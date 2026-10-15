"""MLX-LM backend adapter — in-process Python API."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from infer_check.analysis.scoring import model_input
from infer_check.backends.base import BackendAdapter
from infer_check.types import InferenceResult, Prompt, TokenScores
from infer_check.utils import format_prompt

if TYPE_CHECKING:
    import mlx.core as mx

__all__ = ["MLXBackend"]

# Tokens per forward pass when scoring. Bounds peak memory: each chunk
# materialises [chunk, vocab] float32 logits (~260 MB for a 128k vocab).
_SCORE_CHUNK = 512


class MLXBackend(BackendAdapter):
    """Backend adapter for mlx-lm (Apple Silicon local inference).

    Lazy-loads the model on the first ``generate()`` call so that
    importing this module alone never triggers a heavy download.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str | None = None,
        revision: str | None = None,
        disable_thinking: bool = True,
        logits_transform: Callable[[mx.array], mx.array] | None = None,
    ) -> None:
        """
        Args:
            logits_transform: Optional function applied to the model's logits at
                every position, identically in generation and scoring. Used to
                build deliberately degraded models as positive controls (see
                ``infer_check.backends.perturb``); leave None for normal use.
        """
        self._model_id = model_id
        self._quantization = quantization
        self._revision = revision
        self._disable_thinking = disable_thinking
        self._logits_transform = logits_transform
        self._model: Any = None
        self._tokenizer: Any = None
        self._tokenizer_fp: str | None = None

    # ------------------------------------------------------------------
    # BackendAdapter protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mlx-lm"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_concurrency(self) -> bool:
        # In-process and synchronous: calls run one at a time and block the
        # event loop, so concurrent callers get no real overlap.
        return False

    @property
    def supports_scoring(self) -> bool:
        return True

    @property
    def tokenizer_fingerprint(self) -> str | None:
        """Hash of the tokenizer's vocabulary; None until the model is loaded.

        Computed once and kept after ``cleanup()``: a sweep frees the baseline
        model before the test levels run, but still needs its fingerprint.
        """
        if self._tokenizer_fp is None and self._tokenizer is not None:
            blob = json.dumps(sorted(self._tokenizer.get_vocab().items()), ensure_ascii=False).encode()
            self._tokenizer_fp = hashlib.sha256(blob).hexdigest()[:16]
        return self._tokenizer_fp

    async def score(
        self,
        prompt: Prompt,
        target_ids: list[int],
        *,
        top_k: int = 64,
        at_ids: npt.NDArray[np.int64] | None = None,
    ) -> TokenScores:
        """Teacher-forced log-probs for ``target_ids`` as a continuation of ``prompt``.

        See ``BackendAdapter.score``. Like ``generate``, this runs synchronously.
        """
        self._ensure_loaded()
        return self._score(prompt, target_ids, top_k=top_k, at_ids=at_ids)

    async def generate(self, prompt: Prompt) -> InferenceResult:
        """Run inference and return an ``InferenceResult``.

        The computation runs synchronously inside this coroutine and blocks
        the event loop. That is deliberate: MLX generation is single-stream,
        and ``supports_concurrency`` is False so callers don't overlap calls.
        Keep this ``async`` — every caller awaits it.

        There is intentionally no fallback path: if generation fails, the
        error propagates. A correctness tool must never silently switch to a
        different way of measuring.
        """
        self._ensure_loaded()
        try:
            return self._generate(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"MLX generation failed for prompt '{prompt.id}' ({prompt.text[:80]!r})\n"
                f"Model: {self._model_id}\nError: {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """Load the model and generate a single token through the real code path."""
        try:
            self._ensure_loaded()
            self._generate(Prompt(id="health-check", text="hi", max_tokens=1))
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Release model references and trigger garbage collection."""
        self._model = None
        self._tokenizer = None
        await asyncio.to_thread(gc.collect)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the model and tokenizer on first use."""
        if self._model is not None:
            return

        try:
            from mlx_lm import load
        except ImportError:
            raise RuntimeError("mlx-lm not installed. Install with: pip install infer-check[mlx]") from None

        from pathlib import Path

        model_path = Path(self._model_id).expanduser()
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}\nCheck the path or use a HuggingFace repo ID instead."
            )

        repo_or_path = str(model_path) if model_path.exists() else self._model_id
        try:
            res = load(repo_or_path, revision=self._revision)
        except Exception as exc:
            msg = str(exc)
            if "404" in msg or "Repository Not Found" in msg:
                raise RuntimeError(
                    f"Model not found: {repo_or_path}\n"
                    f"Check the exact repo ID at https://huggingface.co/mlx-community\n"
                    f"Common issue: 'Llama-3.1-8B' vs 'Meta-Llama-3.1-8B' naming."
                ) from exc
            if "gated" in msg.lower() or "authenticated" in msg.lower():
                raise RuntimeError(
                    f"Model '{repo_or_path}' requires authentication.\n"
                    f"Run: huggingface-cli login\n"
                    f"Then accept the license at: https://huggingface.co/{self._model_id}"
                ) from exc
            raise RuntimeError(f"Failed to load model '{repo_or_path}': {exc}") from exc

        self._model = res[0]
        self._tokenizer = res[1]

        _ = self.tokenizer_fingerprint  # compute now, while the tokenizer is loaded

    def _format(self, prompt: Prompt) -> str:
        """Apply the model's chat template (if any) to the prompt text."""
        return format_prompt(
            model_input(prompt),
            tokenizer=self._tokenizer,
            revision=self._revision,
            disable_thinking=self._disable_thinking,
        )

    def _encode(self, formatted: str) -> list[int]:
        """Encode a (possibly chat-templated) string without adding a second BOS.

        Chat templates usually emit the BOS token themselves (e.g. Llama 3's
        ``<|begin_of_text|>``); letting the tokenizer add another one produces
        an off-distribution double-BOS prompt. Mirrors mlx-lm's own logic in
        ``stream_generate`` for string prompts.
        """
        bos = getattr(self._tokenizer, "bos_token", None)
        add_special = not (bos and formatted.startswith(bos))
        return list(self._tokenizer.encode(formatted, add_special_tokens=add_special))

    @staticmethod
    def _top_k(lp: mx.array, k: int) -> tuple[list[float], list[int]]:
        """Top-``k`` log-probs (descending) and their token IDs from a full-vocab vector."""
        import mlx.core as mx

        k = min(k, int(lp.shape[-1]))
        idx = mx.argpartition(-lp, kth=k - 1)[:k]
        vals = lp[idx]
        order = mx.argsort(-vals)
        return cast(list[float], vals[order].tolist()), cast(list[int], idx[order].tolist())

    def _generate(self, prompt: Prompt) -> InferenceResult:
        """Generate via ``mlx_lm.stream_generate``, capturing per-token log-probs.

        ``stream_generate`` (unlike the lower-level ``generate_step``) stops at
        the model's end-of-sequence tokens, honours ``max_tokens``, and decodes
        text with a streaming detokenizer so multi-byte characters split across
        tokens come out intact.
        """
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        meta = prompt.metadata or {}
        temp = float(meta.get("temperature", 0.0))
        top_k = int(meta.get("top_k_logprobs", 10))

        prompt_ids = self._encode(self._format(prompt))

        token_ids: list[int] = []
        logprobs: list[float] = []
        distributions: list[list[float]] = []
        distribution_metadata: list[dict[str, int | str]] = []
        pieces: list[str] = []
        finish_reason: str | None = None

        def record(token: int, lp: mx.array) -> None:
            lp = lp.astype(mx.float32)  # bf16/fp16 log-probs lose precision
            token_ids.append(token)
            # .item() is typed int | float | complex; lp is float32 so it's a float.
            logprobs.append(cast(float, lp[token].item()))
            if top_k > 0:
                vals, ids = self._top_k(lp, top_k)
                distributions.append(vals)
                distribution_metadata.append({f"id_{i}": t for i, t in enumerate(ids)})

        extra: dict[str, Any] = {}
        if self._logits_transform is not None:
            transform = self._logits_transform
            extra["logits_processors"] = [lambda _tokens, logits: transform(logits)]

        start = time.perf_counter()
        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt_ids,  # pass IDs so BOS handling is under our control
            max_tokens=prompt.max_tokens,
            sampler=make_sampler(temp=temp),
            **extra,
        ):
            # Always keep the text: the final response carries the detokenizer's flush.
            pieces.append(resp.text)
            if resp.finish_reason == "stop":
                # resp.token is the end-of-sequence token itself: not part of the output.
                finish_reason = "stop"
                break
            record(int(resp.token), resp.logprobs)
            if resp.finish_reason == "length":
                # The last real token is only delivered in this final response.
                finish_reason = "length"
                break
        elapsed_s = time.perf_counter() - start

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self._model_id,
            quantization=self._quantization,
            # Per-token display strings only: they can split multi-byte
            # characters. Use ``token_ids`` / ``text`` for comparisons.
            tokens=[self._tokenizer.decode([t]) for t in token_ids],
            token_ids=token_ids,
            logprobs=logprobs or None,
            distributions=distributions or None,
            distribution_metadata=distribution_metadata or None,
            text="".join(pieces),
            latency_ms=elapsed_s * 1000,
            tokens_per_second=len(token_ids) / elapsed_s if elapsed_s > 0 else None,
            metadata={"finish_reason": finish_reason, "prompt_tokens": len(prompt_ids)},
        )

    def _score(
        self,
        prompt: Prompt,
        target_ids: list[int],
        *,
        top_k: int,
        at_ids: npt.NDArray[np.int64] | None,
        chunk_size: int = _SCORE_CHUNK,
    ) -> TokenScores:
        """Teacher-forced forward pass over ``prompt + target_ids``.

        The prompt is encoded exactly as in ``_generate``. Everything but the
        last prompt token is prefilled into a KV cache (their logits aren't
        needed); then ``[last_prompt_token, *target_ids[:-1]]`` is fed in chunks,
        so logits row ``i`` is the distribution that predicts ``target_ids[i]``.
        Only small per-position summaries leave the device — the full
        ``[T, vocab]`` matrix is never materialised at once.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        n = len(target_ids)
        if at_ids is not None and (at_ids.ndim != 2 or at_ids.shape[0] != n):
            raise ValueError(f"at_ids must have shape [{n}, M], got {at_ids.shape}")

        prompt_ids = self._encode(self._format(prompt))
        if not prompt_ids:
            raise ValueError("prompt encodes to zero tokens")

        top_ids: list[npt.NDArray[np.int64]] = []
        top_lps: list[npt.NDArray[np.float32]] = []
        argmaxes: list[npt.NDArray[np.int64]] = []
        at_lps: list[npt.NDArray[np.float32]] = []

        if n > 0:
            cache = make_prompt_cache(self._model)
            prefix = prompt_ids[:-1]
            for s in range(0, len(prefix), chunk_size):
                self._model(mx.array(prefix[s : s + chunk_size])[None], cache=cache)
                mx.eval([c.state for c in cache])

            feed = [prompt_ids[-1], *target_ids[:-1]]
            for s in range(0, n, chunk_size):
                logits = self._model(mx.array(feed[s : s + chunk_size])[None], cache=cache)[0]
                logits = logits.astype(mx.float32)
                if self._logits_transform is not None:
                    logits = self._logits_transform(logits)
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

                outputs: list[mx.array] = [mx.argmax(lp, axis=-1)]
                k = min(top_k, int(lp.shape[-1]))
                if k > 0:
                    idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[:, :k]
                    vals = mx.take_along_axis(lp, idx, axis=-1)
                    order = mx.argsort(-vals, axis=-1)
                    outputs += [mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(vals, order, axis=-1)]
                if at_ids is not None:
                    want = mx.array(at_ids[s : s + chunk_size].astype(np.int32))
                    outputs.append(mx.take_along_axis(lp, want, axis=-1))
                mx.eval(outputs)

                argmaxes.append(np.array(outputs[0], dtype=np.int64))
                if k > 0:
                    top_ids.append(np.array(outputs[1], dtype=np.int64))
                    top_lps.append(np.array(outputs[2], dtype=np.float32))
                if at_ids is not None:
                    at_lps.append(np.array(outputs[-1], dtype=np.float32))

        k_out = top_ids[0].shape[1] if top_ids else 0
        return TokenScores(
            top_ids=np.concatenate(top_ids) if top_ids else np.zeros((n, k_out), dtype=np.int64),
            top_logprobs=np.concatenate(top_lps) if top_lps else np.zeros((n, k_out), dtype=np.float32),
            argmax=np.concatenate(argmaxes) if argmaxes else np.zeros((0,), dtype=np.int64),
            at_logprobs=(
                np.concatenate(at_lps)
                if at_lps
                else (np.zeros((0, at_ids.shape[1]), dtype=np.float32) if at_ids is not None else None)
            ),
        )
