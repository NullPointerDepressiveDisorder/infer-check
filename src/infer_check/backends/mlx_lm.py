"""MLX-LM backend adapter — in-process Python API."""

from __future__ import annotations

import asyncio
import gc
import time
from typing import TYPE_CHECKING, Any, cast

from infer_check.backends.base import BackendAdapter
from infer_check.types import InferenceResult, Prompt
from infer_check.utils import format_prompt

if TYPE_CHECKING:
    import mlx.core as mx

__all__ = ["MLXBackend"]


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
    ) -> None:
        self._model_id = model_id
        self._quantization = quantization
        self._revision = revision
        self._disable_thinking = disable_thinking
        self._model: Any = None
        self._tokenizer: Any = None

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

    def _format(self, prompt: Prompt) -> str:
        """Apply the model's chat template (if any) to the prompt text."""
        return format_prompt(
            prompt.text,
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

        start = time.perf_counter()
        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt_ids,  # pass IDs so BOS handling is under our control
            max_tokens=prompt.max_tokens,
            sampler=make_sampler(temp=temp),
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
