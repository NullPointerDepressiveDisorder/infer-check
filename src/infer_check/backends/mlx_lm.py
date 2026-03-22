"""MLX-LM backend adapter — in-process Python API."""

from __future__ import annotations

import gc
import time
from typing import Any, cast

from infer_check.types import InferenceResult, Prompt

__all__ = ["MLXBackend"]


class MLXBackend:
    """Backend adapter for mlx-lm (Apple Silicon local inference).

    Lazy-loads the model on the first ``generate()`` call so that
    importing this module alone never triggers a heavy download.
    """

    def __init__(self, model_id: str, quantization: str | None = None) -> None:
        self._model_id = model_id
        self._quantization = quantization
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # BackendAdapter protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mlx-lm"

    async def generate(self, prompt: Prompt) -> InferenceResult:
        """Run inference and return an ``InferenceResult``.

        Uses ``mlx_lm.generate_step`` when available so that per-token
        logprobs can be captured.  Falls back to the simpler
        ``mlx_lm.generate`` otherwise.

        The actual computation is synchronous (MLX is single-threaded),
        but the method is async to satisfy the ``BackendAdapter`` protocol.
        """
        self._ensure_loaded()

        try:
            return self._generate_with_logprobs(prompt)
        except Exception as exc:
            import logging

            logging.debug("generate_step failed (%s), falling back to simple generate", exc)
            try:
                return self._generate_simple(prompt)
            except Exception as inner:
                raise RuntimeError(
                    f"MLX generation failed for prompt '{prompt.text[:80]}...'\nModel: {self._model_id}\nError: {inner}"
                ) from inner

    async def generate_batch(self, prompts: list[Prompt]) -> list[InferenceResult]:
        """Generate inference results for a batch of prompts.

        MLX is single-threaded so we run sequentially rather than
        using ``asyncio.gather`` which would not yield parallelism.
        """
        return [await self.generate(p) for p in prompts]

    async def health_check(self) -> bool:
        """Load the model and generate a single token."""
        try:
            self._ensure_loaded()
            from mlx_lm import generate as mlx_generate

            mlx_generate(
                self._model,
                self._tokenizer,
                prompt="hi",
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Release model references and trigger garbage collection."""
        self._model = None
        self._tokenizer = None
        gc.collect()

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
            res = load(repo_or_path)
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

    def _format_prompt(self, text: str) -> str:
        """Apply chat template if the tokenizer has one (Instruct models).

        Raw prompts sent to Instruct models produce undefined behavior that
        varies across quantization levels, making comparisons meaningless.
        """
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": text}]
            return str(self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return text

    def _generate_simple(self, prompt: Prompt) -> InferenceResult:
        """Generate using the high-level ``mlx_lm.generate`` API."""
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        temp = prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0
        sampler = make_sampler(temp=temp)
        formatted = self._format_prompt(prompt.text)
        start = time.perf_counter()
        text: str = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=prompt.max_tokens,
            sampler=sampler,
        )
        elapsed_s = time.perf_counter() - start

        tokens = text.split()
        tps = len(tokens) / elapsed_s if elapsed_s > 0 else None

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self._model_id,
            quantization=self._quantization,
            tokens=tokens,
            logprobs=None,
            text=text,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )

    def _generate_with_logprobs(self, prompt: Prompt) -> InferenceResult:
        """Generate token-by-token via ``generate_step`` to capture logprobs."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        temp = prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0
        sampler = make_sampler(temp=temp)
        formatted = self._format_prompt(prompt.text)
        input_ids = self._tokenizer.encode(formatted, return_tensors="mlx")

        # Configurable top-K to avoid memory explosion. Default to 10.
        top_k = prompt.metadata.get("top_k_logprobs", 10) if prompt.metadata else 10

        tokens: list[str] = []
        logprobs: list[float] = []
        distributions: list[list[float]] = []
        distribution_metadata: list[dict[str, int | str]] = []

        start = time.perf_counter()

        for step_idx, (token, logprob_dist) in enumerate(
            generate_step(
                prompt=input_ids,
                model=self._model,
                sampler=sampler,
            )
        ):
            if step_idx >= prompt.max_tokens:
                break

            # logprob_dist is an mlx array of full-vocab logprobs.
            # We only keep top-K to save memory.
            if top_k > 0:
                # mx.argpartition is not always available or might be slow for small K.
                # Since we need to move to CPU anyway for serialization, we can do it there.
                # But to avoid huge tolist(), we can use mx.topk if available or just move a bit.

                # Clamp K to the vocabulary size to avoid out-of-bounds issues.
                vocab_size = int(logprob_dist.shape[0])
                if vocab_size <= 0:
                    # Nothing to record for this step.
                    continue
                effective_top_k = int(top_k)
                if effective_top_k < 1:
                    # Should not happen due to the outer condition, but guard defensively.
                    continue
                if effective_top_k > vocab_size:
                    effective_top_k = vocab_size

                # Get top-K indices and values
                top_k_indices = mx.argpartition(-logprob_dist, effective_top_k - 1)[:effective_top_k]
                top_k_values = logprob_dist[top_k_indices]

                # Sort them for consistency
                sort_idx = mx.argsort(-top_k_values)
                top_k_indices = top_k_indices[sort_idx]
                top_k_values = top_k_values[sort_idx]

                dist_list = cast(list[float], top_k_values.tolist())
                dist_indices = cast(list[int], top_k_indices.tolist())

                distributions.append(dist_list)
                meta: dict[str, int | str] = {}
                for i, idx in enumerate(dist_indices):
                    meta[f"id_{i}"] = int(idx)
                distribution_metadata.append(meta)

            token_id = int(token.item())
            token_str = self._tokenizer.decode([token_id])
            tokens.append(token_str)

            # The logprob of the chosen token (from the full distribution)
            logprobs.append(float(logprob_dist[token_id]))

        elapsed_s = time.perf_counter() - start
        text = "".join(tokens)
        tps = len(tokens) / elapsed_s if elapsed_s > 0 else None

        # Ensure any pending MLX computation is done.
        mx.eval()

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self._model_id,
            quantization=self._quantization,
            tokens=tokens,
            logprobs=logprobs if logprobs else None,
            distributions=distributions if distributions else None,
            distribution_metadata=distribution_metadata if distribution_metadata else None,
            text=text,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )
