from typing import Any, Literal, Protocol, cast, get_args

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from infer_check.types import InferenceResult, Prompt, TokenScores

__all__ = ["BackendAdapter", "BackendConfig", "get_backend", "get_backend_for_model"]


class BackendAdapter(Protocol):
    """Interface every inference backend implements.

    Backends subclass this explicitly so mypy checks conformance where each
    backend is defined: members below with ``...`` bodies are treated as
    abstract, so forgetting one is a type error rather than a silent ``None``
    at runtime.
    """

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. ``"mlx-lm"``."""
        ...

    @property
    def model_id(self) -> str:
        """The model this backend instance serves (repo ID, path, or server-side name)."""
        ...

    @property
    def supports_concurrency(self) -> bool:
        """True only if concurrent ``generate()`` calls genuinely overlap.

        HTTP backends return True (the server schedules/batches requests).
        In-process backends that block the event loop return False; callers
        must not rely on overlapping calls to them (e.g. for stress tests).
        """
        ...

    @property
    def supports_scoring(self) -> bool:
        """True if ``score()`` is implemented (teacher-forced log-probs)."""
        ...

    async def generate(self, prompt: Prompt) -> InferenceResult:
        """Generate a single inference result for a prompt."""
        ...

    async def score(
        self,
        prompt: Prompt,
        target_ids: list[int],
        *,
        top_k: int = 64,
        at_ids: npt.NDArray[np.int64] | None = None,
    ) -> TokenScores:
        """Teacher-forced scoring: log-probs for a *given* continuation of ``prompt``.

        Runs the model over the prompt (templated and encoded exactly as in
        ``generate``) followed by ``target_ids``, without sampling. Row ``i`` of
        the result is the distribution predicting ``target_ids[i]``.

        Args:
            prompt: The prompt the continuation belongs to.
            target_ids: The token sequence to score (typically another model's output).
            top_k: How many most-likely tokens to return per position (0 for none).
            at_ids: Optional ``[len(target_ids), M]`` token IDs; their log-probs are
                returned in ``TokenScores.at_logprobs``.

        Raises:
            NotImplementedError: if ``supports_scoring`` is False.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the backend is ready and responsive."""
        ...

    async def cleanup(self) -> None:
        """Release any resources held by the backend."""
        ...


BackendType = Literal["mlx-lm", "llama-cpp", "vllm-mlx", "openai-compat"]


class BackendConfig(BaseModel):
    """Configuration for an inference backend."""

    model_config = ConfigDict(extra="allow")

    backend_type: BackendType
    model_id: str
    quantization: str | None = None
    hf_revision: str | None = None
    base_url: str | None = None
    disable_thinking: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


def get_backend(config: BackendConfig) -> BackendAdapter:
    """Factory function to instantiate the correct backend adapter."""
    if config.backend_type == "mlx-lm":
        from infer_check.backends.mlx_lm import MLXBackend

        return MLXBackend(
            model_id=config.model_id,
            quantization=config.quantization,
            revision=config.hf_revision,
            disable_thinking=config.disable_thinking,
        )
    elif config.backend_type == "llama-cpp":
        from infer_check.backends.llama_cpp import LlamaCppBackend

        url = config.base_url or "http://127.0.0.1:8080"
        return LlamaCppBackend(
            model_id=config.model_id,
            base_url=url,
            revision=config.hf_revision,
            disable_thinking=config.disable_thinking,
        )
    elif config.backend_type == "vllm-mlx":
        from infer_check.backends.vllm_mlx import VLLMMLXBackend

        url = config.base_url or "http://127.0.0.1:8000"
        return VLLMMLXBackend(
            model_id=config.model_id,
            base_url=url,
            chat=config.extra.get("chat", True),
            revision=config.hf_revision,
            disable_thinking=config.disable_thinking,
        )
    elif config.backend_type == "openai-compat":
        from infer_check.backends.openai_compat import OpenAICompatBackend

        if not config.base_url:
            raise ValueError(
                "openai-compat backend requires --base-url. Example: --base-url http://127.0.0.1:11434 (Ollama)"
            )
        return OpenAICompatBackend(
            base_url=config.base_url,
            model_id=config.model_id,
            api_key=config.extra.get("api_key"),
            chat=config.extra.get("chat", True),
            revision=config.hf_revision,
            disable_thinking=config.disable_thinking,
        )
    else:
        supported = ", ".join(["mlx-lm", "llama-cpp", "vllm-mlx", "openai-compat"])
        raise ValueError(f"Unknown backend type: '{config.backend_type}'. Supported: {supported}")


def get_backend_for_model(
    model_str: str,
    backend_type: str | None = None,
    base_url: str | None = None,
    quantization: str | None = None,
    disable_thinking: bool = True,
    chat: bool = True,
) -> BackendAdapter:
    """Resolve model string to a backend and instantiate it.

    If backend_type is provided, it forces that backend. Otherwise, it resolves
    based on the model string.
    """
    from infer_check.resolve import resolve_model

    # Always normalize the model string first to ensure consistent model_id/base_url
    resolved = resolve_model(model_str, base_url=base_url)
    chosen = backend_type or resolved.backend
    valid: tuple[str, ...] = get_args(BackendType)
    if chosen not in valid:
        raise ValueError(f"Unknown backend '{chosen}'. Choose one of: {', '.join(valid)}.")
    config = BackendConfig(
        backend_type=cast(BackendType, chosen),
        model_id=resolved.model_id,
        base_url=base_url or resolved.base_url,
        quantization=quantization or resolved.label,
        hf_revision=resolved.revision,
        disable_thinking=disable_thinking,
        extra={"chat": chat},
    )

    return get_backend(config)
