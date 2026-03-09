import asyncio
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from infer_check.types import InferenceResult, Prompt

__all__ = ["BackendAdapter", "BackendConfig", "get_backend"]


class BackendAdapter(Protocol):
    """Protocol defining the interface for all inference backends."""

    @property
    def name(self) -> str:
        """The name of the backend."""
        ...

    async def generate(self, prompt: Prompt) -> InferenceResult:
        """Generate a single inference result for a prompt."""
        ...

    async def generate_batch(self, prompts: list[Prompt]) -> list[InferenceResult]:
        """Generate inference results for a batch of prompts."""
        return await asyncio.gather(*(self.generate(prompt) for prompt in prompts))

    async def health_check(self) -> bool:
        """Check if the backend is ready and responsive."""
        ...

    async def cleanup(self) -> None:
        """Release any resources held by the backend."""
        ...


class BackendConfig(BaseModel):
    """Configuration for an inference backend."""

    model_config = ConfigDict(extra="allow")

    backend_type: Literal["mlx-lm", "llama-cpp", "vllm-mlx", "openai-compat"]
    model_id: str
    quantization: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def get_backend(config: BackendConfig) -> BackendAdapter:
    """Factory function to instantiate the correct backend adapter."""
    if config.backend_type == "mlx-lm":
        from infer_check.backends.mlx_lm import MLXBackend

        return MLXBackend(
            model_id=config.model_id,
            quantization=config.quantization,
        )
    elif config.backend_type == "llama-cpp":
        from infer_check.backends.llama_cpp import LlamaCppBackend

        url = config.base_url or "http://localhost:8080"
        return LlamaCppBackend(base_url=url)
    elif config.backend_type == "vllm-mlx":
        from infer_check.backends.vllm_mlx import VLLMMLXBackend

        url = config.base_url or "http://localhost:8000"
        return VLLMMLXBackend(
            model_id=config.model_id,
            base_url=url,
            chat=config.extra.get("chat", False),
        )
    elif config.backend_type == "openai-compat":
        from infer_check.backends.openai_compat import OpenAICompatBackend

        if not config.base_url:
            raise ValueError(
                "openai-compat backend requires --base-url. "
                "Example: --base-url http://localhost:11434/v1 (Ollama)"
            )
        return OpenAICompatBackend(
            base_url=config.base_url,
            model_id=config.model_id,
            api_key=config.extra.get("api_key"),
            chat=config.extra.get("chat", False),
        )
    else:
        supported = ", ".join(["mlx-lm", "llama-cpp", "vllm-mlx", "openai-compat"])
        raise ValueError(f"Unknown backend type: '{config.backend_type}'. Supported: {supported}")
