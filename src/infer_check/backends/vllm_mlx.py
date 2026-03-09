"""vLLM-MLX backend adapter — thin wrapper over ``OpenAICompatBackend``."""

from __future__ import annotations

import httpx

from infer_check.backends.openai_compat import OpenAICompatBackend

__all__ = ["VLLMMLXBackend"]


class VLLMMLXBackend(OpenAICompatBackend):
    """Specialised adapter for vllm-mlx servers.

    Inherits all generation logic from :class:`OpenAICompatBackend` and
    adds a model-aware health check and a convenience class method for
    starting the server process.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        chat: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            chat=chat,
        )

    @property
    def name(self) -> str:
        return "vllm-mlx"

    async def health_check(self) -> bool:
        """Verify the server is up *and* the expected model is loaded."""
        try:
            resp = await self._client.get("/v1/models")
            if resp.status_code != 200:
                return False

            data = resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            return self._model_id in model_ids
        except httpx.ConnectError:
            return False

    @classmethod
    def from_model(
        cls,
        model_id: str,
        quantization: str | None = None,
        base_url: str = "http://localhost:8000",
    ) -> VLLMMLXBackend:
        """Create a backend for *model_id*.

        .. note::

           This does **not** start the vllm-mlx server automatically.
           Start it manually before calling this method::

               python -m vllm.entrypoints.openai.api_server \\
                   --model <model_id> \\
                   --quantization <quantization> \\
                   --port 8000

        Args:
            model_id: HuggingFace model identifier.
            quantization: Optional quantization string (e.g. ``"4bit"``).
            base_url: Server URL (default ``http://localhost:8000``).

        Returns:
            A configured :class:`VLLMMLXBackend` instance.
        """
        return cls(
            model_id=model_id,
            base_url=base_url,
        )
