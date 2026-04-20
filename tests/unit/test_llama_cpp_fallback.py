from unittest.mock import patch

import httpx
import pytest

from infer_check.backends.llama_cpp import LlamaCppBackend
from infer_check.types import Prompt


@pytest.mark.asyncio
async def test_llama_cpp_model_id_fallback() -> None:
    model_id = "test-model-gguf"
    backend = LlamaCppBackend(model_id=model_id, base_url="http://127.0.0.1:8080")
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    # Response missing "model" field
    mock_response = httpx.Response(
        200,
        json={"content": " world", "timings": {"predicted_per_second": 10.0}},
        request=httpx.Request("POST", "http://127.0.0.1:8080/completion"),
    )

    try:
        with (
            patch("infer_check.backends.llama_cpp.format_prompt", return_value="Hello"),
            patch("httpx.AsyncClient.post", return_value=mock_response),
        ):
            res = await backend.generate(prompt)

            # Verify it falls back to backend's model_id instead of "unknown"
            assert res.model_id == model_id
    finally:
        await backend.cleanup()
