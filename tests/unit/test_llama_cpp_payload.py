from unittest.mock import patch

import httpx
import pytest

from infer_check.backends.llama_cpp import LlamaCppBackend
from infer_check.types import Prompt


@pytest.mark.asyncio
async def test_llama_cpp_includes_model_in_payload() -> None:
    model_id = "test-model-gguf"
    backend = LlamaCppBackend(model_id=model_id, base_url="http://127.0.0.1:8080")
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={"content": " world", "model": model_id, "timings": {"predicted_per_second": 10.0}},
        request=httpx.Request("POST", "http://127.0.0.1:8080/completion"),
    )

    try:
        with (
            patch("infer_check.backends.llama_cpp.format_prompt", return_value="Hello"),
            patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post,
        ):
            res = await backend.generate(prompt)

            # Verify the call to post
            assert mock_post.called
            args, kwargs = mock_post.call_args
            assert args[0] == "/completion"
            payload = kwargs["json"]
            assert payload["model"] == model_id
            assert payload["prompt"] == "Hello"
            assert payload["n_predict"] == 10

            # Verify result
            assert res.text == " world"
            assert res.model_id == model_id
    finally:
        await backend.cleanup()
