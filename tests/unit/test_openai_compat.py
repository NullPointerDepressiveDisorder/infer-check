import asyncio
from unittest.mock import patch

import httpx
import pytest

from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.types import Prompt


def test_generate_chat_success() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "world"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        res = asyncio.run(backend.generate(prompt))
        assert res.text == "world"
        assert res.tokens == ["world"]
        assert res.logprobs is None


def test_generate_chat_connection_refused() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "Cannot connect to server" in str(exc.value)


def test_generate_chat_timeout() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "timed out" in str(exc.value)


def test_generate_completions_404() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000", model_id="dummy", chat=False)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(404, request=httpx.Request("POST", "http://localhost:8000/v1/completions"))
    with patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.HTTPStatusError("404 Not Found", request=mock_response.request, response=mock_response),
    ):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "returned 404" in str(exc.value)


def test_generate_chat_malformed_json() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        text="Not valid JSON",
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "non-JSON response" in str(exc.value)


def test_generate_empty_choices() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={"choices": []},
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "empty or malformed response" in str(exc.value)
