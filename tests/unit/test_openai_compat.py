import asyncio
from unittest.mock import patch

import httpx
import pytest

from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.types import Prompt


def test_generate_chat_success() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "world"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        res = asyncio.run(backend.generate(prompt))
        assert res.text == "world"
        assert res.tokens == ["world"]
        assert res.logprobs is None


def test_generate_chat_connection_refused() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "Cannot connect to server" in str(exc.value)


def test_generate_chat_timeout() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "timed out" in str(exc.value)


def test_generate_completions_404() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=False)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(404, request=httpx.Request("POST", "http://127.0.0.1:8000/v1/completions"))
    with patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.HTTPStatusError("404 Not Found", request=mock_response.request, response=mock_response),
    ):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "returned 404" in str(exc.value)


def test_generate_chat_malformed_json() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        text="Not valid JSON",
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "non-JSON response" in str(exc.value)


def test_generate_empty_choices() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={"choices": []},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(backend.generate(prompt))
        assert "empty or malformed response" in str(exc.value)


def test_generate_chat_reasoning_fallback() -> None:
    """Test fallback to message.reasoning_content when message.content is missing/empty."""
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    # Content empty, reasoning_content present
    mock_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "", "reasoning_content": "Thinking... world"}}],
            "usage": {"completion_tokens": 2},
        },
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        res = asyncio.run(backend.generate(prompt))
        assert res.text == "Thinking... world"
        assert res.tokens == ["Thinking...", "world"]

    # Content missing, reasoning_content present
    mock_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"reasoning_content": "Just thinking"}}],
            "usage": {"completion_tokens": 2},
        },
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        res = asyncio.run(backend.generate(prompt))
        assert res.text == "Just thinking"
        assert res.tokens == ["Just", "thinking"]


def test_generate_chat_with_logprobs() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": "world"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "world",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "world", "logprob": -0.1},
                                    {"token": "earth", "logprob": -2.0},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {"completion_tokens": 1},
        },
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        res = asyncio.run(backend.generate(prompt))
        assert res.text == "world"
        assert res.tokens == ["world"]
        assert res.logprobs == [-0.1]
        # top_logprobs are sorted by token: earth (-2.0), world (-0.1)
        assert res.distributions == [[-2.0, -0.1]]
        assert res.distribution_metadata == [{"id_0": "earth", "id_1": "world"}]


def test_generate_chat_with_logprobs_nan_and_missing() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        content=b"{}",
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )

    # Use patch to return NaN for the raw logprob values to avoid JSON serialization issues in httpx.Response
    with (
        patch("httpx.AsyncClient.post", return_value=mock_response),
        patch(
            "httpx.Response.json",
            return_value={
                "choices": [
                    {
                        "message": {"content": "world"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "world",
                                    "logprob": float("nan"),
                                    "top_logprobs": [
                                        {"token": "world", "logprob": float("nan")},
                                        {"token": "earth"},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 1},
            },
        ),
    ):
        res = asyncio.run(backend.generate(prompt))
        assert res.logprobs == [-9999.0]
        # top_logprobs sorted by token: earth (-9999.0), world (-9999.0)
        assert res.distributions == [[-9999.0, -9999.0]]
        assert res.distribution_metadata == [{"id_0": "earth", "id_1": "world"}]
