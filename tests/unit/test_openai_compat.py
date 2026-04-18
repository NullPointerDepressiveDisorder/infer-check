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
    with (
        patch("infer_check.backends.openai_compat.format_prompt", return_value="Hello"),
        patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.HTTPStatusError("404 Not Found", request=mock_response.request, response=mock_response),
        ),
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


def test_generate_chat_disable_thinking_payload() -> None:
    """With disable_thinking, payload carries cross-backend reasoning-off hints."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",
        model_id="dummy",
        chat=True,
        disable_thinking=True,
    )
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        sent = post.call_args.kwargs["json"]
        assert sent["chat_template_kwargs"] == {"enable_thinking": False, "thinking": False}
        assert sent["think"] is False
        assert sent["reasoning"] == {"enabled": False}
        assert sent["reasoning_effort"] == "minimal"


def test_generate_chat_enable_thinking_omits_keys() -> None:
    """When disable_thinking=False, none of the reasoning-off hints are sent."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",
        model_id="dummy",
        chat=True,
        disable_thinking=False,
    )
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        sent = post.call_args.kwargs["json"]
        assert "chat_template_kwargs" not in sent
        assert "think" not in sent
        assert "reasoning" not in sent
        assert "reasoning_effort" not in sent


def test_generate_chat_disable_thinking_strips_think_token_from_message() -> None:
    """Ollama <|think|> trigger is stripped from user content when disabled."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",
        model_id="dummy",
        chat=True,
        disable_thinking=True,
    )
    prompt = Prompt(id="p1", text="<|think|>what is 2+2?", max_tokens=10)
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "4"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        sent = post.call_args.kwargs["json"]
        assert sent["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "what is 2+2?"},
        ]


def test_generate_chat_enable_thinking_preserves_think_token() -> None:
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",
        model_id="dummy",
        chat=True,
        disable_thinking=False,
    )
    prompt = Prompt(id="p1", text="<|think|>what is 2+2?", max_tokens=10)
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "4"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        sent = post.call_args.kwargs["json"]
        assert sent["messages"] == [{"role": "user", "content": "<|think|>what is 2+2?"}]


def test_generate_chat_disable_thinking_prepends_empty_system() -> None:
    """Empty system message overrides server-side SYSTEM defaults (e.g. Modelfile)."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",  # not Ollama port → /v1/chat path
        model_id="dummy",
        chat=True,
        disable_thinking=True,
    )
    prompt = Prompt(id="p1", text="hi", max_tokens=5)
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "hey"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        sent = post.call_args.kwargs["json"]
        assert sent["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ]


def test_generate_chat_enable_thinking_no_system_message() -> None:
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",
        model_id="dummy",
        chat=True,
        disable_thinking=False,
    )
    prompt = Prompt(id="p1", text="hi", max_tokens=5)
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "hey"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        sent = post.call_args.kwargs["json"]
        assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_ollama_native_route_on_disable_thinking() -> None:
    """When talking to Ollama with thinking disabled, route to /api/chat."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:11434",
        model_id="gemma3:4b",
        chat=True,
        disable_thinking=True,
    )
    prompt = Prompt(id="p1", text="<|think|>hello", max_tokens=5)

    native_response = httpx.Response(
        200,
        json={
            "message": {"content": "hi there"},
            "eval_count": 2,
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )

    with patch("httpx.AsyncClient.post", return_value=native_response) as post:
        res = asyncio.run(backend.generate(prompt))
        assert post.call_args.args[0] == "/api/chat"
        sent = post.call_args.kwargs["json"]
        assert sent["think"] is False
        assert sent["stream"] is False
        assert sent["options"]["num_predict"] == 5
        assert sent["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hello"},
        ]
        assert res.text == "hi there"
        assert res.logprobs is None


def test_ollama_port_with_thinking_enabled_uses_openai_route() -> None:
    """Ollama detection only flips routing when disable_thinking is set."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:11434",
        model_id="gemma3:4b",
        chat=True,
        disable_thinking=False,
    )
    prompt = Prompt(id="p1", text="hi", max_tokens=5)
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "hey"}}], "usage": {"completion_tokens": 1}},
        request=httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_response) as post:
        asyncio.run(backend.generate(prompt))
        assert post.call_args.args[0] == "/v1/chat/completions"


def test_generate_chat_disable_thinking_retry_on_400() -> None:
    """Server rejects unknown params; backend retries once without them."""
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8000",
        model_id="dummy",
        chat=True,
        disable_thinking=True,
    )
    prompt = Prompt(id="p1", text="Hello", max_tokens=10)

    bad_request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
    bad_response = httpx.Response(400, text="unknown field", request=bad_request)
    good_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}},
        request=bad_request,
    )

    call_count = {"n": 0}

    def fake_post(*_args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call raises via raise_for_status
            return bad_response
        return good_response

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        res = asyncio.run(backend.generate(prompt))
        assert res.text == "ok"
        assert call_count["n"] == 2
        # State should be sticky — future requests won't include the keys.
        assert backend._thinking_keys_supported is False


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
