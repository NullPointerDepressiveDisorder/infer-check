"""finish_reason / prompt_tokens / token_ids are recorded by every HTTP backend."""

import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from infer_check.backends.llama_cpp import LlamaCppBackend
from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.types import Prompt
from infer_check.utils import normalize_finish_reason

PROMPT = Prompt(id="p1", text="Hello", max_tokens=10)


def _resp(url: str, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("POST", url))


# ── normalize_finish_reason ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("eos", "stop"),
        ("word", "stop"),
        ("end_turn", "stop"),
        ("STOP", "stop"),
        ("length", "length"),
        ("limit", "length"),
        ("max_tokens", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
        ("none", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_finish_reason(raw: Any, expected: str | None) -> None:
    assert normalize_finish_reason(raw) == expected


# ── OpenAI-compatible ───────────────────────────────────────────────


def test_chat_records_finish_reason_and_prompt_tokens() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    body = {
        "choices": [{"message": {"content": "world"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 10},
    }
    with patch("httpx.AsyncClient.post", return_value=_resp("http://x/v1/chat/completions", body)):
        res = asyncio.run(backend.generate(PROMPT))
    assert res.metadata["finish_reason"] == "length"
    assert res.metadata["prompt_tokens"] == 12


def test_chat_tolerates_missing_or_null_usage() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=True)
    body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": None}
    with patch("httpx.AsyncClient.post", return_value=_resp("http://x/v1/chat/completions", body)):
        res = asyncio.run(backend.generate(PROMPT))
    assert res.metadata["finish_reason"] == "stop"
    assert res.metadata["prompt_tokens"] is None


def test_completions_records_finish_reason() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="dummy", chat=False)
    body = {"choices": [{"text": " world", "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}
    with patch("httpx.AsyncClient.post", return_value=_resp("http://x/v1/completions", body)):
        res = asyncio.run(backend.generate(PROMPT))
    assert res.metadata["finish_reason"] == "stop"
    assert res.metadata["prompt_tokens"] == 3


def test_ollama_native_records_done_reason() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:11434", model_id="llama3", chat=True)
    body = {"message": {"content": "hi"}, "done_reason": "length", "prompt_eval_count": 7, "eval_count": 10}
    with patch("httpx.AsyncClient.post", return_value=_resp("http://x/api/chat", body)):
        res = asyncio.run(backend.generate(PROMPT))
    assert res.metadata["finish_reason"] == "length"
    assert res.metadata["prompt_tokens"] == 7


# ── llama.cpp ───────────────────────────────────────────────────────


def test_llama_cpp_stop_type_and_token_ids() -> None:
    backend = LlamaCppBackend(model_id="m")
    body = {"content": " world", "stop_type": "eos", "tokens_evaluated": 5, "tokens": [1917, 13]}
    with patch("httpx.AsyncClient.post", return_value=_resp("http://x/completion", body)) as mock_post:
        res = asyncio.run(backend.generate(PROMPT))
    assert mock_post.call_args.kwargs["json"]["return_tokens"] is True
    assert res.metadata["finish_reason"] == "stop"
    assert res.metadata["prompt_tokens"] == 5
    assert res.token_ids == [1917, 13]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"stopped_eos": True}, "stop"),
        ({"stopped_word": True}, "stop"),
        ({"stopped_limit": True}, "length"),
        ({}, None),
    ],
)
def test_llama_cpp_legacy_stop_flags(flags: dict[str, bool], expected: str | None) -> None:
    backend = LlamaCppBackend(model_id="m")
    body = {"content": " world", **flags}
    with patch("httpx.AsyncClient.post", return_value=_resp("http://x/completion", body)):
        res = asyncio.run(backend.generate(PROMPT))
    assert res.metadata["finish_reason"] == expected
    assert res.token_ids is None
