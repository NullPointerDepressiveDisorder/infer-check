"""Comparisons with servers that return text only (no log-probs), and whitespace.

Seen in gate 3 against vllm-mlx 0.6, which accepts ``logprobs`` and returns none:
its outputs were split into words and compared position by position with real
model tokens, so 97 of 100 prompts "diverged at token 1", identical ones included.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.runner import TestRunner, first_divergence, has_token_info
from infer_check.types import InferenceResult, Prompt


def _res(text: str, tokens: list[str] | None = None, token_ids: list[int] | None = None) -> InferenceResult:
    return InferenceResult(
        prompt_id="p0",
        backend_name="b",
        model_id="m",
        tokens=tokens or [],
        token_ids=token_ids,
        text=text,
        latency_ms=1.0,
    )


MODEL_TOKENS = _res("To solve this", tokens=["To", " solve", " this"], token_ids=[1, 2, 3])
TEXT_ONLY = _res("To solve this")


# ── token positions need tokens on both sides ──────────────────────────


def test_text_only_result_has_no_token_info() -> None:
    assert has_token_info(MODEL_TOKENS)
    assert not has_token_info(TEXT_ONLY)
    assert has_token_info(_res(""))  # an empty output trivially has its (zero) tokens


def test_no_divergence_index_against_a_text_only_side() -> None:
    assert first_divergence(MODEL_TOKENS, TEXT_ONLY) is None
    assert first_divergence(TEXT_ONLY, MODEL_TOKENS) is None


def test_real_tokens_are_still_compared() -> None:
    other = _res("To fix this", tokens=["To", " fix", " this"], token_ids=[1, 9, 3])
    assert first_divergence(MODEL_TOKENS, other) == 1
    assert first_divergence(MODEL_TOKENS, _res("")) == 0


def test_diff_reports_no_divergence_index_for_text_only_servers(tmp_path: Path) -> None:
    comp = TestRunner(cache_dir=tmp_path)._compare(MODEL_TOKENS, TEXT_ONLY)
    assert comp.text_similarity == 1.0 and comp.token_divergence_index is None


# ── whitespace: exact stays primary, whitespace-only is flagged ────────


@pytest.mark.parametrize(
    ("baseline", "test", "similarity_is_1", "whitespace_only"),
    [
        ("this gives us ", "this gives us", False, True),  # server dropped the final space token
        ("\nanswer", "answer", False, True),
        ("same", "same", True, False),
        ("this gives us 4", "this gives us 5", False, False),
        ("a b", "a  b", False, False),  # inner whitespace is a real difference
    ],
)
def test_whitespace_only_differences_are_flagged_not_hidden(
    tmp_path: Path, baseline: str, test: str, similarity_is_1: bool, whitespace_only: bool
) -> None:
    comp = TestRunner(cache_dir=tmp_path)._compare(_res(baseline), _res(test))
    assert (comp.text_similarity == 1.0) is similarity_is_1  # exact comparison is unchanged
    assert comp.metadata["whitespace_only"] is whitespace_only


# ── the HTTP backend says once that it got no log-probs ────────────────


def _chat_response(**extra: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "To solve this"}, "finish_reason": "stop", **extra}]},
        request=httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions"),
    )


def test_missing_logprobs_give_no_tokens_and_one_warning(capsys: pytest.CaptureFixture[str]) -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="m", chat=True)
    with patch("httpx.AsyncClient.post", return_value=_chat_response()):
        first = asyncio.run(backend.generate(Prompt(id="p0", text="2+2?")))
        asyncio.run(backend.generate(Prompt(id="p1", text="3+3?")))
    assert first.tokens == [] and first.text == "To solve this"
    assert capsys.readouterr().out.count("returned no log-probs") == 1


def test_no_warning_when_logprobs_come_back(capsys: pytest.CaptureFixture[str]) -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8000", model_id="m", chat=True)
    logprobs = {"content": [{"token": "To", "logprob": -0.1, "top_logprobs": [{"token": "To", "logprob": -0.1}]}]}
    with patch("httpx.AsyncClient.post", return_value=_chat_response(logprobs=logprobs)):
        res = asyncio.run(backend.generate(Prompt(id="p0", text="2+2?")))
    assert res.tokens == ["To"]
    assert "returned no log-probs" not in capsys.readouterr().out


# ── determinism doesn't invent a position for text-only outputs ────────


def test_determinism_records_no_position_when_tokens_are_unknown(tmp_path: Path) -> None:
    texts = iter(["4", "4", "5"])
    backend = MagicMock()
    backend.name = "server"
    backend.generate = AsyncMock(side_effect=lambda prompt: _res(next(texts)))
    [result] = asyncio.run(TestRunner(cache_dir=tmp_path).determinism(backend, [Prompt(id="p0", text="2+2?")], 3))
    assert result.identical_count == 2
    assert result.divergence_positions == []  # unknown, not a bogus 0
