"""Token IDs from servers that report them (mlx_lm.server), and the near-tie check.

Seen in gate 3a against mlx_lm.server: it reports token IDs, raw vocabulary
strings ("Ġdetermine") and a trailing "<|eot_id|>". Comparing the strings made
every output "diverge at token 1", and the string keys kept prefix KL from lining
up with the in-process backend's ID-keyed top-K lists.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from infer_check.analysis.divergence import divergence_is_near_tie
from infer_check.backends.openai_compat import OpenAICompatBackend, parse_content_logprobs
from infer_check.runner import TestRunner
from infer_check.types import InferenceResult, Prompt

EOT = {
    "id": 128009,
    "token": "<|eot_id|>",
    "logprob": -0.01,
    "top_logprobs": [{"id": 128009, "token": "<|eot_id|>", "logprob": -0.01}],
}


def _entry(tid: int, tok: str, lp: float, *alts: tuple[int, str, float]) -> dict[str, Any]:
    top = [{"id": tid, "token": tok, "logprob": lp}] + [{"id": i, "token": t, "logprob": v} for i, t, v in alts]
    return {"id": tid, "token": tok, "logprob": lp, "top_logprobs": top}


CONTENT = [_entry(791, "The", -0.2, (32, "A", -1.9)), _entry(4320, "Ġanswer", -0.1, (1121, "Ġresult", -2.5)), EOT]


# ── parsing ────────────────────────────────────────────────────────────


def test_ids_are_used_and_top_k_is_keyed_by_id() -> None:
    tokens, ids, lps, dists, meta = parse_content_logprobs(CONTENT, "The answer", "stop")
    assert ids == [791, 4320]
    assert tokens == ["The", "Ġanswer"] and lps == [-0.2, -0.1]
    assert meta[0] == {"id_0": 32, "id_1": 791} and dists[0] == [-1.9, -0.2]  # sorted by id


def test_trailing_end_of_turn_is_dropped_only_for_stopped_outputs() -> None:
    assert parse_content_logprobs(CONTENT, "The answer", "stop")[1] == [791, 4320]
    assert parse_content_logprobs(CONTENT, "The answer", "length")[1] == [791, 4320, 128009]
    # a literal "<b>"-like token that is part of the text stays
    html = [_entry(791, "The", -0.2), _entry(65, "<b>", -0.3)]
    assert parse_content_logprobs(html, "The<b>", "stop")[1] == [791, 65]


def test_without_ids_falls_back_to_strings() -> None:
    no_ids = [{"token": "The", "logprob": -0.2, "top_logprobs": [{"token": "The", "logprob": -0.2}]}]
    _, ids, _, _, meta = parse_content_logprobs(no_ids, "The", "stop")
    assert ids is None and meta[0] == {"id_0": "The"}


def _response(path: str, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("POST", f"http://127.0.0.1:8080{path}"))


def test_chat_endpoint_returns_token_ids() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8080", model_id="m", chat=True)
    body = {
        "choices": [{"message": {"content": "The answer"}, "finish_reason": "stop", "logprobs": {"content": CONTENT}}]
    }
    with patch("httpx.AsyncClient.post", return_value=_response("/v1/chat/completions", body)):
        res = asyncio.run(backend.generate(Prompt(id="p0", text="?")))
    assert res.token_ids == [791, 4320] and res.distribution_metadata is not None


def test_completions_endpoint_reads_the_chat_style_format() -> None:
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8080", model_id="m")
    body = {"choices": [{"text": "The answer", "finish_reason": "stop", "logprobs": {"content": CONTENT}}]}
    with patch("httpx.AsyncClient.post", return_value=_response("/v1/completions", body)):
        res = asyncio.run(backend.generate(Prompt(id="p0", text="?")))
    assert res.token_ids == [791, 4320]


# ── near-tie check ─────────────────────────────────────────────────────


def _res(ids: list[int], tops: list[dict[int, float]], text: str) -> InferenceResult:
    return InferenceResult(
        prompt_id="p0",
        backend_name="b",
        model_id="m",
        tokens=[str(i) for i in ids],
        token_ids=ids,
        logprobs=[tops[n][i] for n, i in enumerate(ids)],
        distributions=[list(t.values()) for t in tops],
        distribution_metadata=[{f"id_{j}": k for j, k in enumerate(t)} for t in tops],
        text=text,
        latency_ms=1.0,
    )


# Position 1 is a tie between token 20 ("minutes") and 21 ("time").
BASE = _res([10, 20], [{10: -0.1, 11: -3.0}, {20: -0.95, 21: -0.95, 22: -4.0}], "x minutes")
TIE = _res([10, 21], [{10: -0.1, 11: -3.0}, {21: -0.95, 20: -0.95, 22: -4.0}], "x time")
NOT_TIE = _res([10, 99], [{10: -0.1, 11: -3.0}, {99: -0.2, 98: -2.0}], "x banana")


def test_mutual_top_k_divergence_is_a_near_tie() -> None:
    assert divergence_is_near_tie(BASE, TIE, 1) is True


def test_divergence_outside_the_other_sides_top_k_is_not() -> None:
    assert divergence_is_near_tie(BASE, NOT_TIE, 1) is False


def test_cannot_check_without_ids_or_past_the_end() -> None:
    no_ids = BASE.model_copy(update={"token_ids": None})
    assert divergence_is_near_tie(no_ids, TIE, 1) is None
    assert divergence_is_near_tie(BASE, TIE, 2) is None
    assert divergence_is_near_tie(BASE, BASE, None) is None


@pytest.mark.parametrize(("test", "expected"), [(TIE, True), (NOT_TIE, False), (BASE, None)])
def test_compare_records_near_tie_and_prefix_kl(tmp_path: Path, test: InferenceResult, expected: bool | None) -> None:
    comp = TestRunner(cache_dir=tmp_path)._compare(BASE, test)
    assert comp.metadata["near_tie"] is expected
    assert comp.kl_method == "prefix"  # ID-keyed top-K on both sides now lines up


# ── same prompt on both sides ──────────────────────────────────────────


def test_chat_request_sends_only_the_user_message() -> None:
    """No empty system message: Qwen and Gemma render it as 5 extra prompt tokens."""
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:8080", model_id="m", chat=True, disable_thinking=True)
    body = {
        "choices": [{"message": {"content": "The answer"}, "finish_reason": "stop", "logprobs": {"content": CONTENT}}]
    }
    with patch("httpx.AsyncClient.post", return_value=_response("/v1/chat/completions", body)) as post:
        asyncio.run(backend.generate(Prompt(id="p0", text="What is 2+2?")))
    payload = post.call_args.kwargs["json"]
    assert payload["messages"] == [{"role": "user", "content": "What is 2+2?"}]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "thinking": False}


def test_different_prompt_lengths_are_flagged_and_get_no_near_tie_verdict(tmp_path: Path) -> None:
    base = BASE.model_copy(update={"metadata": {"prompt_tokens": 31}})
    same = TIE.model_copy(update={"metadata": {"prompt_tokens": 31}})
    longer = TIE.model_copy(update={"metadata": {"prompt_tokens": 36}})
    runner = TestRunner(cache_dir=tmp_path)
    ok = runner._compare(base, same)
    assert ok.metadata["prompt_mismatch"] is False and ok.metadata["near_tie"] is True
    bad = runner._compare(base, longer)
    assert bad.metadata["prompt_mismatch"] is True and bad.metadata["near_tie"] is None
