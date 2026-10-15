import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from infer_check.types import ComparisonResult, InferenceResult, Prompt


def test_prompt_serialization() -> None:
    prompt = Prompt(id="test-1", text="Hello world", max_tokens=10)
    json_str = prompt.model_dump_json()
    data = json.loads(json_str)
    assert data["id"] == "test-1"
    assert data["text"] == "Hello world"
    assert data["max_tokens"] == 10

    prompt_loaded = Prompt.model_validate_json(json_str)
    assert prompt_loaded.id == prompt.id
    assert prompt_loaded.text == prompt.text


def test_inference_result() -> None:
    res = InferenceResult(
        prompt_id="test-1",
        backend_name="test_backend",
        model_id="test_model",
        tokens=["Hello", " world"],
        logprobs=[-0.1, -0.2],
        text="Hello world",
        latency_ms=10.5,
        tokens_per_second=100.0,
    )
    assert res.prompt_id == "test-1"
    assert res.backend_name == "test_backend"
    assert res.model_id == "test_model"
    assert res.tokens == ["Hello", " world"]
    assert res.text == "Hello world"


def test_comparison_result_is_failure() -> None:
    baseline = InferenceResult(prompt_id="test", backend_name="a", model_id="m", tokens=[], text="", latency_ms=10.0)
    test_run = InferenceResult(prompt_id="test", backend_name="b", model_id="m", tokens=[], text="", latency_ms=10.0)

    # Identical
    res1 = ComparisonResult(baseline=baseline, test=test_run, kl_divergence=0.0, text_similarity=1.0, is_failure=False)
    assert not res1.is_failure

    # Diverged
    res2 = ComparisonResult(
        baseline=baseline,
        test=test_run,
        token_divergence_index=5,
        kl_divergence=0.5,
        text_similarity=0.8,
        is_failure=True,
        failure_reason="Diverged at index 5",
    )
    assert res2.is_failure

    # Error string
    res3 = ComparisonResult(
        baseline=baseline,
        test=test_run,
        text_similarity=0.0,
        is_failure=True,
        failure_reason="Backend failure",
    )
    assert res3.is_failure


@pytest.mark.parametrize(
    ("answer_type", "expected", "message"),
    [
        ("boolean", "maybe", "yes/no/true/false"),
        ("integer", "3.5", "not an integer"),
        ("integer", "three", "not a number"),
        ("decimal", "NaN", "finite"),
        ("decimal", "Infinity", "finite"),
        ("choice", "AB", "single letter or digit"),
        ("choice", "?", "single letter or digit"),
    ],
)
def test_prompt_rejects_malformed_answer_keys(answer_type: str, expected: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Prompt(text="x", answer_type=answer_type, expected=expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("answer_type", "expected"),
    [
        ("integer", "18.0"),
        ("integer", "1e5"),
        ("integer", "-7"),
        ("integer", " 42 "),
        ("decimal", "0.125"),
        ("decimal", "-1.5e-3"),
        ("boolean", "Yes"),
        ("boolean", "false"),
        ("choice", "b"),
        ("choice", "3"),
    ],
)
def test_prompt_accepts_canonical_answer_keys(answer_type: str, expected: str) -> None:
    assert Prompt(text="x", answer_type=answer_type, expected=expected).is_graded  # type: ignore[arg-type]


def test_malformed_key_fails_when_the_suite_loads(tmp_path: Path) -> None:
    from infer_check.suites.loader import load_suite

    suite = tmp_path / "bad.jsonl"
    suite.write_text(
        '{"id": "q1", "text": "Is it?", "answer_type": "boolean", "expected": "maybe"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_suite(suite)
