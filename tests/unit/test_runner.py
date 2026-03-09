import pytest

from infer_check.runner import TestRunner
from infer_check.types import InferenceResult


@pytest.fixture
def runner() -> TestRunner:
    return TestRunner(cache_dir=".test_cache")


def test_compare_identical(runner: TestRunner) -> None:
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["A", "B", "C"],
        text="A B C",
        latency_ms=10.0,
    )
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["A", "B", "C"],
        text="A B C",
        latency_ms=10.0,
    )

    comp = runner._compare(baseline, test_res, threshold=0.5)
    assert comp.is_failure is False
    assert comp.text_similarity == 1.0
    assert comp.metadata["severity"] == "identical"
    assert comp.token_divergence_index is None


def test_compare_minor_difference(runner: TestRunner) -> None:
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["The", "quick", "fox"],
        text="The quick fox",
        latency_ms=10.0,
    )
    # Make it very similar to hit >= 0.8
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["The", "quick", "red", "fox"],
        text="The quick red fox",
        latency_ms=10.0,
    )

    comp = runner._compare(baseline, test_res, threshold=0.5)
    assert comp.is_failure is False
    assert comp.text_similarity >= 0.8
    assert comp.text_similarity < 1.0
    assert comp.metadata["severity"] == "minor"
    assert comp.token_divergence_index == 2


def test_compare_severe_difference(runner: TestRunner) -> None:
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["A", "valid", "JSON", "response"],
        text="A valid JSON response",
        latency_ms=10.0,
    )
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["Completely", "broken"],
        text="Completely broken",
        latency_ms=10.0,
    )

    comp = runner._compare(baseline, test_res, threshold=0.5)
    assert comp.is_failure is True
    assert comp.text_similarity < 0.5
    assert comp.metadata["severity"] == "severe"
    assert comp.token_divergence_index == 0


def test_compare_threshold_edge_cases(runner: TestRunner) -> None:
    # Setting threshold high so even minor diffs fail
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["A", "B", "C"],
        text="A B C",
        latency_ms=10.0,
    )
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["A", "B", "D"],
        text="A B D",
        latency_ms=10.0,
    )

    # The threshold logic in the runner is: is_failure = text_similarity < (1.0 - threshold)
    # So threshold is actually "allowed divergence ratio".
    # default 0.5 threshold -> passes as sim 0.8 >= 0.5
    comp_default = runner._compare(baseline, test_res, threshold=0.5)
    assert comp_default.is_failure is False

    # an allowed divergence of 0.1 (strict) means passing requires text_similarity >= 0.9
    comp_strict = runner._compare(baseline, test_res, threshold=0.1)
    assert comp_strict.is_failure is True
