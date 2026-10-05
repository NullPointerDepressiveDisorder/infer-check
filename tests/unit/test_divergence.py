from infer_check.analysis.divergence import (
    find_divergence_onset,
    sequence_similarity,
)
from infer_check.types import InferenceResult


def test_sequence_similarity() -> None:
    sim = sequence_similarity("hello world", "hello world")
    assert sim == 1.0

    sim = sequence_similarity("hello world", "hello friend")
    assert 0.0 < sim < 1.0


def test_find_divergence_onset() -> None:
    res1 = InferenceResult(
        prompt_id="test",
        backend_name="a",
        model_id="m",
        tokens=["Hello", " ", "world"],
        text="Hello world",
        latency_ms=10.0,
    )
    res2 = InferenceResult(
        prompt_id="test",
        backend_name="b",
        model_id="m",
        tokens=["Hello", " ", "world"],
        text="Hello world",
        latency_ms=10.0,
    )

    idx = find_divergence_onset(res1, res2)
    assert idx is None

    res3 = InferenceResult(
        prompt_id="test",
        backend_name="b",
        model_id="m",
        tokens=["Hello", " ", "friend"],
        text="Hello friend",
        latency_ms=10.0,
    )
    idx = find_divergence_onset(res1, res3)
    assert idx == 2  # Diverges at index 2
