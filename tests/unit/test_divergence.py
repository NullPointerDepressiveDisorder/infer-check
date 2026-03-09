import numpy as np

from infer_check.analysis.divergence import (
    find_divergence_onset,
    kl_divergence,
    sequence_similarity,
)
from infer_check.types import InferenceResult


def test_kl_divergence() -> None:
    # Identical distributions
    # these are logprobs: log(0.1) ~ -2.3, log(0.9) ~ -0.1
    p = [-2.3, -0.1]
    q = [-2.3, -0.1]
    # KL(P||Q) should be close to 0
    assert np.isclose(kl_divergence(p, q), 0.0, atol=1e-5)

    # Different distributions
    p = [-0.1, -2.3]
    q = [-2.3, -0.1]
    # Expect KL > 0
    kl = kl_divergence(p, q)
    assert kl > 0.5

    # Check handling of 0s
    p = [0.0, -100.0]  # ~1.0, ~0.0
    q = [-100.0, 0.0]  # ~0.0, ~1.0
    kl = kl_divergence(p, q)
    assert kl > 10.0


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
