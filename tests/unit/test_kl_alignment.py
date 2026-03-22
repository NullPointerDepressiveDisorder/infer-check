from infer_check.runner import TestRunner
from infer_check.types import InferenceResult


def test_kl_alignment_mlx() -> None:
    """Test KL computation for full-vocab aligned distributions (MLX style)."""
    runner = TestRunner()

    # 2 tokens, 4 vocab size
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b1",
        model_id="m1",
        text="hi",
        tokens=["h", "i"],
        distributions=[[0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]],
        distribution_metadata=[{"is_aligned": 1}, {"is_aligned": 1}],
        latency_ms=10.0,
    )
    test = InferenceResult(
        prompt_id="p1",
        backend_name="b2",
        model_id="m1",
        text="hi",
        tokens=["h", "i"],
        distributions=[[0.1, 0.6, 0.2, 0.1], [0.1, 0.2, 0.6, 0.1]],
        distribution_metadata=[{"is_aligned": 1}, {"is_aligned": 1}],
        latency_ms=10.0,
    )

    result = runner._compare(baseline, test)
    assert result.kl_divergence is not None
    assert result.kl_divergence > 0


def test_kl_alignment_llama_cpp() -> None:
    """Test KL computation for top-K aligned distributions (llama.cpp style)."""
    runner = TestRunner()

    # Token 1:
    # Baseline: ID 10 (0.8), ID 11 (0.2)
    # Test: ID 10 (0.7), ID 12 (0.3)
    # Union: ID 10, 11, 12

    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b1",
        model_id="m1",
        text="hi",
        tokens=["h"],
        distributions=[[0.8, 0.2]],
        distribution_metadata=[{"id_0": 10, "id_1": 11}],
        latency_ms=10.0,
    )
    test = InferenceResult(
        prompt_id="p1",
        backend_name="b2",
        model_id="m1",
        text="hi",
        tokens=["h"],
        distributions=[[0.7, 0.3]],
        distribution_metadata=[{"id_0": 10, "id_1": 12}],
        latency_ms=10.0,
    )

    result = runner._compare(baseline, test)
    assert result.kl_divergence is not None
    # Manual check:
    # Baseline prob: ID 10: 0.8, ID 11: 0.2, ID 12: 0.0
    # Test prob:     ID 10: 0.7, ID 11: 0.0, ID 12: 0.3
    # With epsilon 1e-10, this should produce a valid KL.
    assert result.kl_divergence > 0


def test_kl_skips_unaligned() -> None:
    """Ensure KL is None if distributions cannot be aligned."""
    runner = TestRunner()

    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b1",
        model_id="m1",
        text="hi",
        tokens=["h"],
        distributions=[[0.8, 0.2]],
        # No metadata
        latency_ms=10.0,
    )
    test = InferenceResult(
        prompt_id="p1",
        backend_name="b2",
        model_id="m1",
        text="hi",
        tokens=["h"],
        distributions=[[0.7, 0.3]],
        # No metadata
        latency_ms=10.0,
    )

    result = runner._compare(baseline, test)
    assert result.kl_divergence is None


def test_kl_skips_mismatched_aligned() -> None:
    """Ensure KL is None if one has aligned metadata and other doesn't."""
    runner = TestRunner()

    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b1",
        model_id="m1",
        text="hi",
        tokens=["h"],
        distributions=[[0.8, 0.2]],
        distribution_metadata=[{"is_aligned": 1}],
        latency_ms=10.0,
    )
    test = InferenceResult(
        prompt_id="p1",
        backend_name="b2",
        model_id="m1",
        text="hi",
        tokens=["h"],
        distributions=[[0.7, 0.3]],
        # Missing metadata
        latency_ms=10.0,
    )

    result = runner._compare(baseline, test)
    assert result.kl_divergence is None
