"""KL math: binned lower bound, teacher-forced stats, prefix fallback, aggregation."""

import numpy as np
import pytest

from infer_check.analysis.divergence import (
    KLStats,
    aggregate_kl,
    binned_kl,
    prefix_kl,
    teacher_forced_kl,
)
from infer_check.types import ComparisonResult, InferenceResult, TokenScores

# ── binned_kl ───────────────────────────────────────────────────────


def test_identical_distributions_give_zero() -> None:
    lp = np.log(np.array([[0.5, 0.3, 0.1], [0.7, 0.2, 0.05]]))
    assert np.allclose(binned_kl(lp, lp), 0.0, atol=1e-12)


def test_exact_when_all_mass_is_covered() -> None:
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.5, 0.3])
    exact = float(np.sum(p * np.log(p / q)))
    assert binned_kl(np.log(p)[None], np.log(q)[None])[0] == pytest.approx(exact)


def test_is_a_lower_bound_on_full_vocab_kl() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.dirichlet(np.ones(100) * 0.3)
        q = rng.dirichlet(np.ones(100) * 0.3)
        exact = float(np.sum(p * np.log(p / q)))
        top = np.argsort(-p)[:10]
        approx = binned_kl(np.log(p[top])[None], np.log(q[top])[None])[0]
        assert -1e-9 <= approx <= exact + 1e-9


def test_tail_bin_catches_mass_moved_off_the_top_k() -> None:
    # Q moved most of P's top-token mass elsewhere; the tail bin must see it.
    p_lp = np.log(np.array([[0.9]]))
    q_lp = np.log(np.array([[0.1]]))
    assert binned_kl(p_lp, q_lp)[0] > 1.0


def test_zero_probability_bins_contribute_zero() -> None:
    lp = np.array([[0.0, -np.inf]])
    assert binned_kl(lp, lp)[0] == 0.0
    # A -inf bin in P adds nothing, even against a finite Q (used to be 0 * -inf = NaN).
    out = binned_kl([[np.log(0.5), np.log(0.5), -np.inf]], [[np.log(0.5), np.log(0.4), np.log(0.1)]])
    assert out[0] == pytest.approx(0.5 * np.log(0.5 / 0.4))


def test_mass_where_q_is_zero_is_infinite_divergence() -> None:
    assert binned_kl([[np.log(0.5), np.log(0.5)]], [[0.0, -np.inf]])[0] == np.inf


def test_teacher_forced_stats_stay_finite_with_masked_tokens() -> None:
    # Fewer unmasked tokens than top-K: the tail of the reference top-K is -inf.
    lp = np.array([[np.log(0.7), np.log(0.3), -np.inf, -np.inf]])
    ref = _scores(lp, [0])
    stats = teacher_forced_kl(ref, _scores(np.zeros((1, 0)), [0], at=lp))
    assert stats is not None and stats.mean == 0.0 and np.isfinite(stats.per_token).all()


def test_rounding_overshoot_in_q_tail_stays_finite() -> None:
    # Real float32 top-K sums overshoot 1 by ~1e-6. That must not read as an empty
    # Q tail: treating it as zero made KL infinite on ordinary bf16-vs-4-bit runs.
    p = np.array([[0.6, 0.399995]])  # tail 5e-6
    q = np.array([[0.6, 0.400002]])  # sums to 1.000002
    out = binned_kl(np.log(p), np.log(q))[0]
    assert np.isfinite(out) and out < 1e-3


def test_empty_q_tail_under_real_p_mass_is_a_large_lower_bound() -> None:
    # P=0.5 on the selected token, Q=1: the true KL is infinite. The floored tail
    # reports a large finite value, still a valid lower bound.
    out = binned_kl(np.log([[0.5]]), np.log([[1.0]]))[0]
    assert 10 < out < np.inf


def test_p_tail_at_or_below_zero_contributes_nothing() -> None:
    lp = np.log([[0.6, 0.400001]])  # P sums to 1.000001
    assert binned_kl(lp, lp)[0] == 0.0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        binned_kl(np.zeros((2, 3)), np.zeros((2, 4)))


# ── teacher_forced_kl ───────────────────────────────────────────────


def _scores(top_lp: np.ndarray, argmax: list[int], at: np.ndarray | None = None) -> TokenScores:
    t, k = top_lp.shape
    return TokenScores(
        top_ids=np.tile(np.arange(k, dtype=np.int64), (t, 1)),
        top_logprobs=top_lp.astype(np.float32),
        argmax=np.array(argmax, dtype=np.int64),
        at_logprobs=None if at is None else at.astype(np.float32),
    )


def test_teacher_forced_self_comparison_is_zero_with_full_agreement() -> None:
    lp = np.log(np.array([[0.7, 0.2], [0.6, 0.3], [0.5, 0.4]]))
    ref = _scores(lp, [0, 0, 0])
    test = _scores(np.zeros((3, 0)), [0, 0, 0], at=lp)
    stats = teacher_forced_kl(ref, test)
    assert stats is not None
    assert stats.method == "teacher_forced"
    assert stats.mean == pytest.approx(0.0, abs=1e-6)
    assert stats.top1_agreement == 1.0
    assert stats.n_positions == 3


def test_teacher_forced_detects_disagreement() -> None:
    ref_lp = np.log(np.array([[0.9, 0.05], [0.9, 0.05]]))
    test_lp = np.log(np.array([[0.9, 0.05], [0.05, 0.9]]))
    stats = teacher_forced_kl(_scores(ref_lp, [0, 0]), _scores(np.zeros((2, 0)), [0, 1], at=test_lp))
    assert stats is not None
    assert stats.per_token[0] == pytest.approx(0.0, abs=1e-6)
    assert stats.per_token[1] > 1.0
    assert stats.max == pytest.approx(stats.per_token[1])
    assert stats.top1_agreement == 0.5


def test_teacher_forced_requires_at_logprobs_and_matching_lengths() -> None:
    lp = np.log(np.array([[0.7, 0.2]]))
    with pytest.raises(ValueError, match="at_ids"):
        teacher_forced_kl(_scores(lp, [0]), _scores(lp, [0]))
    with pytest.raises(ValueError, match="length"):
        teacher_forced_kl(_scores(lp, [0]), _scores(np.zeros((2, 0)), [0, 0], at=np.zeros((2, 2))))


def test_teacher_forced_empty_sequence_is_none() -> None:
    empty = _scores(np.zeros((0, 2)), [], at=np.zeros((0, 2)))
    assert teacher_forced_kl(empty, empty) is None


# ── prefix_kl ───────────────────────────────────────────────────────


def _res(ids: list[int], dists: list[list[float]], metas: list[dict[str, int | str]]) -> InferenceResult:
    return InferenceResult(
        prompt_id="p",
        backend_name="b",
        model_id="m",
        tokens=[str(i) for i in ids],
        token_ids=ids,
        text="".join(str(i) for i in ids),
        distributions=dists,
        distribution_metadata=metas,
        latency_ms=1.0,
    )


def test_prefix_stops_after_the_first_divergent_position() -> None:
    lp = [float(np.log(0.8)), float(np.log(0.2))]
    meta: dict[str, int | str] = {"id_0": 1, "id_1": 2}
    # Five positions stored, generations diverge at index 1: positions 0 and 1 share context.
    base = _res([1, 1, 1, 1, 1], [lp] * 5, [meta] * 5)
    test = _res([1, 2, 1, 1, 1], [lp] * 5, [meta] * 5)
    stats = prefix_kl(base, test, divergence_index=1)
    assert stats is not None
    assert stats.method == "prefix"
    assert stats.n_positions == 2


def test_prefix_uses_all_positions_when_identical() -> None:
    lp = [float(np.log(0.8)), float(np.log(0.2))]
    meta: dict[str, int | str] = {"id_0": 1, "id_1": 2}
    base = _res([1, 1, 1], [lp] * 3, [meta] * 3)
    stats = prefix_kl(base, base, divergence_index=None)
    assert stats is not None
    assert stats.n_positions == 3
    assert stats.mean == pytest.approx(0.0, abs=1e-9)


def test_prefix_aligns_by_token_id_not_rank() -> None:
    # Same distribution, but the test side lists the tokens in the other order.
    base = _res([1], [[float(np.log(0.8)), float(np.log(0.2))]], [{"id_0": 10, "id_1": 11}])
    test = _res([1], [[float(np.log(0.2)), float(np.log(0.8))]], [{"id_0": 11, "id_1": 10}])
    stats = prefix_kl(base, test, divergence_index=None)
    assert stats is not None
    assert stats.mean == pytest.approx(0.0, abs=1e-9)


def test_prefix_none_without_ids_or_with_mixed_id_types() -> None:
    lp = [[-0.2, -1.8]]
    no_ids = InferenceResult(
        prompt_id="p", backend_name="b", model_id="m", tokens=["a"], text="a", distributions=lp, latency_ms=1.0
    )
    assert prefix_kl(no_ids, no_ids, None) is None
    ints = _res([1], lp, [{"id_0": 1, "id_1": 2}])
    strs = _res([1], lp, [{"id_0": "a", "id_1": "b"}])
    assert prefix_kl(ints, strs, None) is None


# ── aggregate_kl ────────────────────────────────────────────────────


def _comp(method: str | None, kl: float | None, n: int | None, top1: float | None = 1.0) -> ComparisonResult:
    r = InferenceResult(prompt_id="p", backend_name="b", model_id="m", tokens=[], text="", latency_ms=1.0)
    return ComparisonResult(
        baseline=r,
        test=r,
        text_similarity=1.0,
        is_failure=False,
        kl_divergence=kl,
        kl_method=method,  # type: ignore[arg-type]
        kl_positions=n,
        top1_agreement=top1,
    )


def test_aggregate_pools_over_positions() -> None:
    agg = aggregate_kl([_comp("teacher_forced", 0.1, 10, 1.0), _comp("teacher_forced", 0.4, 30, 0.5)])
    assert agg["method"] == "teacher_forced"
    assert agg["mean_kl"] == pytest.approx((0.1 * 10 + 0.4 * 30) / 40)
    assert agg["mean_top1_agreement"] == pytest.approx((1.0 * 10 + 0.5 * 30) / 40)
    assert agg["n_positions"] == 40


def test_aggregate_never_mixes_methods() -> None:
    agg = aggregate_kl([_comp("teacher_forced", 0.1, 10), _comp("prefix", 9.0, 10)])
    assert agg["method"] == "teacher_forced"
    assert agg["mean_kl"] == pytest.approx(0.1)
    assert agg["n_comparisons"] == 1


def test_aggregate_falls_back_to_prefix_then_none() -> None:
    assert aggregate_kl([_comp("prefix", 0.2, 5)])["method"] == "prefix"
    empty = aggregate_kl([_comp(None, None, None, None)])
    assert empty["method"] is None and empty["mean_kl"] is None


def test_klstats_summary_fields() -> None:
    stats = KLStats.from_per_token("teacher_forced", np.array([0.0, 0.1, 1.0]), 0.5)
    assert stats.mean == pytest.approx(1.1 / 3)
    assert stats.max == 1.0
    assert stats.n_positions == 3
