"""How TestRunner._compare fills in KL fields."""

import numpy as np

from infer_check.analysis.divergence import KLStats
from infer_check.runner import TestRunner
from infer_check.types import InferenceResult

LP = [float(np.log(0.8)), float(np.log(0.2))]


def _res(ids: list[int], with_dists: bool = True) -> InferenceResult:
    return InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=[str(i) for i in ids],
        token_ids=ids,
        text=" ".join(str(i) for i in ids),
        distributions=[LP] * len(ids) if with_dists else None,
        distribution_metadata=[{"id_0": 1, "id_1": 2}] * len(ids) if with_dists else None,
        latency_ms=1.0,
    )


def test_falls_back_to_prefix_kl_on_shared_context_only() -> None:
    runner = TestRunner(cache_dir=".test_cache")
    comp = runner._compare(_res([1, 1, 1, 1]), _res([1, 2, 1, 1]))
    assert comp.token_divergence_index == 1
    assert comp.kl_method == "prefix"
    # Positions 0 and 1 are conditioned on the same text; 2 and 3 are not.
    assert comp.kl_positions == 2
    assert comp.kl_per_token is not None and len(comp.kl_per_token) == 2


def test_uses_supplied_teacher_forced_stats() -> None:
    runner = TestRunner(cache_dir=".test_cache")
    stats = KLStats.from_per_token("teacher_forced", np.array([0.01, 0.03, 0.02]), top1=1.0)
    comp = runner._compare(_res([1, 1, 1]), _res([1, 2, 1]), kl=stats)
    assert comp.kl_method == "teacher_forced"
    assert comp.kl_divergence == stats.mean
    assert comp.kl_max == 0.03
    assert comp.kl_positions == 3
    assert comp.top1_agreement == 1.0


def test_no_kl_without_distributions() -> None:
    runner = TestRunner(cache_dir=".test_cache")
    comp = runner._compare(_res([1, 2], with_dists=False), _res([1, 2], with_dists=False))
    assert comp.kl_divergence is None
    assert comp.kl_method is None
    assert comp.top1_agreement is None
