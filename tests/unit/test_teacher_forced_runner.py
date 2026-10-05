"""Teacher-forced KL wiring through compare / sweep / diff, using an analytic stub model.

Each stub "model" emits the same distribution at every position:
``log_softmax(BASE_LOGITS / tau)``. Greedy generation always picks token 0, so
all models generate identical text, yet their distributions differ by ``tau`` —
exactly the case where text similarity says "identical" and KL must not.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from infer_check.backends.base import BackendAdapter
from infer_check.runner import TestRunner
from infer_check.types import InferenceResult, Prompt, TokenScores

BASE_LOGITS = np.array([3.0, 1.5, 1.0, 0.0, -1.0, -2.0])
N_TOKENS = 5


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def exact_kl(tau_p: float, tau_q: float) -> float:
    p_lp, q_lp = log_softmax(BASE_LOGITS / tau_p), log_softmax(BASE_LOGITS / tau_q)
    return float(np.sum(np.exp(p_lp) * (p_lp - q_lp)))


class AnalyticBackend(BackendAdapter):
    def __init__(self, name: str, tau: float, scoring: bool = True) -> None:
        self._name = name
        self._lp = log_softmax(BASE_LOGITS / tau)
        self._scoring = scoring
        self.score_calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return f"analytic-{self._name}"

    @property
    def supports_concurrency(self) -> bool:
        return False

    @property
    def supports_scoring(self) -> bool:
        return self._scoring

    async def generate(self, prompt: Prompt) -> InferenceResult:
        order = np.argsort(-self._lp)
        dist = [float(self._lp[i]) for i in order]
        meta: dict[str, int | str] = {f"id_{r}": int(i) for r, i in enumerate(order)}
        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self._name,
            model_id=self.model_id,
            tokens=["a"] * N_TOKENS,
            token_ids=[0] * N_TOKENS,
            distributions=[dist] * N_TOKENS,
            distribution_metadata=[meta] * N_TOKENS,
            text="a" * N_TOKENS,
            latency_ms=1.0,
            metadata={"finish_reason": "stop"},
        )

    async def score(
        self,
        prompt: Prompt,
        target_ids: list[int],
        *,
        top_k: int = 64,
        at_ids: Any = None,
    ) -> TokenScores:
        if not self._scoring:
            raise NotImplementedError
        self.score_calls.append({"prompt": prompt.id, "n": len(target_ids), "top_k": top_k, "at": at_ids is not None})
        t = len(target_ids)
        k = min(top_k, len(self._lp))
        order = np.argsort(-self._lp)[:k]
        return TokenScores(
            top_ids=np.tile(order.astype(np.int64), (t, 1)),
            top_logprobs=np.tile(self._lp[order].astype(np.float32), (t, 1)),
            argmax=np.full(t, int(np.argmax(self._lp)), dtype=np.int64),
            at_logprobs=None if at_ids is None else self._lp[at_ids].astype(np.float32),
        )

    async def health_check(self) -> bool:
        return True

    async def cleanup(self) -> None:
        pass


PROMPTS = [Prompt(id=f"p{i}", text=f"prompt {i}") for i in range(3)]


@pytest.fixture
def runner(tmp_path: Path) -> TestRunner:
    return TestRunner(cache_dir=tmp_path)


# ── compare ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_same_model_is_zero(runner: TestRunner) -> None:
    res = await runner.compare(AnalyticBackend("a", 1.0), AnalyticBackend("b", 1.0), PROMPTS)
    assert res.kl_method == "teacher_forced"
    assert res.mean_kl_divergence == pytest.approx(0.0, abs=1e-6)
    assert res.mean_top1_agreement == 1.0


@pytest.mark.asyncio
async def test_compare_reports_exact_kl_when_text_is_identical(runner: TestRunner) -> None:
    a, b = AnalyticBackend("a", 1.0), AnalyticBackend("b", 2.0)
    res = await runner.compare(a, b, PROMPTS)

    # Text is identical, so similarity can't see the difference...
    assert all(c.text_similarity == 1.0 for c in res.comparisons)
    # ...but teacher-forced KL does, and matches the closed form (vocab < top-K, so exact).
    assert res.kl_method == "teacher_forced"
    assert res.mean_kl_divergence == pytest.approx(exact_kl(1.0, 2.0), rel=1e-5)
    for c in res.comparisons:
        assert c.kl_positions == N_TOKENS
        assert c.kl_per_token is not None and len(c.kl_per_token) == N_TOKENS

    # A scored its own tokens for the reference; B scored A's tokens at A's top IDs.
    assert all(call["top_k"] > 0 and not call["at"] for call in a.score_calls)
    assert all(call["at"] for call in b.score_calls)
    assert len(a.score_calls) == len(b.score_calls) == len(PROMPTS)


@pytest.mark.asyncio
async def test_compare_falls_back_to_prefix_when_one_side_cannot_score(runner: TestRunner) -> None:
    res = await runner.compare(AnalyticBackend("a", 1.0), AnalyticBackend("b", 2.0, scoring=False), PROMPTS)
    assert res.kl_method == "prefix"
    assert res.mean_kl_divergence is not None and res.mean_kl_divergence > 0


# ── sweep ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_kl_per_level_and_self_check(runner: TestRunner) -> None:
    backends: dict[str, BackendAdapter] = {
        "bf16": AnalyticBackend("bf16", 1.0),
        "8bit": AnalyticBackend("8bit", 1.2),
        "4bit": AnalyticBackend("4bit", 2.0),
    }
    res = await runner.sweep(backends, PROMPTS, ["bf16", "8bit", "4bit"], model_id="analytic")
    by_quant = res.summary["kl_by_quant"]

    # Self-check re-scores with the same model: exactly zero.
    assert by_quant["bf16"]["method"] == "teacher_forced"
    assert by_quant["bf16"]["mean_kl"] == pytest.approx(0.0, abs=1e-6)
    # Levels are reported separately and ordered by how far they drift.
    assert by_quant["8bit"]["mean_kl"] == pytest.approx(exact_kl(1.0, 1.2), rel=1e-5)
    assert by_quant["4bit"]["mean_kl"] == pytest.approx(exact_kl(1.0, 2.0), rel=1e-5)
    assert by_quant["8bit"]["mean_kl"] < by_quant["4bit"]["mean_kl"]
    # The back-compat pooled number excludes the self-check.
    assert res.summary["kl_method"] == "teacher_forced"
    assert res.summary["mean_kl_divergence"] > 0


# ── diff ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diff_skips_reference_scoring_when_no_test_backend_can_score(runner: TestRunner) -> None:
    base = AnalyticBackend("base", 1.0)
    comps = await runner.diff(base, [AnalyticBackend("http", 2.0, scoring=False)], PROMPTS)
    assert base.score_calls == []  # no wasted forward passes
    assert {c.kl_method for c in comps} == {"prefix"}


@pytest.mark.asyncio
async def test_diff_uses_teacher_forced_when_test_backend_can_score(runner: TestRunner) -> None:
    comps = await runner.diff(AnalyticBackend("base", 1.0), [AnalyticBackend("mlx", 2.0)], PROMPTS)
    assert {c.kl_method for c in comps} == {"teacher_forced"}
