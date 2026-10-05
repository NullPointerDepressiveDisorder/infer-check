"""Divergence analysis for inference results.

This module computes divergence metrics between baseline and test inference
results. It operates exclusively on previously collected InferenceResult and
ComparisonResult objects — it never calls backends.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from infer_check.types import ComparisonResult, InferenceResult, TokenScores

__all__ = [
    "KLStats",
    "binned_kl",
    "teacher_forced_kl",
    "prefix_kl",
    "aggregate_kl",
    "token_level_divergence",
    "sequence_similarity",
    "find_divergence_onset",
    "compute_sweep_statistics",
]

KLMethod = Literal["teacher_forced", "prefix"]


@dataclass(frozen=True)
class KLStats:
    """Per-comparison KL summary. All KL values are KL(baseline || test) in nats."""

    method: KLMethod
    mean: float
    p99: float
    max: float
    top1_agreement: float
    per_token: list[float] = field(repr=False)

    @property
    def n_positions(self) -> int:
        return len(self.per_token)

    @classmethod
    def from_per_token(cls, method: KLMethod, per_token: npt.NDArray[np.float64], top1: float) -> KLStats:
        return cls(
            method=method,
            mean=float(per_token.mean()),
            p99=float(np.percentile(per_token, 99)),
            max=float(per_token.max()),
            top1_agreement=top1,
            per_token=[float(x) for x in per_token],
        )


def binned_kl(p_lp: npt.ArrayLike, q_lp: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Per-position KL(P || Q) over a shared set of tokens plus one "everything else" bin.

    Args:
        p_lp: ``[T, K]`` log-probs of P at K token IDs per position.
        q_lp: ``[T, K]`` log-probs of Q at the *same* token IDs.

    Both must be normalised over the full vocabulary (not just the K tokens),
    so the leftover mass ``1 - sum(p)`` is the probability of all other tokens.

    Merging tokens into one bin can only shrink KL (data-processing inequality),
    so this is a **lower bound** on the full-vocabulary KL. With P's top-64
    tokens the gap is negligible in practice.

    Returns:
        ``[T]`` array of non-negative per-position KL values.
    """
    p_log = np.atleast_2d(np.asarray(p_lp, dtype=np.float64))
    q_log = np.atleast_2d(np.asarray(q_lp, dtype=np.float64))
    if p_log.shape != q_log.shape:
        raise ValueError(f"shape mismatch: {p_log.shape} vs {q_log.shape}")

    p, q = np.exp(p_log), np.exp(q_log)
    kl = np.sum(p * (p_log - q_log), axis=-1)

    p_rest = np.clip(1.0 - p.sum(axis=-1), 0.0, 1.0)
    q_rest = np.clip(1.0 - q.sum(axis=-1), 1e-12, 1.0)
    tail = np.where(p_rest > 0, p_rest * (np.log(np.maximum(p_rest, 1e-12)) - np.log(q_rest)), 0.0)
    # Tiny negatives can appear from float error when P == Q.
    out: npt.NDArray[np.float64] = np.maximum(kl + tail, 0.0)
    return out


def teacher_forced_kl(ref: TokenScores, test: TokenScores) -> KLStats | None:
    """KL between two models that scored the *same* token sequence.

    Args:
        ref: Baseline scores, from ``score(..., top_k=K)``.
        test: Test-model scores for the same tokens, from
            ``score(..., at_ids=ref.top_ids)`` so ``test.at_logprobs`` lines up
            with ``ref.top_logprobs``.
    """
    if test.at_logprobs is None:
        raise ValueError("test scores must be computed with at_ids=ref.top_ids")
    if len(ref) != len(test):
        raise ValueError(f"length mismatch: {len(ref)} vs {len(test)} positions")
    if len(ref) == 0:
        return None
    per_token = binned_kl(ref.top_logprobs, test.at_logprobs)
    top1 = float(np.mean(ref.argmax == test.argmax))
    return KLStats.from_per_token("teacher_forced", per_token, top1)


def _topk_map(dist: list[float], meta: dict[str, int | str]) -> dict[int | str, float]:
    """Token ID -> log-prob for one stored top-K distribution (keys are ``id_<rank>``)."""
    out: dict[int | str, float] = {}
    for key, tok in meta.items():
        if key.startswith("id_"):
            rank = int(key[3:])
            if rank < len(dist):
                out[tok] = float(dist[rank])
    return out


def prefix_kl(baseline: InferenceResult, test: InferenceResult, divergence_index: int | None) -> KLStats | None:
    """Approximate KL from stored top-K distributions, for backends that can't score.

    Only positions where both generations still share the same context are
    usable: every position up to and including the first divergent token
    (position ``i`` is conditioned on ``tokens[:i]``). Past that point the two
    models are predicting continuations of different text, and comparing their
    distributions is meaningless — this was the source of the old ~19 nats/token
    figures.

    Each position uses the tokens present in *both* top-K lists, with all other
    mass merged into one bin: a lower bound on the true KL.

    Returns None if distributions or token IDs are unavailable on either side.
    """
    b_d, t_d = baseline.distributions, test.distributions
    b_m, t_m = baseline.distribution_metadata, test.distribution_metadata
    if not (b_d and t_d and b_m and t_m):
        return None

    n = min(len(b_d), len(t_d), len(b_m), len(t_m))
    if divergence_index is not None:
        n = min(n, divergence_index + 1)

    per_token: list[float] = []
    agree: list[bool] = []
    for i in range(n):
        p_map, q_map = _topk_map(b_d[i], b_m[i]), _topk_map(t_d[i], t_m[i])
        if not p_map or not q_map:
            continue
        # Mixed ID types (e.g. int IDs vs token strings) can't be aligned.
        if {type(k) for k in p_map} != {type(k) for k in q_map}:
            return None
        shared = [k for k in p_map if k in q_map]
        p_lp = np.array([p_map[k] for k in shared], dtype=np.float64)
        q_lp = np.array([q_map[k] for k in shared], dtype=np.float64)
        per_token.append(float(binned_kl(p_lp[None, :], q_lp[None, :])[0]) if shared else 0.0)
        agree.append(max(p_map, key=lambda k: p_map[k]) == max(q_map, key=lambda k: q_map[k]))

    if not per_token:
        return None
    return KLStats.from_per_token("prefix", np.array(per_token), float(np.mean(agree)))


def aggregate_kl(comparisons: list[ComparisonResult]) -> dict[str, Any]:
    """Pool per-token KL across comparisons without mixing methods.

    If any comparison has teacher-forced KL, only teacher-forced ones are used;
    otherwise prefix KL. The mean is pooled over all token positions (so long
    outputs weigh more, matching llama.cpp's "Mean KLD"); top-1 agreement is
    pooled the same way.

    Returns:
        ``{"method", "mean_kl", "mean_top1_agreement", "n_comparisons", "n_positions"}``
        with ``None`` values when no comparison has KL.
    """
    method: KLMethod | None = None
    if any(c.kl_method == "teacher_forced" for c in comparisons):
        method = "teacher_forced"
    elif any(c.kl_method == "prefix" for c in comparisons):
        method = "prefix"

    chosen = [c for c in comparisons if method is not None and c.kl_method == method and c.kl_positions]
    total = sum(c.kl_positions or 0 for c in chosen)
    if not chosen or total == 0:
        return {"method": None, "mean_kl": None, "mean_top1_agreement": None, "n_comparisons": 0, "n_positions": 0}

    kl_sum = sum((c.kl_divergence or 0.0) * (c.kl_positions or 0) for c in chosen)
    top1_sum = sum((c.top1_agreement or 0.0) * (c.kl_positions or 0) for c in chosen)
    return {
        "method": method,
        "mean_kl": kl_sum / total,
        "mean_top1_agreement": top1_sum / total,
        "n_comparisons": len(chosen),
        "n_positions": total,
    }


def token_level_divergence(baseline: InferenceResult, test: InferenceResult) -> list[float]:
    """Compute per-token absolute logprob difference between two results.

    Args:
        baseline: The baseline inference result (must have logprobs).
        test: The test inference result (must have logprobs).

    Returns:
        List of per-position absolute divergence values, or an empty list
        if either result lacks logprobs.

    Examples:
        >>> from infer_check.types import InferenceResult
        >>> b = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx", model_id="m",
        ...     tokens=["a"], logprobs=[-0.1], text="a", latency_ms=10.0,
        ... )
        >>> t = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx", model_id="m",
        ...     tokens=["a"], logprobs=[-0.3], text="a", latency_ms=10.0,
        ... )
        >>> token_level_divergence(b, t)
        [0.19999999999999998]
    """
    if baseline.logprobs is None or test.logprobs is None:
        return []

    length = min(len(baseline.logprobs), len(test.logprobs))
    b_lp = np.array(baseline.logprobs[:length], dtype=np.float64)
    t_lp = np.array(test.logprobs[:length], dtype=np.float64)

    return [float(x) for x in np.abs(b_lp - t_lp)]


def sequence_similarity(text_a: str, text_b: str) -> float:
    """Compute similarity score between two text sequences.

    Uses ``difflib.SequenceMatcher`` ratio as the primary metric. Also
    internally computes exact_match and common_prefix_length (available
    via the return docstring for documentation, but the float ratio is
    the returned value).

    Args:
        text_a: First text sequence.
        text_b: Second text sequence.

    Returns:
        Similarity ratio in [0, 1] from SequenceMatcher.

    Examples:
        >>> sequence_similarity("hello world", "hello world")
        1.0
        >>> sequence_similarity("abc", "xyz")
        0.0
        >>> 0.0 < sequence_similarity("hello", "help") < 1.0
        True
    """
    # Primary metric: SequenceMatcher ratio.
    ratio: float = difflib.SequenceMatcher(None, text_a, text_b).ratio()

    # Additional metrics (computed but not returned).
    _exact_match: bool = text_a == text_b
    _common_prefix_length: int = 0
    for idx, (a_char, b_char) in enumerate(zip(text_a, text_b, strict=False)):
        if a_char != b_char:
            break
        _common_prefix_length = idx + 1

    return ratio


def find_divergence_onset(baseline: InferenceResult, test: InferenceResult) -> int | None:
    """Find the first token index where outputs diverge.

    Comparison is done token-by-token on the ``tokens`` lists, not
    character-by-character on the text output.

    Args:
        baseline: The baseline inference result.
        test: The test inference result.

    Returns:
        Index of the first diverging token, or ``None`` if the token
        sequences are identical.

    Examples:
        >>> from infer_check.types import InferenceResult
        >>> b = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx", model_id="m",
        ...     tokens=["a", "b", "c"], text="abc", latency_ms=10.0,
        ... )
        >>> t = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx", model_id="m",
        ...     tokens=["a", "b", "x"], text="abx", latency_ms=10.0,
        ... )
        >>> find_divergence_onset(b, t)
        2
    """
    b_tokens = baseline.tokens
    t_tokens = test.tokens

    length = min(len(b_tokens), len(t_tokens))
    for i in range(length):
        if b_tokens[i] != t_tokens[i]:
            return i

    # If one sequence is longer, divergence starts at the end of the shorter.
    if len(b_tokens) != len(t_tokens):
        return length

    return None


def compute_sweep_statistics(
    comparisons: list[ComparisonResult],
) -> dict[str, Any]:
    """Compute aggregate statistics for a sweep of comparisons.

    Args:
        comparisons: All comparison results from a sweep (e.g., across
            quantization levels for a set of prompts).

    Returns:
        Dictionary containing:
            - **mean_kl** (*float*): Mean KL divergence across comparisons.
            - **median_kl** (*float*): Median KL divergence.
            - **failure_rate** (*float*): Fraction of comparisons where
              ``is_failure`` is True.
            - **failure_count** (*int*): Absolute number of failures.
            - **mean_text_similarity** (*float*): Mean text similarity score.
            - **worst_prompts** (*list[str]*): Top-5 highest-divergence
              prompt IDs (by KL divergence, descending).
            - **degradation_cliff** (*str | None*): The quantization level
              where the failure rate jumps more than 2× compared to the
              previous level, or None if no such cliff exists.

    Examples:
        >>> compute_sweep_statistics([])
        {'mean_kl': 0.0, 'median_kl': 0.0, 'failure_rate': 0.0, \
'failure_count': 0, 'mean_text_similarity': 0.0, 'worst_prompts': [], \
'degradation_cliff': None}
    """
    if not comparisons:
        return {
            "mean_kl": 0.0,
            "median_kl": 0.0,
            "failure_rate": 0.0,
            "failure_count": 0,
            "mean_text_similarity": 0.0,
            "worst_prompts": [],
            "degradation_cliff": None,
        }

    # KL divergence statistics.
    kl_values = np.array(
        [c.kl_divergence if c.kl_divergence is not None else 0.0 for c in comparisons],
        dtype=np.float64,
    )
    mean_kl = float(np.mean(kl_values))
    median_kl = float(np.median(kl_values))

    # Failure statistics.
    failure_count = sum(1 for c in comparisons if c.is_failure)
    failure_rate = failure_count / len(comparisons)

    # Text similarity.
    similarities = np.array([c.text_similarity for c in comparisons], dtype=np.float64)
    mean_text_similarity = float(np.mean(similarities))

    # Worst prompts by KL divergence (top-5, descending).
    indexed = sorted(
        enumerate(comparisons),
        key=lambda ic: ic[1].kl_divergence if ic[1].kl_divergence is not None else 0.0,
        reverse=True,
    )
    worst_prompts = [comparisons[i].baseline.prompt_id for i, _ in indexed[:5]]

    # Degradation cliff: find quantization level where failure_rate jumps >2×.
    degradation_cliff = _find_degradation_cliff(comparisons)

    return {
        "mean_kl": mean_kl,
        "median_kl": median_kl,
        "failure_rate": failure_rate,
        "failure_count": failure_count,
        "mean_text_similarity": mean_text_similarity,
        "worst_prompts": worst_prompts,
        "degradation_cliff": degradation_cliff,
    }


def _find_degradation_cliff(
    comparisons: list[ComparisonResult],
) -> str | None:
    """Find the quantization level where failure rate jumps >2×.

    Args:
        comparisons: All comparisons from a sweep.

    Returns:
        The quantization level string where the cliff occurs, or None.
    """
    # Group comparisons by quantization level of the test result.
    quant_groups: dict[str, list[ComparisonResult]] = {}
    for c in comparisons:
        quant = c.test.quantization or "unknown"
        quant_groups.setdefault(quant, []).append(c)

    if len(quant_groups) < 2:
        return None

    # Sort quantization levels lexicographically (e.g., "q4_0", "q8_0").
    sorted_levels = sorted(quant_groups.keys())
    prev_rate: float | None = None

    for level in sorted_levels:
        group = quant_groups[level]
        rate = sum(1 for c in group if c.is_failure) / len(group)
        if prev_rate is not None and prev_rate > 0 and rate > 2 * prev_rate:
            return level
        # Also catch jump from 0 to non-trivial failure rate.
        if prev_rate is not None and prev_rate == 0 and rate > 0:
            return level
        prev_rate = rate

    return None
