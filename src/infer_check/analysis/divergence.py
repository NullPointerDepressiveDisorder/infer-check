"""Divergence analysis for inference results.

This module computes divergence metrics between baseline and test inference
results. It operates exclusively on previously collected InferenceResult and
ComparisonResult objects — it never calls backends.
"""

from __future__ import annotations

import difflib
from typing import Any

import numpy as np

from infer_check.types import ComparisonResult, InferenceResult

__all__ = [
    "kl_divergence",
    "token_level_divergence",
    "sequence_similarity",
    "find_divergence_onset",
    "compute_sweep_statistics",
]


def kl_divergence(baseline_logprobs: list[float], test_logprobs: list[float]) -> float:
    """Compute KL(baseline || test) for aligned logprob sequences.

    Converts logprobs to probabilities, clips to [1e-10, 1.0] for numerical
    stability, and uses log-sum-exp internally. Length mismatches are handled
    by truncating to the shorter sequence.

    Args:
        baseline_logprobs: Per-token log-probabilities from the baseline run.
        test_logprobs: Per-token log-probabilities from the test run.

    Returns:
        Mean KL divergence across token positions.

    Examples:
        >>> kl_divergence([-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3])
        0.0
        >>> score = kl_divergence([-0.1, -0.5], [-0.2, -1.0])
        >>> score > 0
        True
    """
    length = min(len(baseline_logprobs), len(test_logprobs))
    if length == 0:
        return 0.0

    baseline_lp = np.array(baseline_logprobs[:length], dtype=np.float64)
    test_lp = np.array(test_logprobs[:length], dtype=np.float64)

    # Convert logprobs to probabilities and clip for numerical stability.
    p = np.clip(np.exp(baseline_lp), 1e-10, 1.0)
    q = np.clip(np.exp(test_lp), 1e-10, 1.0)

    # KL(P || Q) = sum(p * log(p / q)) per position.
    # Use log-sum-exp style: log(p/q) = log(p) - log(q).
    log_p = np.log(p)
    log_q = np.log(q)
    kl_per_token = p * (log_p - log_q)

    return float(np.mean(kl_per_token))


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
