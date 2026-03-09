"""Determinism analysis for inference results.

This module quantifies non-determinism across repeated runs of the same
prompt. It operates exclusively on previously collected InferenceResult
objects — it never calls backends.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from infer_check.types import DeterminismResult, InferenceResult

__all__ = [
    "analyze_determinism",
    "determinism_summary",
]


def analyze_determinism(
    results_per_prompt: dict[str, list[InferenceResult]],
) -> list[DeterminismResult]:
    """Analyze determinism across repeated inference runs per prompt.

    For each prompt, counts how many runs produced identical text output,
    identifies all unique outputs and their frequencies, and finds the
    first token position where any pair of runs diverges.

    Args:
        results_per_prompt: Mapping from prompt_id to a list of
            ``InferenceResult`` objects, one per run.

    Returns:
        A list of ``DeterminismResult`` objects, one per prompt.

    Examples:
        >>> from infer_check.types import InferenceResult
        >>> r1 = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx", model_id="m",
        ...     tokens=["a", "b"], text="ab", latency_ms=10.0,
        ... )
        >>> r2 = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx", model_id="m",
        ...     tokens=["a", "b"], text="ab", latency_ms=11.0,
        ... )
        >>> results = analyze_determinism({"p1": [r1, r2]})
        >>> results[0].determinism_score
        1.0
        >>> results[0].identical_count
        2
    """
    determinism_results: list[DeterminismResult] = []

    for prompt_id, runs in results_per_prompt.items():
        if not runs:
            continue

        # Use the first run for metadata.
        first = runs[0]

        # Count identical outputs.
        text_counts = Counter(r.text for r in runs)
        most_common_count = text_counts.most_common(1)[0][1]

        # Determinism score: fraction of runs matching the most common output.
        score = most_common_count / len(runs)

        # Find all divergence positions across every pair of runs.
        divergence_positions = _find_all_divergence_positions(runs)

        determinism_results.append(
            DeterminismResult(
                prompt_id=prompt_id,
                model_id=first.model_id,
                backend_name=first.backend_name,
                quantization=first.quantization,
                num_runs=len(runs),
                identical_count=most_common_count,
                divergence_positions=divergence_positions,
                determinism_score=score,
            )
        )

    return determinism_results


def _find_all_divergence_positions(runs: list[InferenceResult]) -> list[int]:
    """Find the first divergence position for each pair of runs.

    Args:
        runs: List of inference results for the same prompt.

    Returns:
        Sorted list of unique first-divergence token indices found
        across all pairs. Empty if all runs are identical.
    """
    positions: set[int] = set()
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pos = _first_token_divergence(runs[i].tokens, runs[j].tokens)
            if pos is not None:
                positions.add(pos)
    return sorted(positions)


def _first_token_divergence(tokens_a: list[str], tokens_b: list[str]) -> int | None:
    """Find the first token index where two token lists diverge.

    Args:
        tokens_a: Token list from the first run.
        tokens_b: Token list from the second run.

    Returns:
        Index of the first diverging token, or None if identical.
    """
    length = min(len(tokens_a), len(tokens_b))
    for i in range(length):
        if tokens_a[i] != tokens_b[i]:
            return i
    if len(tokens_a) != len(tokens_b):
        return length
    return None


def determinism_summary(
    results: list[DeterminismResult],
) -> dict[str, Any]:
    """Compute aggregate statistics over a set of determinism results.

    Args:
        results: List of ``DeterminismResult`` objects (one per prompt).

    Returns:
        Dictionary containing:
            - **overall_determinism_score** (*float*): Mean determinism
              score across all prompts.
            - **fully_deterministic_count** (*int*): Number of prompts
              where all runs matched (score == 1.0).
            - **non_deterministic_count** (*int*): Number of prompts
              where at least one run diverged.
            - **mean_divergence_position** (*float | None*): Average
              first-diverge token index across non-deterministic prompts,
              or None if all prompts are deterministic.
            - **worst_prompts** (*list[str]*): Top-5 least deterministic
              prompt IDs (by determinism_score, ascending).

    Examples:
        >>> from infer_check.types import DeterminismResult
        >>> d = DeterminismResult(
        ...     prompt_id="p1", model_id="m", backend_name="mlx",
        ...     num_runs=3, identical_count=3, divergence_positions=[],
        ...     determinism_score=1.0,
        ... )
        >>> summary = determinism_summary([d])
        >>> summary["overall_determinism_score"]
        1.0
        >>> summary["fully_deterministic_count"]
        1
    """
    if not results:
        return {
            "overall_determinism_score": 0.0,
            "fully_deterministic_count": 0,
            "non_deterministic_count": 0,
            "mean_divergence_position": None,
            "worst_prompts": [],
        }

    scores = np.array([r.determinism_score for r in results], dtype=np.float64)
    overall_determinism_score = float(np.mean(scores))

    fully_deterministic_count = int(np.sum(scores == 1.0))
    non_deterministic_count = len(results) - fully_deterministic_count

    # Mean divergence position across non-deterministic prompts.
    diverge_positions: list[int] = []
    for r in results:
        if r.divergence_positions:
            diverge_positions.append(min(r.divergence_positions))

    mean_divergence_position: float | None = None
    if diverge_positions:
        mean_divergence_position = float(np.mean(diverge_positions))

    # Worst prompts: lowest determinism scores.
    indexed = sorted(
        results,
        key=lambda r: r.determinism_score,
    )
    worst_prompts = [r.prompt_id for r in indexed[:5]]

    return {
        "overall_determinism_score": overall_determinism_score,
        "fully_deterministic_count": fully_deterministic_count,
        "non_deterministic_count": non_deterministic_count,
        "mean_divergence_position": mean_divergence_position,
        "worst_prompts": worst_prompts,
    }
