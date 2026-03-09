"""Clustering analysis for inference failures."""

from __future__ import annotations

from infer_check.types import ComparisonResult, Prompt

__all__ = ["cluster_failures", "summarize_clusters"]


def cluster_failures(
    comparisons: list[ComparisonResult], prompts: dict[str, Prompt]
) -> dict[str, list[ComparisonResult]]:
    """Group failed comparisons by shared characteristics.

    Dimensions:
    - category: from prompt metadata
    - length: short (<100 tokens), medium (100-500 tokens), long (>500 tokens)
    - onset: early (<10 tokens), mid (10-50 tokens), late (>50 tokens)

    Args:
        comparisons: List of comparison results.
        prompts: Dictionary mapping prompt IDs to Prompt objects.

    Returns:
        Dictionary mapping cluster names to lists of failed comparison results.
    """
    clusters: dict[str, list[ComparisonResult]] = {}

    for comp in comparisons:
        if not comp.is_failure:
            continue

        prompt_id = comp.baseline.prompt_id
        prompt = prompts.get(prompt_id)

        if not prompt:
            # If prompt is missing for some reason, skip clustering this comparison.
            continue

        # 1. Category bucket
        cat_bucket = f"category:{prompt.category}"
        clusters.setdefault(cat_bucket, []).append(comp)

        # 2. Length bucket
        # Approximate tokens heavily by whitespace split, as we don't have the explicit tokenizer.
        # Another proxy could be len(prompt.text) / 4.
        # Using word count is simple and robust enough for text.
        approx_length = len(prompt.text.split())
        if approx_length < 100:
            len_bucket = "length:short"
        elif approx_length <= 500:
            len_bucket = "length:medium"
        else:
            len_bucket = "length:long"

        clusters.setdefault(len_bucket, []).append(comp)

        # 3. Onset bucket
        if comp.token_divergence_index is not None:
            idx = comp.token_divergence_index
            if idx < 10:
                onset_bucket = "onset:early"
            elif idx <= 50:
                onset_bucket = "onset:mid"
            else:
                onset_bucket = "onset:late"

            clusters.setdefault(onset_bucket, []).append(comp)

    return clusters


def summarize_clusters(clusters: dict[str, list[ComparisonResult]]) -> str:
    """Generate a human-readable summary of the largest failure clusters.

    Args:
        clusters: Dictionary mapping cluster names to lists of failed comparisons.

    Returns:
        A formatted string summarizing the top components of the failures.
    """
    if not clusters:
        return "No failures to cluster."

    # Find total unique failures across all clusters
    distinct_failures = {id(c) for c_list in clusters.values() for c in c_list}
    total_failures = len(distinct_failures)

    if total_failures == 0:
        return "No failures to cluster."

    # Sort clusters by number of failures (descending)
    sorted_clusters = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)

    parts: list[str] = []
    # Take top 3 to form the summary
    for i, (cluster_name, comp_list) in enumerate(sorted_clusters[:3]):
        pct = int(round(len(comp_list) / total_failures * 100))

        # Translate cluster names to human readable components
        if cluster_name.startswith("category:"):
            cat = cluster_name.split(":", 1)[1]
            readable_name = f"{cat} prompts"
        elif cluster_name.startswith("onset:"):
            onset = cluster_name.split(":", 1)[1]
            readable_name = f"{onset}-diverging outputs"
        elif cluster_name.startswith("length:"):
            length = cluster_name.split(":", 1)[1]
            readable_name = f"{length} prompts"
        else:
            readable_name = cluster_name

        if i == 0:
            parts.append(f"{readable_name} ({pct}% of failures)")
        else:
            parts.append(f"{readable_name} ({pct}%)")

    return f"Failures concentrated in: {', '.join(parts)}"
