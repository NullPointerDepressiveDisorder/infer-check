"""GitHub issue formatter for infer-check comparison failures.

Produces Markdown-formatted GitHub issue bodies from ``ComparisonResult``
objects. The output is designed to be filed directly as a reproducible bug
report against an LLM inference engine.
"""

from __future__ import annotations

import platform
from collections import defaultdict

from infer_check.types import ComparisonResult

__all__ = [
    "format_issue",
    "format_issues_batch",
]

_REPRO_CMD = (
    "infer-check diff --model {model_id} --backends {baseline},{test} --prompts <prompt_file>"
)


def format_issue(comparison: ComparisonResult, include_repro: bool = True) -> str:
    """Generate a Markdown-formatted GitHub issue body for a single failure.

    Args:
        comparison: A ``ComparisonResult`` describing the divergence between
            a baseline and test backend.
        include_repro: If ``True``, append a **Reproduction** section with
            the ``infer-check diff`` CLI command needed to reproduce the bug.

    Returns:
        A Markdown string suitable for use as a GitHub issue body.

    Examples:
        >>> from infer_check.types import InferenceResult, ComparisonResult
        >>> b = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx-lm", model_id="Llama-3.1-8B",
        ...     quantization="q8_0", tokens=["Hi"], text="Hi", latency_ms=10.0,
        ... )
        >>> t = InferenceResult(
        ...     prompt_id="p1", backend_name="llama-cpp", model_id="Llama-3.1-8B",
        ...     quantization="q4_0", tokens=["Hello"], text="Hello", latency_ms=9.0,
        ... )
        >>> c = ComparisonResult(
        ...     baseline=b, test=t, text_similarity=0.5, is_failure=True,
        ...     kl_divergence=0.3, token_divergence_index=0, failure_reason="Low sim",
        ... )
        >>> body = format_issue(c)
        >>> "llama-cpp" in body and "q4_0" in body
        True
    """
    baseline = comparison.baseline
    test = comparison.test

    category = test.metadata.get("category", baseline.metadata.get("category", "general"))
    quant = test.quantization or "unspecified"
    baseline_quant = baseline.quantization or "unspecified"

    similarity_pct = f"{comparison.text_similarity * 100:.1f}"
    kl_str = f"{comparison.kl_divergence:.4f}" if comparison.kl_divergence is not None else "N/A"
    div_idx = (
        str(comparison.token_divergence_index)
        if comparison.token_divergence_index is not None
        else "N/A"
    )

    platform_info = f"{platform.system()} {platform.machine()} (Python {platform.python_version()})"

    # Truncate outputs for the issue body to keep it readable.
    baseline_text = (baseline.text[:500] + "…") if len(baseline.text) > 500 else baseline.text
    test_text = (test.text[:500] + "…") if len(test.text) > 500 else test.text

    # Baseline token at divergence point (if available).
    baseline_token_line = ""
    test_token_line = ""
    if comparison.token_divergence_index is not None:
        idx = comparison.token_divergence_index
        if idx < len(baseline.tokens):
            b_tok = baseline.tokens[idx]
            b_lp = (
                f"{baseline.logprobs[idx]:.3f}"
                if baseline.logprobs and idx < len(baseline.logprobs)
                else "N/A"
            )
            baseline_token_line = f"  Baseline token: `{b_tok}` (logprob: {b_lp})\n"
        if idx < len(test.tokens):
            t_tok = test.tokens[idx]
            t_lp = (
                f"{test.logprobs[idx]:.3f}" if test.logprobs and idx < len(test.logprobs) else "N/A"
            )
            test_token_line = f"  Actual token: `{t_tok}` (logprob: {t_lp})\n"

    title_line = (
        f"## [Bug] {test.backend_name} produces divergent output "
        f"at {quant} for {category} prompts\n"
    )

    env_section = f"""**Environment**
- Backend: {test.backend_name}
- Baseline backend: {baseline.backend_name}
- Model: {test.model_id}
- Quantization (test): {quant}
- Quantization (baseline): {baseline_quant}
- Platform: {platform_info}
"""

    desc_section = f"""**Description**
Output diverges from baseline at token position {div_idx}.
Text similarity: {similarity_pct}%, KL divergence: {kl_str}

**Baseline output** (truncated to 500 chars)
```
{baseline_text}
```

**Actual output** (truncated to 500 chars)
```
{test_text}
```
"""

    if comparison.token_divergence_index is not None:
        desc_section += (
            f"\n**First divergence at token {div_idx}**\n" + baseline_token_line + test_token_line
        )

    if comparison.failure_reason:
        desc_section += f"\nFailure reason: {comparison.failure_reason}\n"

    parts = [title_line, env_section, desc_section]

    if include_repro:
        cmd = _REPRO_CMD.format(
            model_id=test.model_id,
            baseline=baseline.backend_name,
            test=test.backend_name,
        )
        repro_section = f"""**Reproduction**
```bash
{cmd}
```
"""
        parts.append(repro_section)

    return "\n".join(parts)


def format_issues_batch(comparisons: list[ComparisonResult]) -> str:
    """Format multiple failures into a single issue body, grouped by category.

    Only failures (where ``is_failure`` is ``True``) are included. Comparisons
    are grouped by the ``category`` field from the test result's metadata,
    falling back to ``"general"`` if not present.

    Args:
        comparisons: A list of ``ComparisonResult`` objects. Non-failures are
            silently skipped.

    Returns:
        A Markdown string with failures organized under category headings.
        Returns an empty string if there are no failures.

    Examples:
        >>> from infer_check.types import InferenceResult, ComparisonResult
        >>> b = InferenceResult(
        ...     prompt_id="p1", backend_name="mlx-lm", model_id="M",
        ...     tokens=["Hi"], text="Hi", latency_ms=10.0,
        ... )
        >>> t = InferenceResult(
        ...     prompt_id="p1", backend_name="llama-cpp", model_id="M",
        ...     tokens=["Hello"], text="Hello", latency_ms=9.0,
        ...     metadata={"category": "math"},
        ... )
        >>> c = ComparisonResult(
        ...     baseline=b, test=t, text_similarity=0.2, is_failure=True,
        ... )
        >>> out = format_issues_batch([c])
        >>> "## Category: math" in out
        True
    """
    failures = [c for c in comparisons if c.is_failure]
    if not failures:
        return ""

    # Group by category.
    by_category: dict[str, list[ComparisonResult]] = defaultdict(list)
    for comp in failures:
        cat = comp.test.metadata.get("category", comp.baseline.metadata.get("category", "general"))
        by_category[str(cat)].append(comp)

    total_failures = len(failures)
    backends = sorted({c.test.backend_name for c in failures})
    quants = sorted({c.test.quantization or "unspecified" for c in failures})

    header = (
        f"# infer-check Batch Failure Report\n\n"
        f"**{total_failures} failure(s)** across backends: {', '.join(backends)}\n"
        f"Quantization levels affected: {', '.join(quants)}\n\n"
        f"---\n"
    )

    sections: list[str] = [header]

    for category in sorted(by_category.keys()):
        group = by_category[category]
        # Sort by worst similarity first.
        group.sort(key=lambda c: c.text_similarity)

        sections.append(f"## Category: {category}\n")
        sections.append(f"*{len(group)} failure(s) in this category.*\n")

        for i, comp in enumerate(group, start=1):
            sections.append(f"### Failure {i} of {len(group)}\n")
            # Embed the individual issue body (without the title line duplicate).
            body = format_issue(comp, include_repro=True)
            # Strip the leading title line (already provided by section header).
            body_lines = body.split("\n")
            if body_lines and body_lines[0].startswith("## [Bug]"):
                body = "\n".join(body_lines[1:]).lstrip("\n")
            sections.append(body)
            sections.append("\n---\n")

    return "\n".join(sections)
