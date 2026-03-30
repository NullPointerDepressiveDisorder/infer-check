"""Command-line interface for infer-check."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _resolve_prompts(prompts: str) -> Path:
    """Resolve a prompt suite name or path to an actual file path."""
    from infer_check.prompt_suites import get_suite_path

    try:
        return get_suite_path(prompts)
    except FileNotFoundError as exc:
        raise click.BadParameter(str(exc)) from exc


@click.group()
@click.version_option(package_name="infer-check")
def main() -> None:
    """infer-check: correctness and reliability testing for LLM inference engines."""


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--models",
    required=True,
    help=(
        "Comma-separated label=model_path pairs. "
        "Example: 'bf16=mlx-community/Llama-3.1-8B-Instruct-bf16,"
        "4bit=mlx-community/Llama-3.1-8B-Instruct-4bit'"
    ),
)
@click.option("--backend", default=None, help="Backend type (auto-detected if omitted).")
@click.option(
    "--prompts",
    required=True,
    help="Bundled suite name (e.g. 'reasoning') or path to a .jsonl file.",
)
@click.option(
    "--output",
    default="./results/sweep/",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option(
    "--baseline",
    default=None,
    help="Baseline label (defaults to first in --models).",
)
@click.option("--base-url", default=None, help="Base URL for HTTP backends.")
def sweep(
    models: str,
    backend: str | None,
    prompts: str,
    output: Path,
    baseline: str | None,
    base_url: str | None,
) -> None:
    """Run a quantization sweep: compare pre-quantized models against a baseline.

    Each model is a separate HuggingFace repo or local path. The first model
    (or --baseline) is the reference; all others are compared against it.

    Example:

        infer-check sweep \\
          --models "bf16=mlx-community/Llama-3.1-8B-Instruct-bf16,
                    4bit=mlx-community/Llama-3.1-8B-Instruct-4bit,
                    3bit=mlx-community/Llama-3.1-8B-Instruct-3bit" \\
          --prompts reasoning
    """
    from infer_check.backends.base import get_backend_for_model
    from infer_check.runner import TestRunner
    from infer_check.suites.loader import load_suite

    # Parse label=model_path pairs
    model_map: dict[str, str] = {}
    for entry in models.split(","):
        entry = entry.strip()
        if "=" in entry:
            label, path = entry.split("=", 1)
            model_map[label.strip()] = path.strip().lstrip("=").strip()
        else:
            # No label provided — use the last path component as label
            label = entry.strip().rsplit("/", 1)[-1]
            model_map[label] = entry.strip()

    if len(model_map) < 2:
        console.print("[red]Need at least 2 models for a sweep (baseline + 1 test).[/red]")
        raise SystemExit(1)

    quant_levels = list(model_map.keys())
    baseline_label = baseline or quant_levels[0]

    if baseline_label not in model_map:
        console.print(f"[red]Baseline '{baseline_label}' not found in model map.[/red]")
        raise SystemExit(1)

    console.print(f"[bold cyan]sweep[/bold cyan] baseline={baseline_label} models={quant_levels}")
    for label, path in model_map.items():
        tag = " (baseline)" if label == baseline_label else ""
        console.print(f"  {label}: {path}{tag}")

    prompt_list = load_suite(_resolve_prompts(prompts))

    # Build a separate backend for each model
    backend_map: dict[str, Any] = {}
    for label, model_path in model_map.items():
        backend_map[label] = get_backend_for_model(
            model_str=model_path,
            backend_type=backend,
            base_url=base_url,
            quantization=label,
        )

    runner = TestRunner()
    result = asyncio.run(
        runner.sweep(
            backend_map=backend_map,
            prompts=prompt_list,
            quantization_levels=quant_levels,
            model_id=model_map[baseline_label],
            baseline_quant=baseline_label,
        )
    )

    # Persist results
    output.mkdir(parents=True, exist_ok=True)
    # Use the resolved backend name in the filename when --backend is omitted
    if backend is not None:
        backend_name = backend
    else:
        baseline_backend = backend_map.get(baseline_label)
        if baseline_backend is None:
            backend_name = "unknown"
        else:
            # Prefer an explicit backend adapter name, fall back to the class name
            backend_name = getattr(baseline_backend, "name", type(baseline_backend).__name__)
    ts = int(result.timestamp.timestamp())
    out_path = output / f"sweep_{model_map[baseline_label].replace('/', '_')}_{backend_name}_{ts}.json"
    result.save(out_path)
    console.print(f"[green]Results saved to {out_path}[/green]")

    # ------------------------------------------------------------------
    # Build summary table per quant level
    # ------------------------------------------------------------------
    # Group comparisons by test quantization
    from collections import defaultdict

    groups: dict[str, list[Any]] = defaultdict(list)
    for comp in result.comparisons:
        quant_key = comp.test.quantization or "unknown"
        groups[quant_key].append(comp)

    table = Table(title="Sweep Summary", show_header=True, header_style="bold magenta")
    table.add_column("quant_level", style="cyan")
    table.add_column("identical", justify="right")
    table.add_column("minor", justify="right")
    table.add_column("moderate", justify="right")
    table.add_column("severe", justify="right", style="red")
    table.add_column("mean_similarity", justify="right")

    for quant_level in quant_levels:
        comps = groups.get(quant_level, [])
        if not comps:
            table.add_row(quant_level, "N/A", "N/A", "N/A", "N/A", "N/A")
            continue

        severities = {"identical": 0, "minor": 0, "moderate": 0, "severe": 0}
        for c in comps:
            sev = c.metadata.get("severity", "unknown") if hasattr(c, "metadata") else "unknown"
            if sev in severities:
                severities[sev] += 1

        n = len(comps)
        mean_sim = sum(c.text_similarity for c in comps) / n

        label = f"{quant_level} (self-check)" if quant_level == baseline_label else quant_level

        # Flag baseline self-check problems
        if quant_level == baseline_label and severities["identical"] < n:
            style = "bold red"
            label += " ⚠"
        elif quant_level == baseline_label:
            style = "bold green"
        else:
            style = ""

        table.add_row(
            f"[{style}]{label}[/{style}]" if style else label,
            f"{severities['identical']}/{n}",
            f"{severities['minor']}/{n}",
            f"{severities['moderate']}/{n}",
            f"{severities['severe']}/{n}",
            f"{mean_sim:.4f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_a")
@click.argument("model_b")
@click.option(
    "--prompts",
    default="adversarial-numerics",
    show_default=True,
    help="Bundled suite name (e.g. 'reasoning') or path to a .jsonl file.",
)
@click.option(
    "--output",
    default="./results/compare/",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option(
    "--base-url",
    default=None,
    help=("Base URL override for HTTP backends. Applied to both models unless they resolve to mlx-lm."),
)
@click.option(
    "--label-a",
    default=None,
    help="Custom label for model A (defaults to auto-derived short name).",
)
@click.option(
    "--label-b",
    default=None,
    help="Custom label for model B (defaults to auto-derived short name).",
)
@click.option(
    "--report/--no-report",
    default=True,
    show_default=True,
    help="Generate an HTML comparison report after the run.",
)
def compare(
    model_a: str,
    model_b: str,
    prompts: str,
    output: Path,
    base_url: str | None,
    label_a: str | None,
    label_b: str | None,
    report: bool,
) -> None:
    """Compare two quantizations of the same model.

    MODEL_A and MODEL_B are model specs — HuggingFace repos, Ollama tags,
    or local GGUF paths.  The backend is auto-detected from the identifier,
    or you can use an explicit prefix (ollama:, mlx:, gguf:, vllm-mlx:).

    \b
    Examples:
        # Two MLX quants
        infer-check compare \\
          mlx-community/Llama-3.1-8B-Instruct-4bit \\
          mlx-community/Llama-3.1-8B-Instruct-8bit

        # MLX native vs Ollama GGUF
        infer-check compare \\
          mlx-community/Llama-3.1-8B-Instruct-4bit \\
          ollama:llama3.1:8b-instruct-q4_K_M

        # Bartowski GGUF vs Unsloth GGUF (both via Ollama)
        infer-check compare \\
          ollama:bartowski/Llama-3.1-8B-Instruct-GGUF \\
          ollama:unsloth/Llama-3.1-8B-Instruct-GGUF
    """
    # ── Resolve both model specs ─────────────────────────────────────
    from infer_check.resolve import resolve_model
    from infer_check.runner import TestRunner
    from infer_check.suites.loader import load_suite

    resolved_a = resolve_model(model_a, base_url=base_url, label=label_a)
    resolved_b = resolve_model(model_b, base_url=base_url, label=label_b)

    console.print(
        f"[bold cyan]compare[/bold cyan] "
        f"A={resolved_a.label} ({resolved_a.backend}) "
        f"vs B={resolved_b.label} ({resolved_b.backend})"
    )

    prompt_list = load_suite(_resolve_prompts(prompts))
    console.print(f"  prompts: {len(prompt_list)} from '{prompts}'")

    # ── Build backends ───────────────────────────────────────────────
    from infer_check.backends.base import BackendConfig, get_backend

    config_a = BackendConfig(
        backend_type=resolved_a.backend,
        model_id=resolved_a.model_id,
        quantization=resolved_a.label,
        base_url=resolved_a.base_url,
        extra={"chat": False},
    )
    config_b = BackendConfig(
        backend_type=resolved_b.backend,
        model_id=resolved_b.model_id,
        quantization=resolved_b.label,
        base_url=resolved_b.base_url,
        extra={"chat": False},
    )
    backend_a = get_backend(config_a)
    backend_b = get_backend(config_b)

    # ── Run comparison ───────────────────────────────────────────────
    runner = TestRunner()
    compare_result = asyncio.run(
        runner.compare(
            backend_a=backend_a,
            backend_b=backend_b,
            prompts=prompt_list,
            label_a=resolved_a.label,
            label_b=resolved_b.label,
        )
    )

    # ── Persist results ──────────────────────────────────────────────
    output.mkdir(parents=True, exist_ok=True)

    from infer_check.utils import sanitize_filename

    safe_a = sanitize_filename(resolved_a.label)
    safe_b = sanitize_filename(resolved_b.label)
    ts = int(compare_result.timestamp.timestamp())
    out_path = output / f"compare_{safe_a}_vs_{safe_b}_{ts}.json"
    compare_result.save(out_path)
    console.print(f"[green]Results saved to {out_path}[/green]")

    # ── Summary table ────────────────────────────────────────────────
    table = Table(
        title=f"Compare: {resolved_a.label} vs {resolved_b.label}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")

    n = len(compare_result.comparisons)
    severities = {"identical": 0, "minor": 0, "moderate": 0, "severe": 0}
    for c in compare_result.comparisons:
        sev = c.metadata.get("severity", "unknown") if hasattr(c, "metadata") else "unknown"
        if sev in severities:
            severities[sev] += 1

    table.add_row("prompts", str(n))
    table.add_row(
        "flip rate",
        f"[{'red' if compare_result.flip_rate > 0.1 else 'green'}]{compare_result.flip_rate:.1%}[/]",
    )
    if compare_result.mean_kl_divergence is not None:
        table.add_row("mean KL divergence", f"{compare_result.mean_kl_divergence:.6f}")
    if compare_result.mean_text_similarity is not None:
        table.add_row("mean text similarity", f"{compare_result.mean_text_similarity:.4f}")
    else:
        table.add_row("mean text similarity", "N/A")
    table.add_row(
        "identical / minor / moderate / severe",
        f"{severities['identical']} / {severities['minor']} / "
        f"{severities['moderate']} / [red]{severities['severe']}[/red]",
    )

    console.print(table)

    # ── Per-category breakdown ───────────────────────────────────────
    if compare_result.per_category_stats:
        cat_table = Table(
            title="Per-Category Breakdown",
            show_header=True,
            header_style="bold magenta",
        )
        cat_table.add_column("category", style="cyan")
        cat_table.add_column("prompts", justify="right")
        cat_table.add_column("flip rate", justify="right")
        cat_table.add_column("mean similarity", justify="right")

        for cat, stats in sorted(compare_result.per_category_stats.items()):
            cat_table.add_row(
                cat,
                str(stats.get("count", 0)),
                f"{stats.get('flip_rate', 0.0):.1%}",
                f"{stats.get('mean_similarity', 0.0):.4f}",
            )

        console.print(cat_table)

    # ── Flipped prompts detail ───────────────────────────────────────
    flipped = [c for c in compare_result.comparisons if c.metadata.get("flipped", False)]
    if flipped:
        flip_table = Table(
            title=f"Flipped Prompts ({len(flipped)})",
            show_header=True,
            header_style="bold magenta",
        )
        flip_table.add_column("prompt", style="dim", max_width=50, no_wrap=True)
        flip_table.add_column("category", style="cyan")
        flip_table.add_column("strategy", style="dim")
        flip_table.add_column(f"{resolved_a.label}", max_width=30, no_wrap=True)
        flip_table.add_column(f"{resolved_b.label}", max_width=30, no_wrap=True)
        flip_table.add_column("similarity", justify="right")

        for c in flipped:
            prompt_text = (
                c.metadata.get("prompt_text")
                or getattr(c.baseline, "prompt_id", None)
                or getattr(c.baseline, "text", None)
                or "???"
            )
            # Truncate long prompt text for display.
            if len(prompt_text) > 47:
                prompt_text = prompt_text[:47] + "..."

            ans_a = c.metadata.get("answer_a", "?")
            ans_b = c.metadata.get("answer_b", "?")
            # Truncate long answers.
            if len(str(ans_a)) > 27:
                ans_a = str(ans_a)[:27] + "..."
            if len(str(ans_b)) > 27:
                ans_b = str(ans_b)[:27] + "..."

            cat = (
                c.metadata.get("prompt_category")
                or c.metadata.get("category")
                or c.baseline.metadata.get("prompt_category")
                or c.baseline.metadata.get("category")
                or c.test.metadata.get("prompt_category")
                or c.test.metadata.get("category")
                or "?"
            )
            flip_table.add_row(
                prompt_text,
                cat,
                c.metadata.get("extraction_strategy", "?"),
                f"[green]{ans_a}[/green]",
                f"[red]{ans_b}[/red]",
                f"{c.text_similarity:.3f}",
            )

        console.print(flip_table)

    # ── Report generation ───────────────────────────────────────────
    if report:
        from infer_check.reporting.html import generate_report

        ts = int(compare_result.timestamp.timestamp())
        report_path = output / f"report_{safe_a}_vs_{safe_b}_{ts}.html"
        generate_report(output, report_path)
        console.print(f"[green]HTML report generated at {report_path}[/green]")
    elif n > 0 and not flipped:
        console.print("[bold green]No answer flips detected.[/bold green]")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="Model ID or HuggingFace path.")
@click.option(
    "--backends",
    required=True,
    help="Comma-separated backend names, e.g. 'mlx-lm,llama-cpp'. First is baseline.",
)
@click.option(
    "--prompts",
    required=True,
    help="Bundled suite name (e.g. 'reasoning') or path to a .jsonl file.",
)
@click.option(
    "--output",
    default="./results/diff/",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option("--quant", default=None, help="Quantization level applied to all backends.")
@click.option(
    "--base-urls",
    default=None,
    help="Comma-separated base URLs for HTTP backends (positionally matched to --backends).",
)
@click.option(
    "--chat/--no-chat",
    default=True,
    show_default=True,
    help="Use /v1/chat/completions for HTTP backends (applies chat template server-side).",
)
def diff(
    model: str,
    backends: str,
    prompts: str,
    output: Path,
    quant: str | None,
    base_urls: str | None,
    chat: bool,
) -> None:
    """Compare outputs across different backends for the same model and prompts."""
    from infer_check.backends.base import BackendConfig, get_backend
    from infer_check.runner import TestRunner
    from infer_check.suites.loader import load_suite

    backend_names = [b.strip() for b in backends.split(",") if b.strip()]
    url_list: list[str | None] = [u.strip() for u in base_urls.split(",")] if base_urls else [None] * len(backend_names)
    # Pad url_list if shorter than backend_names
    while len(url_list) < len(backend_names):
        url_list.append(None)

    console.print(f"[bold cyan]diff[/bold cyan] model={model} backends={backend_names} quant={quant}")

    prompt_list = load_suite(_resolve_prompts(prompts))

    backend_instances = []
    for name, url in zip(backend_names, url_list, strict=True):
        config = BackendConfig(
            backend_type=name,  # type: ignore[arg-type]
            model_id=model,
            quantization=quant,
            base_url=url,
            extra={"chat": chat} if name in ("openai-compat", "vllm-mlx") else {},
        )
        backend_instances.append(get_backend(config))

    baseline_backend = backend_instances[0]
    test_backends = backend_instances[1:]

    runner = TestRunner()
    comparisons = asyncio.run(
        runner.diff(
            baseline_backend=baseline_backend,
            test_backends=test_backends,
            prompts=prompt_list,
        )
    )

    # Persist results
    output.mkdir(parents=True, exist_ok=True)
    ts = int(datetime.now(UTC).timestamp())
    out_path = output / f"diff_{model.replace('/', '_')}_{ts}.json"
    out_path.write_text(
        json.dumps(
            [c.model_dump(mode="json") for c in comparisons],
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Results saved to {out_path}[/green]")

    # Summary table
    table = Table(title="Diff Summary", show_header=True, header_style="bold magenta")
    table.add_column("test_backend", style="cyan")
    table.add_column("failures", justify="right")
    table.add_column("failure_rate", justify="right")
    table.add_column("mean_similarity", justify="right")

    from collections import defaultdict

    groups: dict[str, list[Any]] = defaultdict(list)
    for comp in comparisons:
        groups[comp.test.backend_name].append(comp)

    for backend_name, comps in groups.items():
        failures = sum(1 for c in comps if c.is_failure)
        failure_rate = failures / len(comps) if comps else 0.0
        mean_sim = sum(c.text_similarity for c in comps) / len(comps) if comps else 0.0
        table.add_row(backend_name, str(failures), f"{failure_rate:.2%}", f"{mean_sim:.4f}")

    console.print(table)


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="Model ID or HuggingFace path.")
@click.option("--backend", default=None, help="Backend type (auto-detected if omitted).")
@click.option(
    "--prompts",
    required=True,
    help="Bundled suite name (e.g. 'reasoning') or path to a .jsonl file.",
)
@click.option(
    "--output",
    default="./results/stress/",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option(
    "--concurrency",
    default="1,2,4,8,16",
    show_default=True,
    help="Comma-separated concurrency levels.",
)
@click.option("--base-url", default=None, help="Base URL for HTTP backends.")
def stress(
    model: str,
    backend: str | None,
    prompts: str,
    output: Path,
    concurrency: str,
    base_url: str | None,
) -> None:
    """Stress-test a backend with varying concurrency levels."""
    from infer_check.backends.base import get_backend_for_model
    from infer_check.runner import TestRunner
    from infer_check.suites.loader import load_suite

    concurrency_levels = [int(c.strip()) for c in concurrency.split(",") if c.strip()]

    backend_instance = get_backend_for_model(
        model_str=model,
        backend_type=backend,
        base_url=base_url,
    )

    console.print(
        f"[bold cyan]stress[/bold cyan] model={model} backend={backend_instance.name} concurrency={concurrency_levels}"
    )

    prompt_list = load_suite(_resolve_prompts(prompts))

    runner = TestRunner()
    stress_results = asyncio.run(
        runner.stress(
            backend=backend_instance,
            prompts=prompt_list,
            concurrency_levels=concurrency_levels,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    ts = int(datetime.now(UTC).timestamp())
    out_path = output / f"stress_{model.replace('/', '_')}_{backend_instance.name}_{ts}.json"
    out_path.write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in stress_results],
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Results saved to {out_path}[/green]")

    table = Table(title="Stress Test Summary", show_header=True, header_style="bold magenta")
    table.add_column("concurrency", style="cyan", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("output_consistency", justify="right")

    for sr in stress_results:
        table.add_row(
            str(sr.concurrency_level),
            str(sr.error_count),
            f"{sr.output_consistency:.2%}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="Model ID or HuggingFace path.")
@click.option("--backend", default=None, help="Backend type (auto-detected if omitted).")
@click.option(
    "--prompts",
    required=True,
    help="Bundled suite name (e.g. 'reasoning') or path to a .jsonl file.",
)
@click.option(
    "--output",
    default="./results/determinism/",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option("--runs", default=100, show_default=True, type=int, help="Number of runs per prompt.")
@click.option("--base-url", default=None, help="Base URL for HTTP backends.")
def determinism(
    model: str,
    backend: str | None,
    prompts: str,
    output: Path,
    runs: int,
    base_url: str | None,
) -> None:
    """Test whether a backend produces identical outputs across repeated runs at temperature=0."""
    from infer_check.backends.base import get_backend_for_model
    from infer_check.runner import TestRunner
    from infer_check.suites.loader import load_suite

    backend_instance = get_backend_for_model(
        model_str=model,
        backend_type=backend,
        base_url=base_url,
    )

    console.print(f"[bold cyan]determinism[/bold cyan] model={model} backend={backend_instance.name} runs={runs}")

    prompt_list = load_suite(_resolve_prompts(prompts))

    runner = TestRunner()
    det_results = asyncio.run(
        runner.determinism(
            backend=backend_instance,
            prompts=prompt_list,
            num_runs=runs,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    ts = int(datetime.now(UTC).timestamp())
    out_path = output / f"determinism_{model.replace('/', '_')}_{backend_instance.name}_{ts}.json"
    out_path.write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in det_results],
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Results saved to {out_path}[/green]")

    table = Table(title="Determinism Summary", show_header=True, header_style="bold magenta")
    table.add_column("prompt_id", style="cyan")
    table.add_column("runs", justify="right")
    table.add_column("identical", justify="right")
    table.add_column("determinism_score", justify="right")

    for dr in det_results:
        style = "green" if dr.determinism_score == 1.0 else "red"
        table.add_row(
            dr.prompt_id,
            str(dr.num_runs),
            str(dr.identical_count),
            f"[{style}]{dr.determinism_score:.2%}[/{style}]",
        )

    console.print(table)

    # Overall aggregate
    if det_results:
        overall = sum(r.determinism_score for r in det_results) / len(det_results)
        console.print(
            f"\n[bold]Overall determinism score:[/bold] [{'green' if overall == 1.0 else 'yellow'}]{overall:.2%}[/]"
        )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@main.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--format",
    "fmt",
    default="html",
    show_default=True,
    type=click.Choice(["html", "json"]),
    help="Output format.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output file path (defaults to <results_dir>/report.html or report.json).",
)
def report(
    results_dir: Path,
    fmt: Literal["html", "json"],
    output: Path | None,
) -> None:
    """Generate a report from previously saved result JSON files."""
    json_files = sorted(results_dir.rglob("*.json"))
    if not json_files:
        console.print(f"[red]No JSON result files found in {results_dir}[/red]")
        raise SystemExit(1)

    console.print(f"[bold cyan]report[/bold cyan] scanning {len(json_files)} result file(s)…")

    # Load all results as raw dicts (they may be heterogeneous types)
    all_data: list[dict[str, Any]] = []
    for path in json_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                all_data.extend(raw)
            elif isinstance(raw, dict):
                all_data.append(raw)
        except (json.JSONDecodeError, OSError) as exc:
            console.print(f"[yellow]  Skipping {path.name}: {exc}[/yellow]")

    if fmt == "html":
        out_path = output or (results_dir / "report.html")
        try:
            from infer_check.reporting.html import generate_report

            generate_report(results_dir, out_path)
        except ImportError:
            # reporting package not yet implemented — emit a minimal HTML placeholder
            _write_minimal_html(all_data, out_path)
        console.print(f"[green]HTML report written to {out_path}[/green]")
        click.launch(str(out_path))
    else:
        out_path = output or (results_dir / "report.json")
        try:
            from infer_check.reporting.json_export import export

            export(results_dir, out_path)
        except ImportError:
            out_path.write_text(json.dumps(all_data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]JSON report written to {out_path}[/green]")


def _write_minimal_html(data: list[dict[str, Any]], path: Path) -> None:
    """Write a minimal HTML file when the reporting package is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, indent=2, default=str)
    html = (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        "<title>infer-check report</title></head>"
        "<body><h1>infer-check report</h1>"
        f"<pre>{body}</pre></body></html>"
    )
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
