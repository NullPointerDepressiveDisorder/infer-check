"""Command-line interface for infer-check."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import click
from rich.console import Console
from rich.table import Table

console = Console()

F = TypeVar("F", bound=Callable[..., Any])


def common_options(f: F) -> F:
    """Add common options to all subcommands."""
    options = [
        click.option(
            "--max-tokens",
            default=None,
            type=click.IntRange(min=1, clamp=True),
            help="Override default max tokens for generation.",
        ),
        click.option(
            "--num-prompts",
            default=None,
            type=click.IntRange(min=1, clamp=True),
            help="Limit number of prompts to use.",
        ),
        click.option(
            "--disable-thinking/--enable-thinking",
            "disable_thinking",
            default=True,
            show_default=True,
            help=(
                "Suppress reasoning/thinking mode on models that support it "
                "(Qwen3, DeepSeek-R1, Ollama think, vLLM chat_template_kwargs, "
                "OpenAI/OpenRouter reasoning). Models without a thinking mode "
                "are unaffected. Defaults to disabled so outputs are directly "
                "comparable across runs; pass --enable-thinking to restore it."
            ),
        ),
        click.option(
            "--chat/--no-chat",
            default=True,
            show_default=True,
            help=(
                "Use /v1/chat/completions for HTTP backends (applies chat "
                "template server-side). Pass --no-chat to use raw "
                "/v1/completions instead. Ignored for mlx-lm."
            ),
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f


def _resolve_prompts(prompts: str) -> Path:
    """Resolve a prompt suite name or path to an actual file path."""
    from infer_check.prompt_suites import get_suite_path

    try:
        return get_suite_path(prompts)
    except FileNotFoundError as exc:
        raise click.BadParameter(str(exc)) from exc


def _load_prompts(ctx: click.Context, prompts: str, max_tokens: int | None, num_prompts: int | None) -> list[Any]:
    """Load and prepare a prompt suite with overrides applied."""
    from infer_check.suites.loader import load_suite

    # Update defaults from subcommand if provided
    if max_tokens is not None:
        ctx.obj["max_tokens"] = max_tokens
    if num_prompts is not None:
        ctx.obj["num_prompts"] = num_prompts

    prompt_list = load_suite(_resolve_prompts(prompts), num_prompts=ctx.obj["num_prompts"])

    # Apply global max_tokens only if not explicitly set in the prompt JSONL
    for p in prompt_list:
        if "max_tokens" not in p.model_fields_set:
            p.max_tokens = ctx.obj["max_tokens"]

    return prompt_list


@click.group()
@click.version_option(package_name="infer-check")
@click.option(
    "--max-tokens",
    default=1024,
    show_default=True,
    help="Default max tokens for generation (applies to all prompts unless they specify their own).",
)
@click.option(
    "--num-prompts",
    default=None,
    type=click.IntRange(min=1, clamp=True),
    help="Limit the number of prompts to use from a suite; if omitted, all prompts are used.",
)
@click.option(
    "--keep-distributions",
    is_flag=True,
    default=False,
    help="Keep per-token top-K distributions in saved results (sweep/compare/diff). "
    "They're only needed to compute KL during the run and make result files ~20x larger.",
)
@click.pass_context
def main(ctx: click.Context, max_tokens: int, num_prompts: int | None, keep_distributions: bool) -> None:
    """infer-check: correctness and reliability testing for LLM inference engines."""
    ctx.ensure_object(dict)
    ctx.obj["max_tokens"] = max_tokens
    ctx.obj["num_prompts"] = num_prompts
    ctx.obj["keep_distributions"] = keep_distributions


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
@common_options
@click.pass_context
def sweep(
    ctx: click.Context,
    models: str,
    backend: str | None,
    prompts: str,
    output: Path,
    baseline: str | None,
    base_url: str | None,
    max_tokens: int | None,
    num_prompts: int | None,
    disable_thinking: bool,
    chat: bool,
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

    prompt_list = _load_prompts(ctx, prompts, max_tokens, num_prompts)

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

    # Build a separate backend for each model
    backend_map: dict[str, Any] = {}
    for label, model_path in model_map.items():
        backend_map[label] = get_backend_for_model(
            model_str=model_path,
            backend_type=backend,
            base_url=base_url,
            quantization=label,
            disable_thinking=disable_thinking,
            chat=chat,
        )

    runner = TestRunner(keep_distributions=ctx.obj.get("keep_distributions", False))
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
    table.add_column("mean KL", justify="right")
    table.add_column("top-1 agree", justify="right")

    for quant_level in quant_levels:
        comps = groups.get(quant_level, [])
        if not comps:
            table.add_row(quant_level, *["N/A"] * 7)
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
            *_kl_cells(comps),
        )

    console.print(table)
    console.print(
        "[dim]KL = mean per-token KL(baseline ‖ test) in nats; "
        "TF = teacher-forced (both models score the baseline's tokens), "
        "prefix = approximate, shared-context positions only.[/dim]"
    )
    _print_correctness_by_group(result.summary.get("correctness_by_quant", {}), "quant_level")


def _fmt_acc(side: dict[str, Any]) -> str:
    return "N/A" if side.get("accuracy") is None else f"{side['accuracy']:.1%}"


def _correctness_cells(table: dict[str, Any] | None) -> tuple[str, str]:
    """(accuracy A → B, "regressions / improvements (p=…)") for a paired correctness table."""
    if not table:
        return "—", "—"
    p = table["mcnemar_p"]
    style = "red" if table["regressions"] > table["improvements"] and p < 0.05 else ""
    flips = f"{table['regressions']} / {table['improvements']} (p={p:.3g})"
    return f"{_fmt_acc(table['a'])} → {_fmt_acc(table['b'])}", f"[{style}]{flips}[/]" if style else flips


def _print_correctness_by_group(tables: dict[str, dict[str, Any] | None], group_label: str) -> None:
    """One row per group (quant level / backend): accuracy vs the baseline and paired flips."""
    rows = {name: t for name, t in tables.items() if t}
    if not rows:
        return
    t = Table(title="Correctness vs answer keys (baseline → this)", header_style="bold magenta")
    t.add_column(group_label, style="cyan")
    t.add_column("graded", justify="right")
    t.add_column("accuracy", justify="right")
    t.add_column("regr / impr (p)", justify="right")
    for name, table in rows.items():
        acc, flips = _correctness_cells(table)
        t.add_row(name, str(table["n_graded"]), acc, flips)
    console.print(t)


def _print_correctness(table: dict[str, Any] | None, label_a: str, label_b: str) -> None:
    """Paired correctness table against answer keys (skipped when the suite has none)."""
    if not table:
        console.print("[dim]No answer keys in this suite: correctness not measured.[/dim]")
        return
    a, b = table["a"], table["b"]

    def acc(side: dict[str, Any]) -> str:
        if side["accuracy"] is None:
            return "N/A"
        lo, hi = side["accuracy_ci95"]
        return f"{side['accuracy']:.1%}  [{lo:.1%}, {hi:.1%}]  ({side['correct']}/{side['scored']})"

    t = Table(title=f"Correctness vs answer keys ({table['n_graded']} graded prompts)", header_style="bold magenta")
    t.add_column("", style="cyan")
    t.add_column(f"{label_b} right", justify="right")
    t.add_column(f"{label_b} wrong", justify="right")
    t.add_row(f"{label_a} right", str(table["both_right"]), f"[red]{table['regressions']}[/red]")
    t.add_row(f"{label_a} wrong", f"[green]{table['improvements']}[/green]", str(table["both_wrong"]))
    console.print(t)
    console.print(f"  accuracy {label_a}: {acc(a)}")
    console.print(f"  accuracy {label_b}: {acc(b)}")
    console.print(
        f"  regressions {table['regressions']}, improvements {table['improvements']}, "
        f"exact McNemar p = {table['mcnemar_p']:.3g}"
    )
    if a["unparsed"] or b["unparsed"] or a["truncated"] or b["truncated"] or table["both_truncated"]:
        console.print(
            f"  [dim]no parseable answer: {a['unparsed']} / {b['unparsed']}; "
            f"hit token limit while the other finished (counted wrong): {a['truncated']} / {b['truncated']}; "
            f"both hit the limit (excluded): {table['both_truncated']}[/dim]"
        )


def _kl_cells(comps: list[Any]) -> tuple[str, str]:
    """Format (mean KL with method tag, top-1 agreement) for a group of comparisons."""
    from infer_check.analysis.divergence import aggregate_kl

    agg = aggregate_kl(comps)
    if agg["mean_kl"] is None:
        return "N/A", "N/A"
    tag = "TF" if agg["method"] == "teacher_forced" else "prefix"
    return f"{agg['mean_kl']:.4f} ({tag})", f"{agg['mean_top1_agreement']:.1%}"


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
@common_options
@click.pass_context
def compare(
    ctx: click.Context,
    model_a: str,
    model_b: str,
    prompts: str,
    output: Path,
    base_url: str | None,
    label_a: str | None,
    label_b: str | None,
    report: bool,
    max_tokens: int | None,
    num_prompts: int | None,
    disable_thinking: bool,
    chat: bool,
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

    prompt_list = _load_prompts(ctx, prompts, max_tokens, num_prompts)

    resolved_a = resolve_model(model_a, base_url=base_url, label=label_a)
    resolved_b = resolve_model(model_b, base_url=base_url, label=label_b)

    console.print(
        f"[bold cyan]compare[/bold cyan] "
        f"A={resolved_a.label} ({resolved_a.backend}) "
        f"vs B={resolved_b.label} ({resolved_b.backend})"
    )

    console.print(f"  prompts: {len(prompt_list)} from '{prompts}'")

    # ── Build backends ───────────────────────────────────────────────
    from infer_check.backends.base import BackendConfig, get_backend

    config_a = BackendConfig(
        backend_type=resolved_a.backend,
        model_id=resolved_a.model_id,
        quantization=resolved_a.label,
        hf_revision=resolved_a.revision,
        base_url=resolved_a.base_url,
        disable_thinking=disable_thinking,
        extra={"chat": chat},
    )
    config_b = BackendConfig(
        backend_type=resolved_b.backend,
        model_id=resolved_b.model_id,
        quantization=resolved_b.label,
        hf_revision=resolved_b.revision,
        base_url=resolved_b.base_url,
        disable_thinking=disable_thinking,
        extra={"chat": chat},
    )
    backend_a = get_backend(config_a)
    backend_b = get_backend(config_b)

    # ── Run comparison ───────────────────────────────────────────────
    runner = TestRunner(keep_distributions=ctx.obj.get("keep_distributions", False))
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
        method = "teacher-forced" if compare_result.kl_method == "teacher_forced" else "prefix, approximate"
        table.add_row(f"mean KL divergence ({method})", f"{compare_result.mean_kl_divergence:.6f}")
    if compare_result.mean_top1_agreement is not None:
        table.add_row("top-1 agreement", f"{compare_result.mean_top1_agreement:.2%}")
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
    _print_correctness(compare_result.correctness, resolved_a.label, resolved_b.label)

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
@common_options
@click.pass_context
def diff(
    ctx: click.Context,
    model: str,
    backends: str,
    prompts: str,
    output: Path,
    quant: str | None,
    base_urls: str | None,
    max_tokens: int | None,
    num_prompts: int | None,
    disable_thinking: bool,
    chat: bool,
) -> None:
    """Compare outputs across different backends for the same model and prompts."""
    from infer_check.backends.base import BackendConfig, get_backend
    from infer_check.resolve import resolve_model
    from infer_check.runner import TestRunner

    prompt_list = _load_prompts(ctx, prompts, max_tokens, num_prompts)

    # Resolve the model to handle @revision and ensure correct base URL
    resolved = resolve_model(model)

    backend_names = [b.strip() for b in backends.split(",") if b.strip()]
    url_list: list[str | None] = [u.strip() for u in base_urls.split(",")] if base_urls else [None] * len(backend_names)
    # Pad url_list if shorter than backend_names
    while len(url_list) < len(backend_names):
        url_list.append(None)

    console.print(f"[bold cyan]diff[/bold cyan] model={resolved.model_id} backends={backend_names} quant={quant}")
    if resolved.revision:
        console.print(f"  revision: {resolved.revision}")

    backend_instances = []
    for name, url in zip(backend_names, url_list, strict=True):
        config = BackendConfig(
            backend_type=name,  # type: ignore[arg-type]
            model_id=resolved.model_id,
            quantization=quant,
            hf_revision=resolved.revision,
            base_url=url or (resolved.base_url if name == resolved.backend else None),
            disable_thinking=disable_thinking,
            extra={"chat": chat} if name in ("openai-compat", "vllm-mlx") else {},
        )
        backend_instances.append(get_backend(config))

    baseline_backend = backend_instances[0]
    test_backends = backend_instances[1:]

    runner = TestRunner(keep_distributions=ctx.obj.get("keep_distributions", False))
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
    out_path = output / f"diff_{resolved.model_id.replace('/', '_')}_{ts}.json"
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
    table.add_column("flip_rate", justify="right")
    table.add_column("mean_similarity", justify="right")

    from collections import defaultdict

    from infer_check.runner import correctness_summary
    from infer_check.runner import flip_rate as compute_flip_rate

    groups: dict[str, list[Any]] = defaultdict(list)
    for comp in comparisons:
        groups[comp.test.backend_name].append(comp)

    for backend_name, comps in groups.items():
        failures = sum(1 for c in comps if c.is_failure)
        failure_rate = failures / len(comps) if comps else 0.0
        flip_rate = compute_flip_rate(comps)
        mean_sim = sum(c.text_similarity for c in comps) / len(comps) if comps else 0.0
        table.add_row(
            backend_name,
            str(failures),
            f"{failure_rate:.2%}",
            f"[{'red' if flip_rate > 0.1 else 'green'}]{flip_rate:.1%}[/]",
            f"{mean_sim:.4f}",
        )

    console.print(table)
    _print_correctness_by_group({name: correctness_summary(comps) for name, comps in groups.items()}, "test_backend")


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
@click.option(
    "--trials",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Batches per prompt per concurrency level, each with freshly drawn co-scheduled prompts.",
)
@click.option("--seed", default=0, show_default=True, help="Seed for choosing co-scheduled prompts.")
@common_options
@click.pass_context
def stress(
    ctx: click.Context,
    model: str,
    backend: str | None,
    prompts: str,
    output: Path,
    concurrency: str,
    base_url: str | None,
    trials: int,
    seed: int,
    max_tokens: int | None,
    num_prompts: int | None,
    disable_thinking: bool,
    chat: bool,
) -> None:
    """Check that outputs don't change with what else the server is processing.

    Each prompt runs alone first (the reference). Then, at each concurrency
    level N, it is sent together with N-1 other prompts from the suite, and its
    output is compared token-for-token with the solo run. Requires an HTTP
    server backend; in-process mlx-lm can't run requests concurrently.
    """
    from infer_check.backends.base import get_backend_for_model
    from infer_check.runner import TestRunner

    prompt_list = _load_prompts(ctx, prompts, max_tokens, num_prompts)

    concurrency_levels = [int(c.strip()) for c in concurrency.split(",") if c.strip()]

    backend_instance = get_backend_for_model(
        model_str=model,
        backend_type=backend,
        base_url=base_url,
        disable_thinking=disable_thinking,
        chat=chat,
    )

    console.print(
        f"[bold cyan]stress[/bold cyan] model={model} backend={backend_instance.name} concurrency={concurrency_levels}"
    )

    if not backend_instance.supports_concurrency:
        raise click.ClickException(
            f"The {backend_instance.name} backend runs requests one at a time, so a stress test can't fail on it. "
            "Point --backend/--base-url at an HTTP server instead (llama-server, vllm-mlx, mlx_lm.server, vLLM)."
        )

    runner = TestRunner()
    stress_results = asyncio.run(
        runner.stress(
            backend=backend_instance,
            prompts=prompt_list,
            concurrency_levels=concurrency_levels,
            trials=trials,
            seed=seed,
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
    table.add_column("in flight", style="cyan", justify="right")
    table.add_column("compared", justify="right")
    table.add_column("identical to solo", justify="right")
    table.add_column("diverged", justify="right")
    table.add_column("errors", justify="right")

    for sr in stress_results:
        is_ref = sr.concurrency_level == 1
        style = "green" if sr.output_consistency == 1.0 else "red"
        table.add_row(
            f"{sr.concurrency_level}{' (reference)' if is_ref else ''}",
            str(sr.compared),
            "—" if is_ref else f"[{style}]{sr.output_consistency:.1%}[/{style}]",
            "—" if is_ref else str(len(sr.divergences)),
            str(sr.error_count),
        )
    console.print(table)

    for sr in stress_results:
        for msg in sr.error_messages[:3]:
            console.print(f"[yellow]  errors at {sr.concurrency_level} in flight: {msg}[/yellow]")

    diverged = [(sr.concurrency_level, d) for sr in stress_results for d in sr.divergences]
    if diverged:
        dt = Table(title="Diverged outputs (vs. the prompt's solo run)", header_style="bold magenta")
        dt.add_column("in flight", justify="right")
        dt.add_column("prompt")
        dt.add_column("first differing token", justify="right")
        dt.add_column("sent with")
        for level, d in diverged[:30]:
            dt.add_row(str(level), d["prompt_id"], str(d["first_divergence"]), ", ".join(d["co_scheduled"][:4]))
        console.print(dt)
        if len(diverged) > 30:
            console.print(f"[dim]... and {len(diverged) - 30} more (see the results file).[/dim]")


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
@common_options
@click.pass_context
def determinism(
    ctx: click.Context,
    model: str,
    backend: str | None,
    prompts: str,
    output: Path,
    runs: int,
    base_url: str | None,
    max_tokens: int | None,
    num_prompts: int | None,
    disable_thinking: bool,
    chat: bool,
) -> None:
    """Test whether a backend produces identical outputs across repeated runs at temperature=0."""
    from infer_check.backends.base import get_backend_for_model
    from infer_check.runner import TestRunner

    prompt_list = _load_prompts(ctx, prompts, max_tokens, num_prompts)

    backend_instance = get_backend_for_model(
        model_str=model,
        backend_type=backend,
        base_url=base_url,
        disable_thinking=disable_thinking,
        chat=chat,
    )

    console.print(f"[bold cyan]determinism[/bold cyan] model={model} backend={backend_instance.name} runs={runs}")

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
