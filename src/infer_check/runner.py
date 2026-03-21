import asyncio
import difflib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.progress import Progress

from infer_check.backends.base import BackendAdapter
from infer_check.types import (
    CompareResult,
    ComparisonResult,
    DeterminismResult,
    InferenceResult,
    Prompt,
    StressResult,
    SweepResult,
)


class TestRunner:
    __test__ = False

    def __init__(self, cache_dir: str | Path = ".infer_check_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, results: Any, path: Path) -> None:
        """Write intermediate results as JSON for resumability."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(results, "model_dump_json"):
            path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
        elif isinstance(results, list):
            dumped = [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
            path.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _compare(
        self, baseline: InferenceResult, test: InferenceResult, threshold: float = 0.5
    ) -> ComparisonResult:
        """Compare a test result against a baseline.

        Threshold is the *minimum* text similarity to pass. Default 0.5
        separates "different wording, same answer" from "actually broken."

        Severity tiers (stored in metadata):
          - identical:  similarity == 1.0
          - minor:      similarity >= 0.8
          - moderate:   similarity >= 0.5
          - severe:     similarity < 0.5
        """
        if baseline.text == test.text:
            text_similarity = 1.0
        else:
            matcher = difflib.SequenceMatcher(None, baseline.text, test.text)
            text_similarity = matcher.ratio()

        is_failure = text_similarity < (1.0 - threshold)

        # Severity tiers
        if text_similarity == 1.0:
            severity = "identical"
        elif text_similarity >= 0.8:
            severity = "minor"
        elif text_similarity >= 0.5:
            severity = "moderate"
        else:
            severity = "severe"

        token_divergence_index = None
        for i, (b_tok, t_tok) in enumerate(zip(baseline.tokens, test.tokens, strict=False)):
            if b_tok != t_tok:
                token_divergence_index = i
                break

        if token_divergence_index is None and len(baseline.tokens) != len(test.tokens):
            token_divergence_index = min(len(baseline.tokens), len(test.tokens))

        return ComparisonResult(
            baseline=baseline,
            test=test,
            text_similarity=text_similarity,
            is_failure=is_failure,
            token_divergence_index=token_divergence_index,
            kl_divergence=None,
            failure_reason=f"[{severity}] Text similarity {text_similarity:.3f} below threshold"
            if is_failure
            else None,
            metadata={"severity": severity},
        )

    async def sweep(
        self,
        backend_map: dict[str, BackendAdapter],
        prompts: list[Prompt],
        quantization_levels: list[str],
        model_id: str,
        baseline_quant: str | None = None,
    ) -> SweepResult:
        """Run prompts through a separate backend per quantization level.

        Args:
            backend_map: Mapping of quant label → BackendAdapter instance.
                         Each backend points at a different pre-quantized model.
            prompts: The prompt suite to evaluate.
            quantization_levels: Ordered list of quant labels (keys into backend_map).
            model_id: Identifier for the model family (used in results metadata).
            baseline_quant: Which label is the baseline. Defaults to first in list.
        """
        baseline_results: dict[str, InferenceResult] = {}
        all_comparisons: list[ComparisonResult] = []
        failure_counts: dict[str, int] = {q: 0 for q in quantization_levels}

        if baseline_quant is None:
            if not quantization_levels:
                raise ValueError("Must provide at least one quantization level")
            baseline_quant = quantization_levels[0]

        # +1 for the baseline self-check (run baseline twice)
        total_steps = (len(quantization_levels) + 1) * len(prompts)

        with Progress() as progress:
            task = progress.add_task("[cyan]Running sweep...", total=total_steps)

            # ---- Baseline pass 1: collect reference outputs ----
            baseline_backend = backend_map[baseline_quant]
            progress.update(task, description=f"[cyan]Baseline pass 1 ({baseline_quant})...")
            for prompt in prompts:
                try:
                    res = await baseline_backend.generate(prompt)
                except Exception as exc:
                    from rich.console import Console

                    Console().print(
                        f"[yellow]  ⚠ Skipping prompt '{prompt.text[:60]}...' "
                        f"at {baseline_quant}: {exc}[/yellow]"
                    )
                    progress.advance(task)
                    continue
                if not res.quantization:
                    res.quantization = baseline_quant
                baseline_results[res.prompt_id] = res
                progress.advance(task)

            # ---- Baseline pass 2: self-check for determinism ----
            progress.update(task, description=f"[cyan]Baseline self-check ({baseline_quant})...")
            for prompt in prompts:
                if prompt.id not in baseline_results:
                    progress.advance(task)
                    continue
                try:
                    res2 = await baseline_backend.generate(prompt)
                except Exception as exc:
                    from rich.console import Console

                    Console().print(
                        f"[yellow]  ⚠ Self-check failed for '{prompt.text[:60]}...': {exc}[/yellow]"
                    )
                    progress.advance(task)
                    continue
                if not res2.quantization:
                    res2.quantization = baseline_quant
                comp = self._compare(baseline_results[prompt.id], res2)
                comp.metadata["is_self_check"] = True
                all_comparisons.append(comp)
                if comp.is_failure:
                    failure_counts[baseline_quant] = failure_counts.get(baseline_quant, 0) + 1
                progress.advance(task)

            await baseline_backend.cleanup()

            # ---- Quantized model passes ----
            for quant in quantization_levels:
                if quant == baseline_quant:
                    continue  # Already handled above

                backend = backend_map[quant]
                progress.update(task, description=f"[cyan]Running {quant}...")

                for prompt in prompts:
                    try:
                        res = await backend.generate(prompt)
                    except Exception as exc:
                        from rich.console import Console

                        Console().print(
                            f"[yellow]  ⚠ Skipping prompt '{prompt.text[:60]}...' "
                            f"at {quant}: {exc}[/yellow]"
                        )
                        progress.advance(task)
                        continue
                    if not res.quantization:
                        res.quantization = quant

                    baseline = baseline_results.get(res.prompt_id)
                    if baseline:
                        comp = self._compare(baseline, res)
                        all_comparisons.append(comp)
                        if comp.is_failure:
                            failure_counts[quant] = failure_counts.get(quant, 0) + 1
                    progress.advance(task)

                # Save checkpoint after each quant completes
                timestamp = int(datetime.now(UTC).timestamp())
                checkpoint_path = self.cache_dir / f"sweep_{model_id}_{quant}_{timestamp}.json"
                self._save_checkpoint(
                    [c for c in all_comparisons if c.test.quantization == quant],
                    checkpoint_path,
                )

                await backend.cleanup()

        summary = {
            "total_prompts": len(prompts),
            "quantization_levels": quantization_levels,
            "baseline_quant": baseline_quant,
            "failure_counts": failure_counts,
        }

        return SweepResult(
            model_id=model_id,
            backend_name=next(iter(backend_map.values())).name,
            quantization_levels=quantization_levels,
            comparisons=all_comparisons,
            timestamp=datetime.now(UTC),
            summary=summary,
        )

    async def compare(
        self,
        backend_a: BackendAdapter,
        backend_b: BackendAdapter,
        prompts: list[Prompt],
        label_a: str = "model_a",
        label_b: str = "model_b",
    ) -> CompareResult:
        """Compare two quantizations (or model variants) head-to-head.

        Unlike ``sweep`` (one baseline, N test quants on the same backend)
        and ``diff`` (same model across different backends), ``compare``
        is designed for arbitrary A-vs-B comparisons — two different quant
        providers, two different bit widths, or even cross-backend pairs.

        Metrics produced:
          - Per-prompt text similarity and severity tier (via ``_compare``).
          - **Flip rate**: fraction of prompts where the functional answer
            changed, determined by category-aware answer extraction
            (numeric, boolean, code, JSON, or raw similarity).
          - **KL divergence** (when both backends expose logprobs).
          - **Per-category breakdown** (keyed on ``Prompt.category``).
        """
        from collections import defaultdict

        results_a: dict[str, InferenceResult] = {}
        results_b: dict[str, InferenceResult] = {}
        comparisons: list[ComparisonResult] = []

        with Progress() as progress:
            total = len(prompts) * 2
            task = progress.add_task("[cyan]Comparing...", total=total)

            # ── Pass 1: generate from model A ────────────────────────
            progress.update(task, description=f"[cyan]Generating from {label_a}...")
            for prompt in prompts:
                try:
                    res = await backend_a.generate(prompt)
                except Exception as exc:
                    from rich.console import Console

                    Console().print(
                        f"[yellow]  ⚠ {label_a} failed for '{prompt.text[:60]}...': {exc}[/yellow]"
                    )
                    progress.advance(task)
                    continue
                if not res.quantization:
                    res.quantization = label_a
                results_a[res.prompt_id] = res
                progress.advance(task)

            # ── Pass 2: generate from model B ────────────────────────
            progress.update(task, description=f"[cyan]Generating from {label_b}...")
            for prompt in prompts:
                try:
                    res = await backend_b.generate(prompt)
                except Exception as exc:
                    from rich.console import Console

                    Console().print(
                        f"[yellow]  ⚠ {label_b} failed for '{prompt.text[:60]}...': {exc}[/yellow]"
                    )
                    progress.advance(task)
                    continue
                if not res.quantization:
                    res.quantization = label_b
                results_b[res.prompt_id] = res
                progress.advance(task)

        # ── Build comparisons with answer extraction ────────────────
        from infer_check.analysis.answer_extract import (
            answers_match,
            extract_answer,
        )

        for prompt in prompts:
            a = results_a.get(prompt.id)
            b = results_b.get(prompt.id)
            if a and b:
                comp = self._compare(a, b)
                comp.metadata["category"] = prompt.category

                # Extract functional answers and check for flips.
                ans_a = extract_answer(a.text, prompt.category)
                ans_b = extract_answer(b.text, prompt.category)
                flipped = not answers_match(ans_a, ans_b)

                comp.metadata["flipped"] = flipped
                comp.metadata["answer_a"] = ans_a.value
                comp.metadata["answer_b"] = ans_b.value
                comp.metadata["extraction_strategy"] = ans_a.strategy
                comp.metadata["extraction_confidence"] = min(
                    ans_a.confidence,
                    ans_b.confidence,
                )
                comparisons.append(comp)

        # ── Aggregate metrics ────────────────────────────────────────
        n = len(comparisons) or 1  # avoid ZeroDivisionError

        # Flip rate via answer extraction (not severity proxy).
        flip_count = sum(1 for c in comparisons if c.metadata.get("flipped", False))
        flip_rate = flip_count / n

        # KL divergence (only if both sides have logprobs).
        kl_values = [c.kl_divergence for c in comparisons if c.kl_divergence is not None]
        mean_kl: float | None = sum(kl_values) / len(kl_values) if kl_values else None

        mean_sim = sum(c.text_similarity for c in comparisons) / n

        # Per-category stats.
        cat_groups: dict[str, list[ComparisonResult]] = defaultdict(list)
        for c in comparisons:
            cat_groups[c.metadata.get("category", "general")].append(c)

        per_category: dict[str, dict[str, float | int]] = {}
        for cat, cat_comps in cat_groups.items():
            cat_n = len(cat_comps)
            cat_flips = sum(1 for c in cat_comps if c.metadata.get("flipped", False))
            per_category[cat] = {
                "count": cat_n,
                "flip_rate": cat_flips / cat_n,
                "mean_similarity": sum(c.text_similarity for c in cat_comps) / cat_n,
            }

        # ── Cleanup ──────────────────────────────────────────────────
        await backend_a.cleanup()
        await backend_b.cleanup()

        # ── Checkpoint ───────────────────────────────────────────────
        timestamp_int = int(datetime.now(UTC).timestamp())
        checkpoint_path = self.cache_dir / f"compare_{label_a}_vs_{label_b}_{timestamp_int}.json"
        self._save_checkpoint(comparisons, checkpoint_path)

        return CompareResult(
            model_a=label_a,
            model_b=label_b,
            backend_a=backend_a.name,
            backend_b=backend_b.name,
            comparisons=comparisons,
            flip_rate=flip_rate,
            mean_kl_divergence=mean_kl,
            mean_text_similarity=mean_sim,
            per_category_stats=per_category,
        )

    async def diff(
        self,
        baseline_backend: BackendAdapter,
        test_backends: list[BackendAdapter],
        prompts: list[Prompt],
    ) -> list[ComparisonResult]:
        """Compare outputs across different backends against a baseline."""
        baseline_results: dict[str, InferenceResult] = {}
        comparisons: list[ComparisonResult] = []

        with Progress() as progress:
            total_tasks = len(prompts) * (1 + len(test_backends))
            task = progress.add_task("[green]Diffing test...", total=total_tasks)

            # Baseline pass
            progress.update(
                task, description=f"[green]Running baseline backend ({baseline_backend.name})..."
            )
            for prompt in prompts:
                try:
                    res = await baseline_backend.generate(prompt)
                except Exception as exc:
                    from rich.console import Console

                    Console().print(
                        f"[yellow]  ⚠ Baseline failed for '{prompt.text[:60]}...': {exc}[/yellow]"
                    )
                    progress.advance(task)
                    continue
                baseline_results[res.prompt_id] = res
                progress.advance(task)

            # Test backends pass
            for test_backend in test_backends:
                progress.update(
                    task, description=f"[green]Running test backend ({test_backend.name})..."
                )
                for prompt in prompts:
                    try:
                        test_res = await test_backend.generate(prompt)
                    except Exception as exc:
                        from rich.console import Console

                        Console().print(
                            f"[yellow]  ⚠ {test_backend.name} failed for "
                            f"'{prompt.text[:60]}...': {exc}[/yellow]"
                        )
                        progress.advance(task)
                        continue
                    baseline = baseline_results.get(test_res.prompt_id)
                    if baseline:
                        comp = self._compare(baseline, test_res)
                        comparisons.append(comp)
                    progress.advance(task)

        return comparisons

    async def stress(
        self,
        backend: BackendAdapter,
        prompts: list[Prompt],
        concurrency_levels: list[int] | None = None,
    ) -> list[StressResult]:
        """Run stress tests with varying concurrency levels and compare for consistency."""
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4, 8]

        results: list[StressResult] = []
        baseline_results: dict[str, InferenceResult] = {}

        with Progress() as progress:
            task = progress.add_task(
                "[magenta]Running stress test...", total=len(concurrency_levels) * len(prompts)
            )

            for concurrency in concurrency_levels:
                sem = asyncio.Semaphore(concurrency)

                async def _run(
                    p: Prompt, _sem: asyncio.Semaphore = sem
                ) -> InferenceResult | Exception:
                    async with _sem:
                        try:
                            return await backend.generate(p)
                        except Exception as e:
                            return e

                coros = [_run(p) for p in prompts]
                # Gather returns results in the same order as coros
                batch_results_or_errs = await asyncio.gather(*coros, return_exceptions=True)
                progress.advance(task, advance=len(prompts))

                valid_results: list[InferenceResult] = []
                error_count = 0
                for r in batch_results_or_errs:
                    if isinstance(r, BaseException):
                        error_count += 1
                    else:
                        valid_results.append(r)

                consistent_count = 0
                total_compared = 0

                if not baseline_results:
                    # Treat the first run (e.g. concurrency=1) as the baseline
                    for valid_res in valid_results:
                        baseline_results[valid_res.prompt_id] = valid_res

                for valid_res in valid_results:
                    b = baseline_results.get(valid_res.prompt_id)
                    if b:
                        total_compared += 1
                        if b.text == valid_res.text:
                            consistent_count += 1

                output_consistency = consistent_count / max(1, total_compared)

                # Fetch model_id from results or backend config
                model_id = "unknown"
                if valid_results:
                    model_id = valid_results[0].model_id
                elif hasattr(backend, "config") and hasattr(backend.config, "model_id"):
                    model_id = backend.config.model_id

                sr = StressResult(
                    model_id=model_id,
                    backend_name=backend.name,
                    concurrency_level=concurrency,
                    results=valid_results,
                    error_count=error_count,
                    output_consistency=output_consistency,
                )
                results.append(sr)

        return results

    async def determinism(
        self, backend: BackendAdapter, prompts: list[Prompt], num_runs: int = 100
    ) -> list[DeterminismResult]:
        """Test determinism over multiple runs at temperature=0."""
        results: list[DeterminismResult] = []

        with Progress() as progress:
            task = progress.add_task(
                "[yellow]Running determinism test...", total=len(prompts) * num_runs
            )

            for prompt in prompts:
                # Force temp 0 for determinism checks
                if prompt.metadata is None:
                    prompt.metadata = {}
                prompt.metadata["temperature"] = 0.0

                run_results: list[InferenceResult] = []
                for _ in range(num_runs):
                    try:
                        res = await backend.generate(prompt)
                    except Exception as exc:
                        from rich.console import Console

                        Console().print(
                            f"[yellow]  ⚠ Determinism run failed for "
                            f"'{prompt.text[:60]}...': {exc}[/yellow]"
                        )
                        progress.advance(task)
                        continue
                    run_results.append(res)
                    progress.advance(task)

                if not run_results:
                    continue

                first = run_results[0]
                identical_count = 0
                divergence_positions = []

                for r in run_results:
                    if r.text == first.text:
                        identical_count += 1
                    else:
                        div_idx = None
                        for i, (t1, t2) in enumerate(zip(first.tokens, r.tokens, strict=False)):
                            if t1 != t2:
                                div_idx = i
                                break
                        if div_idx is None:
                            div_idx = min(len(first.tokens), len(r.tokens))
                        divergence_positions.append(div_idx)

                det_res = DeterminismResult(
                    prompt_id=prompt.id,
                    model_id=first.model_id,
                    backend_name=backend.name,
                    quantization=first.quantization,
                    num_runs=num_runs,
                    identical_count=identical_count,
                    divergence_positions=divergence_positions,
                    determinism_score=identical_count / max(1, num_runs),
                )
                results.append(det_res)

                timestamp = int(datetime.now(UTC).timestamp())
                checkpoint_path = self.cache_dir / f"determinism_{prompt.id}_{timestamp}.json"
                self._save_checkpoint(det_res, checkpoint_path)

        return results
