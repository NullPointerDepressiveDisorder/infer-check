import asyncio
import difflib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.progress import Progress

from infer_check.analysis.divergence import KLStats, aggregate_kl, prefix_kl, teacher_forced_kl
from infer_check.backends.base import BackendAdapter
from infer_check.types import (
    CompareResult,
    ComparisonResult,
    DeterminismResult,
    InferenceResult,
    Prompt,
    StressResult,
    SweepResult,
    TokenScores,
)
from infer_check.utils import sanitize_filename

_RESULT_DISTRIBUTIONS = {"distributions", "distribution_metadata"}
_DISTRIBUTION_FIELDS: dict[str, Any] = {"baseline": _RESULT_DISTRIBUTIONS, "test": _RESULT_DISTRIBUTIONS}

# Baseline tokens kept per position for teacher-forced KL. The remaining mass is
# merged into one bin, so KL is a (very tight) lower bound.
REFERENCE_TOP_K = 64


def _warn(message: str) -> None:
    from rich.console import Console

    Console().print(f"[yellow]  ⚠ {message}[/yellow]")


def flip_rate(comparisons: list[ComparisonResult]) -> float:
    """Fraction of flip-eligible comparisons whose answer flipped.

    Comparisons with ``metadata["flipped"] is None`` (no meaningful answer to
    compare) are excluded from the denominator.
    """
    eligible = [c for c in comparisons if c.metadata.get("flipped") is not None]
    return sum(1 for c in eligible if c.metadata["flipped"]) / len(eligible) if eligible else 0.0


def tokenizers_conflict(a: BackendAdapter, b: BackendAdapter) -> bool:
    """True only when both tokenizer fingerprints are known and differ.

    Token IDs from one tokenizer mean different tokens to another, so KL, top-1
    agreement and token divergence between such backends are meaningless. An
    unknown fingerprint (HTTP backends) means "can't tell", not "conflict".
    """
    fa, fb = a.tokenizer_fingerprint, b.tokenizer_fingerprint
    return fa is not None and fb is not None and fa != fb


def find_degradation_cliff(
    levels: list[str],
    baseline: str,
    correctness_by_level: dict[str, dict[str, Any] | None],
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """First level, in the order given, with a statistically significant accuracy loss.

    "Significant" means more regressions than improvements against the baseline
    with exact McNemar p < ``alpha``. Levels are walked in the order the user
    listed them (typically decreasing precision), never sorted by name.
    Without answer keys there is no accuracy to lose, so this returns None.

    Returns:
        ``{"level", "regressions", "improvements", "mcnemar_p"}`` or None.
    """
    for level in levels:
        if level == baseline:
            continue
        table = correctness_by_level.get(level)
        if not table:
            continue
        if table["regressions"] > table["improvements"] and table["mcnemar_p"] < alpha:
            return {
                "level": level,
                "regressions": table["regressions"],
                "improvements": table["improvements"],
                "mcnemar_p": table["mcnemar_p"],
            }
    return None


def correctness_summary(comparisons: list[ComparisonResult]) -> dict[str, Any] | None:
    """Paired correctness table (baseline = A, test = B) over the graded comparisons."""
    from infer_check.analysis.scoring import paired_correctness

    pairs = [
        (c.metadata["grade_a"], c.metadata["grade_b"])
        for c in comparisons
        if c.metadata.get("grade_a") is not None and c.metadata.get("grade_b") is not None
    ]
    return paired_correctness(pairs)


def first_divergence(a: InferenceResult, b: InferenceResult) -> int | None:
    """Index of the first token where two generations differ, or None if identical.

    Compares token IDs when both results have them (exact). Otherwise falls
    back to per-token display strings, which can split multi-byte characters
    and so over-report divergence. If one sequence is a prefix of the other,
    the divergence index is the shorter length.
    """
    seq_a: list[int] | list[str]
    seq_b: list[int] | list[str]
    if a.token_ids is not None and b.token_ids is not None:
        seq_a, seq_b = a.token_ids, b.token_ids
    else:
        seq_a, seq_b = a.tokens, b.tokens
    for i, (x, y) in enumerate(zip(seq_a, seq_b, strict=False)):
        if x != y:
            return i
    if len(seq_a) != len(seq_b):
        return min(len(seq_a), len(seq_b))
    return None


class TestRunner:
    __test__ = False

    def __init__(self, cache_dir: str | Path = ".infer_check_cache", keep_distributions: bool = False):
        """
        Args:
            cache_dir: Where intermediate checkpoints are written.
            keep_distributions: Keep each result's per-token top-K distributions in
                the returned and saved results. They are only inputs to the KL
                computation, which happens during the run, and they dominate file
                size (~400 KB per prompt), so by default they are dropped once
                every comparison needing them has been computed.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.keep_distributions = keep_distributions
        self._tokenizer_warned: set[tuple[int, int]] = set()

    def _token_level(self, a: BackendAdapter, b: BackendAdapter) -> bool:
        """Whether token-level metrics are meaningful between ``a`` and ``b`` (warns once per pair)."""
        if not tokenizers_conflict(a, b):
            return True
        if (id(a), id(b)) not in self._tokenizer_warned:
            self._tokenizer_warned.add((id(a), id(b)))
            _warn(
                f"{a.name} ({a.model_id}) and {b.name} ({b.model_id}) use different tokenizers: "
                "token-level metrics (KL, top-1 agreement, token divergence) skipped. "
                "Correctness and text similarity are still reported."
            )
        return False

    def _drop_distributions(self, comparisons: list[ComparisonResult]) -> None:
        """Release stored top-K distributions once KL has been computed (unless keeping them)."""
        if self.keep_distributions:
            return
        for comp in comparisons:
            for res in (comp.baseline, comp.test):
                res.distributions = None
                res.distribution_metadata = None

    def _annotate_flip_metadata(
        self,
        comp: ComparisonResult,
        prompt: Prompt,
    ) -> None:
        """Grade both sides and decide whether the answer flipped, updating comparison metadata.

        Graded prompts (with an answer key): each side gets a grade
        (correct / wrong / unparsed / truncated); a flip is one side correct and
        the other wrong. As in ``paired_correctness``, an unparsed response, or a
        truncation where the other side finished, counts as wrong.

        Free prompts: falls back to category-based answer extraction. Prompts
        whose category has no real extractor, or where no answer could be
        extracted from one side, would only be compared by raw text similarity,
        which says nothing about the answer. They get ``flipped = None`` and are
        excluded from the flip rate.
        """
        from infer_check.analysis.answer_extract import answers_match, extract_answer
        from infer_check.analysis.scoring import final_answer, grade

        text_a, text_b = comp.baseline.text, comp.test.text

        if prompt.is_graded:
            grade_a, grade_b = grade(comp.baseline, prompt), grade(comp.test, prompt)
            comp.metadata["expected"] = prompt.expected
            comp.metadata["answer_type"] = prompt.answer_type
            comp.metadata["grade_a"] = grade_a
            comp.metadata["grade_b"] = grade_b
            comp.metadata["answer_a"] = final_answer(text_a)
            comp.metadata["answer_b"] = final_answer(text_b)
            comp.metadata["extraction_strategy"] = "answer_key"

            # Same rules as paired_correctness: no parseable answer, or running out of
            # budget where the other side finished, is a wrong answer.
            def eff(own: str | None, other: str | None) -> str | None:
                if own == "unparsed" or (own == "truncated" and other != "truncated"):
                    return "wrong"
                return own

            eff_a, eff_b = eff(grade_a, grade_b), eff(grade_b, grade_a)
            gradable = {"correct", "wrong"}
            comp.metadata["flipped"] = eff_a in gradable and eff_b in gradable and eff_a != eff_b
            return

        ans_a = extract_answer(text_a, prompt.category)
        ans_b = extract_answer(text_b, prompt.category)
        comp.metadata["answer_a"] = ans_a.value
        comp.metadata["answer_b"] = ans_b.value
        comp.metadata["extraction_strategy"] = ans_a.strategy
        comp.metadata["extraction_confidence"] = min(ans_a.confidence, ans_b.confidence)
        if ans_a.strategy == "raw" or not ans_a.value or not ans_b.value:
            # No extractor for this category, or no answer found on one side:
            # there's nothing to compare, so this prompt isn't flip-eligible.
            comp.metadata["flipped"] = None
        else:
            comp.metadata["flipped"] = not answers_match(ans_a, ans_b)

    def _save_checkpoint(self, results: Any, path: Path) -> None:
        """Write intermediate results as JSON for resumability."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # A sweep checkpoint can be written while the shared baseline results still
        # hold distributions (they're dropped at the end), so exclude them here too.
        exclude = None if self.keep_distributions else _DISTRIBUTION_FIELDS
        if hasattr(results, "model_dump_json"):
            path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
        elif isinstance(results, list):
            dumped = [
                r.model_dump(mode="json", exclude=exclude if isinstance(r, ComparisonResult) else None)
                if hasattr(r, "model_dump")
                else r
                for r in results
            ]
            path.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _compare(
        self,
        baseline: InferenceResult,
        test: InferenceResult,
        threshold: float = 0.5,
        kl: KLStats | None = None,
        token_level: bool = True,
    ) -> ComparisonResult:
        """Compare a test result against a baseline.

        Threshold is the *minimum* text similarity to pass. Default 0.5
        separates "different wording, same answer" from "actually broken."

        Severity tiers (stored in metadata):
          - identical:  similarity == 1.0
          - minor:      similarity >= 0.8
          - moderate:   similarity >= 0.5
          - severe:     similarity < 0.5

        KL: pass teacher-forced ``kl`` stats when both backends could score the
        baseline's tokens. Otherwise an approximate prefix KL is computed from
        the stored top-K distributions (see ``prefix_kl``).
        """
        if baseline.text == test.text:
            text_similarity = 1.0
        else:
            matcher = difflib.SequenceMatcher(None, baseline.text, test.text)
            text_similarity = matcher.ratio()

        is_failure = text_similarity < threshold

        # Severity tiers
        if text_similarity == 1.0:
            severity = "identical"
        elif text_similarity >= 0.8:
            severity = "minor"
        elif text_similarity >= 0.5:
            severity = "moderate"
        else:
            severity = "severe"

        # Token IDs are only comparable when both sides share a tokenizer.
        token_divergence_index = first_divergence(baseline, test) if token_level else None
        if not token_level:
            kl = None
        elif kl is None:
            kl = prefix_kl(baseline, test, token_divergence_index)

        return ComparisonResult(
            baseline=baseline,
            test=test,
            text_similarity=text_similarity,
            is_failure=is_failure,
            token_divergence_index=token_divergence_index,
            kl_divergence=kl.mean if kl else None,
            kl_method=kl.method if kl else None,
            kl_max=kl.max if kl else None,
            kl_p99=kl.p99 if kl else None,
            kl_positions=kl.n_positions if kl else None,
            kl_per_token=kl.per_token if kl else None,
            top1_agreement=kl.top1_agreement if kl else None,
            failure_reason=f"[{severity}] Text similarity {text_similarity:.3f} below threshold"
            if is_failure
            else None,
            metadata={
                "severity": severity,
                "prompt_text": baseline.metadata.get("prompt_text"),
                "prompt_category": baseline.metadata.get("prompt_category"),
                "category": baseline.metadata.get("prompt_category") or baseline.metadata.get("category"),
            },
        )

    async def _reference_scores(
        self,
        backend: BackendAdapter,
        prompts: list[Prompt],
        results: dict[str, InferenceResult],
    ) -> dict[str, TokenScores]:
        """Score each baseline result's own tokens with the baseline model.

        These top-K distributions are the reference that test models are
        compared against. Must run while the baseline model is still loaded.
        Returns an empty dict if the backend can't score.
        """
        refs: dict[str, TokenScores] = {}
        if not backend.supports_scoring:
            return refs
        for prompt in prompts:
            res = results.get(prompt.id)
            if res is None or not res.token_ids:
                continue
            try:
                refs[prompt.id] = await backend.score(prompt, res.token_ids, top_k=REFERENCE_TOP_K)
            except Exception as exc:
                _warn(f"Reference scoring failed for '{prompt.id}' on {backend.name}: {exc}")
        return refs

    async def _teacher_forced(
        self,
        backend: BackendAdapter,
        prompt: Prompt,
        baseline: InferenceResult,
        ref: TokenScores | None,
        token_level: bool = True,
    ) -> KLStats | None:
        """Score the baseline's tokens with ``backend`` and compare against ``ref``.

        The baseline's token IDs are fed to the test model as-is, so this only
        makes sense when both share a tokenizer; callers pass ``token_level``
        from ``_token_level``. Returns None when token-level metrics are off,
        or when either side can't score (the caller then falls back to prefix KL).
        """
        if not token_level or ref is None or not backend.supports_scoring or not baseline.token_ids:
            return None
        try:
            scores = await backend.score(prompt, baseline.token_ids, top_k=0, at_ids=ref.top_ids)
        except Exception as exc:
            _warn(f"Teacher-forced scoring failed for '{prompt.id}' on {backend.name}: {exc}")
            return None
        return teacher_forced_kl(ref, scores)

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
        failure_counts: dict[str, int] = dict.fromkeys(quantization_levels, 0)

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
                    res.metadata["prompt_text"] = prompt.text
                    res.metadata["prompt_category"] = prompt.category
                except Exception as exc:
                    from rich.console import Console

                    Console().print(
                        f"[yellow]  ⚠ Skipping prompt '{prompt.text[:60]}...' at {baseline_quant}: {exc}[/yellow]"
                    )
                    progress.advance(task)
                    continue
                if not res.quantization:
                    res.quantization = baseline_quant
                baseline_results[res.prompt_id] = res
                progress.advance(task)

            # ---- Reference scores for teacher-forced KL (baseline still loaded) ----
            if baseline_backend.supports_scoring:
                progress.update(task, description=f"[cyan]Scoring baseline reference ({baseline_quant})...")
            refs = await self._reference_scores(baseline_backend, prompts, baseline_results)

            # ---- Baseline pass 2: self-check for determinism ----
            progress.update(task, description=f"[cyan]Baseline self-check ({baseline_quant})...")
            for prompt in prompts:
                if prompt.id not in baseline_results:
                    progress.advance(task)
                    continue
                try:
                    res2 = await baseline_backend.generate(prompt)
                    res2.metadata["prompt_text"] = prompt.text
                    res2.metadata["prompt_category"] = prompt.category
                except Exception as exc:
                    from rich.console import Console

                    Console().print(f"[yellow]  ⚠ Self-check failed for '{prompt.text[:60]}...': {exc}[/yellow]")
                    progress.advance(task)
                    continue
                if not res2.quantization:
                    res2.quantization = baseline_quant
                ref_res = baseline_results[prompt.id]
                # Re-scoring with the same model should give KL == 0 exactly;
                # anything else means scoring itself is nondeterministic.
                kl = await self._teacher_forced(baseline_backend, prompt, ref_res, refs.get(prompt.id))
                comp = self._compare(ref_res, res2, kl=kl)
                self._annotate_flip_metadata(comp, prompt)
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
                        res.metadata["prompt_text"] = prompt.text
                        res.metadata["prompt_category"] = prompt.category
                    except Exception as exc:
                        from rich.console import Console

                        Console().print(
                            f"[yellow]  ⚠ Skipping prompt '{prompt.text[:60]}...' at {quant}: {exc}[/yellow]"
                        )
                        progress.advance(task)
                        continue
                    if not res.quantization:
                        res.quantization = quant

                    baseline = baseline_results.get(res.prompt_id)
                    if baseline:
                        tl = self._token_level(baseline_backend, backend)
                        kl = await self._teacher_forced(backend, prompt, baseline, refs.get(prompt.id), tl)
                        comp = self._compare(baseline, res, kl=kl, token_level=tl)
                        self._annotate_flip_metadata(comp, prompt)
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

        # KL per quantization level. Pooling across levels would be meaningless.
        kl_by_quant = {
            quant: aggregate_kl([c for c in all_comparisons if (c.test.quantization or "unknown") == quant])
            for quant in quantization_levels
        }
        # Back-compat key: pooled over non-baseline levels only.
        pooled = aggregate_kl([c for c in all_comparisons if not c.metadata.get("is_self_check")])

        summary = {
            "total_prompts": len(prompts),
            "quantization_levels": quantization_levels,
            "baseline_quant": baseline_quant,
            "failure_counts": failure_counts,
            "mean_kl_divergence": pooled["mean_kl"],
            "kl_method": pooled["method"],
            "kl_by_quant": kl_by_quant,
            # Paired correctness of each level against the baseline (None without answer keys).
            "correctness_by_quant": {
                quant: correctness_summary([c for c in all_comparisons if (c.test.quantization or "unknown") == quant])
                for quant in quantization_levels
            },
        }
        summary["degradation_cliff"] = find_degradation_cliff(
            quantization_levels, baseline_quant, summary["correctness_by_quant"]
        )

        self._drop_distributions(all_comparisons)

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
                    res.metadata["prompt_text"] = prompt.text
                    res.metadata["prompt_category"] = prompt.category
                except Exception as exc:
                    from rich.console import Console

                    Console().print(f"[yellow]  ⚠ {label_a} failed for '{prompt.text[:60]}...': {exc}[/yellow]")
                    progress.advance(task)
                    continue
                if not res.quantization:
                    res.quantization = label_a
                results_a[res.prompt_id] = res
                progress.advance(task)

            # ── Reference scores from A (teacher-forced KL) ──────────
            if backend_a.supports_scoring:
                progress.update(task, description=f"[cyan]Scoring reference ({label_a})...")
            refs = await self._reference_scores(backend_a, prompts, results_a)

            # ── Pass 2: generate from model B ────────────────────────
            progress.update(task, description=f"[cyan]Generating from {label_b}...")
            for prompt in prompts:
                try:
                    res = await backend_b.generate(prompt)
                    res.metadata["prompt_text"] = prompt.text
                    res.metadata["prompt_category"] = prompt.category
                except Exception as exc:
                    from rich.console import Console

                    Console().print(f"[yellow]  ⚠ {label_b} failed for '{prompt.text[:60]}...': {exc}[/yellow]")
                    progress.advance(task)
                    continue
                if not res.quantization:
                    res.quantization = label_b
                results_b[res.prompt_id] = res
                progress.advance(task)

        # ── Build comparisons with answer extraction ────────────────
        tl = self._token_level(backend_a, backend_b)
        for prompt in prompts:
            a = results_a.get(prompt.id)
            b = results_b.get(prompt.id)
            if a and b:
                kl = await self._teacher_forced(backend_b, prompt, a, refs.get(prompt.id), tl)
                comp = self._compare(a, b, kl=kl, token_level=tl)
                self._annotate_flip_metadata(comp, prompt)
                comparisons.append(comp)

        # ── Aggregate metrics ────────────────────────────────────────
        n = len(comparisons)

        # Flip rate over flip-eligible prompts; correctness over prompts with answer keys.
        overall_flip_rate = flip_rate(comparisons)
        correctness = correctness_summary(comparisons)

        kl_summary = aggregate_kl(comparisons)

        mean_sim: float | None = sum(c.text_similarity for c in comparisons) / n if n > 0 else None

        # Per-category stats.
        cat_groups: dict[str, list[ComparisonResult]] = defaultdict(list)
        for c in comparisons:
            cat_groups[c.metadata.get("prompt_category", "general")].append(c)

        per_category: dict[str, dict[str, Any]] = {}
        for cat, cat_comps in cat_groups.items():
            cat_n = len(cat_comps)
            cat_kl = aggregate_kl(cat_comps)
            per_category[cat] = {
                "count": cat_n,
                "flip_rate": flip_rate(cat_comps),
                "correctness": correctness_summary(cat_comps),
                "mean_similarity": sum(c.text_similarity for c in cat_comps) / cat_n,
                "mean_kl_divergence": cat_kl["mean_kl"],
                "mean_top1_agreement": cat_kl["mean_top1_agreement"],
            }

        self._drop_distributions(comparisons)

        # ── Cleanup ──────────────────────────────────────────────────
        await backend_a.cleanup()
        await backend_b.cleanup()

        # ── Checkpoint ───────────────────────────────────────────────
        timestamp_int = int(datetime.now(UTC).timestamp())

        # Sanitize labels to avoid path traversal and filesystem issues
        safe_a = sanitize_filename(label_a)
        safe_b = sanitize_filename(label_b)

        checkpoint_path = self.cache_dir / f"compare_{safe_a}_vs_{safe_b}_{timestamp_int}.json"
        self._save_checkpoint(comparisons, checkpoint_path)

        return CompareResult(
            model_a=label_a,
            model_b=label_b,
            backend_a=backend_a.name,
            backend_b=backend_b.name,
            comparisons=comparisons,
            flip_rate=overall_flip_rate,
            correctness=correctness,
            mean_kl_divergence=kl_summary["mean_kl"],
            kl_method=kl_summary["method"],
            mean_top1_agreement=kl_summary["mean_top1_agreement"],
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
            progress.update(task, description=f"[green]Running baseline backend ({baseline_backend.name})...")
            for prompt in prompts:
                try:
                    res = await baseline_backend.generate(prompt)
                    res.metadata["prompt_text"] = prompt.text
                    res.metadata["prompt_category"] = prompt.category
                except Exception as exc:
                    from rich.console import Console

                    Console().print(f"[yellow]  ⚠ Baseline failed for '{prompt.text[:60]}...': {exc}[/yellow]")
                    progress.advance(task)
                    continue
                baseline_results[res.prompt_id] = res
                progress.advance(task)

            # Reference scores for teacher-forced KL (only used if a test backend can score too).
            refs: dict[str, TokenScores] = {}
            if any(tb.supports_scoring for tb in test_backends):
                refs = await self._reference_scores(baseline_backend, prompts, baseline_results)

            # Test backends pass
            for test_backend in test_backends:
                progress.update(task, description=f"[green]Running test backend ({test_backend.name})...")
                for prompt in prompts:
                    try:
                        test_res = await test_backend.generate(prompt)
                        test_res.metadata["prompt_text"] = prompt.text
                        test_res.metadata["prompt_category"] = prompt.category
                    except Exception as exc:
                        from rich.console import Console

                        Console().print(
                            f"[yellow]  ⚠ {test_backend.name} failed for '{prompt.text[:60]}...': {exc}[/yellow]"
                        )
                        progress.advance(task)
                        continue
                    baseline = baseline_results.get(test_res.prompt_id)
                    if baseline:
                        tl = self._token_level(baseline_backend, test_backend)
                        kl = await self._teacher_forced(test_backend, prompt, baseline, refs.get(prompt.id), tl)
                        comp = self._compare(baseline, test_res, kl=kl, token_level=tl)
                        self._annotate_flip_metadata(comp, prompt)
                        comparisons.append(comp)
                    progress.advance(task)

        self._drop_distributions(comparisons)
        return comparisons

    @staticmethod
    async def _safe_generate(backend: BackendAdapter, prompt: Prompt) -> InferenceResult | Exception:
        try:
            res = await backend.generate(prompt)
        except Exception as exc:
            return exc
        res.metadata["prompt_text"] = prompt.text
        res.metadata["prompt_category"] = prompt.category
        return res

    async def stress(
        self,
        backend: BackendAdapter,
        prompts: list[Prompt],
        concurrency_levels: list[int] | None = None,
        *,
        fillers: list[Prompt] | None = None,
        trials: int = 1,
        seed: int = 0,
    ) -> list[StressResult]:
        """Check that a prompt's output doesn't depend on what else is in flight.

        Batching servers compute each request together with whatever else is
        scheduled, and some kernels give slightly different results depending on
        batch size and composition. Under greedy decoding a single flipped token
        changes the rest of the output. This is the main source of serving
        nondeterminism.

        - Level 1: every prompt runs alone, one at a time. This is the reference.
        - Level N > 1: each prompt (the target) is sent at the same time as N-1
          *other* prompts drawn at random (seeded) from ``fillers`` (default: the
          suite itself), and its output is compared token-for-token with its solo
          run. Each trial draws new fillers, so batch composition varies.

        Args:
            backend: Must support real concurrency (an HTTP server). In-process
                backends run requests one at a time, so the test could never fail.
            prompts: Targets.
            concurrency_levels: Requests in flight per batch; 1 is always run first.
            fillers: Prompts to co-schedule with each target (default: ``prompts``).
            trials: Batches per target per level, each with fresh fillers.
            seed: Seed for filler selection, so runs are reproducible.

        Raises:
            ValueError: if the backend doesn't support concurrency.
        """
        import random

        if not backend.supports_concurrency:
            raise ValueError(
                f"The {backend.name} backend runs requests one at a time, so a stress test can't fail on it. "
                "Run stress against an HTTP server (llama-server, vllm-mlx, mlx_lm.server, vLLM, ...)."
            )
        levels = sorted({lvl for lvl in (concurrency_levels or [1, 2, 4, 8]) if lvl >= 1} | {1})
        pool = fillers or prompts
        rng = random.Random(seed)  # nosec B311 - reproducible test sampling, not security

        total = len(prompts) * (1 + trials * (len(levels) - 1))
        results: list[StressResult] = []

        with Progress() as progress:
            task = progress.add_task("[magenta]Stress: solo reference...", total=total)

            def note(messages: list[str], exc: Exception) -> None:
                msg = f"{type(exc).__name__}: {exc}"[:300]
                if msg not in messages and len(messages) < 10:
                    messages.append(msg)

            # ── Level 1: each prompt alone ──────────────────────────────
            solo: dict[str, InferenceResult] = {}
            solo_errors = 0
            solo_messages: list[str] = []
            for prompt in prompts:
                out = await self._safe_generate(backend, prompt)
                if isinstance(out, Exception):
                    solo_errors += 1
                    note(solo_messages, out)
                    _warn(f"Solo run failed for '{prompt.id}': {out}")
                else:
                    solo[prompt.id] = out
                progress.advance(task)
            results.append(
                StressResult(
                    model_id=next(iter(solo.values())).model_id if solo else backend.model_id,
                    backend_name=backend.name,
                    concurrency_level=1,
                    results=list(solo.values()),
                    error_count=solo_errors,
                    output_consistency=1.0,  # the reference, by definition
                    compared=len(solo),
                    error_messages=solo_messages,
                )
            )

            # ── Level N: each target alongside N-1 other prompts ────────
            for level in levels[1:]:
                progress.update(task, description=f"[magenta]Stress: {level} in flight...")
                outputs: list[InferenceResult] = []
                divergences: list[dict[str, Any]] = []
                messages: list[str] = []
                errors = consistent = compared = 0
                for trial in range(trials):
                    for target in prompts:
                        reference = solo.get(target.id)
                        if reference is None:
                            progress.advance(task)
                            continue
                        others = [p for p in pool if p.id != target.id] or pool
                        k = level - 1
                        chosen = rng.sample(others, k) if len(others) >= k else [rng.choice(others) for _ in range(k)]
                        batch = await asyncio.gather(*(self._safe_generate(backend, p) for p in [target, *chosen]))
                        for o in batch:
                            if isinstance(o, Exception):
                                errors += 1
                                note(messages, o)
                        out = batch[0]
                        progress.advance(task)
                        if isinstance(out, Exception):
                            continue
                        compared += 1
                        outputs.append(out)
                        if out.text == reference.text:
                            consistent += 1
                        else:
                            divergences.append(
                                {
                                    "prompt_id": target.id,
                                    "trial": trial,
                                    "first_divergence": first_divergence(reference, out),
                                    "co_scheduled": [p.id for p in chosen],
                                }
                            )
                results.append(
                    StressResult(
                        model_id=outputs[0].model_id if outputs else backend.model_id,
                        backend_name=backend.name,
                        concurrency_level=level,
                        results=outputs,
                        error_count=errors,
                        output_consistency=consistent / compared if compared else 0.0,
                        trials=trials,
                        compared=compared,
                        divergences=divergences,
                        error_messages=messages,
                    )
                )

        return results

    async def determinism(
        self, backend: BackendAdapter, prompts: list[Prompt], num_runs: int = 100
    ) -> list[DeterminismResult]:
        """Test determinism over multiple runs at temperature=0."""
        results: list[DeterminismResult] = []

        with Progress() as progress:
            task = progress.add_task("[yellow]Running determinism test...", total=len(prompts) * num_runs)

            for prompt in prompts:
                # Force temp 0 for determinism checks
                if prompt.metadata is None:
                    prompt.metadata = {}
                prompt.metadata["temperature"] = 0.0

                run_results: list[InferenceResult] = []
                for _ in range(num_runs):
                    try:
                        res = await backend.generate(prompt)
                        res.metadata["prompt_text"] = prompt.text
                        res.metadata["prompt_category"] = prompt.category
                    except Exception as exc:
                        from rich.console import Console

                        Console().print(
                            f"[yellow]  ⚠ Determinism run failed for '{prompt.text[:60]}...': {exc}[/yellow]"
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
                        div_idx = first_divergence(first, r)
                        if div_idx is None:
                            # Same tokens but different text shouldn't happen; record the
                            # end of the sequence rather than dropping the divergence.
                            div_idx = len(first.token_ids or first.tokens)
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
