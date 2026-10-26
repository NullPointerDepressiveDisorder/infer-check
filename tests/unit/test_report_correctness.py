"""The HTML report leads with answer-key correctness and labels the KL method."""

from pathlib import Path

from infer_check.analysis.scoring import paired_correctness
from infer_check.reporting.html import generate_report
from infer_check.types import CompareResult, ComparisonResult, InferenceResult, SweepResult


def _res(text: str, backend: str = "b", quant: str | None = None) -> InferenceResult:
    return InferenceResult(
        prompt_id="p1", backend_name=backend, model_id="m", tokens=[], text=text, latency_ms=1.0, quantization=quant
    )


def _comp(grade_a: str, grade_b: str, quant: str | None = None) -> ComparisonResult:
    return ComparisonResult(
        baseline=_res("Final answer: 1", "mlx-lm"),
        test=_res("Final answer: 2", "mlx-lm", quant),
        text_similarity=0.9,
        is_failure=False,
        kl_divergence=0.05,
        kl_method="teacher_forced",
        kl_positions=10,
        top1_agreement=0.95,
        metadata={"grade_a": grade_a, "grade_b": grade_b, "flipped": grade_a != grade_b},
    )


def test_compare_report_shows_correctness_and_kl_method(tmp_path: Path) -> None:
    pairs = [("correct", "wrong")] * 12 + [("correct", "correct")] * 20 + [("wrong", "correct")]
    comps = [_comp(a, b) for a, b in pairs]
    CompareResult(
        model_a="bf16",
        model_b="4bit",
        backend_a="mlx-lm",
        backend_b="mlx-lm",
        comparisons=comps,
        flip_rate=13 / 33,
        correctness=paired_correctness(pairs),  # type: ignore[arg-type]
        mean_kl_divergence=0.05,
        kl_method="teacher_forced",
        mean_top1_agreement=0.95,
        mean_text_similarity=0.9,
    ).save(tmp_path / "compare_x.json")

    html = generate_report(tmp_path, tmp_path / "report.html").read_text()

    assert "Correctness vs answer keys (33 graded prompts)" in html
    assert "12 regressions / 1 improvements" in html
    assert "teacher-forced" in html
    assert "Top-1 Agreement" in html
    # Executive summary verdict is about answers, and flags the significant loss.
    assert "Graded against answer keys: 12 regression(s) and 1 improvement(s)" in html
    assert "Statistically significant accuracy loss: bf16 → 4bit" in html
    # The old misleading verdict is gone.
    assert "correctness issue(s) found" not in html


def test_ungraded_report_says_text_differences_are_not_wrong_answers(tmp_path: Path) -> None:
    comp = ComparisonResult(
        baseline=_res("hello"), test=_res("goodbye"), text_similarity=0.2, is_failure=True, metadata={}
    )
    CompareResult(
        model_a="a",
        model_b="b",
        backend_a="x",
        backend_b="x",
        comparisons=[comp],
        flip_rate=0.0,
    ).save(tmp_path / "compare_y.json")
    html = generate_report(tmp_path, tmp_path / "report.html").read_text()
    assert "Text differences alone don&#39;t mean wrong answers" in html
    assert "Correctness vs answer keys" not in html


def test_sweep_report_has_per_level_table(tmp_path: Path) -> None:
    pairs_8 = [("correct", "correct")] * 10
    pairs_4 = [("correct", "wrong")] * 6 + [("correct", "correct")] * 4
    SweepResult(
        model_id="m",
        backend_name="mlx-lm",
        quantization_levels=["bf16", "8bit", "4bit"],
        comparisons=[_comp(a, b, "8bit") for a, b in pairs_8] + [_comp(a, b, "4bit") for a, b in pairs_4],
        timestamp=__import__("datetime").datetime.now(),
        summary={
            "baseline_quant": "bf16",
            "kl_by_quant": {
                "bf16": {"method": "teacher_forced", "mean_kl": 0.0, "mean_top1_agreement": 1.0},
                "8bit": {"method": "teacher_forced", "mean_kl": 0.0006, "mean_top1_agreement": 0.994},
                "4bit": {"method": "teacher_forced", "mean_kl": 0.09, "mean_top1_agreement": 0.92},
            },
            "correctness_by_quant": {
                "bf16": None,
                "8bit": paired_correctness(pairs_8),  # type: ignore[arg-type]
                "4bit": paired_correctness(pairs_4),  # type: ignore[arg-type]
            },
        },
    ).save(tmp_path / "sweep_z.json")
    html = generate_report(tmp_path, tmp_path / "report.html").read_text()
    assert "By level, against the baseline" in html
    assert "0.0900 (teacher-forced)" in html
    assert "100.0% → 40.0%" in html
    assert "(self-check)" in html
    assert "Text similarity per prompt" in html
