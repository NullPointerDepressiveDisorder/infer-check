"""Answer-key scoring: parsing, grading, paired statistics."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from infer_check.analysis.scoring import (
    final_answer,
    grade,
    is_correct,
    mcnemar_exact,
    model_input,
    paired_correctness,
    parse_number,
    wilson_interval,
)
from infer_check.types import InferenceResult, Prompt

# ── final_answer ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Working...\nFinal answer: 42", "42"),
        ("Final Answer: 42.", "42"),
        ("**Final answer:** 42", "42"),
        ("Final answer**: 42", "42"),
        ("final answer：42", "42"),  # fullwidth colon
        ("Final answer: $1,234$", "1,234"),
        ("Final answer: 1\nwait, recheck\nFinal answer: 2", "2"),  # last one wins
        ("So the result is \\boxed{17}.", "17"),
        ("\\boxed{\\frac{1}{2}} ... \\boxed{3}", "3"),
        ("Final answer: 5 and \\boxed{9}", "5 and \\boxed{9}"),  # explicit line beats boxed
    ],
)
def test_final_answer_formats(text: str, expected: str) -> None:
    assert final_answer(text) == expected


@pytest.mark.parametrize("text", ["The answer is 42.", "Final answer 42", "final answers: 3", ""])
def test_final_answer_requires_the_marker(text: str) -> None:
    # A space instead of a colon, or no marker, is not a committed answer.
    assert final_answer(text) is None


# ── parse_number ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ans", "value"),
    [
        ("42", "42"),
        ("1,234,567", "1234567"),
        ("-17", "-17"),
        ("\u22125", "-5"),  # unicode minus
        ("3.25", "3.25"),
        ("1.5e9", "1500000000"),
        ("1.5 × 10^9", "1500000000"),
        ("2 x 10^{-3}", "0.002"),
        ("10^308", "1e308"),
        ("10**15", "1000000000000000"),
        ("-1/8", "-0.125"),
        ("1627/2520", str(Decimal(1627) / Decimal(2520))),
        ("$184,345,654.33", "184345654.33"),
        ("about 7 apples", "7"),
    ],
)
def test_parse_number(ans: str, value: str) -> None:
    assert parse_number(ans) == Decimal(value)


def test_parse_number_none() -> None:
    assert parse_number("no idea") is None


# ── is_correct ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected", "answer_type", "result"),
    [
        ("Final answer: 500500", "500500", "integer", True),
        ("Final answer: 500,500", "500500", "integer", True),
        ("Final answer: 500501", "500500", "integer", False),
        ("Final answer: 18.0", "18", "integer", True),
        ("Final answer: 1 × 10^16", "10000000000000000", "integer", True),
        ("Final answer: 179.625", "179.625", "decimal", True),
        ("Final answer: 184345654.33", "184345654.3348", "decimal", True),  # within 1e-6 relative
        ("Final answer: 184000000", "184345654.3348", "decimal", False),
        ("Final answer: Yes.", "yes", "boolean", True),
        ("Final answer: **True**", "yes", "boolean", True),
        ("Final answer: no", "yes", "boolean", False),
        ("Final answer: (B)", "B", "choice", True),
        ("Final answer: c", "C", "choice", True),
        ("Final answer: A", "C", "choice", False),
        ("No marker here: 500500", "500500", "integer", None),
        ("Final answer: unsure", "500500", "integer", None),
        ("Final answer: maybe", "yes", "boolean", None),
    ],
)
def test_is_correct(text: str, expected: str, answer_type: str, result: bool | None) -> None:
    assert is_correct(text, expected, answer_type) is result  # type: ignore[arg-type]


def test_explicit_line_wins_over_boxed() -> None:
    assert is_correct("Final answer: 5 and \\boxed{9}", "5", "integer") is True


def test_bad_answer_key_raises() -> None:
    with pytest.raises(ValueError, match="answer key"):
        is_correct("Final answer: 3", "three", "integer")


# ── grade / prompts ─────────────────────────────────────────────────


def _result(text: str, finish: str | None = "stop") -> InferenceResult:
    return InferenceResult(
        prompt_id="p",
        backend_name="b",
        model_id="m",
        tokens=[],
        text=text,
        latency_ms=1.0,
        metadata={"finish_reason": finish},
    )


def test_grade() -> None:
    p = Prompt(id="p", text="1+1?", answer_type="integer", expected="2")
    assert grade(_result("Final answer: 2"), p) == "correct"
    assert grade(_result("Final answer: 3"), p) == "wrong"
    assert grade(_result("It's two."), p) == "unparsed"
    # Cut off before answering: not the model's mistake.
    assert grade(_result("Let me think step by step...", finish="length"), p) == "truncated"
    # But an answer given before the cutoff still counts.
    assert grade(_result("Final answer: 2\nAlso", finish="length"), p) == "correct"
    assert grade(_result("anything"), Prompt(text="free")) is None


def test_model_input_appends_instruction_only_for_graded_prompts() -> None:
    graded = Prompt(text="What is 2+2?", answer_type="integer", expected="4")
    assert model_input(graded).startswith("What is 2+2?\n\n")
    assert "Final answer: <integer>" in model_input(graded)
    assert model_input(Prompt(text="Explain X.")) == "Explain X."


def test_prompt_validates_answer_keys() -> None:
    with pytest.raises(ValidationError, match="requires 'expected'"):
        Prompt(text="x", answer_type="integer")
    with pytest.raises(ValidationError, match="answer_type is 'free'"):
        Prompt(text="x", expected="4")
    assert Prompt(text="x", answer_type="boolean", expected="yes").is_graded


# ── statistics ──────────────────────────────────────────────────────


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 10 regressions, 0 improvements: 2 * (1/2)^10
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 10) == mcnemar_exact(10, 0)
    # Known value: b=8, c=2 -> 2 * P(X<=2 | n=10) = 2 * 56/1024
    assert mcnemar_exact(8, 2) == pytest.approx(112 / 1024)


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(50, 100)  # type: ignore[misc]
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(10, 10)  # type: ignore[misc]
    assert hi == pytest.approx(1.0) and 0.6 < lo < 0.75


def test_paired_correctness_table() -> None:
    pairs = [
        ("correct", "correct"),
        ("correct", "wrong"),  # regression
        ("correct", "wrong"),  # regression
        ("wrong", "correct"),  # improvement
        ("wrong", "wrong"),
        ("correct", "unparsed"),  # excluded from 2x2, counts as wrong for B's accuracy
    ]
    t = paired_correctness(pairs)  # type: ignore[arg-type]
    assert t is not None
    assert (t["both_right"], t["regressions"], t["improvements"], t["both_wrong"]) == (1, 2, 1, 1)
    assert t["n_graded"] == 6 and t["n_paired"] == 5 and t["both_truncated"] == 0
    assert t["a"]["scored"] == 6 and t["a"]["correct"] == 4
    assert t["b"]["scored"] == 6 and t["b"]["correct"] == 2 and t["b"]["unparsed"] == 1
    assert t["mcnemar_p"] == pytest.approx(mcnemar_exact(2, 1))
    assert paired_correctness([]) is None


def test_truncating_where_the_other_side_finished_counts_as_wrong() -> None:
    t = paired_correctness(
        [
            ("correct", "truncated"),  # B looped where A answered correctly -> regression
            ("wrong", "truncated"),  # -> both wrong
            ("truncated", "correct"),  # A ran out where B answered -> improvement
            ("unparsed", "truncated"),  # A stopped without an answer: B wrong, pair stays out of 2x2
        ]
    )
    assert t is not None
    assert (t["both_right"], t["regressions"], t["improvements"], t["both_wrong"]) == (0, 1, 1, 1)
    assert t["a"]["truncated"] == 1 and t["b"]["truncated"] == 3
    # Truncations count against accuracy: B answered 1 of 4 correctly.
    assert t["b"]["scored"] == 4 and t["b"]["correct"] == 1
    assert t["a"]["scored"] == 4 and t["a"]["correct"] == 1


def test_prompts_where_both_sides_truncate_are_excluded() -> None:
    t = paired_correctness([("truncated", "truncated"), ("correct", "correct")])
    assert t is not None
    assert t["both_truncated"] == 1 and t["n_graded"] == 2
    assert t["a"]["scored"] == 1 and t["b"]["scored"] == 1
    assert t["a"]["truncated"] == 0 and t["b"]["truncated"] == 0
    assert t["n_paired"] == 1 and t["both_right"] == 1
