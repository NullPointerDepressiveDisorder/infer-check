"""Answer-key scoring: parsing, grading, paired statistics."""

from decimal import Decimal
from fractions import Fraction

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
        ("$184,345,654.33", "184345654.33"),
        ("about 7 apples", "7"),
    ],
)
def test_parse_number(ans: str, value: str) -> None:
    assert parse_number(ans) == Decimal(value)


@pytest.mark.parametrize(
    ("ans", "value"),
    [
        ("10^1000000", "1E+1000000"),
        ("10^369693099", "1E+369693099"),
        ("2 x 10^1000000", "2E+1000000"),
    ],
)
def test_parse_number_handles_huge_exponents(ans: str, value: str) -> None:
    assert parse_number(ans) == Decimal(value)


@pytest.mark.parametrize(
    "ans",
    [
        "10^" + "9" * 5000,  # exponent larger than Python's 4300-digit int-conversion limit
        "10^" + "9" * 25,  # exponent larger than Decimal's widened container
    ],
)
def test_unevaluable_exponent_is_unparsed_not_its_leading_10(ans: str) -> None:
    assert parse_number(ans) is None


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


@pytest.mark.parametrize("answer_type", ["integer", "decimal"])
def test_huge_values_grade_wrong_instead_of_raising(answer_type: str) -> None:
    # Parses fine, then used to overflow in the decimal tolerance subtraction.
    assert is_correct("Final answer: 1e1000000", "2", answer_type) is False  # type: ignore[arg-type]
    assert is_correct("Final answer: 10^369693099", "369693100", answer_type) is False  # type: ignore[arg-type]


def test_unrepresentable_plain_number_is_unparsed() -> None:
    # An exponent this large can't be constructed as a Decimal at all.
    assert is_correct("Final answer: 1e999999999999999999999", "2", "decimal") is None


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


def test_grade_survives_a_huge_answer() -> None:
    p = Prompt(id="p", text="How many digits does 9^(9^9) have?", answer_type="integer", expected="369693100")
    assert grade(_result("Roughly 10^369693099.\nFinal answer: 10^369693099"), p) == "wrong"


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


# ── fractions that can't be evaluated are unparsed ──────────────────


def test_zero_denominator_is_unparsed_not_its_numerator() -> None:
    assert parse_number("1/0") is None
    assert is_correct("Final answer: 1/0", "1", "integer") is None  # was graded correct


@pytest.mark.parametrize("ans", ["9" * 5000 + "/7", "7/" + "9" * 5000])
def test_oversized_fraction_evaluates_without_error(ans: str) -> None:
    # Operands are parsed as Decimals, which have no 4300-digit limit.
    assert parse_number(ans) is not None
    assert is_correct(f"Final answer: {ans}", "2", "decimal") is False


def test_ordinary_fractions_still_parse() -> None:
    assert parse_number("-1/8") == Decimal("-0.125")
    assert parse_number("3/4 of the cake") == Decimal("0.75")
    assert parse_number("6/3") == Decimal(2)  # whole-number fractions are exact
    # Repeating decimals are kept as exact rationals.
    assert parse_number("1627/2520") == Fraction(1627, 2520)


# ── leading-dot decimals ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ans", "value"),
    [(".5", "0.5"), ("-.5", "-0.5"), (".5e2", "50"), (".5 × 10^3", "500"), ("-.25 x 10^-2", "-0.0025")],
)
def test_leading_dot_decimals(ans: str, value: str) -> None:
    assert parse_number(ans) == Decimal(value)


def test_leading_dot_answers_grade_correctly() -> None:
    assert is_correct("Final answer: .5", "0.5", "decimal") is True  # was False (parsed as 5)
    assert is_correct("Final answer: .5", "5", "integer") is False  # was True


# ── boolean keys with surrounding whitespace ────────────────────────


def test_boolean_key_whitespace_is_ignored() -> None:
    assert is_correct("Final answer: yes", " yes ", "boolean") is True
    p = Prompt(text="x", answer_type="boolean", expected=" yes ")
    assert p.expected == "yes"  # stored canonically


# ── fraction forms: LaTeX, signed and comma-grouped operands ────────


@pytest.mark.parametrize(
    ("ans", "value"),
    [
        ("\\frac{1}{2}", "0.5"),
        ("-\\frac{3}{4}", "-0.75"),
        ("\\dfrac{10}{4}", "2.5"),
        ("\\frac{-1}{2}", "-0.5"),
        ("1/-2", "-0.5"),
        ("1,000/2", "500"),
        ("1.5/3", "0.5"),
        ("-1/8", "-0.125"),
    ],
)
def test_fraction_forms(ans: str, value: str) -> None:
    assert parse_number(ans) == Decimal(value)


def test_unevaluable_latex_fraction_is_unparsed_not_its_inner_number() -> None:
    assert parse_number("\\frac{\\pi}{2}") is None


def test_slash_followed_by_a_word_is_not_a_fraction() -> None:
    assert parse_number("$1,000/month") == Decimal(1000)


@pytest.mark.parametrize(
    ("text", "expected", "answer_type", "result"),
    [
        ("\\boxed{\\frac{1}{2}}", "1", "integer", False),  # was a false positive: parsed as 1
        ("\\boxed{\\frac{1}{2}}", "0.5", "decimal", True),  # was a false negative
        ("Final answer: 1/-2", "1", "integer", False),  # was parsed as 1
        ("Final answer: 1/-2", "-0.5", "decimal", True),
        ("Final answer: 1,000/2", "500", "integer", True),  # was parsed as 1000
        ("Final answer: 1,000/2", "1000", "integer", False),
    ],
)
def test_fraction_grading_regressions(text: str, expected: str, answer_type: str, result: bool) -> None:
    assert is_correct(text, expected, answer_type) is result  # type: ignore[arg-type]


# ── decimal tolerance is relative, not absolute ─────────────────────


@pytest.mark.parametrize(
    ("text", "expected", "result"),
    [
        ("Final answer: 0", "1e-9", False),  # was True: a 100% error inside an absolute 1e-6 window
        ("Final answer: 1.0000005e-9", "1e-9", True),  # within 1e-6 relative
        ("Final answer: 1.00001e-9", "1e-9", False),
        ("Final answer: -0.1250001", "-0.125", True),
        ("Final answer: -0.1251", "-0.125", False),
        ("Final answer: 0", "0", True),
        ("Final answer: 0.0", "0", True),
        ("Final answer: 0.000001", "0", False),  # a zero key needs an exact zero
        ("Final answer: 1/3", "0.3333333", True),  # exact rational against a decimal key
        ("Final answer: 1/3", "0.3333", False),
    ],
)
def test_decimal_tolerance_is_relative(text: str, expected: str, result: bool) -> None:
    assert is_correct(text, expected, "decimal") is result


# ── integer grading is exact: no fixed precision to round at ────────

BIG = 10**50 + 1  # 51 digits: one more than the old 50-digit working precision


@pytest.mark.parametrize(
    ("ans", "key", "result"),
    [
        (f"{BIG}/1", str(BIG), True),  # was False: rounded to 10^50
        (f"{BIG}/1", str(10**50), False),  # was True
        (f"{BIG} × 10^0", str(BIG), True),  # scientific notation no longer multiplies and rounds
        (f"{BIG} × 10^0", str(10**50), False),
        (f"{BIG}/{10**50}", "1", False),  # not an integer; was rounded to exactly 1
        (f"{2 * BIG}/2", str(BIG), True),  # exact rational reduces to the integer
        (f"{BIG}e0", str(BIG), True),
    ],
)
def test_integer_grading_is_exact(ans: str, key: str, result: bool) -> None:
    assert is_correct(f"Final answer: {ans}", key, "integer") is result


def test_parsed_values_are_exact() -> None:
    assert parse_number(f"{BIG}/{10**50}") == Fraction(BIG, 10**50)
    assert parse_number(f"{BIG} × 10^2") == Decimal(BIG * 100)
    assert parse_number("1e369693099") == Decimal("1E+369693099")  # still no huge-int materialisation
