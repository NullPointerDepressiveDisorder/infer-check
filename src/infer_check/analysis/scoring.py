"""Correctness scoring against answer keys.

Prompts with ``answer_type != "free"`` carry an ``expected`` answer. The model
is asked to end with ``Final answer: <answer>``; that line (or, failing that,
the last ``\\boxed{...}``) is parsed according to ``answer_type`` and compared
with the key.

Comparisons between two configurations are reported as a paired 2x2 table
(both right / regressions / improvements / both wrong) with an exact McNemar
test on the discordant pairs. Responses with no parseable answer are counted
separately and never as a flip. A response cut off by the token limit counts as
wrong when the other configuration finished the same prompt; a prompt where
both are cut off is excluded (the budget, not the model, is the likely cause).
"""

from __future__ import annotations

import contextlib
import decimal
import math
import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import comb
from typing import Any, Literal

from infer_check.types import AnswerType, InferenceResult, Prompt

__all__ = [
    "Grade",
    "answer_instruction",
    "model_input",
    "final_answer",
    "parse_number",
    "is_correct",
    "grade",
    "mcnemar_exact",
    "wilson_interval",
    "paired_correctness",
]

Grade = Literal["correct", "wrong", "unparsed", "truncated"]

# ── Prompting ───────────────────────────────────────────────────────

# Ask for the working *before* the answer. Without "step by step", small models
# tend to jump straight to "Final answer: <guess>" and accuracy collapses
# (Qwen2.5-0.5B on GSM8K: 17% with a bare format instruction).
_FORMAT = "Work through the problem step by step, then end your response with a final line of the form: Final answer: "
_INSTRUCTIONS: dict[str, str] = {
    "integer": _FORMAT + "<integer>",
    "decimal": _FORMAT + "<number>",
    "boolean": _FORMAT + "<yes or no>",
    "choice": _FORMAT + "<option letter>",
}


def answer_instruction(answer_type: AnswerType) -> str | None:
    """The fixed instruction appended to graded prompts, or None for free ones."""
    return _INSTRUCTIONS.get(answer_type)


def model_input(prompt: Prompt) -> str:
    """The text actually sent to the model: the prompt plus the answer-format instruction."""
    instruction = answer_instruction(prompt.answer_type)
    return f"{prompt.text}\n\n{instruction}" if instruction else prompt.text


# ── Parsing ─────────────────────────────────────────────────────────

# "Final answer: 42", "**Final answer:** 42", "Final Answer**: 42", fullwidth colon.
_FINAL = re.compile(r"final\s+answer\s*\**\s*[:：]\s*\**\s*(.+)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

# Mantissas may omit the leading zero (".5", "-.5"); otherwise ".5" would parse as 5.
_SCI = re.compile(
    r"(-?(?:\d+(?:\.\d+)?|\.\d+))"  # mantissa
    r"\s*(?:[x×*·]|\\times|\\cdot)\s*10\s*(?:\^|\*\*)\s*\{?\s*([-+]?\d+)\s*\}?"  # × 10^exponent
)
_POW10 = re.compile(r"(?<![\d.])(-?)10\s*(?:\^|\*\*)\s*\{?\s*([-+]?\d+)\s*\}?")
_OPERAND = r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
# a/b with signed, comma-grouped or decimal operands ("1,000/2", "1/-2", "1.5/3"). The
# lookarounds stop it starting mid-number ("1,000/2" must not match "000/2").
_FRAC = re.compile(rf"(?<![\w.,])({_OPERAND})\s*/\s*({_OPERAND})(?!\w|\.\d)")
_LATEX_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_NUMBER_ONLY = re.compile(rf"\s*{_OPERAND}\s*")
_NUM = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_BOOL = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_CHOICE = re.compile(r"\(?\b([A-Za-z0-9])\b\)?")


def final_answer(text: str) -> str | None:
    """The answer the model committed to: last ``Final answer:`` line, else last ``\\boxed{}``."""
    hits = _FINAL.findall(text)
    if hits:
        ans = hits[-1].strip().strip("*`$ ").rstrip(".").strip()
        return ans or None
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip() or None
    return None


Number = Decimal | Fraction
# Relative tolerance for ``decimal`` answers.
DECIMAL_REL_TOL = Decimal("1e-6")

# Default contexts overflow past 1e999999; model output like "10^369693099" is legal.
_Wide = decimal.Context(prec=50, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)


def _normalize_latex_fractions(s: str) -> tuple[str, list[int]]:
    """Rewrite numeric ``\\frac{a}{b}`` as ``a/b``.

    Non-numeric fractions (``\\frac{\\pi}{2}``) can't be evaluated: they become a
    marker no pattern matches, and their position is returned so the number
    inside them ("2") is never taken as the answer.
    """
    out: list[str] = []
    failed: list[int] = []
    last = 0
    for m in _LATEX_FRAC.finditer(s):
        out.append(s[last : m.start()])
        num, den = m.group(1).strip(), m.group(2).strip()
        if _NUMBER_ONLY.fullmatch(num) and _NUMBER_ONLY.fullmatch(den):
            out.append(f"{num}/{den}")
        else:
            failed.append(sum(len(part) for part in out))
            out.append("\x00")
        last = m.end()
    out.append(s[last:])
    return "".join(out), failed


def parse_number(ans: str) -> Number | None:
    """Parse the first number in ``ans``: plain, comma-grouped, ``1.5e9``,
    ``1.5 × 10^9``, ``10^9``, or a fraction ``a/b``. None if there isn't one.

    Values are exact: decimals are built straight from their digits (a
    ``Decimal`` from a string is never rounded, whatever its exponent), and
    fractions are exact ``Fraction``s. No fixed precision means no boundary
    past which grading silently rounds: ``(10^50 + 1)/1`` stays 10^50 + 1.
    """
    s = ans.replace("\u2212", "-").replace("\u00a0", " ").replace("$", "")
    s, failed = _normalize_latex_fractions(s)
    candidates: list[tuple[int, int, Number]] = []  # (start, -length, value)

    with decimal.localcontext(_Wide):
        for m in _SCI.finditer(s):
            try:
                # "m × 10^e" is exactly the decimal string "me": no multiplication, no rounding.
                candidates.append((m.start(), -len(m.group(0)), Decimal(f"{m.group(1)}e{int(m.group(2))}")))
            except (ArithmeticError, ValueError):  # exponent beyond Decimal's range, or >4300 digits
                failed.append(m.start())
        for m in _POW10.finditer(s):
            try:
                candidates.append((m.start(), -len(m.group(0)), Decimal(f"{m.group(1)}1e{int(m.group(2))}")))
            except (ArithmeticError, ValueError):
                failed.append(m.start())
        for m in _FRAC.finditer(s):
            try:
                num = Fraction(Decimal(m.group(1).replace(",", "")))
                den = Fraction(Decimal(m.group(2).replace(",", "")))
                candidates.append((m.start(), -len(m.group(0)), num / den))  # exact; x/0 raises
            except (ArithmeticError, ValueError):
                failed.append(m.start())
    plain = _NUM.search(s)
    if plain:
        with contextlib.suppress(InvalidOperation):
            candidates.append((plain.start(), -len(plain.group(0)), Decimal(plain.group(0).replace(",", ""))))

    if not candidates:
        return None
    best = min(candidates, key=lambda c: (c[0], c[1]))
    # Never fall back to the digits inside a form we failed to evaluate
    # (the "10" of "10^<huge>", the "1" of "1/0", the "2" of "\\frac{\\pi}{2}").
    if any(start <= best[0] for start in failed):
        return None
    return best[2]


def _within_relative_tolerance(got: Number, exp: Decimal) -> bool:
    """``|got - exp| <= 1e-6 * |exp|``. A key of 0 requires exactly 0.

    Purely relative: an absolute floor would let ``0`` pass for a key of
    ``1e-9``, a 100% error.
    """
    if exp == 0:
        return got == 0
    if isinstance(got, Fraction):
        key = Fraction(exp)
        return abs(got - key) <= Fraction(DECIMAL_REL_TOL) * abs(key)
    with decimal.localcontext(_Wide):
        try:
            return abs(got - exp) <= DECIMAL_REL_TOL * abs(exp)
        except ArithmeticError:
            return False  # a value beyond even the wide range can't be within tolerance of a key


def is_correct(text: str, expected: str, answer_type: AnswerType) -> bool | None:
    """True/False, or None when no answer can be parsed.

    None is reported separately and is never counted as a flip.
    """
    ans = final_answer(text)
    if ans is None:
        return None
    if answer_type in ("integer", "decimal"):
        got = parse_number(ans)
        if got is None:
            return None
        try:
            exp = Decimal(expected)
        except InvalidOperation as exc:
            raise ValueError(f"answer key {expected!r} is not a number") from exc
        if answer_type == "integer":
            return got == exp  # exact: Decimal and Fraction compare without rounding
        return _within_relative_tolerance(got, exp)
    if answer_type == "boolean":
        m = _BOOL.search(ans)
        if not m:
            return None
        norm = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
        key = expected.strip().lower()
        return norm[m.group(1).lower()] == norm.get(key, key)
    if answer_type == "choice":
        m = _CHOICE.search(ans)
        if not m:
            return None
        return m.group(1).upper() == expected.strip().upper()
    return None


def grade(result: InferenceResult, prompt: Prompt) -> Grade | None:
    """Grade one response. None for prompts without an answer key."""
    if not prompt.is_graded or prompt.expected is None:
        return None
    ok = is_correct(result.text, prompt.expected, prompt.answer_type)
    if ok is None:
        # No answer because generation hit the token limit isn't the model's mistake.
        return "truncated" if result.metadata.get("finish_reason") == "length" else "unparsed"
    return "correct" if ok else "wrong"


# ── Statistics ──────────────────────────────────────────────────────


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs.

    b = A right & B wrong (regressions), c = A wrong & B right (improvements).
    Under H0 (no difference), each discordant pair is a fair coin flip.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p: float = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))
    return p


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion (well-behaved near 0 and 1)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_correctness(pairs: Iterable[tuple[Grade, Grade]]) -> dict[str, Any] | None:
    """Paired 2x2 correctness table for configuration A vs B.

    Truncation rule: hitting the token limit on a prompt where the *other*
    configuration finished counts as wrong. Running out of budget there is the
    configuration's own failure (typically a repetition loop), not a budget
    problem with the prompt. Only when *both* sides truncate is the prompt
    excluded, since then the budget itself is the likely cause.

    Unparseable answers (the model stopped without a ``Final answer:``) count as
    wrong for accuracy but stay out of the 2x2 table and McNemar test, so a
    formatting slip is never reported as a flip.

    Returns None if there are no graded pairs.
    """
    raw = list(pairs)
    if not raw:
        return None

    both_truncated = sum(1 for a, b in raw if a == "truncated" and b == "truncated")
    kept = [(a, b) for a, b in raw if not (a == "truncated" and b == "truncated")]

    def effective(own: Grade, other: Grade) -> Grade:
        return "wrong" if own == "truncated" and other != "truncated" else own

    eff = [(effective(a, b), effective(b, a)) for a, b in kept]

    def side(idx: int) -> dict[str, Any]:
        grades = [p[idx] for p in eff]
        correct = grades.count("correct")
        scored = len(grades)
        ci = wilson_interval(correct, scored)
        return {
            "correct": correct,
            "scored": scored,
            "accuracy": correct / scored if scored else None,
            "accuracy_ci95": list(ci) if ci else None,
            "unparsed": grades.count("unparsed"),
            # Truncated where the other side finished; already counted as wrong above.
            "truncated": sum(1 for p in kept if p[idx] == "truncated"),
        }

    both_right = regressions = improvements = both_wrong = 0
    for a, b in eff:
        if a not in ("correct", "wrong") or b not in ("correct", "wrong"):
            continue
        if a == "correct" and b == "correct":
            both_right += 1
        elif a == "correct":
            regressions += 1
        elif b == "correct":
            improvements += 1
        else:
            both_wrong += 1

    return {
        "n_graded": len(raw),
        "n_paired": both_right + regressions + improvements + both_wrong,
        "both_truncated": both_truncated,
        "a": side(0),
        "b": side(1),
        "both_right": both_right,
        "regressions": regressions,
        "improvements": improvements,
        "both_wrong": both_wrong,
        "mcnemar_p": mcnemar_exact(regressions, improvements),
    }
