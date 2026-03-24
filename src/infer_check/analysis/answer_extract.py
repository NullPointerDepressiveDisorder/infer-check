"""Answer extraction and flip-rate computation.

Extracts the "functional answer" from LLM outputs so that two responses
can be compared semantically rather than by raw text similarity.  A
*flip* occurs when two models give substantively different answers to the
same prompt — not merely different wording.

Extraction strategies are selected by prompt category:

  - **numeric** (arithmetic, precision, large_numbers, floating_point,
    underflow, formatting, word_problem, multi_digit_arithmetic,
    precision_numerics, large_number_reasoning, algebraic_reasoning,
    logical_puzzle): extract the last number/expression.
  - **boolean** (logic, edge_case): extract yes / no / true / false.
  - **code** (python, debugging, completion, precise_syntax,
    code_translation): extract fenced code blocks
    and compare after whitespace normalisation.
  - **json** (json): parse and compare structurally.
  - **fallback**: character-level similarity via ``difflib``.
"""

from __future__ import annotations

import difflib
import json as json_mod
import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ExtractedAnswer",
    "extract_answer",
    "answers_match",
    "compute_flip_rate",
]


# ── Categories → extraction strategy ────────────────────────────────

_NUMERIC_CATEGORIES = frozenset(
    {
        "arithmetic",
        "precision",
        "large_numbers",
        "floating_point",
        "underflow",
        "formatting",
        "word_problem",
        "multi_digit_arithmetic",
        "precision_numerics",
        "large_number_reasoning",
        "algebraic_reasoning",
        "logical_puzzle",
    }
)

_BOOLEAN_CATEGORIES = frozenset(
    {
        "logic",
        "edge_case",
    }
)

_CODE_CATEGORIES = frozenset(
    {
        "python",
        "debugging",
        "completion",
        "precise_syntax",
        "code_translation",
    }
)

_JSON_CATEGORIES = frozenset(
    {
        "json",
    }
)


# ── Regex patterns ──────────────────────────────────────────────────

# Matches integers, decimals, scientific notation, and comma-separated
# numbers like 1,234,567.89.  Also matches negative numbers.
_NUMBER_RE = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?",
)

# Fenced code block: ```...``` (with optional language tag).
_CODE_BLOCK_RE = re.compile(
    r"```(?:\w+)?\s*\n(.*?)```",
    re.DOTALL,
)

# Boolean-ish final answer patterns.
_BOOLEAN_RE = re.compile(
    r"\b(yes|no|true|false|correct|incorrect|definitely|not necessarily)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    """The extracted functional answer from an LLM response."""

    strategy: str  # "numeric", "boolean", "code", "json", "raw"
    value: str  # normalised answer string for comparison
    raw_text: str  # original full response text
    confidence: float  # 0–1, how confident we are in extraction


# ── Extraction helpers ──────────────────────────────────────────────


def _extract_numeric(text: str) -> ExtractedAnswer:
    """Pull the last number from a math/numeric response."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return ExtractedAnswer(
            strategy="numeric",
            value="",
            raw_text=text,
            confidence=0.0,
        )
    # Take the last number — models tend to state the final answer last.
    last = matches[-1].replace(",", "")
    return ExtractedAnswer(
        strategy="numeric",
        value=last,
        raw_text=text,
        confidence=0.9,
    )


def _extract_boolean(text: str) -> ExtractedAnswer:
    """Pull the boolean conclusion from a logic response.

    Scans for yes/no/true/false keywords.  When multiple are found,
    the *last* one wins (models often hedge before concluding).
    Negation context is checked: "not correct" → no, "not true" → no.
    """
    matches: list[tuple[int, str]] = [(m.start(), m.group(0).lower()) for m in _BOOLEAN_RE.finditer(text)]
    if not matches:
        return ExtractedAnswer(
            strategy="boolean",
            value="",
            raw_text=text,
            confidence=0.0,
        )

    _pos = {"yes", "correct", "definitely", "true"}
    _neg = {"no", "incorrect", "not necessarily", "false"}

    pos, raw = matches[-1]
    text_lower = text.lower()

    # Check for "not" within 5 chars before a positive keyword.
    negated = False
    if raw in _pos:
        prefix = text_lower[max(0, pos - 5) : pos]
        if "not" in prefix.split():
            negated = True

    if raw in _neg or negated:
        normalised = "no"
    elif raw in _pos:
        normalised = "yes"
    else:
        normalised = raw

    return ExtractedAnswer(
        strategy="boolean",
        value=normalised,
        raw_text=text,
        confidence=0.85,
    )


def _extract_code(text: str) -> ExtractedAnswer:
    """Extract fenced code blocks and normalise whitespace."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        # No fenced block — treat entire response as code (common for
        # models that don't fence).  Strip leading prose heuristically.
        lines = text.strip().splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            # Heuristic: code lines start with def, class, import,
            # return, if, for, while, #, or are indented.
            if (
                in_code
                or stripped.startswith(
                    (
                        "def ",
                        "class ",
                        "import ",
                        "from ",
                        "return ",
                        "if ",
                        "for ",
                        "while ",
                        "#",
                    )
                )
                or line.startswith(("    ", "\t"))
                or stripped == ""
            ):
                code_lines.append(line)
                in_code = True
        code = "\n".join(code_lines).strip()
        confidence = 0.5 if code else 0.0
    else:
        code = "\n\n".join(b.strip() for b in blocks)
        confidence = 0.9

    # Normalise: collapse whitespace runs, strip trailing ws per line.
    normalised = "\n".join(line.rstrip() for line in code.splitlines()).strip()
    return ExtractedAnswer(
        strategy="code",
        value=normalised,
        raw_text=text,
        confidence=confidence,
    )


def _extract_json(text: str) -> ExtractedAnswer:
    """Extract and canonicalise JSON from the response."""
    # Try to find a JSON block in fences first.
    blocks = _CODE_BLOCK_RE.findall(text)
    candidates = blocks if blocks else [text]

    for candidate in candidates:
        candidate = candidate.strip()
        # Find the first { or [ and try to parse from there.
        for start_char in ("{", "["):
            idx = candidate.find(start_char)
            if idx == -1:
                continue
            try:
                parsed = json_mod.loads(candidate[idx:])
                canonical = json_mod.dumps(
                    parsed,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return ExtractedAnswer(
                    strategy="json",
                    value=canonical,
                    raw_text=text,
                    confidence=0.95,
                )
            except json_mod.JSONDecodeError:
                continue

    return ExtractedAnswer(
        strategy="json",
        value="",
        raw_text=text,
        confidence=0.0,
    )


def _extract_raw(text: str) -> ExtractedAnswer:
    """Fallback: use the full text, lightly normalised."""
    normalised = " ".join(text.lower().split())
    return ExtractedAnswer(
        strategy="raw",
        value=normalised,
        raw_text=text,
        confidence=0.3,
    )


# ── Public API ──────────────────────────────────────────────────────


def extract_answer(text: str, category: str = "general") -> ExtractedAnswer:
    """Extract the functional answer from an LLM response.

    Selects an extraction strategy based on the prompt category.

    Args:
        text: The full LLM response text.
        category: The prompt category (from ``Prompt.category``).

    Returns:
        An ``ExtractedAnswer`` with the normalised value and metadata.
    """
    cat = category.lower()
    if cat in _NUMERIC_CATEGORIES:
        return _extract_numeric(text)
    if cat in _BOOLEAN_CATEGORIES:
        return _extract_boolean(text)
    if cat in _CODE_CATEGORIES:
        return _extract_code(text)
    if cat in _JSON_CATEGORIES:
        return _extract_json(text)
    return _extract_raw(text)


def answers_match(
    a: ExtractedAnswer,
    b: ExtractedAnswer,
    *,
    similarity_threshold: float = 0.85,
) -> bool:
    """Determine whether two extracted answers are functionally equivalent.

    For numeric, boolean, and json strategies the comparison is exact
    (after normalisation).  For code and raw strategies, a similarity
    threshold is used.

    Args:
        a: First extracted answer.
        b: Second extracted answer.
        similarity_threshold: Minimum ``SequenceMatcher`` ratio for
            code/raw comparisons to be considered a match.

    Returns:
        ``True`` if the answers are functionally equivalent.
    """
    # If both extraction failed, they match as non-answers.
    if not a.value and not b.value:
        return True

    # If only one extraction failed, fall back to raw similarity.
    if not a.value or not b.value:
        ratio = difflib.SequenceMatcher(
            None,
            a.raw_text or "",
            b.raw_text or "",
        ).ratio()
        return ratio >= similarity_threshold

    strategy = a.strategy  # both should share strategy if same prompt

    if strategy in ("numeric", "boolean", "json"):
        return a.value == b.value

    # Code and raw: use sequence similarity.
    ratio = difflib.SequenceMatcher(None, a.value, b.value).ratio()
    return ratio >= similarity_threshold


@dataclass(frozen=True, slots=True)
class FlipDetail:
    """Per-prompt flip analysis result."""

    prompt_id: str
    category: str
    flipped: bool
    answer_a: ExtractedAnswer
    answer_b: ExtractedAnswer


def compute_flip_rate(
    pairs: Sequence[tuple[str, str, str, str]],
    *,
    similarity_threshold: float = 0.85,
) -> tuple[float, list[FlipDetail]]:
    """Compute the flip rate across a set of response pairs.

    Args:
        pairs: Sequence of ``(prompt_id, category, text_a, text_b)``
            tuples.
        similarity_threshold: Passed through to ``answers_match`` for
            code/raw comparisons.

    Returns:
        A tuple of ``(flip_rate, details)`` where ``flip_rate`` is in
        [0, 1] and ``details`` is a list of per-prompt ``FlipDetail``
        objects.
    """
    if not pairs:
        return 0.0, []

    details: list[FlipDetail] = []
    flip_count = 0

    for prompt_id, category, text_a, text_b in pairs:
        ans_a = extract_answer(text_a, category)
        ans_b = extract_answer(text_b, category)
        flipped = not answers_match(
            ans_a,
            ans_b,
            similarity_threshold=similarity_threshold,
        )
        if flipped:
            flip_count += 1
        details.append(
            FlipDetail(
                prompt_id=prompt_id,
                category=category,
                flipped=flipped,
                answer_a=ans_a,
                answer_b=ans_b,
            )
        )

    return flip_count / len(pairs), details
