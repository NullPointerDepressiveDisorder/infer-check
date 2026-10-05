import re
from decimal import Decimal, InvalidOperation
from math import comb

_FINAL = re.compile(r"final answer\s*[:: ]\s*\**\s*(.+)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def final_answer(text: str) -> str | None:
    hits = _FINAL.findall(text)
    return hits[-1].strip().strip("*.` ") if hits else None


def is_correct(text: str, expected: str, answer_type: str) -> bool | None:
    """True/False, or None when no answer can be parsed.
    None is reported seperately and is never counted as a flip."""
    ans = final_answer(text)
    if ans is None:
        return None
    if answer_type in ("integer", "decimal"):
        m = _NUM.search(ans)
        if not m:
            return None
        try:
            got, exp = Decimal(m.group(0).replace(",", "")), Decimal(expected)
        except InvalidOperation:
            return None
        if answer_type == "integer":
            return got == exp
        return abs(got - exp) < Decimal("1e-6") * max(Decimal(1), abs(exp))
    if answer_type == "boolean":
        norm = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
        return norm.get(ans.split()[0].lower().strip(".,")) == expected.lower()
    if answer_type == "choice":
        return ans.split()[0].strip("().").upper() == expected.upper()
    return None


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs.
    b = A right & B wrong (regressions), c = A wrong & B right (improvements)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p: float = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))
    return p
