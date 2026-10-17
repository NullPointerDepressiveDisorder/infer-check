"""Attach computed answer keys to bundled prompt suites.

Usage:
    python scripts/build_answer_keys.py           # rewrite suites in place
    python scripts/build_answer_keys.py --check   # exit 1 if any stored key differs

Every key is computed here in Python, never typed in by hand. Only prompts
with a single, unambiguous answer are keyed: multi-part questions ("Compute X.
Is it Y?") and convention-dependent ones (0^0, (-3) mod 5, IEEE edge cases)
stay free-form, because "Final answer: <x>" can't capture them. Representation
answers (bit strings like 0b10000000) aren't keyed either: the numeric grader
would read them as decimal numbers.

This script is authoritative: prompts it doesn't list have any answer key
removed, so dropping an entry here un-keys the prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections.abc import Callable
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

SUITES = Path(__file__).resolve().parent.parent / "src" / "infer_check" / "prompt_suites"

# Graded prompts ask for step-by-step working before "Final answer:"; 256 tokens
# truncated most responses (29/43 for Qwen2.5-0.5B bf16), so keyed prompts get
# at least this budget.
MIN_GRADED_MAX_TOKENS = 1024


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def nth_prime(n: int) -> int:
    count, k = 0, 1
    while count < n:
        k += 1
        if all(k % d for d in range(2, math.isqrt(k) + 1)):
            count += 1
    return k


def digits_of_9_pow_9_pow_9() -> int:
    getcontext().prec = 60
    return int((Decimal(9**9) * Decimal(9).log10()).to_integral_value(rounding="ROUND_FLOOR")) + 1


def float32_from_bits(bits: str) -> Decimal:
    (value,) = struct.unpack(">f", int(bits.replace(" ", ""), 2).to_bytes(4, "big"))
    return Decimal(value)  # exact binary value


def frac(f: Fraction) -> str:
    getcontext().prec = 40
    return str(Decimal(f.numerator) / Decimal(f.denominator))


# id -> (answer_type, compute)
Key = tuple[str, Callable[[], object]]
NUMERICS: dict[str, Key] = {
    "numerics-002": ("integer", lambda: 9999999999999999 + 1),
    "numerics-003": ("integer", lambda: 10**308),
    "numerics-005": ("boolean", lambda: "yes" if 2**31 - 1 == 2147483647 else "no"),
    "numerics-010": ("integer", lambda: len(str(math.factorial(170)))),
    "numerics-013": ("decimal", lambda: Decimal("1234567.89") * Decimal("149.32")),
    "numerics-015": ("integer", lambda: 366 * 24 * 60 * 60),
    "numerics-019": ("integer", lambda: 2),  # sum of 1/2^k for k >= 0
    "numerics-023": ("decimal", lambda: Decimal(int("10110011", 2)) + Decimal(int("101", 2)) / 8),
    "numerics-024": ("integer", lambda: int("9" * 50) + 1),
    "numerics-031": ("integer", lambda: 999999999999999999 * 999999999999999999),
    "numerics-032": ("boolean", lambda: "yes" if 2**64 == 18446744073709551616 else "no"),
    "numerics-034": ("integer", lambda: 123456789 * 987654321),
    "numerics-035": ("integer", lambda: fib(50)),  # same under F0=0 and F1=F2=1 conventions
    "numerics-037": ("integer", lambda: (-1) ** 7),
    "numerics-043": ("integer", lambda: sum(range(1, 1001))),
    "numerics-044": ("decimal", lambda: frac(Fraction(-2) ** -3)),
    "numerics-047": ("boolean", lambda: "yes"),  # 2^127 - 1 is the Mersenne prime M127
    "numerics-049": ("integer", lambda: (10**15 + 1) - 10**15),
    "numerics-050": ("integer", lambda: 17**17),
    "numerics-051": ("integer", lambda: sum(100 // 5**k for k in range(1, 4))),
    "numerics-054": ("integer", lambda: 2**10 * 3**5 * 7**2),
    "numerics-055": ("integer", lambda: pow(10, 100, 7)),
    "numerics-057": ("integer", digits_of_9_pow_9_pow_9),
    "numerics-058": ("integer", lambda: 0xDEADBEEF),
    "numerics-060": ("integer", lambda: 0),  # rows are linearly dependent
    "numerics-061": ("integer", lambda: math.gcd(123456789, 987654321)),
    "numerics-062": ("decimal", lambda: Decimal("3.14159265") * Decimal("2.71828182")),
    "numerics-064": ("integer", lambda: 1111111 * 1111111),
    "numerics-067": ("integer", lambda: math.comb(15, 5)),
    "numerics-068": ("integer", lambda: nth_prime(100)),
    "numerics-069": ("integer", lambda: (255 << 24) | (255 << 16) | (255 << 8) | 0),
    "numerics-076": ("integer", lambda: 999**3),
    "numerics-083": ("integer", lambda: 10**7 * 10**8),
    "numerics-084": ("integer", lambda: 12345679 * 63),
    "numerics-085": ("integer", lambda: pow(2, 100, 10)),
    "numerics-089": ("decimal", lambda: frac(sum(Fraction((-1) ** (k + 1), k) for k in range(1, 11)))),
    "numerics-092": ("decimal", lambda: (2 - Fraction(1, 2**23)) * 2**127),
    "numerics-093": ("integer", lambda: 3**33),
    "numerics-094": ("integer", lambda: sum(2 * k - 1 for k in range(1, 101))),
    "numerics-097": ("integer", lambda: math.isqrt(3**2 + 4**2)),
    "numerics-098": ("decimal", lambda: float32_from_bits("0 10000000 10010010000111111011011")),
    "numerics-099": ("integer", lambda: 99999**2),
}

KEYS: dict[str, dict[str, Key]] = {"adversarial-numerics.jsonl": NUMERICS}


def render(value: object) -> str:
    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else frac(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify stored keys instead of writing.")
    args = parser.parse_args()

    mismatches = 0
    for filename, keys in KEYS.items():
        path = SUITES / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = {r["id"] for r in rows}
        unknown = set(keys) - ids
        if unknown:
            raise SystemExit(f"{filename}: keys for unknown prompt ids {sorted(unknown)}")

        for row in rows:
            if row["id"] not in keys:
                if "answer_type" in row or "expected" in row:  # stale key from an older list
                    mismatches += 1
                    if args.check:
                        print(f"{filename} {row['id']}: has an answer key but isn't listed here")
                    row.pop("answer_type", None)
                    row.pop("expected", None)
                continue
            answer_type, compute = keys[row["id"]]
            expected = render(compute())
            max_tokens = max(int(row.get("max_tokens", 0)), MIN_GRADED_MAX_TOKENS)
            if (
                row.get("answer_type") != answer_type
                or row.get("expected") != expected
                or row.get("max_tokens") != max_tokens
            ):
                mismatches += 1
                if args.check:
                    print(f"{filename} {row['id']}: stored {row.get('expected')!r}, computed {expected!r}")
            row["answer_type"] = answer_type
            row["expected"] = expected
            row["max_tokens"] = max_tokens

        if not args.check:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            print(f"{filename:32s} {len(keys):3d} prompts keyed of {len(rows)}")

    return 1 if args.check and mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
