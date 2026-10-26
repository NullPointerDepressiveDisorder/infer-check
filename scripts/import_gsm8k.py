"""Import the GSM8K test split as a bundled, answer-keyed prompt suite.

Usage:
    python scripts/import_gsm8k.py             # writes src/infer_check/prompt_suites/gsm8k.jsonl
    python scripts/import_gsm8k.py --out PATH  # write somewhere else

Source: OpenAI's grade-school-math repository (Cobbe et al., 2021, "Training
Verifiers to Solve Math Word Problems"), MIT license. Each problem's reference
solution ends with "#### <answer>"; that number becomes the answer key.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "infer_check" / "prompt_suites" / "gsm8k.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Token budget per problem (room for working).")
    args = parser.parse_args()

    with urllib.request.urlopen(URL, timeout=60) as resp:  # nosec B310 - fixed https URL
        lines = resp.read().decode("utf-8").splitlines()

    out_rows = []
    for i, line in enumerate(ln for ln in lines if ln.strip()):
        row = json.loads(line)
        raw = row["answer"].split("####")[-1].strip().replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            print(f"skipping problem {i}: unparseable answer {raw!r}", file=sys.stderr)
            continue
        is_int = value == value.to_integral_value()
        out_rows.append(
            {
                "id": f"gsm8k-{i + 1:04d}",
                "text": row["question"].strip(),
                "category": "word_problem",
                "max_tokens": args.max_tokens,
                "answer_type": "integer" if is_int else "decimal",
                "expected": str(int(value)) if is_int else str(value),
            }
        )

    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    print(f"wrote {len(out_rows)} problems to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
