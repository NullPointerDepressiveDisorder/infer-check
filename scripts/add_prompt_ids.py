"""Assign stable IDs to bundled prompt suites.

Usage:
    python scripts/add_prompt_ids.py            # rewrite suites in place
    python scripts/add_prompt_ids.py --check    # exit 1 if any prompt lacks an ID

Existing IDs are never changed, so the script is safe to re-run after
appending prompts: only new lines get IDs, numbered after the highest
existing one. Everything except the inserted "id" key is written back
byte-for-byte (key order, spacing, and non-ASCII characters are preserved).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUITES_DIR = Path(__file__).resolve().parent.parent / "src" / "infer_check" / "prompt_suites"

# Short, readable prefixes per suite file.
PREFIXES = {
    "adversarial-numerics": "numerics",
    "code": "code",
    "determinism": "determinism",
    "long-context": "longctx",
    "quant-sensitive": "quant",
    "reasoning": "reasoning",
}


def _dump(obj: dict) -> str:
    # Matches the existing file formatting: default separators, raw unicode.
    return json.dumps(obj, ensure_ascii=False)


def process(path: Path, check_only: bool) -> tuple[int, int]:
    prefix = PREFIXES.get(path.stem, path.stem)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [json.loads(line) for line in lines]

    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    existing = [int(m.group(1)) for r in rows if "id" in r and (m := pattern.match(str(r["id"])))]
    next_n = max(existing, default=0) + 1

    missing = sum(1 for r in rows if "id" not in r)
    if check_only or missing == 0:
        return len(rows), missing

    out: list[str] = []
    for original, row in zip(lines, rows, strict=True):
        if "id" in row:
            out.append(original)
            continue
        new_row = {"id": f"{prefix}-{next_n:03d}", **row}
        next_n += 1
        # Safety: removing the id must reproduce the original line exactly.
        rest = {k: v for k, v in new_row.items() if k != "id"}
        if _dump(rest) != original:
            raise SystemExit(f"{path.name}: line would not round-trip cleanly:\n{original}")
        out.append(_dump(new_row))

    ids = [json.loads(line)["id"] for line in out]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path.name}: duplicate IDs after assignment")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(rows), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Only report prompts without IDs.")
    args = parser.parse_args()

    total_missing = 0
    for path in sorted(SUITES_DIR.glob("*.jsonl")):
        n, missing = process(path, args.check)
        total_missing += missing
        verb = "missing" if args.check else "assigned"
        print(f"{path.name:32s} {n:4d} prompts, {missing:4d} {verb}")

    if args.check and total_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
