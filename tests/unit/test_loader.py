import json
from pathlib import Path

import pytest

from infer_check.suites.loader import load_suite
from infer_check.types import Prompt


def test_load_suite_actual_reasoning() -> None:
    # Attempt to load the actual suites
    suite_path = Path("src/infer_check/prompt_suites/reasoning.jsonl")
    if not suite_path.exists():
        pytest.skip(f"Test suite not found at {suite_path}")

    prompts = load_suite(suite_path)
    assert len(prompts) > 0
    assert isinstance(prompts[0], Prompt)
    assert prompts[0].id is not None
    assert prompts[0].text is not None


def test_load_suite_malformed(tmp_path: Path) -> None:
    malformed_file = tmp_path / "bad.jsonl"
    with open(malformed_file, "w") as f:
        f.write("not json\\n")

    with pytest.raises(ValueError):
        load_suite(malformed_file)


def test_load_suite_missing_fields(tmp_path: Path) -> None:
    bad_fields = tmp_path / "fields.jsonl"
    with open(bad_fields, "w") as f:
        f.write(json.dumps({"id": "only_id"}) + "\\n")

    with pytest.raises(ValueError):
        load_suite(bad_fields)
