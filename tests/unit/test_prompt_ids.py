"""Stable prompt IDs: bundled suites declare them, the loader derives them otherwise."""

import json
from pathlib import Path

import pytest

from infer_check.prompt_suites import get_suite_path, list_suites
from infer_check.suites.loader import load_suite


@pytest.mark.parametrize("suite", list_suites())
def test_bundled_suites_have_explicit_unique_ids(suite: str) -> None:
    path = get_suite_path(suite)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [i for i, r in enumerate(rows, start=1) if "id" not in r]
    assert not missing, f"{suite}: lines without an explicit id: {missing[:10]}"
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{suite}: duplicate ids"


def test_ids_are_identical_across_loads() -> None:
    path = get_suite_path("reasoning")
    first = [p.id for p in load_suite(path)]
    second = [p.id for p in load_suite(path)]
    assert first == second
    assert first[0] == "reasoning-001"


def test_missing_ids_are_derived_deterministically(tmp_path: Path) -> None:
    suite = tmp_path / "custom.jsonl"
    suite.write_text('{"text": "alpha"}\n{"text": "beta"}\n', encoding="utf-8")
    a = [p.id for p in load_suite(suite)]
    b = [p.id for p in load_suite(suite)]
    assert a == b
    assert all(i.startswith("custom-") for i in a)
    assert a[0] != a[1]


def test_derived_ids_survive_reordering(tmp_path: Path) -> None:
    one = tmp_path / "custom.jsonl"
    one.write_text('{"text": "alpha"}\n{"text": "beta"}\n', encoding="utf-8")
    ids_before = {p.text: p.id for p in load_suite(one)}
    one.write_text('{"text": "beta"}\n{"text": "alpha"}\n', encoding="utf-8")
    ids_after = {p.text: p.id for p in load_suite(one)}
    assert ids_before == ids_after


def test_repeated_text_without_ids_gets_suffix(tmp_path: Path) -> None:
    suite = tmp_path / "rep.jsonl"
    suite.write_text('{"text": "same"}\n{"text": "same"}\n{"text": "same"}\n', encoding="utf-8")
    ids = [p.id for p in load_suite(suite)]
    assert len(set(ids)) == 3
    assert ids[1] == f"{ids[0]}-2" and ids[2] == f"{ids[0]}-3"


def test_duplicate_explicit_ids_are_rejected(tmp_path: Path) -> None:
    suite = tmp_path / "dup.jsonl"
    suite.write_text('{"id": "x", "text": "a"}\n{"id": "x", "text": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate prompt id 'x'"):
        load_suite(suite)


@pytest.mark.parametrize("suite", list_suites())
def test_bundled_suites_load_and_answer_keys_validate(suite: str) -> None:
    # Prompt's validator rejects keyed prompts without `expected` (and vice versa).
    prompts = load_suite(get_suite_path(suite))
    assert prompts


def test_numerics_answer_keys_match_their_computation() -> None:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_answer_keys.py"
    proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_gsm8k_is_fully_keyed() -> None:
    prompts = load_suite(get_suite_path("gsm8k"))
    assert len(prompts) == 1319
    assert all(p.is_graded and p.answer_type == "integer" for p in prompts)


def test_keyed_numerics_prompts_ask_a_single_question() -> None:
    """A keyed prompt must ask only for the graded quantity.

    "Compute X. How many digits does it have?" invites a final answer of X, which
    the grader then compares against the digit count.
    """
    import re

    compound = re.compile(r"^(Compute|What is)\b[^?]*[.?]\s+\S.*\?$")
    for p in load_suite(get_suite_path("adversarial-numerics")):
        if p.is_graded:
            assert p.text.count("?") <= 1 and not compound.match(p.text), f"{p.id}: {p.text!r}"


def test_bit_string_prompt_is_not_keyed() -> None:
    # "0b10000000" would be read as the decimal number 0 by the numeric grader.
    prompts = {p.id: p for p in load_suite(get_suite_path("adversarial-numerics"))}
    assert not prompts["numerics-075"].is_graded
