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
