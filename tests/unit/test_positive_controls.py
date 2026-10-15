"""scripts/positive_controls.py: the checks that decide PASS/FAIL."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "positive_controls.py"


@pytest.fixture(scope="module")
def pc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("positive_controls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(regressions: int, improvements: int, p: float) -> dict[str, Any]:
    return {"regressions": regressions, "improvements": improvements, "mcnemar_p": p}


def test_significant_loss_at_max_damage_passes(pc: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert pc.check_detects_damage("2bit", _table(23, 0, 2.4e-7)) is True
    assert "PASS" in capsys.readouterr().out


@pytest.mark.parametrize(
    "table",
    [_table(5, 1, 0.22), _table(1, 9, 0.02), _table(4, 4, 1.0)],
    ids=["not-significant", "significant-improvement", "tie"],
)
def test_undetected_or_inverted_damage_fails(
    pc: ModuleType, capsys: pytest.CaptureFixture[str], table: dict[str, Any]
) -> None:
    assert pc.check_detects_damage("2bit", table) is False
    assert "FAIL" in capsys.readouterr().out


def test_no_answer_keys_is_a_skip_not_a_failure(pc: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert pc.check_detects_damage("2bit", None) is True
    assert "SKIP" in capsys.readouterr().out


def test_check_monotonic(pc: ModuleType) -> None:
    # Real Phase 4 quantization KL values pass; an inversion or a missing value fails.
    assert pc.check_monotonic("KL", [0.0, 0.0005, 0.0789, 0.6763, 10.3418]) is True
    assert pc.check_monotonic("KL", [0.0, 0.0789, 0.0005]) is False
    assert pc.check_monotonic("KL", [0.0, None, 0.5]) is False


def test_disagreement(pc: ModuleType) -> None:
    assert pc.disagreement({"mean_top1_agreement": 0.93}) == pytest.approx(0.07)
    assert pc.disagreement({"mean_top1_agreement": None}) is None
