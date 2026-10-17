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
    return {
        "regressions": regressions,
        "improvements": improvements,
        "mcnemar_p": p,
        "n_paired": regressions + improvements + 10,
    }


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


def test_all_truncated_table_renders_and_fails_clearly(pc: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    from infer_check.analysis.scoring import paired_correctness

    table = paired_correctness([("truncated", "truncated")] * 3)
    row = pc.fmt_row("2bit", {"mean_kl": 1.0, "mean_top1_agreement": 0.5}, table)  # used to raise TypeError
    assert "N/A -> N/A" in row
    assert pc.check_detects_damage("2bit", table) is False
    assert "no gradable pairs" in capsys.readouterr().out


def test_quant_cache_dir_includes_the_repo_owner(pc: ModuleType, tmp_path: Path) -> None:
    a = pc.quant_cache_dir(tmp_path, "org-a/model-bf16", 3)
    b = pc.quant_cache_dir(tmp_path, "org-b/model-bf16", 3)
    assert a != b
    assert a.name == "org-a--model-bf16-3bit"
    assert pc.quant_cache_dir(tmp_path, "org-a/model-bf16", 2) != a
    assert pc.quant_cache_dir(tmp_path, "/Users/x/models/m", 3).name == "Users--x--models--m-3bit"
