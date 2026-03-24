from pathlib import Path

import pytest
from click.testing import CliRunner

from infer_check.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "infer-check" in result.output
    assert "sweep" in result.output
    assert "diff" in result.output


def test_sweep_fewer_than_2_models(runner: CliRunner, tmp_path: Path) -> None:
    dummy_suite = tmp_path / "dummy.jsonl"
    dummy_suite.touch()
    result = runner.invoke(main, ["sweep", "--models", "just-one-model", "--prompts", str(dummy_suite)])
    assert result.exit_code != 0
    assert "Need at least 2 models" in result.output


def test_sweep_invalid_baseline(runner: CliRunner, tmp_path: Path) -> None:
    dummy_suite = tmp_path / "dummy.jsonl"
    dummy_suite.touch()
    result = runner.invoke(
        main,
        [
            "sweep",
            "--models",
            "a=m1,b=m2",
            "--prompts",
            str(dummy_suite),
            "--baseline",
            "c",  # c is not in models
        ],
    )
    assert result.exit_code != 0
    assert "Baseline 'c' not found in model map" in result.output


def test_sweep_parsing_logic(runner: CliRunner, tmp_path: Path) -> None:
    # Just asserting the parsing setup works correctly
    # We create a dummy suite to bypass load_suite failing
    dummy_suite = tmp_path / "dummy.jsonl"
    dummy_suite.write_text('{"id":"p1", "text":"hi"}')

    # We will patch get_backend and runner.TestRunner.sweep
    from unittest.mock import patch

    with (
        patch("infer_check.backends.base.get_backend"),
        patch("infer_check.runner.TestRunner.sweep") as mock_sweep,
    ):
        from datetime import UTC, datetime

        from infer_check.types import SweepResult

        mock_sweep.return_value = SweepResult(
            model_id="m1",
            backend_name="mlx-lm",
            quantization_levels=["a", "b"],
            comparisons=[],
            timestamp=datetime.now(UTC),
            summary={},
        )

        result = runner.invoke(
            main,
            [
                "sweep",
                "--models",
                "a=m1,m2",  # m2 should become label 'm2'
                "--prompts",
                str(dummy_suite),
                "--output",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert mock_sweep.called
        kwargs = mock_sweep.call_args.kwargs
        # Ensure a=m1 and m2=m2 parsing worked correctly
        assert list(kwargs["backend_map"].keys()) == ["a", "m2"]
        assert kwargs["baseline_quant"] == "a"
