from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from infer_check.cli import main
from infer_check.types import ComparisonResult, InferenceResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_diff_summary_table(runner: CliRunner, tmp_path: Path) -> None:
    # Setup dummy prompts file
    dummy_suite = tmp_path / "dummy.jsonl"
    dummy_suite.write_text(
        '{"id":"p1", "text":"hi", "category":"general"}\n{"id":"p2", "text":"bye", "category":"general"}'
    )

    # Mock InferenceResults
    inf_res_baseline_1 = InferenceResult(
        prompt_id="p1", backend_name="baseline", model_id="m", tokens=["hi"], text="hi", latency_ms=1.0
    )
    inf_res_test_1 = InferenceResult(
        prompt_id="p1", backend_name="test", model_id="m", tokens=["hi"], text="hi", latency_ms=1.0
    )

    inf_res_baseline_2 = InferenceResult(
        prompt_id="p2", backend_name="baseline", model_id="m", tokens=["bye"], text="bye", latency_ms=1.0
    )
    inf_res_test_2 = InferenceResult(
        prompt_id="p2", backend_name="test", model_id="m", tokens=["hello"], text="hello", latency_ms=1.0
    )

    # Mock ComparisonResults
    # comp1: not flipped
    comp1 = ComparisonResult(
        baseline=inf_res_baseline_1,
        test=inf_res_test_1,
        text_similarity=1.0,
        is_failure=False,
        metadata={"flipped": False},
    )
    # comp2: flipped
    comp2 = ComparisonResult(
        baseline=inf_res_baseline_2,
        test=inf_res_test_2,
        text_similarity=0.4,
        is_failure=True,
        metadata={"flipped": True},
    )

    with (
        patch("infer_check.backends.base.get_backend"),
        patch("infer_check.runner.TestRunner.diff") as mock_diff,
    ):
        # We need an async mock that returns the list of comparisons
        async def mock_diff_async(*args: Any, **kwargs: Any) -> list[ComparisonResult]:
            return [comp1, comp2]

        mock_diff.side_effect = mock_diff_async

        result = runner.invoke(
            main,
            [
                "diff",
                "--model",
                "m1",
                "--backends",
                "mlx-lm,llama-cpp",
                "--prompts",
                str(dummy_suite),
                "--output",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        # Output should contain the table headers
        assert "test_backend" in result.output
        assert "failures" in result.output
        assert "failure_rate" in result.output
        assert "flip_rate" in result.output
        assert "mean_similarity" in result.output

        # Check backend name and metrics
        assert "llama-cpp" in result.output  # Backend name used in runner.diff?
        # Actually in cli.py it uses backend_names = [b.strip() for b in backends.split(",")]
        # and it pads it. In my mock, the Comparisons results have backend_name from inf_res.
        # But groups = defaultdict(list)
        # for comp in comparisons:
        #     groups[comp.test.backend_name].append(comp)

        assert "test" in result.output  # backend_name in inf_res_test_*

        # 2 prompts, 1 failure -> 50.00%
        assert "50.00%" in result.output

        # 1 flip out of 2 -> 50.0%
        # The formatting in cli.py is f"[{'red' if flip_rate > 0.1 else 'green'}]{flip_rate:.1%}[/]"
        # Rich markup might be stripped or present depending on how CliRunner handles it.
        # Usually CliRunner output doesn't have the color codes unless we tell it to.
        assert "50.0%" in result.output

        # mean similarity: (1.0 + 0.4) / 2 = 0.7
        assert "0.7000" in result.output


def test_cli_diff_summary_no_flips(runner: CliRunner, tmp_path: Path) -> None:
    dummy_suite = tmp_path / "dummy_no_flips.jsonl"
    dummy_suite.write_text('{"id":"p1", "text":"hi", "category":"general"}')

    inf_res_baseline = InferenceResult(
        prompt_id="p1", backend_name="baseline", model_id="m", tokens=["hi"], text="hi", latency_ms=1.0
    )
    inf_res_test = InferenceResult(
        prompt_id="p1", backend_name="test", model_id="m", tokens=["hi"], text="hi", latency_ms=1.0
    )

    comp = ComparisonResult(
        baseline=inf_res_baseline, test=inf_res_test, text_similarity=1.0, is_failure=False, metadata={"flipped": False}
    )

    with (
        patch("infer_check.backends.base.get_backend"),
        patch("infer_check.runner.TestRunner.diff") as mock_diff,
    ):

        async def mock_diff_async(*args: Any, **kwargs: Any) -> list[ComparisonResult]:
            return [comp]

        mock_diff.side_effect = mock_diff_async

        result = runner.invoke(
            main,
            [
                "diff",
                "--model",
                "m1",
                "--backends",
                "mlx-lm,llama-cpp",
                "--prompts",
                str(dummy_suite),
                "--output",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "0.0%" in result.output
