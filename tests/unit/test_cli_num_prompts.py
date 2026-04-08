import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from infer_check.cli import main
from infer_check.runner import TestRunner
from infer_check.types import SweepResult


def test_num_prompts_limit() -> None:
    """Test that --num-prompts correctly limits the number of prompts used."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        prompt_path = Path("test_prompts.jsonl")
        prompts = [
            {"text": "Prompt 1"},
            {"text": "Prompt 2"},
            {"text": "Prompt 3"},
            {"text": "Prompt 4"},
        ]
        prompt_path.write_text("\n".join(json.dumps(p) for p in prompts), encoding="utf-8")

        mock_result = SweepResult(
            model_id="m1",
            backend_name="mlx",
            quantization_levels=["a", "b"],
            comparisons=[],
            timestamp=datetime.now(UTC),
            summary={},
        )

        with (
            patch("infer_check.cli._resolve_prompts", return_value=prompt_path),
            patch("infer_check.backends.base.get_backend_for_model"),
            patch.object(TestRunner, "sweep", return_value=mock_result) as mock_sweep,
        ):
            # 1. Test global --num-prompts
            result = runner.invoke(
                main, ["--num-prompts", "2", "sweep", "--models", "a=m1,b=m2", "--prompts", "test_prompts.jsonl"]
            )
            assert result.exit_code == 0
            args, kwargs = mock_sweep.call_args
            captured_prompts = kwargs.get("prompts") or args[2]
            assert len(captured_prompts) == 2
            assert captured_prompts[0].text == "Prompt 1"
            assert captured_prompts[1].text == "Prompt 2"

            # 2. Test subcommand --num-prompts override
            result = runner.invoke(
                main,
                [
                    "--num-prompts",
                    "2",
                    "sweep",
                    "--models",
                    "a=m1,b=m2",
                    "--prompts",
                    "test_prompts.jsonl",
                    "--num-prompts",
                    "3",
                ],
            )
            assert result.exit_code == 0
            args, kwargs = mock_sweep.call_args
            captured_prompts = kwargs.get("prompts") or args[2]
            assert len(captured_prompts) == 3

            # 3. Test without --num-prompts (should use all)
            result = runner.invoke(main, ["sweep", "--models", "a=m1,b=m2", "--prompts", "test_prompts.jsonl"])
            assert result.exit_code == 0
            args, kwargs = mock_sweep.call_args
            captured_prompts = kwargs.get("prompts") or args[2]
            assert len(captured_prompts) == 4
