import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from infer_check.cli import main


def test_max_tokens_prompt_precedence() -> None:
    """Test that prompt-level max_tokens override takes precedence over global --max-tokens."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        prompt_path = Path("test_prompts.jsonl")
        # One prompt with explicit max_tokens, one without
        prompts = [{"text": "Override", "max_tokens": 123}, {"text": "Default"}]
        prompt_path.write_text("\n".join(json.dumps(p) for p in prompts), encoding="utf-8")

        from datetime import UTC, datetime

        from infer_check.runner import TestRunner
        from infer_check.types import SweepResult

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
            # Run with global --max-tokens 512
            result = runner.invoke(
                main, ["--max-tokens", "512", "sweep", "--models", "a=m1,b=m2", "--prompts", "test_prompts.jsonl"]
            )
            assert result.exit_code == 0

            # Check the call to sweep
            args, kwargs = mock_sweep.call_args
            captured_prompts = kwargs.get("prompts") or args[2]

            assert len(captured_prompts) == 2
            # Prompt 1: should keep its own 123
            assert captured_prompts[0].text == "Override"
            assert captured_prompts[0].max_tokens == 123
            # Prompt 2: should get the global 512
            assert captured_prompts[1].text == "Default"
            assert captured_prompts[1].max_tokens == 512


def test_max_tokens_propagation_override() -> None:
    """Test that a custom global --max-tokens is propagated."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        prompt_path = Path("test_prompts.jsonl")
        prompt_path.write_text(json.dumps({"text": "Hello world"}), encoding="utf-8")

        from datetime import UTC, datetime

        from infer_check.runner import TestRunner
        from infer_check.types import SweepResult

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
            # Run with --max-tokens 512
            result = runner.invoke(
                main, ["--max-tokens", "512", "sweep", "--models", "a=m1,b=m2", "--prompts", "test_prompts.jsonl"]
            )
            if result.exception:
                print(result.exception)
                raise result.exception
            assert result.exit_code == 0

            args, kwargs = mock_sweep.call_args
            captured_prompts = kwargs.get("prompts") or args[2]
            assert captured_prompts[0].max_tokens == 512
