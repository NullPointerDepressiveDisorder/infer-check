from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from infer_check.types import SweepResult


def test_sweep_model_parsing_robustness() -> None:
    """Test that sweep command parses model paths robustly, handling extra equals signs."""
    # Create a mock SweepResult to return from runner.sweep
    mock_sweep_result = SweepResult(
        model_id="test-model",
        backend_name="test-backend",
        quantization_levels=["bf16", "4bit"],
        comparisons=[],
        timestamp=datetime.now(UTC),
        summary={},
    )

    # We mock get_backend_for_model and TestRunner.sweep to avoid actual initialization
    with (
        patch("infer_check.backends.base.get_backend_for_model") as mock_get_backend,
        patch("infer_check.runner.TestRunner.sweep", new_callable=Mock),
        patch("infer_check.suites.loader.load_suite", return_value=[MagicMock()]),
        patch("infer_check.cli._resolve_prompts", return_value=Path("dummy.jsonl")),
        patch("asyncio.run", return_value=mock_sweep_result),
    ):
        # Simulating the command: infer-check sweep --models "bf16==path/to/model" --prompts dummy
        # We call the function directly as click command
        from click.testing import CliRunner

        from infer_check.cli import main

        runner = CliRunner()
        # Using a subset of arguments to trigger the parsing logic
        runner.invoke(main, ["sweep", "--models", "bf16==bartowski/Qwen,4bit=bartowski/Qwen", "--prompts", "reasoning"])

        # Check if get_backend_for_model was called with cleaned paths
        # It should be called twice: once for bf16 and once for 4bit
        assert mock_get_backend.call_count == 2

        # Check first call (bf16)
        args, kwargs = mock_get_backend.call_args_list[0]
        assert kwargs["model_str"] == "bartowski/Qwen"
        assert kwargs["quantization"] == "bf16"

        # Check second call (4bit)
        args, kwargs = mock_get_backend.call_args_list[1]
        assert kwargs["model_str"] == "bartowski/Qwen"
        assert kwargs["quantization"] == "4bit"
