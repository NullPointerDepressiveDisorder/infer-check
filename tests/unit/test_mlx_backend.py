import asyncio
import sys
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from infer_check.backends.mlx_lm import MLXBackend
from infer_check.utils import format_prompt


@pytest.fixture
def mock_mlx() -> Generator[tuple[MagicMock, MagicMock, MagicMock]]:
    mock_mlx_lm = MagicMock()
    sys.modules["mlx_lm"] = mock_mlx_lm

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

    yield mock_mlx_lm.load, mock_model, mock_tokenizer

    del sys.modules["mlx_lm"]


def test_mlx_chat_template(mock_mlx: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_tokenizer = mock_mlx
    mock_tokenizer.chat_template = "dummy_template"
    mock_tokenizer.apply_chat_template.return_value = "<chat>hello</chat>"

    backend = MLXBackend(model_id="dummy-model")
    # Need to mock the actual generation to avoid entering mlx framework deep logic
    # We will just patch `_generate_with_logprobs` or `_generate_simple`
    # Or properly mock mlx_lm.generate

    backend._model = mock_mlx[1]
    backend._tokenizer = mock_tokenizer

    formatted = format_prompt("hello", tokenizer=backend._tokenizer)
    assert formatted == "<chat>hello</chat>"
    mock_tokenizer.apply_chat_template.assert_called_once()


def test_mlx_load_error_handling(mock_mlx: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    mock_load, _, _ = mock_mlx

    backend = MLXBackend(model_id="non-existent")

    mock_load.side_effect = ValueError("404 Repository Not Found")
    with pytest.raises(RuntimeError) as exc_info:
        backend._ensure_loaded()
    assert "Model not found" in str(exc_info.value)

    backend = MLXBackend(model_id="gated-model")
    mock_load.side_effect = ValueError("gated repo")
    with pytest.raises(RuntimeError) as exc_info:
        backend._ensure_loaded()
    assert "requires authentication" in str(exc_info.value)


def test_mlx_cleanup(mock_mlx: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    backend = MLXBackend(model_id="dummy-model")
    # trigger load
    backend._ensure_loaded()
    assert backend._model is not None
    assert backend._tokenizer is not None

    asyncio.run(backend.cleanup())
    assert backend._model is None
    assert backend._tokenizer is None


@pytest.mark.asyncio
async def test_mlx_generate_fallback(mock_mlx: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    from unittest.mock import patch

    from infer_check.types import InferenceResult, Prompt

    backend = MLXBackend(model_id="dummy-model")
    backend._model = mock_mlx[1]
    backend._tokenizer = mock_mlx[2]

    prompt = Prompt(text="test prompt")
    simple_result = InferenceResult(
        prompt_id=prompt.id,
        backend_name="mlx-lm",
        model_id="dummy-model",
        tokens=["hello"],
        text="hello",
        latency_ms=10.0,
    )

    with (
        patch.object(MLXBackend, "_generate_with_logprobs") as mock_logprobs,
        patch.object(MLXBackend, "_generate_simple") as mock_simple,
        patch("rich.console.Console.print") as mock_print,
    ):
        mock_logprobs.side_effect = Exception("Logprobs failed")
        mock_simple.return_value = simple_result

        result = await backend.generate(prompt)

        assert result == simple_result
        mock_logprobs.assert_called_once_with(prompt)
        mock_simple.assert_called_once_with(prompt)
        mock_print.assert_called_once()
        args, _ = mock_print.call_args
        assert "generate_step failed, falling back to simple generate" in args[0]
        assert "Logprobs failed" in args[0]


@pytest.mark.asyncio
async def test_mlx_generate_double_failure(mock_mlx: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    from unittest.mock import patch

    from infer_check.types import Prompt

    backend = MLXBackend(model_id="dummy-model")
    backend._model = mock_mlx[1]
    backend._tokenizer = mock_mlx[2]

    prompt = Prompt(text="test prompt")

    with (
        patch.object(MLXBackend, "_generate_with_logprobs") as mock_logprobs,
        patch.object(MLXBackend, "_generate_simple") as mock_simple,
        patch("rich.console.Console.print"),
    ):
        mock_logprobs.side_effect = Exception("Logprobs failed")
        mock_simple.side_effect = Exception("Simple failed")

        with pytest.raises(RuntimeError) as exc_info:
            await backend.generate(prompt)

        assert "MLX generation failed" in str(exc_info.value)
        assert "Simple failed" in str(exc_info.value)
