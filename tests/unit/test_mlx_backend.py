import asyncio
import sys
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from infer_check.backends.mlx_lm import MLXBackend


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

    formatted = backend._format_prompt("hello")
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
