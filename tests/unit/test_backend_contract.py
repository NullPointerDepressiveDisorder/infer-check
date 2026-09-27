"""Every backend satisfies the BackendAdapter contract.

Guards against regressions like making ``generate`` synchronous: every
caller in the runner does ``await backend.generate(...)``.
"""

import inspect

import pytest

from infer_check.backends.base import BackendAdapter
from infer_check.backends.llama_cpp import LlamaCppBackend
from infer_check.backends.mlx_lm import MLXBackend
from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.backends.vllm_mlx import VLLMMLXBackend

BACKENDS = [
    (MLXBackend, {"model_id": "org/model"}, False),
    (LlamaCppBackend, {"model_id": "org/model"}, True),
    (OpenAICompatBackend, {"base_url": "http://127.0.0.1:8000", "model_id": "org/model"}, True),
    (VLLMMLXBackend, {"model_id": "org/model"}, True),
]


@pytest.mark.parametrize(("cls", "kwargs", "concurrent"), BACKENDS)
def test_backend_contract(cls: type, kwargs: dict[str, str], concurrent: bool) -> None:
    # BackendAdapter isn't @runtime_checkable, so check explicit inheritance via the MRO.
    assert BackendAdapter in cls.__mro__
    backend = cls(**kwargs)
    assert backend.model_id == "org/model"
    assert isinstance(backend.name, str) and backend.name
    assert backend.supports_concurrency is concurrent


@pytest.mark.parametrize(("cls", "kwargs", "concurrent"), BACKENDS)
@pytest.mark.parametrize("method", ["generate", "health_check", "cleanup"])
def test_protocol_methods_are_coroutines(cls: type, kwargs: dict[str, str], concurrent: bool, method: str) -> None:
    assert inspect.iscoroutinefunction(getattr(cls, method)), f"{cls.__name__}.{method} must be async"
