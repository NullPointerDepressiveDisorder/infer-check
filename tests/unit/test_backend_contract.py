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

# (class, constructor kwargs, supports_concurrency, supports_scoring)
BACKENDS = [
    (MLXBackend, {"model_id": "org/model"}, False, True),
    (LlamaCppBackend, {"model_id": "org/model"}, True, False),
    (OpenAICompatBackend, {"base_url": "http://127.0.0.1:8000", "model_id": "org/model"}, True, False),
    (VLLMMLXBackend, {"model_id": "org/model"}, True, False),
]
PARAMS = ("cls", "kwargs", "concurrent", "scoring")


@pytest.mark.parametrize(PARAMS, BACKENDS)
def test_backend_contract(cls: type, kwargs: dict[str, str], concurrent: bool, scoring: bool) -> None:
    # BackendAdapter isn't @runtime_checkable, so check explicit inheritance via the MRO.
    assert BackendAdapter in cls.__mro__
    backend = cls(**kwargs)
    assert backend.model_id == "org/model"
    assert isinstance(backend.name, str) and backend.name
    assert backend.supports_concurrency is concurrent
    assert backend.supports_scoring is scoring


@pytest.mark.parametrize(PARAMS, BACKENDS)
@pytest.mark.parametrize("method", ["generate", "score", "health_check", "cleanup"])
def test_protocol_methods_are_coroutines(
    cls: type, kwargs: dict[str, str], concurrent: bool, scoring: bool, method: str
) -> None:
    assert inspect.iscoroutinefunction(getattr(cls, method)), f"{cls.__name__}.{method} must be async"


@pytest.mark.parametrize(PARAMS, [b for b in BACKENDS if not b[3]])
def test_non_scoring_backends_raise_not_implemented(
    cls: type, kwargs: dict[str, str], concurrent: bool, scoring: bool
) -> None:
    import asyncio

    from infer_check.types import Prompt

    with pytest.raises(NotImplementedError):
        asyncio.run(cls(**kwargs).score(Prompt(text="hi"), [1, 2, 3]))
