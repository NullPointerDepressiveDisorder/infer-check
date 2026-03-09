"""Inference backend adapters."""

from infer_check.backends.base import BackendAdapter, BackendConfig, get_backend
from infer_check.backends.llama_cpp import LlamaCppBackend
from infer_check.backends.mlx_lm import MLXBackend
from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.backends.vllm_mlx import VLLMMLXBackend

__all__ = [
    "BackendAdapter",
    "BackendConfig",
    "LlamaCppBackend",
    "MLXBackend",
    "OpenAICompatBackend",
    "VLLMMLXBackend",
    "get_backend",
]
