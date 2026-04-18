"""Model resolution for multiple commands (compare, sweep, stress, determinism).

Takes a model spec string and resolves it to a backend type + model path.
Supports HuggingFace repo IDs, local paths, and Ollama-style tags.

Resolution rules (in order):
  1. Explicit prefix  — ``ollama:llama3.1:8b-q4`` → openai-compat + Ollama
  2. Explicit prefix  — ``mlx:mlx-community/...`` → mlx-lm
  3. Explicit prefix  — ``gguf:/path/to/model.gguf`` → llama-cpp
  4. Local .gguf file — path exists and ends with ``.gguf`` → llama-cpp
  5. HF repo with ``-mlx`` or ``mlx-community/`` → mlx-lm
  6. HF repo with ``-GGUF`` or ``-gguf`` → llama-cpp (default: http://127.0.0.1:8080)
  7. Fallback — assume mlx-lm (most common local Mac use case)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["ResolvedModel", "resolve_model"]

BackendType = Literal["mlx-lm", "llama-cpp", "openai-compat", "vllm-mlx"]

# Prefixes that explicitly select a backend.
_PREFIX_MAP: dict[str, BackendType] = {
    "ollama": "openai-compat",
    "mlx": "mlx-lm",
    "gguf": "llama-cpp",
    "vllm-mlx": "vllm-mlx",
}

# Default base URLs per backend (can be overridden via CLI).
_DEFAULT_URLS: dict[BackendType, str] = {
    "openai-compat": "http://127.0.0.1:11434",  # Ollama (backend adds /v1/... paths)
    "llama-cpp": "http://127.0.0.1:8080",
    "vllm-mlx": "http://127.0.0.1:8000",
}


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A fully resolved model specification."""

    backend: BackendType
    model_id: str
    base_url: str | None
    label: str  # short human-readable label for tables / reports
    revision: str | None = None

    def __str__(self) -> str:
        res = f"{self.label} ({self.backend})"
        if self.revision:
            res += f" @ {self.revision}"
        return res


def _make_label(model_id: str) -> str:
    """Derive a short label from a model identifier.

    Examples:
        >>> _make_label("mlx-community/Llama-3.1-8B-Instruct-4bit")
        'Llama-3.1-8B-Instruct-4bit'
        >>> _make_label("bartowski/Llama-3.1-8B-Instruct-GGUF")
        'Llama-3.1-8B-Instruct-GGUF'
        >>> _make_label("llama3.1:8b-instruct-q4_K_M")
        'llama3.1:8b-instruct-q4_K_M'
    """
    # Strip HF org prefix if present.
    if "/" in model_id:
        return model_id.rsplit("/", 1)[-1]
    return model_id


def resolve_model(
    spec: str,
    *,
    base_url: str | None = None,
    label: str | None = None,
) -> ResolvedModel:
    """Resolve a model spec string into a backend type and model path.

    Args:
        spec: Model identifier. Can be prefixed (``ollama:model``,
              ``mlx:repo/model``, ``gguf:/path/to/file.gguf``) or bare.
              Optionally includes a revision after '@' (e.g. ``repo/model@main``).
        base_url: Override the default base URL for HTTP backends.
        label: Override the auto-derived label.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty model spec")

    revision: str | None = None
    if "@" in spec and not spec.startswith("@"):
        # Split on the LAST @ to allow for potential @ in paths if they exist (unlikely but safer)
        # Actually usually it's repo@rev.
        spec, revision = spec.rsplit("@", 1)

    # ── 1. Check for explicit prefix ─────────────────────────────────
    for prefix, backend in _PREFIX_MAP.items():
        pattern = f"^{re.escape(prefix)}:"
        if re.match(pattern, spec, re.IGNORECASE):
            model_id = spec[len(prefix) + 1 :]
            return ResolvedModel(
                backend=backend,
                model_id=model_id,
                base_url=base_url or _DEFAULT_URLS.get(backend),
                label=label or _make_label(model_id),
                revision=revision,
            )

    # ── 2. Local .gguf file path ─────────────────────────────────────
    local_path = Path(spec)
    if local_path.suffix.lower() == ".gguf":
        if local_path.exists():
            return ResolvedModel(
                backend="llama-cpp",
                model_id=str(local_path.resolve()),
                base_url=base_url or _DEFAULT_URLS["llama-cpp"],
                label=label or local_path.stem,
                revision=revision,
            )
        # Even if it doesn't exist yet, honour the extension.
        return ResolvedModel(
            backend="llama-cpp",
            model_id=spec,
            base_url=base_url or _DEFAULT_URLS["llama-cpp"],
            label=label or local_path.stem,
            revision=revision,
        )

    # ── 3. HuggingFace repo heuristics ──────────────────────────────
    spec_lower = spec.lower()

    # MLX repos (mlx-community org or -mlx suffix).
    if (
        spec_lower.startswith("mlx-community/")
        or spec_lower.endswith("-mlx")
        or "/mlx-" in spec_lower
        or spec_lower.startswith("mlx-")
    ):
        return ResolvedModel(
            backend="mlx-lm",
            model_id=spec,
            base_url=None,  # mlx-lm loads locally, no URL
            label=label or _make_label(spec),
            revision=revision,
        )

    # GGUF repos (typically served via Ollama or llama-cpp).
    gguf_indicators = ["gguf", "bartowski", "maziyarpanahi", "mradermacher"]
    if any(ind in spec_lower for ind in gguf_indicators):
        # Prefer llama-cpp for explicit GGUF repos unless it's an Ollama tag.
        return ResolvedModel(
            backend="llama-cpp",
            model_id=spec,
            base_url=base_url or _DEFAULT_URLS["llama-cpp"],
            label=label or _make_label(spec),
            revision=revision,
        )

    # ── 4. Ollama-style tags (contain colon but no slash) ────────────
    #    e.g. "llama3.1:8b-instruct-q4_K_M"
    if ":" in spec and "/" not in spec:
        return ResolvedModel(
            backend="openai-compat",
            model_id=spec,
            base_url=base_url or _DEFAULT_URLS["openai-compat"],
            label=label or _make_label(spec),
            revision=revision,
        )

    # ── 5. Fallback — assume mlx-lm (Mac-first user base) ───────────
    return ResolvedModel(
        backend="mlx-lm",
        model_id=spec,
        base_url=None,
        label=label or _make_label(spec),
        revision=revision,
    )
