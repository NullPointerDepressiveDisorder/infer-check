"""Shared utilities for infer-check."""

from __future__ import annotations

import contextlib
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from transformers import SentencePieceBackend, TokenizersBackend

# Tokens that trigger reasoning mode on specific model/runner combos. When
# ``disable_thinking`` is set we strip them from the prompt so that a stray
# token in user input can't re-enable thinking. ``<|think|>`` is Ollama's
# system-prompt trigger for gpt-oss-style models; ``<think>…</think>`` is the
# DeepSeek-R1 / Qwen reasoning wrapper.
_THINKING_TOKEN_PATTERNS = (
    re.compile(r"<\|think\|>", re.IGNORECASE),
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"</?think>", re.IGNORECASE),
)


def strip_thinking_tokens(text: str) -> str:
    """Remove reasoning-trigger tokens from ``text``."""
    for pattern in _THINKING_TOKEN_PATTERNS:
        text = pattern.sub("", text)
    return text


def sanitize_filename(label: str) -> str:
    """Sanitize a label for safe use in filenames across platforms.

    - Replaces path separators (/ \\) and parent directory references (..)
    - Replaces other filesystem-unsafe characters (< > : " | ? *)
    - Replaces control characters (\\x00-\\x1f)
    - Collapses multiple underscores into one
    - Strips leading and trailing dots, underscores and whitespace
    - Appends an underscore to Windows reserved names (CON, PRN, etc.)
    """
    # Replace path separators and parent directory references
    safe = label.replace("/", "_").replace("\\", "_")
    safe = safe.replace("..", "_")

    # Replace other filesystem-unsafe characters and control characters
    # Matches: / \ : * ? " < > | and \x00-\x1f
    safe = re.sub(r'[<>:"|?*\x00-\x1f]', "_", safe)

    # Collapse multiple underscores into one
    safe = re.sub(r"_+", "_", safe)

    # Strip leading/trailing underscores, dots, and whitespace
    # Windows doesn't allow trailing dots or spaces
    safe = safe.strip("._ ")

    # Handle Windows reserved device names
    # CON, PRN, AUX, NUL, COM1-9, LPT1-9
    reserved_pattern = r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$"
    stem = safe.split(".", 1)[0]
    if re.match(reserved_pattern, stem, re.IGNORECASE):
        if "." in safe:
            name, ext = safe.split(".", 1)
            safe = f"{name}_{'.' + ext}"
        else:
            safe += "_"

    # Ensure we have something left
    return safe if safe else "model"


@lru_cache(maxsize=8)
def _get_tokenizer(model_id: str, revision: str | None = None) -> Any:
    """Helper to load and cache HuggingFace tokenizers."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def format_prompt(
    text: str,
    tokenizer: TokenizersBackend | SentencePieceBackend | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    disable_thinking: bool = False,
) -> str:
    """Apply chat template client-side.

    Uses an existing tokenizer if provided (mlx-lm path),
    or loads one from HuggingFace by model_id (HTTP backend path).

    When ``disable_thinking`` is True, attempt to turn off reasoning/thinking
    mode via the chat template. This works across model families that expose a
    template flag (Qwen3 and derivatives use ``enable_thinking``; some DeepSeek
    and HunYuan variants use ``thinking``). Templates that don't know the flag
    ignore it; templates that reject unknown kwargs trigger a graceful fallback
    to normal rendering, so non-thinking models keep working unchanged.
    """
    if disable_thinking:
        text = strip_thinking_tokens(text)

    if tokenizer is None and model_id:
        # Only attempt to load from HF if it looks like a HF repo (owner/name)
        # or an absolute/relative path. Ollama tags (name:tag) or local GGUF
        # files should be skipped as they'll fail or hang from_pretrained.
        is_hf_id = "/" in model_id or (model_id.count(":") == 0 and "." not in model_id)
        if is_hf_id:
            with contextlib.suppress(Exception):
                tokenizer = _get_tokenizer(model_id, revision)

    if tokenizer and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": text}]
        if disable_thinking:
            for kwargs in ({"enable_thinking": False}, {"thinking": False}):
                try:
                    return str(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            **cast(dict[str, Any], kwargs),
                        )
                    )
                except TypeError:
                    continue
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return text
