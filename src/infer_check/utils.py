"""Shared utilities for infer-check."""

from __future__ import annotations

import re

from transformers import SentencePieceBackend, TokenizersBackend


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


def format_prompt(
    text: str,
    tokenizer: TokenizersBackend | SentencePieceBackend | None = None,
    model_id: str | None = None,
    revision: str | None = None,
) -> str:
    """Apply chat template client-side.

    Uses an existing tokenizer if provided (mlx-lm path),
    or loads one from HuggingFace by model_id (HTTP backend path).
    """
    if tokenizer is None and model_id:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    if tokenizer and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": text}]
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return text
