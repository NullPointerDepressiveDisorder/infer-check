"""Shared utilities for infer-check."""

from __future__ import annotations

import re


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
    if re.match(reserved_pattern, safe, re.IGNORECASE):
        safe += "_"

    # Ensure we have something left
    return safe if safe else "model"
