"""Bundled prompt suites for infer-check.

Use ``get_suite_path("reasoning")`` to get the path to a bundled suite,
or ``list_suites()`` to see all available suites.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["get_suite_path", "list_suites"]

_PACKAGE = "infer_check.prompt_suites"


def list_suites() -> list[str]:
    """Return names of all bundled prompt suites (without .jsonl extension)."""
    suites = []
    for item in resources.files(_PACKAGE).iterdir():
        if hasattr(item, "name") and item.name.endswith(".jsonl"):
            suites.append(item.name.removesuffix(".jsonl"))
    return sorted(suites)


def get_suite_path(name: str) -> Path:
    """Resolve a suite name to a file path.

    Accepts either:
      - A bare name like ``"reasoning"`` (resolves to the bundled suite)
      - An existing file path (returned as-is)
    """
    # If it's already a path that exists, return it
    p = Path(name)
    if p.exists():
        return p

    # Try as a bundled suite name
    clean = name.removesuffix(".jsonl")
    ref = resources.files(_PACKAGE) / f"{clean}.jsonl"
    # resources.as_file() gives us a real filesystem path
    if ref.is_file():
        return Path(str(ref))

    available = list_suites()
    raise FileNotFoundError(
        f"Prompt suite '{name}' not found.\n"
        f"Available bundled suites: {', '.join(available)}\n"
        f"Or pass a path to a .jsonl file."
    )
