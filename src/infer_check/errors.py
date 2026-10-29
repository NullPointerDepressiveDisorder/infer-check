"""What a run does when a prompt fails to generate or score.

A failed prompt is never dropped silently. Each error is recorded, the summary
reports it, and the CLI exits non-zero. Errors are not random: a model that loops
writes the longest outputs and is the likeliest to time out or run out of memory,
so quietly leaving its errors out makes a damaged model look healthier.

Errors that leave the rest of a run worthless stop it instead: running out of GPU
memory (every later generation fails too), a server that refuses connections even
after retries, or several errors in a row.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Literal

from rich.console import Console

from infer_check.types import RunError

# Text that means GPU memory is exhausted, e.g. MLX's
# "[METAL] Command buffer execution failed: Insufficient Memory
# (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".
_OUT_OF_MEMORY = ("insufficient memory", "outofmemory", "out of memory")

MAX_CONSECUTIVE_ERRORS = 3


class RunAbortedError(RuntimeError):
    """An error after which carrying on would only produce more errors."""


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def systemic_reason(exc: BaseException) -> str | None:
    """Why ``exc`` means the run can't usefully continue, or None if it's isolated.

    Backends wrap the original error (``raise ... from exc``), so the whole chain
    is checked.
    """
    import httpx

    for e in _chain(exc):
        if isinstance(e, MemoryError) or any(m in str(e).lower() for m in _OUT_OF_MEMORY):
            return "out of GPU memory"
        if isinstance(e, httpx.ConnectError):
            return "the server refused the connection"
    return None


def describe(exc: BaseException) -> str:
    """One line naming the root cause, e.g. ``RuntimeError: [METAL] ... Insufficient Memory``."""
    root = list(_chain(exc))[-1]
    for candidate in (root, exc):
        lines = [line.strip() for line in str(candidate).splitlines() if line.strip()]
        if lines:
            return f"{type(root).__name__}: {lines[0]}"[:300]
    return type(root).__name__


class ErrorLog:
    """Errors from one run, and the rule for when to stop it."""

    def __init__(self, max_consecutive: int = MAX_CONSECUTIVE_ERRORS) -> None:
        self.errors: list[RunError] = []
        self._max_consecutive = max_consecutive
        self._streak = 0

    def ok(self) -> None:
        """Note a success; only errors in a row count toward stopping."""
        self._streak = 0

    def record(
        self,
        label: str,
        prompt_id: str,
        exc: BaseException,
        stage: Literal["generate", "score"] = "generate",
        *,
        streak: bool = True,
    ) -> None:
        """Record an error, raising ``RunAbortedError`` if the run should stop.

        ``streak=False`` is for stress tests, where errors under load are what is
        being measured: they are recorded but don't count toward stopping.
        """
        error = RunError(label=label, prompt_id=prompt_id, stage=stage, error=describe(exc))
        self.errors.append(error)
        Console().print(f"[yellow]  ⚠ {label}: {stage} failed for '{prompt_id}': {error.error}[/yellow]")
        if streak:
            self._streak += 1
        reason = systemic_reason(exc)
        if reason is None and self._streak >= self._max_consecutive:
            reason = f"{self._streak} errors in a row"
        if reason is not None:
            raise RunAbortedError(
                f"Run stopped: {reason}. Last error: {stage} of '{prompt_id}' at {label}: {error.error}"
            ) from exc

    def counts(self) -> dict[str, int]:
        """Errors per label (quant level, model or backend)."""
        return dict(Counter(e.label for e in self.errors))
