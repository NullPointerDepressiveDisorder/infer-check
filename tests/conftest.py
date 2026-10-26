"""Shared test configuration."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _results_snapshot() -> set[Path]:
    return set(_REPO_RESULTS.rglob("*")) if _REPO_RESULTS.exists() else set()


@pytest.fixture(autouse=True)
def _no_writes_to_checkout_results() -> Iterator[None]:
    """Fail any test that writes into the checkout's results/ directory.

    CLI commands save there by default; tests must pass ``--output`` under
    ``tmp_path``, or stray files contaminate later ``infer-check report`` runs.
    """
    before = _results_snapshot()
    yield
    new = sorted(str(p.relative_to(_REPO_RESULTS)) for p in _results_snapshot() - before)
    assert not new, f"test wrote into the checkout's results/: {new[:5]}"


def pytest_configure(config: object) -> None:
    """Optionally pin MLX to the CPU.

    Linux CI runs MLX on the CPU (there is no Metal); set
    ``INFER_CHECK_MLX_DEVICE=cpu`` to reproduce that on a Mac (``make test-mlx-cpu``).
    """
    if os.environ.get("INFER_CHECK_MLX_DEVICE", "").lower() != "cpu":
        return
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.set_default_device(mx.cpu)
