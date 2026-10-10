"""Shared test configuration."""

import os


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
