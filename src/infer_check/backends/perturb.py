"""Deliberately degraded models, for positive controls.

A correctness tool has to show it *can* detect damage, not only that it found
none. These transforms make a model measurably worse by a controlled amount, so
KL, top-1 agreement and accuracy should all degrade monotonically as the
damage grows. If they don't, the metrics are broken.

Example::

    from infer_check.backends.mlx_lm import MLXBackend
    from infer_check.backends.perturb import fixed_logit_noise

    damaged = MLXBackend(
        "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        quantization="noise σ=1.0",
        logits_transform=fixed_logit_noise(sigma=1.0, seed=0),
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

__all__ = ["fixed_logit_noise"]


def fixed_logit_noise(sigma: float, seed: int = 0) -> Callable[[mx.array], mx.array]:
    """Add a fixed random bias, drawn once from N(0, sigma²) per vocabulary entry, to every logit row.

    The same bias applies at every position, in generation and in scoring alike,
    so the damaged model is deterministic and teacher-forced KL measures exactly
    the damage. ``sigma=0`` is the identity.
    """
    cache: dict[int, mx.array] = {}

    def transform(logits: mx.array) -> mx.array:
        import mlx.core as mx

        if sigma == 0:
            return logits
        vocab = int(logits.shape[-1])
        if vocab not in cache:
            cache[vocab] = mx.random.normal((vocab,), key=mx.random.key(seed)) * sigma
        return logits + cache[vocab].astype(logits.dtype)

    return transform
