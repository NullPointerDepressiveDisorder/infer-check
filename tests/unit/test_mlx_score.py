"""MLXBackend._score against a tiny randomly initialised Llama (real forward passes, no download).

The reference for every check is one uncached forward pass over
``prompt + target``: row ``i`` of the scores must equal the log-softmax at
position ``len(prompt) - 1 + i`` — the position that predicts ``target[i]``.
"""

import asyncio
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

from mlx_lm.models.llama import Model, ModelArgs  # noqa: E402

from infer_check.backends.mlx_lm import MLXBackend  # noqa: E402
from infer_check.types import Prompt  # noqa: E402

VOCAB = 50
PROMPT_IDS = [1, 7, 3, 9, 4, 11]
TARGET_IDS = [5, 2, 8, 8, 13, 21, 3, 0, 17, 6, 30]


@pytest.fixture(scope="module")
def tiny_model() -> Any:
    mx.random.seed(0)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
        )
    )
    mx.eval(model.parameters())
    return model


@pytest.fixture
def backend(tiny_model: Any, monkeypatch: pytest.MonkeyPatch) -> MLXBackend:
    be = MLXBackend(model_id="tiny/random")
    be._model = tiny_model
    be._tokenizer = object()
    # Bypass chat templating/tokenization: the prompt is already token IDs.
    monkeypatch.setattr(MLXBackend, "_format", lambda self, prompt: "")
    monkeypatch.setattr(MLXBackend, "_encode", lambda self, formatted: list(PROMPT_IDS))
    return be


def reference_logprobs(model: Any, target: list[int]) -> np.ndarray:
    """Uncached single forward pass; rows aligned so row i predicts target[i]."""
    logits = model(mx.array(PROMPT_IDS + target)[None])[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    start = len(PROMPT_IDS) - 1
    return np.array(lp[start : start + len(target)])


def test_rows_align_with_the_tokens_they_predict(backend: MLXBackend, tiny_model: Any) -> None:
    ref = reference_logprobs(tiny_model, TARGET_IDS)
    scores = backend._score(Prompt(text="x"), TARGET_IDS, top_k=5, at_ids=np.array([[t] for t in TARGET_IDS]))

    assert len(scores) == len(TARGET_IDS)
    # Log-prob of each actual target token matches the reference at the right row.
    assert scores.at_logprobs is not None
    np.testing.assert_allclose(
        scores.at_logprobs[:, 0], ref[np.arange(len(TARGET_IDS)), TARGET_IDS], rtol=1e-4, atol=1e-4
    )
    np.testing.assert_array_equal(scores.argmax, ref.argmax(axis=-1))
    # Off-by-one would make these misalign: shifted rows must NOT match.
    assert not np.allclose(scores.at_logprobs[1:, 0], ref[np.arange(len(TARGET_IDS) - 1), TARGET_IDS[1:]], atol=1e-3)


def test_top_k_is_sorted_and_matches_reference(backend: MLXBackend, tiny_model: Any) -> None:
    ref = reference_logprobs(tiny_model, TARGET_IDS)
    scores = backend._score(Prompt(text="x"), TARGET_IDS, top_k=8, at_ids=None)

    assert scores.top_ids.shape == scores.top_logprobs.shape == (len(TARGET_IDS), 8)
    assert np.all(np.diff(scores.top_logprobs, axis=1) <= 1e-6)  # descending
    expected = -np.sort(-ref, axis=1)[:, :8]
    np.testing.assert_allclose(scores.top_logprobs, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        np.take_along_axis(ref, scores.top_ids, axis=1), scores.top_logprobs, rtol=1e-4, atol=1e-4
    )
    assert scores.at_logprobs is None


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 1000])
def test_chunking_does_not_change_results(backend: MLXBackend, chunk_size: int) -> None:
    full = backend._score(Prompt(text="x"), TARGET_IDS, top_k=6, at_ids=None, chunk_size=1000)
    chunked = backend._score(Prompt(text="x"), TARGET_IDS, top_k=6, at_ids=None, chunk_size=chunk_size)
    np.testing.assert_array_equal(full.argmax, chunked.argmax)
    np.testing.assert_allclose(full.top_logprobs, chunked.top_logprobs, rtol=1e-4, atol=1e-4)


def test_top_k_zero_and_clamping(backend: MLXBackend) -> None:
    none = backend._score(Prompt(text="x"), TARGET_IDS, top_k=0, at_ids=None)
    assert none.top_ids.shape == (len(TARGET_IDS), 0)
    assert len(none.argmax) == len(TARGET_IDS)
    clamped = backend._score(Prompt(text="x"), TARGET_IDS, top_k=10_000, at_ids=None)
    assert clamped.top_ids.shape == (len(TARGET_IDS), VOCAB)
    # Full vocab covered: probabilities sum to 1.
    np.testing.assert_allclose(np.exp(clamped.top_logprobs).sum(axis=1), 1.0, atol=1e-4)


def test_empty_target_and_bad_at_ids(backend: MLXBackend) -> None:
    empty = backend._score(Prompt(text="x"), [], top_k=4, at_ids=np.zeros((0, 3), dtype=np.int64))
    assert len(empty) == 0
    assert empty.top_ids.shape[0] == 0 and len(empty.argmax) == 0
    assert empty.at_logprobs is not None and empty.at_logprobs.shape == (0, 3)
    with pytest.raises(ValueError, match="at_ids"):
        backend._score(Prompt(text="x"), TARGET_IDS, top_k=4, at_ids=np.zeros((2, 3), dtype=np.int64))


def test_async_score_entry_point(backend: MLXBackend) -> None:
    scores = asyncio.run(backend.score(Prompt(text="x"), TARGET_IDS, top_k=3))
    assert scores.top_ids.shape == (len(TARGET_IDS), 3)


# ── logits_transform (positive-control damage) ──────────────────────


def test_logit_noise_is_applied_in_scoring_and_is_deterministic(
    tiny_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infer_check.backends.perturb import fixed_logit_noise

    monkeypatch.setattr(MLXBackend, "_format", lambda self, prompt: "")
    monkeypatch.setattr(MLXBackend, "_encode", lambda self, formatted: list(PROMPT_IDS))
    noisy = MLXBackend(model_id="tiny/random", logits_transform=fixed_logit_noise(sigma=1.0, seed=3))
    noisy._model, noisy._tokenizer = tiny_model, object()

    # Reference: the clean logits plus the same fixed bias, renormalised.
    logits = tiny_model(mx.array(PROMPT_IDS + TARGET_IDS)[None])[0].astype(mx.float32)
    bias = mx.random.normal((VOCAB,), key=mx.random.key(3)) * 1.0
    noisy_logits = logits + bias
    lp = noisy_logits - mx.logsumexp(noisy_logits, axis=-1, keepdims=True)
    start = len(PROMPT_IDS) - 1
    ref = np.array(lp[start : start + len(TARGET_IDS)])

    ids = np.array([[t] for t in TARGET_IDS])
    first = noisy._score(Prompt(text="x"), TARGET_IDS, top_k=0, at_ids=ids)
    second = noisy._score(Prompt(text="x"), TARGET_IDS, top_k=0, at_ids=ids)
    assert first.at_logprobs is not None and second.at_logprobs is not None
    np.testing.assert_allclose(first.at_logprobs[:, 0], ref[np.arange(len(TARGET_IDS)), TARGET_IDS], atol=1e-4)
    np.testing.assert_array_equal(first.at_logprobs, second.at_logprobs)


def test_zero_noise_is_identity(backend: MLXBackend, tiny_model: Any) -> None:
    from infer_check.backends.perturb import fixed_logit_noise

    clean = backend._score(Prompt(text="x"), TARGET_IDS, top_k=5, at_ids=None)
    backend._logits_transform = fixed_logit_noise(sigma=0.0)
    same = backend._score(Prompt(text="x"), TARGET_IDS, top_k=5, at_ids=None)
    np.testing.assert_array_equal(clean.top_ids, same.top_ids)
    np.testing.assert_allclose(clean.top_logprobs, same.top_logprobs)
