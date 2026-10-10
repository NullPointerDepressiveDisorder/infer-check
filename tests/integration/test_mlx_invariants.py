"""End-to-end invariants for the MLX backend against a real (small) model.

These download a ~300 MB model on first run, so they're marked ``slow`` and
excluded by default. Run them on Apple Silicon with:

    pytest -m slow tests/integration

Override the model with INFER_CHECK_TEST_MODEL=<mlx-community repo>.
"""

import asyncio
import os

import numpy as np
import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

from infer_check.analysis.divergence import teacher_forced_kl  # noqa: E402
from infer_check.backends.mlx_lm import MLXBackend  # noqa: E402
from infer_check.types import InferenceResult, Prompt  # noqa: E402

pytestmark = pytest.mark.slow

MODEL = os.environ.get("INFER_CHECK_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")


@pytest.fixture(scope="module")
def backend() -> MLXBackend:
    be = MLXBackend(MODEL)
    be._ensure_loaded()
    return be


def gen(backend: MLXBackend, text: str, max_tokens: int) -> InferenceResult:
    return asyncio.run(backend.generate(Prompt(id="t", text=text, max_tokens=max_tokens)))


def test_short_answer_stops_at_end_of_message(backend: MLXBackend) -> None:
    r = gen(backend, "Reply with exactly one word: hello.", 64)
    assert r.metadata["finish_reason"] == "stop"
    assert r.token_ids is not None and 0 < len(r.token_ids) < 64


def test_no_special_tokens_leak_into_text(backend: MLXBackend) -> None:
    r = gen(backend, "Say hello in one short sentence.", 64)
    for special in backend._tokenizer.all_special_tokens:
        assert special not in r.text, f"special token {special!r} leaked into output"


def test_streamed_text_matches_whole_sequence_decode(backend: MLXBackend) -> None:
    # Multi-byte characters are the case per-token decoding used to break.
    r = gen(backend, "Write these symbols separated by spaces: ÷ ∫ ≡ ∞ ∂ π", 64)
    assert r.token_ids is not None
    assert r.text.strip() == backend._tokenizer.decode(r.token_ids).strip()
    assert "\ufffd" not in r.text


def test_length_cap_respects_max_tokens(backend: MLXBackend) -> None:
    # The old generate_step path silently capped every output at 256 tokens.
    r = gen(backend, "Write the numbers from 1 to 500, one per line.", 300)
    assert r.token_ids is not None
    if r.metadata["finish_reason"] == "length":
        assert len(r.token_ids) == 300
    else:
        assert len(r.token_ids) < 300


def test_prompt_has_single_bos(backend: MLXBackend) -> None:
    ids = backend._encode(backend._format(Prompt(text="hi")))
    bos = getattr(backend._tokenizer, "bos_token_id", None)
    if bos is not None:
        assert not (len(ids) > 1 and ids[0] == bos and ids[1] == bos)


def test_greedy_generation_is_repeatable(backend: MLXBackend) -> None:
    a = gen(backend, "What is 12 * 13? Answer briefly.", 64)
    b = gen(backend, "What is 12 * 13? Answer briefly.", 64)
    assert a.token_ids == b.token_ids and a.text == b.text


# ── Teacher-forced scoring ──────────────────────────────────────────


def test_scoring_reproduces_greedy_generation(backend: MLXBackend) -> None:
    """Scoring the model's own greedy output must agree with how it was generated.

    Not bit-exact: one chunked forward pass and step-by-step decoding use
    different kernels, so near-ties can flip. That's why the runner only ever
    compares scored-vs-scored. But agreement must be near-total and log-probs
    close; an off-by-one alignment bug would wreck both.
    """
    prompt = Prompt(id="t", text="Explain in two sentences why the sky is blue.", max_tokens=96)
    r = asyncio.run(backend.generate(prompt))
    assert r.token_ids and r.logprobs
    ids = np.array(r.token_ids)
    s = asyncio.run(backend.score(prompt, r.token_ids, top_k=8, at_ids=ids[:, None]))

    assert len(s) == len(ids)
    assert np.mean(s.argmax == ids) >= 0.95
    assert s.at_logprobs is not None
    assert np.max(np.abs(s.at_logprobs[:, 0] - np.array(r.logprobs))) < 0.15


def test_rescoring_with_same_model_gives_zero_kl(backend: MLXBackend) -> None:
    prompt = Prompt(id="t", text="List three primary colors.", max_tokens=48)
    r = asyncio.run(backend.generate(prompt))
    assert r.token_ids
    ref = asyncio.run(backend.score(prompt, r.token_ids, top_k=64))
    again = asyncio.run(backend.score(prompt, r.token_ids, top_k=0, at_ids=ref.top_ids))
    stats = teacher_forced_kl(ref, again)
    assert stats is not None
    assert stats.mean == 0.0 and stats.top1_agreement == 1.0
