"""End-to-end invariants for the MLX backend against a real (small) model.

These download a ~300 MB model on first run, so they're marked ``slow`` and
excluded by default. Run them on Apple Silicon with:

    pytest -m slow tests/integration

Override the model with INFER_CHECK_TEST_MODEL=<mlx-community repo>.
"""

import asyncio
import os

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

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
