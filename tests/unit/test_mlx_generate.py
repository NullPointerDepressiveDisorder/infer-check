"""MLX backend generation logic, with ``mlx_lm.stream_generate`` faked.

Uses real ``mlx.core`` arrays (skipped where MLX isn't installed) but no model:
each test scripts the exact ``GenerationResponse`` sequence mlx-lm would yield,
including the extra final response that ``stream_generate`` always emits.
"""

import sys
import types
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from infer_check.backends.mlx_lm import MLXBackend  # noqa: E402
from infer_check.types import Prompt  # noqa: E402

VOCAB = 16
EOS = 2


@dataclass
class FakeResponse:
    """Mirrors the fields of mlx_lm.generate.GenerationResponse that the backend reads."""

    text: str
    token: int
    logprobs: Any
    finish_reason: str | None = None


def peaked(token: int) -> Any:
    """A normalised log-prob vector whose most likely token is ``token``."""
    logits = np.full(VOCAB, -4.0, dtype=np.float32)
    logits[token] = 2.0
    lp = logits - np.log(np.exp(logits).sum())
    return mx.array(lp)


def resp(text: str, token: int, finish: str | None = None) -> FakeResponse:
    return FakeResponse(text=text, token=token, logprobs=peaked(token), finish_reason=finish)


class FakeTokenizer:
    bos_token = "<s>"
    chat_template = None  # format_prompt returns the raw text

    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return ([1] if add_special_tokens else []) + [3 + (ord(c) % 10) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture
def fake_mlx_lm() -> Generator[dict[str, Any]]:
    """Install fake ``mlx_lm`` / ``mlx_lm.sample_utils`` modules.

    Tests put the response script in ``state["responses"]`` (or an exception in
    ``state["raises"]``); the fake records the arguments it was called with.
    """
    state: dict[str, Any] = {"responses": [], "raises": None, "calls": []}

    def stream_generate(model: Any, tokenizer: Any, prompt: Any, **kwargs: Any) -> Iterator[FakeResponse]:
        state["calls"].append({"prompt": prompt, **kwargs})
        if state["raises"] is not None:
            raise state["raises"]
        yield from state["responses"]

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = stream_generate  # type: ignore[attr-defined]
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kw: ("sampler", kw)  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"mlx_lm": mlx_lm, "mlx_lm.sample_utils": sample_utils}):
        yield state


@pytest.fixture
def backend() -> MLXBackend:
    be = MLXBackend(model_id="fake/model")
    be._model = object()  # skip _ensure_loaded's real load
    be._tokenizer = FakeTokenizer()
    return be


def run(backend: MLXBackend, prompt: Prompt) -> Any:
    import asyncio

    return asyncio.run(backend.generate(prompt))


# ── stop / length handling ──────────────────────────────────────────


def test_stop_excludes_eos_token_but_keeps_final_text(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    # stream_generate yields each token, then a final response whose token is the
    # EOS itself and whose text is the detokenizer's flush.
    fake_mlx_lm["responses"] = [resp("Hel", 5), resp("lo", 6), resp("!", EOS, "stop")]
    res = run(backend, Prompt(id="p", text="hi", max_tokens=50))

    assert res.text == "Hello!"
    assert res.token_ids == [5, 6]
    assert res.metadata["finish_reason"] == "stop"
    assert len(res.logprobs) == len(res.distributions) == len(res.tokens) == 2


def test_length_records_the_final_token(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    # On "length", the last real token is only delivered in the final response.
    fake_mlx_lm["responses"] = [resp("a", 5), resp("b", 6), resp("c", 7, "length")]
    res = run(backend, Prompt(id="p", text="hi", max_tokens=3))

    assert res.text == "abc"
    assert res.token_ids == [5, 6, 7]
    assert res.metadata["finish_reason"] == "length"
    assert len(res.logprobs) == 3


def test_immediate_eos_gives_empty_output(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["responses"] = [resp("", EOS, "stop")]
    res = run(backend, Prompt(id="p", text="hi", max_tokens=50))

    assert res.text == ""
    assert res.token_ids == []
    assert res.logprobs is None and res.distributions is None
    assert res.metadata["finish_reason"] == "stop"


def test_nothing_after_the_final_response_is_consumed(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    # Defensive: even if a generator kept yielding, we stop at the first finish_reason.
    fake_mlx_lm["responses"] = [resp("a", 5), resp("", EOS, "stop"), resp("junk", 9)]
    res = run(backend, Prompt(id="p", text="hi", max_tokens=50))
    assert res.text == "a"
    assert res.token_ids == [5]


# ── what gets passed to mlx-lm ──────────────────────────────────────


def test_passes_max_tokens_and_token_ids(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["responses"] = [resp("", EOS, "stop")]
    res = run(backend, Prompt(id="p", text="hi", max_tokens=777))

    call = fake_mlx_lm["calls"][0]
    assert call["max_tokens"] == 777  # the old generate_step path silently capped at 256
    assert isinstance(call["prompt"], list) and all(isinstance(t, int) for t in call["prompt"])
    assert res.metadata["prompt_tokens"] == len(call["prompt"])


def test_temperature_reaches_the_sampler(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["responses"] = [resp("", EOS, "stop")]
    run(backend, Prompt(id="p", text="hi", metadata={"temperature": 0.7}))
    assert fake_mlx_lm["calls"][0]["sampler"] == ("sampler", {"temp": 0.7})


# ── BOS handling ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("formatted", "expect_add_special"),
    [
        ("<s>templated prompt", False),  # template already emitted BOS: don't add another
        ("raw prompt", True),  # no template: let the tokenizer add BOS
    ],
)
def test_bos_added_exactly_once(
    backend: MLXBackend, fake_mlx_lm: dict[str, Any], formatted: str, expect_add_special: bool
) -> None:
    fake_mlx_lm["responses"] = [resp("", EOS, "stop")]
    with patch.object(MLXBackend, "_format", return_value=formatted):
        run(backend, Prompt(id="p", text="hi"))
    tok = backend._tokenizer
    assert tok.encode_calls == [(formatted, expect_add_special)]


# ── log-prob bookkeeping ────────────────────────────────────────────


def test_top_k_distributions_are_sorted_and_consistent(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["responses"] = [resp("a", 5), resp("", EOS, "stop")]
    res = run(backend, Prompt(id="p", text="hi", metadata={"top_k_logprobs": 4}))

    dist, ids = res.distributions[0], res.distribution_metadata[0]
    assert len(dist) == 4 and len(ids) == 4
    assert dist == sorted(dist, reverse=True)
    assert ids["id_0"] == 5  # most likely token is the one generated here
    assert res.logprobs[0] == pytest.approx(dist[0])
    assert np.exp(dist).sum() < 1.0 + 1e-6  # normalised over the full vocab


def test_top_k_clamped_to_vocab_and_zero_disables(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["responses"] = [resp("a", 5), resp("", EOS, "stop")]
    res = run(backend, Prompt(id="p", text="hi", metadata={"top_k_logprobs": 10_000}))
    assert len(res.distributions[0]) == VOCAB

    fake_mlx_lm["responses"] = [resp("a", 5), resp("", EOS, "stop")]
    res = run(backend, Prompt(id="p", text="hi", metadata={"top_k_logprobs": 0}))
    assert res.distributions is None and res.logprobs is not None


# ── failure behaviour ───────────────────────────────────────────────


def test_errors_propagate_without_fallback(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["raises"] = ValueError("kaboom")
    with pytest.raises(RuntimeError, match="MLX generation failed for prompt 'p42'") as exc_info:
        run(backend, Prompt(id="p42", text="hi"))
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert len(fake_mlx_lm["calls"]) == 1  # no second, different generation path


def test_health_check_uses_real_generation_path(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    import asyncio

    fake_mlx_lm["responses"] = [resp("h", 5, "length")]
    assert asyncio.run(backend.health_check()) is True
    assert fake_mlx_lm["calls"][0]["max_tokens"] == 1

    fake_mlx_lm["raises"] = RuntimeError("broken")
    assert asyncio.run(backend.health_check()) is False


def test_logits_transform_reaches_generation(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    seen: list[Any] = []
    backend._logits_transform = lambda logits: seen.append(logits) or logits
    fake_mlx_lm["responses"] = [resp("", EOS, "stop")]
    run(backend, Prompt(id="p", text="hi"))
    processors = fake_mlx_lm["calls"][0]["logits_processors"]
    assert len(processors) == 1
    marker = object()
    assert processors[0]([1, 2], marker) is marker and seen == [marker]


def test_no_logits_processors_by_default(backend: MLXBackend, fake_mlx_lm: dict[str, Any]) -> None:
    fake_mlx_lm["responses"] = [resp("", EOS, "stop")]
    run(backend, Prompt(id="p", text="hi"))
    assert "logits_processors" not in fake_mlx_lm["calls"][0]
