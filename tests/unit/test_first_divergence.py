"""first_divergence: token-ID based comparison with a string fallback."""

from infer_check.runner import first_divergence
from infer_check.types import InferenceResult


def _res(tokens: list[str], ids: list[int] | None = None) -> InferenceResult:
    return InferenceResult(
        prompt_id="p",
        backend_name="b",
        model_id="m",
        tokens=tokens,
        token_ids=ids,
        text="".join(tokens),
        latency_ms=1.0,
    )


def test_identical_ids_are_not_divergent_even_if_display_strings_differ() -> None:
    # Per-token decoding can mangle a multi-byte char into "\ufffd" pieces on one
    # side; the token IDs are what actually matter.
    a = _res(["x", "\ufffd", "\ufffd"], [10, 20, 21])
    b = _res(["x", "÷", ""], [10, 20, 21])
    assert first_divergence(a, b) is None


def test_first_differing_id() -> None:
    assert first_divergence(_res(["a", "b", "c"], [1, 2, 3]), _res(["a", "b", "d"], [1, 2, 4])) == 2


def test_prefix_diverges_at_shorter_length() -> None:
    assert first_divergence(_res(["a", "b"], [1, 2]), _res(["a", "b", "c"], [1, 2, 3])) == 2


def test_falls_back_to_strings_when_either_side_lacks_ids() -> None:
    assert first_divergence(_res(["a", "b"], [1, 2]), _res(["a", "c"])) == 1
    assert first_divergence(_res(["a", "b"]), _res(["a", "b"])) is None
