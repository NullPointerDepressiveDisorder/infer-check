"""Phase 5 cleanup: degradation cliff order, flip eligibility, --max-tokens notice,
backend validation, and the stress section of the HTML report."""

import json
from pathlib import Path
from typing import Any

import click
import pytest

from infer_check.runner import TestRunner, find_degradation_cliff
from infer_check.types import ComparisonResult, InferenceResult, Prompt, StressResult


def _table(regressions: int, improvements: int, p: float) -> dict[str, Any]:
    return {"regressions": regressions, "improvements": improvements, "mcnemar_p": p}


# ── degradation cliff ───────────────────────────────────────────────


def test_cliff_follows_the_given_order_not_alphabetical() -> None:
    corr = {
        "8bit": _table(2, 3, 0.9),
        "4bit": _table(11, 1, 0.006),
        "3bit": _table(19, 1, 4e-5),
    }
    # Alphabetically "3bit" < "4bit" < "8bit", which the old code would have walked.
    cliff = find_degradation_cliff(["bf16", "8bit", "4bit", "3bit"], "bf16", corr)
    assert cliff is not None and cliff["level"] == "4bit"
    assert cliff["regressions"] == 11 and cliff["mcnemar_p"] == 0.006


def test_no_cliff_without_significant_loss_or_answer_keys() -> None:
    assert find_degradation_cliff(["bf16", "8bit"], "bf16", {"8bit": _table(5, 1, 0.2)}) is None
    assert find_degradation_cliff(["bf16", "8bit"], "bf16", {"8bit": _table(1, 9, 0.02)}) is None  # improvement
    assert find_degradation_cliff(["bf16", "8bit"], "bf16", {"bf16": None, "8bit": None}) is None


# ── flip eligibility for prompts without answer keys ────────────────


def _res(text: str) -> InferenceResult:
    return InferenceResult(prompt_id="p", backend_name="b", model_id="m", tokens=[], text=text, latency_ms=1.0)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("The answer is 42", "The answer is 43", True),
        ("The answer is 42", "It is 42", False),
        ("The answer is 42", "I'm not sure.", None),  # no number on one side: nothing to compare
        ("No idea.", "I'm not sure.", None),
    ],
)
def test_free_prompt_flip_needs_answers_on_both_sides(a: str, b: str, expected: bool | None) -> None:
    comp = ComparisonResult(baseline=_res(a), test=_res(b), text_similarity=0.5, is_failure=False)
    TestRunner(cache_dir=".test_cache")._annotate_flip_metadata(comp, Prompt(text="?", category="arithmetic"))
    assert comp.metadata["flipped"] is expected


# ── --max-tokens notice ─────────────────────────────────────────────


def test_max_tokens_notice_when_prompts_set_their_own(capsys: pytest.CaptureFixture[str]) -> None:
    from infer_check.cli import _load_prompts, main

    ctx = click.Context(main)
    ctx.obj = {"max_tokens": 64, "num_prompts": 3, "max_tokens_explicit": True}
    prompts = _load_prompts(ctx, "gsm8k", None, None)
    # Prompt-level values keep precedence...
    assert all(p.max_tokens == 1024 for p in prompts)
    # ...and the user is told the flag changed nothing.
    out = capsys.readouterr().out
    assert "applied to 0 of 3 prompts" in out


def test_no_notice_when_flag_not_given(capsys: pytest.CaptureFixture[str]) -> None:
    from infer_check.cli import _load_prompts, main

    ctx = click.Context(main)
    ctx.obj = {"max_tokens": 1024, "num_prompts": 2, "max_tokens_explicit": False}
    _load_prompts(ctx, "gsm8k", None, None)
    assert "applied to" not in capsys.readouterr().out


# ── backend validation ──────────────────────────────────────────────


def test_unknown_backend_type_is_rejected() -> None:
    from infer_check.backends.base import get_backend_for_model

    with pytest.raises(ValueError, match="Unknown backend 'vllm-cuda'"):
        get_backend_for_model("org/model", backend_type="vllm-cuda")


# ── HTML stress section ─────────────────────────────────────────────


def test_report_stress_section_lists_divergences(tmp_path: Path) -> None:
    from infer_check.reporting.html import generate_report

    results = [
        StressResult(
            model_id="m",
            backend_name="openai-compat",
            concurrency_level=1,
            results=[],
            error_count=0,
            output_consistency=1.0,
            compared=24,
        ),
        StressResult(
            model_id="m",
            backend_name="openai-compat",
            concurrency_level=4,
            results=[],
            error_count=2,
            output_consistency=0.667,
            compared=24,
            divergences=[
                {
                    "prompt_id": "reasoning-005",
                    "trial": 0,
                    "first_divergence": 83,
                    "co_scheduled": ["reasoning-009", "reasoning-020"],
                }
            ],
            error_messages=["RuntimeError: Cannot connect to server"],
        ),
    ]
    (tmp_path / "stress_x.json").write_text(json.dumps([r.model_dump(mode="json") for r in results]))
    html = generate_report(tmp_path, tmp_path / "report.html").read_text()
    assert "Identical to solo" in html and "(reference)" in html
    assert "reasoning-005" in html and "first differing token 83" in html
    assert "Cannot connect to server" in html
