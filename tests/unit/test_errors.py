"""Run errors: recorded and reported, never dropped; systemic ones stop the run."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from infer_check.errors import ErrorLog, RunAbortedError, describe, systemic_reason
from infer_check.runner import TestRunner
from infer_check.types import InferenceResult, Prompt, RunError

METAL_OOM = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


def _wrapped(inner: BaseException, outer: str = "MLX generation failed for prompt 'p0'") -> RuntimeError:
    """An error the way backends raise it: ``raise RuntimeError(...) from inner``."""
    try:
        try:
            raise inner
        except BaseException as exc:
            raise RuntimeError(outer) from exc
    except RuntimeError as wrapped:
        return wrapped
    raise AssertionError("unreachable")


# ── classification ─────────────────────────────────────────────────────


def test_metal_out_of_memory_is_systemic_through_the_wrapper() -> None:
    assert systemic_reason(_wrapped(RuntimeError(METAL_OOM))) == "out of GPU memory"


def test_refused_connection_is_systemic() -> None:
    exc = _wrapped(httpx.ConnectError("Connection refused"), "Cannot connect to server")
    assert systemic_reason(exc) == "the server refused the connection"


@pytest.mark.parametrize(
    "exc",
    [
        _wrapped(httpx.ReadTimeout("timed out"), "Request timed out after 120s."),
        ValueError("could not parse the response"),
        RuntimeError("Server returned HTTP 400: bad request"),
    ],
)
def test_other_errors_are_isolated(exc: BaseException) -> None:
    assert systemic_reason(exc) is None


def test_describe_names_the_root_cause() -> None:
    assert describe(_wrapped(RuntimeError(METAL_OOM))) == f"RuntimeError: {METAL_OOM}"


# ── stopping rule ──────────────────────────────────────────────────────


def test_isolated_errors_are_recorded_and_the_run_continues() -> None:
    log = ErrorLog()
    log.record("4bit", "p0", ValueError("bad"))
    log.ok()
    log.record("4bit", "p3", ValueError("bad"), stage="score")
    assert [(e.label, e.prompt_id, e.stage) for e in log.errors] == [
        ("4bit", "p0", "generate"),
        ("4bit", "p3", "score"),
    ]
    assert log.counts() == {"4bit": 2}


def test_errors_in_a_row_stop_the_run() -> None:
    log = ErrorLog(max_consecutive=3)
    log.record("2bit", "p0", ValueError("bad"))
    log.record("2bit", "p1", ValueError("bad"))
    with pytest.raises(RunAbortedError, match="3 errors in a row"):
        log.record("2bit", "p2", ValueError("bad"))
    assert len(log.errors) == 3  # the last one is recorded before stopping


def test_a_success_resets_the_streak() -> None:
    log = ErrorLog(max_consecutive=2)
    for i in range(5):
        log.record("2bit", f"p{i}", ValueError("bad"))
        log.ok()
    assert len(log.errors) == 5


def test_out_of_memory_stops_at_once() -> None:
    log = ErrorLog()
    with pytest.raises(RunAbortedError, match="out of GPU memory"):
        log.record("b", "p0", _wrapped(RuntimeError(METAL_OOM)))
    assert len(log.errors) == 1


def test_stress_errors_do_not_count_toward_the_streak_but_systemic_still_stops() -> None:
    log = ErrorLog(max_consecutive=2)
    for i in range(5):
        log.record("8 in flight", f"p{i}", TimeoutError("slow"), streak=False)
    assert len(log.errors) == 5
    with pytest.raises(RunAbortedError):
        log.record("8 in flight", "p5", _wrapped(httpx.ConnectError("refused")), streak=False)


# ── runner ─────────────────────────────────────────────────────────────


def _result(prompt_id: str, backend: str) -> InferenceResult:
    return InferenceResult(
        prompt_id=prompt_id, backend_name=backend, model_id="m", tokens=["ok"], text="ok", latency_ms=1.0
    )


def _backend(name: str, fail: dict[str, BaseException] | None = None) -> MagicMock:
    backend = MagicMock()
    backend.name = name
    backend.supports_scoring = False

    async def _generate(prompt: Prompt) -> InferenceResult:
        if fail and prompt.id in fail:
            raise fail[prompt.id]
        return _result(prompt.id, name)

    backend.generate = _generate
    backend.cleanup = AsyncMock()
    return backend


PROMPTS = [Prompt(id=f"p{i}", text=f"Prompt {i}", category="general") for i in range(3)]


def test_compare_reports_a_failed_prompt_instead_of_dropping_it(tmp_path: Path) -> None:
    runner = TestRunner(cache_dir=tmp_path)
    b = _backend("b", fail={"p1": RuntimeError("bad output")})
    result = asyncio.run(runner.compare(_backend("a"), b, PROMPTS, label_a="a", label_b="b"))
    assert [c.baseline.prompt_id for c in result.comparisons] == ["p0", "p2"]
    assert result.errors == [RunError(label="b", prompt_id="p1", stage="generate", error="RuntimeError: bad output")]


def test_compare_stops_when_the_gpu_runs_out_of_memory(tmp_path: Path) -> None:
    runner = TestRunner(cache_dir=tmp_path)
    b = _backend("b", fail={"p0": _wrapped(RuntimeError(METAL_OOM))})
    with pytest.raises(RunAbortedError, match="out of GPU memory"):
        asyncio.run(runner.compare(_backend("a"), b, PROMPTS, label_a="a", label_b="b"))


def test_each_run_starts_with_an_empty_log(tmp_path: Path) -> None:
    runner = TestRunner(cache_dir=tmp_path)
    asyncio.run(runner.compare(_backend("a"), _backend("b", fail={"p1": ValueError("x")}), PROMPTS))
    result = asyncio.run(runner.compare(_backend("a"), _backend("b"), PROMPTS))
    assert result.errors == [] and runner.errors.errors == []


def test_sweep_summary_counts_errors_per_level(tmp_path: Path) -> None:
    runner = TestRunner(cache_dir=tmp_path)
    levels = {"bf16": _backend("mlx"), "4bit": _backend("mlx", fail={"p2": ValueError("x")})}
    result = asyncio.run(runner.sweep(levels, PROMPTS, ["bf16", "4bit"], model_id="m"))
    assert result.summary["errors"] == {"4bit": 1}
    assert [(e.label, e.prompt_id) for e in result.errors] == [("4bit", "p2")]


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_exits_1_when_prompts_failed_unless_allowed() -> None:
    from infer_check.cli import _report_errors

    errors = [RunError(label="4bit", prompt_id="p1", stage="generate", error="ValueError: x")]
    _report_errors([], allow_failures=False)  # nothing to report
    _report_errors(errors, allow_failures=True)  # listed, but accepted
    with pytest.raises(SystemExit) as exit_info:
        _report_errors(errors, allow_failures=False)
    assert exit_info.value.code == 1


def test_cli_exits_1_when_a_run_is_stopped() -> None:
    from infer_check.cli import _run

    async def stopped() -> None:
        raise RunAbortedError("Run stopped: out of GPU memory.")

    with pytest.raises(SystemExit) as exit_info:
        _run(stopped())
    assert exit_info.value.code == 1
