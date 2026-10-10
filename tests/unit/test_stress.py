"""Stress test: outputs must not depend on what else is in flight."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from infer_check.backends.base import BackendAdapter
from infer_check.runner import TestRunner
from infer_check.types import InferenceResult, Prompt, TokenScores

PROMPTS = [Prompt(id=f"p{i}", text=f"prompt {i}") for i in range(6)]


class FakeServer(BackendAdapter):
    """Answers concurrently. Prompts in ``fragile`` come out differently whenever
    other requests are in flight, like a batch-variant kernel would cause."""

    def __init__(self, fragile: set[str] | None = None, concurrent: bool = True) -> None:
        self.fragile = fragile or set()
        self._concurrent = concurrent
        self.in_flight = 0
        self.max_seen = 0
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "fake-server"

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def supports_concurrency(self) -> bool:
        return self._concurrent

    @property
    def supports_scoring(self) -> bool:
        return False

    async def score(self, prompt: Prompt, target_ids: list[int], *, top_k: int = 64, at_ids: Any = None) -> TokenScores:
        raise NotImplementedError

    async def generate(self, prompt: Prompt) -> InferenceResult:
        self.calls.append(prompt.id)
        self.in_flight += 1
        await asyncio.sleep(0.001)  # let every request in the batch start
        seen = self.in_flight
        self.max_seen = max(self.max_seen, seen)
        self.in_flight -= 1
        ids = [1, 2, 3, 4]
        if prompt.id in self.fragile and seen > 1:
            ids = [1, 2, 9, 9]  # a token flips at position 2
        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self.model_id,
            tokens=[str(i) for i in ids],
            token_ids=ids,
            text=" ".join(str(i) for i in ids),
            latency_ms=1.0,
        )

    async def health_check(self) -> bool:
        return True

    async def cleanup(self) -> None:
        pass


@pytest.fixture
def runner(tmp_path: Path) -> TestRunner:
    return TestRunner(cache_dir=tmp_path)


def test_refuses_backends_without_real_concurrency(runner: TestRunner) -> None:
    with pytest.raises(ValueError, match="one at a time"):
        asyncio.run(runner.stress(FakeServer(concurrent=False), PROMPTS))


def test_stable_server_is_fully_consistent(runner: TestRunner) -> None:
    server = FakeServer()
    results = asyncio.run(runner.stress(server, PROMPTS, [1, 2, 4]))
    assert [r.concurrency_level for r in results] == [1, 2, 4]
    assert all(r.output_consistency == 1.0 and not r.divergences for r in results)
    assert server.max_seen == 4  # requests really were concurrent


def test_detects_batch_dependent_outputs(runner: TestRunner) -> None:
    server = FakeServer(fragile={"p1", "p4"})
    results = asyncio.run(runner.stress(server, PROMPTS, [1, 4]))
    solo, batched = results
    # Solo runs are the reference: fragile prompts are fine on their own.
    assert solo.concurrency_level == 1 and solo.compared == len(PROMPTS)
    assert batched.compared == len(PROMPTS)
    assert batched.output_consistency == pytest.approx(4 / 6)
    assert {d["prompt_id"] for d in batched.divergences} == {"p1", "p4"}
    for d in batched.divergences:
        assert d["first_divergence"] == 2
        assert len(d["co_scheduled"]) == 3 and d["prompt_id"] not in d["co_scheduled"]
    # Only the targets' outputs are kept, not the fillers'.
    assert len(batched.results) == len(PROMPTS)


def test_level_one_always_runs_first(runner: TestRunner) -> None:
    results = asyncio.run(runner.stress(FakeServer(), PROMPTS, [8, 2]))
    assert [r.concurrency_level for r in results] == [1, 2, 8]


def test_fillers_are_reproducible_with_a_seed(runner: TestRunner) -> None:
    def co_scheduled(seed: int) -> list[list[str]]:
        res = asyncio.run(runner.stress(FakeServer(fragile={p.id for p in PROMPTS}), PROMPTS, [3], seed=seed))
        return [d["co_scheduled"] for d in res[1].divergences]

    assert co_scheduled(7) == co_scheduled(7)
    assert co_scheduled(7) != co_scheduled(8)


def test_trials_draw_new_batches(runner: TestRunner) -> None:
    server = FakeServer(fragile={"p0"})
    res = asyncio.run(runner.stress(server, PROMPTS, [2], trials=3))
    assert res[1].trials == 3
    assert res[1].compared == 3 * len(PROMPTS)
    assert [d["trial"] for d in res[1].divergences] == [0, 1, 2]


def test_more_in_flight_than_prompts_samples_with_replacement(runner: TestRunner) -> None:
    two = PROMPTS[:2]
    res = asyncio.run(runner.stress(FakeServer(), two, [5]))
    assert res[1].compared == 2


def test_cli_refuses_in_process_backend() -> None:
    from click.testing import CliRunner

    from infer_check.cli import main

    result = CliRunner().invoke(
        main, ["stress", "--model", "mlx-community/x-4bit", "--backend", "mlx-lm", "--prompts", "reasoning"]
    )
    assert result.exit_code != 0
    assert "one at a time" in result.output


def test_error_messages_are_recorded(runner: TestRunner) -> None:
    class Flaky(FakeServer):
        async def generate(self, prompt: Prompt) -> InferenceResult:
            if prompt.id == "p5":
                raise TimeoutError("server took too long")
            return await super().generate(prompt)

    res = asyncio.run(runner.stress(Flaky(), PROMPTS, [1, 3]))
    assert res[0].error_count == 1 and res[0].error_messages == ["TimeoutError: server took too long"]
    assert res[1].error_messages == ["TimeoutError: server took too long"]  # deduplicated
