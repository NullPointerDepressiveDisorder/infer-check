import asyncio

import pytest

from infer_check.runner import TestRunner
from infer_check.types import InferenceResult, Prompt


@pytest.fixture
def runner() -> TestRunner:
    return TestRunner(cache_dir=".test_cache")


def test_compare_identical(runner: TestRunner) -> None:
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["A", "B", "C"],
        text="A B C",
        latency_ms=10.0,
    )
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["A", "B", "C"],
        text="A B C",
        latency_ms=10.0,
    )

    comp = runner._compare(baseline, test_res, threshold=0.5)
    assert comp.is_failure is False
    assert comp.text_similarity == 1.0
    assert comp.metadata["severity"] == "identical"
    assert comp.token_divergence_index is None


def test_compare_minor_difference(runner: TestRunner) -> None:
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["The", "quick", "fox"],
        text="The quick fox",
        latency_ms=10.0,
    )
    # Make it very similar to hit >= 0.8
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["The", "quick", "red", "fox"],
        text="The quick red fox",
        latency_ms=10.0,
    )

    comp = runner._compare(baseline, test_res, threshold=0.5)
    assert comp.is_failure is False
    assert comp.text_similarity >= 0.8
    assert comp.text_similarity < 1.0
    assert comp.metadata["severity"] == "minor"
    assert comp.token_divergence_index == 2


def test_compare_severe_difference(runner: TestRunner) -> None:
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["A", "valid", "JSON", "response"],
        text="A valid JSON response",
        latency_ms=10.0,
    )
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["Completely", "broken"],
        text="Completely broken",
        latency_ms=10.0,
    )

    comp = runner._compare(baseline, test_res, threshold=0.5)
    assert comp.is_failure is True
    assert comp.text_similarity < 0.5
    assert comp.metadata["severity"] == "severe"
    assert comp.token_divergence_index == 0


def test_compare_threshold_edge_cases(runner: TestRunner) -> None:
    # Setting threshold high so even minor diffs fail
    baseline = InferenceResult(
        prompt_id="p1",
        backend_name="b",
        model_id="m",
        tokens=["A", "B", "C"],
        text="A B C",
        latency_ms=10.0,
    )
    test_res = InferenceResult(
        prompt_id="p1",
        backend_name="t",
        model_id="m",
        tokens=["A", "B", "D"],
        text="A B D",
        latency_ms=10.0,
    )

    # The threshold logic in the runner is: is_failure = text_similarity < threshold
    # So threshold is the minimum similarity to pass.
    # default 0.5 threshold -> passes as sim 0.8 >= 0.5
    comp_default = runner._compare(baseline, test_res, threshold=0.5)
    assert comp_default.is_failure is False

    # A minimum similarity of 0.9 (strict) means passing requires text_similarity >= 0.9
    # Since similarity is ~0.8, it should fail.
    comp_strict = runner._compare(baseline, test_res, threshold=0.9)
    assert comp_strict.is_failure is True


class StubBackend:
    def __init__(self, name: str, responses: dict[str, str]):
        self._name = name
        self._responses = responses
        self.cleanup_called = False

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, prompt: Prompt) -> InferenceResult:
        text = self._responses.get(prompt.id, "Default response")
        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self._name,
            model_id="stub-model",
            tokens=text.split(),
            text=text,
            latency_ms=1.0,
            metadata={},
        )

    async def generate_batch(self, prompts: list[Prompt]) -> list[InferenceResult]:
        return await asyncio.gather(*(self.generate(p) for p in prompts))

    async def health_check(self) -> bool:
        return True

    async def cleanup(self) -> None:
        self.cleanup_called = True


@pytest.mark.asyncio
async def test_diff_flip_detection(runner: TestRunner) -> None:
    # Setup prompts
    p1 = Prompt(id="p1", text="What is 2+2?", category="arithmetic")
    p2 = Prompt(id="p2", text="What is the capital of France?", category="general")
    prompts = [p1, p2]

    # Setup backends
    # p1: baseline says 4, test says 5 (FLIP)
    # p2: both say Paris (NO FLIP)
    baseline_backend = StubBackend("baseline", {"p1": "The answer is 4", "p2": "Paris"})
    test_backend = StubBackend("test", {"p1": "The answer is 5", "p2": "Paris"})

    # Run diff
    comparisons = await runner.diff(
        baseline_backend=baseline_backend,
        test_backends=[test_backend],
        prompts=prompts,
    )

    assert len(comparisons) == 2

    # Check p1 (flip)
    comp_p1 = next(c for c in comparisons if c.baseline.prompt_id == "p1")
    assert comp_p1.metadata["flipped"] is True
    assert comp_p1.metadata["answer_a"] == "4"
    assert comp_p1.metadata["answer_b"] == "5"
    assert "extraction_confidence" in comp_p1.metadata

    # Check p2 (no flip)
    comp_p2 = next(c for c in comparisons if c.baseline.prompt_id == "p2")
    assert comp_p2.metadata["flipped"] is False
    assert comp_p2.metadata["answer_a"].lower() == "paris"
    assert comp_p2.metadata["answer_b"].lower() == "paris"
    assert "extraction_confidence" in comp_p2.metadata


@pytest.mark.asyncio
async def test_diff_multiple_test_backends(runner: TestRunner) -> None:
    p1 = Prompt(id="p1", text="Test", category="general")
    prompts = [p1]

    baseline = StubBackend("baseline", {"p1": "A"})
    test1 = StubBackend("test1", {"p1": "B"})
    test2 = StubBackend("test2", {"p1": "A"})

    comparisons = await runner.diff(
        baseline_backend=baseline,
        test_backends=[test1, test2],
        prompts=prompts,
    )

    assert len(comparisons) == 2
    assert comparisons[0].test.backend_name == "test1"
    assert comparisons[0].metadata["flipped"] is True
    assert comparisons[1].test.backend_name == "test2"
    assert comparisons[1].metadata["flipped"] is False
