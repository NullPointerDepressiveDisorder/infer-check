"""Tests for the compare command — CLI, runner, and types."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from infer_check.cli import main
from infer_check.runner import TestRunner
from infer_check.types import (
    CompareResult,
    InferenceResult,
    Prompt,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def test_runner(tmp_path: Path) -> TestRunner:
    return TestRunner(cache_dir=str(tmp_path / ".cache"))


def _make_inference(
    prompt_id: str,
    backend: str,
    text: str,
    tokens: list[str] | None = None,
) -> InferenceResult:
    """Helper to create InferenceResult with minimal boilerplate."""
    return InferenceResult(
        prompt_id=prompt_id,
        backend_name=backend,
        model_id="test-model",
        tokens=tokens or text.split(),
        text=text,
        latency_ms=10.0,
    )


def _make_prompts(n: int = 3) -> list[Prompt]:
    return [Prompt(id=f"p{i}", text=f"Prompt {i}", category="general") for i in range(n)]


# ═════════════════════════════════════════════════════════════════════
# CLI tests
# ═════════════════════════════════════════════════════════════════════


class TestCompareCLI:
    def test_compare_appears_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "compare" in result.output

    def test_compare_subcommand_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "MODEL_A" in result.output
        assert "MODEL_B" in result.output
        assert "--prompts" in result.output
        assert "--label-a" in result.output
        assert "--report" in result.output

    def test_compare_end_to_end_mocked(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Full CLI flow with mocked backends."""
        suite = tmp_path / "suite.jsonl"
        suite.write_text('{"id":"p1","text":"hi","category":"general"}\n')

        mock_result = CompareResult(
            model_a="4bit",
            model_b="8bit",
            backend_a="mlx-lm",
            backend_b="mlx-lm",
            comparisons=[],
            flip_rate=0.0,
            mean_kl_divergence=None,
            mean_text_similarity=1.0,
            per_category_stats={},
        )

        with (
            patch("infer_check.backends.base.get_backend"),
            patch(
                "infer_check.runner.TestRunner.compare",
                return_value=mock_result,
            ),
        ):
            result = cli_runner.invoke(
                main,
                [
                    "compare",
                    "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    "mlx-community/Llama-3.1-8B-Instruct-8bit",
                    "--prompts",
                    str(suite),
                    "--output",
                    str(tmp_path / "out"),
                    "--no-report",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "compare" in result.output.lower()

    def test_compare_missing_model_args(self, cli_runner: CliRunner) -> None:
        """compare requires two positional args."""
        result = cli_runner.invoke(main, ["compare"])
        assert result.exit_code != 0

    def test_compare_resolves_backends(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Verify that model specs are resolved to correct backends."""
        suite = tmp_path / "suite.jsonl"
        suite.write_text('{"id":"p1","text":"hi","category":"general"}\n')

        configs_seen: list[MagicMock] = []

        # original_get_backend = None

        def capture_config(config: MagicMock) -> MagicMock:
            configs_seen.append(config)
            # Return a mock backend
            from unittest.mock import MagicMock

            mock = MagicMock()
            mock.name = config.backend_type
            return mock

        mock_result = CompareResult(
            model_a="a",
            model_b="b",
            backend_a="mlx-lm",
            backend_b="openai-compat",
            comparisons=[],
            flip_rate=0.0,
            mean_kl_divergence=None,
            mean_text_similarity=1.0,
            per_category_stats={},
        )

        with (
            patch(
                "infer_check.backends.base.get_backend",
                side_effect=capture_config,
            ),
            patch(
                "infer_check.runner.TestRunner.compare",
                return_value=mock_result,
            ),
        ):
            cli_runner.invoke(
                main,
                [
                    "compare",
                    "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    "ollama:llama3.1:8b-q4",
                    "--prompts",
                    str(suite),
                    "--output",
                    str(tmp_path / "out"),
                    "--no-report",
                ],
            )

        assert len(configs_seen) == 2
        assert configs_seen[0].backend_type == "mlx-lm"
        assert configs_seen[1].backend_type == "openai-compat"


# ═════════════════════════════════════════════════════════════════════
# Runner tests
# ═════════════════════════════════════════════════════════════════════


class TestCompareRunner:
    def _make_mock_backend(
        self,
        name: str,
        responses: dict[str, str],
    ) -> MagicMock:
        """Create a mock backend that returns fixed text per prompt_id."""
        from unittest.mock import MagicMock

        backend = MagicMock()
        backend.name = name

        async def _generate(prompt: Prompt) -> InferenceResult:
            text = responses.get(prompt.id, "default response")
            return _make_inference(prompt.id, name, text)

        backend.generate = _generate
        backend.cleanup = AsyncMock()
        return backend

    def test_identical_outputs(self, test_runner: TestRunner) -> None:
        prompts = _make_prompts(3)
        responses = {p.id: f"Answer to {p.id}" for p in prompts}

        backend_a = self._make_mock_backend("mlx-a", responses)
        backend_b = self._make_mock_backend("mlx-b", responses)

        result = asyncio.run(
            test_runner.compare(
                backend_a=backend_a,
                backend_b=backend_b,
                prompts=prompts,
                label_a="4bit",
                label_b="8bit",
            )
        )

        assert isinstance(result, CompareResult)
        assert result.model_a == "4bit"
        assert result.model_b == "8bit"
        assert len(result.comparisons) == 3
        assert result.flip_rate == 0.0
        assert result.mean_text_similarity == 1.0
        for c in result.comparisons:
            assert c.metadata["severity"] == "identical"
            assert c.baseline.metadata["prompt_text"] is not None
            assert c.baseline.metadata["prompt_category"] is not None
            assert c.test.metadata["prompt_text"] is not None
            assert c.test.metadata["prompt_category"] is not None

    def test_severe_divergence_flip_rate(self, test_runner: TestRunner) -> None:
        """When outputs differ severely, flip_rate reflects it."""
        prompts = _make_prompts(4)

        # Model A gives consistent answers
        resp_a = {p.id: f"The answer is {i}" for i, p in enumerate(prompts)}
        # Model B diverges severely on half
        resp_b = {
            prompts[0].id: "The answer is 0",  # identical
            prompts[1].id: "The answer is 1",  # identical
            prompts[2].id: "XXXXX",  # severe
            prompts[3].id: "YYYYY",  # severe
        }

        backend_a = self._make_mock_backend("a", resp_a)
        backend_b = self._make_mock_backend("b", resp_b)

        result = asyncio.run(
            test_runner.compare(
                backend_a=backend_a,
                backend_b=backend_b,
                prompts=prompts,
            )
        )

        assert len(result.comparisons) == 4
        # 2 out of 4 should be severe
        assert result.flip_rate == pytest.approx(0.5)
        assert result.mean_text_similarity is not None
        assert result.mean_text_similarity < 1.0

    def test_per_category_stats(self, test_runner: TestRunner) -> None:
        """Per-category breakdown is computed correctly."""
        prompts = [
            Prompt(id="r1", text="Reason 1", category="reasoning"),
            Prompt(id="r2", text="Reason 2", category="reasoning"),
            Prompt(id="c1", text="Code 1", category="code"),
        ]
        resp_a = {"r1": "Yes", "r2": "Yes", "c1": "print('hi')"}
        resp_b = {"r1": "Yes", "r2": "NOPE", "c1": "print('hi')"}

        backend_a = self._make_mock_backend("a", resp_a)
        backend_b = self._make_mock_backend("b", resp_b)

        result = asyncio.run(
            test_runner.compare(
                backend_a=backend_a,
                backend_b=backend_b,
                prompts=prompts,
            )
        )

        assert "reasoning" in result.per_category_stats
        assert "code" in result.per_category_stats
        assert result.per_category_stats["reasoning"]["count"] == 2
        assert result.per_category_stats["code"]["count"] == 1
        # code category should be identical
        assert result.per_category_stats["code"]["mean_similarity"] == 1.0

    def test_backend_errors_skipped_gracefully(self, test_runner: TestRunner) -> None:
        """If one backend errors on a prompt, that prompt is skipped."""
        prompts = _make_prompts(2)
        from unittest.mock import MagicMock

        backend_a = MagicMock()
        backend_a.name = "a"
        call_count = 0

        async def _gen_a(prompt: Prompt) -> InferenceResult:
            nonlocal call_count
            call_count += 1
            if prompt.id == "p0":
                raise RuntimeError("GPU OOM")
            return _make_inference(prompt.id, "a", "ok")

        backend_a.generate = _gen_a
        backend_a.cleanup = AsyncMock()

        resp_b = {p.id: "ok" for p in prompts}
        backend_b = self._make_mock_backend("b", resp_b)

        result = asyncio.run(
            test_runner.compare(
                backend_a=backend_a,
                backend_b=backend_b,
                prompts=prompts,
            )
        )

        # p0 failed on backend_a so only p1 has both sides
        assert len(result.comparisons) == 1
        assert result.comparisons[0].baseline.prompt_id == "p1"

    def test_cleanup_called(self, test_runner: TestRunner) -> None:
        """Both backends get cleanup() called."""
        prompts = _make_prompts(1)
        resp = {prompts[0].id: "ok"}

        backend_a = self._make_mock_backend("a", resp)
        backend_b = self._make_mock_backend("b", resp)

        asyncio.run(
            test_runner.compare(
                backend_a=backend_a,
                backend_b=backend_b,
                prompts=prompts,
            )
        )

        backend_a.cleanup.assert_awaited_once()
        backend_b.cleanup.assert_awaited_once()

    def test_checkpoint_written(self, test_runner: TestRunner, tmp_path: Path) -> None:
        """A checkpoint file is written to cache_dir."""
        prompts = _make_prompts(1)
        resp = {prompts[0].id: "ok"}

        backend_a = self._make_mock_backend("a", resp)
        backend_b = self._make_mock_backend("b", resp)

        # Run a compare so that a checkpoint should be written.
        asyncio.run(
            test_runner.compare(
                backend_a=backend_a,
                backend_b=backend_b,
                prompts=prompts,
            )
        )

        cache_dir = Path(test_runner.cache_dir)
        assert cache_dir.exists() and cache_dir.is_dir()

        checkpoint_files = list(cache_dir.glob("compare_*.json"))
        assert checkpoint_files, "Expected at least one compare_*.json checkpoint file to be written"
