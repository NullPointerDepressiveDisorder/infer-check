import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AnswerType",
    "Prompt",
    "InferenceResult",
    "ComparisonResult",
    "SweepResult",
    "StressResult",
    "DeterminismResult",
    "CompareResult",
    "TokenScores",
]


class BaseInferModel(BaseModel):
    """Base model providing JSON serialization and save/load classmethods."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
    )

    def save(self, path: str | Path) -> None:
        """Save the model instance to a JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a model instance from a JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _generate_uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


AnswerType = Literal["integer", "decimal", "boolean", "choice", "free"]


class Prompt(BaseInferModel):
    id: str = Field(default_factory=_generate_uuid)
    text: str
    category: str = "general"
    max_tokens: int = 1024
    # How to grade the response against ``expected``. "free" prompts have no
    # answer key: they contribute KL/similarity but not correctness metrics.
    answer_type: AnswerType = "free"
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_answer_key(self) -> Self:
        if self.answer_type != "free" and not self.expected:
            raise ValueError(f"prompt '{self.id}': answer_type '{self.answer_type}' requires 'expected'")
        if self.answer_type == "free" and self.expected is not None:
            raise ValueError(f"prompt '{self.id}': 'expected' given but answer_type is 'free'")
        return self

    @property
    def is_graded(self) -> bool:
        return self.answer_type != "free"


class InferenceResult(BaseInferModel):
    """One generation from one backend.

    ``metadata`` conventions (set by every backend where the server exposes them):
      - ``finish_reason``: ``"stop"`` (end-of-message / stop sequence),
        ``"length"`` (hit max_tokens), another server-specific string, or ``None``
        if unknown.
      - ``prompt_tokens``: number of prompt tokens after chat templating, or ``None``.
    """

    prompt_id: str
    backend_name: str
    model_id: str
    quantization: str | None = None
    tokens: list[str]
    # Token IDs, when the backend exposes them. Prefer these over ``tokens`` for
    # comparisons: ``tokens`` are per-token display strings and can split
    # multi-byte characters.
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    distributions: list[list[float]] | None = None
    distribution_metadata: list[dict[str, int | str]] | None = None
    text: str
    latency_ms: float
    tokens_per_second: float | None = None
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonResult(BaseInferModel):
    baseline: InferenceResult
    test: InferenceResult
    # Mean per-token KL(baseline || test), in nats. How it was computed is in
    # ``kl_method``; values from different methods must not be averaged together.
    kl_divergence: float | None = None
    # "teacher_forced": both models scored the baseline's exact token sequence, so
    #   every position compares distributions over the same context.
    # "prefix": approximate fallback for backends that can't score; uses only the
    #   positions up to and including the first divergent token, where both
    #   generations still share a context.
    kl_method: Literal["teacher_forced", "prefix"] | None = None
    kl_max: float | None = None
    kl_p99: float | None = None
    kl_positions: int | None = None
    kl_per_token: list[float] | None = None
    # Fraction of positions where both models' most likely next token is the same.
    top1_agreement: float | None = None
    token_divergence_index: int | None = None
    text_similarity: float
    is_failure: bool
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SweepResult(BaseInferModel):
    model_id: str
    backend_name: str
    quantization_levels: list[str]
    comparisons: list[ComparisonResult]
    timestamp: datetime
    summary: dict[str, Any]


class StressResult(BaseInferModel):
    model_id: str
    backend_name: str
    concurrency_level: int
    results: list[InferenceResult]
    error_count: int
    output_consistency: float


class DeterminismResult(BaseInferModel):
    prompt_id: str
    model_id: str
    backend_name: str
    quantization: str | None = None
    num_runs: int
    identical_count: int
    divergence_positions: list[int]
    determinism_score: float


class CompareResult(BaseInferModel):
    """Result of comparing two inference configurations (e.g., models, backends, or quantizations)."""

    model_a: str  # label or repo ID for configuration A (e.g., model or run variant)
    model_b: str  # label or repo ID for configuration B (e.g., model or run variant)
    backend_a: str
    backend_b: str
    comparisons: list[ComparisonResult]
    flip_rate: float  # fraction of flip-eligible prompts where the answer changed
    # Paired correctness against answer keys (see analysis.scoring.paired_correctness);
    # None when no prompt in the suite has an answer key.
    correctness: dict[str, Any] | None = None
    mean_kl_divergence: float | None = None
    kl_method: Literal["teacher_forced", "prefix"] | None = None
    mean_top1_agreement: float | None = None
    mean_text_similarity: float | None = None
    per_category_stats: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TokenScores:
    """Teacher-forced next-token log-probs for a fixed target token sequence.

    Row ``i`` describes the distribution that predicts ``target_ids[i]`` given
    the prompt and ``target_ids[:i]``. All log-probs are normalised over the
    model's full vocabulary. In-memory only (holds numpy arrays; not serialised).
    """

    top_ids: npt.NDArray[np.int64]  # [T, K] token IDs, by descending log-prob
    top_logprobs: npt.NDArray[np.float32]  # [T, K]
    argmax: npt.NDArray[np.int64]  # [T] most likely token at each position
    # [T, M] log-probs at caller-requested token IDs (``at_ids`` passed to score()).
    at_logprobs: npt.NDArray[np.float32] | None = None

    def __len__(self) -> int:
        return int(self.argmax.shape[0])
