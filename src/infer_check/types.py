import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Prompt",
    "InferenceResult",
    "ComparisonResult",
    "SweepResult",
    "StressResult",
    "DeterminismResult",
    "CompareResult",
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


class Prompt(BaseInferModel):
    id: str = Field(default_factory=_generate_uuid)
    text: str
    category: str = "general"
    max_tokens: int = 256
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResult(BaseInferModel):
    prompt_id: str
    backend_name: str
    model_id: str
    quantization: str | None = None
    tokens: list[str]
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
    kl_divergence: float | None = None
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
    """Result of comparing two quantizations of the same model."""

    model_a: str  # label or repo ID for model A
    model_b: str  # label or repo ID for model B
    backend_a: str
    backend_b: str
    comparisons: list[ComparisonResult]
    flip_rate: float  # fraction of prompts where the "answer" changed
    mean_kl_divergence: float | None = None
    mean_text_similarity: float | None = None
    per_category_stats: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
