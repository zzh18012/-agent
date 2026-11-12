from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.metadata import InformationSet, TargetSummary
from mmos.schemas.registry import TaskType


class TemporalSplit(MMOSModel):
    strategy: Literal["rolling_origin"] = "rolling_origin"
    horizon: int = Field(ge=1)
    min_train_size: int = Field(ge=2)
    step: int = Field(ge=1)
    n_splits: int = Field(ge=1)

    @model_validator(mode="after")
    def _windows_do_not_overlap(self) -> "TemporalSplit":
        if self.step < self.horizon:
            raise ValueError("step must be >= horizon so test windows do not overlap")
        return self


class FoldWindow(MMOSModel):
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @model_validator(mode="after")
    def _time_order_is_preserved(self) -> "FoldWindow":
        if not (self.train_start < self.train_end <= self.test_start < self.test_end):
            raise ValueError("fold must satisfy train_start < train_end <= test_start < test_end")
        return self


class MaseFoldDetail(MMOSModel):
    fold_id: int
    train_start: int
    train_end: int
    scale_lag: int
    scale_denominator: float
    absolute_forecast_error: float
    scaled_error: float | None = None
    status: str = "defined"


class EvaluationRecord(MMOSModel):
    model_id: str
    n_observations: int
    sse: float | None = None
    sst: float | None = None
    r2: float | None = None
    r2_reason: str | None = None
    target_mean: float | None = None
    target_std_population: float | None = None
    rmse: float | None = None
    mae: float | None = None
    rmse_eval_std_ratio: float | None = None
    mase: float | None = None
    mase_details: list[MaseFoldDetail] = Field(default_factory=list)


class MetricSummary(MMOSModel):
    metric: str
    point_estimate: float
    fold_values: list[float] = Field(default_factory=list)
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    n: int = 0
    notes: str = ""


class ExperimentSpec(MMOSModel):
    """Everything needed to reproduce one experiment without asking an agent."""

    experiment_id: str
    problem_id: str
    proposal_id: str
    dataset_version: str | None = None
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_column: str = Field(min_length=1)
    feature_set: list[str] = Field(default_factory=list)
    split_strategy: str
    validation_strategy: str
    n_splits: int = Field(ge=2)
    models: list[str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    hyperparameter_space: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(min_length=1)
    random_seed: int
    required_artifacts: list[str] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    failure_criteria: list[str] = Field(min_length=1)
    compute_requirements: str
    expected_runtime_class: Literal["seconds", "minutes", "hours"]
    time_column: str | None = None
    seasonal_period: int | None = Field(default=None, ge=2)
    temporal_split: TemporalSplit | None = None


class ExperimentResult(MMOSModel):
    """Cross-validated metrics produced by a real run."""

    experiment_id: str
    dataset_hash: str
    task_type: TaskType
    feature_set: list[str]
    split_strategy: str
    validation_strategy: str
    random_seed: int
    n_splits: int
    models: dict[str, dict[str, float]]
    fold_metrics: dict[str, list[dict[str, float]]] = Field(default_factory=dict)
    metric_summaries: dict[str, dict[str, MetricSummary]] = Field(default_factory=dict)
    fold_windows: list[FoldWindow] = Field(default_factory=list)
    target_summary: TargetSummary | None = None
    model_determinism: dict[str, bool] = Field(default_factory=dict)
    feature_availability: dict[str, str] = Field(default_factory=dict)
    dataset_metadata_sha256: str | None = None
    evaluated_rows: int = 0
    evaluations: dict[str, EvaluationRecord] = Field(default_factory=dict)
    information_sets: dict[str, "InformationSet"] = Field(default_factory=dict)
    generated_at: datetime
