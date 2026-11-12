from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel
from mmos.schemas.registry import TaskType


class ExperimentSpec(MMOSModel):
    """Everything needed to reproduce one experiment without asking an agent."""

    experiment_id: str
    problem_id: str
    proposal_id: str
    dataset_version: str | None = None
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_column: str = Field(min_length=1)
    feature_set: list[str] = Field(min_length=1)
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
    generated_at: datetime
