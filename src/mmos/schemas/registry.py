from enum import StrEnum

from pydantic import Field

from mmos.schemas.common import MMOSModel


class TaskType(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    CAUSAL = "causal"
    SENSITIVITY = "sensitivity"


class Interpretability(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComputeClass(StrEnum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ModelCapability(MMOSModel):
    model_id: str
    name: str
    task_types: list[TaskType] = Field(min_length=1)
    description: str
    required_packages: list[str] = Field(default_factory=list)
    supports_probabilistic: bool = False
    supports_exogenous: bool = False
    supports_constraints: bool = False
    supports_multivariate: bool = False
    interpretability: Interpretability = Interpretability.MEDIUM
    compute_class: ComputeClass = ComputeClass.SMALL
    recommended_validation: str
    known_failure_modes: list[str] = Field(default_factory=list)
    documentation: str = ""
    is_baseline: bool = False
