import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class ExperimentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"
    VALIDATED = "validated"


class VariableType(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class DecisionVariable(MMOSModel):
    name: str = Field(min_length=1)
    type: VariableType = VariableType.CONTINUOUS
    lower: float = 0.0
    upper: float = 1.0e12

    @model_validator(mode="after")
    def _bounds(self) -> "DecisionVariable":
        if self.lower > self.upper:
            raise ValueError(f"variable {self.name!r} has lower > upper")
        if self.type is VariableType.BINARY and (self.lower < 0 or self.upper > 1):
            raise ValueError(f"binary variable {self.name!r} must lie in [0, 1]")
        return self


class LinearConstraint(MMOSModel):
    name: str = Field(min_length=1)
    coefficients: dict[str, float] = Field(min_length=1)
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _bounds(self) -> "LinearConstraint":
        if self.lower is None and self.upper is None:
            raise ValueError(f"constraint {self.name!r} needs a lower or upper bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"constraint {self.name!r} has lower > upper")
        return self


class OptimizationModel(MMOSModel):
    name: str = Field(min_length=1)
    sense: Literal["minimize", "maximize"]
    variables: list[DecisionVariable] = Field(min_length=1)
    objective: dict[str, float]
    constraints: list[LinearConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references(self) -> "OptimizationModel":
        names = [variable.name for variable in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique")
        known = set(names)
        unknown = set(self.objective) - known
        if unknown:
            raise ValueError(f"objective references unknown variables: {sorted(unknown)}")
        for constraint in self.constraints:
            missing = set(constraint.coefficients) - known
            if missing:
                raise ValueError(
                    f"constraint {constraint.name!r} references unknown variables: "
                    f"{sorted(missing)}"
                )
        return self


def relative_improvement(
    sense: Literal["minimize", "maximize"], candidate: float, baseline: float
) -> float | None:
    """Positive means the candidate is better under the objective sense."""
    if baseline == 0:
        return None
    if sense == "maximize":
        return (candidate - baseline) / abs(baseline)
    return (baseline - candidate) / abs(baseline)


def optimization_instance_hash(model: OptimizationModel) -> str:
    payload = {
        "sense": model.sense,
        "variables": sorted(
            (
                {
                    "name": variable.name,
                    "type": variable.type.value,
                    "lower": variable.lower,
                    "upper": variable.upper,
                }
                for variable in model.variables
            ),
            key=lambda item: item["name"],
        ),
        "objective": {name: model.objective[name] for name in sorted(model.objective)},
        "constraints": sorted(
            (
                {
                    "name": constraint.name,
                    "coefficients": {
                        name: constraint.coefficients[name]
                        for name in sorted(constraint.coefficients)
                    },
                    "lower": constraint.lower,
                    "upper": constraint.upper,
                }
                for constraint in model.constraints
            ),
            key=lambda item: item["name"],
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class OptimizationSpec(MMOSModel):
    experiment_id: str
    problem_id: str
    model: OptimizationModel
    solver: str = "ortools.cbc"
    solver_parameters: dict[str, str] = Field(default_factory=dict)
    time_limit_seconds: float = Field(default=30.0, gt=0)
    baseline: Literal["greedy_feasible", "none"] = "greedy_feasible"
    instance_hash: str | None = None
    random_seed: int | None = None
    acceptance_criteria: list[str] = Field(default_factory=list)


class OptimizationSolution(MMOSModel):
    method: str
    status: Literal["optimal", "feasible", "infeasible", "unbounded", "unknown"]
    objective_value: float | None = None
    best_bound: float | None = None
    optimality_gap: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    feasible: bool
    constraint_violations: list[str] = Field(default_factory=list)
    runtime_seconds: float = 0.0
    solver: str = ""
    solver_version: str = "unavailable"
    termination_condition: str | None = None
    absolute_gap: float | None = None
    node_count: int | None = None


class EnumerationResult(MMOSModel):
    method: str = "exhaustive_enumeration"
    status: Literal["complete", "skipped"]
    evaluated_subsets: int = 0
    optimum_value: float | None = None
    optimal_subsets: int | None = None
    matches_solver_result: bool | None = None
    reason: str | None = None


class OptimizationResult(MMOSModel):
    experiment_id: str
    problem_id: str
    model_name: str
    sense: Literal["minimize", "maximize"]
    solutions: list[OptimizationSolution] = Field(min_length=1)
    selected_method: str | None = None
    enumeration: EnumerationResult | None = None
    instance_hash: str | None = None
    solver_backend: str = "unavailable"
    solver_version: str = "unavailable"
    generated_at: datetime


class OptimizationExperimentRecord(MMOSModel):
    experiment_id: str
    project_id: str
    task_family: Literal["optimization"] = "optimization"
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.CREATED
    instance_hash: str
    config_hash: str
    git_commit: str = "unknown"
    solver: str = ""
    solver_version: str = "unavailable"
    solver_backend: str = "unavailable"
    mlflow_run_id: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    validation_status: dict[str, str] = Field(default_factory=dict)
    linked_experiment_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
