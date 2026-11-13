from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


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


class OptimizationSpec(MMOSModel):
    experiment_id: str
    problem_id: str
    model: OptimizationModel
    solver: str = "ortools.cbc"
    time_limit_seconds: float = Field(default=30.0, gt=0)
    baseline: Literal["greedy_feasible"] = "greedy_feasible"


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


class EnumerationResult(MMOSModel):
    status: Literal["complete", "skipped"]
    evaluated_subsets: int = 0
    optimum_value: float | None = None
    optimal_subsets: int | None = None
    reason: str | None = None


class OptimizationResult(MMOSModel):
    experiment_id: str
    problem_id: str
    model_name: str
    sense: Literal["minimize", "maximize"]
    solutions: list[OptimizationSolution] = Field(min_length=1)
    selected_method: str | None = None
    enumeration: EnumerationResult | None = None
    generated_at: datetime
