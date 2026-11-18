from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mmos.schemas.common import MMOSModel

INTEGRAL_BOUND_TOLERANCE = 1e-9


class _StrictModel(MMOSModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MILPVariableType(StrEnum):
    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"


class MILPSense(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintSense(StrEnum):
    LE = "le"
    GE = "ge"
    EQ = "eq"


class MILPSolverStatus(StrEnum):
    OPTIMAL = "optimal"
    FEASIBLE_NOT_PROVEN = "feasible_not_proven"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NO_SOLUTION = "no_solution"
    SOLVER_ERROR = "solver_error"


class MILPVariableSpec(_StrictModel):
    variable_id: str = Field(min_length=1)
    name: str = ""
    type: MILPVariableType
    lower_bound: float
    upper_bound: float

    @model_validator(mode="after")
    def _bounds(self) -> MILPVariableSpec:
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"variable {self.variable_id!r} has lower_bound > upper_bound")
        if self.type is MILPVariableType.BINARY:
            if self.lower_bound < 0.0 or self.upper_bound > 1.0:
                raise ValueError(
                    f"binary variable {self.variable_id!r} must lie inside [0, 1]"
                )
            for bound in (self.lower_bound, self.upper_bound):
                if abs(bound - round(bound)) > INTEGRAL_BOUND_TOLERANCE:
                    raise ValueError(
                        f"binary variable {self.variable_id!r} has a non-0/1 bound"
                    )
        if self.type is MILPVariableType.INTEGER:
            for bound in (self.lower_bound, self.upper_bound):
                if abs(bound - round(bound)) > INTEGRAL_BOUND_TOLERANCE:
                    raise ValueError(
                        f"integer variable {self.variable_id!r} has a non-integral bound"
                    )
        return self


class LinearTerm(_StrictModel):
    variable_id: str = Field(min_length=1)
    coefficient: float


class LinearExpression(_StrictModel):
    constant: float = 0.0
    terms: list[LinearTerm] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_terms(self) -> LinearExpression:
        ids = [term.variable_id for term in self.terms]
        if len(ids) != len(set(ids)):
            raise ValueError("linear expression repeats a variable_id")
        return self


class LinearObjectiveSpec(_StrictModel):
    sense: MILPSense
    expression: LinearExpression


class LinearConstraintSpec(_StrictModel):
    constraint_id: str = Field(min_length=1)
    expression: LinearExpression
    sense: ConstraintSense
    rhs: float


class MILPSolverConfig(_StrictModel):
    solver: Literal["cbc"] = "cbc"
    time_limit_seconds: float = Field(default=30.0, gt=0.0)
    relative_gap: float | None = Field(default=None, ge=0.0)
    feasibility_tolerance: float = Field(default=1e-6, gt=0.0)
    integrality_tolerance: float = Field(default=1e-6, gt=0.0)


class LinearMILPSpec(_StrictModel):
    model_name: str = Field(min_length=1)
    variables: list[MILPVariableSpec] = Field(min_length=1)
    objective: LinearObjectiveSpec
    constraints: list[LinearConstraintSpec] = Field(default_factory=list)
    solver_config: MILPSolverConfig = Field(default_factory=MILPSolverConfig)

    @model_validator(mode="after")
    def _references(self) -> LinearMILPSpec:
        variable_ids = [variable.variable_id for variable in self.variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("variable_id values must be unique")
        known = set(variable_ids)
        unknown = {term.variable_id for term in self.objective.expression.terms} - known
        if unknown:
            raise ValueError(f"objective references unknown variables: {sorted(unknown)}")
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint_id values must be unique")
        for constraint in self.constraints:
            missing = {term.variable_id for term in constraint.expression.terms} - known
            if missing:
                raise ValueError(
                    f"constraint {constraint.constraint_id!r} references unknown variables: "
                    f"{sorted(missing)}"
                )
        if not any(
            variable.type in (MILPVariableType.BINARY, MILPVariableType.INTEGER)
            for variable in self.variables
        ):
            raise ValueError(
                "a linear MILP must declare at least one binary or integer decision variable"
            )
        return self


class MILPConstraintActivity(_StrictModel):
    constraint_id: str
    lhs: float
    rhs: float
    sense: ConstraintSense
    slack: float
    satisfied: bool


class LinearMILPResult(_StrictModel):
    experiment_id: str
    model_name: str
    instance_hash: str
    variable_values: dict[str, float] = Field(default_factory=dict)
    objective_value: float | None = None
    objective_sense: MILPSense
    constraint_activities: list[MILPConstraintActivity] = Field(default_factory=list)
    solver_status: MILPSolverStatus
    termination_reason: str
    best_bound: float | None = None
    gap: float | None = None
    relative_gap_limit: float | None = None
    solver_name: str = "CBC"
    solver_version: str = "unavailable"
    solver_time_limit_seconds: float = 0.0
    solve_wall_seconds: float = 0.0
    feasibility_tolerance: float
    integrality_tolerance: float
    artifact_hash: str = ""


PolicyValue = bool | int | float | str


def policy_value_key(value: PolicyValue) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    return (3, value)


class DiscreteDecisionVariable(_StrictModel):
    variable_id: str = Field(min_length=1)
    domain: list[PolicyValue] = Field(min_length=1)

    @model_validator(mode="after")
    def _domain(self) -> DiscreteDecisionVariable:
        keys = [policy_value_key(value) for value in self.domain]
        if len(keys) != len(set(keys)):
            raise ValueError(f"decision {self.variable_id!r} has duplicate domain values")
        for value in self.domain:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"decision {self.variable_id!r} has a non-finite domain value")
            if isinstance(value, str) and not value:
                raise ValueError(f"decision {self.variable_id!r} has an empty string domain value")
        return self

    @property
    def canonical_domain(self) -> list[PolicyValue]:
        return sorted(self.domain, key=policy_value_key)


class FrozenParameterProvenance(_StrictModel):
    name: str = Field(min_length=1)
    value: float
    source_evidence_id: str = Field(min_length=1)
    source_artifact_hash: str = Field(min_length=1)
    source_experiment_id: str = Field(min_length=1)
    uncertainty_disclosure: Literal[
        "decision conditional on point estimates; uncertainty not propagated"
    ] = "decision conditional on point estimates; uncertainty not propagated"


class FinitePolicySpec(_StrictModel):
    model_name: str = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    parameters: dict[str, float] = Field(default_factory=dict)
    parameter_provenance: list[FrozenParameterProvenance] = Field(default_factory=list)
    point_estimate_conditioned: bool = False
    objective_sense: Literal["minimize"] = "minimize"
    max_enumerated_states: int = Field(gt=0)

    @model_validator(mode="after")
    def _provenance(self) -> FinitePolicySpec:
        names = [item.name for item in self.parameter_provenance]
        if len(names) != len(set(names)):
            raise ValueError("parameter_provenance repeats a parameter name")
        for item in self.parameter_provenance:
            if item.name not in self.parameters:
                raise ValueError(f"provenance names unknown parameter {item.name!r}")
            if abs(self.parameters[item.name] - item.value) > 1e-12:
                raise ValueError(
                    f"provenance value for {item.name!r} does not match the frozen parameter"
                )
        if self.parameter_provenance and not self.point_estimate_conditioned:
            raise ValueError(
                "provenance-backed parameters require point_estimate_conditioned=true"
            )
        if self.point_estimate_conditioned and not self.parameter_provenance:
            raise ValueError(
                "point_estimate_conditioned=true requires parameter provenance"
            )
        return self


class FinitePolicyExecutionStatus(StrEnum):
    COMPLETED = "completed"
    NOT_EXECUTED_STATE_SPACE_LIMIT = "not_executed_state_space_limit"


class FinitePolicyResult(_StrictModel):
    experiment_id: str
    model_name: str
    evaluator_id: str
    evaluator_version: str
    evaluator_hash: str
    parameter_hash: str
    provenance_hash: str
    parameters: dict[str, float]
    parameter_provenance: list[FrozenParameterProvenance] = Field(default_factory=list)
    point_estimate_conditioned: bool = False
    uncertainty_disclosure: str | None = None
    objective_sense: Literal["minimize"] = "minimize"
    decision_variables: list[DiscreteDecisionVariable]
    state_space_size: int
    max_enumerated_states: int
    evaluated_state_count: int
    feasible_policy_count: int
    selected_policy: dict[str, PolicyValue] | None = None
    objective_value: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    objective_components: dict[str, float] | None = None
    optimal_policy_count: int = 0
    tie_break_rule: Literal["canonical_lexicographic"] = "canonical_lexicographic"
    execution_status: FinitePolicyExecutionStatus
    artifact_hash: str = ""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonical_expression(expression: LinearExpression) -> dict[str, object]:
    return {
        "constant": expression.constant,
        "terms": [
            {"variable_id": term.variable_id, "coefficient": term.coefficient}
            for term in sorted(expression.terms, key=lambda item: item.variable_id)
        ],
    }


def linear_milp_canonical_payload(spec: LinearMILPSpec) -> dict[str, object]:
    return {
        "kind": "linear_milp.v1",
        "variables": [
            {
                "variable_id": variable.variable_id,
                "name": variable.name,
                "type": variable.type.value,
                "lower_bound": variable.lower_bound,
                "upper_bound": variable.upper_bound,
            }
            for variable in sorted(spec.variables, key=lambda item: item.variable_id)
        ],
        "objective": {
            "sense": spec.objective.sense.value,
            "expression": _canonical_expression(spec.objective.expression),
        },
        "constraints": [
            {
                "constraint_id": constraint.constraint_id,
                "sense": constraint.sense.value,
                "rhs": constraint.rhs,
                "expression": _canonical_expression(constraint.expression),
            }
            for constraint in sorted(
                spec.constraints, key=lambda item: item.constraint_id
            )
        ],
        "solver_config": {
            "solver": spec.solver_config.solver,
            "time_limit_seconds": spec.solver_config.time_limit_seconds,
            "relative_gap": spec.solver_config.relative_gap,
            "feasibility_tolerance": spec.solver_config.feasibility_tolerance,
            "integrality_tolerance": spec.solver_config.integrality_tolerance,
        },
    }


def linear_milp_instance_hash(spec: LinearMILPSpec) -> str:
    payload = linear_milp_canonical_payload(spec)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def policy_parameter_hash(
    evaluator_id: str, evaluator_version: str, parameters: dict[str, float]
) -> str:
    payload = {
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "parameters": {name: parameters[name] for name in sorted(parameters)},
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def policy_provenance_hash(provenance: list[FrozenParameterProvenance]) -> str:
    payload = [
        {
            "name": item.name,
            "value": item.value,
            "source_evidence_id": item.source_evidence_id,
            "source_artifact_hash": item.source_artifact_hash,
            "source_experiment_id": item.source_experiment_id,
            "uncertainty_disclosure": item.uncertainty_disclosure,
        }
        for item in sorted(provenance, key=lambda entry: entry.name)
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def policy_state_space_size(decision_variables: list[DiscreteDecisionVariable]) -> int:
    size = 1
    for decision in decision_variables:
        size *= len(decision.domain)
    return size


__all__ = [
    "INTEGRAL_BOUND_TOLERANCE",
    "ConstraintSense",
    "DiscreteDecisionVariable",
    "FinitePolicyExecutionStatus",
    "FinitePolicyResult",
    "FinitePolicySpec",
    "FrozenParameterProvenance",
    "LinearConstraintSpec",
    "LinearExpression",
    "LinearMILPResult",
    "LinearMILPSpec",
    "LinearObjectiveSpec",
    "LinearTerm",
    "MILPConstraintActivity",
    "MILPSense",
    "MILPSolverConfig",
    "MILPSolverStatus",
    "MILPVariableSpec",
    "MILPVariableType",
    "PolicyValue",
    "linear_milp_canonical_payload",
    "linear_milp_instance_hash",
    "policy_parameter_hash",
    "policy_provenance_hash",
    "policy_state_space_size",
    "policy_value_key",
]
