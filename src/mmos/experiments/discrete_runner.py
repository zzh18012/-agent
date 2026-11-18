from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

from ortools.linear_solver import pywraplp
from pydantic import BaseModel

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.experiments.policy_evaluators import (
    get_policy_evaluator,
    validate_parameters,
)
from mmos.schemas.discrete import (
    ConstraintSense,
    DiscreteDecisionVariable,
    FinitePolicyExecutionStatus,
    FinitePolicyResult,
    FinitePolicySpec,
    LinearConstraintSpec,
    LinearExpression,
    LinearMILPResult,
    LinearMILPSpec,
    MILPConstraintActivity,
    MILPSense,
    MILPSolverStatus,
    MILPVariableType,
    PolicyValue,
    linear_milp_canonical_payload,
    linear_milp_instance_hash,
    policy_parameter_hash,
    policy_provenance_hash,
    policy_state_space_size,
)

RESULT_EXCLUDE = ("artifact_hash",)


class DiscreteRunError(RuntimeError):
    pass


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_hashed_result[T: BaseModel](
    paths: ProjectPaths, experiment_id: str, result: T, filename: str
) -> tuple[Path, T]:
    directory = paths.experiment_dir(experiment_id)
    target = directory / "artifacts" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    for key in RESULT_EXCLUDE:
        payload.pop(key, None)
    artifact_hash = _canonical_hash(payload)
    result = result.model_copy(update={"artifact_hash": artifact_hash})
    atomic_write_text(target, result.model_dump_json(indent=2) + "\n")
    return target, result


def _activity(expression: LinearExpression, values: dict[str, float]) -> float:
    return expression.constant + sum(
        term.coefficient * values[term.variable_id] for term in expression.terms
    )


def _slack(constraint: LinearConstraintSpec, lhs: float) -> float:
    if constraint.sense is ConstraintSense.LE:
        return constraint.rhs - lhs
    if constraint.sense is ConstraintSense.GE:
        return lhs - constraint.rhs
    return -abs(lhs - constraint.rhs)


def _satisfied(
    constraint: LinearConstraintSpec, lhs: float, tolerance: float
) -> bool:
    if constraint.sense is ConstraintSense.LE:
        return lhs <= constraint.rhs + tolerance
    if constraint.sense is ConstraintSense.GE:
        return lhs >= constraint.rhs - tolerance
    return abs(lhs - constraint.rhs) <= tolerance


def _incumbent_is_valid(
    spec: LinearMILPSpec, values: dict[str, float]
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    tolerance = spec.solver_config.feasibility_tolerance
    integrality = spec.solver_config.integrality_tolerance
    for variable in spec.variables:
        value = values.get(variable.variable_id)
        if value is None or not math.isfinite(value):
            issues.append(f"{variable.variable_id}: missing or non-finite value")
            continue
        if value < variable.lower_bound - tolerance or value > variable.upper_bound + tolerance:
            issues.append(f"{variable.variable_id}: outside declared bounds")
        if variable.type is MILPVariableType.INTEGER:
            if abs(value - round(value)) > integrality:
                issues.append(f"{variable.variable_id}: not integral")
        elif variable.type is MILPVariableType.BINARY:
            if abs(value - round(value)) > integrality:
                issues.append(f"{variable.variable_id}: binary value is not integral")
            elif round(value) not in (0, 1):
                issues.append(f"{variable.variable_id}: binary value is not 0 or 1")
    for constraint in spec.constraints:
        lhs = _activity(constraint.expression, values)
        if not _satisfied(constraint, lhs, tolerance):
            issues.append(f"{constraint.constraint_id}: constraint violated")
    return not issues, issues


def _status_from_solver(status_code: int) -> MILPSolverStatus:
    mapping = {
        pywraplp.Solver.OPTIMAL: MILPSolverStatus.OPTIMAL,
        pywraplp.Solver.FEASIBLE: MILPSolverStatus.FEASIBLE_NOT_PROVEN,
        pywraplp.Solver.INFEASIBLE: MILPSolverStatus.INFEASIBLE,
        pywraplp.Solver.UNBOUNDED: MILPSolverStatus.UNBOUNDED,
        pywraplp.Solver.ABNORMAL: MILPSolverStatus.SOLVER_ERROR,
        pywraplp.Solver.NOT_SOLVED: MILPSolverStatus.NO_SOLUTION,
    }
    return mapping.get(status_code, MILPSolverStatus.SOLVER_ERROR)


def _termination_reason(status: MILPSolverStatus) -> str:
    return {
        MILPSolverStatus.OPTIMAL: "proven_optimal",
        MILPSolverStatus.FEASIBLE_NOT_PROVEN: "feasible_incumbent_at_limit",
        MILPSolverStatus.INFEASIBLE: "proven_infeasible",
        MILPSolverStatus.UNBOUNDED: "proven_unbounded",
        MILPSolverStatus.NO_SOLUTION: "not_solved",
        MILPSolverStatus.SOLVER_ERROR: "solver_error",
    }[status]


def _build_solver(spec: LinearMILPSpec) -> tuple[pywraplp.Solver, dict[str, object]]:
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise DiscreteRunError("OR-Tools CBC solver is unavailable")
    variables: dict[str, object] = {}
    for variable in spec.variables:
        if variable.type is MILPVariableType.BINARY:
            variables[variable.variable_id] = solver.IntVar(
                variable.lower_bound, variable.upper_bound, variable.variable_id
            )
        elif variable.type is MILPVariableType.INTEGER:
            variables[variable.variable_id] = solver.IntVar(
                round(variable.lower_bound),
                round(variable.upper_bound),
                variable.variable_id,
            )
        else:
            variables[variable.variable_id] = solver.NumVar(
                variable.lower_bound, variable.upper_bound, variable.variable_id
            )

    objective = solver.Objective()
    for term in spec.objective.expression.terms:
        objective.SetCoefficient(variables[term.variable_id], term.coefficient)
    if spec.objective.expression.constant:
        objective.SetOffset(spec.objective.expression.constant)
    if spec.objective.sense is MILPSense.MAXIMIZE:
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    for constraint in spec.constraints:
        rhs = constraint.rhs - constraint.expression.constant
        if constraint.sense is ConstraintSense.LE:
            handle = solver.Constraint(-solver.infinity(), rhs, constraint.constraint_id)
        elif constraint.sense is ConstraintSense.GE:
            handle = solver.Constraint(rhs, solver.infinity(), constraint.constraint_id)
        else:
            handle = solver.Constraint(rhs, rhs, constraint.constraint_id)
        for term in constraint.expression.terms:
            handle.SetCoefficient(variables[term.variable_id], term.coefficient)
    return solver, variables


def run_linear_milp(
    paths: ProjectPaths, spec: LinearMILPSpec, *, experiment_id: str
) -> LinearMILPResult:
    directory = paths.experiment_dir(experiment_id)
    (directory / "artifacts").mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        directory / "linear_milp_spec.json", spec.model_dump_json(indent=2) + "\n"
    )
    instance_hash = linear_milp_instance_hash(spec)
    canonical = linear_milp_canonical_payload(spec)
    atomic_write_text(
        directory / "artifacts" / "canonical_instance.json",
        json.dumps(
            {"instance_hash": instance_hash, "canonical_instance": canonical},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    solver, handles = _build_solver(spec)
    config = spec.solver_config
    solver.SetTimeLimit(int(config.time_limit_seconds * 1000))
    applied_relative_gap = False
    if config.relative_gap is not None:
        try:
            solver.SetSolverSpecificParametersAsString(f"ratioGap {config.relative_gap}")
            applied_relative_gap = True
        except Exception:
            applied_relative_gap = False
    started = time.perf_counter()
    status_code = solver.Solve()
    wall = time.perf_counter() - started
    status = _status_from_solver(status_code)

    if status in (MILPSolverStatus.OPTIMAL, MILPSolverStatus.FEASIBLE_NOT_PROVEN):
        values = {
            variable_id: float(handle.solution_value())  # type: ignore[attr-defined]
            for variable_id, handle in handles.items()
        }
        valid, _issues = _incumbent_is_valid(spec, values)
        if not valid:
            status = MILPSolverStatus.SOLVER_ERROR
            termination = "incumbent_violates_declared_tolerances"
        else:
            termination = _termination_reason(status)
        activities = [
            MILPConstraintActivity(
                constraint_id=constraint.constraint_id,
                lhs=lhs,
                rhs=constraint.rhs,
                sense=constraint.sense,
                slack=_slack(constraint, lhs),
                satisfied=_satisfied(
                    constraint, lhs, config.feasibility_tolerance
                ),
            )
            for constraint in spec.constraints
            for lhs in [_activity(constraint.expression, values)]
        ]
        objective_value = _activity(spec.objective.expression, values)
        best_bound: float | None = None
        gap: float | None = None
        if status is MILPSolverStatus.OPTIMAL:
            best_bound = objective_value
            gap = 0.0
        else:
            raw_bound = float(solver.Objective().BestBound())
            if math.isfinite(raw_bound):
                best_bound = raw_bound
                gap = abs(raw_bound - objective_value) / max(abs(objective_value), 1e-9)
        result = LinearMILPResult(
            experiment_id=experiment_id,
            model_name=spec.model_name,
            instance_hash=instance_hash,
            variable_values=values,
            objective_value=objective_value,
            objective_sense=spec.objective.sense,
            constraint_activities=activities,
            solver_status=status,
            termination_reason=termination,
            best_bound=best_bound,
            gap=gap,
            relative_gap_limit=config.relative_gap if applied_relative_gap else None,
            solver_name="CBC",
            solver_version=version("ortools"),
            solver_time_limit_seconds=config.time_limit_seconds,
            solve_wall_seconds=wall,
            feasibility_tolerance=config.feasibility_tolerance,
            integrality_tolerance=config.integrality_tolerance,
        )
    else:
        result = LinearMILPResult(
            experiment_id=experiment_id,
            model_name=spec.model_name,
            instance_hash=instance_hash,
            objective_sense=spec.objective.sense,
            solver_status=status,
            termination_reason=_termination_reason(status),
            relative_gap_limit=config.relative_gap if applied_relative_gap else None,
            solver_name="CBC",
            solver_version=version("ortools"),
            solver_time_limit_seconds=config.time_limit_seconds,
            solve_wall_seconds=wall,
            feasibility_tolerance=config.feasibility_tolerance,
            integrality_tolerance=config.integrality_tolerance,
        )
    _target, result = _write_hashed_result(
        paths, experiment_id, result, "linear_milp_result.json"
    )
    return result


def run_finite_policy(
    paths: ProjectPaths, spec: FinitePolicySpec, *, experiment_id: str
) -> FinitePolicyResult:
    directory = paths.experiment_dir(experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        directory / "finite_policy_spec.json", spec.model_dump_json(indent=2) + "\n"
    )
    evaluator = get_policy_evaluator(spec.evaluator_id)
    validate_parameters(evaluator, spec.parameters)
    decisions = sorted(evaluator.decision_variables, key=lambda item: item.variable_id)
    state_space_size = policy_state_space_size(list(decisions))
    parameter_hash = policy_parameter_hash(
        evaluator.evaluator_id, evaluator.version, spec.parameters
    )
    provenance_hash = policy_provenance_hash(spec.parameter_provenance)
    disclosure = (
        "decision conditional on point estimates; uncertainty not propagated"
        if spec.point_estimate_conditioned
        else None
    )
    common = {
        "experiment_id": experiment_id,
        "model_name": spec.model_name,
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_version": evaluator.version,
        "evaluator_hash": evaluator.implementation_hash,
        "parameter_hash": parameter_hash,
        "provenance_hash": provenance_hash,
        "parameters": {name: spec.parameters[name] for name in sorted(spec.parameters)},
        "parameter_provenance": spec.parameter_provenance,
        "point_estimate_conditioned": spec.point_estimate_conditioned,
        "uncertainty_disclosure": disclosure,
        "objective_sense": spec.objective_sense,
        "decision_variables": [
            decision.model_copy(update={"domain": decision.canonical_domain})
            for decision in decisions
        ],
        "state_space_size": state_space_size,
        "max_enumerated_states": spec.max_enumerated_states,
    }
    if state_space_size > spec.max_enumerated_states:
        result = FinitePolicyResult(
            **common,
            evaluated_state_count=0,
            feasible_policy_count=0,
            selected_policy=None,
            objective_value=None,
            optimal_policy_count=0,
            execution_status=FinitePolicyExecutionStatus.NOT_EXECUTED_STATE_SPACE_LIMIT,
        )
        _target, result = _write_hashed_result(
            paths, experiment_id, result, "finite_policy_result.json"
        )
        return result

    best_objective: float | None = None
    selected: dict[str, PolicyValue] | None = None
    optimal_policy_count = 0
    selected_metrics: dict[str, float] = {}
    selected_components: dict[str, float] | None = None
    feasible_count = 0
    for combination in itertools.product(
        *(decision.canonical_domain for decision in decisions)
    ):
        policy = {
            decision.variable_id: value
            for decision, value in zip(decisions, combination, strict=True)
        }
        evaluation = evaluator.evaluate(policy, spec.parameters)
        if not evaluation.feasible:
            continue
        if evaluation.objective is None or not math.isfinite(evaluation.objective):
            raise DiscreteRunError("registered evaluator returned a non-finite objective")
        feasible_count += 1
        objective = float(evaluation.objective)
        if best_objective is None or objective < best_objective - 1e-9:
            best_objective = objective
            selected = dict(policy)
            selected_metrics = dict(evaluation.metrics)
            selected_components = (
                dict(evaluation.objective_components)
                if evaluation.objective_components is not None
                else None
            )
            optimal_policy_count = 1
        elif abs(objective - best_objective) <= 1e-9:
            optimal_policy_count += 1
    result = FinitePolicyResult(
        **common,
        evaluated_state_count=state_space_size,
        feasible_policy_count=feasible_count,
        selected_policy=selected,
        objective_value=best_objective,
        metrics=selected_metrics,
        objective_components=selected_components,
        optimal_policy_count=optimal_policy_count,
        execution_status=FinitePolicyExecutionStatus.COMPLETED,
    )
    _target, result = _write_hashed_result(
        paths, experiment_id, result, "finite_policy_result.json"
    )
    return result


def canonical_policy_order(
    decisions: list[DiscreteDecisionVariable],
) -> list[tuple[PolicyValue, ...]]:
    return list(
        itertools.product(*(decision.canonical_domain for decision in decisions))
    )


__all__ = [
    "DiscreteRunError",
    "canonical_policy_order",
    "run_finite_policy",
    "run_linear_milp",
]
