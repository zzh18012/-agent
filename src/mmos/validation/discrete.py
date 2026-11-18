from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ortools.linear_solver import pywraplp

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    finite_policy_result_file,
    finite_policy_spec_file,
    linear_milp_result_file,
    linear_milp_spec_file,
    load_finite_policy_result,
    load_finite_policy_spec,
    load_linear_milp_result,
    load_linear_milp_spec,
)
from mmos.experiments.policy_evaluators import get_policy_evaluator
from mmos.schemas.data import Severity
from mmos.schemas.discrete import (
    ConstraintSense,
    FinitePolicyExecutionStatus,
    LinearConstraintSpec,
    LinearExpression,
    LinearMILPSpec,
    MILPSense,
    MILPSolverStatus,
    MILPVariableType,
    PolicyValue,
    linear_milp_canonical_payload,
    linear_milp_instance_hash,
    policy_parameter_hash,
    policy_provenance_hash,
    policy_value_key,
)
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_ORACLE_TOLERANCE = 1e-7
_TIE_TOLERANCE = 1e-9
_OBJECTIVE_RELATIVE_TOLERANCE = 1e-9
_LP_TIME_LIMIT_MS = 5000


def _artifact_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("artifact_hash", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _artifact_integrity(path: Path) -> tuple[bool, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get("artifact_hash", ""))
    if not recorded:
        return False, "artifact_hash is missing"
    if recorded != _artifact_hash(payload):
        return False, "artifact_hash does not match the artifact body"
    return True, ""


@dataclass(frozen=True)
class ExhaustiveOutcome:
    status: Literal["complete", "skipped"]
    optimum: float | None
    feasible_count: int
    reason: str | None = None
    used_continuous_lp: bool = False


def _o_lhs(expression: LinearExpression, values: dict[str, float]) -> float:
    total = expression.constant
    for term in expression.terms:
        total += term.coefficient * values[term.variable_id]
    return total


def _o_satisfied(
    constraint: LinearConstraintSpec, lhs: float, tolerance: float
) -> bool:
    if constraint.sense is ConstraintSense.LE:
        return lhs <= constraint.rhs + tolerance
    if constraint.sense is ConstraintSense.GE:
        return lhs >= constraint.rhs - tolerance
    return abs(lhs - constraint.rhs) <= tolerance


def _o_feasible(
    spec: LinearMILPSpec, values: dict[str, float], tolerance: float
) -> bool:
    for variable in spec.variables:
        value = values[variable.variable_id]
        if value < variable.lower_bound - tolerance:
            return False
        if value > variable.upper_bound + tolerance:
            return False
    return all(
        _o_satisfied(constraint, _o_lhs(constraint.expression, values), tolerance)
        for constraint in spec.constraints
    )


def _discrete_domains(
    spec: LinearMILPSpec,
) -> list[tuple[str, list[float]]]:
    domains: list[tuple[str, list[float]]] = []
    for variable in spec.variables:
        if variable.type is MILPVariableType.BINARY:
            domains.append((variable.variable_id, [0.0, 1.0]))
        elif variable.type is MILPVariableType.INTEGER:
            low = round(variable.lower_bound)
            high = round(variable.upper_bound)
            domains.append(
                (variable.variable_id, [float(value) for value in range(low, high + 1)])
            )
    return domains


def _continuous_lp_objective(
    spec: LinearMILPSpec, fixed: dict[str, float]
) -> tuple[Literal["optimal", "infeasible", "error"], float | None]:
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return "error", None
    handles = {}
    for variable in spec.variables:
        if variable.type is MILPVariableType.CONTINUOUS:
            handles[variable.variable_id] = solver.NumVar(
                variable.lower_bound, variable.upper_bound, variable.variable_id
            )
    objective = solver.Objective()
    constant = spec.objective.expression.constant
    for term in spec.objective.expression.terms:
        if term.variable_id in fixed:
            constant += term.coefficient * fixed[term.variable_id]
        else:
            objective.SetCoefficient(handles[term.variable_id], term.coefficient)
    if constant:
        objective.SetOffset(constant)
    if spec.objective.sense is MILPSense.MAXIMIZE:
        objective.SetMaximization()
    else:
        objective.SetMinimization()
    for constraint in spec.constraints:
        rhs = constraint.rhs
        for term in constraint.expression.terms:
            if term.variable_id in fixed:
                rhs -= term.coefficient * fixed[term.variable_id]
        rhs -= constraint.expression.constant
        if constraint.sense is ConstraintSense.LE:
            handle = solver.Constraint(-solver.infinity(), rhs, constraint.constraint_id)
        elif constraint.sense is ConstraintSense.GE:
            handle = solver.Constraint(rhs, solver.infinity(), constraint.constraint_id)
        else:
            handle = solver.Constraint(rhs, rhs, constraint.constraint_id)
        for term in constraint.expression.terms:
            if term.variable_id not in fixed:
                handle.SetCoefficient(handles[term.variable_id], term.coefficient)
    solver.SetTimeLimit(_LP_TIME_LIMIT_MS)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        return "optimal", float(objective.Value())
    if status == pywraplp.Solver.INFEASIBLE:
        return "infeasible", None
    return "error", None


def exhaustive_milp_optimum(
    spec: LinearMILPSpec, *, max_states: int = 100_000
) -> ExhaustiveOutcome:
    domains = _discrete_domains(spec)
    if not domains:
        return ExhaustiveOutcome(
            status="skipped", optimum=None, feasible_count=0, reason="no discrete variables"
        )
    count = 1
    for _name, values in domains:
        count *= len(values)
    if count > max_states:
        return ExhaustiveOutcome(
            status="skipped",
            optimum=None,
            feasible_count=0,
            reason=f"state space {count} exceeds oracle limit {max_states}",
        )
    has_continuous = any(
        variable.type is MILPVariableType.CONTINUOUS for variable in spec.variables
    )
    best: float | None = None
    feasible_count = 0
    for combination in itertools.product(*(values for _name, values in domains)):
        fixed = {
            name: value for (name, _values), value in zip(domains, combination, strict=True)
        }
        if not has_continuous:
            if not _o_feasible(spec, fixed, _ORACLE_TOLERANCE):
                continue
            feasible_count += 1
            objective = _o_lhs(spec.objective.expression, fixed)
        else:
            outcome, objective = _continuous_lp_objective(spec, fixed)
            if outcome == "error":
                return ExhaustiveOutcome(
                    status="skipped",
                    optimum=None,
                    feasible_count=feasible_count,
                    reason="continuous sub-LP could not be solved",
                    used_continuous_lp=True,
                )
            if outcome == "infeasible":
                continue
            feasible_count += 1
            assert objective is not None
        if best is None:
            best = objective
        elif spec.objective.sense is MILPSense.MAXIMIZE:
            best = max(best, objective)
        else:
            best = min(best, objective)
    return ExhaustiveOutcome(
        status="complete",
        optimum=best,
        feasible_count=feasible_count,
        used_continuous_lp=has_continuous,
    )


def _comparison_tolerance(reported: float, base: float) -> float:
    return max(base, _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(reported)))


def validate_linear_milp(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = load_linear_milp_spec(paths, experiment_id)
    result = load_linear_milp_result(paths, experiment_id)
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(
            ValidationIssue(code=code, severity=Severity.WARNING, message=message)
        )

    integrity, integrity_reason = _artifact_integrity(
        linear_milp_result_file(paths, experiment_id)
    )
    if not integrity:
        error("artifact_hash_mismatch", integrity_reason)
    expected_hash = linear_milp_instance_hash(spec)
    if result.instance_hash != expected_hash:
        error(
            "instance_hash_mismatch",
            f"recorded instance hash {result.instance_hash} != recomputed {expected_hash}",
        )
    canonical_path = (
        paths.experiment_dir(experiment_id) / "artifacts" / "canonical_instance.json"
    )
    if not canonical_path.exists():
        error("missing_canonical_instance", "canonical_instance.json is missing")
    else:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        if canonical.get("instance_hash") != expected_hash:
            error(
                "canonical_hash_mismatch",
                "canonical instance hash does not match the recomputed instance hash",
            )
        if canonical.get("canonical_instance") != linear_milp_canonical_payload(spec):
            error(
                "canonical_payload_mismatch",
                "canonical instance payload does not match the spec",
            )

    if result.feasibility_tolerance != spec.solver_config.feasibility_tolerance:
        error("tolerance_mismatch", "result feasibility tolerance differs from the spec")
    if result.integrality_tolerance != spec.solver_config.integrality_tolerance:
        error("integrality_tolerance_mismatch", "result integrality tolerance differs")
    if result.solver_name != "CBC":
        error("unexpected_solver", f"expected CBC, recorded {result.solver_name}")

    status = result.solver_status
    if status in (MILPSolverStatus.OPTIMAL, MILPSolverStatus.FEASIBLE_NOT_PROVEN):
        if result.objective_value is None:
            error("missing_objective", "feasible result has no objective value")
        variable_ids = {variable.variable_id for variable in spec.variables}
        if set(result.variable_values) != variable_ids:
            error(
                "variable_set_mismatch",
                "variable_values do not match the declared variable set",
            )
        tolerance = result.feasibility_tolerance
        integrality = result.integrality_tolerance
        for variable in spec.variables:
            value = result.variable_values.get(variable.variable_id)
            if value is None or not math.isfinite(value):
                error("missing_variable_value", f"{variable.variable_id}: missing/non-finite")
                continue
            if (
                value < variable.lower_bound - tolerance
                or value > variable.upper_bound + tolerance
            ):
                error(
                    "bound_violation",
                    f"{variable.variable_id}={value} outside declared bounds",
                )
            if variable.type in (
                MILPVariableType.BINARY,
                MILPVariableType.INTEGER,
            ) and abs(value - round(value)) > integrality:
                error(
                    "integrality_violation",
                    f"{variable.variable_id}={value} is not integral",
                )
            if variable.type is MILPVariableType.BINARY and round(value) not in (0, 1):
                error(
                    "binary_violation",
                    f"{variable.variable_id}={value} is not a 0/1 value",
                )
        recorded = {item.constraint_id: item for item in result.constraint_activities}
        expected_ids = {constraint.constraint_id for constraint in spec.constraints}
        if set(recorded) != expected_ids:
            error(
                "constraint_activity_set_mismatch",
                "constraint activities do not match the declared constraints",
            )
        for constraint in spec.constraints:
            values = result.variable_values
            if set(values) != variable_ids:
                break
            lhs = _o_lhs(constraint.expression, values)
            satisfied = _o_satisfied(constraint, lhs, tolerance)
            slack = (
                constraint.rhs - lhs
                if constraint.sense is ConstraintSense.LE
                else lhs - constraint.rhs
                if constraint.sense is ConstraintSense.GE
                else -abs(lhs - constraint.rhs)
            )
            activity = recorded.get(constraint.constraint_id)
            if activity is None:
                continue
            if abs(activity.lhs - lhs) > _comparison_tolerance(lhs, tolerance):
                error(
                    "constraint_activity_mismatch",
                    f"{constraint.constraint_id}: recorded {activity.lhs}, recomputed {lhs}",
                )
            if abs(activity.slack - slack) > _comparison_tolerance(slack, tolerance):
                error(
                    "constraint_slack_mismatch",
                    f"{constraint.constraint_id}: recorded slack {activity.slack}, "
                    f"recomputed {slack}",
                )
            if activity.satisfied != satisfied:
                error(
                    "constraint_satisfaction_mismatch",
                    f"{constraint.constraint_id}: recorded satisfied={activity.satisfied}, "
                    f"recomputed={satisfied}",
                )
            if not satisfied:
                error(
                    "constraint_violation",
                    f"{constraint.constraint_id}: recomputed activity {lhs} violates "
                    f"{constraint.sense.value} {constraint.rhs}",
                )
        if set(result.variable_values) == variable_ids:
            recomputed_objective = _o_lhs(spec.objective.expression, result.variable_values)
            if result.objective_value is None:
                pass
            elif abs(result.objective_value - recomputed_objective) > _comparison_tolerance(
                result.objective_value, tolerance
            ):
                error(
                    "objective_mismatch",
                    f"recorded {result.objective_value}, recomputed {recomputed_objective}",
                )
            oracle = exhaustive_milp_optimum(spec)
            if oracle.status == "complete":
                if oracle.optimum is None:
                    error(
                        "oracle_contradiction",
                        "exhaustive oracle found no feasible assignment",
                    )
                elif result.objective_value is not None:
                    allowed = _comparison_tolerance(oracle.optimum, _ORACLE_TOLERANCE)
                    if status is MILPSolverStatus.OPTIMAL and (
                        abs(result.objective_value - oracle.optimum) > allowed
                    ):
                        error(
                            "optimum_mismatch",
                            f"solver objective {result.objective_value} != exhaustive "
                            f"optimum {oracle.optimum}",
                        )
                    if (
                        status is MILPSolverStatus.FEASIBLE_NOT_PROVEN
                        and result.objective_value < oracle.optimum - allowed
                    ):
                        error(
                            "better_than_optimum",
                            "incumbent is better than the exhaustive optimum",
                        )
            else:
                warning(
                    "oracle_skipped",
                    f"exhaustive oracle skipped: {oracle.reason}",
                )
        if status is MILPSolverStatus.OPTIMAL:
            if result.gap not in (None, 0.0) and abs(result.gap) > tolerance:
                error("optimal_gap_nonzero", f"optimal status with gap {result.gap}")
            if result.best_bound is None:
                error("missing_best_bound", "optimal result has no best bound")
            elif result.objective_value is not None and abs(
                result.best_bound - result.objective_value
            ) > _comparison_tolerance(result.objective_value, tolerance):
                error(
                    "best_bound_mismatch",
                    f"best bound {result.best_bound} != objective {result.objective_value}",
                )
            if result.termination_reason != "proven_optimal":
                error(
                    "termination_mismatch",
                    f"optimal status records termination {result.termination_reason}",
                )
        else:
            if result.termination_reason != "feasible_incumbent_at_limit":
                warning(
                    "termination_unexpected",
                    f"non-optimal incumbent records termination "
                    f"{result.termination_reason}",
                )
            if result.gap is None and result.best_bound is not None:
                error("missing_gap", "best bound present but no optimality gap recorded")
            if (
                result.relative_gap_limit is not None
                and result.gap is not None
                and result.gap > result.relative_gap_limit + 1e-6
            ):
                warning(
                    "gap_exceeds_configured_limit",
                    f"reported gap {result.gap} exceeds configured {result.relative_gap_limit}",
                )
    elif status is MILPSolverStatus.INFEASIBLE:
        if result.variable_values or result.objective_value is not None:
            error(
                "infeasible_with_solution",
                "infeasible result must not carry variable values or an objective",
            )
        if result.constraint_activities:
            error(
                "infeasible_with_activities",
                "infeasible result must not carry constraint activities",
            )
        if result.termination_reason != "proven_infeasible":
            error(
                "termination_mismatch",
                f"infeasible status records termination {result.termination_reason}",
            )
        oracle = exhaustive_milp_optimum(spec)
        if oracle.status == "complete" and oracle.feasible_count > 0:
            error(
                "oracle_contradiction",
                f"oracle found {oracle.feasible_count} feasible assignments",
            )
        elif oracle.status == "skipped":
            warning(
                "infeasibility_not_independently_confirmed",
                f"exhaustive oracle skipped: {oracle.reason}",
            )
    elif status is MILPSolverStatus.UNBOUNDED:
        if result.variable_values or result.objective_value is not None:
            error("unbounded_with_solution", "unbounded result carries solution data")
        warning(
            "unbounded_not_independently_confirmed",
            "the exhaustive oracle does not certify unboundedness",
        )
    else:
        error(
            "unusable_solver_status",
            f"solver status {status.value} cannot support evidence",
        )

    return build_report(
        "linear_milp",
        experiment_id,
        issues,
        evidence=[str(linear_milp_spec_file(paths, experiment_id)),
                  str(linear_milp_result_file(paths, experiment_id))],
        recommendations=[
            "recompute every constraint and the objective from the frozen variable values",
            "compare the optimum with the exhaustive oracle for small discrete instances",
        ],
    )


def _policy_equal(left: dict[str, PolicyValue], right: dict[str, PolicyValue]) -> bool:
    if set(left) != set(right):
        return False
    return all(policy_value_key(left[key]) == policy_value_key(right[key]) for key in left)


def validate_finite_policy(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = load_finite_policy_spec(paths, experiment_id)
    result = load_finite_policy_result(paths, experiment_id)
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(
            ValidationIssue(code=code, severity=Severity.WARNING, message=message)
        )

    integrity, integrity_reason = _artifact_integrity(
        finite_policy_result_file(paths, experiment_id)
    )
    if not integrity:
        error("artifact_hash_mismatch", integrity_reason)
    try:
        evaluator = get_policy_evaluator(spec.evaluator_id)
    except Exception as exc:
        error("unknown_evaluator", str(exc))
        return build_report("finite_policy", experiment_id, issues)
    if evaluator.version != result.evaluator_version:
        error("evaluator_version_mismatch", "result was produced by a different version")
    if evaluator.implementation_hash != result.evaluator_hash:
        error(
            "stale_evaluator",
            "registered evaluator implementation hash differs from the artifact",
        )
    decisions = sorted(evaluator.decision_variables, key=lambda item: item.variable_id)
    canonical_decisions = [
        decision.model_copy(update={"domain": decision.canonical_domain})
        for decision in decisions
    ]
    if [decision.model_dump() for decision in result.decision_variables] != [
        decision.model_dump() for decision in canonical_decisions
    ]:
        error(
            "decision_variable_mismatch",
            "artifact decision variables differ from the registered evaluator",
        )
    expected_size = 1
    for decision in canonical_decisions:
        expected_size *= len(decision.domain)
    if result.state_space_size != expected_size:
        error(
            "state_space_mismatch",
            f"recorded {result.state_space_size}, recomputed {expected_size}",
        )
    if result.max_enumerated_states != spec.max_enumerated_states:
        error("limit_mismatch", "artifact enumeration limit differs from the spec")
    if spec.parameters != result.parameters:
        error("parameter_mismatch", "artifact parameters differ from the spec")
    expected_parameter_hash = policy_parameter_hash(
        evaluator.evaluator_id, evaluator.version, spec.parameters
    )
    if result.parameter_hash != expected_parameter_hash:
        error("parameter_hash_mismatch", "parameter hash does not match the parameters")
    if [item.model_dump() for item in result.parameter_provenance] != [
        item.model_dump() for item in spec.parameter_provenance
    ]:
        error("provenance_mismatch", "artifact provenance differs from the spec")
    if result.provenance_hash != policy_provenance_hash(spec.parameter_provenance):
        error("provenance_hash_mismatch", "provenance hash does not match the spec")
    if spec.point_estimate_conditioned:
        expected_disclosure = (
            "decision conditional on point estimates; uncertainty not propagated"
        )
        if result.uncertainty_disclosure != expected_disclosure:
            error(
                "missing_uncertainty_disclosure",
                "point-estimate-conditioned result lacks the required disclosure",
            )
    elif result.uncertainty_disclosure is not None:
        error(
            "unexpected_uncertainty_disclosure",
            "disclosure present without provenance-backed parameters",
        )

    if expected_size > spec.max_enumerated_states:
        limit_status = FinitePolicyExecutionStatus.NOT_EXECUTED_STATE_SPACE_LIMIT
        if result.execution_status is not limit_status:
            error(
                "state_space_limit_not_enforced",
                "state space exceeds the limit but the search was marked complete",
            )
        if (
            result.evaluated_state_count
            or result.feasible_policy_count
            or result.optimal_policy_count
            or result.selected_policy is not None
            or result.objective_value is not None
        ):
            error(
                "state_space_limit_with_results",
                "an unexecuted search must not report policies or objectives",
            )
        return build_report(
            "finite_policy",
            experiment_id,
            issues,
            evidence=[str(finite_policy_spec_file(paths, experiment_id)),
                      str(finite_policy_result_file(paths, experiment_id))],
            recommendations=["reduce the declared policy space or raise the explicit limit"],
        )

    if result.execution_status is not FinitePolicyExecutionStatus.COMPLETED:
        error("unexpected_execution_status", "state space fits the limit but search did not run")
        return build_report("finite_policy", experiment_id, issues)

    best: float | None = None
    selected: dict[str, PolicyValue] | None = None
    selected_metrics: dict[str, float] = {}
    selected_components: dict[str, float] | None = None
    feasible_count = 0
    optimal_count = 0
    for combination in itertools.product(
        *(decision.canonical_domain for decision in canonical_decisions)
    ):
        policy = {
            decision.variable_id: value
            for decision, value in zip(canonical_decisions, combination, strict=True)
        }
        evaluation = evaluator.evaluate(policy, spec.parameters)
        if not evaluation.feasible:
            continue
        if evaluation.objective is None or not math.isfinite(evaluation.objective):
            error(
                "non_finite_objective",
                "registered evaluator returned a non-finite objective",
            )
            continue
        feasible_count += 1
        objective = float(evaluation.objective)
        if best is None or objective < best - _TIE_TOLERANCE:
            best = objective
            selected = dict(policy)
            selected_metrics = dict(evaluation.metrics)
            selected_components = (
                dict(evaluation.objective_components)
                if evaluation.objective_components is not None
                else None
            )
            optimal_count = 1
        elif abs(objective - best) <= _TIE_TOLERANCE:
            optimal_count += 1

    if result.evaluated_state_count != expected_size:
        error(
            "evaluated_state_count_mismatch",
            f"recorded {result.evaluated_state_count}, expected {expected_size}",
        )
    if result.feasible_policy_count != feasible_count:
        error(
            "feasible_count_mismatch",
            f"recorded {result.feasible_policy_count}, recomputed {feasible_count}",
        )
    if result.optimal_policy_count != optimal_count:
        error(
            "optimal_count_mismatch",
            f"recorded {result.optimal_policy_count}, recomputed {optimal_count}",
        )
    if feasible_count == 0:
        if result.selected_policy is not None or result.objective_value is not None:
            error("unexpected_optimum", "no feasible policy exists but an optimum was recorded")
        return build_report(
            "finite_policy",
            experiment_id,
            issues,
            evidence=[str(finite_policy_spec_file(paths, experiment_id)),
                      str(finite_policy_result_file(paths, experiment_id))],
        )
    if result.selected_policy is None or selected is None:
        error("missing_selected_policy", "a feasible optimum exists but none was recorded")
        return build_report("finite_policy", experiment_id, issues)
    if not _policy_equal(result.selected_policy, selected):
        error(
            "selected_policy_mismatch",
            "recorded policy is not the canonical lexicographic optimum",
        )
    if best is None or result.objective_value is None:
        error("missing_objective", "optimum objective is missing")
    elif abs(result.objective_value - best) > max(
        _TIE_TOLERANCE, _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(best))
    ):
        error(
            "objective_mismatch",
            f"recorded {result.objective_value}, recomputed {best}",
        )
    if set(result.metrics) != set(selected_metrics):
        error("metrics_keys_mismatch", "artifact metrics do not match the evaluator output")
    else:
        for key, value in selected_metrics.items():
            if abs(result.metrics[key] - value) > max(
                _TIE_TOLERANCE, _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(value))
            ):
                error(
                    "metric_mismatch",
                    f"{key}: recorded {result.metrics[key]}, recomputed {value}",
                )
    if result.objective_components is not None and result.objective_value is not None:
        component_sum = sum(result.objective_components.values())
        if abs(component_sum - result.objective_value) > max(
            _TIE_TOLERANCE,
            _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(result.objective_value)),
        ):
            error(
                "component_sum_mismatch",
                f"components sum to {component_sum}, objective is {result.objective_value}",
            )
        if selected_components is not None:
            if set(result.objective_components) != set(selected_components):
                error("component_keys_mismatch", "component keys differ from the evaluator")
            else:
                for key, value in selected_components.items():
                    if abs(result.objective_components[key] - value) > max(
                        _TIE_TOLERANCE,
                        _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(value)),
                    ):
                        error(
                            "component_mismatch",
                            f"{key}: recorded {result.objective_components[key]}, "
                            f"recomputed {value}",
                        )
    elif selected_components is not None:
        warning(
            "components_missing",
            "evaluator declares additive components but the artifact omits them",
        )
    if result.tie_break_rule != "canonical_lexicographic":
        error("tie_break_rule_mismatch", "unexpected tie-break rule")
    return build_report(
        "finite_policy",
        experiment_id,
        issues,
        evidence=[str(finite_policy_spec_file(paths, experiment_id)),
                  str(finite_policy_result_file(paths, experiment_id))],
        recommendations=[
            "re-enumerate the declared finite policy space independently",
            "verify the canonical lexicographic optimum and tie count",
        ],
    )


def validate_linear_milp_experiment(
    paths: ProjectPaths, experiment_id: str
) -> list[ValidationReport]:
    report = validate_linear_milp(paths, experiment_id)
    _write_validation(paths, experiment_id, "linear_milp", report)
    spec = load_linear_milp_spec(paths, experiment_id)
    oracle = exhaustive_milp_optimum(spec)
    atomic_write_text(
        paths.experiment_dir(experiment_id) / "validation" / "exhaustive_oracle.json",
        json.dumps(
            {
                "status": oracle.status,
                "optimum": oracle.optimum,
                "feasible_count": oracle.feasible_count,
                "reason": oracle.reason,
                "used_continuous_lp": oracle.used_continuous_lp,
            },
            indent=2,
        )
        + "\n",
    )
    return [report]


def validate_finite_policy_experiment(
    paths: ProjectPaths, experiment_id: str
) -> list[ValidationReport]:
    report = validate_finite_policy(paths, experiment_id)
    _write_validation(paths, experiment_id, "finite_policy", report)
    return [report]


def _write_validation(
    paths: ProjectPaths, experiment_id: str, name: str, report: ValidationReport
) -> None:
    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / f"{name}.json", report.model_dump_json(indent=2) + "\n")
    summary = {
        "experiment_id": experiment_id,
        "blocking": report.blocking,
        "reports": {
            report.validator: {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
        },
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")


__all__ = [
    "ExhaustiveOutcome",
    "exhaustive_milp_optimum",
    "validate_finite_policy",
    "validate_finite_policy_experiment",
    "validate_linear_milp",
    "validate_linear_milp_experiment",
]
