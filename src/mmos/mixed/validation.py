import json
from pathlib import Path

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_record,
    load_experiment_spec,
    load_optimization_result,
    load_optimization_spec,
)
from mmos.experiments.optimization_runner import check_solution
from mmos.mixed.artifacts import (
    ArtifactContractError,
    contract_for,
    load_envelope,
    verify_forecast_artifact,
)
from mmos.schemas.data import Severity
from mmos.schemas.mixed import DAGRunRecord, DAGStatus, MixedExecutionPlan, NodeStatus
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.states import TaskFamily
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-6


def run_record_path(paths: ProjectPaths, run_id: str) -> Path:
    return paths.mixed_dir / "runs" / run_id / "dag_run.json"


def _load_run(paths: ProjectPaths, run_id: str) -> DAGRunRecord | None:
    target = run_record_path(paths, run_id)
    if not target.exists():
        return None
    return DAGRunRecord.model_validate_json(target.read_text(encoding="utf-8"))


def _summary_passed(paths: ProjectPaths, experiment_id: str) -> tuple[bool, str]:
    summary_path = paths.experiment_dir(experiment_id) / "validation" / "summary.json"
    if not summary_path.exists():
        return False, "validation summary is missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != experiment_id:
        return False, "validation summary belongs to another experiment"
    if summary.get("blocking"):
        return False, "validation summary contains blocking issues"
    reports = summary.get("reports", {})
    failed = [
        name
        for name, info in reports.items()
        if info.get("status") != "passed" or info.get("blocking")
    ]
    if failed:
        return False, f"validation reports did not pass: {', '.join(sorted(failed))}"
    return True, ""


def _evidence_frozen(paths: ProjectPaths, key: str) -> bool:
    provenance_path = paths.verified_dir / "provenance.json"
    if not provenance_path.exists():
        return False
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    return key in provenance.get("experiments", {})


def validate_mixed_run(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.WARNING, message=message))

    if plan.plan_hash != plan.computed_hash():
        error("plan_hash_mismatch", "plan hash does not match its content")
        return build_report("mixed", run_id, issues)

    try:
        plan.topological_order()
    except ValueError as exc:
        error("dag_cycle", str(exc))
        return build_report("mixed", run_id, issues)

    record = _load_run(paths, run_id)
    if record is None:
        error("missing_run_record", f"DAG run record is missing for {run_id}")
        return build_report("mixed", run_id, issues)

    if record.plan_hash != plan.plan_hash:
        error("run_plan_mismatch", "run record plan hash does not match the plan")
    if record.status is not DAGStatus.SUCCEEDED:
        error("run_not_succeeded", f"DAG run status is {record.status.value}")
    if not record.provenance_complete:
        error("provenance_incomplete", "DAG run did not record complete provenance")
    for node in record.nodes:
        if node.execution_status is not NodeStatus.SUCCEEDED:
            error(
                "node_not_succeeded",
                f"node {node.node_id} is {node.execution_status.value}",
            )
        if node.experiment_id is None:
            error("node_missing_experiment", f"node {node.node_id} has no experiment id")

    contract = contract_for(plan.spec)
    envelope_path = paths.mixed_dir / "runs" / run_id / "artifacts" / "forecast_envelope.json"
    artifact = None
    envelope = None
    if not envelope_path.exists():
        error("missing_envelope", f"forecast envelope is missing: {envelope_path}")
    else:
        envelope = load_envelope(envelope_path)
        try:
            artifact = verify_forecast_artifact(paths, envelope, contract)
        except ArtifactContractError as exc:
            error("artifact_contract", str(exc))

    if artifact is None or envelope is None:
        return build_report("mixed", run_id, issues)

    if artifact.timestamps != plan.spec.planning_periods:
        error("timestamp_mismatch", "forecast timestamps do not match planning periods")
    forecast_node = next(
        (node for node in record.nodes if node.task_family is TaskFamily.FORECASTING),
        None,
    )
    if forecast_node is None:
        error("missing_forecast_node", "the DAG run has no forecasting node")
    elif artifact.forecast_experiment_id != forecast_node.experiment_id:
        error("forecast_experiment_mismatch", "artifact and node disagree on the forecast run")
    if record.forecast_artifact_id != envelope.artifact_id:
        error("run_artifact_id_mismatch", "run record artifact id does not match the envelope")
    if record.forecast_artifact_hash != envelope.content_hash:
        error("run_artifact_hash_mismatch", "run record artifact hash does not match the envelope")

    forecast_spec = None
    forecast_spec_path = (
        paths.experiment_dir(artifact.forecast_experiment_id) / "experiment_spec.json"
    )
    if not forecast_spec_path.exists():
        error("missing_forecast_spec", "forecast experiment spec is missing")
    else:
        forecast_spec = load_experiment_spec(paths, artifact.forecast_experiment_id)
        if forecast_spec.dataset_hash != artifact.source_dataset_hash:
            error(
                "dataset_hash_mismatch",
                "forecast artifact dataset hash does not match the experiment spec",
            )
        if forecast_spec.target_column != artifact.target_name:
            error("target_mismatch", "forecast artifact target does not match the experiment spec")
        if forecast_spec.temporal_split is None:
            error("missing_temporal_split", "forecast experiment has no temporal split")
        elif forecast_spec.temporal_split.horizon < plan.spec.forecast_horizon:
            error(
                "horizon_mismatch",
                "forecast validation horizon is shorter than the planning horizon",
            )
        if not _evidence_frozen(paths, artifact.forecast_experiment_id):
            error("forecast_evidence_missing", "forecast evidence was not frozen")
    forecast_passed, forecast_reason = _summary_passed(paths, artifact.forecast_experiment_id)
    if not forecast_passed:
        error("forecast_validation", f"forecast validation failed: {forecast_reason}")

    optimization_id = record.optimization_experiment_id
    if optimization_id is None:
        optimization_nodes = [
            node for node in record.nodes if node.task_family is TaskFamily.OPTIMIZATION
        ]
        if optimization_nodes:
            optimization_id = optimization_nodes[0].experiment_id
    if optimization_id is None:
        error("missing_optimization_experiment", "no optimization experiment was recorded")
        return build_report("mixed", run_id, issues)

    optimization_record = load_experiment_record(paths, optimization_id)
    if optimization_record.status is not ExperimentStatus.VALIDATED:
        error("optimization_status", f"optimization status is {optimization_record.status.value}")
    if optimization_record.consumed_artifact_ids != [envelope.artifact_id]:
        error(
            "consumed_artifact_id_mismatch",
            "optimization record does not consume the verified forecast artifact",
        )
    if optimization_record.consumed_artifact_hashes != [envelope.content_hash]:
        error(
            "consumed_artifact_hash_mismatch",
            "optimization record consumed a different artifact hash",
        )
    if optimization_record.dependency_contract_status != "passed":
        error("dependency_contract", "dependency contract status is not passed")
    if artifact.forecast_experiment_id not in optimization_record.linked_experiment_ids:
        error("linked_experiment_missing", "optimization record does not link the forecast run")

    optimization_passed, optimization_reason = _summary_passed(paths, optimization_id)
    if not optimization_passed:
        error("optimization_validation", f"optimization validation failed: {optimization_reason}")
    if not _evidence_frozen(paths, f"optimization:{optimization_id}"):
        error("optimization_evidence_missing", "optimization evidence was not frozen")

    optimization_spec = load_optimization_spec(paths, optimization_id)
    result = load_optimization_result(paths, optimization_id)
    if optimization_spec.instance_hash != result.instance_hash:
        error("instance_hash_mismatch", "optimization spec and result disagree on the hash")

    solver = next(
        (solution for solution in result.solutions if solution.method == "ortools.cbc"),
        None,
    )
    if solver is None:
        error("missing_solver_solution", "no CBC solution was recorded")
        return build_report("mixed", run_id, issues)
    if not solver.feasible or solver.status not in {"optimal", "feasible"}:
        error("infeasible_plan", "the production plan is not feasible")
        return build_report("mixed", run_id, issues)

    feasible, violations, recomputed = check_solution(
        optimization_spec.model, solver.variable_values
    )
    if not feasible:
        for violation in violations[:5]:
            error("plan_violation", violation)
    if solver.objective_value is None:
        error("missing_objective", "the production plan has no objective value")
    elif abs(solver.objective_value - recomputed) > _TOLERANCE:
        error(
            "objective_mismatch",
            f"reported objective {solver.objective_value} != recomputed {recomputed}",
        )
    if record.final_objective is None or abs(record.final_objective - recomputed) > _TOLERANCE:
        error("run_objective_mismatch", "run record objective does not match the plan")
    if record.final_feasible is not True:
        error("run_feasibility_mismatch", "run record does not mark the plan feasible")

    balance_constraints = {
        constraint.name: constraint
        for constraint in optimization_spec.model.constraints
        if constraint.name.startswith("inventory_balance_")
    }
    for index, demand in enumerate(artifact.point_forecasts):
        constraint = balance_constraints.get(f"inventory_balance_{index}")
        if constraint is None:
            error("missing_balance", f"inventory balance {index} is missing")
            continue
        if constraint.lower is None or constraint.upper is None:
            error("non_equality_balance", f"inventory balance {index} is not an equality")
            continue
        actual = constraint.lower
        expected = (
            float(demand) - plan.spec.initial_inventory
            if index == 0
            else float(demand)
        )
        if abs(actual - expected) > _TOLERANCE:
            error(
                "balance_demand_mismatch",
                f"inventory balance {index} uses {actual}, artifact demand is {expected}",
            )
        production = solver.variable_values.get(f"production_{index}")
        inventory = solver.variable_values.get(f"inventory_{index}")
        if production is None or inventory is None:
            error("missing_plan_variable", f"period {index} is missing plan variables")
            continue
        if production < -_TOLERANCE or inventory < -_TOLERANCE:
            error("negative_plan", f"period {index} has a negative production or inventory")
        capacity = plan.spec.capacities[index]
        if production > capacity + _TOLERANCE:
            error(
                "capacity_violation",
                f"period {index} production {production} exceeds capacity {capacity}",
            )
        previous = (
            plan.spec.initial_inventory
            if index == 0
            else solver.variable_values.get(f"inventory_{index - 1}", 0.0)
        )
        residual = previous + production - inventory - float(demand)
        if abs(residual) > _TOLERANCE:
            error(
                "inventory_residual",
                f"period {index} inventory balance residual {residual}",
            )

    expected_objective = sum(
        float(plan.spec.production_costs[index])
        * solver.variable_values.get(f"production_{index}", 0.0)
        + float(plan.spec.holding_costs[index])
        * solver.variable_values.get(f"inventory_{index}", 0.0)
        for index in range(len(plan.spec.planning_periods))
    )
    if abs(expected_objective - recomputed) > _TOLERANCE:
        error(
            "objective_formula_mismatch",
            f"objective from the frozen spec is {expected_objective}, "
            f"model objective is {recomputed}",
        )

    if record.dependency_validation.get("status") != "passed":
        warning("dependency_trace_missing", "run record does not report dependency validation")

    return build_report(
        "mixed",
        run_id,
        issues,
        evidence=[
            envelope_path.relative_to(paths.root).as_posix(),
            run_record_path(paths, run_id).relative_to(paths.root).as_posix(),
        ],
        recommendations=[
            "recompute the production plan from the verified artifact",
            "keep the artifact contract checks independent of the executor",
        ],
    )


def validate_mixed_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
) -> list[ValidationReport]:
    report = validate_mixed_run(paths, plan, run_id)
    output_dir = paths.mixed_dir / "runs" / run_id / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "mixed.json", report.model_dump_json(indent=2) + "\n"
    )
    summary = {
        "experiment_id": run_id,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
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
    return [report]


__all__ = ["run_record_path", "validate_mixed_experiment", "validate_mixed_run"]
