import json
import math
import statistics
from pathlib import Path

import polars as pl

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_record,
    load_experiment_spec,
    load_optimization_result,
    load_optimization_spec,
)
from mmos.data.table import load_single_table
from mmos.experiments.optimization_runner import check_solution
from mmos.experiments.simulation_runner import (
    confidence_interval,
    derive_seed_schedule,
    seed_schedule_hash,
)
from mmos.mixed.artifacts import (
    ArtifactContractError,
    contract_for,
    contract_for_forecast_error,
    contract_for_production_plan,
    load_envelope,
    verify_forecast_artifact,
    verify_forecast_error_artifact,
    verify_production_plan_artifact,
)
from mmos.mixed.simulation import (
    load_mixed_simulation_record,
    load_mixed_simulation_result,
    load_mixed_simulation_spec,
    percentile,
)
from mmos.schemas.data import Severity
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.mixed import (
    ArtifactEnvelope,
    DAGRunRecord,
    DAGStatus,
    ForecastArtifact,
    ForecastErrorArtifact,
    MixedExecutionPlan,
    NodeStatus,
    ProductionPlanArtifact,
)
from mmos.schemas.mixed_simulation import MixedSimulationResult, MixedSimulationSpec
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
    if plan.has_simulation_node():
        _simulation_link_issues(paths, plan, record, run_id, error, warning)

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


def _simulation_artifacts(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    error,
) -> (
    tuple[
        ArtifactEnvelope,
        ForecastArtifact,
        ArtifactEnvelope,
        ForecastErrorArtifact,
        ArtifactEnvelope,
        ProductionPlanArtifact,
    ]
    | None
):
    artifact_dir = paths.mixed_dir / "runs" / run_id / "artifacts"
    forecast_path = artifact_dir / "forecast_envelope.json"
    error_path = artifact_dir / "forecast_error_envelope.json"
    plan_path = artifact_dir / "production_plan_envelope.json"
    missing = [
        path.name
        for path in (forecast_path, error_path, plan_path)
        if not path.exists()
    ]
    if missing:
        error("missing_simulation_artifact", f"missing artifacts: {', '.join(missing)}")
        return None
    try:
        forecast_envelope = load_envelope(forecast_path)
        forecast = verify_forecast_artifact(
            paths, forecast_envelope, contract_for(plan.spec)
        )
        error_envelope = load_envelope(error_path)
        error_artifact = verify_forecast_error_artifact(
            paths,
            error_envelope,
            contract_for_forecast_error(
                plan.spec,
                forecast.forecast_experiment_id,
                forecast_envelope.content_hash,
            ),
        )
        plan_envelope = load_envelope(plan_path)
        plan_artifact = verify_production_plan_artifact(
            paths,
            plan_envelope,
            contract_for_production_plan(
                plan.spec,
                forecast_envelope.artifact_id,
                forecast_envelope.content_hash,
            ),
        )
    except ArtifactContractError as exc:
        error("simulation_artifact_contract", str(exc))
        return None
    return (
        forecast_envelope,
        forecast,
        error_envelope,
        error_artifact,
        plan_envelope,
        plan_artifact,
    )


def _check_residual_provenance(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    forecast: ForecastArtifact,
    forecast_envelope,
    error_artifact: ForecastErrorArtifact,
    error,
) -> None:
    if error_artifact.source_forecast_artifact_id != forecast_envelope.artifact_id:
        error(
            "residual_source_artifact",
            "forecast-error artifact does not reference the consumed forecast artifact id",
        )
    experiment_id = error_artifact.source_forecast_experiment_id
    passed, reason = _summary_passed(paths, experiment_id)
    if not passed:
        error(
            "residual_source_validation",
            f"residual source experiment validation failed: {reason}",
        )
    experiment_dir = paths.experiment_dir(experiment_id)
    predictions_path = experiment_dir / "artifacts" / "predictions.parquet"
    metrics_path = experiment_dir / "artifacts" / "metrics.json"
    if not predictions_path.exists() or not metrics_path.exists():
        error("residual_predictions_missing", "out-of-fold predictions are missing")
        return
    result = ExperimentResult.model_validate_json(
        metrics_path.read_text(encoding="utf-8")
    )
    history = load_single_table(paths).sort(plan.spec.time_column)
    timestamps = [str(value) for value in history[plan.spec.time_column].to_list()]
    predictions = pl.read_parquet(predictions_path).filter(
        pl.col("model") == plan.spec.forecast_model_id
    )
    prediction_map = {
        int(row["row_index"]): (float(row["actual"]), float(row["prediction"]))
        for row in predictions.iter_rows(named=True)
    }
    expected_indices = {
        index
        for window in result.fold_windows
        for index in range(window.test_start, window.test_end)
    }
    residual_indices = {item.row_index for item in error_artifact.residuals}
    if residual_indices != expected_indices:
        error(
            "residual_coverage",
            "forecast-error rows are not exactly the rolling-origin test windows",
        )
    for item in error_artifact.residuals:
        if item.row_index not in prediction_map:
            error(
                "residual_not_prediction",
                f"residual row {item.row_index} has no out-of-fold prediction",
            )
            continue
        actual, prediction = prediction_map[item.row_index]
        if abs(item.actual - actual) > _TOLERANCE or abs(
            item.forecast - prediction
        ) > _TOLERANCE:
            error(
                "residual_value_mismatch",
                f"residual row {item.row_index} does not match the frozen prediction",
            )
        window = next(
            (
                index
                for index, window in enumerate(result.fold_windows)
                if window.test_start <= item.row_index < window.test_end
            ),
            None,
        )
        if window is None or window != item.fold_id:
            error(
                "residual_fold_mismatch",
                f"residual row {item.row_index} is not in fold {item.fold_id}",
            )
        if item.row_index >= len(timestamps) or timestamps[item.row_index] != item.timestamp:
            error(
                "residual_timestamp_mismatch",
                f"residual row {item.row_index} timestamp is not the frozen series value",
            )
    if error_artifact.forecast_horizon < plan.spec.forecast_horizon:
        error("residual_horizon", "residual validation horizon is shorter than the plan")
    if error_artifact.unit != forecast.unit:
        error("residual_unit", "residual unit does not match the forecast unit")


def _check_simulation_replications(
    spec: MixedSimulationSpec,
    result: MixedSimulationResult,
    forecast: ForecastArtifact,
    plan_artifact: ProductionPlanArtifact,
    residual_pool: list[float],
    error,
) -> None:
    schedule = derive_seed_schedule(spec.master_seed, spec.replications)
    if seed_schedule_hash(schedule) != result.replication_seed_hash:
        error("seed_hash_mismatch", "replication seed hash does not match the schedule")
    expected_ids = list(range(spec.replications))
    actual_ids = sorted(replication.replication_id for replication in result.replications)
    if actual_ids != expected_ids:
        error("replication_ids", "replication ids are not complete and unique")
    if result.n_replications != spec.replications:
        error("replication_count", "simulation result replication count is wrong")
    if result.determinism.get("match") is not True:
        error("simulation_determinism", "deterministic rerun did not match")

    seed_by_id = {
        entry["replication_id"]: entry["replication_seed"] for entry in schedule
    }
    plan_values = plan_artifact.production
    for replication in result.replications:
        if replication.replication_seed != seed_by_id.get(replication.replication_id):
            error(
                "replication_seed",
                f"replication {replication.replication_id} seed does not match the schedule",
            )
        if len(replication.periods) != spec.planning_horizon:
            error(
                "replication_periods",
                f"replication {replication.replication_id} has wrong period count",
            )
            continue
        previous_inventory = 0.0
        total_demand = 0.0
        total_served = 0.0
        total_shortage = 0.0
        for index, period in enumerate(replication.periods):
            if period.timestamp != spec.timestamps[index]:
                error(
                    "replication_timestamp",
                    f"replication {replication.replication_id} timestamp mismatch",
                )
            point = forecast.point_forecasts[index]
            if abs(period.point_forecast - point) > _TOLERANCE:
                error(
                    "replication_point_forecast",
                    f"replication {replication.replication_id} point forecast mismatch",
                )
            if not any(
                abs(period.sampled_residual - value) <= 1e-9
                for value in residual_pool
            ):
                error(
                    "residual_sampling",
                    f"replication {replication.replication_id} sampled a value outside "
                    "the out-of-sample residual pool",
                )
            expected_demand = (
                max(0.0, point + period.sampled_residual)
                if spec.demand_floor_policy == "max_zero"
                else point + period.sampled_residual
            )
            if abs(period.simulated_demand - expected_demand) > _TOLERANCE:
                error(
                    "demand_policy",
                    f"replication {replication.replication_id} demand floor mismatch",
                )
            if abs(period.production - plan_values[index]) > _TOLERANCE:
                error(
                    "plan_mutation",
                    f"replication {replication.replication_id} changed the frozen plan",
                )
            available = previous_inventory + period.production
            served = min(available, period.simulated_demand)
            shortage = max(0.0, period.simulated_demand - available)
            ending = max(0.0, available - period.simulated_demand)
            if abs(period.available - available) > _TOLERANCE:
                error("available_arithmetic", "available inventory arithmetic mismatch")
            if abs(period.served - served) > _TOLERANCE:
                error("served_arithmetic", "served arithmetic mismatch")
            if abs(period.shortage - shortage) > _TOLERANCE:
                error("shortage_arithmetic", "shortage arithmetic mismatch")
            if abs(period.ending_inventory - ending) > _TOLERANCE:
                error("inventory_arithmetic", "ending inventory arithmetic mismatch")
            previous_inventory = ending
            total_demand += period.simulated_demand
            total_served += served
            total_shortage += shortage
        if abs(replication.total_demand - total_demand) > _TOLERANCE:
            error("total_demand", "replication total demand mismatch")
        if abs(replication.total_served - total_served) > _TOLERANCE:
            error("total_served", "replication total served mismatch")
        if abs(replication.total_shortage - total_shortage) > _TOLERANCE:
            error("total_shortage", "replication total shortage mismatch")
        expected_fill = total_served / total_demand if total_demand > 0 else 1.0
        if abs(replication.fill_rate - expected_fill) > _TOLERANCE:
            error("fill_rate", "fill rate mismatch")


def _check_simulation_metrics(
    spec: MixedSimulationSpec,
    result: MixedSimulationResult,
    error,
) -> None:
    with_shortage = sum(
        1
        for replication in result.replications
        if replication.total_shortage > 1e-9
    )
    expected_probability = with_shortage / len(result.replications)
    if abs(result.any_shortage_probability - expected_probability) > _TOLERANCE:
        error("any_shortage_probability", "any-shortage probability mismatch")
    distinct_paths = {
        tuple(period.sampled_residual for period in replication.periods)
        for replication in result.replications
    }
    if result.distinct_residual_paths != len(distinct_paths):
        error("distinct_paths", "distinct residual path count mismatch")
    floor_activations = sum(
        1
        for replication in result.replications
        for period in replication.periods
        if period.point_forecast + period.sampled_residual < 0
    )
    if result.demand_floor_activations != floor_activations:
        error("floor_activations", "demand-floor activation count mismatch")
    for metric in spec.metrics:
        summary = result.metrics.get(metric)
        if summary is None:
            error("missing_metric", f"simulation metric {metric!r} is missing")
            continue
        values = [float(getattr(replication, metric)) for replication in result.replications]
        expected_mean = statistics.fmean(values)
        expected_std = statistics.stdev(values) if len(values) > 1 else 0.0
        expected_median = statistics.median(values)
        expected_low, expected_high = confidence_interval(values, spec.confidence_level)
        checks = {
            "mean": (summary.mean, expected_mean),
            "std": (summary.std, expected_std),
            "median": (summary.median, expected_median),
            "p05": (summary.p05, percentile(values, 5)),
            "p25": (summary.p25, percentile(values, 25)),
            "p75": (summary.p75, percentile(values, 75)),
            "p95": (summary.p95, percentile(values, 95)),
            "ci_low": (summary.ci_low, expected_low),
            "ci_high": (summary.ci_high, expected_high),
        }
        for name, (reported, recomputed) in checks.items():
            if abs(reported - recomputed) > _TOLERANCE:
                error(
                    "metric_summary_mismatch",
                    f"{metric}.{name} reported {reported} != recomputed {recomputed}",
                )
        if summary.n != len(values):
            error("metric_count", f"{metric} summary count is wrong")


def _validate_simulation_core(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    error,
    warning,
) -> None:
    inputs = _simulation_artifacts(paths, plan, run_id, error)
    if inputs is None:
        return
    (
        forecast_envelope,
        forecast,
        error_envelope,
        error_artifact,
        plan_envelope,
        plan_artifact,
    ) = inputs
    experiment_id = f"exp-{plan.plan_id}-simulation"
    try:
        spec = load_mixed_simulation_spec(paths, experiment_id)
        result = load_mixed_simulation_result(paths, experiment_id)
        record = load_mixed_simulation_record(paths, experiment_id)
    except (FileNotFoundError, ValueError) as exc:
        error("simulation_experiment_missing", f"simulation artifacts are missing: {exc}")
        return
    if spec.spec_hash != spec.computed_hash():
        error("simulation_spec_hash", "simulation spec hash does not match its content")
    if result.spec_hash != spec.spec_hash:
        error("simulation_result_spec", "simulation result and spec disagree on the hash")
    if spec.simulation_experiment_id != experiment_id:
        error("simulation_experiment_id", "simulation experiment id mismatch")
    identity_checks = {
        "plan_id": (spec.production_plan_artifact_id, plan_envelope.artifact_id),
        "plan_hash": (spec.production_plan_artifact_hash, plan_envelope.content_hash),
        "forecast_id": (spec.forecast_artifact_id, forecast_envelope.artifact_id),
        "forecast_hash": (spec.forecast_artifact_hash, forecast_envelope.content_hash),
        "error_id": (spec.forecast_error_artifact_id, error_envelope.artifact_id),
        "error_hash": (spec.forecast_error_artifact_hash, error_envelope.content_hash),
    }
    for name, (left, right) in identity_checks.items():
        if left != right:
            error("simulation_identity", f"simulation spec {name} does not match the artifact")
    result_checks = {
        "plan_id": (result.production_plan_artifact_id, plan_envelope.artifact_id),
        "plan_hash": (result.production_plan_artifact_hash, plan_envelope.content_hash),
        "forecast_id": (result.forecast_artifact_id, forecast_envelope.artifact_id),
        "forecast_hash": (result.forecast_artifact_hash, forecast_envelope.content_hash),
        "error_id": (
            result.forecast_error_artifact_id,
            error_envelope.artifact_id,
        ),
        "error_hash": (
            result.forecast_error_artifact_hash,
            error_envelope.content_hash,
        ),
    }
    for name, (left, right) in result_checks.items():
        if left != right:
            error("simulation_result_identity", f"result {name} does not match the artifact")
    units = {
        spec.unit,
        result.unit,
        forecast.unit,
        error_artifact.unit,
        plan_artifact.unit,
    }
    if len(units) != 1:
        error("cross_task_unit", f"cross-task unit mismatch: {sorted(units)}")
    if (
        plan_artifact.timestamps != spec.timestamps
        or forecast.timestamps != spec.timestamps
        or result.timestamps != spec.timestamps
        or len(spec.timestamps) != plan.spec.forecast_horizon
    ):
        error("cross_task_horizon", "cross-task horizon or timestamp alignment mismatch")
    if plan_artifact.forecast_artifact_id != forecast_envelope.artifact_id:
        error("plan_forecast_identity", "plan artifact references the wrong forecast")
    if record.status not in {ExperimentStatus.SUCCEEDED, ExperimentStatus.VALIDATED}:
        error("simulation_status", f"simulation status is {record.status.value}")
    if record.consumed_artifact_ids != [
        plan_envelope.artifact_id,
        error_envelope.artifact_id,
    ]:
        error("simulation_consumed_ids", "simulation consumed artifact ids mismatch")
    if record.consumed_artifact_hashes != [
        plan_envelope.content_hash,
        error_envelope.content_hash,
    ]:
        error("simulation_consumed_hashes", "simulation consumed artifact hashes mismatch")
    if record.dependency_contract_status != "passed":
        error("simulation_dependency_contract", "simulation dependency contract failed")
    if record.replication_seed_hash != result.replication_seed_hash:
        error("simulation_record_seed", "simulation record seed hash mismatch")
    mlflow_path = paths.experiment_dir(experiment_id) / "mlflow_run.json"
    if record.mlflow_run_id is None or not mlflow_path.exists():
        error("simulation_mlflow", "simulation experiment has no MLflow record")
    _check_residual_provenance(
        paths, plan, forecast, forecast_envelope, error_artifact, error
    )
    _check_simulation_replications(
        spec,
        result,
        forecast,
        plan_artifact,
        [item.residual for item in error_artifact.residuals],
        error,
    )
    _check_simulation_metrics(spec, result, error)
    if error_artifact.sample_count < max(2, plan.spec.forecast_horizon):
        warning(
            "small_residual_pool",
            "out-of-sample residual pool is small; uncertainty is limited",
        )


def validate_mixed_simulation_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
) -> list[ValidationReport]:
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.WARNING, message=message))

    _validate_simulation_core(paths, plan, run_id, error, warning)
    report = build_report(
        "mixed_simulation",
        run_id,
        issues,
        evidence=[
            f"experiments/{plan.plan_id}/simulation",
            run_record_path(paths, run_id).relative_to(paths.root).as_posix(),
        ],
        recommendations=[
            "recompute every replication from the frozen plan and residual pool",
            "keep the plan immutable during simulation",
        ],
    )
    experiment_id = f"exp-{plan.plan_id}-simulation"
    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "mixed_simulation.json",
        report.model_dump_json(indent=2) + "\n",
    )
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
    return [report]


def _simulation_link_issues(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    record: DAGRunRecord,
    run_id: str,
    error,
    warning,
) -> None:
    experiment_id = f"exp-{plan.plan_id}-simulation"
    simulation_node = next(
        (node for node in record.nodes if node.task_family is TaskFamily.SIMULATION),
        None,
    )
    if simulation_node is None:
        error("missing_simulation_node", "the DAG has no simulation node")
        return
    if record.simulation_experiment_id != experiment_id:
        error("run_simulation_experiment", "run record simulation experiment mismatch")
    if simulation_node.experiment_id != experiment_id:
        error("node_simulation_experiment", "simulation node experiment mismatch")
    try:
        spec = load_mixed_simulation_spec(paths, experiment_id)
        result = load_mixed_simulation_result(paths, experiment_id)
    except (FileNotFoundError, ValueError):
        error("simulation_experiment_missing", "simulation experiment is missing")
        return
    if record.production_plan_artifact_id != spec.production_plan_artifact_id:
        error("run_plan_id", "run record production plan id mismatch")
    if record.production_plan_artifact_hash != spec.production_plan_artifact_hash:
        error("run_plan_hash", "run record production plan hash mismatch")
    if record.forecast_error_artifact_id != spec.forecast_error_artifact_id:
        error("run_error_id", "run record forecast-error id mismatch")
    if record.forecast_error_artifact_hash != spec.forecast_error_artifact_hash:
        error("run_error_hash", "run record forecast-error hash mismatch")
    if record.simulation_replications != result.n_replications:
        error("run_simulation_replications", "run record replication count mismatch")
    if not record.contract_chain_complete:
        error("contract_chain", "run record contract chain is incomplete")
    try:
        reported_shortage = float(record.shortage_summary.get("mean_total_shortage", "nan"))
        reported_fill = float(record.service_summary.get("mean_fill_rate", "nan"))
    except ValueError:
        error("run_simulation_summary", "run record simulation summary is malformed")
        return
    if math.isnan(reported_shortage) or abs(
        reported_shortage - result.metrics["total_shortage"].mean
    ) > _TOLERANCE:
        error("run_shortage_summary", "run record shortage summary mismatch")
    if math.isnan(reported_fill) or abs(
        reported_fill - result.metrics["fill_rate"].mean
    ) > _TOLERANCE:
        error("run_service_summary", "run record service summary mismatch")
    passed, reason = _summary_passed(paths, experiment_id)
    if not passed:
        error("simulation_validation", f"simulation validation failed: {reason}")
    if not _evidence_frozen(paths, f"mixed:{run_id}"):
        warning("simulation_evidence_pending", "mixed DAG evidence is not frozen yet")


def validate_mixed_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
) -> list[ValidationReport]:
    reports = [validate_mixed_run(paths, plan, run_id)]
    if plan.has_simulation_node():
        reports.extend(validate_mixed_simulation_experiment(paths, plan, run_id))
    output_dir = paths.mixed_dir / "runs" / run_id / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "mixed.json", reports[0].model_dump_json(indent=2) + "\n"
    )
    if len(reports) > 1:
        atomic_write_text(
            output_dir / "mixed_simulation.json",
            reports[1].model_dump_json(indent=2) + "\n",
        )
    blocking = any(report.blocking for report in reports)
    summary = {
        "experiment_id": run_id,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "blocking": blocking,
        "reports": {
            report.validator: {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
            for report in reports
        },
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    return reports


__all__ = [
    "run_record_path",
    "validate_mixed_experiment",
    "validate_mixed_run",
    "validate_mixed_simulation_experiment",
]
