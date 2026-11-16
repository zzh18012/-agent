import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_manifest,
    save_experiment_record,
    save_experiment_spec,
    save_optimization_spec,
    sha256_file,
)
from mmos.data import DataAgent
from mmos.data.table import load_single_table
from mmos.evidence import freeze_experiment, freeze_optimization
from mmos.experiments.optimization_runner import (
    build_experiment_record,
    run_optimization,
)
from mmos.experiments.workflow import run_experiment_workflow
from mmos.mixed.artifacts import (
    ArtifactContractError,
    build_forecast_artifact,
    build_forecast_error_artifact,
    build_production_plan_artifact,
    contract_for,
    contract_for_forecast_error,
    contract_for_production_plan,
    load_envelope,
    save_forecast_artifact,
    save_forecast_error_artifact,
    save_production_plan_artifact,
    verify_forecast_artifact,
    verify_forecast_error_artifact,
    verify_production_plan_artifact,
)
from mmos.mixed.plan import MixedPlanError
from mmos.mixed.simulation import (
    MixedSimulationError,
    build_mixed_simulation_record,
    run_mixed_simulation,
    save_mixed_simulation_record,
    save_mixed_simulation_spec,
)
from mmos.mixed.staleness import apply_stale_propagation
from mmos.mixed.validation import validate_mixed_simulation_experiment
from mmos.schemas.experiment import ExperimentResult, ExperimentSpec, TemporalSplit
from mmos.schemas.mixed import (
    ArtifactEnvelope,
    DAGRunRecord,
    DAGStatus,
    ForecastArtifact,
    ForecastErrorArtifact,
    MixedExecutionPlan,
    NodeStatus,
    ProductionPlanArtifact,
    TaskNode,
)
from mmos.schemas.mixed_simulation import (
    MixedSimulationRecord,
    MixedSimulationResult,
    MixedSimulationSpec,
)
from mmos.schemas.optimization import (
    DecisionVariable,
    ExperimentStatus,
    LinearConstraint,
    OptimizationModel,
    OptimizationResult,
    OptimizationSpec,
    VariableType,
    optimization_instance_hash,
)
from mmos.schemas.states import TaskFamily
from mmos.tracking import log_mixed_simulation_run, log_optimization_run
from mmos.validation import validate_experiment
from mmos.validation.optimization import validate_optimization_experiment


class MixedExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ForecastNodeOutputs:
    experiment_spec: ExperimentSpec
    forecast: ForecastArtifact
    forecast_envelope: ArtifactEnvelope
    forecast_envelope_path: Path
    error: ForecastErrorArtifact | None
    error_envelope: ArtifactEnvelope | None
    error_envelope_path: Path | None


@dataclass(frozen=True)
class OptimizationNodeOutputs:
    spec: OptimizationSpec
    objective: float
    feasible: bool
    plan: ProductionPlanArtifact
    plan_envelope: ArtifactEnvelope
    plan_envelope_path: Path
    validation: dict[str, str]


@dataclass(frozen=True)
class SimulationNodeOutputs:
    spec: MixedSimulationSpec
    result: MixedSimulationResult
    record: MixedSimulationRecord
    validation: dict[str, str]


def _timestamp() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _run_record_path(paths: ProjectPaths, run_id: str) -> Path:
    return paths.mixed_dir / "runs" / run_id / "dag_run.json"


def _write_run_record(paths: ProjectPaths, record: DAGRunRecord) -> Path:
    target = _run_record_path(paths, record.run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, record.model_dump_json(indent=2) + "\n")
    return target


def _node(record: DAGRunRecord, node_id: str) -> TaskNode:
    return next(node for node in record.nodes if node.node_id == node_id)


def _update_node(
    record: DAGRunRecord,
    node_id: str,
    *,
    status: NodeStatus,
    reason: str | None = None,
    experiment_id: str | None = None,
) -> DAGRunRecord:
    nodes = [
        node.model_copy(
            update={
                "execution_status": status,
                "status_reason": reason,
                "experiment_id": experiment_id or node.experiment_id,
            }
        )
        if node.node_id == node_id
        else node
        for node in record.nodes
    ]
    return record.model_copy(update={"nodes": nodes})


def _forecast_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
) -> ForecastNodeOutputs:
    spec = plan.spec
    data_result = DataAgent(paths).analyze(target=spec.target_column)
    if data_result.quality.blocking or data_result.leakage.blocking:
        raise MixedExecutionError("data gate blocked the forecasting node")
    df = load_single_table(paths)
    if spec.time_column not in df.columns:
        raise MixedExecutionError(f"time column {spec.time_column!r} is missing")
    df = df.sort(spec.time_column)
    if df[spec.target_column].null_count() > 0:
        raise MixedExecutionError("target contains nulls")
    y = df[spec.target_column].to_numpy()
    if not np.issubdtype(y.dtype, np.number):
        raise MixedExecutionError("forecast target must be numeric")
    n_rows = df.height
    timestamps = [str(value) for value in df[spec.time_column].to_list()]
    history_payload = [
        [str(period), float(value)]
        for period, value in zip(timestamps, y.tolist(), strict=True)
    ]
    history_digest = hashlib.sha256(
        json.dumps(history_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    min_train_size = n_rows - spec.n_splits * spec.forecast_horizon
    if min_train_size < 10:
        raise MixedExecutionError("not enough rows for the frozen forecast folds")
    experiment_id = f"exp-{plan.plan_id}-forecast"
    experiment_spec = ExperimentSpec(
        experiment_id=experiment_id,
        problem_id=plan.project_id,
        proposal_id="mixed-forecast-node",
        dataset_hash=dataset_fingerprint(load_manifest(paths)),
        target_column=spec.target_column,
        feature_set=[],
        split_strategy="rolling_origin",
        validation_strategy="rolling_origin",
        n_splits=spec.n_splits,
        models=sorted({spec.forecast_model_id, "naive.last_value"}),
        metrics=["rmse", "mae", "mase"],
        random_seed=spec.random_seed,
        required_artifacts=["metrics.json", "predictions.parquet"],
        acceptance_criteria=["produce a validated future forecast artifact"],
        failure_criteria=["temporal validation contains a blocking issue"],
        compute_requirements="temporal baseline (cpu)",
        expected_runtime_class="seconds",
        time_column=spec.time_column,
        seasonal_period=spec.seasonal_period,
        temporal_split=TemporalSplit(
            horizon=spec.forecast_horizon,
            min_train_size=min_train_size,
            step=spec.forecast_horizon,
            n_splits=spec.n_splits,
        ),
    )
    save_experiment_spec(paths, experiment_spec)
    execution = run_experiment_workflow(paths, experiment_id)
    if not execution.succeeded():
        raise MixedExecutionError("forecasting workflow failed")
    reports = validate_experiment(paths, experiment_id)
    if any(report.blocking for report in reports):
        raise MixedExecutionError("forecasting validation is blocking")
    freeze_experiment(paths, experiment_id)
    forecast = build_forecast_artifact(
        spec,
        y,
        forecast_experiment_id=experiment_id,
        source_dataset_hash=experiment_spec.dataset_hash,
    )
    experiment_dir = paths.experiment_dir(experiment_id)
    mlflow_path = experiment_dir / "mlflow_run.json"
    validation_summary = experiment_dir / "validation" / "summary.json"
    provenance = {
        "dataset_hash": experiment_spec.dataset_hash,
        "forecast_experiment_id": experiment_id,
        "forecast_spec_path": experiment_dir.joinpath("experiment_spec.json")
        .relative_to(paths.root)
        .as_posix(),
        "validation_summary_path": validation_summary.relative_to(paths.root).as_posix(),
        "mlflow_run_path": mlflow_path.relative_to(paths.root).as_posix()
        if mlflow_path.exists()
        else "",
        "plan_hash": plan.plan_hash,
        "history_rows": str(n_rows),
        "history_digest": history_digest,
    }
    _, forecast_envelope_path, forecast_envelope = save_forecast_artifact(
        paths, plan, run_id, forecast, provenance=provenance
    )
    error: ForecastErrorArtifact | None = None
    error_envelope: ArtifactEnvelope | None = None
    error_envelope_path: Path | None = None
    if plan.has_simulation_node():
        result = ExperimentResult.model_validate_json(
            experiment_dir.joinpath("artifacts", "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        predictions_path = experiment_dir / "artifacts" / "predictions.parquet"
        error = build_forecast_error_artifact(
            spec,
            pl.read_parquet(predictions_path),
            timestamps,
            result.fold_windows,
            forecast_experiment_id=experiment_id,
            forecast_artifact_id=forecast_envelope.artifact_id,
            forecast_artifact_hash=forecast_envelope.content_hash,
        )
        error_provenance = {
            **provenance,
            "source_forecast_artifact_id": forecast_envelope.artifact_id,
            "source_forecast_artifact_hash": forecast_envelope.content_hash,
            "predictions_path": predictions_path.relative_to(paths.root).as_posix(),
            "predictions_sha256": sha256_file(predictions_path),
            "residual_source": "rolling_origin_oof",
            "residual_definition": "actual_minus_forecast",
        }
        _, error_envelope_path, error_envelope = save_forecast_error_artifact(
            paths, plan, run_id, error, provenance=error_provenance
        )
    return ForecastNodeOutputs(
        experiment_spec=experiment_spec,
        forecast=forecast,
        forecast_envelope=forecast_envelope,
        forecast_envelope_path=forecast_envelope_path,
        error=error,
        error_envelope=error_envelope,
        error_envelope_path=error_envelope_path,
    )


def _optimization_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    artifact: ForecastArtifact,
    envelope: ArtifactEnvelope,
) -> OptimizationNodeOutputs:
    spec = plan.spec
    periods = len(spec.planning_periods)
    demand = artifact.point_forecasts
    if any(value < -1e-9 for value in demand):
        raise MixedExecutionError("forecast contains negative demand")
    variables = []
    objective = {}
    constraints = []
    for index in range(periods):
        production = f"production_{index}"
        inventory = f"inventory_{index}"
        variables.append(
            DecisionVariable(
                name=production,
                type=VariableType.CONTINUOUS,
                lower=0.0,
                upper=float(spec.capacities[index]),
            )
        )
        variables.append(
            DecisionVariable(
                name=inventory, type=VariableType.CONTINUOUS, lower=0.0
            )
        )
        objective[production] = float(spec.production_costs[index])
        objective[inventory] = float(spec.holding_costs[index])
        if index == 0:
            coefficients = {production: 1.0, inventory: -1.0}
            rhs = float(demand[0]) - spec.initial_inventory
        else:
            coefficients = {
                production: 1.0,
                inventory: -1.0,
                f"inventory_{index - 1}": 1.0,
            }
            rhs = float(demand[index])
        constraints.append(
            LinearConstraint(
                name=f"inventory_balance_{index}",
                coefficients=coefficients,
                lower=rhs,
                upper=rhs,
            )
        )
        constraints.append(
            LinearConstraint(
                name=f"capacity_{index}",
                coefficients={production: 1.0},
                upper=float(spec.capacities[index]),
            )
        )
    model = OptimizationModel(
        name=f"production_planning_{plan.plan_id}",
        sense="minimize",
        variables=variables,
        objective=objective,
        constraints=constraints,
    )
    experiment_id = f"exp-{plan.plan_id}-optimization"
    optimization_spec = OptimizationSpec(
        experiment_id=experiment_id,
        problem_id=plan.project_id,
        model=model,
        solver="ortools.cbc",
        time_limit_seconds=spec.solver_time_limit_seconds,
        baseline="none",
        acceptance_criteria=["inventory balance and capacities must hold"],
    )
    optimization_spec = optimization_spec.model_copy(
        update={"instance_hash": optimization_instance_hash(model)}
    )
    save_optimization_spec(paths, optimization_spec)
    result: OptimizationResult = run_optimization(paths, experiment_id)
    solution = next(
        (item for item in result.solutions if item.method == "ortools.cbc"), None
    )
    if solution is None or not solution.feasible or solution.objective_value is None:
        raise MixedExecutionError("production-planning LP is infeasible")
    experiment_dir = paths.experiment_dir(experiment_id)
    plan_artifact = build_production_plan_artifact(
        spec,
        result,
        optimization_experiment_id=experiment_id,
        forecast_artifact_id=envelope.artifact_id,
        forecast_artifact_hash=envelope.content_hash,
    )
    plan_provenance = {
        "plan_hash": plan.plan_hash,
        "optimization_experiment_id": experiment_id,
        "optimization_spec_path": experiment_dir.joinpath("optimization_spec.json")
        .relative_to(paths.root)
        .as_posix(),
        "optimization_spec_sha256": sha256_file(
            experiment_dir / "optimization_spec.json"
        ),
        "optimization_result_path": experiment_dir.joinpath(
            "artifacts", "optimization_result.json"
        )
        .relative_to(paths.root)
        .as_posix(),
        "optimization_result_sha256": sha256_file(
            experiment_dir / "artifacts" / "optimization_result.json"
        ),
        "forecast_artifact_id": envelope.artifact_id,
        "forecast_artifact_hash": envelope.content_hash,
        "instance_hash": optimization_spec.instance_hash or "",
        "solver": result.solver_backend,
        "solver_version": result.solver_version,
    }
    _, plan_envelope_path, plan_envelope = save_production_plan_artifact(
        paths, plan, run_id, plan_artifact, provenance=plan_provenance
    )
    record = build_experiment_record(
        paths,
        optimization_spec,
        result,
        status=ExperimentStatus.SUCCEEDED,
        linked_experiment_ids=[artifact.forecast_experiment_id],
    ).model_copy(
        update={
            "consumed_artifact_ids": [envelope.artifact_id],
            "consumed_artifact_hashes": [envelope.content_hash],
            "dependency_contract_status": "passed",
        }
    )
    mlflow = log_optimization_run(paths, optimization_spec, result, record)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_experiment_record(paths, record)
    reports = validate_optimization_experiment(paths, experiment_id)
    validation = {report.validator: report.status.value for report in reports}
    if any(report.blocking for report in reports):
        record = record.model_copy(update={"status": ExperimentStatus.INVALID})
        save_experiment_record(paths, record)
        raise MixedExecutionError("production-planning validation is blocking")
    record = record.model_copy(
        update={"status": ExperimentStatus.VALIDATED, "validation_status": validation}
    )
    save_experiment_record(paths, record)
    freeze_optimization(paths, experiment_id)
    return OptimizationNodeOutputs(
        spec=optimization_spec,
        objective=float(solution.objective_value),
        feasible=bool(solution.feasible),
        plan=plan_artifact,
        plan_envelope=plan_envelope,
        plan_envelope_path=plan_envelope_path,
        validation=validation,
    )


def _simulation_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    forecast: ArtifactEnvelope,
    forecast_artifact: ForecastArtifact,
    error: ArtifactEnvelope,
    error_artifact: ForecastErrorArtifact,
    plan_envelope: ArtifactEnvelope,
    plan_artifact: ProductionPlanArtifact,
) -> SimulationNodeOutputs:
    spec = plan.spec
    if spec.simulation_replications is None:
        raise MixedExecutionError("simulation node requires simulation replications")
    experiment_id = f"exp-{plan.plan_id}-simulation"
    simulation_spec = MixedSimulationSpec(
        simulation_experiment_id=experiment_id,
        simulation_id=f"sim-{plan.plan_id}-stress",
        problem_id=plan.project_id,
        project_id=paths.root.name,
        unit=spec.unit,
        planning_horizon=spec.forecast_horizon,
        timestamps=list(spec.planning_periods),
        production_plan_artifact_id=plan_envelope.artifact_id,
        production_plan_artifact_hash=plan_envelope.content_hash,
        forecast_artifact_id=forecast.artifact_id,
        forecast_artifact_hash=forecast.content_hash,
        forecast_error_artifact_id=error.artifact_id,
        forecast_error_artifact_hash=error.content_hash,
        replications=spec.simulation_replications,
        master_seed=spec.simulation_master_seed,
        residual_sampling_method=spec.residual_sampling_method,
        demand_floor_policy=spec.demand_floor_policy,
        confidence_level=spec.confidence_level,
    ).with_hash()
    save_mixed_simulation_spec(paths, simulation_spec)
    result = run_mixed_simulation(
        paths, simulation_spec, plan_artifact, forecast_artifact, error_artifact
    )
    record = build_mixed_simulation_record(
        paths, simulation_spec, result, status=ExperimentStatus.SUCCEEDED
    ).model_copy(
        update={
            "consumed_artifact_ids": [plan_envelope.artifact_id, error.artifact_id],
            "consumed_artifact_hashes": [
                plan_envelope.content_hash,
                error.content_hash,
            ],
            "dependency_contract_status": "passed",
        }
    )
    mlflow = log_mixed_simulation_run(paths, simulation_spec, result, record)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_mixed_simulation_record(paths, record)
    reports = validate_mixed_simulation_experiment(paths, plan, run_id)
    validation = {report.validator: report.status.value for report in reports}
    if any(report.blocking for report in reports):
        record = record.model_copy(update={"status": ExperimentStatus.INVALID})
        save_mixed_simulation_record(paths, record)
        raise MixedExecutionError("mixed simulation validation is blocking")
    record = record.model_copy(
        update={"status": ExperimentStatus.VALIDATED, "validation_status": validation}
    )
    save_mixed_simulation_record(paths, record)
    return SimulationNodeOutputs(
        spec=simulation_spec,
        result=result,
        record=record,
        validation=validation,
    )


def run_mixed_execution(
    paths: ProjectPaths, plan: MixedExecutionPlan
) -> DAGRunRecord:
    if plan.plan_hash != plan.computed_hash():
        raise MixedPlanError("plan hash does not match its content")
    order = plan.topological_order()
    if not order:
        raise MixedPlanError("plan has no executable nodes")
    run_id = f"mixed-{plan.plan_hash[:8]}-{secrets.token_hex(3)}"
    started = _timestamp()
    record = DAGRunRecord(
        run_id=run_id,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        project_id=paths.root.name,
        status=DAGStatus.RUNNING,
        nodes=[node.model_copy() for node in plan.nodes],
        started_at=started,
    )
    _write_run_record(paths, record)
    produced: dict[tuple[str, str], Path] = {}
    node_id = order[0]
    try:
        for node_id in order:
            node = _node(record, node_id)
            for edge in plan.edges:
                if edge.consumer_node != node_id:
                    continue
                upstream = _node(record, edge.producer_node)
                if upstream.execution_status is not NodeStatus.SUCCEEDED:
                    record = _update_node(
                        record,
                        node_id,
                        status=NodeStatus.BLOCKED_UPSTREAM,
                        reason=f"upstream {edge.producer_node} did not succeed",
                    )
                    record = record.model_copy(
                        update={"status": DAGStatus.BLOCKED, "completed_at": _timestamp()}
                    )
                    _write_run_record(paths, record)
                    return record
                if (edge.producer_node, edge.producer_output) not in produced:
                    record = _update_node(
                        record,
                        node_id,
                        status=NodeStatus.BLOCKED_UPSTREAM,
                        reason=(
                            f"artifact {edge.producer_output} from "
                            f"{edge.producer_node} is missing"
                        ),
                    )
                    record = record.model_copy(
                        update={"status": DAGStatus.BLOCKED, "completed_at": _timestamp()}
                    )
                    _write_run_record(paths, record)
                    return record
            record = _update_node(record, node_id, status=NodeStatus.RUNNING)
            _write_run_record(paths, record)
            if node.task_family is TaskFamily.FORECASTING:
                outputs = _forecast_experiment(paths, plan, run_id)
                produced[("forecast_node", "demand_forecast")] = (
                    outputs.forecast_envelope_path
                )
                if outputs.error_envelope_path is not None:
                    produced[("forecast_node", "forecast_error")] = (
                        outputs.error_envelope_path
                    )
                apply_stale_propagation(
                    paths,
                    plan.plan_id,
                    forecast_hash=outputs.forecast_envelope.content_hash,
                    forecast_error_hash=outputs.error_envelope.content_hash
                    if outputs.error_envelope is not None
                    else None,
                )
                record = _update_node(
                    record,
                    node_id,
                    status=NodeStatus.SUCCEEDED,
                    experiment_id=outputs.experiment_spec.experiment_id,
                ).model_copy(
                    update={
                        "forecast_artifact_id": outputs.forecast_envelope.artifact_id,
                        "forecast_artifact_hash": outputs.forecast_envelope.content_hash,
                        "forecast_error_artifact_id": outputs.error_envelope.artifact_id
                        if outputs.error_envelope is not None
                        else None,
                        "forecast_error_artifact_hash": outputs.error_envelope.content_hash
                        if outputs.error_envelope is not None
                        else None,
                    }
                )
            elif node.task_family is TaskFamily.OPTIMIZATION:
                forecast_key = ("forecast_node", "demand_forecast")
                if forecast_key not in produced:
                    raise MixedExecutionError("forecast artifact is missing")
                forecast_envelope = load_envelope(produced[forecast_key])
                forecast_artifact = verify_forecast_artifact(
                    paths, forecast_envelope, contract_for(plan.spec)
                )
                outputs = _optimization_experiment(
                    paths, plan, run_id, forecast_artifact, forecast_envelope
                )
                produced[("optimization_node", "production_plan")] = (
                    outputs.plan_envelope_path
                )
                apply_stale_propagation(
                    paths,
                    plan.plan_id,
                    production_plan_hash=outputs.plan_envelope.content_hash,
                )
                record = _update_node(
                    record,
                    node_id,
                    status=NodeStatus.SUCCEEDED,
                    experiment_id=outputs.spec.experiment_id,
                ).model_copy(
                    update={
                        "optimization_experiment_id": outputs.spec.experiment_id,
                        "production_plan_artifact_id": outputs.plan_envelope.artifact_id,
                        "production_plan_artifact_hash": outputs.plan_envelope.content_hash,
                        "final_objective": outputs.objective,
                        "final_feasible": outputs.feasible,
                        "dependency_validation": {
                            "status": "passed",
                            "artifact_id": forecast_envelope.artifact_id,
                            "artifact_hash": forecast_envelope.content_hash,
                        },
                    }
                )
            elif node.task_family is TaskFamily.SIMULATION:
                forecast_key = ("forecast_node", "demand_forecast")
                error_key = ("forecast_node", "forecast_error")
                plan_key = ("optimization_node", "production_plan")
                missing_keys = [
                    name
                    for name, key in (
                        ("forecast", forecast_key),
                        ("forecast_error", error_key),
                        ("production_plan", plan_key),
                    )
                    if key not in produced
                ]
                if missing_keys:
                    raise MixedExecutionError(
                        "simulation inputs are missing: " + ", ".join(missing_keys)
                    )
                forecast_envelope = load_envelope(produced[forecast_key])
                forecast_artifact = verify_forecast_artifact(
                    paths, forecast_envelope, contract_for(plan.spec)
                )
                error_envelope = load_envelope(produced[error_key])
                error_artifact = verify_forecast_error_artifact(
                    paths,
                    error_envelope,
                    contract_for_forecast_error(
                        plan.spec,
                        forecast_artifact.forecast_experiment_id,
                        forecast_envelope.content_hash,
                    ),
                )
                plan_envelope = load_envelope(produced[plan_key])
                plan_artifact = verify_production_plan_artifact(
                    paths,
                    plan_envelope,
                    contract_for_production_plan(
                        plan.spec,
                        forecast_envelope.artifact_id,
                        forecast_envelope.content_hash,
                    ),
                )
                outputs = _simulation_experiment(
                    paths,
                    plan,
                    run_id,
                    forecast_envelope,
                    forecast_artifact,
                    error_envelope,
                    error_artifact,
                    plan_envelope,
                    plan_artifact,
                )
                shortage_summary = {
                    "mean_total_shortage": f"{outputs.result.metrics['total_shortage'].mean:.6f}",
                    "mean_shortage_period_count": (
                        f"{outputs.result.metrics['shortage_period_count'].mean:.6f}"
                    ),
                    "max_shortage_p95": f"{outputs.result.metrics['max_shortage'].p95:.6f}",
                }
                service_summary = {
                    "mean_fill_rate": f"{outputs.result.metrics['fill_rate'].mean:.6f}",
                    "any_shortage_probability": (
                        f"{outputs.result.any_shortage_probability:.6f}"
                    ),
                    "replications": str(outputs.result.n_replications),
                }
                record = _update_node(
                    record,
                    node_id,
                    status=NodeStatus.SUCCEEDED,
                    experiment_id=outputs.spec.simulation_experiment_id,
                ).model_copy(
                    update={
                        "simulation_experiment_id": (
                            outputs.spec.simulation_experiment_id
                        ),
                        "simulation_replications": outputs.result.n_replications,
                        "shortage_summary": shortage_summary,
                        "service_summary": service_summary,
                        "contract_chain_complete": True,
                    }
                )
            _write_run_record(paths, record)
    except (MixedExecutionError, ArtifactContractError, MixedSimulationError) as exc:
        record = _update_node(
            record,
            node_id,
            status=NodeStatus.FAILED,
            reason=str(exc),
        ).model_copy(
            update={
                "status": DAGStatus.FAILED,
                "completed_at": _timestamp(),
                "notes": [*record.notes, str(exc)],
            }
        )
        _write_run_record(paths, record)
        return record
    except Exception as exc:
        record = _update_node(
            record,
            node_id,
            status=NodeStatus.FAILED,
            reason=str(exc),
        ).model_copy(
            update={
                "status": DAGStatus.FAILED,
                "completed_at": _timestamp(),
                "notes": [*record.notes, f"unexpected failure: {exc}"],
            }
        )
        _write_run_record(paths, record)
        raise
    record = record.model_copy(
        update={
            "status": DAGStatus.SUCCEEDED,
            "completed_at": _timestamp(),
            "provenance_complete": True,
        }
    )
    _write_run_record(paths, record)
    return record


__all__ = [
    "MixedExecutionError",
    "run_mixed_execution",
]
