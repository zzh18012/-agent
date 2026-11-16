import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_manifest,
    save_experiment_record,
    save_experiment_spec,
    save_optimization_spec,
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
    contract_for,
    load_envelope,
    save_forecast_artifact,
    verify_forecast_artifact,
)
from mmos.mixed.plan import MixedPlanError
from mmos.schemas.experiment import ExperimentSpec, TemporalSplit
from mmos.schemas.mixed import (
    DAGRunRecord,
    DAGStatus,
    ForecastArtifact,
    MixedExecutionPlan,
    NodeStatus,
    TaskNode,
)
from mmos.schemas.optimization import (
    DecisionVariable,
    ExperimentStatus,
    LinearConstraint,
    OptimizationModel,
    OptimizationSpec,
    VariableType,
    optimization_instance_hash,
)
from mmos.tracking import log_optimization_run
from mmos.validation import validate_experiment
from mmos.validation.optimization import validate_optimization_experiment


class MixedExecutionError(RuntimeError):
    pass


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
) -> tuple[ExperimentSpec, ForecastArtifact, Path, Path, dict[str, str]]:
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
    history_payload = [
        [str(period), float(value)]
        for period, value in zip(
            df[spec.time_column].to_list(),
            y.tolist(),
            strict=True,
        )
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
    artifact = build_forecast_artifact(
        spec,
        y,
        forecast_experiment_id=experiment_id,
        source_dataset_hash=experiment_spec.dataset_hash,
    )
    mlflow_path = paths.experiment_dir(experiment_id) / "mlflow_run.json"
    validation_summary = (
        paths.experiment_dir(experiment_id) / "validation" / "summary.json"
    )
    provenance = {
        "dataset_hash": experiment_spec.dataset_hash,
        "forecast_experiment_id": experiment_id,
        "forecast_spec_path": paths.experiment_dir(experiment_id)
        .joinpath("experiment_spec.json")
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
    payload_path, envelope_path, envelope = save_forecast_artifact(
        paths, plan, run_id, artifact, provenance=provenance
    )
    return experiment_spec, artifact, payload_path, envelope_path, {
        "artifact_id": envelope.artifact_id,
        "artifact_hash": envelope.content_hash,
        "payload_hash": envelope.payload_hash,
        "envelope_path": envelope_path.relative_to(paths.root).as_posix(),
    }


def _optimization_experiment(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    artifact: ForecastArtifact,
    envelope_hash: str,
    artifact_id: str,
) -> tuple[OptimizationSpec, float, bool, dict[str, object]]:
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
    result = run_optimization(paths, experiment_id)
    solution = next(
        (item for item in result.solutions if item.method == "ortools.cbc"), None
    )
    if solution is None or not solution.feasible or solution.objective_value is None:
        raise MixedExecutionError("production-planning MILP is infeasible")
    record = build_experiment_record(
        paths,
        optimization_spec,
        result,
        status=ExperimentStatus.SUCCEEDED,
        linked_experiment_ids=[artifact.forecast_experiment_id],
    ).model_copy(
        update={
            "consumed_artifact_ids": [artifact_id],
            "consumed_artifact_hashes": [envelope_hash],
            "dependency_contract_status": "passed",
        }
    )
    mlflow = log_optimization_run(paths, optimization_spec, result, record)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_experiment_record(paths, record)
    reports = validate_optimization_experiment(paths, experiment_id)
    if any(report.blocking for report in reports):
        record = record.model_copy(update={"status": ExperimentStatus.INVALID})
        save_experiment_record(paths, record)
        raise MixedExecutionError("production-planning validation is blocking")
    record = record.model_copy(
        update={
            "status": ExperimentStatus.VALIDATED,
            "validation_status": {
                report.validator: report.status.value for report in reports
            },
        }
    )
    save_experiment_record(paths, record)
    freeze_optimization(paths, experiment_id)
    return (
        optimization_spec,
        solution.objective_value,
        solution.feasible,
        {
            "optimization_experiment_id": experiment_id,
            "mlflow_run_id": mlflow.run_id,
            "validation": {
                report.validator: report.status.value for report in reports
            },
        },
    )


def _mark_previous_runs_stale(
    paths: ProjectPaths, plan: MixedExecutionPlan, current_hash: str
) -> None:
    runs_dir = paths.mixed_dir / "runs"
    if not runs_dir.exists():
        return
    for path in sorted(runs_dir.glob("*/dag_run.json")):
        record = DAGRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.plan_id != plan.plan_id:
            continue
        if (
            record.status is DAGStatus.SUCCEEDED
            and record.forecast_artifact_hash
            and record.forecast_artifact_hash != current_hash
        ):
            stale = record.model_copy(
                update={
                    "status": DAGStatus.STALE,
                    "notes": [
                        *record.notes,
                        "upstream forecast artifact hash changed; downstream plan is stale",
                    ],
                }
            )
            atomic_write_text(path, stale.model_dump_json(indent=2) + "\n")


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
    envelopes: dict[str, Path] = {}
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
                if edge.producer_node not in envelopes:
                    record = _update_node(
                        record,
                        node_id,
                        status=NodeStatus.BLOCKED_UPSTREAM,
                        reason=f"artifact from {edge.producer_node} is missing",
                    )
                    record = record.model_copy(
                        update={"status": DAGStatus.BLOCKED, "completed_at": _timestamp()}
                    )
                    _write_run_record(paths, record)
                    return record
            record = _update_node(record, node_id, status=NodeStatus.RUNNING)
            _write_run_record(paths, record)
            if node.task_family.value == "forecasting":
                experiment_spec, artifact, _, envelope_path, artifact_info = (
                    _forecast_experiment(paths, plan, run_id)
                )
                envelopes[node_id] = envelope_path
                _mark_previous_runs_stale(
                    paths, plan, artifact_info["artifact_hash"]
                )
                record = _update_node(
                    record,
                    node_id,
                    status=NodeStatus.SUCCEEDED,
                    experiment_id=experiment_spec.experiment_id,
                ).model_copy(
                    update={
                        "forecast_artifact_id": artifact_info["artifact_id"],
                        "forecast_artifact_hash": artifact_info["artifact_hash"],
                    }
                )
            elif node.task_family.value == "optimization":
                envelope = load_envelope(envelopes["forecast_node"])
                artifact = verify_forecast_artifact(
                    paths, envelope, contract_for(plan.spec)
                )
                (
                    optimization_spec,
                    objective,
                    feasible,
                    optimization_info,
                ) = _optimization_experiment(
                    paths,
                    plan,
                    run_id,
                    artifact,
                    envelope.content_hash,
                    envelope.artifact_id,
                )
                record = _update_node(
                    record,
                    node_id,
                    status=NodeStatus.SUCCEEDED,
                    experiment_id=optimization_spec.experiment_id,
                ).model_copy(
                    update={
                        "optimization_experiment_id": optimization_info[
                            "optimization_experiment_id"
                        ],
                        "final_objective": objective,
                        "final_feasible": feasible,
                        "dependency_validation": {
                            "status": "passed",
                            "artifact_id": envelope.artifact_id,
                            "artifact_hash": envelope.content_hash,
                        },
                    }
                )
            _write_run_record(paths, record)
    except (MixedExecutionError, ArtifactContractError) as exc:
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


__all__ = ["MixedExecutionError", "run_mixed_execution"]
