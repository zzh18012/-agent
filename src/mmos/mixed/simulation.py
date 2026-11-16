import csv
import hashlib
import io
import json
import math
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
)
from mmos.experiments.simulation_runner import (
    confidence_interval,
    derive_seed_schedule,
    seed_schedule_hash,
)
from mmos.schemas.mixed import ForecastArtifact, ForecastErrorArtifact, ProductionPlanArtifact
from mmos.schemas.mixed_simulation import (
    MIXED_SIMULATION_METRICS,
    MixedSimulationMetricSummary,
    MixedSimulationRecord,
    MixedSimulationReplication,
    MixedSimulationResult,
    MixedSimulationSpec,
    SimulationPeriodResult,
)
from mmos.schemas.optimization import ExperimentStatus


class MixedSimulationError(RuntimeError):
    pass


def mixed_simulation_spec_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return paths.experiment_dir(experiment_id) / "mixed_simulation_spec.json"


def mixed_simulation_result_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return (
        paths.experiment_dir(experiment_id)
        / "artifacts"
        / "mixed_simulation_result.json"
    )


def mixed_simulation_record_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return paths.experiment_dir(experiment_id) / "mixed_simulation_record.json"


def save_mixed_simulation_spec(
    paths: ProjectPaths, spec: MixedSimulationSpec
) -> Path:
    target = mixed_simulation_spec_file(paths, spec.simulation_experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, spec.model_dump_json(indent=2) + "\n")
    return target


def load_mixed_simulation_spec(
    paths: ProjectPaths, experiment_id: str
) -> MixedSimulationSpec:
    return MixedSimulationSpec.model_validate_json(
        mixed_simulation_spec_file(paths, experiment_id).read_text(encoding="utf-8")
    )


def save_mixed_simulation_result(
    paths: ProjectPaths, result: MixedSimulationResult
) -> Path:
    target = mixed_simulation_result_file(paths, result.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, result.model_dump_json(indent=2) + "\n")
    return target


def load_mixed_simulation_result(
    paths: ProjectPaths, experiment_id: str
) -> MixedSimulationResult:
    return MixedSimulationResult.model_validate_json(
        mixed_simulation_result_file(paths, experiment_id).read_text(encoding="utf-8")
    )


def save_mixed_simulation_record(
    paths: ProjectPaths, record: MixedSimulationRecord
) -> Path:
    target = mixed_simulation_record_file(paths, record.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, record.model_dump_json(indent=2) + "\n")
    return target


def load_mixed_simulation_record(
    paths: ProjectPaths, experiment_id: str
) -> MixedSimulationRecord:
    return MixedSimulationRecord.model_validate_json(
        mixed_simulation_record_file(paths, experiment_id).read_text(encoding="utf-8")
    )


def percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _replication_digest(replications: list[MixedSimulationReplication]) -> str:
    payload = [replication.model_dump(mode="json") for replication in replications]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _simulate_replication(
    spec: MixedSimulationSpec,
    forecast: ForecastArtifact,
    plan: ProductionPlanArtifact,
    residuals: list[float],
    seeds: dict[str, int],
) -> MixedSimulationReplication:
    rng = random.Random(seeds["replication_seed"])
    sampled = [rng.choice(residuals) for _ in range(spec.planning_horizon)]
    periods: list[SimulationPeriodResult] = []
    inventory_previous = 0.0
    for index, timestamp in enumerate(spec.timestamps):
        point = float(forecast.point_forecasts[index])
        residual = float(sampled[index])
        demand = (
            max(0.0, point + residual)
            if spec.demand_floor_policy == "max_zero"
            else point + residual
        )
        production = float(plan.production[index])
        available = inventory_previous + production
        served = min(available, demand)
        shortage = max(0.0, demand - available)
        ending = max(0.0, available - demand)
        periods.append(
            SimulationPeriodResult(
                timestamp=timestamp,
                point_forecast=point,
                sampled_residual=residual,
                simulated_demand=demand,
                production=production,
                available=available,
                served=served,
                shortage=shortage,
                ending_inventory=ending,
            )
        )
        inventory_previous = ending
    total_demand = sum(period.simulated_demand for period in periods)
    total_served = sum(period.served for period in periods)
    total_shortage = sum(period.shortage for period in periods)
    return MixedSimulationReplication(
        replication_id=seeds["replication_id"],
        replication_seed=seeds["replication_seed"],
        periods=periods,
        total_demand=total_demand,
        total_served=total_served,
        total_shortage=total_shortage,
        shortage_period_count=sum(
            1 for period in periods if period.shortage > 1e-9
        ),
        fill_rate=total_served / total_demand if total_demand > 0 else 1.0,
        ending_inventory=periods[-1].ending_inventory,
        average_inventory=sum(period.ending_inventory for period in periods)
        / len(periods),
        max_shortage=max(period.shortage for period in periods),
    )


def _summaries(
    spec: MixedSimulationSpec, replications: list[MixedSimulationReplication]
) -> dict[str, MixedSimulationMetricSummary]:
    summaries: dict[str, MixedSimulationMetricSummary] = {}
    for metric in spec.metrics:
        values = [float(getattr(replication, metric)) for replication in replications]
        low, high = confidence_interval(values, spec.confidence_level)
        summaries[metric] = MixedSimulationMetricSummary(
            metric=metric,
            mean=statistics.fmean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            median=statistics.median(values),
            p05=percentile(values, 5),
            p25=percentile(values, 25),
            p75=percentile(values, 75),
            p95=percentile(values, 95),
            ci_low=low,
            ci_high=high,
            confidence_level=spec.confidence_level,
            n=len(values),
        )
    return summaries


def _write_replications(
    paths: ProjectPaths, result: MixedSimulationResult
) -> Path:
    target = (
        paths.experiment_dir(result.experiment_id)
        / "artifacts"
        / "mixed_simulation_replications.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "replication_id",
            "replication_seed",
            "timestamp",
            "point_forecast",
            "sampled_residual",
            "simulated_demand",
            "production",
            "available",
            "served",
            "shortage",
            "ending_inventory",
        ]
    )
    for replication in result.replications:
        for period in replication.periods:
            writer.writerow(
                [
                    replication.replication_id,
                    replication.replication_seed,
                    period.timestamp,
                    period.point_forecast,
                    period.sampled_residual,
                    period.simulated_demand,
                    period.production,
                    period.available,
                    period.served,
                    period.shortage,
                    period.ending_inventory,
                ]
            )
    atomic_write_text(target, buffer.getvalue())
    return target


def run_mixed_simulation(
    paths: ProjectPaths,
    spec: MixedSimulationSpec,
    plan: ProductionPlanArtifact,
    forecast: ForecastArtifact,
    error: ForecastErrorArtifact,
) -> MixedSimulationResult:
    if spec.spec_hash != spec.computed_hash():
        raise MixedSimulationError("mixed simulation spec hash does not match its content")
    if spec.production_plan_artifact_hash != plan.computed_content_hash():
        raise MixedSimulationError("production plan does not match the simulation spec")
    if spec.forecast_artifact_hash != forecast.computed_content_hash():
        raise MixedSimulationError("forecast does not match the simulation spec")
    if spec.forecast_error_artifact_hash != error.computed_content_hash():
        raise MixedSimulationError("forecast-error artifact does not match the spec")
    if plan.timestamps != spec.timestamps or forecast.timestamps != spec.timestamps:
        raise MixedSimulationError("simulation timestamps do not align across artifacts")
    if error.source_forecast_experiment_id != forecast.forecast_experiment_id:
        raise MixedSimulationError("residual pool is not from the consumed forecast run")
    if error.unit != spec.unit or plan.unit != spec.unit or forecast.unit != spec.unit:
        raise MixedSimulationError("simulation unit does not match the artifacts")
    residuals = [item.residual for item in error.residuals]
    if not residuals:
        raise MixedSimulationError("forecast-error pool is empty")
    schedule = derive_seed_schedule(spec.master_seed, spec.replications)
    started = datetime.now(UTC).replace(microsecond=0)
    started_clock = time.perf_counter()
    replications = [
        _simulate_replication(spec, forecast, plan, residuals, seeds)
        for seeds in schedule
    ]
    second_run = [
        _simulate_replication(spec, forecast, plan, residuals, seeds)
        for seeds in schedule
    ]
    first_digest = _replication_digest(replications)
    second_digest = _replication_digest(second_run)
    if first_digest != second_digest:
        raise MixedSimulationError(
            "two independent runs with the same seed schedule differ"
        )
    summaries = _summaries(spec, replications)
    any_shortage = sum(
        1 for replication in replications if replication.total_shortage > 1e-9
    ) / len(replications)
    distinct_paths = len(
        {
            tuple(period.sampled_residual for period in replication.periods)
            for replication in replications
        }
    )
    floor_activations = sum(
        1
        for replication in replications
        for period in replication.periods
        if period.point_forecast + period.sampled_residual < 0
    )
    result = MixedSimulationResult(
        experiment_id=spec.simulation_experiment_id,
        simulation_id=spec.simulation_id,
        spec_hash=spec.spec_hash,
        unit=spec.unit,
        planning_horizon=spec.planning_horizon,
        timestamps=list(spec.timestamps),
        production_plan_artifact_id=spec.production_plan_artifact_id,
        production_plan_artifact_hash=spec.production_plan_artifact_hash,
        forecast_artifact_id=spec.forecast_artifact_id,
        forecast_artifact_hash=spec.forecast_artifact_hash,
        forecast_error_artifact_id=spec.forecast_error_artifact_id,
        forecast_error_artifact_hash=spec.forecast_error_artifact_hash,
        master_seed=spec.master_seed,
        replication_seed_hash=seed_schedule_hash(schedule),
        n_replications=len(replications),
        metrics=summaries,
        any_shortage_probability=any_shortage,
        distinct_residual_paths=distinct_paths,
        demand_floor_activations=floor_activations,
        replications=replications,
        determinism={
            "first_run_digest": first_digest,
            "second_run_digest": second_digest,
            "match": True,
            "runs": 2,
            "note": (
                "Two independent in-process executions with the same seed schedule; "
                "same process/platform, not a cross-platform check."
            ),
        },
        runtime_seconds=time.perf_counter() - started_clock,
        started_at=started,
        completed_at=datetime.now(UTC).replace(microsecond=0),
        notes=[
            "The production plan is frozen; the simulation never re-optimizes production.",
            "Residuals are sampled with replacement (iid) from rolling-origin "
            "out-of-sample errors.",
            "Lost-sales semantics: served = min(available, demand); "
            "shortage = max(0, demand - available).",
            "Demand floor policy is max(0, point forecast + sampled residual).",
            "Risk probabilities are empirical estimates from a finite Monte Carlo sample.",
        ],
    )
    save_mixed_simulation_result(paths, result)
    _write_replications(paths, result)
    return result


def build_mixed_simulation_record(
    paths: ProjectPaths,
    spec: MixedSimulationSpec,
    result: MixedSimulationResult,
    *,
    status: ExperimentStatus = ExperimentStatus.SUCCEEDED,
) -> MixedSimulationRecord:
    return MixedSimulationRecord(
        experiment_id=spec.simulation_experiment_id,
        project_id=paths.root.name,
        spec_hash=spec.spec_hash,
        git_commit=current_git_commit(),
        master_seed=spec.master_seed,
        replication_seed_hash=result.replication_seed_hash,
        status=status,
    )


__all__ = [
    "MIXED_SIMULATION_METRICS",
    "MixedSimulationError",
    "build_mixed_simulation_record",
    "load_mixed_simulation_record",
    "load_mixed_simulation_result",
    "load_mixed_simulation_spec",
    "mixed_simulation_record_file",
    "mixed_simulation_result_file",
    "mixed_simulation_spec_file",
    "percentile",
    "run_mixed_simulation",
    "save_mixed_simulation_record",
    "save_mixed_simulation_result",
    "save_mixed_simulation_spec",
]
