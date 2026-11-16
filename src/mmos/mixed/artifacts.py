import statistics
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.experiments.engines import forecast_baseline
from mmos.schemas.experiment import FoldWindow
from mmos.schemas.mixed import (
    ArtifactContract,
    ArtifactEnvelope,
    ForecastArtifact,
    ForecastErrorArtifact,
    ForecastErrorContract,
    ForecastResidual,
    MixedExecutionPlan,
    MixedSpec,
    ProductionPlanArtifact,
    ProductionPlanContract,
)
from mmos.schemas.optimization import OptimizationResult
from mmos.schemas.states import TaskFamily


class ArtifactContractError(RuntimeError):
    pass


def build_forecast_artifact(
    spec: MixedSpec,
    history: np.ndarray,
    *,
    forecast_experiment_id: str,
    source_dataset_hash: str,
) -> ForecastArtifact:
    if spec.forecast_model_id not in {
        "naive.last_value",
        "naive.drift",
        "naive.seasonal",
    }:
        raise ArtifactContractError("mixed forecasting supports temporal baselines only")
    points = forecast_baseline(
        spec.forecast_model_id,
        history,
        spec.forecast_horizon,
        spec.seasonal_period,
    )
    return ForecastArtifact(
        forecast_horizon=spec.forecast_horizon,
        timestamps=list(spec.forecast_timestamps),
        target_name=spec.target_column,
        unit=spec.unit,
        point_forecasts=[float(value) for value in points],
        model_id=spec.forecast_model_id,
        source_dataset_hash=source_dataset_hash,
        forecast_experiment_id=forecast_experiment_id,
        validation_strategy="rolling_origin",
        validation_status="verified",
    )


def save_forecast_artifact(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    artifact: ForecastArtifact,
    *,
    provenance: dict[str, str],
) -> tuple[Path, Path, ArtifactEnvelope]:
    artifact_dir = paths.mixed_dir / "runs" / run_id / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload_path = artifact_dir / "forecast_artifact.json"
    atomic_write_text(payload_path, artifact.model_dump_json(indent=2) + "\n")
    payload_hash = sha256_file(payload_path)
    envelope = ArtifactEnvelope(
        artifact_id=f"artifact-{run_id}-demand-forecast",
        artifact_type="demand_forecast",
        schema_version="1.0",
        producer_project_id=paths.root.name,
        producer_node_id="forecast_node",
        producer_task_family=TaskFamily.FORECASTING,
        producer_experiment_id=artifact.forecast_experiment_id,
        content_hash=artifact.computed_content_hash(),
        validation_status="verified",
        created_at=datetime.now(UTC).replace(microsecond=0),
        payload_path=payload_path.relative_to(paths.root).as_posix(),
        payload_hash=payload_hash,
        provenance=provenance,
    )
    envelope_path = artifact_dir / "forecast_envelope.json"
    atomic_write_text(envelope_path, envelope.model_dump_json(indent=2) + "\n")
    return payload_path, envelope_path, envelope


def load_envelope(path: Path) -> ArtifactEnvelope:
    return ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))


def contract_for(spec: MixedSpec) -> ArtifactContract:
    return ArtifactContract(
        artifact_type="demand_forecast",
        schema_version="1.0",
        target_name=spec.target_column,
        unit=spec.unit,
        horizon=spec.forecast_horizon,
        timestamps=list(spec.planning_periods),
    )


def verify_forecast_artifact(
    paths: ProjectPaths,
    envelope: ArtifactEnvelope,
    contract: ArtifactContract,
) -> ForecastArtifact:
    if envelope.artifact_type != contract.artifact_type:
        raise ArtifactContractError(
            f"artifact type {envelope.artifact_type!r} != {contract.artifact_type!r}"
        )
    if envelope.schema_version != contract.schema_version:
        raise ArtifactContractError(
            f"schema version {envelope.schema_version!r} != {contract.schema_version!r}"
        )
    if envelope.validation_status != "verified":
        raise ArtifactContractError("forecast artifact is not verified")
    payload_path = paths.root / envelope.payload_path
    if not payload_path.exists():
        raise ArtifactContractError(f"forecast payload missing: {envelope.payload_path}")
    if sha256_file(payload_path) != envelope.payload_hash:
        raise ArtifactContractError("forecast payload hash mismatch")
    artifact = ForecastArtifact.model_validate_json(
        payload_path.read_text(encoding="utf-8")
    )
    if artifact.computed_content_hash() != envelope.content_hash:
        raise ArtifactContractError("forecast content hash mismatch")
    if artifact.validation_status != "verified":
        raise ArtifactContractError("forecast payload is not verified")
    if artifact.forecast_experiment_id != envelope.producer_experiment_id:
        raise ArtifactContractError("forecast experiment identity mismatch")
    if artifact.forecast_horizon != contract.horizon:
        raise ArtifactContractError("forecast horizon mismatch")
    if artifact.timestamps != contract.timestamps:
        raise ArtifactContractError("forecast timestamp alignment mismatch")
    if artifact.target_name != contract.target_name:
        raise ArtifactContractError("forecast target name mismatch")
    if artifact.unit != contract.unit:
        raise ArtifactContractError("forecast unit mismatch")
    return artifact


def build_forecast_error_artifact(
    spec: MixedSpec,
    predictions: pl.DataFrame,
    timestamps: list[str],
    fold_windows: list[FoldWindow],
    *,
    forecast_experiment_id: str,
    forecast_artifact_id: str,
    forecast_artifact_hash: str,
) -> ForecastErrorArtifact:
    frame = predictions.filter(pl.col("model") == spec.forecast_model_id).sort(
        "row_index"
    )
    if frame.height == 0:
        raise ArtifactContractError("no out-of-fold predictions for the forecast model")
    residuals: list[ForecastResidual] = []
    for row in frame.iter_rows(named=True):
        row_index = int(row["row_index"])
        fold_id = next(
            (
                index
                for index, window in enumerate(fold_windows)
                if window.test_start <= row_index < window.test_end
            ),
            None,
        )
        if fold_id is None:
            raise ArtifactContractError(
                f"prediction row {row_index} is not in any rolling-origin test window"
            )
        if row_index >= len(timestamps):
            raise ArtifactContractError("prediction row is outside the time series")
        actual = float(row["actual"])
        forecast = float(row["prediction"])
        residuals.append(
            ForecastResidual(
                fold_id=fold_id,
                row_index=row_index,
                timestamp=str(timestamps[row_index]),
                actual=actual,
                forecast=forecast,
                residual=actual - forecast,
            )
        )
    values = [item.residual for item in residuals]
    return ForecastErrorArtifact(
        forecast_horizon=spec.forecast_horizon,
        unit=spec.unit,
        source_forecast_experiment_id=forecast_experiment_id,
        source_forecast_artifact_id=forecast_artifact_id,
        source_forecast_artifact_hash=forecast_artifact_hash,
        residuals=residuals,
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        sample_count=len(values),
    )


def save_forecast_error_artifact(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    artifact: ForecastErrorArtifact,
    *,
    provenance: dict[str, str],
) -> tuple[Path, Path, ArtifactEnvelope]:
    artifact_dir = paths.mixed_dir / "runs" / run_id / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload_path = artifact_dir / "forecast_error_artifact.json"
    atomic_write_text(payload_path, artifact.model_dump_json(indent=2) + "\n")
    envelope = ArtifactEnvelope(
        artifact_id=f"artifact-{run_id}-forecast-error",
        artifact_type="forecast_error",
        schema_version="1.0",
        producer_project_id=paths.root.name,
        producer_node_id="forecast_node",
        producer_task_family=TaskFamily.FORECASTING,
        producer_experiment_id=artifact.source_forecast_experiment_id,
        content_hash=artifact.computed_content_hash(),
        validation_status="verified",
        created_at=datetime.now(UTC).replace(microsecond=0),
        payload_path=payload_path.relative_to(paths.root).as_posix(),
        payload_hash=sha256_file(payload_path),
        provenance=provenance,
    )
    envelope_path = artifact_dir / "forecast_error_envelope.json"
    atomic_write_text(envelope_path, envelope.model_dump_json(indent=2) + "\n")
    return payload_path, envelope_path, envelope


def contract_for_forecast_error(
    spec: MixedSpec,
    forecast_experiment_id: str,
    forecast_artifact_hash: str,
    *,
    min_sample_count: int = 1,
) -> ForecastErrorContract:
    return ForecastErrorContract(
        unit=spec.unit,
        source_forecast_experiment_id=forecast_experiment_id,
        source_forecast_artifact_hash=forecast_artifact_hash,
        min_sample_count=min_sample_count,
    )


def verify_forecast_error_artifact(
    paths: ProjectPaths, envelope: ArtifactEnvelope, contract: ForecastErrorContract
) -> ForecastErrorArtifact:
    if envelope.artifact_type != contract.artifact_type:
        raise ArtifactContractError(
            f"artifact type {envelope.artifact_type!r} != {contract.artifact_type!r}"
        )
    if envelope.schema_version != contract.schema_version:
        raise ArtifactContractError(
            f"schema version {envelope.schema_version!r} != {contract.schema_version!r}"
        )
    if envelope.validation_status != "verified":
        raise ArtifactContractError("forecast-error artifact is not verified")
    payload_path = paths.root / envelope.payload_path
    if not payload_path.exists():
        raise ArtifactContractError(f"forecast-error payload missing: {envelope.payload_path}")
    if sha256_file(payload_path) != envelope.payload_hash:
        raise ArtifactContractError("forecast-error payload hash mismatch")
    artifact = ForecastErrorArtifact.model_validate_json(
        payload_path.read_text(encoding="utf-8")
    )
    if artifact.computed_content_hash() != envelope.content_hash:
        raise ArtifactContractError("forecast-error content hash mismatch")
    if artifact.validation_status != "verified":
        raise ArtifactContractError("forecast-error payload is not verified")
    if artifact.source_forecast_experiment_id != contract.source_forecast_experiment_id:
        raise ArtifactContractError("forecast-error source experiment mismatch")
    if artifact.source_forecast_artifact_hash != contract.source_forecast_artifact_hash:
        raise ArtifactContractError("forecast-error source artifact hash mismatch")
    if artifact.residual_definition != contract.residual_definition:
        raise ArtifactContractError("residual definition mismatch")
    if artifact.residual_source != contract.residual_source:
        raise ArtifactContractError("residual source must be rolling-origin OOF")
    if artifact.unit != contract.unit:
        raise ArtifactContractError("forecast-error unit mismatch")
    if artifact.sample_count < contract.min_sample_count:
        raise ArtifactContractError("forecast-error sample count is too small")
    return artifact


def build_production_plan_artifact(
    spec: MixedSpec,
    result: OptimizationResult,
    *,
    optimization_experiment_id: str,
    forecast_artifact_id: str,
    forecast_artifact_hash: str,
) -> ProductionPlanArtifact:
    solution = next(
        (item for item in result.solutions if item.method == "ortools.cbc"), None
    )
    if (
        solution is None
        or not solution.feasible
        or solution.objective_value is None
        or solution.status not in {"optimal", "feasible"}
    ):
        raise ArtifactContractError("production plan requires a feasible solver solution")
    production = [
        float(solution.variable_values.get(f"production_{index}", 0.0))
        for index in range(spec.forecast_horizon)
    ]
    inventory = [
        float(solution.variable_values.get(f"inventory_{index}", 0.0))
        for index in range(spec.forecast_horizon)
    ]
    return ProductionPlanArtifact(
        planning_horizon=spec.forecast_horizon,
        timestamps=list(spec.planning_periods),
        unit=spec.unit,
        production=production,
        planned_inventory=inventory,
        capacity=[float(value) for value in spec.capacities],
        objective_value=float(solution.objective_value),
        forecast_artifact_id=forecast_artifact_id,
        forecast_artifact_hash=forecast_artifact_hash,
        optimization_experiment_id=optimization_experiment_id,
    )


def save_production_plan_artifact(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    artifact: ProductionPlanArtifact,
    *,
    provenance: dict[str, str],
) -> tuple[Path, Path, ArtifactEnvelope]:
    artifact_dir = paths.mixed_dir / "runs" / run_id / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload_path = artifact_dir / "production_plan_artifact.json"
    atomic_write_text(payload_path, artifact.model_dump_json(indent=2) + "\n")
    envelope = ArtifactEnvelope(
        artifact_id=f"artifact-{run_id}-production-plan",
        artifact_type="production_plan",
        schema_version="1.0",
        producer_project_id=paths.root.name,
        producer_node_id="optimization_node",
        producer_task_family=TaskFamily.OPTIMIZATION,
        producer_experiment_id=artifact.optimization_experiment_id,
        content_hash=artifact.computed_content_hash(),
        validation_status="verified",
        created_at=datetime.now(UTC).replace(microsecond=0),
        payload_path=payload_path.relative_to(paths.root).as_posix(),
        payload_hash=sha256_file(payload_path),
        provenance=provenance,
    )
    envelope_path = artifact_dir / "production_plan_envelope.json"
    atomic_write_text(envelope_path, envelope.model_dump_json(indent=2) + "\n")
    return payload_path, envelope_path, envelope


def contract_for_production_plan(
    spec: MixedSpec,
    forecast_artifact_id: str,
    forecast_artifact_hash: str,
) -> ProductionPlanContract:
    return ProductionPlanContract(
        unit=spec.unit,
        horizon=spec.forecast_horizon,
        timestamps=list(spec.planning_periods),
        forecast_artifact_id=forecast_artifact_id,
        forecast_artifact_hash=forecast_artifact_hash,
    )


def verify_production_plan_artifact(
    paths: ProjectPaths, envelope: ArtifactEnvelope, contract: ProductionPlanContract
) -> ProductionPlanArtifact:
    if envelope.artifact_type != contract.artifact_type:
        raise ArtifactContractError(
            f"artifact type {envelope.artifact_type!r} != {contract.artifact_type!r}"
        )
    if envelope.schema_version != contract.schema_version:
        raise ArtifactContractError(
            f"schema version {envelope.schema_version!r} != {contract.schema_version!r}"
        )
    if envelope.validation_status != "verified":
        raise ArtifactContractError("production plan artifact is not verified")
    payload_path = paths.root / envelope.payload_path
    if not payload_path.exists():
        raise ArtifactContractError(f"production plan payload missing: {envelope.payload_path}")
    if sha256_file(payload_path) != envelope.payload_hash:
        raise ArtifactContractError("production plan payload hash mismatch")
    artifact = ProductionPlanArtifact.model_validate_json(
        payload_path.read_text(encoding="utf-8")
    )
    if artifact.computed_content_hash() != envelope.content_hash:
        raise ArtifactContractError("production plan content hash mismatch")
    if artifact.validation_status != "verified":
        raise ArtifactContractError("production plan payload is not verified")
    if artifact.forecast_artifact_id != contract.forecast_artifact_id:
        raise ArtifactContractError("production plan forecast artifact id mismatch")
    if artifact.forecast_artifact_hash != contract.forecast_artifact_hash:
        raise ArtifactContractError("production plan forecast artifact hash mismatch")
    if artifact.planning_horizon != contract.horizon:
        raise ArtifactContractError("production plan horizon mismatch")
    if artifact.timestamps != contract.timestamps:
        raise ArtifactContractError("production plan timestamp alignment mismatch")
    if artifact.unit != contract.unit:
        raise ArtifactContractError("production plan unit mismatch")
    return artifact


__all__ = [
    "ArtifactContractError",
    "build_forecast_artifact",
    "build_forecast_error_artifact",
    "build_production_plan_artifact",
    "contract_for",
    "contract_for_forecast_error",
    "contract_for_production_plan",
    "load_envelope",
    "save_forecast_artifact",
    "save_forecast_error_artifact",
    "save_production_plan_artifact",
    "verify_forecast_artifact",
    "verify_forecast_error_artifact",
    "verify_production_plan_artifact",
]
