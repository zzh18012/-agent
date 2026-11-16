from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.experiments.engines import forecast_baseline
from mmos.schemas.mixed import (
    ArtifactContract,
    ArtifactEnvelope,
    ForecastArtifact,
    MixedExecutionPlan,
    MixedSpec,
)
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


__all__ = [
    "ArtifactContractError",
    "build_forecast_artifact",
    "contract_for",
    "load_envelope",
    "save_forecast_artifact",
    "verify_forecast_artifact",
]
