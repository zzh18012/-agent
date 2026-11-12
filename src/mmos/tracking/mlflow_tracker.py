from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.experiment import ExperimentResult, ExperimentSpec
from mmos.schemas.tracking import MLflowRunRecord


def tracking_uri(paths: ProjectPaths) -> str:
    """Each project owns its ledger; no global or remote store is assumed."""
    database = paths.experiments_dir / "mlflow.db"
    return f"sqlite:///{database.as_posix()}"


def _experiment_id(client: MlflowClient, paths: ProjectPaths) -> str:
    name = paths.root.name
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    artifact_root: Path = paths.experiments_dir / "mlflow_artifacts"
    return client.create_experiment(name, artifact_location=artifact_root.as_uri())


def log_experiment_run(
    paths: ProjectPaths, spec: ExperimentSpec, result: ExperimentResult
) -> MLflowRunRecord:
    """Record one formal run. Failures propagate; tracking is not optional."""
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)

    with mlflow.start_run(experiment_id=experiment_id, run_name=spec.experiment_id) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "problem_id": spec.problem_id,
                "proposal_id": spec.proposal_id,
                "dataset_hash": spec.dataset_hash,
            }
        )
        mlflow.log_params(
            {
                "dataset_hash": spec.dataset_hash,
                "random_seed": spec.random_seed,
                "n_splits": spec.n_splits,
                "models": ",".join(spec.models),
                "metrics": ",".join(spec.metrics),
                "target_column": spec.target_column,
                "feature_count": len(spec.feature_set),
            }
        )
        for model_id, metrics in result.models.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_id}.{metric_name}", value)

        experiment_dir = paths.experiments_dir / spec.experiment_id
        mlflow.log_artifacts(str(experiment_dir), artifact_path="experiment")
        run_id = run.info.run_id

    record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        paths.experiments_dir / spec.experiment_id / "mlflow_run.json",
        record.model_dump_json(indent=2) + "\n",
    )
    return record
