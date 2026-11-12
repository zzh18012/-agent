from pathlib import Path

from mlflow.tracking import MlflowClient

from mmos.experiments.runner import run_experiment
from mmos.schemas.tracking import MLflowRunRecord
from tests.helpers import make_experiment_project, regression_frame


def test_runner_records_run_in_project_local_mlflow(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae"],
    )

    run_experiment(paths, spec.experiment_id)

    record_path = paths.experiments_dir / spec.experiment_id / "mlflow_run.json"
    assert record_path.exists()
    record = MLflowRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))

    assert (paths.experiments_dir / "mlflow.db").exists()
    assert record.tracking_uri.startswith("sqlite:///")
    assert record.experiment_name == paths.root.name

    client = MlflowClient(tracking_uri=record.tracking_uri)
    run = client.get_run(record.run_id)

    assert run.info.status == "FINISHED"
    assert run.data.tags["project_id"] == paths.root.name
    assert run.data.tags["dataset_hash"] == spec.dataset_hash
    assert run.data.params["random_seed"] == "0"
    assert "sklearn.linear_regression.rmse" in run.data.metrics

    artifacts = client.list_artifacts(record.run_id, "experiment/artifacts")
    assert any(artifact.path.endswith("metrics.json") for artifact in artifacts)
