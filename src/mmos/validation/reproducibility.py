import polars as pl
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import (
    ProjectPaths,
    experiment_spec_file,
    load_experiment_spec,
    load_manifest,
)
from mmos.data.table import load_single_table
from mmos.schemas.data import Severity
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.tracking import MLflowRunRecord
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report


def validate_reproducibility(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    experiment_dir = paths.experiment_dir(experiment_id)
    artifacts = experiment_dir / "artifacts"
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    required = {
        "experiment_spec": experiment_spec_file(paths, experiment_id),
        "metrics": artifacts / "metrics.json",
        "predictions": artifacts / "predictions.parquet",
        "run_record": experiment_dir / "run_record.json",
        "mlflow_run": experiment_dir / "mlflow_run.json",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    for label in missing:
        error("missing_artifact", f"missing artifact: {label}")

    if missing:
        return build_report(
            "reproducibility",
            experiment_id,
            issues,
            evidence=[str(path) for path in required.values()],
        )

    spec = load_experiment_spec(paths, experiment_id)
    result = ExperimentResult.model_validate_json(
        (artifacts / "metrics.json").read_text(encoding="utf-8")
    )

    if spec.dataset_hash != result.dataset_hash:
        error("dataset_hash_mismatch", "spec and result disagree on the dataset hash")
    if dataset_fingerprint(load_manifest(paths)) != spec.dataset_hash:
        error("stale_dataset", "the project data changed after this experiment was planned")

    record = MLflowRunRecord.model_validate_json(
        (experiment_dir / "mlflow_run.json").read_text(encoding="utf-8")
    )
    try:
        run = MlflowClient(tracking_uri=record.tracking_uri).get_run(record.run_id)
    except MlflowException as exc:
        error("mlflow_run_unavailable", f"cannot read MLflow run: {exc}")
    else:
        if run.info.status != "FINISHED":
            error("mlflow_run_not_finished", f"MLflow run status is {run.info.status}")
        if run.data.tags.get("dataset_hash") != spec.dataset_hash:
            error("mlflow_dataset_hash_mismatch", "MLflow tags disagree with the spec")

    predictions = pl.read_parquet(artifacts / "predictions.parquet")
    if predictions.is_empty():
        error("empty_predictions", "prediction artifact has no rows")
    else:
        models = set(predictions["model"].unique().to_list())
        if models != set(spec.models):
            error(
                "prediction_models_mismatch",
                f"prediction models {sorted(models)} do not match spec {sorted(spec.models)}",
            )
        counts = predictions.group_by("row_index").len()
        if counts["len"].min() != len(spec.models) or counts["len"].max() != len(spec.models):
            error(
                "prediction_coverage_mismatch",
                "not every row has exactly one prediction per model",
            )
        if spec.temporal_split is not None:
            expected_rows = sum(
                window.test_end - window.test_start for window in result.fold_windows
            )
        else:
            expected_rows = load_single_table(paths).height
        predicted_rows = predictions["row_index"].n_unique()
        if predicted_rows != expected_rows:
            error(
                "row_count_mismatch",
                f"{predicted_rows} predicted rows for {expected_rows} data rows",
            )

    return build_report(
        "reproducibility",
        experiment_id,
        issues,
        evidence=[str(path) for path in required.values()],
    )
