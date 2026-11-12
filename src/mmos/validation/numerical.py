import math

import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mmos.core.project import ProjectPaths
from mmos.data.table import load_single_table
from mmos.schemas.data import Severity
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.registry import TaskType
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-9


def _load_result(paths: ProjectPaths, experiment_id: str) -> ExperimentResult:
    target = paths.experiments_dir / experiment_id / "artifacts" / "metrics.json"
    return ExperimentResult.model_validate_json(target.read_text(encoding="utf-8"))


def validate_numerical(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    """Recompute reported metrics from out-of-fold predictions and check ranges."""
    result = _load_result(paths, experiment_id)
    artifacts = paths.experiments_dir / experiment_id / "artifacts"
    predictions_path = artifacts / "predictions.parquet"
    predictions = pl.read_parquet(predictions_path)
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    for model_id, metrics in result.models.items():
        for name, value in metrics.items():
            if name in {"rmse", "mae"} and value < 0:
                error("negative_error_metric", f"{model_id}/{name} is negative: {value}")
            if name == "r2" and value > 1.0 + 1e-9:
                error("invalid_r2", f"{model_id}/r2 exceeds 1: {value}")
            if name in {"accuracy", "f1_macro", "roc_auc"} and not 0.0 <= value <= 1.0:
                error("probability_metric_out_of_range", f"{model_id}/{name} is {value}")

        frame = predictions.filter(pl.col("model") == model_id)
        if frame.is_empty():
            error("missing_predictions", f"no out-of-fold predictions for {model_id}")
            continue
        actual = frame["actual"].to_numpy()
        predicted = frame["prediction"].to_numpy()

        if result.task_type in (TaskType.REGRESSION, TaskType.FORECASTING):
            recomputed = {
                "rmse": math.sqrt(mean_squared_error(actual, predicted)),
                "mae": float(mean_absolute_error(actual, predicted)),
                "r2": float(r2_score(actual, predicted)),
            }
        elif result.task_type is TaskType.CLASSIFICATION:
            recomputed = {
                "accuracy": float(accuracy_score(actual, predicted)),
                "f1_macro": float(
                    f1_score(actual, predicted, average="macro", zero_division=0)  # type: ignore[arg-type]
                ),
            }
        else:
            continue

        for name in set(metrics) & set(recomputed):
            reported = metrics[name]
            value = recomputed[name]
            if abs(reported - value) > _TOLERANCE:
                error(
                    "metric_mismatch",
                    f"{model_id}/{name}: reported {reported}, recomputed {value}",
                )

    target = result.target_summary
    if target is not None:
        series = load_single_table(paths)[target.name]

        def numeric(value: object) -> float | None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return None

        recomputed_target = {
            "mean": numeric(series.mean()) if series.dtype.is_numeric() else None,
            "std": numeric(series.std()) if series.dtype.is_numeric() else None,
            "median": numeric(series.median()) if series.dtype.is_numeric() else None,
            "min": numeric(series.min()) if series.dtype.is_numeric() else None,
            "max": numeric(series.max()) if series.dtype.is_numeric() else None,
        }
        for name, recomputed in recomputed_target.items():
            reported = getattr(target, name)
            if (
                recomputed is None
                or not isinstance(reported, (int, float))
                or isinstance(reported, bool)
            ):
                continue
            if abs(float(reported) - recomputed) > 1e-9 + 1e-9 * abs(recomputed):
                error(
                    "target_stat_mismatch",
                    f"target {name}: reported {reported}, recomputed from data {recomputed}",
                )
        if target.std:
            for model_id, metrics in result.models.items():
                ratio = metrics.get("rmse_std_ratio")
                rmse = metrics.get("rmse")
                if ratio is not None and rmse is not None:
                    expected_ratio = rmse / target.std
                    if abs(ratio - expected_ratio) > 1e-9 + 1e-9 * abs(expected_ratio):
                        error(
                            "normalized_metric_mismatch",
                            f"{model_id}/rmse_std_ratio: reported {ratio}, "
                            f"recomputed {expected_ratio}",
                        )

    return build_report(
        "numerical",
        experiment_id,
        issues,
        evidence=[
            str(artifacts / "metrics.json"),
            str(predictions_path),
        ],
        recommendations=[
            "roc_auc and MASE cannot be recomputed from labels alone and are range-checked",
        ]
        if any("roc_auc" in metrics for metrics in result.models.values())
        else [],
    )
