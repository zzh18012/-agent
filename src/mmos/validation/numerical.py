import math

import numpy as np
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

    for model_id, evaluation in result.evaluations.items():
        frame = predictions.filter(pl.col("model") == model_id)
        if frame.is_empty():
            continue
        actual = frame["actual"].to_numpy().astype(float)
        predicted = frame["prediction"].to_numpy().astype(float)
        errors = predicted - actual
        sse = float(np.sum(errors**2))
        mean = float(np.mean(actual))
        sst = float(np.sum((actual - mean) ** 2))
        rmse = math.sqrt(sse / len(actual))
        mae = float(np.mean(np.abs(errors)))
        std_population = math.sqrt(sst / len(actual))
        ratio = rmse / std_population if std_population > 0 else None
        r2 = 1.0 - sse / sst if sst > 0 else None

        def close(reported: float, recomputed: float) -> bool:
            return abs(reported - recomputed) <= 1e-9 + 1e-9 * abs(recomputed)

        checks = (
            ("n_observations", evaluation.n_observations, len(actual)),
            ("sse", evaluation.sse, sse),
            ("sst", evaluation.sst, sst),
            ("target_mean", evaluation.target_mean, mean),
            ("target_std_population", evaluation.target_std_population, std_population),
            ("rmse", evaluation.rmse, rmse),
            ("mae", evaluation.mae, mae),
            ("r2", evaluation.r2, r2),
            ("rmse_eval_std_ratio", evaluation.rmse_eval_std_ratio, ratio),
        )
        for name, reported, recomputed in checks:
            if reported is None or recomputed is None:
                continue
            if not close(float(reported), float(recomputed)):
                error(
                    "evaluation_component_mismatch",
                    f"{model_id}/{name}: reported {reported}, recomputed {recomputed}",
                )
        if sst == 0 and evaluation.r2 is not None:
            error("r2_should_be_undefined", f"{model_id}: R2 reported for a constant target")
        if sst > 0 and evaluation.r2 is None:
            error("r2_missing", f"{model_id}: R2 missing although SST > 0")

        scaled_values = []
        for detail in evaluation.mase_details:
            if detail.scale_denominator > 0:
                expected_scaled = detail.absolute_forecast_error / detail.scale_denominator
                if detail.scaled_error is None or not close(detail.scaled_error, expected_scaled):
                    error(
                        "mase_scaled_error_mismatch",
                        f"{model_id}/fold {detail.fold_id}: scaled error does not match "
                        "absolute error / training denominator",
                    )
                else:
                    scaled_values.append(detail.scaled_error)
            elif detail.scaled_error is not None:
                error(
                    "mase_undefined_but_reported",
                    f"{model_id}/fold {detail.fold_id}: zero denominator but scaled error",
                )
        if scaled_values and evaluation.mase is not None:
            expected_mase = float(np.mean(scaled_values))
            if not close(evaluation.mase, expected_mase):
                error(
                    "mase_mismatch",
                    f"{model_id}: reported MASE {evaluation.mase}, recomputed {expected_mase}",
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
        for model_id, metrics in result.models.items():
            ratio = metrics.get("rmse_eval_std_ratio")
            rmse = metrics.get("rmse")
            evaluation = result.evaluations.get(model_id)
            if (
                ratio is not None
                and rmse is not None
                and evaluation is not None
                and evaluation.target_std_population
            ):
                expected_ratio = rmse / evaluation.target_std_population
                if abs(ratio - expected_ratio) > 1e-9 + 1e-9 * abs(expected_ratio):
                    error(
                        "normalized_metric_mismatch",
                        f"{model_id}/rmse_eval_std_ratio: reported {ratio}, "
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
