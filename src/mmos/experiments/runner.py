import argparse
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold

from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_spec,
    load_manifest,
)
from mmos.data.metadata import (
    build_target_summary,
    dataset_metadata_record,
    load_dataset_metadata,
)
from mmos.data.table import load_single_table
from mmos.experiments.engines import (
    TEMPORAL_BASELINES,
    build_estimator,
    forecast_baseline,
    is_stochastic,
    task_type_for,
)
from mmos.experiments.evaluation import metrics_from_evaluation, regression_evaluation
from mmos.experiments.metrics import compute_metrics, regression_metrics
from mmos.schemas.experiment import (
    EvaluationRecord,
    ExperimentResult,
    ExperimentSpec,
    FoldWindow,
    MaseFoldDetail,
    MetricSummary,
    TargetSummary,
)
from mmos.schemas.metadata import DatasetMetadata, InformationSet
from mmos.schemas.registry import TaskType
from mmos.tracking import log_experiment_run

_REGRESSION_METRICS = {"rmse", "mae", "r2"}
_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 0


def _summarize_metric(
    metric: str, point_estimate: float, fold_values: list[float]
) -> MetricSummary:
    if not fold_values:
        return MetricSummary(
            metric=metric,
            point_estimate=point_estimate,
            n=0,
            notes="no fold-level values available",
        )
    values = np.array(fold_values, dtype=float)
    low = high = None
    notes = ""
    if len(values) >= 2:
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        samples = rng.choice(values, size=(_BOOTSTRAP_RESAMPLES, len(values)), replace=True)
        means = samples.mean(axis=1)
        low = float(np.percentile(means, 2.5))
        high = float(np.percentile(means, 97.5))
        notes = (
            f"percentile bootstrap over {len(values)} fold metrics "
            f"({_BOOTSTRAP_RESAMPLES} resamples, seed {_BOOTSTRAP_SEED})"
        )
    else:
        notes = "single fold; no interval"
    return MetricSummary(
        metric=metric,
        point_estimate=point_estimate,
        fold_values=[float(value) for value in values],
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) >= 2 else None,
        median=float(np.median(values)),
        min=float(values.min()),
        max=float(values.max()),
        confidence_interval_low=low,
        confidence_interval_high=high,
        n=len(values),
        notes=notes,
    )


def _mase_fold_detail(
    fold_id: int,
    window: FoldWindow,
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    seasonal_period: int | None,
) -> MaseFoldDetail:
    lag = seasonal_period if seasonal_period is not None else 1
    absolute_error = float(np.mean(np.abs(y_test - predictions)))
    if len(y_train) <= lag:
        return MaseFoldDetail(
            fold_id=fold_id,
            train_start=window.train_start,
            train_end=window.train_end,
            scale_lag=lag,
            scale_denominator=0.0,
            absolute_forecast_error=absolute_error,
            scaled_error=None,
            status="undefined: insufficient training history for scale_lag",
        )
    denominator = float(np.mean(np.abs(y_train[lag:] - y_train[:-lag])))
    if denominator == 0:
        return MaseFoldDetail(
            fold_id=fold_id,
            train_start=window.train_start,
            train_end=window.train_end,
            scale_lag=lag,
            scale_denominator=0.0,
            absolute_forecast_error=absolute_error,
            scaled_error=None,
            status="undefined: zero naive scaling error",
        )
    return MaseFoldDetail(
        fold_id=fold_id,
        train_start=window.train_start,
        train_end=window.train_end,
        scale_lag=lag,
        scale_denominator=denominator,
        absolute_forecast_error=absolute_error,
        scaled_error=absolute_error / denominator,
        status="defined",
    )


def _regression_fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, requested: list[str]
) -> dict[str, float]:
    base = [metric for metric in requested if metric in _REGRESSION_METRICS]
    if len(y_true) < 2:
        base = [metric for metric in base if metric != "r2"]
    return regression_metrics(y_true, y_pred, base) if base else {}


def _temporal_windows(n_rows: int, spec: ExperimentSpec) -> list[FoldWindow]:
    split = spec.temporal_split
    if split is None:
        raise ValueError("temporal experiment is missing a temporal split")
    windows: list[FoldWindow] = []
    for index in range(split.n_splits):
        train_end = split.min_train_size + index * split.step
        test_start = train_end
        test_end = test_start + split.horizon
        if test_end > n_rows:
            raise ValueError(
                "not enough rows for the requested rolling-origin folds: "
                f"need {test_end}, have {n_rows}"
            )
        windows.append(
            FoldWindow(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return windows


def _information_sets(
    spec: ExperimentSpec, metadata: DatasetMetadata | None
) -> dict[str, InformationSet]:
    availability = (
        {feature.name: feature.availability.value for feature in metadata.features}
        if metadata is not None
        else {}
    )
    horizon = spec.temporal_split.horizon if spec.temporal_split is not None else 0
    information_sets: dict[str, InformationSet] = {}
    for model_id in spec.models:
        if model_id in TEMPORAL_BASELINES:
            information_sets[model_id] = InformationSet(
                information_set_id=f"historical_y_only_h{horizon}",
                target_history_available=True,
                forecast_horizon=horizon,
                lookback_policy="expanding" if horizon else "unknown",
            )
        elif model_id == "sklearn.dummy_regressor":
            information_sets[model_id] = InformationSet(
                information_set_id="constant",
                target_history_available=False,
                forecast_horizon=horizon,
                lookback_policy="none",
            )
        else:
            feature_availability = {
                feature: availability.get(feature, "unknown") for feature in spec.feature_set
            }
            identifier = (
                "features:"
                + ",".join(sorted(spec.feature_set))
                + "|"
                + ",".join(
                    f"{name}={status}" for name, status in sorted(feature_availability.items())
                )
            )
            information_sets[model_id] = InformationSet(
                information_set_id=identifier,
                target_history_available=False,
                feature_columns=list(spec.feature_set),
                feature_availability=feature_availability,
                forecast_horizon=horizon,
                lookback_policy="none",
            )
    return information_sets


def _summaries_from_folds(
    metrics: dict[str, float], fold_scores: list[dict[str, float]]
) -> dict[str, MetricSummary]:
    return {
        metric_name: _summarize_metric(
            metric_name,
            value,
            [fold[metric_name] for fold in fold_scores if metric_name in fold],
        )
        for metric_name, value in metrics.items()
    }


def _run_temporal(
    spec: ExperimentSpec,
    df: pl.DataFrame,
    target_summary: TargetSummary,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, list[dict[str, float]]],
    dict[str, dict[str, MetricSummary]],
    list[FoldWindow],
    list[dict[str, object]],
    dict[str, bool],
    dict[str, EvaluationRecord],
]:
    if spec.time_column is None or spec.time_column not in df.columns:
        raise ValueError("time_column is missing from the data")
    df = df.sort(spec.time_column)
    time_values = df[spec.time_column]
    if time_values.null_count() > 0:
        raise ValueError("time column contains nulls")
    if time_values.n_unique() != df.height:
        raise ValueError("time column must be unique")
    missing = [column for column in spec.feature_set if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns in data: {', '.join(missing)}")
    y = df[spec.target_column].to_numpy()
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("temporal evaluation requires a numeric target")
    if df[spec.target_column].null_count() > 0:
        raise ValueError("null target values; cleaning must be an explicit step")
    x = df.select(spec.feature_set).to_numpy() if spec.feature_set else None

    split = spec.temporal_split
    if split is None:
        raise ValueError("temporal experiment is missing a temporal split")
    windows = _temporal_windows(df.height, spec)
    evaluated_indices = np.concatenate(
        [np.arange(window.test_start, window.test_end) for window in windows]
    )
    y_eval = y[evaluated_indices]

    metrics_by_model: dict[str, dict[str, float]] = {}
    fold_metrics_by_model: dict[str, list[dict[str, float]]] = {}
    summaries_by_model: dict[str, dict[str, MetricSummary]] = {}
    evaluations: dict[str, EvaluationRecord] = {}
    prediction_rows: list[dict[str, object]] = []
    determinism: dict[str, bool] = {}

    for model_id in spec.models:
        fold_scores: list[dict[str, float]] = []
        mase_details: list[MaseFoldDetail] = []
        predictions = np.empty(len(evaluated_indices), dtype=float)
        position = 0

        for fold_index, window in enumerate(windows):
            y_train = y[window.train_start : window.train_end]
            y_test = y[window.test_start : window.test_end]
            if model_id in TEMPORAL_BASELINES:
                fold_pred = forecast_baseline(
                    model_id, y_train, split.horizon, spec.seasonal_period
                )
            else:
                if x is None:
                    raise ValueError(f"model {model_id!r} requires at least one feature column")
                estimator = build_estimator(model_id, random_state=spec.random_seed)
                estimator.fit(x[window.train_start : window.train_end], y_train)
                fold_pred = estimator.predict(x[window.test_start : window.test_end])

            count = window.test_end - window.test_start
            predictions[position : position + count] = fold_pred
            position += count

            detail = _mase_fold_detail(
                fold_index, window, y_train, y_test, fold_pred, spec.seasonal_period
            )
            mase_details.append(detail)
            scores = _regression_fold_metrics(y_test, fold_pred, spec.metrics)
            if detail.scaled_error is not None:
                scores["mase"] = detail.scaled_error
            fold_scores.append(scores)

        record = regression_evaluation(model_id, y_eval, predictions, mase_details=mase_details)
        evaluations[model_id] = record
        metrics_by_model[model_id] = metrics_from_evaluation(record)
        fold_metrics_by_model[model_id] = fold_scores
        summaries_by_model[model_id] = _summaries_from_folds(
            metrics_by_model[model_id], fold_scores
        )
        determinism[model_id] = not is_stochastic(model_id)
        for row_index, prediction, actual in zip(
            evaluated_indices, predictions, y_eval, strict=True
        ):
            prediction_rows.append(
                {
                    "row_index": int(row_index),
                    "model": model_id,
                    "prediction": float(prediction),
                    "actual": float(actual),
                }
            )

    return (
        metrics_by_model,
        fold_metrics_by_model,
        summaries_by_model,
        windows,
        prediction_rows,
        determinism,
        evaluations,
    )


def _run_standard(
    spec: ExperimentSpec,
    df: pl.DataFrame,
    target_summary: TargetSummary,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, list[dict[str, float]]],
    dict[str, dict[str, MetricSummary]],
    list[dict[str, object]],
    dict[str, bool],
    dict[str, EvaluationRecord],
]:
    required = [*spec.feature_set, spec.target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns in data: {', '.join(missing)}")
    if not spec.feature_set:
        raise ValueError("non-temporal experiments require at least one feature")

    features = df.select(spec.feature_set)
    if features.null_count().to_numpy().sum() > 0 or df[spec.target_column].null_count() > 0:
        raise ValueError(
            "null values in features or target; cleaning must be an explicit, recorded step"
        )
    x = features.to_numpy()
    y = df[spec.target_column].to_numpy()
    task_type = task_type_for(spec.models[0])
    for model_id in spec.models[1:]:
        if model_id in TEMPORAL_BASELINES:
            continue
        if task_type_for(model_id) is not task_type:
            raise ValueError(f"experiment mixes task types: {spec.models[0]!r} and {model_id!r}")

    if task_type is TaskType.CLASSIFICATION:
        splitter = StratifiedKFold(
            n_splits=spec.n_splits, shuffle=True, random_state=spec.random_seed
        )
        split_iter = splitter.split(x, y)
        classes = np.unique(y)
        class_positions = {value: index for index, value in enumerate(classes)}
    elif task_type is TaskType.REGRESSION:
        splitter = KFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.random_seed)
        split_iter = splitter.split(x, y)
        classes = None
        class_positions = {}
    else:
        raise NotImplementedError(
            f"cross-validated execution is not implemented for {task_type.value}"
        )

    folds = list(split_iter)
    metrics_by_model: dict[str, dict[str, float]] = {}
    fold_metrics_by_model: dict[str, list[dict[str, float]]] = {}
    summaries_by_model: dict[str, dict[str, MetricSummary]] = {}
    evaluations: dict[str, EvaluationRecord] = {}
    prediction_rows: list[dict[str, object]] = []
    determinism: dict[str, bool] = {}
    fold_metric_names = [name for name in spec.metrics if name != "roc_auc"]

    for model_id in spec.models:
        oof_pred = np.full(len(y), np.nan)
        oof_proba = np.full((len(y), len(classes)), np.nan) if classes is not None else None
        fold_scores: list[dict[str, float]] = []

        for train_index, test_index in folds:
            estimator = build_estimator(model_id, random_state=spec.random_seed)
            estimator.fit(x[train_index], y[train_index])
            fold_pred = estimator.predict(x[test_index])
            oof_pred[test_index] = fold_pred
            fold_scores.append(
                compute_metrics(task_type, y[test_index], fold_pred, None, fold_metric_names)
            )
            if oof_proba is not None:
                fold_proba = estimator.predict_proba(x[test_index])
                for column_index, class_value in enumerate(estimator.classes_):
                    oof_proba[test_index, class_positions[class_value]] = fold_proba[
                        :, column_index
                    ]

        if np.isnan(oof_pred).any():
            raise RuntimeError(f"incomplete out-of-fold predictions for {model_id}")
        if oof_proba is not None and np.isnan(oof_proba).any():
            raise RuntimeError(f"incomplete out-of-fold probabilities for {model_id}")

        if task_type is TaskType.REGRESSION:
            record = regression_evaluation(model_id, y, oof_pred)
            evaluations[model_id] = record
            metrics_by_model[model_id] = metrics_from_evaluation(record)
            summaries_by_model[model_id] = _summaries_from_folds(
                metrics_by_model[model_id], fold_scores
            )
        else:
            metrics_by_model[model_id] = compute_metrics(
                task_type, y, oof_pred, oof_proba, spec.metrics
            )
            summaries_by_model[model_id] = _summaries_from_folds(
                metrics_by_model[model_id], fold_scores
            )
        fold_metrics_by_model[model_id] = fold_scores
        determinism[model_id] = not is_stochastic(model_id)
        prediction_rows.extend(
            {
                "row_index": index,
                "model": model_id,
                "prediction": oof_pred[index].item(),
                "actual": y[index].item(),
            }
            for index in range(len(y))
        )

    return (
        metrics_by_model,
        fold_metrics_by_model,
        summaries_by_model,
        prediction_rows,
        determinism,
        evaluations,
    )


def run_experiment(paths: ProjectPaths, experiment_id: str) -> ExperimentResult:
    spec = load_experiment_spec(paths, experiment_id)
    manifest = load_manifest(paths)
    fingerprint = dataset_fingerprint(manifest)
    if fingerprint != spec.dataset_hash:
        raise ValueError(
            "dataset hash mismatch: the project data changed after this experiment was planned "
            f"(spec {spec.dataset_hash[:12]}, current {fingerprint[:12]})"
        )

    df = load_single_table(paths)
    metadata = load_dataset_metadata(paths)
    metadata_record = dataset_metadata_record(paths)
    if metadata is not None and metadata.target_name != spec.target_column:
        raise ValueError(
            f"dataset metadata declares target {metadata.target_name!r}, "
            f"but the experiment target is {spec.target_column!r}"
        )
    target_summary = build_target_summary(df, spec.target_column, metadata)
    feature_availability = (
        {feature.name: feature.availability.value for feature in metadata.features}
        if metadata is not None
        else {}
    )

    if spec.temporal_split is not None:
        (
            metrics_by_model,
            fold_metrics_by_model,
            metric_summaries,
            fold_windows,
            prediction_rows,
            determinism,
            evaluations,
        ) = _run_temporal(spec, df, target_summary)
        task_type = TaskType.FORECASTING
        evaluated_rows = sum(window.test_end - window.test_start for window in fold_windows)
    else:
        (
            metrics_by_model,
            fold_metrics_by_model,
            metric_summaries,
            prediction_rows,
            determinism,
            evaluations,
        ) = _run_standard(spec, df, target_summary)
        fold_windows = []
        task_type = task_type_for(spec.models[0])
        evaluated_rows = len(df)

    result = ExperimentResult(
        experiment_id=spec.experiment_id,
        dataset_hash=spec.dataset_hash,
        task_type=task_type,
        feature_set=spec.feature_set,
        split_strategy=spec.split_strategy,
        validation_strategy=spec.validation_strategy,
        random_seed=spec.random_seed,
        n_splits=spec.n_splits,
        models=metrics_by_model,
        fold_metrics=fold_metrics_by_model,
        metric_summaries=metric_summaries,
        fold_windows=fold_windows,
        target_summary=target_summary,
        model_determinism=determinism,
        feature_availability=feature_availability,
        dataset_metadata_sha256=metadata_record.sha256 if metadata_record else None,
        evaluated_rows=evaluated_rows,
        evaluations=evaluations,
        information_sets=_information_sets(spec, metadata),
        generated_at=datetime.now(UTC).replace(microsecond=0),
    )

    experiment_dir = paths.experiments_dir / experiment_id
    artifacts = experiment_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_write_text(artifacts / "metrics.json", result.model_dump_json(indent=2) + "\n")
    pl.DataFrame(prediction_rows).write_parquet(artifacts / "predictions.parquet")
    run_record = {
        "experiment_id": experiment_id,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "polars_version": pl.__version__,
        "platform": platform.platform(),
        "created_at": result.generated_at.isoformat(),
    }
    atomic_write_text(experiment_dir / "run_record.json", json.dumps(run_record, indent=2) + "\n")
    log_experiment_run(paths, spec, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one planned experiment.")
    parser.add_argument("--project", required=True, help="Project directory.")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)

    paths = ProjectPaths(Path(args.project).resolve())
    result = run_experiment(paths, args.experiment_id)
    print(json.dumps(result.models, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
