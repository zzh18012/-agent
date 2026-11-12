from pathlib import Path

import polars as pl
import pytest

from mmos.core.project import save_experiment_spec
from mmos.experiments.runner import run_experiment
from mmos.schemas.experiment import ExperimentResult
from tests.helpers import (
    classification_frame,
    make_experiment_project,
    regression_frame,
)


def test_runner_writes_metrics_predictions_and_run_record(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae", "r2"],
    )

    result = run_experiment(paths, spec.experiment_id)

    assert set(result.models) == {"sklearn.dummy_regressor", "sklearn.linear_regression"}
    assert (
        result.models["sklearn.linear_regression"]["rmse"]
        < result.models["sklearn.dummy_regressor"]["rmse"]
    )

    artifacts = paths.experiments_dir / spec.experiment_id / "artifacts"
    saved = ExperimentResult.model_validate_json(
        (artifacts / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved == result

    predictions = pl.read_parquet(artifacts / "predictions.parquet")
    assert predictions.height == 40 * 2
    assert set(predictions["model"].unique().to_list()) == set(spec.models)

    run_record = artifacts.parent / "run_record.json"
    assert run_record.exists()
    assert "sklearn_version" in run_record.read_text(encoding="utf-8")


def test_runner_rejects_stale_dataset_hash(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
    )
    stale = spec.model_copy(update={"dataset_hash": "b" * 64})
    save_experiment_spec(paths, stale)

    with pytest.raises(ValueError, match="dataset hash mismatch"):
        run_experiment(paths, stale.experiment_id)


def test_runner_rejects_null_features(tmp_path: Path) -> None:
    df = regression_frame()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("x1")).alias("x1")
    )
    paths, spec = make_experiment_project(
        tmp_path, df, models=["sklearn.linear_regression"], metrics=["rmse"]
    )

    with pytest.raises(ValueError, match="null values"):
        run_experiment(paths, spec.experiment_id)


def test_runner_rejects_models_without_engine(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["naive.last_value"],
        metrics=["rmse"],
    )

    with pytest.raises(NotImplementedError, match="no executable engine"):
        run_experiment(paths, spec.experiment_id)


def test_runner_rejects_unknown_metric(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["mape"],
    )

    with pytest.raises(ValueError, match="unsupported metric"):
        run_experiment(paths, spec.experiment_id)


def test_runner_rejects_missing_feature(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
        feature_set=["x1", "x9"],
    )

    with pytest.raises(ValueError, match="missing columns"):
        run_experiment(paths, spec.experiment_id)


def test_classification_runner_computes_probability_metrics(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        classification_frame(),
        models=["sklearn.logistic_regression"],
        metrics=["accuracy", "f1_macro", "roc_auc"],
    )

    result = run_experiment(paths, spec.experiment_id)

    metrics = result.models["sklearn.logistic_regression"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
