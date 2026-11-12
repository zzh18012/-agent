import math
from pathlib import Path

import polars as pl
import pytest

from mmos.core.dataset import dataset_fingerprint
from mmos.core.intake import ingest_files
from mmos.core.project import (
    create_project,
    load_manifest,
    save_experiment_spec,
)
from mmos.experiments.runner import run_experiment
from mmos.schemas.experiment import ExperimentResult, ExperimentSpec
from mmos.schemas.project import SourceRole


def regression_frame(rows: int = 40) -> pl.DataFrame:
    x1 = [float(index % 10) for index in range(rows)]
    x2 = [float(index % 7) for index in range(rows)]
    y = [
        3.0 * a + 2.0 * b + 0.01 * math.sin(index)
        for index, (a, b) in enumerate(zip(x1, x2, strict=True))
    ]
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def classification_frame(rows: int = 60) -> pl.DataFrame:
    x1 = [float(index % 12) - 6.0 for index in range(rows)]
    x2 = [float((index * 5) % 9) - 4.0 for index in range(rows)]
    y = [1 if a + 0.5 * b > 0 else 0 for a, b in zip(x1, x2, strict=True)]
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def make_project(
    tmp_path: Path,
    df: pl.DataFrame,
    *,
    models: list[str],
    metrics: list[str],
    feature_set: list[str] | None = None,
    target_column: str = "y",
) -> tuple:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = tmp_path / "data.csv"
    df.write_csv(source)
    ingest_files(paths, [source], SourceRole.DATA)

    spec = ExperimentSpec(
        experiment_id="exp-test",
        problem_id="demo",
        proposal_id="p1",
        dataset_hash=dataset_fingerprint(load_manifest(paths)),
        target_column=target_column,
        feature_set=feature_set or ["x1", "x2"],
        split_strategy="kfold",
        validation_strategy="cross_validation",
        n_splits=3,
        models=models,
        metrics=metrics,
        random_seed=0,
        required_artifacts=["metrics.json", "predictions.parquet"],
        acceptance_criteria=["beat the baseline"],
        failure_criteria=["worse than baseline"],
        compute_requirements="test",
        expected_runtime_class="seconds",
    )
    save_experiment_spec(paths, spec)
    return paths, spec


def test_runner_writes_metrics_predictions_and_run_record(tmp_path: Path) -> None:
    paths, spec = make_project(
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
    paths, spec = make_project(
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
    paths, spec = make_project(tmp_path, df, models=["sklearn.linear_regression"], metrics=["rmse"])

    with pytest.raises(ValueError, match="null values"):
        run_experiment(paths, spec.experiment_id)


def test_runner_rejects_models_without_engine(tmp_path: Path) -> None:
    paths, spec = make_project(
        tmp_path,
        regression_frame(),
        models=["naive.last_value"],
        metrics=["rmse"],
    )

    with pytest.raises(NotImplementedError, match="no executable engine"):
        run_experiment(paths, spec.experiment_id)


def test_runner_rejects_unknown_metric(tmp_path: Path) -> None:
    paths, spec = make_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["mape"],
    )

    with pytest.raises(ValueError, match="unsupported metric"):
        run_experiment(paths, spec.experiment_id)


def test_runner_rejects_missing_feature(tmp_path: Path) -> None:
    paths, spec = make_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
        feature_set=["x1", "x9"],
    )

    with pytest.raises(ValueError, match="missing columns"):
        run_experiment(paths, spec.experiment_id)


def test_classification_runner_computes_probability_metrics(tmp_path: Path) -> None:
    paths, spec = make_project(
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
