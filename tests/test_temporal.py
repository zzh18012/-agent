import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from mmos.core.dataset import dataset_fingerprint
from mmos.core.intake import ingest_files
from mmos.core.project import create_project, load_manifest, save_experiment_spec
from mmos.evidence import freeze_experiment, load_evidence_index
from mmos.experiments.engines import forecast_baseline
from mmos.experiments.runner import run_experiment
from mmos.paper import build_paper, render_markdown
from mmos.schemas.experiment import ExperimentSpec, FoldWindow, TemporalSplit
from mmos.schemas.metadata import (
    DatasetMetadata,
    DatasetProvenance,
    FeatureAvailability,
    FeatureMetadata,
    TemporalMetadata,
)
from mmos.schemas.project import SourceRole
from mmos.validation import validate_experiment
from mmos.validation.temporal import validate_temporal
from tests.helpers import make_experiment_project, regression_frame


def write_dataset(directory: Path, rows: int = 40) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "series.csv"
    periods = list(range(1, rows + 1))
    x1 = [float((period * 3) % 10) for period in periods]
    x2 = [float((period * 5) % 7) for period in periods]
    y = [
        10.0 + 3.0 * a + 2.0 * b + 0.3 * math.sin(period)
        for period, a, b in zip(periods, x1, x2, strict=True)
    ]
    pl.DataFrame({"period": periods, "x1": x1, "x2": x2, "y": y}).write_csv(path)
    return path


def write_metadata(
    directory: Path, availability: FeatureAvailability = FeatureAvailability.KNOWN_FUTURE
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset_metadata.json"
    metadata = DatasetMetadata(
        target_name="y",
        target_description="demand per period",
        target_unit=None,
        unit_status="synthetic_units",
        temporal=TemporalMetadata(
            is_temporal=True,
            time_column="period",
            frequency="1 period",
            forecast_horizon=1,
            seasonal_period=None,
            ordering_required=True,
            future_feature_policy="known_future_only",
        ),
        features=[
            FeatureMetadata(name="x1", availability=availability),
            FeatureMetadata(name="x2", availability=availability),
        ],
        provenance=DatasetProvenance(
            is_synthetic=True,
            generator="linear_demand_generator",
            generation_parameters={
                "coefficients": "x1=3, x2=2",
                "noise": "0.3*sin(period)",
            },
            seed=20261112,
            code_path="tests/test_temporal.py",
            description="synthetic demonstration series",
        ),
    )
    path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return path


def make_temporal_project(
    tmp_path: Path,
    *,
    temporal_experiment: bool = True,
    availability: FeatureAvailability = FeatureAvailability.KNOWN_FUTURE,
):
    paths, _ = create_project(tmp_path / "projects", "temporal-demo")
    data_path = write_dataset(tmp_path / "input")
    metadata_path = write_metadata(tmp_path / "input", availability)
    ingest_files(paths, [data_path], SourceRole.DATA)
    ingest_files(paths, [metadata_path], SourceRole.ATTACHMENT)

    split = (
        TemporalSplit(horizon=1, min_train_size=35, step=1, n_splits=5)
        if temporal_experiment
        else None
    )
    spec = ExperimentSpec(
        experiment_id="exp-temporal",
        problem_id="demo",
        proposal_id="p1",
        dataset_hash=dataset_fingerprint(load_manifest(paths)),
        target_column="y",
        feature_set=["x1", "x2"],
        split_strategy="rolling_origin" if temporal_experiment else "kfold",
        validation_strategy="rolling_origin" if temporal_experiment else "cross_validation",
        n_splits=5,
        models=(
            ["naive.last_value", "naive.drift", "sklearn.linear_regression"]
            if temporal_experiment
            else ["sklearn.linear_regression"]
        ),
        metrics=["rmse", "mae", "r2", "mase"] if temporal_experiment else ["rmse", "mae"],
        random_seed=0,
        required_artifacts=["metrics.json", "predictions.parquet"],
        acceptance_criteria=["beat baseline"],
        failure_criteria=["worse than baseline"],
        compute_requirements="test",
        expected_runtime_class="seconds",
        time_column="period" if temporal_experiment else None,
        temporal_split=split,
    )
    save_experiment_spec(paths, spec)
    return paths, spec


def test_forecast_baselines_follow_their_definitions() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert forecast_baseline("naive.last_value", y, 2, None).tolist() == [3.0, 3.0]
    assert forecast_baseline("naive.drift", y, 2, None).tolist() == [4.0, 5.0]
    assert forecast_baseline("naive.seasonal", y, 3, 2).tolist() == [2.0, 3.0, 2.0]
    with pytest.raises(ValueError, match="seasonal_period"):
        forecast_baseline("naive.seasonal", y, 1, None)


def test_temporal_experiment_uses_ordered_folds_and_baselines(tmp_path: Path) -> None:
    paths, spec = make_temporal_project(tmp_path)

    result = run_experiment(paths, spec.experiment_id)

    assert len(result.fold_windows) == 5
    for window in result.fold_windows:
        assert window.train_end <= window.test_start
    assert result.target_summary is not None
    assert result.target_summary.is_synthetic is True
    assert result.target_summary.generator == "linear_demand_generator"
    assert result.target_summary.generation_seed == 20261112
    assert result.target_summary.unit_status.value == "synthetic_units"
    assert result.evaluations["sklearn.linear_regression"].n_observations == 5
    assert result.information_sets["naive.last_value"].information_set_id.startswith(
        "historical_y_only"
    )
    assert result.information_sets["sklearn.linear_regression"].information_set_id.startswith(
        "features:"
    )
    for record in result.evaluations.values():
        assert record.sse is not None and record.sst is not None
        assert record.target_std_population and record.target_std_population > 0
        assert record.r2 is not None
        identity = (record.rmse / record.target_std_population) ** 2
        assert abs(identity - (1.0 - record.r2)) < 1e-9
        assert len(record.mase_details) == 5
        for detail in record.mase_details:
            assert detail.scale_denominator > 0
            assert detail.train_end <= detail.fold_id + 35

    for metrics in result.models.values():
        assert "mase" in metrics
        assert "rmse_eval_std_ratio" in metrics
    assert (
        result.models["sklearn.linear_regression"]["rmse"]
        < result.models["naive.last_value"]["rmse"]
    )
    assert all(result.model_determinism.values())

    predictions = pl.read_parquet(
        paths.experiment_dir(spec.experiment_id) / "artifacts" / "predictions.parquet"
    )
    assert predictions.height == 5 * 3
    assert predictions["row_index"].n_unique() == 5

    reports = validate_experiment(paths, spec.experiment_id)
    temporal_report = next(report for report in reports if report.validator == "temporal")
    assert temporal_report.blocking is False

    freeze_experiment(paths, spec.experiment_id)
    index = load_evidence_index(paths)
    assert any(record.key == "target.std" for record in index.records.values())
    assert any(record.key == "temporal.horizon" for record in index.records.values())
    assert any(record.key == "target.generation_parameters" for record in index.records.values())
    assert any(
        record.key.startswith("uncertainty.sklearn.linear_regression.rmse")
        for record in index.records.values()
    )

    paper = build_paper(paths, spec.experiment_id)
    text = render_markdown(paper)
    assert "Target context" in text
    assert "Temporal evaluation" in text
    assert "Rolling-origin" in text
    assert "synthetic demonstration dataset" in text
    assert "best-scoring candidate in this synthetic sanity run" in text
    assert "mase" in text


def test_temporal_metadata_with_random_split_is_blocked(tmp_path: Path) -> None:
    paths, spec = make_temporal_project(tmp_path, temporal_experiment=False)
    run_experiment(paths, spec.experiment_id)

    report = validate_temporal(paths, spec.experiment_id)

    assert report.blocking is True
    assert any(issue.code == "temporal_random_split" for issue in report.issues)


def test_historical_only_feature_is_blocked(tmp_path: Path) -> None:
    paths, spec = make_temporal_project(tmp_path, availability=FeatureAvailability.HISTORICAL_ONLY)
    run_experiment(paths, spec.experiment_id)

    report = validate_temporal(paths, spec.experiment_id)

    assert report.blocking is True
    assert any(issue.code == "future_unavailable_feature" for issue in report.issues)


def test_fold_window_must_preserve_time_order() -> None:
    with pytest.raises(ValidationError):
        FoldWindow(train_start=0, train_end=10, test_start=9, test_end=12)


def test_provisional_paper_wording_is_not_final(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse", "mae"],
    )
    run_experiment(paths, spec.experiment_id)
    validate_experiment(paths, spec.experiment_id)
    freeze_experiment(paths, spec.experiment_id)

    paper = build_paper(paths, spec.experiment_id, model_status="provisional_winner")
    text = render_markdown(paper)
    assert "provisional leading candidate" in text
    assert "verified leading candidate" not in text
