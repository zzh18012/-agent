from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from mmos.experiments.equivalence import paired_comparison
from mmos.experiments.tournament import Tournament
from mmos.registries import ModelRegistry
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.metadata import InformationSet
from mmos.schemas.registry import TaskType

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"
NOW = datetime(2026, 11, 12, tzinfo=UTC)


def make_info(identifier: str) -> InformationSet:
    return InformationSet(
        information_set_id=identifier,
        target_history_available=identifier == "historical",
        feature_columns=[] if identifier != "features:x1,x2" else ["x1", "x2"],
        feature_availability={"x1": "known_future", "x2": "known_future"}
        if identifier == "features:x1,x2"
        else {},
    )


def predictions_frame(offsets: dict[str, float]) -> pl.DataFrame:
    rows = []
    for index in range(20):
        actual = float(index + 1)
        for model_id, offset in offsets.items():
            rows.append(
                {
                    "row_index": index,
                    "model": model_id,
                    "prediction": actual + offset * (1 if index % 2 == 0 else -1),
                    "actual": actual,
                }
            )
    return pl.DataFrame(rows)


def test_identical_predictions_are_equivalent() -> None:
    frame = predictions_frame({"a": 0.0, "b": 0.0})
    comparison = paired_comparison(
        frame,
        "a",
        "b",
        candidate_information=make_info("features:x1,x2"),
        baseline_information=make_info("features:x1,x2"),
        threshold=None,
    )
    assert comparison.status == "equivalent"
    assert comparison.n_pairs == 20
    assert comparison.relative_improvement == 0.0
    assert comparison.ci_low == 0.0 and comparison.ci_high == 0.0


def test_clear_difference_is_detected() -> None:
    frame = predictions_frame({"good": 0.0, "bad": 2.0})
    comparison = paired_comparison(
        frame,
        "good",
        "bad",
        candidate_information=make_info("features:x1,x2"),
        baseline_information=make_info("features:x1,x2"),
        threshold=None,
    )
    assert comparison.status == "different"
    assert comparison.relative_improvement == 1.0


def test_information_set_mismatch_is_incomparable() -> None:
    frame = predictions_frame({"candidate": 0.0, "naive": 1.0})
    comparison = paired_comparison(
        frame,
        "candidate",
        "naive",
        candidate_information=make_info("features:x1,x2"),
        baseline_information=make_info("historical"),
        threshold=None,
    )
    assert comparison.status == "incomparable"
    assert comparison.information_set_match is False


def test_threshold_can_declare_small_differences_equivalent() -> None:
    frame = predictions_frame({"a": 2.0, "b": 2.01})
    comparison = paired_comparison(
        frame,
        "a",
        "b",
        candidate_information=make_info("features:x1,x2"),
        baseline_information=make_info("features:x1,x2"),
        threshold=0.1,
    )
    assert comparison.status == "equivalent"


def test_tournament_returns_equivalent_for_matched_information() -> None:
    offsets = {
        "sklearn.linear_regression": 2.0,
        "sklearn.ridge": 2.01,
        "naive.last_value": 5.0,
        "sklearn.dummy_regressor": 4.0,
    }
    frame = predictions_frame(offsets)

    def rmse(model_id: str) -> float:
        subset = frame.filter(pl.col("model") == model_id)
        errors = subset["prediction"].to_numpy() - subset["actual"].to_numpy()
        return float(np.sqrt(np.mean(errors**2)))

    metrics = {
        model_id: {"rmse": rmse(model_id), "mae": abs(offset)}
        for model_id, offset in offsets.items()
    }
    result = ExperimentResult(
        experiment_id="exp-equivalence",
        dataset_hash="a" * 64,
        task_type=TaskType.FORECASTING,
        feature_set=["x1", "x2"],
        split_strategy="rolling_origin",
        validation_strategy="rolling_origin",
        random_seed=0,
        n_splits=5,
        models=metrics,
        information_sets={
            "sklearn.linear_regression": make_info("features:x1,x2"),
            "sklearn.ridge": make_info("features:x1,x2"),
            "naive.last_value": make_info("historical"),
            "sklearn.dummy_regressor": make_info("constant"),
        },
        generated_at=NOW,
    )

    tournament = Tournament(ModelRegistry.load(REGISTRY_PATH)).run(
        [result],
        primary_metric="rmse",
        higher_is_better=False,
        predictions=frame,
        improvement_threshold=0.01,
    )

    assert tournament.selection_status == "equivalent"
    assert tournament.selected_model == "sklearn.linear_regression"
    assert tournament.challenger_model == "sklearn.ridge"
    matched = [
        comparison for comparison in tournament.equivalence if comparison.information_set_match
    ]
    assert matched and matched[0].status == "equivalent"
    external = [
        comparison for comparison in tournament.equivalence if not comparison.information_set_match
    ]
    assert external and all(comparison.status == "incomparable" for comparison in external)
