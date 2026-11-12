from datetime import UTC, datetime
from pathlib import Path

import pytest

from mmos.core.project import create_project, save_tournament_result
from mmos.experiments.tournament import Tournament
from mmos.registries import ModelRegistry
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.registry import TaskType
from mmos.schemas.tournament import TournamentResult

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"
NOW = datetime(2026, 11, 12, tzinfo=UTC)


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load(REGISTRY_PATH)


def make_result(
    values: dict[str, float],
    *,
    metric: str = "rmse",
    experiment_id: str = "exp-1",
    dataset_hash: str = "a" * 64,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        dataset_hash=dataset_hash,
        task_type=TaskType.REGRESSION,
        feature_set=["x1"],
        split_strategy="kfold",
        validation_strategy="cross_validation",
        random_seed=0,
        n_splits=5,
        models={model_id: {metric: value} for model_id, value in values.items()},
        generated_at=NOW,
    )


def test_best_candidate_that_beats_baseline_is_selected(registry: ModelRegistry) -> None:
    result = make_result(
        {
            "sklearn.dummy_regressor": 2.0,
            "sklearn.linear_regression": 0.5,
            "sklearn.ridge": 0.7,
        }
    )

    tournament = Tournament(registry).run([result], primary_metric="rmse", higher_is_better=False)

    assert tournament.blocked is False
    assert tournament.selected_model == "sklearn.linear_regression"
    assert tournament.challenger_model == "sklearn.dummy_regressor"
    ranks = {entry.model_id: entry.rank for entry in tournament.entries}
    assert ranks["sklearn.linear_regression"] == 1
    assert ranks["sklearn.ridge"] == 2
    assert ranks["sklearn.dummy_regressor"] == 3


def test_higher_is_better_direction(registry: ModelRegistry) -> None:
    result = make_result(
        {"sklearn.dummy_regressor": 0.0, "sklearn.linear_regression": 0.9}, metric="r2"
    )

    tournament = Tournament(registry).run([result], primary_metric="r2", higher_is_better=True)

    assert tournament.selected_model == "sklearn.linear_regression"


def test_tournament_without_baseline_is_blocked(registry: ModelRegistry) -> None:
    result = make_result({"sklearn.linear_regression": 0.5})

    tournament = Tournament(registry).run([result], primary_metric="rmse", higher_is_better=False)

    assert tournament.blocked is True
    assert tournament.selected_model is None
    assert any("no registered baseline" in note for note in tournament.notes)


def test_baseline_wins_when_no_candidate_is_better(registry: ModelRegistry) -> None:
    result = make_result({"sklearn.dummy_regressor": 1.0, "sklearn.linear_regression": 2.0})

    tournament = Tournament(registry).run([result], primary_metric="rmse", higher_is_better=False)

    assert tournament.blocked is True
    assert tournament.selected_model is None
    assert tournament.challenger_model == "sklearn.dummy_regressor"
    assert any("baseline wins" in note for note in tournament.notes)


def test_incomparable_datasets_are_rejected(registry: ModelRegistry) -> None:
    first = make_result({"sklearn.dummy_regressor": 1.0}, dataset_hash="a" * 64)
    second = make_result(
        {"sklearn.linear_regression": 0.5},
        experiment_id="exp-2",
        dataset_hash="b" * 64,
    )

    with pytest.raises(ValueError, match="not comparable: dataset_hash"):
        Tournament(registry).run([first, second], primary_metric="rmse", higher_is_better=False)


def test_missing_primary_metric_is_rejected(registry: ModelRegistry) -> None:
    result = make_result({"sklearn.dummy_regressor": 1.0}, metric="mae")

    with pytest.raises(ValueError, match="not present"):
        Tournament(registry).run([result], primary_metric="rmse", higher_is_better=False)


def test_tournament_result_persists(tmp_path: Path, registry: ModelRegistry) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    tournament = Tournament(registry).run(
        [make_result({"sklearn.dummy_regressor": 2.0, "sklearn.linear_regression": 0.5})],
        primary_metric="rmse",
        higher_is_better=False,
    )

    target = save_tournament_result(paths, tournament)

    assert target.exists()
    saved = TournamentResult.model_validate_json(target.read_text(encoding="utf-8"))
    assert saved == tournament
