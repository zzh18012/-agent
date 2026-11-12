from pathlib import Path

import pytest

from mmos.core.project import (
    create_project,
    load_experiment_spec,
    save_experiment_spec,
)
from mmos.experiments import ExperimentPlanner
from mmos.registries import ModelRegistry
from mmos.schemas.proposal import Complexity, ExpertRole, ModelProposal

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"
DATASET_HASH = "a" * 64


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load(REGISTRY_PATH)


def make_proposal(model_ids: list[str]) -> ModelProposal:
    return ModelProposal(
        proposal_id="p1",
        problem_id="proj-1",
        subproblem_id="q1",
        name="Ridge regression",
        model_family="linear",
        expert_role=ExpertRole.STATISTICIAN,
        description="regularized linear model",
        estimated_complexity=Complexity.LOW,
        registry_model_ids=model_ids,
    )


def test_plan_produces_reproducible_contract(registry: ModelRegistry) -> None:
    planner = ExperimentPlanner(registry)

    spec = planner.plan(
        make_proposal(["sklearn.ridge"]),
        dataset_hash=DATASET_HASH,
        feature_set=["x1", "x2"],
    )

    assert spec.experiment_id.startswith("exp-p1-")
    assert spec.dataset_hash == DATASET_HASH
    assert spec.models == ["sklearn.ridge"]
    assert spec.metrics == ["rmse", "mae", "r2"]
    assert spec.split_strategy == "kfold"
    assert spec.validation_strategy == "cross_validation"
    assert spec.n_splits == 5
    assert spec.random_seed == 0
    assert spec.expected_runtime_class == "seconds"
    assert spec.acceptance_criteria and spec.failure_criteria


def test_forecasting_uses_time_series_split(registry: ModelRegistry) -> None:
    planner = ExperimentPlanner(registry)

    spec = planner.plan(
        make_proposal(["naive.last_value"]),
        dataset_hash=DATASET_HASH,
        feature_set=["y"],
    )

    assert spec.split_strategy == "rolling_origin"
    assert spec.validation_strategy == "time_series_cross_validation"
    assert spec.metrics == ["mae", "rmse", "mase"]


def test_explicit_metrics_override_defaults(registry: ModelRegistry) -> None:
    spec = ExperimentPlanner(registry).plan(
        make_proposal(["sklearn.ridge"]),
        dataset_hash=DATASET_HASH,
        feature_set=["x1"],
        metrics=["mae"],
    )
    assert spec.metrics == ["mae"]


def test_proposal_without_registry_model_cannot_be_planned(registry: ModelRegistry) -> None:
    with pytest.raises(ValueError, match="names no registry model"):
        ExperimentPlanner(registry).plan(
            make_proposal([]),
            dataset_hash=DATASET_HASH,
            feature_set=["x1"],
        )


def test_unknown_registry_model_is_rejected(registry: ModelRegistry) -> None:
    with pytest.raises(KeyError, match="unknown model_id"):
        ExperimentPlanner(registry).plan(
            make_proposal(["not.registered"]),
            dataset_hash=DATASET_HASH,
            feature_set=["x1"],
        )


def test_spec_roundtrips_through_the_project(tmp_path: Path, registry: ModelRegistry) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    spec = ExperimentPlanner(registry).plan(
        make_proposal(["sklearn.ridge"]),
        dataset_hash=DATASET_HASH,
        feature_set=["x1"],
    )

    target = save_experiment_spec(paths, spec)

    assert target.exists()
    assert load_experiment_spec(paths, spec.experiment_id) == spec
