from pathlib import Path

import pytest

from mmos.registries import ModelRegistry
from mmos.schemas.registry import ModelCapability, TaskType

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"


def make_capability(model_id: str = "demo.model") -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        name="Demo",
        task_types=[TaskType.REGRESSION],
        description="demo",
        recommended_validation="kfold",
    )


def test_committed_registry_loads() -> None:
    registry = ModelRegistry.load(REGISTRY_PATH)
    assert len(registry) >= 10


def test_registry_contains_baselines_for_regression_and_forecasting() -> None:
    registry = ModelRegistry.load(REGISTRY_PATH)
    assert any(capability.is_baseline for capability in registry.find(TaskType.REGRESSION))
    assert any(capability.is_baseline for capability in registry.find(TaskType.FORECASTING))


def test_unknown_model_id_raises_helpful_error() -> None:
    registry = ModelRegistry.load(REGISTRY_PATH)
    with pytest.raises(KeyError, match="unknown model_id"):
        registry.get("does.not.exist")


def test_duplicate_model_ids_are_rejected() -> None:
    capability = make_capability("duplicate")
    with pytest.raises(ValueError, match="duplicate model_id"):
        ModelRegistry([capability, capability])


def test_find_filters_by_task_type() -> None:
    registry = ModelRegistry.load(REGISTRY_PATH)
    regression = registry.find(TaskType.REGRESSION)
    assert regression
    assert all(TaskType.REGRESSION in capability.task_types for capability in regression)
