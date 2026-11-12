from collections.abc import Iterable
from pathlib import Path

import yaml

from mmos.schemas.registry import ModelCapability, TaskType


class ModelRegistry:
    """Capability catalog. It describes models; it does not run them."""

    def __init__(self, capabilities: Iterable[ModelCapability]) -> None:
        self._by_id: dict[str, ModelCapability] = {}
        for capability in capabilities:
            if capability.model_id in self._by_id:
                raise ValueError(f"duplicate model_id in registry: {capability.model_id}")
            self._by_id[capability.model_id] = capability

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = data["models"] if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError(f"registry file must contain a list of models: {path}")
        return cls(ModelCapability.model_validate(entry) for entry in entries)

    def get(self, model_id: str) -> ModelCapability:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise KeyError(
                f"unknown model_id {model_id!r}; see model_registry/registry.yaml"
            ) from None

    def require(self, model_ids: Iterable[str]) -> list[ModelCapability]:
        return [self.get(model_id) for model_id in model_ids]

    def has(self, model_id: str) -> bool:
        return model_id in self._by_id

    def ids(self) -> set[str]:
        return set(self._by_id)

    def find(self, task_type: TaskType | None = None) -> list[ModelCapability]:
        capabilities = list(self._by_id.values())
        if task_type is None:
            return capabilities
        return [capability for capability in capabilities if task_type in capability.task_types]

    def __len__(self) -> int:
        return len(self._by_id)
