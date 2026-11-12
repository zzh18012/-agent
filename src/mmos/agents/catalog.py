from mmos.experiments.engines import ENGINES, TEMPORAL_BASELINES
from mmos.registries import ModelRegistry

EXECUTABLE_MODEL_IDS = frozenset(ENGINES) | TEMPORAL_BASELINES


def registry_catalog(registry: ModelRegistry) -> str:
    lines = []
    for capability in registry.find():
        task_types = ", ".join(task.value for task in capability.task_types)
        lines.append(
            f"- {capability.model_id} [{task_types}] "
            f"(interpretability: {capability.interpretability.value}, "
            f"compute: {capability.compute_class.value}): {capability.description}"
        )
    return "\n".join(lines)


def executable_catalog(registry: ModelRegistry) -> str:
    lines = []
    for model_id in sorted(EXECUTABLE_MODEL_IDS):
        if not registry.has(model_id):
            continue
        capability = registry.get(model_id)
        task_types = ", ".join(task.value for task in capability.task_types)
        lines.append(
            f"- {capability.model_id} [{task_types}] "
            f"(interpretability: {capability.interpretability.value}, "
            f"compute: {capability.compute_class.value}): {capability.description}"
        )
    return "\n".join(lines)
