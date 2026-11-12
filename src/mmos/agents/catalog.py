from mmos.registries import ModelRegistry


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
