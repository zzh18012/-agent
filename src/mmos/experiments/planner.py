import secrets
from typing import Literal

from mmos.registries import ModelRegistry
from mmos.schemas.experiment import ExperimentSpec
from mmos.schemas.proposal import ModelProposal
from mmos.schemas.registry import ComputeClass, TaskType

DEFAULT_METRICS: dict[TaskType, list[str]] = {
    TaskType.REGRESSION: ["rmse", "mae", "r2"],
    TaskType.CLASSIFICATION: ["accuracy", "f1_macro", "roc_auc"],
    TaskType.FORECASTING: ["mae", "rmse", "mase"],
}

RuntimeClass = Literal["seconds", "minutes", "hours"]

_RUNTIME_CLASS: dict[ComputeClass, RuntimeClass] = {
    ComputeClass.TINY: "seconds",
    ComputeClass.SMALL: "seconds",
    ComputeClass.MEDIUM: "minutes",
    ComputeClass.LARGE: "hours",
}

_SPLIT: dict[TaskType, tuple[str, str]] = {
    TaskType.FORECASTING: ("rolling_origin", "time_series_cross_validation"),
}
_DEFAULT_SPLIT = ("kfold", "cross_validation")


class ExperimentPlanner:
    """Turns a proposal into a reproducible experiment contract.

    The planner never decides which model is best; it only specifies a fair,
    comparable run that the executor can carry out without further agent input.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def plan(
        self,
        proposal: ModelProposal,
        *,
        dataset_hash: str,
        feature_set: list[str],
        metrics: list[str] | None = None,
        n_splits: int = 5,
        random_seed: int = 0,
        dataset_version: str | None = None,
        acceptance_criteria: list[str] | None = None,
        failure_criteria: list[str] | None = None,
    ) -> ExperimentSpec:
        if not proposal.registry_model_ids:
            raise ValueError(
                f"proposal {proposal.proposal_id!r} names no registry model; "
                "an experiment cannot be planned"
            )
        capabilities = self.registry.require(proposal.registry_model_ids)
        task_type = capabilities[0].task_types[0]

        resolved_metrics = metrics or DEFAULT_METRICS.get(task_type)
        if not resolved_metrics:
            raise ValueError(
                f"no default metrics for task type {task_type.value!r}; pass metrics explicitly"
            )

        split_strategy, validation_strategy = _SPLIT.get(task_type, _DEFAULT_SPLIT)
        compute = capabilities[0].compute_class

        return ExperimentSpec(
            experiment_id=f"exp-{proposal.proposal_id}-{secrets.token_hex(3)}",
            problem_id=proposal.problem_id,
            proposal_id=proposal.proposal_id,
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            feature_set=feature_set,
            split_strategy=split_strategy,
            validation_strategy=validation_strategy,
            n_splits=n_splits,
            models=proposal.registry_model_ids,
            metrics=resolved_metrics,
            random_seed=random_seed,
            required_artifacts=["metrics.json", "predictions.parquet"],
            acceptance_criteria=acceptance_criteria
            or [
                "beats every registered baseline on the primary metric",
                "matches the planned split and metric definitions",
            ],
            failure_criteria=failure_criteria
            or [
                "worse than baseline on the primary metric",
                "validation report contains a blocking issue",
            ],
            compute_requirements=", ".join(
                f"{capability.model_id} ({capability.compute_class.value})"
                for capability in capabilities
            ),
            expected_runtime_class=_RUNTIME_CLASS[compute],
        )
