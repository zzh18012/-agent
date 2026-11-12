from mmos.core.project import ProjectPaths, load_experiment_spec
from mmos.data.metadata import load_dataset_metadata
from mmos.experiments.engines import TEMPORAL_BASELINES
from mmos.schemas.data import Severity
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.metadata import FeatureAvailability
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report


def validate_temporal(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = load_experiment_spec(paths, experiment_id)
    result = ExperimentResult.model_validate_json(
        (paths.experiment_dir(experiment_id) / "artifacts" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = load_dataset_metadata(paths)
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.WARNING, message=message))

    declared_temporal = bool(metadata is not None and metadata.temporal.is_temporal)
    if spec.temporal_split is None:
        if declared_temporal:
            error(
                "temporal_random_split",
                "dataset metadata declares temporal data, but the experiment does not use "
                "a rolling-origin split",
            )
        return build_report(
            "temporal",
            experiment_id,
            issues,
            evidence=[str(paths.experiment_dir(experiment_id) / "experiment_spec.json")],
            recommendations=["random KFold is forbidden for declared temporal data"],
        )

    if spec.split_strategy != "rolling_origin":
        error("temporal_split_strategy", "temporal experiments must use rolling_origin")
    if not result.fold_windows:
        error("missing_fold_windows", "temporal result has no fold windows")
    for window in result.fold_windows:
        if window.train_end > window.test_start:
            error(
                "temporal_order_violation",
                "a fold uses test data before or during the training window",
            )
    if result.target_summary is None:
        error("missing_target_context", "temporal results must record target metadata")

    baseline_ids = {model_id for model_id in spec.models if model_id in TEMPORAL_BASELINES}
    if not baseline_ids:
        error(
            "missing_temporal_baseline",
            "temporal experiments must include a naive/drift/seasonal baseline",
        )
    if "naive.seasonal" in spec.models and spec.seasonal_period is None:
        error("seasonal_without_period", "seasonal naive requires a confirmed seasonal_period")
    if spec.seasonal_period is not None and "naive.seasonal" not in spec.models:
        warning(
            "seasonal_baseline_missing",
            "a seasonal period is known, but no seasonal naive baseline was evaluated",
        )
    if spec.time_column is not None and spec.time_column in spec.feature_set:
        warning(
            "time_column_as_feature",
            "the time column is used as a model feature; justify that this is not leakage",
        )

    if metadata is not None:
        availability = {feature.name: feature.availability for feature in metadata.features}
        for feature in spec.feature_set:
            status = availability.get(feature)
            if status is None or status is FeatureAvailability.UNKNOWN:
                warning(
                    "unknown_feature_availability",
                    f"availability of feature {feature!r} at prediction time is unknown",
                )
            elif status is FeatureAvailability.HISTORICAL_ONLY:
                error(
                    "future_unavailable_feature",
                    f"feature {feature!r} is historical-only and cannot be used at forecast time",
                )
            elif status is FeatureAvailability.FORECASTED_FUTURE:
                warning(
                    "forecasted_feature",
                    f"feature {feature!r} must itself be forecast; document that dependency",
                )

    return build_report(
        "temporal",
        experiment_id,
        issues,
        evidence=[
            str(paths.experiment_dir(experiment_id) / "experiment_spec.json"),
            str(paths.experiment_dir(experiment_id) / "artifacts" / "metrics.json"),
        ],
        recommendations=[
            "verify that every fold satisfies max(train time) < min(test time)",
            "compare the candidate against the strongest naive baseline",
        ],
    )
