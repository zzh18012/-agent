import statistics

import polars as pl

from mmos.core.project import ProjectPaths
from mmos.schemas.data import Severity
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

MIN_SAMPLE_SIZE = 30
MIN_FOLDS = 5
HIGH_VARIATION_RATIO = 0.5


def validate_statistical(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    artifacts = paths.experiments_dir / experiment_id / "artifacts"
    result = ExperimentResult.model_validate_json(
        (artifacts / "metrics.json").read_text(encoding="utf-8")
    )
    predictions = pl.read_parquet(artifacts / "predictions.parquet")
    issues: list[ValidationIssue] = []

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.WARNING, message=message))

    if result.n_splits < MIN_FOLDS:
        warning("few_folds", f"only {result.n_splits} folds; estimates are noisy")

    if not predictions.is_empty():
        first_model = next(iter(result.models))
        row_count = predictions.filter(pl.col("model") == first_model).height
        if 0 < row_count < MIN_SAMPLE_SIZE:
            warning("small_sample", f"only {row_count} rows; inference is fragile")

    if not result.fold_metrics:
        warning("missing_fold_metrics", "no per-fold metrics recorded; variance cannot be audited")
    else:
        for model_id, folds in result.fold_metrics.items():
            if len(folds) != result.n_splits:
                warning(
                    "fold_count_mismatch",
                    f"{model_id}: {len(folds)} fold records for {result.n_splits} folds",
                )
            metric_names = set(folds[0]) if folds else set()
            for metric_name in sorted(metric_names):
                values = [fold[metric_name] for fold in folds if metric_name in fold]
                if len(values) < 2:
                    continue
                mean = statistics.fmean(values)
                deviation = statistics.stdev(values)
                if mean != 0 and deviation > HIGH_VARIATION_RATIO * abs(mean):
                    warning(
                        "high_fold_variance",
                        f"{model_id}/{metric_name}: fold std {deviation:.4g} "
                        f"exceeds {HIGH_VARIATION_RATIO:.0%} of mean {mean:.4g}",
                    )

    return build_report(
        "statistical", experiment_id, issues, evidence=[str(artifacts / "metrics.json")]
    )
