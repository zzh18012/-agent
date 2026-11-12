from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import ProjectPaths, load_experiment_spec, load_manifest
from mmos.data.leakage import check_leakage
from mmos.data.profiling import profile_table
from mmos.data.table import load_single_table
from mmos.schemas.data import Severity
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report


def validate_leakage(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = load_experiment_spec(paths, experiment_id)
    issues: list[ValidationIssue] = []
    evidence = [str(paths.experiment_dir(experiment_id) / "experiment_spec.json")]

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    if spec.target_column in spec.feature_set:
        error("target_in_features", f"target {spec.target_column!r} is also a feature")

    fingerprint = dataset_fingerprint(load_manifest(paths))
    if fingerprint != spec.dataset_hash:
        error("stale_dataset", "the project data changed after this experiment was planned")

    df = load_single_table(paths)
    if spec.target_column not in df.columns:
        error("missing_target", f"target column {spec.target_column!r} is not in the data")
    else:
        profile = profile_table("data", "data/raw", df)
        for finding in check_leakage(profile, df, target=spec.target_column):
            issues.append(
                ValidationIssue(
                    code=finding.code,
                    severity=finding.severity,
                    message=finding.message,
                )
            )

    return build_report(
        "leakage",
        experiment_id,
        issues,
        evidence=evidence,
        recommendations=[
            "train/test overlap and true temporal leakage require split-level review",
        ],
    )
