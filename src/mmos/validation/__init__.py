import json
from collections.abc import Callable

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.validation import ValidationReport
from mmos.validation.leakage import validate_leakage
from mmos.validation.numerical import validate_numerical
from mmos.validation.reproducibility import validate_reproducibility
from mmos.validation.statistical import validate_statistical

Validator = Callable[[ProjectPaths, str], ValidationReport]

VALIDATORS: dict[str, Validator] = {
    "leakage": validate_leakage,
    "numerical": validate_numerical,
    "statistical": validate_statistical,
    "reproducibility": validate_reproducibility,
}


def validate_experiment(
    paths: ProjectPaths, experiment_id: str, validators: list[str] | None = None
) -> list[ValidationReport]:
    names = validators or list(VALIDATORS)
    unknown = [name for name in names if name not in VALIDATORS]
    if unknown:
        raise ValueError(
            f"unknown validator(s): {', '.join(unknown)}; available: {', '.join(VALIDATORS)}"
        )

    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[ValidationReport] = []
    for name in names:
        report = VALIDATORS[name](paths, experiment_id)
        atomic_write_text(output_dir / f"{name}.json", report.model_dump_json(indent=2) + "\n")
        reports.append(report)

    summary = {
        "experiment_id": experiment_id,
        "blocking": any(report.blocking for report in reports),
        "reports": {
            report.validator: {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
            for report in reports
        },
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    return reports


__all__ = ["VALIDATORS", "Validator", "validate_experiment"]
