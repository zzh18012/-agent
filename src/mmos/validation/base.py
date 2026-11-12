from datetime import UTC, datetime

from mmos.schemas.data import Severity
from mmos.schemas.validation import ValidationIssue, ValidationReport, ValidationStatus

_SEVERITY_ORDER = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.ERROR: 2,
}


def build_report(
    validator: str,
    subject_id: str,
    issues: list[ValidationIssue],
    *,
    evidence: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> ValidationReport:
    blocking = any(issue.severity is Severity.ERROR for issue in issues)
    severity = max(
        (issue.severity for issue in issues), key=_SEVERITY_ORDER.__getitem__, default=Severity.INFO
    )
    return ValidationReport(
        validator=validator,
        subject_id=subject_id,
        status=ValidationStatus.FAILED if blocking else ValidationStatus.PASSED,
        severity=severity,
        issues=issues,
        evidence=evidence or [],
        recommendations=recommendations or [],
        blocking=blocking,
        generated_at=datetime.now(UTC).replace(microsecond=0),
    )
