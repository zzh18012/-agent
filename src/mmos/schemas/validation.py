from datetime import datetime
from enum import StrEnum

from pydantic import Field

from mmos.schemas.common import MMOSModel
from mmos.schemas.data import Severity


class ValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class ValidationIssue(MMOSModel):
    code: str
    severity: Severity
    message: str


class ValidationReport(MMOSModel):
    validator: str
    subject_id: str
    status: ValidationStatus
    severity: Severity
    issues: list[ValidationIssue] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    blocking: bool
    generated_at: datetime
