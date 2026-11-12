from enum import StrEnum
from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class Issue(MMOSModel):
    code: str
    severity: Severity
    message: str
    table: str | None = None
    column: str | None = None


class ColumnProfile(MMOSModel):
    name: str
    dtype: str
    null_count: int = Field(ge=0)
    null_fraction: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)
    min: str | None = None
    max: str | None = None
    mean: float | None = None
    std: float | None = None
    p50: float | None = None


class TableProfile(MMOSModel):
    name: str
    path: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    duplicate_row_count: int = Field(ge=0)
    columns: list[ColumnProfile]


class DatasetProfile(MMOSModel):
    tables: list[TableProfile]

    def table(self, name: str) -> TableProfile:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"unknown table: {name}")


class ColumnSpec(MMOSModel):
    name: str
    dtype: str
    nullable: bool


class TableSchema(MMOSModel):
    name: str
    path: str
    columns: list[ColumnSpec]
    candidate_keys: list[str] = Field(default_factory=list)


class DataSchema(MMOSModel):
    tables: list[TableSchema]


class DataQualityReport(MMOSModel):
    status: QualityStatus
    blocking: bool
    tables_checked: list[str]
    issues: list[Issue] = Field(default_factory=list)


class LeakageFinding(MMOSModel):
    code: str
    severity: Severity
    message: str
    table: str
    columns: list[str] = Field(default_factory=list)


class LeakageReport(MMOSModel):
    status: Literal["passed", "failed"]
    blocking: bool
    findings: list[LeakageFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
