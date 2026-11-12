from datetime import datetime
from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel


class EvidenceRecord(MMOSModel):
    """One verified number or artifact entry. The paper layer may cite only these."""

    evidence_id: str
    key: str
    raw_value: float | int | str
    display_value: str
    unit: str | None = None
    experiment_id: str
    artifact_path: str
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_field: str
    validation_reports: list[str] = Field(min_length=1)
    verified_at: datetime
    provenance: dict[str, str] = Field(default_factory=dict)


class EvidenceIndex(MMOSModel):
    schema_version: Literal[1] = 1
    records: dict[str, EvidenceRecord] = Field(default_factory=dict)


class ClaimRecord(MMOSModel):
    claim_id: str
    statement: str
    evidence_ids: list[str] = Field(min_length=1)
    required_checks: list[str] = Field(default_factory=list)
    status: Literal["proposed", "supported", "rejected"] = "proposed"
    reviewer: str | None = None
    notes: str = ""
