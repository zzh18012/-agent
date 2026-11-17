from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.competition.clock import ensure_aware
from mmos.schemas.common import MMOSModel


class HistoricalBenchmarkState(StrEnum):
    SEALED = "sealed"
    PROBLEM_REVEALED = "problem_revealed"
    PROBLEM_ANALYZED = "problem_analyzed"
    MODELING = "modeling"
    EXPERIMENTING = "experimenting"
    EVIDENCE_BUILDING = "evidence_building"
    PAPER_WRITING = "paper_writing"
    REVIEWING = "reviewing"
    FROZEN = "frozen"
    PASS = "pass"
    BLOCKED = "blocked"
    INVALIDATED_INFRASTRUCTURE = "invalidated_infrastructure"


class ProtocolValidity(StrEnum):
    VALID = "valid"
    INVALIDATED_INFRASTRUCTURE = "invalidated_infrastructure"


class RequirementStatus(StrEnum):
    ANSWERED_WITH_EVIDENCE = "answered_with_evidence"
    ANSWERED_WITH_LIMITATION = "answered_with_limitation"
    UNRESOLVED = "unresolved"


class BenchmarkFile(MMOSModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class BenchmarkSourceManifest(MMOSModel):
    benchmark_source_id: str
    competition: str
    year: int = Field(ge=2000, le=2100)
    official_source: str
    downloaded_at: datetime
    archive_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_size: int = Field(ge=0)
    extracted_files: list[BenchmarkFile] = Field(default_factory=list)
    manifest_hash: str = ""

    @model_validator(mode="after")
    def _aware(self) -> BenchmarkSourceManifest:
        ensure_aware(self.downloaded_at, "downloaded_at")
        return self


class SelectionCandidate(MMOSModel):
    label: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: int = Field(ge=1)


class ProblemSelectionManifest(MMOSModel):
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels: list[str] = Field(min_length=1)
    ordered: list[SelectionCandidate] = Field(min_length=1)
    primary_label: str
    reserve_labels: list[str] = Field(default_factory=list)
    created_at: datetime
    selection_hash: str = ""

    @model_validator(mode="after")
    def _aware(self) -> ProblemSelectionManifest:
        ensure_aware(self.created_at, "created_at")
        return self


class RevealRecord(MMOSModel):
    selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    problem_label: str
    revealed_at: datetime
    attempt_id: str
    reveal_hash: str = ""

    @model_validator(mode="after")
    def _aware(self) -> RevealRecord:
        ensure_aware(self.revealed_at, "revealed_at")
        return self


class BlindnessPolicy(MMOSModel):
    operational_blind: Literal[True] = True
    training_contamination_verifiable: Literal[False] = False
    web_access_allowed: Literal[False] = False
    solution_artifacts_allowed: Literal[False] = False
    best_of_n_allowed: Literal[False] = False
    one_valid_attempt: Literal[True] = True
    note: str = (
        "Operational blindness verifies runtime information isolation, not absence of "
        "relevant material from pretrained model parameters."
    )


class LeakageEvent(StrEnum):
    EXTERNAL_FILE_ACCESS = "external_file_access"
    WEB_ACCESS_ATTEMPT = "web_access_attempt"
    MANUAL_MATERIAL = "manual_material"
    SOLUTION_PATH_REJECTED = "solution_path_rejected"
    HUMAN_INTERVENTION = "human_intervention"
    SOURCE_ACCESS = "source_access"


class LeakageLogEntry(MMOSModel):
    event_type: LeakageEvent
    occurred_at: datetime
    detail: str
    prohibited: bool = True

    @model_validator(mode="after")
    def _aware(self) -> LeakageLogEntry:
        ensure_aware(self.occurred_at, "occurred_at")
        return self


class BenchmarkCodeFreeze(MMOSModel):
    benchmark_run_id: str
    git_commit: str
    worktree_clean: bool
    test_status: str
    ruff_status: str
    pyright_status: str
    code_tree_hash: str
    created_at: datetime
    freeze_hash: str = ""

    @model_validator(mode="after")
    def _aware(self) -> BenchmarkCodeFreeze:
        ensure_aware(self.created_at, "created_at")
        return self


class ProblemRequirement(MMOSModel):
    requirement_id: str
    source_location: str
    question_text: str
    question_hash: str
    required_output: str
    status: RequirementStatus = RequirementStatus.UNRESOLVED
    model_component: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    paper_section: str | None = None


class BenchmarkBlindnessAudit(MMOSModel):
    checks: dict[str, bool] = Field(default_factory=dict)
    prohibited_access_count: int = Field(ge=0)
    human_scientific_intervention_count: int = Field(ge=0)
    passed: bool
    notes: list[str] = Field(default_factory=list)


class CapabilityGap(MMOSModel):
    gap_id: str
    problem_requirement: str
    failure_mode: str
    existing_capability_attempted: str
    why_insufficient: str
    scientific_consequence: str
    severity: Literal["critical", "major", "minor"]
    possible_future_capability: str


class HistoricalBenchmarkRun(MMOSModel):
    benchmark_run_id: str
    competition: str
    year: int
    problem_label: str | None = None
    state: HistoricalBenchmarkState = HistoricalBenchmarkState.SEALED
    protocol_validity: ProtocolValidity = ProtocolValidity.VALID
    source_manifest_hash: str = ""
    selection_manifest_hash: str = ""
    code_freeze_commit: str = ""
    problem_revealed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    invalidation_reason: str | None = None
    run_hash: str = ""

    @model_validator(mode="after")
    def _aware(self) -> HistoricalBenchmarkRun:
        ensure_aware(self.created_at, "created_at")
        ensure_aware(self.updated_at, "updated_at")
        return self


class BenchmarkResult(MMOSModel):
    benchmark_run_id: str
    competition: str
    year: int
    problem_label: str
    source_archive_hash: str
    selection_manifest_hash: str
    code_freeze_commit: str
    protocol_validity: ProtocolValidity
    blindness_audit: BenchmarkBlindnessAudit | None = None
    router_result: dict[str, object] = Field(default_factory=dict)
    requirement_coverage: dict[str, int] = Field(default_factory=dict)
    experiment_ids: list[str] = Field(default_factory=list)
    failed_experiment_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    model_freeze_hash: str | None = None
    evidence_freeze_hash: str | None = None
    paper_freeze_hash: str | None = None
    paper_hash: str | None = None
    number_audit: str | None = None
    math_audit: str | None = None
    red_team: str | None = None
    judge: str | None = None
    scientific_verdict: str
    external_quality_assessment: Literal["not_yet_assessed"] = "not_yet_assessed"
    unresolved_issues: list[str] = Field(default_factory=list)
    new_capability_gaps: list[str] = Field(default_factory=list)
    human_intervention_count: int = Field(default=0, ge=0)
    result_hash: str = ""


__all__ = [
    "BenchmarkBlindnessAudit",
    "BenchmarkCodeFreeze",
    "BenchmarkFile",
    "BenchmarkResult",
    "BenchmarkSourceManifest",
    "BlindnessPolicy",
    "CapabilityGap",
    "HistoricalBenchmarkRun",
    "HistoricalBenchmarkState",
    "LeakageEvent",
    "LeakageLogEntry",
    "ProblemRequirement",
    "ProblemSelectionManifest",
    "ProtocolValidity",
    "RequirementStatus",
    "RevealRecord",
    "SelectionCandidate",
]
