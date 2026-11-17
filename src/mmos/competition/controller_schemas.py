from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.competition.clock import ensure_aware, zone
from mmos.schemas.common import MMOSModel
from mmos.schemas.competition import ResolutionClass


class CompetitionControllerState(StrEnum):
    CONFIGURED = "configured"
    ACTIVE_EXPLORATION = "active_exploration"
    CANDIDATE_CONVERGENCE = "candidate_convergence"
    MODEL_FROZEN = "model_frozen"
    EVIDENCE_COMPLETION = "evidence_completion"
    EVIDENCE_FROZEN = "evidence_frozen"
    PAPER_FINALIZATION = "paper_finalization"
    PAPER_FROZEN = "paper_frozen"
    READY_FOR_SUBMISSION_CHECKS = "ready_for_submission_checks"
    CLOSED = "closed"


STATE_ORDER: tuple[CompetitionControllerState, ...] = tuple(
    CompetitionControllerState
)


class ActionCategory(StrEnum):
    READ_ONLY = "read_only"
    COMPUTE = "compute"
    MODEL_MUTATION = "model_mutation"
    EVIDENCE_MUTATION = "evidence_mutation"
    PAPER_MUTATION = "paper_mutation"
    RELEASE_MUTATION = "release_mutation"
    SUBMISSION_PREPARATION = "submission_preparation"


MUTATION_ACTIONS = frozenset(
    {
        ActionCategory.COMPUTE,
        ActionCategory.MODEL_MUTATION,
        ActionCategory.EVIDENCE_MUTATION,
        ActionCategory.PAPER_MUTATION,
        ActionCategory.RELEASE_MUTATION,
        ActionCategory.SUBMISSION_PREPARATION,
    }
)


class WorkItemKind(StrEnum):
    SCIENTIFIC_CRITICAL = "scientific_critical"
    EXPERIMENT_REQUEST = "experiment_request"
    VALIDATION_REPAIR = "validation_repair"
    PAPER_REPAIR = "paper_repair"
    CLAIM_SCOPE_REDUCTION = "claim_scope_reduction"
    REVIEW_ISSUE = "review_issue"
    SUBMISSION_PREPARATION = "submission_preparation"
    OPTIONAL_IMPROVEMENT = "optional_improvement"


class WorkItemStatus(StrEnum):
    PENDING = "pending"
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class WorkSeverity(IntEnum):
    CRITICAL = 0
    MAJOR = 1
    MINOR = 2
    NONE = 3


class ClaimCriticality(IntEnum):
    CORE = 0
    SUPPORTING = 1
    OPTIONAL = 2


class GuardVerdict(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    DEFER = "defer"
    CANCEL = "cancel"


class CompetitionTimingProfile(MMOSModel):
    profile_id: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    timezone: str = Field(min_length=1)
    contest_start_at: datetime
    exploration_end_at: datetime
    model_freeze_at: datetime
    evidence_freeze_at: datetime
    paper_freeze_at: datetime
    work_stop_at: datetime
    submission_deadline_at: datetime
    source: str = Field(min_length=1)
    source_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _timing_is_valid(self) -> CompetitionTimingProfile:
        zone(self.timezone)
        fields = [
            ("contest_start_at", self.contest_start_at),
            ("exploration_end_at", self.exploration_end_at),
            ("model_freeze_at", self.model_freeze_at),
            ("evidence_freeze_at", self.evidence_freeze_at),
            ("paper_freeze_at", self.paper_freeze_at),
            ("work_stop_at", self.work_stop_at),
            ("submission_deadline_at", self.submission_deadline_at),
        ]
        for name, value in fields:
            ensure_aware(value, name)
        if not self.contest_start_at < self.exploration_end_at:
            raise ValueError("contest_start_at must precede exploration_end_at")
        if self.exploration_end_at > self.model_freeze_at:
            raise ValueError("exploration_end_at must not be after model_freeze_at")
        if self.model_freeze_at > self.evidence_freeze_at:
            raise ValueError("model_freeze_at must not be after evidence_freeze_at")
        if self.evidence_freeze_at > self.paper_freeze_at:
            raise ValueError("evidence_freeze_at must not be after paper_freeze_at")
        if self.paper_freeze_at > self.work_stop_at:
            raise ValueError("paper_freeze_at must not be after work_stop_at")
        if not self.contest_start_at < self.work_stop_at:
            raise ValueError("contest_start_at must precede work_stop_at")
        if self.work_stop_at > self.submission_deadline_at:
            raise ValueError("work_stop_at must not be after submission_deadline_at")
        return self

    @property
    def timezone_info(self) -> object:
        return zone(self.timezone)

    def local(self, value: datetime) -> datetime:
        return ensure_aware(value, "datetime").astimezone(zone(self.timezone))

    def milestones(self) -> list[tuple[str, datetime]]:
        return [
            ("exploration_end_at", self.exploration_end_at),
            ("model_freeze_at", self.model_freeze_at),
            ("evidence_freeze_at", self.evidence_freeze_at),
            ("paper_freeze_at", self.paper_freeze_at),
            ("work_stop_at", self.work_stop_at),
            ("submission_deadline_at", self.submission_deadline_at),
        ]


class CompetitionWorkItem(MMOSModel):
    work_item_id: str = Field(min_length=1)
    kind: WorkItemKind
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    blocking_release: bool
    scientific_severity: WorkSeverity
    claim_criticality: ClaimCriticality
    capability_status: ResolutionClass | None = None
    expected_runtime_class: Literal["seconds", "minutes", "hours"] | None = None
    dependencies: list[str] = Field(default_factory=list)
    created_at: datetime
    action_category: ActionCategory
    status: WorkItemStatus = WorkItemStatus.PENDING
    blocked_reason: str | None = None
    p0_repair: bool = False

    @model_validator(mode="after")
    def _time_is_aware(self) -> CompetitionWorkItem:
        ensure_aware(self.created_at, "created_at")
        return self


class DeadlineGuardDecision(MMOSModel):
    decision: GuardVerdict
    reason_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    competition_state: CompetitionControllerState
    now: datetime
    relevant_deadline: datetime | None = None
    work_item_id: str | None = None
    action_category: ActionCategory


class WorkQueueEntry(MMOSModel):
    work_item_id: str
    status: WorkItemStatus
    decision: DeadlineGuardDecision
    priority_key: list[str] = Field(default_factory=list)


class WorkQueue(MMOSModel):
    competition_run_id: str
    state: CompetitionControllerState
    now: datetime
    next_milestone: str | None
    remaining_seconds: float
    ordered_work_items: list[WorkQueueEntry] = Field(default_factory=list)
    allowed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    deferred: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    queue_hash: str = ""


class ModelFreezeRecord(MMOSModel):
    freeze_id: str
    revision: int = Field(default=1, ge=1)
    competition_run_id: str
    frozen_at: datetime
    primary_model_refs: list[str] = Field(default_factory=list)
    challenger_refs: list[str] = Field(default_factory=list)
    fallback_refs: list[str] = Field(default_factory=list)
    source_project_id: str
    experiment_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    artifact_hashes: list[str] = Field(default_factory=list)
    selection_evidence_ids: list[str] = Field(default_factory=list)
    freeze_hash: str = ""


class EvidenceFreezeRecord(MMOSModel):
    freeze_id: str
    revision: int = Field(default=1, ge=1)
    competition_run_id: str
    frozen_at: datetime
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_catalog_hash: str
    source_release_hash: str
    augmentation_hashes: list[str] = Field(default_factory=list)
    validation_summary: str = Field(min_length=1)
    unresolved_issues: list[str] = Field(default_factory=list)
    freeze_hash: str = ""


class PaperFreezeRecord(MMOSModel):
    freeze_id: str
    revision: int = Field(default=1, ge=1)
    competition_run_id: str
    paper_run_id: str
    report_path: str
    report_hash: str
    evidence_freeze_id: str
    evidence_freeze_hash: str
    number_audit_status: str
    math_audit_status: str
    red_team_status: str
    judge_status: str
    frozen_at: datetime
    freeze_hash: str = ""


class CompetitionCheckpoint(MMOSModel):
    checkpoint_id: str
    competition_run_id: str
    controller_state: CompetitionControllerState
    clock_snapshot: datetime
    queue_hash: str
    model_freeze_hash: str | None = None
    evidence_freeze_hash: str | None = None
    paper_freeze_hash: str | None = None
    unresolved_blockers: list[str] = Field(default_factory=list)
    checkpoint_hash: str = ""


class ScopeReductionDecision(MMOSModel):
    decision_id: str
    competition_run_id: str
    created_at: datetime
    action: str = Field(min_length=1)
    target_work_item_ids: list[str] = Field(default_factory=list)
    target_claims: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    status: Literal["applied", "rejected"] = "applied"


class HumanDecisionAction(StrEnum):
    CANCEL_WORK_ITEM = "cancel_work_item"
    USE_FALLBACK = "use_fallback"
    REDUCE_CLAIM = "reduce_claim"
    DEFER_NEW_CAPABILITY = "defer_new_capability"


class HumanDecisionRecord(MMOSModel):
    decision_id: str
    competition_run_id: str
    created_at: datetime
    action: HumanDecisionAction
    target_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    status: Literal["applied", "rejected"] = "applied"


class CompetitionBudgetLedger(MMOSModel):
    competition_run_id: str
    phase: CompetitionControllerState
    planned_work_items: int = Field(ge=0)
    completed_work_items: int = Field(ge=0)
    blocked_work_items: int = Field(ge=0)
    elapsed_seconds: float
    remaining_seconds: float
    next_milestone: str | None
    time_to_next_milestone_seconds: float | None


class CompetitionFixtureRecord(MMOSModel):
    source_paper_run_id: str
    judge_status: str
    number_audit_ok: bool
    math_audit_ok: bool
    items: list[CompetitionWorkItem] = Field(default_factory=list)


class CompetitionControllerRun(MMOSModel):
    competition_run_id: str
    profile_id: str
    profile_hash: str
    source_project_id: str
    source_paper_run_id: str
    source_release: str
    source_release_hash: str
    source_verified_numbers_hash: str
    state: CompetitionControllerState
    state_history: list[CompetitionControllerState] = Field(default_factory=list)
    work_stopped: bool = False
    scientific_gate_failed: bool = False
    deadline_expired: bool = False
    requires_human_review: bool = False
    model_freeze_id: str | None = None
    model_freeze_hash: str | None = None
    evidence_freeze_id: str | None = None
    evidence_freeze_hash: str | None = None
    paper_freeze_id: str | None = None
    paper_freeze_hash: str | None = None
    unresolved_blockers: list[str] = Field(default_factory=list)
    queue_hash: str | None = None
    created_at: datetime
    updated_at: datetime


class TimelineEvent(MMOSModel):
    event_type: str = Field(min_length=1)
    occurred_at: datetime
    state: CompetitionControllerState
    payload: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "MUTATION_ACTIONS",
    "STATE_ORDER",
    "ActionCategory",
    "ClaimCriticality",
    "CompetitionBudgetLedger",
    "CompetitionCheckpoint",
    "CompetitionControllerRun",
    "CompetitionControllerState",
    "CompetitionFixtureRecord",
    "CompetitionTimingProfile",
    "CompetitionWorkItem",
    "DeadlineGuardDecision",
    "EvidenceFreezeRecord",
    "GuardVerdict",
    "HumanDecisionAction",
    "HumanDecisionRecord",
    "ModelFreezeRecord",
    "PaperFreezeRecord",
    "ScopeReductionDecision",
    "TimelineEvent",
    "WorkItemKind",
    "WorkItemStatus",
    "WorkQueue",
    "WorkQueueEntry",
    "WorkSeverity",
]
