from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class SectionKind(StrEnum):
    PROBLEM_RESTATEMENT = "problem_restatement"
    ASSUMPTIONS = "assumptions"
    NOTATION = "notation"
    MODEL = "model"
    SOLUTION = "solution"
    RESULTS = "results"
    SENSITIVITY = "sensitivity"
    EVALUATION = "evaluation"
    LIMITATIONS = "limitations"


SECTION_ORDER = tuple(SectionKind)


class TableTemplate(StrEnum):
    METRIC_COMPARISON = "metric_comparison"
    EXPERIMENT_INVENTORY = "experiment_inventory"
    EVIDENCE_LINEAGE = "evidence_lineage"


class FigureTemplate(StrEnum):
    PIPELINE_SCHEMATIC = "pipeline_schematic"
    METRIC_BAR = "metric_bar"
    UNCERTAINTY = "uncertainty"


class Capability(StrEnum):
    PREDICTION = "prediction"
    FORECASTING = "forecasting"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    GRAPH = "graph"
    MULTI_OBJECTIVE = "multi_objective"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"


class MathSeverity(StrEnum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class FindingKind(StrEnum):
    SUPPORTED = "supported"
    TENTATIVE = "tentative"
    UNSUPPORTED = "unsupported"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimPoint(MMOSModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class NotationEntry(MMOSModel):
    symbol: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    unit: str | None = None


class SectionPlan(MMOSModel):
    heading: str = Field(min_length=1)
    kind: SectionKind
    purpose: str = Field(min_length=1)
    blocks: list[str] = Field(min_length=1)
    claim_points: list[ClaimPoint] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)


class TablePlan(MMOSModel):
    table_id: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    template: TableTemplate
    evidence_ids: list[str] = Field(min_length=1)


class FigurePlan(MMOSModel):
    figure_id: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    template: FigureTemplate
    evidence_ids: list[str] = Field(min_length=1)
    metric_prefix: str | None = None
    experiment_id: str | None = None

    @model_validator(mode="after")
    def _template_fields(self) -> "FigurePlan":
        if self.template is FigureTemplate.METRIC_BAR and not self.metric_prefix:
            raise ValueError("metric_bar figures require a metric_prefix")
        return self


class CapabilityNeed(MMOSModel):
    capability: Capability
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    proposed_experiment: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)
    priority: Priority = Priority.MEDIUM
    required: bool = False
    supporting_evidence_ids: list[str] = Field(min_length=1)


class CompetitionPaperPlan(MMOSModel):
    title: str = Field(min_length=1)
    abstract_points: list[ClaimPoint] = Field(min_length=1)
    sections: list[SectionPlan] = Field(min_length=1)
    tables: list[TablePlan] = Field(default_factory=list)
    figures: list[FigurePlan] = Field(default_factory=list)
    assumptions: list[str] = Field(min_length=1)
    notation: list[NotationEntry] = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    capability_needs: list[CapabilityNeed] = Field(default_factory=list)

    @model_validator(mode="after")
    def _section_shape(self) -> "CompetitionPaperPlan":
        kinds = [section.kind for section in self.sections]
        if len(kinds) != len(set(kinds)):
            raise ValueError("section kinds must be unique")
        if set(kinds) != set(SECTION_ORDER):
            missing = sorted(kind.value for kind in set(SECTION_ORDER) - set(kinds))
            extra = sorted(kind.value for kind in set(kinds) - set(SECTION_ORDER))
            raise ValueError(
                f"section kinds must cover all kinds; missing={missing}, extra={extra}"
            )
        order = [SECTION_ORDER.index(kind) for kind in kinds]
        if order != sorted(order):
            raise ValueError("sections must follow the canonical order")
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError("table ids must be unique")
        figure_ids = [figure.figure_id for figure in self.figures]
        if len(figure_ids) != len(set(figure_ids)):
            raise ValueError("figure ids must be unique")
        return self


class MathAuditIssue(MMOSModel):
    issue_id: str = Field(min_length=1)
    severity: MathSeverity
    category: str = Field(min_length=1)
    target: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    required_fix: str = Field(min_length=1)


class MathAuditReport(MMOSModel):
    issues: list[MathAuditIssue] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class ExperimentFinding(MMOSModel):
    finding_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    kind: FindingKind


class ExperimentAnalysisReport(MMOSModel):
    findings: list[ExperimentFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requested_checks: list[CapabilityNeed] = Field(default_factory=list)


class FigurePlanBatch(MMOSModel):
    figures: list[FigurePlan] = Field(default_factory=list, max_length=2)


class CompetitionExperimentRequest(MMOSModel):
    request_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    capability: Capability
    proposed_experiment: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)
    priority: Priority
    supporting_evidence_ids: list[str] = Field(min_length=1)
    status: Literal["proposed"] = "proposed"
    executed: Literal[False] = False
    supported: bool
    reason: str | None = None


class CompetitionExperimentManifest(MMOSModel):
    requests: list[CompetitionExperimentRequest] = Field(default_factory=list)
    supported_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    created_at: datetime

    @model_validator(mode="after")
    def _counts(self) -> "CompetitionExperimentManifest":
        supported = sum(1 for request in self.requests if request.supported)
        if supported != self.supported_count:
            raise ValueError("supported_count does not match requests")
        if len(self.requests) - supported != self.unsupported_count:
            raise ValueError("unsupported_count does not match requests")
        return self


class NumberAuditFinding(MMOSModel):
    token: str
    location: str
    reason: str


class NumberAuditReport(MMOSModel):
    checked_fragments: int = Field(ge=0)
    checked_tokens: int = Field(ge=0)
    unverified: list[NumberAuditFinding] = Field(default_factory=list)
    allowed_metadata_numbers: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return bool(self.unverified)


class ResolutionClass(StrEnum):
    EXECUTABLE_EXISTING_CAPABILITY = "executable_existing_capability"
    REQUIRES_NEW_CAPABILITY = "requires_new_capability"
    REQUIRES_HUMAN_DECISION = "requires_human_decision"
    NOT_JUSTIFIED = "not_justified"
    INVALID = "invalid"


class ExecutionStatus(StrEnum):
    PROPOSED = "proposed"
    RESOLVED = "resolved"
    EXECUTABLE = "executable"
    BLOCKED = "blocked"
    EXECUTING = "executing"
    EXECUTED = "executed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EVIDENCE_REGISTERED = "evidence_registered"
    FAILED = "failed"


class RequestResolution(MMOSModel):
    request_id: str
    classification: ResolutionClass
    mapped_kind: str | None = None
    rationale: str = Field(min_length=1)
    blocker_reason: str | None = None
    executable: bool = False


class RequestExecution(MMOSModel):
    request_id: str
    status: ExecutionStatus
    experiment_ids: list[str] = Field(default_factory=list)
    validation_status: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    executed: bool = False
    validated: bool = False
    evidence_registered: bool = False


class CompetitionAugmentationRun(MMOSModel):
    augmentation_id: str
    source_project_id: str
    source_release: str
    source_release_hash_before: str
    source_release_hash_after: str
    source_verified_numbers_hash_before: str
    source_verified_numbers_hash_after: str
    source_context_hash: str
    source_competition_run_id: str
    request_manifest_hash: str
    resolutions: list[RequestResolution]
    executions: list[RequestExecution]
    new_evidence_ids: list[str] = Field(default_factory=list)
    evidence_store_path: str
    evidence_store_hash: str
    new_competition_run_id: str | None = None
    new_paper_path: str | None = None
    new_release_path: str | None = None
    new_paper_error: str | None = None
    status: Literal["completed", "partial", "blocked"]
    created_at: datetime


class CompetitionPaperRun(MMOSModel):
    run_id: str
    source_release: str
    source_release_hash: str
    context_hash: str
    plan_hash: str
    math_audit_hash: str
    experiment_analysis_hash: str
    number_audit_hash: str
    request_manifest_hash: str
    status: Literal["released", "blocked"]
    red_team_overall: str | None = None
    paper_path: str
    review_path: str | None = None
    release_path: str | None = None
    created_at: datetime


__all__ = [
    "SECTION_ORDER",
    "Capability",
    "CapabilityNeed",
    "ClaimPoint",
    "CompetitionAugmentationRun",
    "CompetitionExperimentManifest",
    "CompetitionExperimentRequest",
    "CompetitionPaperPlan",
    "CompetitionPaperRun",
    "ExecutionStatus",
    "ExperimentAnalysisReport",
    "ExperimentFinding",
    "FigurePlan",
    "FigurePlanBatch",
    "FigureTemplate",
    "FindingKind",
    "MathAuditIssue",
    "MathAuditReport",
    "MathSeverity",
    "NotationEntry",
    "NumberAuditFinding",
    "NumberAuditReport",
    "Priority",
    "RequestExecution",
    "RequestResolution",
    "ResolutionClass",
    "SectionKind",
    "SectionPlan",
    "TablePlan",
    "TableTemplate",
]
