from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.states import TaskFamily


class RoutingStatus(StrEnum):
    SINGLE_TASK = "SINGLE_TASK"
    MULTI_TASK = "MULTI_TASK"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


class ExecutionStatus(StrEnum):
    READY_FOR_SINGLE_TASK = "READY_FOR_SINGLE_TASK"
    BLOCKED_PENDING_MIXED_DAG = "BLOCKED_PENDING_MIXED_DAG"
    BLOCKED_NEEDS_CLARIFICATION = "BLOCKED_NEEDS_CLARIFICATION"
    BLOCKED_UNSUPPORTED = "BLOCKED_UNSUPPORTED"


class CapabilityLevel(StrEnum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class ProblemSignals(MMOSModel):
    has_supervised_target: bool = False
    target_description: str | None = None
    requires_future_prediction: bool = False
    temporal_order_matters: bool = False

    has_decision_variables: bool = False
    has_hard_constraints: bool = False
    objective_count: int = Field(default=0, ge=0, le=99)
    objectives_simultaneously_optimized: bool = False
    objective_preference_given: bool = False

    stochastic_process_evolution: bool = False
    requires_replications_or_distributional_output: bool = False

    graph_structure_is_primary: bool = False
    graph_native_problem: bool = False

    reported_evaluation_metrics: list[str] = Field(default_factory=list)
    insufficient_information: bool = False
    missing_information: list[str] = Field(default_factory=list)
    unsupported_requirements: list[str] = Field(default_factory=list)
    notes: str = ""


class SubproblemSignals(MMOSModel):
    subproblem_id: str
    signals: ProblemSignals


class ProblemSignalExtraction(MMOSModel):
    subproblems: list[SubproblemSignals] = Field(min_length=1)
    overall_notes: str = ""


class CapabilityReport(MMOSModel):
    task_family: TaskFamily
    current_supported_scope: str
    execution_readiness: CapabilityLevel
    limitations: list[str] = Field(default_factory=list)


class DependencyHint(MMOSModel):
    from_task: TaskFamily
    to_task: TaskFamily


class SubproblemRoute(MMOSModel):
    subproblem_id: str
    task_families: list[TaskFamily]
    reason_codes: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)


class RoutingDecision(MMOSModel):
    routing_id: str
    problem_hash: str
    problem_spec_hash: str
    status: RoutingStatus
    primary_task: TaskFamily | None = None
    required_task_families: list[TaskFamily] = Field(default_factory=list)
    subproblem_routes: list[SubproblemRoute] = Field(default_factory=list)
    dependency_hints: list[DependencyHint] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    clarifications_needed: list[str] = Field(default_factory=list)
    unsupported_requirements: list[str] = Field(default_factory=list)
    capabilities: list[CapabilityReport] = Field(default_factory=list)
    execution_status: ExecutionStatus
    suggested_command: str | None = None
    provider: str
    model: str
    git_commit: str = "unknown"
    created_at: datetime

    @model_validator(mode="after")
    def _status_invariants(self) -> "RoutingDecision":
        if self.status is RoutingStatus.SINGLE_TASK:
            if self.primary_task is None:
                raise ValueError("SINGLE_TASK requires a primary_task")
            if self.required_task_families != [self.primary_task]:
                raise ValueError("SINGLE_TASK requires exactly the primary task")
        else:
            if self.primary_task is not None:
                raise ValueError("primary_task is only valid for SINGLE_TASK")
        if self.status is RoutingStatus.MULTI_TASK and len(
            set(self.required_task_families)
        ) < 2:
            raise ValueError("MULTI_TASK requires at least two task families")
        if self.status is RoutingStatus.AMBIGUOUS and not self.clarifications_needed:
            raise ValueError("AMBIGUOUS requires clarifications_needed")
        if self.status is RoutingStatus.UNSUPPORTED and not self.unsupported_requirements:
            raise ValueError("UNSUPPORTED requires unsupported_requirements")
        families = {capability.task_family for capability in self.capabilities}
        if not set(self.required_task_families) <= families:
            raise ValueError("capabilities do not cover required task families")
        for hint in self.dependency_hints:
            if hint.from_task not in self.required_task_families or (
                hint.to_task not in self.required_task_families
            ):
                raise ValueError("dependency hint references an unrequired task family")
        return self


class RoutingIssue(MMOSModel):
    code: str
    message: str


class RoutingAuditResult(MMOSModel):
    routing_id: str
    passed: bool
    critical_unsafe: bool
    issues: list[RoutingIssue] = Field(default_factory=list)


class RoutingBenchmarkCase(MMOSModel):
    case_id: str
    problem: str
    problem_hash: str
    expected_status: RoutingStatus
    expected_tasks: list[TaskFamily] = Field(default_factory=list)
    expected_dependency_hints: list[str] = Field(default_factory=list)


class RoutingBenchmarkManifest(MMOSModel):
    suite_id: str
    version: str
    cases: list[RoutingBenchmarkCase] = Field(min_length=1)
    manifest_hash: str


class RoutingBenchmarkCaseResult(MMOSModel):
    case_id: str
    run: int
    expected_status: RoutingStatus
    expected_tasks: list[TaskFamily]
    expected_dependency_hints: list[str] = Field(default_factory=list)
    predicted_status: RoutingStatus
    predicted_primary: TaskFamily | None = None
    predicted_tasks: list[TaskFamily] = Field(default_factory=list)
    predicted_dependency_hints: list[str] = Field(default_factory=list)
    status_correct: bool
    single_task_exact: bool
    critical_unsafe: bool
    audit_passed: bool
    predicted_route: str


class RoutingBenchmarkRun(MMOSModel):
    run: int
    cases: list[RoutingBenchmarkCaseResult]
    status_accuracy: float
    single_task_accuracy: float
    multi_task_recall: float
    ambiguous_detection: float
    unsupported_detection: float
    critical_unsafe_count: int


class RoutingBenchmarkReport(MMOSModel):
    suite_id: str
    manifest_hash: str
    run_1: RoutingBenchmarkRun
    run_2: RoutingBenchmarkRun
    stability: float
    confusion_matrix: dict[str, dict[str, int]]
    misroutes: list[str] = Field(default_factory=list)
    critical_unsafe_count: int
    limitations: list[str] = Field(default_factory=list)
    created_at: datetime
