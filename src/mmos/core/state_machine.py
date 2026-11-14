from mmos.schemas.states import ESCAPE_STATES, ProjectState, TaskFamily

NEXT_STATE: dict[ProjectState, ProjectState] = {
    ProjectState.INIT: ProjectState.PROJECT_INGESTED,
    ProjectState.PROJECT_INGESTED: ProjectState.PROBLEM_UNDERSTOOD,
    ProjectState.PROBLEM_UNDERSTOOD: ProjectState.DATA_VALIDATED,
    ProjectState.DATA_VALIDATED: ProjectState.RESEARCH_COMPLETED,
    ProjectState.RESEARCH_COMPLETED: ProjectState.CANDIDATES_PROPOSED,
    ProjectState.CANDIDATES_PROPOSED: ProjectState.PROTOTYPING,
    ProjectState.PROTOTYPING: ProjectState.PROTOTYPED,
    ProjectState.PROTOTYPED: ProjectState.FULL_EXPERIMENT,
    ProjectState.FULL_EXPERIMENT: ProjectState.EXPERIMENT_COMPLETED,
    ProjectState.EXPERIMENT_COMPLETED: ProjectState.VALIDATING,
    ProjectState.VALIDATING: ProjectState.VERIFIED,
    ProjectState.VERIFIED: ProjectState.EVIDENCE_FROZEN,
    ProjectState.EVIDENCE_FROZEN: ProjectState.PAPER_DRAFTED,
    ProjectState.PAPER_DRAFTED: ProjectState.RED_TEAM_REVIEW,
    ProjectState.RED_TEAM_REVIEW: ProjectState.READY_FOR_RELEASE,
    ProjectState.READY_FOR_RELEASE: ProjectState.RELEASED,
}

LINEAR_ORDER: tuple[ProjectState, ...] = (
    ProjectState.INIT,
    ProjectState.PROJECT_INGESTED,
    ProjectState.PROBLEM_UNDERSTOOD,
    ProjectState.DATA_VALIDATED,
    ProjectState.RESEARCH_COMPLETED,
    ProjectState.CANDIDATES_PROPOSED,
    ProjectState.PROTOTYPING,
    ProjectState.PROTOTYPED,
    ProjectState.FULL_EXPERIMENT,
    ProjectState.EXPERIMENT_COMPLETED,
    ProjectState.VALIDATING,
    ProjectState.VERIFIED,
    ProjectState.EVIDENCE_FROZEN,
    ProjectState.PAPER_DRAFTED,
    ProjectState.RED_TEAM_REVIEW,
    ProjectState.READY_FOR_RELEASE,
    ProjectState.RELEASED,
)

OPTIMIZATION_NEXT_STATE: dict[ProjectState, ProjectState] = {
    ProjectState.INIT: ProjectState.PROJECT_INGESTED,
    ProjectState.PROJECT_INGESTED: ProjectState.PROBLEM_UNDERSTOOD,
    ProjectState.PROBLEM_UNDERSTOOD: ProjectState.PROTOTYPING,
    ProjectState.PROTOTYPING: ProjectState.PROTOTYPED,
    ProjectState.PROTOTYPED: ProjectState.EXPERIMENT_COMPLETED,
    ProjectState.EXPERIMENT_COMPLETED: ProjectState.VALIDATING,
    ProjectState.VALIDATING: ProjectState.VERIFIED,
    ProjectState.VERIFIED: ProjectState.EVIDENCE_FROZEN,
    ProjectState.EVIDENCE_FROZEN: ProjectState.PAPER_DRAFTED,
    ProjectState.PAPER_DRAFTED: ProjectState.RED_TEAM_REVIEW,
    ProjectState.RED_TEAM_REVIEW: ProjectState.READY_FOR_RELEASE,
    ProjectState.READY_FOR_RELEASE: ProjectState.RELEASED,
}

OPTIMIZATION_ORDER: tuple[ProjectState, ...] = (
    ProjectState.INIT,
    ProjectState.PROJECT_INGESTED,
    ProjectState.PROBLEM_UNDERSTOOD,
    ProjectState.PROTOTYPING,
    ProjectState.PROTOTYPED,
    ProjectState.EXPERIMENT_COMPLETED,
    ProjectState.VALIDATING,
    ProjectState.VERIFIED,
    ProjectState.EVIDENCE_FROZEN,
    ProjectState.PAPER_DRAFTED,
    ProjectState.RED_TEAM_REVIEW,
    ProjectState.READY_FOR_RELEASE,
    ProjectState.RELEASED,
)

TERMINAL_STATES = frozenset({ProjectState.RELEASED})


def task_transition_table(task: TaskFamily) -> dict[ProjectState, ProjectState]:
    if task in {TaskFamily.PREDICTION, TaskFamily.FORECASTING}:
        return NEXT_STATE
    if task in {TaskFamily.OPTIMIZATION, TaskFamily.SIMULATION}:
        return OPTIMIZATION_NEXT_STATE
    raise ValueError(f"task family {task.value!r} has no lifecycle yet")


def task_order(task: TaskFamily) -> tuple[ProjectState, ...]:
    if task in {TaskFamily.PREDICTION, TaskFamily.FORECASTING}:
        return LINEAR_ORDER
    if task in {TaskFamily.OPTIMIZATION, TaskFamily.SIMULATION}:
        return OPTIMIZATION_ORDER
    raise ValueError(f"task family {task.value!r} has no lifecycle yet")


class InvalidTransitionError(ValueError):
    def __init__(
        self, current: ProjectState, target: ProjectState, allowed: frozenset[ProjectState]
    ):
        self.current = current
        self.target = target
        self.allowed = allowed
        allowed_text = ", ".join(sorted(state.value for state in allowed)) or "none"
        message = f"invalid transition {current.value} -> {target.value}; allowed: {allowed_text}"
        super().__init__(message)


def allowed_targets(
    current: ProjectState,
    resume_state: ProjectState | None = None,
    task: TaskFamily = TaskFamily.PREDICTION,
) -> frozenset[ProjectState]:
    if current in TERMINAL_STATES:
        return frozenset()
    if current in ESCAPE_STATES:
        # Leaving an escape state returns to the state that was interrupted.
        return frozenset({resume_state}) if resume_state is not None else frozenset()
    targets = set(ESCAPE_STATES)
    next_state = task_transition_table(task).get(current)
    if next_state is not None:
        targets.add(next_state)
    return frozenset(targets)


def check_transition(
    current: ProjectState,
    target: ProjectState,
    resume_state: ProjectState | None = None,
    task: TaskFamily = TaskFamily.PREDICTION,
) -> None:
    allowed = allowed_targets(current, resume_state, task)
    if target not in allowed:
        raise InvalidTransitionError(current, target, allowed)
