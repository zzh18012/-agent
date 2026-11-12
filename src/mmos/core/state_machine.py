from mmos.schemas.states import ESCAPE_STATES, ProjectState

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

TERMINAL_STATES = frozenset({ProjectState.RELEASED})


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
    current: ProjectState, resume_state: ProjectState | None = None
) -> frozenset[ProjectState]:
    if current in TERMINAL_STATES:
        return frozenset()
    if current in ESCAPE_STATES:
        # Leaving an escape state returns to the state that was interrupted.
        return frozenset({resume_state}) if resume_state is not None else frozenset()
    targets = set(ESCAPE_STATES)
    next_state = NEXT_STATE.get(current)
    if next_state is not None:
        targets.add(next_state)
    return frozenset(targets)


def check_transition(
    current: ProjectState,
    target: ProjectState,
    resume_state: ProjectState | None = None,
) -> None:
    allowed = allowed_targets(current, resume_state)
    if target not in allowed:
        raise InvalidTransitionError(current, target, allowed)
