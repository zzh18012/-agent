from itertools import pairwise

import pytest

from mmos.core.project import create_project
from mmos.core.state_machine import InvalidTransitionError, allowed_targets, check_transition
from mmos.orchestration import Orchestrator
from mmos.schemas.project import SourceRole
from mmos.schemas.states import ESCAPE_STATES, ProjectState, TaskFamily

LINEAR_PATH = [
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
]


def test_full_linear_path_is_allowed() -> None:
    for current, target in pairwise(LINEAR_PATH):
        check_transition(current, target)


def test_skipping_a_state_is_rejected() -> None:
    with pytest.raises(InvalidTransitionError):
        check_transition(ProjectState.INIT, ProjectState.DATA_VALIDATED)


def test_escape_states_are_reachable_from_working_states() -> None:
    for escape_state in ESCAPE_STATES:
        check_transition(ProjectState.PROTOTYPING, escape_state)


def test_escape_state_requires_resume_target() -> None:
    assert allowed_targets(ProjectState.REQUIRES_HUMAN_REVIEW) == frozenset()
    with pytest.raises(InvalidTransitionError):
        check_transition(ProjectState.REQUIRES_HUMAN_REVIEW, ProjectState.PROTOTYPING)
    check_transition(
        ProjectState.REQUIRES_HUMAN_REVIEW,
        ProjectState.PROTOTYPING,
        resume_state=ProjectState.PROTOTYPING,
    )


def test_released_is_terminal() -> None:
    assert allowed_targets(ProjectState.RELEASED) == frozenset()
    with pytest.raises(InvalidTransitionError):
        check_transition(ProjectState.RELEASED, ProjectState.BLOCKED)


def test_optimization_lifecycle_skips_prediction_states(tmp_path) -> None:
    paths, _ = create_project(tmp_path / "projects", "opt", task=TaskFamily.OPTIMIZATION)
    problem = tmp_path / "problem.md"
    problem.write_text("# Optimize\n", encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")

    allowed = allowed_targets(ProjectState.PROBLEM_UNDERSTOOD, task=TaskFamily.OPTIMIZATION)
    assert ProjectState.PROTOTYPING in allowed
    assert ProjectState.DATA_VALIDATED not in allowed
    with pytest.raises(InvalidTransitionError):
        check_transition(
            ProjectState.PROBLEM_UNDERSTOOD,
            ProjectState.DATA_VALIDATED,
            task=TaskFamily.OPTIMIZATION,
        )

    updated = orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED, reason="test optimization lifecycle"
    )
    assert updated.state is ProjectState.EXPERIMENT_COMPLETED
    history = [
        entry
        for entry in paths.state_history_file.read_text(encoding="utf-8").splitlines()
        if entry
    ]
    assert len(history) == 5
