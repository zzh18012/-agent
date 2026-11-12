import json
from pathlib import Path

import pytest

from mmos.core.project import create_project, load_project_config
from mmos.core.state_machine import InvalidTransitionError
from mmos.orchestration import Orchestrator
from mmos.schemas.project import SourceRole
from mmos.schemas.states import ProjectState


def make_project(tmp_path: Path):
    return create_project(tmp_path / "projects", "demo")


def history_lines(paths) -> list[dict]:
    if not paths.state_history_file.exists():
        return []
    return [
        json.loads(line)
        for line in paths.state_history_file.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_ingest_transitions_to_project_ingested(tmp_path: Path) -> None:
    paths, _ = make_project(tmp_path)
    source = tmp_path / "train.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")

    records, state = Orchestrator(paths).ingest([source], SourceRole.DATA)

    assert len(records) == 1
    assert state is ProjectState.PROJECT_INGESTED
    assert load_project_config(paths).state is ProjectState.PROJECT_INGESTED
    entries = history_lines(paths)
    assert len(entries) == 1
    assert entries[0]["from_state"] == "INIT"
    assert entries[0]["to_state"] == "PROJECT_INGESTED"


def test_invalid_transition_leaves_state_untouched(tmp_path: Path) -> None:
    paths, _ = make_project(tmp_path)

    with pytest.raises(InvalidTransitionError):
        Orchestrator(paths).transition(ProjectState.DATA_VALIDATED, reason="too early")

    assert load_project_config(paths).state is ProjectState.INIT
    assert history_lines(paths) == []


def test_escape_state_records_resume_target(tmp_path: Path) -> None:
    paths, _ = make_project(tmp_path)
    orchestrator = Orchestrator(paths)

    blocked = orchestrator.transition(ProjectState.REQUIRES_HUMAN_REVIEW, reason="ambiguity")
    assert blocked.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert blocked.resume_state is ProjectState.INIT

    resumed = orchestrator.transition(ProjectState.INIT, reason="approved")
    assert resumed.state is ProjectState.INIT
    assert resumed.resume_state is None
    assert len(history_lines(paths)) == 2
