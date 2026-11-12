import json
from pathlib import Path

import pytest

from mmos.core.project import create_project, load_project_config
from mmos.orchestration import Orchestrator
from mmos.pipeline import select_model_ids
from mmos.schemas.project import SourceRole
from mmos.schemas.proposal import (
    Complexity,
    CouncilDecision,
    ExpertRole,
    ModelProposal,
    ProposalScore,
)
from mmos.schemas.states import ProjectState


def make_proposal(proposal_id: str, model_ids: list[str]) -> ModelProposal:
    return ModelProposal(
        proposal_id=proposal_id,
        problem_id="p",
        subproblem_id="q1",
        name="model",
        model_family="linear",
        expert_role=ExpertRole.STATISTICIAN,
        description="d",
        estimated_complexity=Complexity.LOW,
        registry_model_ids=model_ids,
    )


def make_decision(selected: list[str]) -> CouncilDecision:
    return CouncilDecision(
        scores=[
            ProposalScore(
                proposal_id=proposal_id,
                problem_match=4,
                theoretical_soundness=4,
                data_fit=4,
                interpretability=4,
                feasibility=4,
                paper_value=3,
                risk=2,
                rationale="r",
            )
            for proposal_id in selected
        ],
        selected_for_prototype=selected,
    )


def test_select_model_ids_unions_and_deduplicates() -> None:
    proposals = [
        make_proposal("p1", ["sklearn.ridge", "sklearn.random_forest_regressor"]),
        make_proposal("p2", ["sklearn.ridge", "sklearn.linear_regression"]),
    ]
    decision = make_decision(["p1", "p2"])

    assert select_model_ids(decision, proposals) == [
        "sklearn.ridge",
        "sklearn.random_forest_regressor",
        "sklearn.linear_regression",
    ]


def test_advance_to_walks_each_state_and_never_moves_backwards(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = tmp_path / "data.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([source], SourceRole.DATA)

    updated = orchestrator.advance_to(ProjectState.DATA_VALIDATED, reason="test: advance")

    assert updated.state is ProjectState.DATA_VALIDATED
    history = [
        json.loads(line)
        for line in paths.state_history_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [entry["to_state"] for entry in history] == [
        "PROJECT_INGESTED",
        "PROBLEM_UNDERSTOOD",
        "DATA_VALIDATED",
    ]

    same = orchestrator.advance_to(ProjectState.PROBLEM_UNDERSTOOD, reason="test: noop")
    assert same.state is ProjectState.DATA_VALIDATED
    assert load_project_config(paths).state is ProjectState.DATA_VALIDATED


def test_advance_to_refuses_from_escape_state(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    orchestrator = Orchestrator(paths)
    orchestrator.transition(ProjectState.REQUIRES_HUMAN_REVIEW, reason="test")

    with pytest.raises(ValueError, match="cannot advance"):
        orchestrator.advance_to(ProjectState.PROJECT_INGESTED, reason="test")
