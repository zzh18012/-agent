from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from mmos.agents import CouncilError, ResearchAgent, normalize_decision, normalize_proposals
from mmos.cli.main import app
from mmos.core.intake import ingest_files
from mmos.core.project import create_project, save_proposals
from mmos.experiments.planner import ExperimentPlanner
from mmos.orchestration import Orchestrator
from mmos.registries import ModelRegistry
from mmos.schemas.problem import ProblemSpec, SubProblem, Variable
from mmos.schemas.project import SourceRole
from mmos.schemas.proposal import (
    Complexity,
    CouncilDecision,
    ExpertRole,
    ModelProposal,
    ProposalBatch,
    ProposalScore,
)
from mmos.schemas.states import ProjectState
from tests.helpers import regression_frame

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"
runner = CliRunner()


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load(REGISTRY_PATH)


@pytest.fixture
def spec() -> ProblemSpec:
    return ProblemSpec(
        problem_id="proj-1",
        title="Demand prediction",
        overall_goal="Predict demand",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="Predict demand from two features",
                goal="minimize prediction error",
                target_variables=[Variable(name="demand")],
            )
        ],
    )


def make_proposal(registry_model_ids: list[str], subproblem_id: str = "q1") -> ModelProposal:
    return ModelProposal(
        proposal_id="llm-invented",
        problem_id="llm-invented",
        subproblem_id=subproblem_id,
        name="Ridge regression",
        model_family="linear",
        expert_role=ExpertRole.ML_SCIENTIST,
        description="regularized linear model",
        estimated_complexity=Complexity.LOW,
        registry_model_ids=registry_model_ids,
    )


def make_score(proposal_id: str) -> ProposalScore:
    return ProposalScore(
        proposal_id=proposal_id,
        problem_match=4,
        theoretical_soundness=4,
        data_fit=4,
        interpretability=4,
        feasibility=4,
        paper_value=3,
        risk=2,
        rationale="test",
    )


def test_research_agent_returns_system_problem_id() -> None:
    sample = {
        "problem_id": "ignored",
        "methods": [
            {
                "method_id": "m1",
                "name": "Ridge regression",
                "problem_types": ["regression"],
                "core_idea": "regularized linear fit",
            }
        ],
    }
    result = ResearchAgent(TestModel(custom_output_args=sample)).research(
        ProblemSpec(
            problem_id="proj-1",
            title="t",
            overall_goal="g",
            subproblems=[SubProblem(subproblem_id="q1", statement="s", goal="g")],
        ),
        registry=ModelRegistry.load(REGISTRY_PATH),
    )
    assert result.problem_id == "proj-1"
    assert result.methods[0].method_id == "m1"


def test_normalize_proposals_assigns_ids_and_drops_invalid(
    registry: ModelRegistry, spec: ProblemSpec
) -> None:
    batch = ProposalBatch(
        proposals=[
            make_proposal(["sklearn.ridge", "not.registered"]),
            make_proposal(["not.registered"]),
            make_proposal(["sklearn.ridge"], subproblem_id="q99"),
        ]
    )

    proposals, notes = normalize_proposals(batch, ExpertRole.STATISTICIAN, spec, registry)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.proposal_id == "prop-statistician-1"
    assert proposal.problem_id == "proj-1"
    assert proposal.registry_model_ids == ["sklearn.ridge"]
    assert len(notes) == 2


def test_normalize_decision_filters_unknown_and_falls_back(
    registry: ModelRegistry,
) -> None:
    proposals = [
        make_proposal(["sklearn.ridge"]).model_copy(update={"proposal_id": "prop-a"}),
        make_proposal(["sklearn.linear_regression"]).model_copy(update={"proposal_id": "prop-b"}),
    ]
    decision = CouncilDecision(
        scores=[make_score("ghost"), make_score("prop-a")],
        selected_for_prototype=["ghost"],
    )

    normalized, notes = normalize_decision(decision, proposals)

    assert normalized.selected_for_prototype
    assert normalized.selected_for_prototype[0] in {"prop-a", "prop-b"}
    assert any("fell back" in note for note in notes)


def test_normalize_decision_without_real_scores_raises() -> None:
    decision = CouncilDecision(
        scores=[make_score("ghost")],
        selected_for_prototype=["ghost"],
    )
    with pytest.raises(CouncilError):
        normalize_decision(decision, [])


def test_planner_can_include_registered_baseline(registry: ModelRegistry) -> None:
    proposal = make_proposal(["sklearn.ridge"])
    spec = ExperimentPlanner(registry).plan(
        proposal,
        dataset_hash="a" * 64,
        target_column="y",
        feature_set=["x1"],
        include_baseline=True,
    )
    assert spec.models[0] == "sklearn.dummy_regressor"
    assert "sklearn.ridge" in spec.models


def test_plan_experiment_cli_creates_spec_and_advances(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    paths, _ = create_project(projects_dir, "demo")
    source = tmp_path / "data.csv"
    regression_frame().write_csv(source)
    ingest_files(paths, [source], SourceRole.DATA)
    proposal = make_proposal(["sklearn.ridge"]).model_copy(update={"proposal_id": "prop-a"})
    save_proposals(paths, [proposal])

    orchestrator = Orchestrator(paths)
    orchestrator.transition(ProjectState.PROJECT_INGESTED, reason="test setup")
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test setup")
    orchestrator.transition(ProjectState.DATA_VALIDATED, reason="test setup")
    orchestrator.transition(ProjectState.RESEARCH_COMPLETED, reason="test setup")
    orchestrator.transition(ProjectState.CANDIDATES_PROPOSED, reason="test setup")

    result = runner.invoke(
        app,
        [
            "plan-experiment",
            paths.root.name,
            "--proposal-id",
            "prop-a",
            "--target",
            "y",
            "--feature",
            "x1",
            "--feature",
            "x2",
            "--splits",
            "3",
            "--projects-dir",
            str(projects_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "PROTOTYPING" in result.output
    assert "sklearn.dummy_regressor" in result.output

    experiment_ids = [path.name for path in paths.experiments_dir.iterdir() if path.is_dir()]
    assert len(experiment_ids) == 1
