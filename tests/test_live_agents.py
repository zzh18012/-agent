from pathlib import Path

import pytest

from mmos.agents import ModelCouncil, ResearchAgent, build_model
from mmos.config import Settings
from mmos.registries import ModelRegistry
from mmos.schemas.problem import ProblemSpec, SubProblem, Variable

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"


@pytest.mark.integration
def test_live_research_and_model_council() -> None:
    settings = Settings(_env_file=".env")
    if settings.llm_api_key is None:
        pytest.skip("requires MMOS_LLM_API_KEY in the environment or .env")

    model = build_model(settings)
    registry = ModelRegistry.load(REGISTRY_PATH)
    spec = ProblemSpec(
        problem_id="live-demand",
        title="Demand prediction",
        overall_goal="Predict product demand and minimize error",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="Predict demand y from two observed features x1 and x2.",
                goal="Minimize out-of-sample prediction error.",
                inputs=["x1", "x2"],
                outputs=["predicted demand"],
                target_variables=[Variable(name="y", meaning="demand")],
                evaluation_criteria=["RMSE on held-out folds"],
                required_deliverables=["predictions", "RMSE table"],
            )
        ],
    )

    research = ResearchAgent(model).research(spec, registry=registry)
    assert research.problem_id == "live-demand"
    assert research.methods

    outcome = ModelCouncil(model).propose(spec, research, registry=registry)
    assert outcome.proposals
    assert outcome.decision.selected_for_prototype
    assert all(
        registry.has(model_id)
        for proposal in outcome.proposals
        for model_id in proposal.registry_model_ids
    )
