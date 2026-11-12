import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from mmos.agents import (
    MissingAPIKeyError,
    ProblemAnalyst,
    UnsupportedProviderError,
    build_model,
)
from mmos.config import Settings

SAMPLE_OUTPUT = {
    "problem_id": "ignored-by-system",
    "title": "Demand forecasting",
    "overall_goal": "Forecast demand and plan capacity",
    "subproblems": [
        {"subproblem_id": "q1", "statement": "forecast", "goal": "predict demand"},
        {
            "subproblem_id": "q2",
            "statement": "plan",
            "goal": "plan capacity",
            "depends_on": ["q1"],
        },
    ],
}


def test_analyst_returns_validated_spec_with_system_problem_id() -> None:
    analyst = ProblemAnalyst(TestModel(custom_output_args=SAMPLE_OUTPUT))

    spec = analyst.analyze("A forecasting task", problem_id="proj-1")

    assert spec.problem_id == "proj-1"
    assert [subproblem.subproblem_id for subproblem in spec.subproblems] == ["q1", "q2"]


def test_missing_api_key_is_reported() -> None:
    with pytest.raises(MissingAPIKeyError):
        build_model(Settings(_env_file=None, llm_api_key=None))


def test_deepseek_model_is_constructed_without_network() -> None:
    model = build_model(Settings(_env_file=None, llm_api_key="test-key"))
    assert isinstance(model, OpenAIChatModel)


def test_openai_model_with_base_url_is_constructed_without_network() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_api_key="test-key",
        llm_base_url="https://example.invalid/v1",
    )
    assert isinstance(build_model(settings), OpenAIChatModel)


def test_anthropic_provider_is_reported_as_unsupported() -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", llm_api_key="test-key")
    with pytest.raises(UnsupportedProviderError):
        build_model(settings)


@pytest.mark.integration
def test_live_problem_analysis() -> None:
    settings = Settings(_env_file=".env")
    if settings.llm_api_key is None:
        pytest.skip("requires MMOS_LLM_API_KEY in the environment or .env")

    spec = ProblemAnalyst(build_model(settings)).analyze(
        "A factory produces two products. Maximize profit subject to machine hours.",
        problem_id="live-1",
    )
    assert spec.problem_id == "live-1"
    assert spec.subproblems
