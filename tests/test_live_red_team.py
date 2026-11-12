from pathlib import Path

import pytest

from mmos.agents import RedTeam, build_model
from mmos.config import Settings
from mmos.evidence import freeze_experiment
from mmos.experiments.runner import run_experiment
from mmos.paper import build_paper
from mmos.validation import validate_experiment
from tests.helpers import make_experiment_project, regression_frame


@pytest.mark.integration
def test_live_red_team_review(tmp_path: Path) -> None:
    settings = Settings(_env_file=".env")
    if settings.llm_api_key is None:
        pytest.skip("requires MMOS_LLM_API_KEY in the environment or .env")

    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae"],
    )
    run_experiment(paths, spec.experiment_id)
    validate_experiment(paths, spec.experiment_id)
    freeze_experiment(paths, spec.experiment_id)
    paper = build_paper(paths, spec.experiment_id, primary_metric="rmse")

    review = RedTeam(build_model(settings)).run(paper, experiment_id=spec.experiment_id)

    attack_ids = {attack.attack_id for attack in review.attacks.attacks}
    assert {response.attack_id for response in review.responses.responses} == attack_ids
    assert {verdict.attack_id for verdict in review.judge.verdicts} == attack_ids
    assert review.experiment_id == spec.experiment_id
