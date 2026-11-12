import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mmos.cli.main import app
from mmos.core.project import create_project, save_tournament_result, sha256_file
from mmos.evidence import add_claim, freeze_experiment, load_evidence_index
from mmos.experiments.runner import run_experiment
from mmos.experiments.tournament import Tournament
from mmos.paper import PaperError, build_paper, render_markdown, render_paper
from mmos.registries import ModelRegistry
from mmos.validation import validate_experiment
from tests.helpers import make_experiment_project, regression_frame

runner = CliRunner()
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"


def make_frozen_project(tmp_path: Path):
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae"],
    )
    result = run_experiment(paths, spec.experiment_id)
    validate_experiment(paths, spec.experiment_id)
    freeze_experiment(paths, spec.experiment_id)

    tournament = Tournament(ModelRegistry.load(REGISTRY_PATH)).run(
        [result], primary_metric="rmse", higher_is_better=False
    )
    save_tournament_result(paths, tournament)

    evidence_id = f"ev-{spec.experiment_id}-sklearn.linear_regression-rmse"
    supported = add_claim(
        paths,
        statement="Linear regression beats the dummy baseline on rmse",
        evidence_ids=[evidence_id],
        accept=True,
        reviewer="test",
    )
    add_claim(
        paths,
        statement="This claim has not been accepted",
        evidence_ids=[evidence_id],
        claim_id="claim-proposed",
    )
    return paths, spec, tournament, supported, evidence_id


def test_paper_is_built_from_verified_evidence(tmp_path: Path) -> None:
    paths, spec, tournament, supported, evidence_id = make_frozen_project(tmp_path)
    index = load_evidence_index(paths)
    record = index.records[evidence_id]

    ast = build_paper(
        paths,
        spec.experiment_id,
        tournament_id=tournament.tournament_id,
        primary_metric="rmse",
        higher_is_better=False,
    )

    assert ast.title.endswith("Technical Report")
    assert len(ast.figures) == 1
    figure = ast.figures[0]
    for relative in (figure.path, figure.pdf_path, figure.svg_path):
        assert relative is not None
        assert (paths.root / relative).exists()
    figure_spec_path = paths.verified_dir / "figures" / f"{figure.figure_id}.json"
    figure_spec = json.loads(figure_spec_path.read_text(encoding="utf-8"))
    assert figure_spec["hashes"]["png"] == sha256_file(paths.root / figure.path)
    assert evidence_id in figure.evidence_ids

    table = ast.tables[0]
    model_row = next(row for row in table.rows if row[0] == "sklearn.linear_regression")
    assert record.display_value in model_row

    assert [claim.claim_id for claim in ast.claims] == [supported.claim_id]

    text = render_markdown(ast)
    assert "sklearn.linear_regression" in text
    assert supported.statement in text
    assert "![" in text
    assert record.display_value in text

    report = render_paper(paths, ast)
    content = report.read_text(encoding="utf-8")
    assert "../verified/figures" in content


def test_paper_requires_verified_evidence(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "empty")
    with pytest.raises(PaperError, match="no verified evidence"):
        build_paper(paths, "exp-missing")


def test_paper_cli_renders_report(tmp_path: Path) -> None:
    paths, spec, tournament, _, _ = make_frozen_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "paper",
            paths.root.name,
            spec.experiment_id,
            "--tournament",
            tournament.tournament_id,
            "--metric",
            "rmse",
            "--projects-dir",
            str(paths.root.parent),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Report:" in result.output
    assert (paths.root / "paper" / "report.md").exists()
