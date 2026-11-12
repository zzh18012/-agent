from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from mmos import __version__
from mmos.cli.main import app
from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import (
    ProjectPaths,
    load_manifest,
    load_problem_spec,
    load_project_config,
    save_experiment_spec,
)
from mmos.orchestration import Orchestrator
from mmos.schemas.experiment import ExperimentSpec
from mmos.schemas.states import ProjectState

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_reports_python() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.stdout


def test_analyze_problem_without_problem_files_fails(tmp_path) -> None:
    init_result = runner.invoke(app, ["init", "Empty", "--projects-dir", str(tmp_path)])
    assert init_result.exit_code == 0
    project_id = next(path for path in tmp_path.iterdir() if path.is_dir()).name

    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n", encoding="utf-8")
    ingest_result = runner.invoke(
        app,
        ["ingest", project_id, str(data), "--role", "data", "--projects-dir", str(tmp_path)],
    )
    assert ingest_result.exit_code == 0

    result = runner.invoke(app, ["analyze-problem", project_id, "--projects-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "no problem files" in result.output


def test_analyze_problem_pipeline_with_test_model(tmp_path, monkeypatch) -> None:
    sample = {
        "problem_id": "ignored",
        "title": "Demo",
        "overall_goal": "goal",
        "subproblems": [{"subproblem_id": "q1", "statement": "s", "goal": "g"}],
    }
    monkeypatch.setattr(
        "mmos.cli.main.build_model",
        lambda settings: TestModel(custom_output_args=sample),
    )

    init_result = runner.invoke(app, ["init", "Pipe", "--projects-dir", str(tmp_path)])
    assert init_result.exit_code == 0
    project_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    project_id = project_dir.name

    problem = tmp_path / "problem.md"
    problem.write_text("# Task\n\nMaximize profit.", encoding="utf-8")
    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            project_id,
            str(problem),
            "--role",
            "problem",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert ingest_result.exit_code == 0

    result = runner.invoke(app, ["analyze-problem", project_id, "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "PROBLEM_UNDERSTOOD" in result.output

    spec = load_problem_spec(ProjectPaths(project_dir))
    assert spec.problem_id == project_id
    assert spec.subproblems[0].subproblem_id == "q1"


def test_analyze_data_flow(tmp_path) -> None:
    init_result = runner.invoke(app, ["init", "Data", "--projects-dir", str(tmp_path)])
    assert init_result.exit_code == 0
    project_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    project_id = project_dir.name

    data = tmp_path / "train.csv"
    data.write_text("id,y,x\n1,10,1\n2,12,2\n3,14,3\n", encoding="utf-8")
    ingest_result = runner.invoke(
        app,
        ["ingest", project_id, str(data), "--role", "data", "--projects-dir", str(tmp_path)],
    )
    assert ingest_result.exit_code == 0

    paths = ProjectPaths(project_dir)
    Orchestrator(paths).transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test setup")

    result = runner.invoke(
        app,
        ["analyze-data", project_id, "--target", "y", "--projects-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "DATA_VALIDATED" in result.output
    assert load_project_config(paths).state is ProjectState.DATA_VALIDATED
    assert paths.data_quality_file.exists()


def test_run_experiment_cli_dry_run(tmp_path) -> None:
    init_result = runner.invoke(app, ["init", "Run", "--projects-dir", str(tmp_path)])
    assert init_result.exit_code == 0
    project_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    project_id = project_dir.name

    data = tmp_path / "train.csv"
    data.write_text("x1,x2,y\n1,2,3\n2,3,5\n3,4,7\n", encoding="utf-8")
    ingest_result = runner.invoke(
        app,
        ["ingest", project_id, str(data), "--role", "data", "--projects-dir", str(tmp_path)],
    )
    assert ingest_result.exit_code == 0

    paths = ProjectPaths(project_dir)
    orchestrator = Orchestrator(paths)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test setup")
    orchestrator.transition(ProjectState.DATA_VALIDATED, reason="test setup")

    spec = ExperimentSpec(
        experiment_id="exp-cli",
        problem_id=project_id,
        proposal_id="p1",
        dataset_hash=dataset_fingerprint(load_manifest(paths)),
        target_column="y",
        feature_set=["x1", "x2"],
        split_strategy="kfold",
        validation_strategy="cross_validation",
        n_splits=2,
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
        random_seed=0,
        required_artifacts=["metrics.json"],
        acceptance_criteria=["beat baseline"],
        failure_criteria=["worse than baseline"],
        compute_requirements="test",
        expected_runtime_class="seconds",
    )
    save_experiment_spec(paths, spec)

    result = runner.invoke(
        app,
        [
            "run-experiment",
            project_id,
            "exp-cli",
            "--dry-run",
            "--projects-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output


def test_init_ingest_and_status_flow(tmp_path) -> None:
    init_result = runner.invoke(app, ["init", "My Demo", "--projects-dir", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.stdout
    assert "INIT" in init_result.stdout

    project_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    project_id = project_dir.name

    source = tmp_path / "train.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")

    ingest_result = runner.invoke(
        app,
        ["ingest", project_id, str(source), "--role", "data", "--projects-dir", str(tmp_path)],
    )
    assert ingest_result.exit_code == 0, ingest_result.stdout
    assert "PROJECT_INGESTED" in ingest_result.stdout

    status_result = runner.invoke(app, ["status", project_id, "--projects-dir", str(tmp_path)])
    assert status_result.exit_code == 0, status_result.stdout
    assert "PROJECT_INGESTED" in status_result.stdout
