from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from mmos import __version__
from mmos.cli.main import app
from mmos.core.project import ProjectPaths, load_problem_spec

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
