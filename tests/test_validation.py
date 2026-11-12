import json
from pathlib import Path

from typer.testing import CliRunner

from mmos.cli.main import app
from mmos.experiments.runner import run_experiment
from mmos.orchestration import Orchestrator
from mmos.schemas.states import ProjectState
from mmos.validation import validate_experiment
from mmos.validation.leakage import validate_leakage
from mmos.validation.numerical import validate_numerical
from mmos.validation.reproducibility import validate_reproducibility
from tests.helpers import make_experiment_project, regression_frame

runner = CliRunner()


def test_clean_experiment_passes_all_validators_and_cli(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae", "r2"],
    )
    orchestrator = Orchestrator(paths)
    orchestrator.transition(ProjectState.PROJECT_INGESTED, reason="test setup")
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test setup")
    orchestrator.transition(ProjectState.DATA_VALIDATED, reason="test setup")
    run_experiment(paths, spec.experiment_id)

    reports = validate_experiment(paths, spec.experiment_id)

    assert {report.validator for report in reports} == {
        "leakage",
        "numerical",
        "statistical",
        "reproducibility",
        "temporal",
    }
    assert all(not report.blocking for report in reports)
    summary = json.loads(
        (paths.experiment_dir(spec.experiment_id) / "validation" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["blocking"] is False

    cli_result = runner.invoke(
        app,
        ["validate", paths.root.name, spec.experiment_id, "--projects-dir", str(projects_dir)],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "Validation passed" in cli_result.output


def test_numerical_detects_tampered_metric(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
    )
    run_experiment(paths, spec.experiment_id)

    metrics_file = paths.experiment_dir(spec.experiment_id) / "artifacts" / "metrics.json"
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    payload["models"]["sklearn.linear_regression"]["rmse"] += 0.5
    metrics_file.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_numerical(paths, spec.experiment_id)

    assert report.blocking is True
    assert any(issue.code == "metric_mismatch" for issue in report.issues)


def test_leakage_detects_target_used_as_feature(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
        feature_set=["x1", "y"],
    )

    report = validate_leakage(paths, spec.experiment_id)

    assert report.blocking is True
    assert any(issue.code == "target_in_features" for issue in report.issues)


def test_reproducibility_requires_mlflow_record(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
    )
    run_experiment(paths, spec.experiment_id)
    (paths.experiment_dir(spec.experiment_id) / "mlflow_run.json").unlink()

    report = validate_reproducibility(paths, spec.experiment_id)

    assert report.blocking is True
    assert any(issue.code == "missing_artifact" for issue in report.issues)
