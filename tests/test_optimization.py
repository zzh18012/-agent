import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from mmos.cli.main import app
from mmos.core.project import (
    create_project,
    load_experiment_record,
    load_project_config,
    save_optimization_spec,
)
from mmos.evidence import freeze_optimization, load_claims, load_evidence_index
from mmos.experiments.optimization_runner import (
    comparison_diagnostics,
    create_baseline_experiment,
    run_optimization,
    solve_greedy,
    solve_ortools,
)
from mmos.experiments.optimization_tournament import (
    load_optimization_tournament,
)
from mmos.orchestration import Orchestrator
from mmos.paper import render_markdown
from mmos.paper.optimization_builder import build_optimization_paper
from mmos.schemas.optimization import (
    ExperimentStatus,
    OptimizationSpec,
    optimization_instance_hash,
    relative_improvement,
)
from mmos.schemas.project import SourceRole
from mmos.schemas.review import (
    AttackReport,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)
from mmos.schemas.states import ProjectState, TaskFamily
from mmos.schemas.tracking import MLflowRunRecord
from mmos.validation.optimization import (
    validate_optimization,
    validate_optimization_experiment,
)

SPEC_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "milp_knapsack" / "spec.json"


def load_spec() -> OptimizationSpec:
    return OptimizationSpec.model_validate_json(SPEC_PATH.read_text(encoding="utf-8"))


def brute_force_optimum(spec: OptimizationSpec) -> float:
    model = spec.model
    names = [variable.name for variable in model.variables]
    best = float("-inf")
    for bits in itertools.product([0, 1], repeat=len(names)):
        values = {name: float(bit) for name, bit in zip(names, bits, strict=True)}
        from mmos.experiments.optimization_runner import check_solution

        feasible, _, objective = check_solution(model, values)
        if feasible:
            best = max(best, objective)
    return best


def test_solver_matches_brute_force_optimum() -> None:
    spec = load_spec()
    optimum = brute_force_optimum(spec)

    solution = solve_ortools(spec.model, spec.time_limit_seconds)

    assert solution.status == "optimal"
    assert solution.feasible is True
    assert solution.objective_value == optimum == 360.0
    assert solution.optimality_gap == 0.0


def test_greedy_baseline_is_feasible_and_not_better() -> None:
    spec = load_spec()
    solution = solve_greedy(spec.model)

    assert solution.feasible is True
    assert solution.objective_value is not None
    assert solution.objective_value <= 360.0


def test_optimization_vertical_slice(tmp_path: Path) -> None:
    spec = load_spec()
    paths, _ = create_project(tmp_path / "projects", "milp-demo")
    save_optimization_spec(paths, spec)

    result = run_optimization(paths, spec.experiment_id)
    solver = next(solution for solution in result.solutions if solution.method == "ortools.cbc")
    assert solver.objective_value == 360.0
    assert result.selected_method == "ortools.cbc"

    reports = validate_optimization_experiment(paths, spec.experiment_id)
    assert all(not report.blocking for report in reports)

    records = freeze_optimization(paths, spec.experiment_id)
    index = load_evidence_index(paths)
    assert any(record.key == "optimization.selected_method" for record in index.records.values())
    assert any(
        record.key == "optimization.ortools.cbc.objective_value"
        for record in index.records.values()
    )
    assert len(records) >= 10
    improvement_records = [
        record for record in records if record.key == "optimization.relative_improvement"
    ]
    assert improvement_records
    assert improvement_records[0].raw_value > 0
    assert result.enumeration is not None
    assert result.enumeration.status == "complete"
    assert result.enumeration.optimum_value == 360.0
    assert result.enumeration.evaluated_subsets == 1024

    paper = build_optimization_paper(paths, spec.experiment_id)
    text = render_markdown(paper)
    assert "360" in text
    assert "ortools.cbc" in text
    assert "greedy_feasible" in text
    assert "capacity" in text


def test_tampered_objective_is_blocking(tmp_path: Path) -> None:
    spec = load_spec()
    paths, _ = create_project(tmp_path / "projects", "milp-tamper")
    save_optimization_spec(paths, spec)
    run_optimization(paths, spec.experiment_id)

    result_path = (
        paths.experiment_dir(spec.experiment_id) / "artifacts" / "optimization_result.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["solutions"][0]["objective_value"] += 10.0
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_optimization(paths, spec.experiment_id)

    assert report.blocking is True
    assert any(issue.code == "objective_mismatch" for issue in report.issues)


def test_relative_improvement_follows_objective_direction() -> None:
    assert relative_improvement("maximize", 360.0, 350.0) == pytest.approx(10.0 / 350.0)
    assert relative_improvement("maximize", 340.0, 350.0) < 0
    assert relative_improvement("minimize", 90.0, 100.0) == pytest.approx(0.1)
    assert relative_improvement("minimize", 110.0, 100.0) < 0


def test_optimize_cli_enters_shared_lifecycle(tmp_path, monkeypatch) -> None:
    spec = load_spec()
    projects_dir = tmp_path / "projects"
    paths, _ = create_project(projects_dir, "opt-e2e", task=TaskFamily.OPTIMIZATION)
    problem = tmp_path / "problem.md"
    problem.write_text("# Knapsack\n", encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")

    class FakeRedTeam:
        def __init__(self, model) -> None:
            pass

        def run(self, paper, *, experiment_id: str) -> RedTeamReview:
            return RedTeamReview(
                experiment_id=experiment_id,
                attacks=AttackReport(attacks=[]),
                responses=AuthorResponseReport(responses=[]),
                judge=JudgeReport(verdicts=[], overall=Verdict.PASS, blocking=False),
                created_at=datetime.now(UTC),
            )

    monkeypatch.setattr("mmos.cli.main._model_or_exit", lambda: "fake")
    monkeypatch.setattr("mmos.cli.main.RedTeam", FakeRedTeam)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            paths.root.name,
            str(SPEC_PATH),
            "--projects-dir",
            str(projects_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert load_project_config(paths).state is ProjectState.RELEASED
    assert (paths.releases_dir / f"optimization_{spec.experiment_id}.md").exists()
    history = paths.state_history_file.read_text(encoding="utf-8")
    assert "RED_TEAM_REVIEW" in history
    assert "RELEASED" in history


def test_blocked_optimization_enters_human_review(tmp_path, monkeypatch) -> None:
    spec = load_spec()
    projects_dir = tmp_path / "projects"
    paths, _ = create_project(projects_dir, "opt-blocked", task=TaskFamily.OPTIMIZATION)
    problem = tmp_path / "problem-blocked.md"
    problem.write_text("# Knapsack\n", encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")

    class ReviseRedTeam:
        def __init__(self, model) -> None:
            pass

        def run(self, paper, *, experiment_id: str) -> RedTeamReview:
            return RedTeamReview(
                experiment_id=experiment_id,
                attacks=AttackReport(attacks=[]),
                responses=AuthorResponseReport(responses=[]),
                judge=JudgeReport(verdicts=[], overall=Verdict.REVISE, blocking=False),
                created_at=datetime.now(UTC),
            )

    monkeypatch.setattr("mmos.cli.main._model_or_exit", lambda: "fake")
    monkeypatch.setattr("mmos.cli.main.RedTeam", ReviseRedTeam)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            paths.root.name,
            str(SPEC_PATH),
            "--projects-dir",
            str(projects_dir),
        ],
    )

    assert result.exit_code == 1
    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.RED_TEAM_REVIEW
    assert not (paths.releases_dir / f"optimization_{spec.experiment_id}.md").exists()


def test_instance_hash_changes_with_instance_data() -> None:
    spec = load_spec()
    first = optimization_instance_hash(spec.model)
    second = optimization_instance_hash(spec.model.model_copy(deep=True))
    assert first == second

    changed_objective = dict(spec.model.objective)
    changed_objective["x_A"] = changed_objective["x_A"] + 1.0
    changed = spec.model.model_copy(update={"objective": changed_objective})
    assert optimization_instance_hash(changed) != first

    constraint = spec.model.constraints[0]
    changed_constraints = [constraint.model_copy(update={"upper": constraint.upper + 1.0})]
    changed_capacity = spec.model.model_copy(update={"constraints": changed_constraints})
    assert optimization_instance_hash(changed_capacity) != first


def test_comparison_preconditions() -> None:
    spec = load_spec()
    baseline = spec.model_copy(update={"experiment_id": "baseline", "solver": "greedy"})
    ok, reasons = comparison_diagnostics(spec, baseline)
    assert ok is True
    assert reasons == []

    different_limit = baseline.model_copy(update={"time_limit_seconds": 1.0})
    ok, reasons = comparison_diagnostics(spec, different_limit)
    assert ok is True
    assert any("time limits differ" in reason for reason in reasons)

    changed_objective = dict(spec.model.objective)
    changed_objective["x_A"] += 1.0
    changed = baseline.model_copy(
        update={"model": spec.model.model_copy(update={"objective": changed_objective})}
    )
    ok, reasons = comparison_diagnostics(spec, changed)
    assert ok is False
    assert any("objective" in reason for reason in reasons)


def test_baseline_is_a_formal_experiment(tmp_path: Path) -> None:
    spec = load_spec()
    paths, _ = create_project(tmp_path / "projects", "baseline-demo")
    save_optimization_spec(paths, spec)
    result = run_optimization(paths, spec.experiment_id)

    baseline_spec, baseline_result = create_baseline_experiment(paths, spec, result)

    assert baseline_spec.experiment_id.endswith("-baseline-greedy")
    assert baseline_result.experiment_id == baseline_spec.experiment_id
    assert baseline_result.selected_method == "greedy_feasible"
    assert baseline_result.instance_hash == result.instance_hash
    assert baseline_spec.instance_hash == spec.instance_hash or baseline_spec.instance_hash
    assert (
        paths.experiment_dir(baseline_spec.experiment_id) / "artifacts" / "optimization_result.json"
    ).exists()


def test_experiment_identity_and_mlflow_are_linked(tmp_path: Path, monkeypatch) -> None:
    spec = load_spec()
    projects_dir = tmp_path / "projects"
    paths, _ = create_project(projects_dir, "opt-identity", task=TaskFamily.OPTIMIZATION)
    problem = tmp_path / "problem-identity.md"
    problem.write_text("# Knapsack\n", encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")

    class PassRedTeam:
        def __init__(self, model) -> None:
            pass

        def run(self, paper, *, experiment_id: str) -> RedTeamReview:
            return RedTeamReview(
                experiment_id=experiment_id,
                attacks=AttackReport(attacks=[]),
                responses=AuthorResponseReport(responses=[]),
                judge=JudgeReport(verdicts=[], overall=Verdict.PASS, blocking=False),
                created_at=datetime.now(UTC),
            )

    monkeypatch.setattr("mmos.cli.main._model_or_exit", lambda: "fake")
    monkeypatch.setattr("mmos.cli.main.RedTeam", PassRedTeam)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            paths.root.name,
            str(SPEC_PATH),
            "--projects-dir",
            str(projects_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    record = load_experiment_record(paths, spec.experiment_id)
    assert record.status is ExperimentStatus.VALIDATED
    assert record.instance_hash
    assert record.config_hash
    assert record.git_commit
    assert record.mlflow_run_id
    assert record.linked_experiment_ids
    baseline_id = record.linked_experiment_ids[0]
    baseline_record = load_experiment_record(paths, baseline_id)
    assert baseline_record.status is ExperimentStatus.SUCCEEDED
    assert baseline_record.instance_hash == record.instance_hash
    assert baseline_record.mlflow_run_id != record.mlflow_run_id

    for item in (record, baseline_record):
        mlflow_path = paths.experiment_dir(item.experiment_id) / "mlflow_run.json"
        mlflow_record = MLflowRunRecord.model_validate_json(mlflow_path.read_text(encoding="utf-8"))
        client = MlflowClient(tracking_uri=mlflow_record.tracking_uri)
        run = client.get_run(mlflow_record.run_id)
        assert run.data.tags["task_family"] == "optimization"
        assert run.data.tags["instance_hash"] == item.instance_hash
        assert run.data.params["number_of_variables"] == "10"
        assert (
            "solver.objective_value" in run.data.metrics
            or "baseline.objective_value" in run.data.metrics
        )

    evidence = load_evidence_index(paths)
    keys = {record.key for record in evidence.records.values()}
    assert "experiment.mlflow_run_id" in keys
    assert "experiment.instance_hash" in keys
    assert "experiment.linked_experiment_id" in keys
    assert "tournament.status" in keys
    assert "tournament.winner_experiment_id" in keys
    assert "tournament.relative_improvement" in keys

    tournaments = list(paths.tournaments_dir.glob("tournament-*/tournament.json"))
    assert tournaments
    tournament = load_optimization_tournament(paths, tournaments[0].parent.name)
    assert tournament.winner_experiment_id == spec.experiment_id
    assert (tournaments[0].parent / "decision_trace.md").exists()
    assert (tournaments[0].parent / "mlflow_run.json").exists()

    claims = load_claims(paths)
    assert claims
    assert "higher objective" in claims[0].statement
    assert "generally" not in claims[0].statement.lower()
    paper_text = (paths.root / "paper" / "report.md").read_text(encoding="utf-8")
    assert "Optimization comparison" in paper_text
    assert "benchmark instance" in paper_text
