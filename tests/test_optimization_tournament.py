from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import (
    create_project,
    experiment_record_file,
    save_experiment_record,
    save_optimization_result,
    save_optimization_spec,
)
from mmos.experiments.optimization_tournament import (
    load_optimization_tournament,
    run_optimization_tournament,
)
from mmos.schemas.optimization import (
    DecisionVariable,
    EnumerationResult,
    ExperimentStatus,
    LinearConstraint,
    OptimizationExperimentRecord,
    OptimizationModel,
    OptimizationResult,
    OptimizationSolution,
    OptimizationSpec,
    VariableType,
    optimization_instance_hash,
)
from mmos.schemas.optimization_tournament import OptimalityClass, WinnerStatus

NOW = datetime(2026, 11, 13, tzinfo=UTC)


def make_model(sense: str = "maximize", coefficient: float = 1.0) -> OptimizationModel:
    return OptimizationModel(
        name="one_variable",
        sense=sense,
        variables=[DecisionVariable(name="x", type=VariableType.INTEGER, lower=0, upper=1000)],
        objective={"x": coefficient},
        constraints=[LinearConstraint(name="cap", coefficients={"x": 1.0}, upper=1000)],
    )


def make_case(
    tmp_path: Path,
    *,
    sense: str = "maximize",
    candidate_objective: float = 360.0,
    baseline_objective: float = 350.0,
    candidate_feasible: bool = True,
    baseline_feasible: bool = True,
    candidate_validated: bool = True,
    baseline_model: OptimizationModel | None = None,
    enumeration: bool = True,
):
    paths, _ = create_project(tmp_path / "projects", "opt-tournament")
    model = make_model(sense)
    baseline_spec_model = baseline_model or model
    candidate_spec = OptimizationSpec(
        experiment_id="cand-exp",
        problem_id="p",
        model=model,
        solver="ortools.cbc",
        instance_hash=optimization_instance_hash(model),
    )
    baseline_spec = OptimizationSpec(
        experiment_id="base-exp",
        problem_id="p",
        model=baseline_spec_model,
        solver="greedy",
        instance_hash=optimization_instance_hash(baseline_spec_model),
    )
    save_optimization_spec(paths, candidate_spec)
    save_optimization_spec(paths, baseline_spec)

    if candidate_feasible:
        candidate_solution = OptimizationSolution(
            method="ortools.cbc",
            status="optimal",
            objective_value=candidate_objective,
            best_bound=candidate_objective,
            optimality_gap=0.0,
            absolute_gap=0.0,
            variable_values={"x": candidate_objective},
            feasible=True,
            runtime_seconds=0.01,
            solver="CBC",
            solver_version="9.15.6755",
            termination_condition="optimal",
        )
    else:
        candidate_solution = OptimizationSolution(
            method="ortools.cbc",
            status="infeasible",
            variable_values={},
            feasible=False,
            runtime_seconds=0.01,
            solver="CBC",
            termination_condition="infeasible",
        )
    if baseline_feasible:
        baseline_solution = OptimizationSolution(
            method="greedy_feasible",
            status="feasible",
            objective_value=baseline_objective,
            variable_values={"x": baseline_objective},
            feasible=True,
            solver="greedy",
            termination_condition="heuristic",
        )
    else:
        baseline_solution = OptimizationSolution(
            method="greedy_feasible",
            status="infeasible",
            variable_values={},
            feasible=False,
            solver="greedy",
            termination_condition="heuristic",
        )

    enum = (
        EnumerationResult(
            status="complete",
            evaluated_subsets=1,
            optimum_value=candidate_objective,
            optimal_subsets=1,
            matches_solver_result=True,
        )
        if enumeration and candidate_feasible
        else None
    )
    candidate_result = OptimizationResult(
        experiment_id="cand-exp",
        problem_id="p",
        model_name=model.name,
        sense=sense,
        solutions=[candidate_solution],
        selected_method="ortools.cbc" if candidate_feasible else None,
        enumeration=enum,
        instance_hash=candidate_spec.instance_hash,
        solver_backend="CBC",
        solver_version="9.15.6755",
        generated_at=NOW,
    )
    baseline_result = OptimizationResult(
        experiment_id="base-exp",
        problem_id="p",
        model_name=baseline_spec_model.name,
        sense=sense,
        solutions=[baseline_solution],
        selected_method="greedy_feasible" if baseline_feasible else None,
        instance_hash=baseline_spec.instance_hash,
        solver_backend="greedy",
        solver_version="n/a",
        generated_at=NOW,
    )
    save_optimization_result(paths, candidate_result)
    save_optimization_result(paths, baseline_result)

    save_experiment_record(
        paths,
        OptimizationExperimentRecord(
            experiment_id="cand-exp",
            project_id=paths.root.name,
            created_at=NOW,
            status=ExperimentStatus.VALIDATED,
            instance_hash=candidate_spec.instance_hash or "",
            config_hash="c" * 64,
            validation_status=({"optimization": "passed"} if candidate_validated else {}),
            solver="ortools.cbc",
        ),
    )
    baseline_record = OptimizationExperimentRecord(
        experiment_id="base-exp",
        project_id=paths.root.name,
        created_at=NOW,
        status=ExperimentStatus.SUCCEEDED,
        instance_hash=baseline_spec.instance_hash or "",
        config_hash="b" * 64,
        solver="greedy",
    )
    save_experiment_record(paths, baseline_record)
    return paths, candidate_spec, baseline_spec, baseline_record


def test_maximize_candidate_wins(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, _ = make_case(tmp_path)

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    assert result.winner_status is WinnerStatus.WINNER
    assert result.winner_experiment_id == "cand-exp"
    assert result.equivalent_experiment_ids == []
    assert any("Relative improvement: 0.02857" in line for line in result.decision_trace)
    candidate = next(entry for entry in result.entries if entry.entry_role.value == "candidate")
    baseline = next(entry for entry in result.entries if entry.entry_role.value == "baseline")
    assert candidate.optimality_class is OptimalityClass.PROVEN_OPTIMAL
    assert baseline.optimality_class is OptimalityClass.FEASIBLE_NOT_PROVEN_OPTIMAL
    assert baseline.baseline_role.value == "sanity_baseline"
    assert baseline.best_bound is None and baseline.relative_gap is None


def test_minimize_candidate_wins_without_sign_regression(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, _ = make_case(
        tmp_path, sense="minimize", candidate_objective=90.0, baseline_objective=100.0
    )

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    assert result.winner_status is WinnerStatus.WINNER
    assert result.winner_experiment_id == "cand-exp"


def test_infeasible_candidate_cannot_win(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, _ = make_case(
        tmp_path, candidate_feasible=False, baseline_feasible=True
    )

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    assert result.winner_status is WinnerStatus.WINNER
    assert result.winner_experiment_id == "base-exp"
    candidate = next(entry for entry in result.entries if entry.entry_role.value == "candidate")
    assert candidate.feasible is False
    assert candidate.optimality_class is OptimalityClass.INFEASIBLE


def test_invalid_experiment_cannot_win(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, _ = make_case(tmp_path, candidate_validated=False)

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    assert result.winner_status is WinnerStatus.INVALID
    assert result.winner_experiment_id is None


def test_incomparable_experiments_have_no_winner(tmp_path: Path) -> None:
    changed = make_model(coefficient=2.0)
    paths, candidate_spec, baseline_spec, _ = make_case(tmp_path, baseline_model=changed)

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    assert result.winner_status is WinnerStatus.INCOMPARABLE
    assert result.winner_experiment_id is None
    assert result.incomparable_pairs
    assert not result.comparison_matrix[0].comparable
    assert any("No improvement is computed" in line for line in result.decision_trace)


def test_equal_objectives_are_equivalent(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, _ = make_case(
        tmp_path, candidate_objective=350.0, baseline_objective=350.0
    )

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    assert result.winner_status is WinnerStatus.EQUIVALENT
    assert result.winner_experiment_id is None
    assert set(result.equivalent_experiment_ids) == {"cand-exp", "base-exp"}


def test_tournament_artifacts_and_experiment_independence(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, baseline_record = make_case(tmp_path)
    record_path = experiment_record_file(paths, baseline_spec.experiment_id)
    before = record_path.read_bytes()

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )
    loaded = load_optimization_tournament(paths, result.tournament_id)

    directory = paths.tournaments_dir / result.tournament_id
    assert (directory / "tournament.json").exists()
    assert (directory / "decision_trace.md").exists()
    assert loaded.winner_experiment_id == "cand-exp"
    assert record_path.read_bytes() == before
    assert not record_path.read_text(encoding="utf-8").count("is_winner")
    assert baseline_record.experiment_id == "base-exp"


def test_solver_optimal_without_enumeration_is_distinguished(tmp_path: Path) -> None:
    paths, candidate_spec, baseline_spec, _ = make_case(tmp_path, enumeration=False)

    result = run_optimization_tournament(
        paths, candidate_spec.experiment_id, baseline_spec.experiment_id
    )

    candidate = next(entry for entry in result.entries if entry.entry_role.value == "candidate")
    assert candidate.optimality_class is OptimalityClass.SOLVER_OPTIMAL
