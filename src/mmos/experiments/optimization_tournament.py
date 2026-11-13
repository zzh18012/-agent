import secrets
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_record,
    load_optimization_result,
    load_optimization_spec,
)
from mmos.experiments.optimization_runner import (
    check_solution,
    comparison_diagnostics,
)
from mmos.schemas.optimization import relative_improvement
from mmos.schemas.optimization_tournament import (
    BaselineRole,
    ComparisonMatrixEntry,
    EntryRole,
    MethodFamily,
    OptimalityClass,
    OptimizationTournamentEntry,
    OptimizationTournamentResult,
    WinnerStatus,
)

_TOLERANCE = 1e-6


def _solution_for(result, role: EntryRole):
    if role is EntryRole.BASELINE:
        return next(
            (solution for solution in result.solutions if solution.method == "greedy_feasible"),
            None,
        )
    return next(
        (solution for solution in result.solutions if solution.method != "greedy_feasible"),
        None,
    )


def _build_entry(
    paths: ProjectPaths, experiment_id: str, role: EntryRole
) -> OptimizationTournamentEntry:
    spec = load_optimization_spec(paths, experiment_id)
    result = load_optimization_result(paths, experiment_id)
    record = load_experiment_record(paths, experiment_id)
    solution = _solution_for(result, role)
    warnings: list[str] = []
    valid = record.status.value not in {"failed", "invalid"}
    if role is EntryRole.CANDIDATE and record.validation_status.get("optimization") != "passed":
        valid = False
        warnings.append("candidate optimization validation did not pass")
    if solution is None:
        valid = False
        warnings.append("no solution for the requested role")
        return OptimizationTournamentEntry(
            experiment_id=experiment_id,
            model_id=spec.model.name,
            method_family=MethodFamily.EXACT_SOLVER,
            solver=spec.solver,
            instance_hash=spec.instance_hash or result.instance_hash or "",
            objective_sense=spec.model.sense,
            feasible=False,
            valid=False,
            validation_status=record.validation_status,
            entry_role=role,
            baseline_role=(BaselineRole.SANITY_BASELINE if role is EntryRole.BASELINE else None),
            warnings=warnings,
            optimality_class=OptimalityClass.INVALID,
        )

    infeasible_status = solution.status in {"infeasible", "unbounded"}
    if infeasible_status or not solution.feasible:
        feasible = False
        violations = []
        warnings.append(f"solver reported {solution.status}")
    else:
        feasible, violations, recomputed = check_solution(spec.model, solution.variable_values)
        if not feasible or violations:
            valid = False
            warnings.append("constraint audit failed")
        if (
            solution.objective_value is None
            or abs(solution.objective_value - recomputed) > _TOLERANCE
        ):
            valid = False
            warnings.append("objective recomputation mismatch")

    verification_method = None
    verification_status = None
    if result.enumeration is not None:
        verification_method = result.enumeration.method
        verification_status = result.enumeration.status
    if not valid:
        optimality = OptimalityClass.INVALID
    elif not feasible:
        optimality = OptimalityClass.INFEASIBLE
    elif solution.status == "optimal" and result.enumeration is not None:
        if (
            result.enumeration.status == "complete"
            and result.enumeration.matches_solver_result is True
        ):
            optimality = OptimalityClass.PROVEN_OPTIMAL
        else:
            optimality = OptimalityClass.SOLVER_OPTIMAL
    elif solution.status == "optimal":
        optimality = OptimalityClass.SOLVER_OPTIMAL
    else:
        optimality = OptimalityClass.FEASIBLE_NOT_PROVEN_OPTIMAL

    if feasible and solution.best_bound is None:
        warnings.append("best bound unavailable")
    if feasible and solution.optimality_gap is None:
        warnings.append("relative gap unavailable")

    return OptimizationTournamentEntry(
        experiment_id=experiment_id,
        model_id=spec.model.name,
        method_family=(
            MethodFamily.HEURISTIC if spec.solver == "greedy" else MethodFamily.EXACT_SOLVER
        ),
        solver=spec.solver,
        instance_hash=spec.instance_hash or result.instance_hash or "",
        objective_sense=spec.model.sense,
        feasible=feasible,
        valid=valid,
        objective_value=solution.objective_value,
        solver_status=solution.status,
        termination_condition=solution.termination_condition,
        best_bound=solution.best_bound,
        absolute_gap=solution.absolute_gap,
        relative_gap=solution.optimality_gap,
        runtime_seconds=solution.runtime_seconds,
        time_limit_seconds=spec.time_limit_seconds,
        validation_status=record.validation_status,
        independent_verification_method=verification_method,
        independent_verification_status=verification_status,
        optimality_class=optimality,
        entry_role=role,
        baseline_role=(BaselineRole.SANITY_BASELINE if role is EntryRole.BASELINE else None),
        warnings=warnings,
    )


def _decision_trace(
    candidate: OptimizationTournamentEntry,
    baseline: OptimizationTournamentEntry,
    *,
    comparable: bool,
    reasons: list[str],
    winner_status: WinnerStatus,
    absolute_improvement: float | None,
    relative: float | None,
) -> list[str]:
    trace = [
        "Comparison preconditions:",
        f"- same instance hash: "
        f"{'PASS' if candidate.instance_hash == baseline.instance_hash else 'FAIL'}",
        f"- same objective definition: "
        f"{'PASS' if not any('objective' in reason for reason in reasons) else 'FAIL'}",
        f"- same objective sense: "
        f"{'PASS' if candidate.objective_sense == baseline.objective_sense else 'FAIL'}",
        f"- same hard constraints: "
        f"{'PASS' if not any('constraint' in reason for reason in reasons) else 'FAIL'}",
        f"Comparison status: {'comparable' if comparable else 'INCOMPARABLE'}",
        f"Validity: candidate={candidate.valid}, baseline={baseline.valid}",
        f"Feasibility: candidate={candidate.feasible}, baseline={baseline.feasible}",
        f"Objective sense: {candidate.objective_sense}",
        f"Objective values: candidate={candidate.objective_value}, "
        f"baseline={baseline.objective_value}",
        f"Absolute improvement (candidate - baseline): {absolute_improvement}",
        f"Relative improvement: {relative}",
        f"Candidate optimality: {candidate.optimality_class.value}; "
        f"verification={candidate.independent_verification_method or 'unavailable'}",
        f"Baseline role: {baseline.baseline_role.value if baseline.baseline_role else 'none'}",
        f"Tournament conclusion: {winner_status.value}",
    ]
    if not comparable:
        trace.append("No improvement is computed for incomparable experiments.")
    return trace


def run_optimization_tournament(
    paths: ProjectPaths,
    candidate_id: str,
    baseline_id: str,
    *,
    objective_tolerance: float = 1e-6,
) -> OptimizationTournamentResult:
    candidate_spec = load_optimization_spec(paths, candidate_id)
    baseline_spec = load_optimization_spec(paths, baseline_id)
    candidate = _build_entry(paths, candidate_id, EntryRole.CANDIDATE)
    baseline = _build_entry(paths, baseline_id, EntryRole.BASELINE)

    comparable, reasons = comparison_diagnostics(candidate_spec, baseline_spec)
    warnings = list(reasons)
    absolute_improvement = None
    relative = None
    equivalent_ids: list[str] = []
    incomparable_pairs: list[list[str]] = []
    winner: str | None = None

    if not comparable:
        winner_status = WinnerStatus.INCOMPARABLE
        incomparable_pairs.append([candidate_id, baseline_id])
    elif not candidate.valid or not baseline.valid:
        winner_status = WinnerStatus.INVALID
    elif not candidate.feasible and not baseline.feasible:
        winner_status = WinnerStatus.NO_WINNER
    elif not candidate.feasible:
        winner_status = WinnerStatus.WINNER
        winner = baseline_id
    elif not baseline.feasible:
        winner_status = WinnerStatus.WINNER
        winner = candidate_id
    else:
        assert candidate.objective_value is not None
        assert baseline.objective_value is not None
        absolute_improvement = candidate.objective_value - baseline.objective_value
        relative = relative_improvement(
            candidate.objective_sense,
            candidate.objective_value,
            baseline.objective_value,
        )
        if abs(absolute_improvement) <= objective_tolerance:
            winner_status = WinnerStatus.EQUIVALENT
            equivalent_ids = [candidate_id, baseline_id]
        else:
            candidate_better = (
                absolute_improvement > 0
                if candidate.objective_sense == "maximize"
                else absolute_improvement < 0
            )
            winner_status = WinnerStatus.WINNER
            winner = candidate_id if candidate_better else baseline_id

    feasibility_relationship = (
        "both feasible"
        if candidate.feasible and baseline.feasible
        else "candidate infeasible"
        if not candidate.feasible
        else "baseline infeasible"
    )
    objective_relationship = (
        f"candidate {candidate.objective_value} vs baseline {baseline.objective_value}"
        if candidate.objective_value is not None and baseline.objective_value is not None
        else "not evaluated"
    )
    optimality_relationship = (
        f"candidate={candidate.optimality_class.value}, baseline={baseline.optimality_class.value}"
    )
    matrix = ComparisonMatrixEntry(
        experiment_a=candidate_id,
        experiment_b=baseline_id,
        comparable=comparable,
        reasons=reasons,
        feasibility_relationship=feasibility_relationship,
        objective_relationship=objective_relationship,
        optimality_relationship=optimality_relationship,
        warnings=warnings,
    )
    trace = _decision_trace(
        candidate,
        baseline,
        comparable=comparable,
        reasons=reasons,
        winner_status=winner_status,
        absolute_improvement=absolute_improvement,
        relative=relative,
    )

    result = OptimizationTournamentResult(
        tournament_id=f"tournament-{secrets.token_hex(4)}",
        project_id=paths.root.name,
        instance_hash=candidate.instance_hash,
        entries=[candidate, baseline],
        comparison_matrix=[matrix],
        winner_experiment_id=winner,
        winner_status=winner_status,
        equivalent_experiment_ids=equivalent_ids,
        baseline_experiment_ids=[baseline_id],
        incomparable_pairs=incomparable_pairs,
        decision_trace=trace,
        warnings=warnings,
        objective_tolerance=objective_tolerance,
        created_at=datetime.now(UTC).replace(microsecond=0),
    )
    save_optimization_tournament(paths, result)
    return result


def load_optimization_tournament(
    paths: ProjectPaths, tournament_id: str
) -> OptimizationTournamentResult:
    return OptimizationTournamentResult.model_validate_json(
        (optimization_tournament_dir(paths, tournament_id) / "tournament.json").read_text(
            encoding="utf-8"
        )
    )


def optimization_tournament_dir(paths: ProjectPaths, tournament_id: str) -> Path:
    return paths.tournaments_dir / tournament_id


def save_optimization_tournament(paths: ProjectPaths, result: OptimizationTournamentResult) -> Path:
    directory = optimization_tournament_dir(paths, result.tournament_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "tournament.json"
    atomic_write_text(target, result.model_dump_json(indent=2) + "\n")
    atomic_write_text(
        directory / "decision_trace.md",
        "\n".join([f"# Tournament {result.tournament_id}", "", *result.decision_trace]) + "\n",
    )
    return target
