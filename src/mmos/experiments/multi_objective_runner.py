import csv
import io
import time
from datetime import UTC, datetime
from importlib import metadata

from ortools.linear_solver import pywraplp

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
    epsilon_solutions_file,
    save_multi_objective_result,
    save_pareto_oracle,
)
from mmos.experiments.multi_objective_oracle import enumerate_pareto_oracle
from mmos.experiments.pareto_utils import hypervolume_2d
from mmos.schemas.multi_objective import (
    EpsilonCoverage,
    EpsilonSensitivity,
    EpsilonSolve,
    FrontComparison,
    MultiObjectiveExperimentRecord,
    MultiObjectiveResult,
    MultiObjectiveSpec,
    ParetoSolution,
    ParetoVector,
)
from mmos.schemas.optimization import ExperimentStatus


class MultiObjectiveError(RuntimeError):
    pass


def _candidate_dominates(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    # Candidate-path maximize dominance, independently written from the oracle.
    if left[0] < right[0] or left[1] < right[1]:
        return False
    return left[0] > right[0] or left[1] > right[1]


def _solve_stage(
    spec: MultiObjectiveSpec,
    epsilon: int,
    *,
    fixed_f1: float | None,
    maximize_secondary: bool,
):
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise MultiObjectiveError("CBC solver is unavailable")
    solver.SetTimeLimit(int(spec.solver_time_limit_seconds * 1000))
    variables = {
        item.project_id: solver.IntVar(0, 1, f"x_{item.project_id}")
        for item in spec.items
    }
    primary = spec.primary_objective_id
    secondary = spec.epsilon_objective_id
    solver.Add(
        sum(item.cost * variables[item.project_id] for item in spec.items)
        <= spec.budget
    )
    f1_expression = sum(
        item.benefits[primary] * variables[item.project_id] for item in spec.items
    )
    f2_expression = sum(
        item.benefits[secondary] * variables[item.project_id] for item in spec.items
    )
    solver.Add(f2_expression >= epsilon)
    if fixed_f1 is not None:
        solver.Add(f1_expression == fixed_f1)
    if maximize_secondary:
        solver.Maximize(f2_expression)
    else:
        solver.Maximize(f1_expression)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        decision = {
            item.project_id: round(variables[item.project_id].solution_value())
            for item in spec.items
        }
        return solver, status, decision
    if status == pywraplp.Solver.INFEASIBLE:
        return solver, status, None
    return solver, status, None


def _status_name(status: int) -> str:
    if status == pywraplp.Solver.OPTIMAL:
        return "optimal"
    if status == pywraplp.Solver.INFEASIBLE:
        return "infeasible"
    if status == pywraplp.Solver.FEASIBLE:
        return "feasible"
    return "error"


def _objectives(
    spec: MultiObjectiveSpec, decision: dict[str, int]
) -> tuple[float, float]:
    primary = spec.primary_objective_id
    secondary = spec.epsilon_objective_id
    f1 = sum(
        item.benefits[primary] * decision[item.project_id]
        for item in spec.items
    )
    f2 = sum(
        item.benefits[secondary] * decision[item.project_id]
        for item in spec.items
    )
    return float(f1), float(f2)


def _levels_for_step(spec: MultiObjectiveSpec, step: int) -> list[int]:
    maximum = int(
        sum(item.benefits[spec.epsilon_objective_id] for item in spec.items)
    )
    levels = list(range(0, maximum + 1, step))
    if not levels or levels[-1] != maximum:
        levels.append(maximum)
    return levels


def _sweep(
    spec: MultiObjectiveSpec, levels: list[int]
) -> tuple[list[EpsilonSolve], list[tuple[float, float, dict[str, int]]], int]:
    raw: list[EpsilonSolve] = []
    solver_calls = 0
    for epsilon in levels:
        stage_started = time.perf_counter()
        _, status1, decision1 = _solve_stage(
            spec, epsilon, fixed_f1=None, maximize_secondary=False
        )
        solver_calls += 1
        stage1_name = _status_name(status1)
        if decision1 is None:
            raw.append(
                EpsilonSolve(
                    epsilon=float(epsilon),
                    status="infeasible" if status1 == pywraplp.Solver.INFEASIBLE else "error",
                    feasible=False,
                    stage1_status=stage1_name,
                    runtime_seconds=time.perf_counter() - stage_started,
                )
            )
            continue
        f1, _ = _objectives(spec, decision1)
        _, status2, decision2 = _solve_stage(
            spec, epsilon, fixed_f1=f1, maximize_secondary=True
        )
        solver_calls += 1
        stage2_name = _status_name(status2)
        if decision2 is None:
            raw.append(
                EpsilonSolve(
                    epsilon=float(epsilon),
                    status="error",
                    feasible=False,
                    stage1_status=stage1_name,
                    stage2_status=stage2_name,
                    runtime_seconds=time.perf_counter() - stage_started,
                )
            )
            continue
        f1_final, f2_final = _objectives(spec, decision2)
        raw.append(
            EpsilonSolve(
                epsilon=float(epsilon),
                status="optimal",
                f1=f1_final,
                f2=f2_final,
                decision_vector=decision2,
                feasible=True,
                stage1_status=stage1_name,
                stage2_status=stage2_name,
                runtime_seconds=time.perf_counter() - stage_started,
            )
        )
    feasible_points: list[tuple[float, float, dict[str, int]]] = []
    for solution in raw:
        if solution.feasible and solution.f1 is not None and solution.f2 is not None:
            feasible_points.append(
                (solution.f1, solution.f2, solution.decision_vector)
            )
    return raw, feasible_points, solver_calls


def _candidate_front(
    unique_vectors: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    return [
        candidate
        for candidate in unique_vectors
        if not any(
            _candidate_dominates(other, candidate)
            for other in unique_vectors
            if other != candidate
        )
    ]


def run_multi_objective(
    paths: ProjectPaths, spec: MultiObjectiveSpec
) -> MultiObjectiveResult:
    if spec.instance_hash != spec.computed_instance_hash():
        raise MultiObjectiveError("instance hash does not match instance content")
    if spec.spec_hash != spec.computed_spec_hash():
        raise MultiObjectiveError("spec hash does not match spec content")
    started = datetime.now(UTC).replace(microsecond=0)
    clock = time.perf_counter()
    levels = spec.epsilon_levels()
    raw, feasible_points, solver_calls = _sweep(spec, levels)
    raw_vectors = [(f1, f2) for f1, f2, _ in feasible_points]
    unique_vectors = sorted(set(raw_vectors), key=lambda point: (point[0], point[1]))
    front_vectors = _candidate_front(unique_vectors)
    front_set = set(front_vectors)
    candidate_solutions: dict[tuple[tuple[str, int], ...], ParetoSolution] = {}
    for f1, f2, decision in feasible_points:
        if (f1, f2) in front_set:
            key = tuple(sorted(decision.items()))
            candidate_solutions[key] = ParetoSolution(
                f1=f1,
                f2=f2,
                decision_vector=decision,
            )
    ordered_solutions = sorted(
        candidate_solutions.values(),
        key=lambda solution: (
            solution.f1,
            solution.f2,
            tuple(sorted(solution.decision_vector.items())),
        ),
    )

    oracle = enumerate_pareto_oracle(spec)
    oracle_vectors = {
        (round(vector.f1, 9), round(vector.f2, 9))
        for vector in oracle.pareto_objective_vectors
    }
    candidate_keys = {
        (round(f1, 9), round(f2, 9)) for f1, f2 in front_vectors
    }
    missing = sorted(oracle_vectors - candidate_keys)
    extra = sorted(candidate_keys - oracle_vectors)
    dominated = [
        (f1, f2)
        for f1, f2 in front_vectors
        if any(
            _candidate_dominates(
                (vector.f1, vector.f2), (f1, f2)
            )
            for vector in oracle.pareto_objective_vectors
        )
    ]
    precision = (
        len(candidate_keys & oracle_vectors) / len(candidate_keys)
        if candidate_keys
        else 0.0
    )
    recall = len(candidate_keys & oracle_vectors) / len(oracle_vectors) if oracle_vectors else 1.0
    sensitivity: list[EpsilonSensitivity] = []
    for step in (2, 5):
        step_levels = _levels_for_step(spec, step)
        _, step_points, _ = _sweep(spec, step_levels)
        step_unique = sorted(
            {(f1, f2) for f1, f2, _ in step_points},
            key=lambda point: (point[0], point[1]),
        )
        step_front = _candidate_front(step_unique)
        step_keys = {(round(f1, 9), round(f2, 9)) for f1, f2 in step_front}
        step_missing = sorted(oracle_vectors - step_keys)
        sensitivity.append(
            EpsilonSensitivity(
                step=step,
                levels=len(step_levels),
                feasible_levels=len(step_points),
                candidate_front_size=len(step_front),
                precision=(
                    len(step_keys & oracle_vectors) / len(step_keys)
                    if step_keys
                    else 0.0
                ),
                recall=(
                    len(step_keys & oracle_vectors) / len(oracle_vectors)
                    if oracle_vectors
                    else 1.0
                ),
                exact_match=step_keys == oracle_vectors,
                missing_vectors=[
                    ParetoVector(f1=f1, f2=f2) for f1, f2 in step_missing
                ],
            )
        )
    comparison = FrontComparison(
        candidate_front_size=len(front_vectors),
        oracle_front_size=oracle.front_size,
        precision=precision,
        recall=recall,
        exact_match=candidate_keys == oracle_vectors,
        missing_vectors=[ParetoVector(f1=f1, f2=f2) for f1, f2 in missing],
        extra_vectors=[ParetoVector(f1=f1, f2=f2) for f1, f2 in extra],
        dominated_candidates=[ParetoVector(f1=f1, f2=f2) for f1, f2 in dominated],
    )
    reference = (float(spec.reference_point[0]), float(spec.reference_point[1]))
    hypervolume = hypervolume_2d(front_vectors, reference)
    coverage = EpsilonCoverage(
        epsilon_min=float(levels[0]),
        epsilon_max=float(levels[-1]),
        epsilon_count=len(levels),
        successful_solves=len(feasible_points),
        infeasible_levels=sum(1 for solution in raw if solution.status == "infeasible"),
        error_levels=sum(1 for solution in raw if solution.status == "error"),
    )
    completed = datetime.now(UTC).replace(microsecond=0)
    result = MultiObjectiveResult(
        experiment_id=spec.experiment_id,
        problem_id=spec.problem_id,
        instance_hash=spec.instance_hash,
        spec_hash=spec.spec_hash,
        status="succeeded",
        method=spec.method,
        solver=spec.solver,
        solver_version=metadata.version("ortools"),
        primary_objective_id=spec.primary_objective_id,
        epsilon_objective_id=spec.epsilon_objective_id,
        epsilon_coverage=coverage,
        reference_point=list(spec.reference_point),
        raw_epsilon_solutions=raw,
        candidate_pareto_front=[
            ParetoVector(f1=f1, f2=f2) for f1, f2 in front_vectors
        ],
        candidate_pareto_set=ordered_solutions,
        candidate_front_size=len(front_vectors),
        candidate_set_size=len(ordered_solutions),
        comparison=comparison,
        epsilon_sensitivity=sensitivity,
        hypervolume=hypervolume,
        hypervolume_gap=abs(hypervolume - oracle.hypervolume),
        runtime_total=time.perf_counter() - clock,
        solver_calls=solver_calls,
        git_commit=current_git_commit(),
        started_at=started,
        completed_at=completed,
        notes=[
            "No unique winner is selected: the result is a Pareto set of alternatives.",
            "Epsilon levels come only from the frozen instance and declared policy.",
            "Two-stage lexicographic solves fix maximum f1, then maximize f2.",
        ],
    )
    save_pareto_oracle(paths, spec.experiment_id, oracle)
    save_multi_objective_result(paths, result)
    _write_epsilon_solutions(paths, result)
    return result


def _write_epsilon_solutions(paths: ProjectPaths, result: MultiObjectiveResult) -> None:
    target = epsilon_solutions_file(paths, result.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "epsilon",
            "status",
            "f1",
            "f2",
            "feasible",
            "stage1_status",
            "stage2_status",
            "runtime_seconds",
            "decision_vector",
        ]
    )
    for solution in result.raw_epsilon_solutions:
        writer.writerow(
            [
                solution.epsilon,
                solution.status,
                "" if solution.f1 is None else solution.f1,
                "" if solution.f2 is None else solution.f2,
                solution.feasible,
                solution.stage1_status,
                solution.stage2_status,
                solution.runtime_seconds,
                ";".join(
                    f"{key}={value}"
                    for key, value in sorted(solution.decision_vector.items())
                ),
            ]
        )
    atomic_write_text(target, buffer.getvalue())


def build_multi_objective_record(
    paths: ProjectPaths,
    spec: MultiObjectiveSpec,
    result: MultiObjectiveResult,
    *,
    status: ExperimentStatus,
) -> MultiObjectiveExperimentRecord:
    return MultiObjectiveExperimentRecord(
        experiment_id=spec.experiment_id,
        project_id=paths.root.name,
        instance_hash=spec.instance_hash,
        spec_hash=spec.spec_hash,
        method=result.method,
        solver=result.solver,
        solver_version=result.solver_version,
        git_commit=result.git_commit,
        status=status,
    )


__all__ = [
    "MultiObjectiveError",
    "build_multi_objective_record",
    "run_multi_objective",
]
