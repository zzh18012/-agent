import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import structlog
from ortools.linear_solver import pywraplp

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
    load_optimization_spec,
    save_experiment_record,
    save_optimization_result,
    save_optimization_spec,
)
from mmos.schemas.optimization import (
    EnumerationResult,
    ExperimentStatus,
    OptimizationExperimentRecord,
    OptimizationModel,
    OptimizationResult,
    OptimizationSolution,
    OptimizationSpec,
    VariableType,
    optimization_instance_hash,
)

log = structlog.get_logger("mmos.optimization")

Status = Literal["optimal", "feasible", "infeasible", "unbounded", "unknown"]

_TOLERANCE = 1e-6


def _activity(coefficients: dict[str, float], values: dict[str, float]) -> float:
    return sum(coefficient * values.get(name, 0.0) for name, coefficient in coefficients.items())


def objective_value(model: OptimizationModel, values: dict[str, float]) -> float:
    return _activity(model.objective, values)


def check_solution(
    model: OptimizationModel, values: dict[str, float]
) -> tuple[bool, list[str], float]:
    violations: list[str] = []
    for variable in model.variables:
        value = values.get(variable.name, variable.lower)
        if value < variable.lower - _TOLERANCE or value > variable.upper + _TOLERANCE:
            violations.append(
                f"{variable.name}={value} outside [{variable.lower}, {variable.upper}]"
            )
    for constraint in model.constraints:
        activity = _activity(constraint.coefficients, values)
        if constraint.lower is not None and activity < constraint.lower - _TOLERANCE:
            violations.append(f"{constraint.name}: activity {activity} < lower {constraint.lower}")
        if constraint.upper is not None and activity > constraint.upper + _TOLERANCE:
            violations.append(f"{constraint.name}: activity {activity} > upper {constraint.upper}")
    return not violations, violations, objective_value(model, values)


def solve_ortools(model: OptimizationModel, time_limit_seconds: float) -> OptimizationSolution:
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver is unavailable")

    variables = {}
    for variable in model.variables:
        if variable.type is VariableType.BINARY:
            variables[variable.name] = solver.IntVar(0.0, 1.0, variable.name)
        elif variable.type is VariableType.INTEGER:
            variables[variable.name] = solver.IntVar(variable.lower, variable.upper, variable.name)
        else:
            variables[variable.name] = solver.NumVar(variable.lower, variable.upper, variable.name)

    objective = solver.Objective()
    for name, coefficient in model.objective.items():
        objective.SetCoefficient(variables[name], coefficient)
    if model.sense == "maximize":
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    for constraint in model.constraints:
        expression = solver.Constraint(
            constraint.lower if constraint.lower is not None else -solver.infinity(),
            constraint.upper if constraint.upper is not None else solver.infinity(),
            constraint.name,
        )
        for name, coefficient in constraint.coefficients.items():
            expression.SetCoefficient(variables[name], coefficient)

    solver.SetTimeLimit(int(time_limit_seconds * 1000))
    started = time.perf_counter()
    status_code = solver.Solve()
    runtime = time.perf_counter() - started
    status_map: dict[int, Status] = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "unknown",
        pywraplp.Solver.NOT_SOLVED: "unknown",
    }
    status = status_map.get(status_code, "unknown")
    if status not in {"optimal", "feasible"}:
        return OptimizationSolution(
            method="ortools.cbc",
            status=status,
            feasible=False,
            runtime_seconds=runtime,
            solver="CBC",
        )

    values = {name: variable.solution_value() for name, variable in variables.items()}
    feasible, violations, _recomputed_objective = check_solution(model, values)
    reported_objective = float(objective.Value())
    best_bound = float(objective.BestBound()) if objective.BestBound() is not None else None
    gap = None
    absolute_gap = None
    if status == "optimal":
        gap = 0.0
        absolute_gap = 0.0
    elif best_bound is not None and math.isfinite(best_bound):
        absolute_gap = abs(best_bound - reported_objective)
        gap = absolute_gap / max(abs(reported_objective), 1e-9)
    return OptimizationSolution(
        method="ortools.cbc",
        status=status,
        objective_value=reported_objective,
        best_bound=best_bound,
        optimality_gap=gap,
        absolute_gap=absolute_gap,
        variable_values=values,
        feasible=feasible,
        constraint_violations=violations,
        runtime_seconds=runtime,
        solver="CBC",
        solver_version=version("ortools"),
        termination_condition=status,
    )


def solve_greedy(model: OptimizationModel) -> OptimizationSolution:
    values = {variable.name: variable.lower for variable in model.variables}
    if any(variable.type is VariableType.BINARY for variable in model.variables):

        def consumption(variable_name: str) -> float:
            total = 0.0
            for constraint in model.constraints:
                if constraint.upper is not None:
                    total += max(constraint.coefficients.get(variable_name, 0.0), 0.0)
            return max(total, 1e-9)

        index = {variable.name: position for position, variable in enumerate(model.variables)}

        def score(variable_name: str) -> float:
            coefficient = model.objective.get(variable_name, 0.0)
            if model.sense == "minimize":
                coefficient = -coefficient
            return coefficient / consumption(variable_name)

        ordered = sorted(
            (variable for variable in model.variables if variable.type is VariableType.BINARY),
            key=lambda variable: (
                score(variable.name),
                model.objective.get(variable.name, 0.0),
                -variable.upper,
                -index[variable.name],
            ),
            reverse=True,
        )
        for variable in ordered:
            candidate = dict(values)
            candidate[variable.name] = 1.0
            if check_solution(model, candidate)[0]:
                values = candidate

    feasible, violations, objective = check_solution(model, values)
    return OptimizationSolution(
        method="greedy_feasible",
        status="feasible" if feasible else "infeasible",
        objective_value=objective if feasible else None,
        variable_values=values,
        feasible=feasible,
        constraint_violations=violations,
        solver="greedy",
        solver_version="n/a",
        termination_condition="heuristic",
    )


def enumerate_binary_model(model: OptimizationModel, max_variables: int = 16) -> EnumerationResult:
    if not all(variable.type is VariableType.BINARY for variable in model.variables):
        return EnumerationResult(
            status="skipped", reason="enumeration is implemented for binary models only"
        )
    if len(model.variables) > max_variables:
        return EnumerationResult(
            status="skipped",
            reason=f"enumeration limited to {max_variables} binary variables",
        )
    names = [variable.name for variable in model.variables]
    best_value: float | None = None
    best_count = 0
    evaluated = 0
    for bits in itertools.product([0, 1], repeat=len(names)):
        values = {name: float(bit) for name, bit in zip(names, bits, strict=True)}
        feasible, _, objective = check_solution(model, values)
        if not feasible:
            continue
        evaluated += 1
        if best_value is None or (
            objective > best_value if model.sense == "maximize" else objective < best_value
        ):
            best_value = objective
            best_count = 1
        elif abs(objective - best_value) <= _TOLERANCE:
            best_count += 1
    return EnumerationResult(
        status="complete",
        evaluated_subsets=2 ** len(names),
        optimum_value=best_value,
        optimal_subsets=best_count,
    )


def _config_hash(spec) -> str:
    return hashlib.sha256(spec.model_dump_json().encode("utf-8")).hexdigest()


def build_experiment_record(
    paths: ProjectPaths,
    spec,
    result: OptimizationResult,
    *,
    status: ExperimentStatus = ExperimentStatus.SUCCEEDED,
    linked_experiment_ids: list[str] | None = None,
    validation_status: dict[str, str] | None = None,
    mlflow_run_id: str | None = None,
) -> OptimizationExperimentRecord:
    artifacts = [
        f"experiments/{spec.experiment_id}/optimization_spec.json",
        f"experiments/{spec.experiment_id}/artifacts/optimization_result.json",
    ]
    run_record = paths.experiment_dir(spec.experiment_id) / "run_record.json"
    if run_record.exists():
        artifacts.append(f"experiments/{spec.experiment_id}/run_record.json")
    solver_log = paths.experiment_dir(spec.experiment_id) / "solver_log.txt"
    if solver_log.exists():
        artifacts.append(f"experiments/{spec.experiment_id}/solver_log.txt")
    record = OptimizationExperimentRecord(
        experiment_id=spec.experiment_id,
        project_id=paths.root.name,
        created_at=result.generated_at,
        status=status,
        instance_hash=result.instance_hash or spec.instance_hash or "",
        config_hash=_config_hash(spec),
        git_commit=current_git_commit(),
        solver=spec.solver,
        solver_version=result.solver_version,
        solver_backend=result.solver_backend,
        mlflow_run_id=mlflow_run_id,
        artifacts=artifacts,
        validation_status=validation_status or {},
        linked_experiment_ids=linked_experiment_ids or [],
    )
    save_experiment_record(paths, record)
    return record


def create_baseline_experiment(
    paths: ProjectPaths, spec, result: OptimizationResult
) -> tuple[OptimizationSpec, OptimizationResult]:
    baseline_id = f"{spec.experiment_id}-baseline-greedy"
    instance_hash = result.instance_hash or spec.instance_hash
    baseline_spec = spec.model_copy(
        update={
            "experiment_id": baseline_id,
            "solver": "greedy",
            "instance_hash": instance_hash,
        }
    )
    save_optimization_spec(paths, baseline_spec)
    greedy = next(solution for solution in result.solutions if solution.method == "greedy_feasible")
    baseline_result = OptimizationResult(
        experiment_id=baseline_id,
        problem_id=spec.problem_id,
        model_name=spec.model.name,
        sense=spec.model.sense,
        solutions=[greedy],
        selected_method=greedy.method if greedy.feasible else None,
        instance_hash=instance_hash,
        solver_backend="greedy",
        solver_version="n/a",
        generated_at=result.generated_at,
    )
    save_optimization_result(paths, baseline_result)
    atomic_write_text(
        paths.experiment_dir(baseline_id) / "run_record.json",
        json.dumps(
            {
                "experiment_id": baseline_id,
                "baseline": "greedy_feasible",
                "python_version": sys.version.split()[0],
                "created_at": result.generated_at.isoformat(),
            },
            indent=2,
        )
        + "\n",
    )
    atomic_write_text(
        paths.experiment_dir(baseline_id) / "solver_log.txt",
        "solver=greedy_feasible\n"
        f"objective_value={greedy.objective_value}\n"
        f"feasible={greedy.feasible}\n"
        "note=deterministic greedy heuristic; no solver log available\n",
    )
    return baseline_spec, baseline_result


def comparison_diagnostics(spec_a, spec_b) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if spec_a.instance_hash != spec_b.instance_hash:
        return False, ["instance hash differs"]
    if spec_a.model.sense != spec_b.model.sense:
        return False, ["objective sense differs"]
    if spec_a.model.objective != spec_b.model.objective:
        return False, ["objective definition differs"]
    constraints_a = sorted(
        (constraint.model_dump(mode="json") for constraint in spec_a.model.constraints),
        key=lambda item: item["name"],
    )
    constraints_b = sorted(
        (constraint.model_dump(mode="json") for constraint in spec_b.model.constraints),
        key=lambda item: item["name"],
    )
    if constraints_a != constraints_b:
        return False, ["hard constraints differ"]
    if spec_a.time_limit_seconds != spec_b.time_limit_seconds:
        reasons.append("time limits differ; disclose when comparing")
    return True, reasons


def solve_enhanced_greedy(model: OptimizationModel) -> OptimizationSolution:
    """Best feasible solution among ratio greedy and a single-item solution."""
    greedy = solve_greedy(model)
    best_single_values = {variable.name: variable.lower for variable in model.variables}
    best_single_objective: float | None = None
    best_single_feasible = False
    for variable in model.variables:
        if variable.type is not VariableType.BINARY:
            continue
        values = dict(best_single_values)
        values[variable.name] = 1.0
        feasible, _, objective = check_solution(model, values)
        if not feasible:
            continue
        if best_single_objective is None or (
            objective > best_single_objective
            if model.sense == "maximize"
            else objective < best_single_objective
        ):
            best_single_objective = objective
            best_single_values = values
            best_single_feasible = True
    if greedy.feasible and greedy.objective_value is not None:
        greedy_is_better = True
        if best_single_feasible and best_single_objective is not None:
            greedy_is_better = (
                greedy.objective_value >= best_single_objective
                if model.sense == "maximize"
                else greedy.objective_value <= best_single_objective
            )
        if greedy_is_better:
            return greedy.model_copy(
                update={
                    "method": "enhanced_greedy_feasible",
                    "solver": "enhanced_greedy",
                    "solver_version": "n/a",
                }
            )
    if not best_single_feasible:
        return OptimizationSolution(
            method="enhanced_greedy_feasible",
            status="infeasible",
            feasible=False,
            solver="enhanced_greedy",
            solver_version="n/a",
            termination_condition="heuristic",
        )
    return OptimizationSolution(
        method="enhanced_greedy_feasible",
        status="feasible",
        objective_value=best_single_objective,
        variable_values=best_single_values,
        feasible=True,
        solver="enhanced_greedy",
        solver_version="n/a",
        termination_condition="heuristic",
    )


def run_optimization(paths: ProjectPaths, experiment_id: str) -> OptimizationResult:
    spec = load_optimization_spec(paths, experiment_id)
    if spec.instance_hash is None:
        spec = spec.model_copy(update={"instance_hash": optimization_instance_hash(spec.model)})
        save_optimization_spec(paths, spec)
    if spec.solver == "ortools.cbc":
        solver_solution = solve_ortools(spec.model, spec.time_limit_seconds)
        solutions = [solver_solution]
        if spec.baseline == "greedy_feasible":
            solutions.append(solve_greedy(spec.model))
        selected = solver_solution.method if solver_solution.feasible else None
        enumeration = enumerate_binary_model(spec.model)
        if (
            enumeration.status == "complete"
            and enumeration.optimum_value is not None
            and solver_solution.objective_value is not None
        ):
            enumeration = enumeration.model_copy(
                update={
                    "matches_solver_result": abs(
                        enumeration.optimum_value - solver_solution.objective_value
                    )
                    <= _TOLERANCE
                }
            )
        solver_backend = "CBC"
        solver_version = solver_solution.solver_version
    elif spec.solver == "greedy":
        solver_solution = solve_greedy(spec.model)
        solutions = [solver_solution]
        selected = solver_solution.method if solver_solution.feasible else None
        enumeration = None
        solver_backend = "greedy"
        solver_version = "n/a"
    elif spec.solver == "enhanced_greedy":
        solver_solution = solve_enhanced_greedy(spec.model)
        solutions = [solver_solution]
        selected = solver_solution.method if solver_solution.feasible else None
        enumeration = None
        solver_backend = "enhanced_greedy"
        solver_version = "n/a"
    else:
        raise NotImplementedError(f"unsupported solver {spec.solver!r}")

    result = OptimizationResult(
        experiment_id=experiment_id,
        problem_id=spec.problem_id,
        model_name=spec.model.name,
        sense=spec.model.sense,
        solutions=solutions,
        selected_method=selected,
        enumeration=enumeration,
        instance_hash=spec.instance_hash,
        solver_backend=solver_backend,
        solver_version=solver_version,
        generated_at=datetime.now(UTC).replace(microsecond=0),
    )
    save_optimization_result(paths, result)
    run_record = {
        "experiment_id": experiment_id,
        "ortools_version": version("ortools"),
        "python_version": sys.version.split()[0],
        "created_at": result.generated_at.isoformat(),
    }
    atomic_write_text(
        paths.experiment_dir(experiment_id) / "run_record.json",
        json.dumps(run_record, indent=2) + "\n",
    )
    atomic_write_text(
        paths.experiment_dir(experiment_id) / "solver_log.txt",
        "\n".join(
            [
                f"solver={spec.solver}",
                f"solver_version={solver_solution.solver_version}",
                f"solver_backend={solver_solution.solver}",
                f"time_limit_seconds={spec.time_limit_seconds}",
                f"status={solver_solution.status}",
                f"termination_condition={solver_solution.termination_condition}",
                f"objective_value={solver_solution.objective_value}",
                f"best_bound={solver_solution.best_bound}",
                f"absolute_gap={solver_solution.absolute_gap}",
                f"relative_gap={solver_solution.optimality_gap}",
                f"runtime_seconds={solver_solution.runtime_seconds}",
                f"feasible={solver_solution.feasible}",
                "note=CBC native console output is not exposed by the OR-Tools "
                "pywraplp wrapper; this log records solver-returned metadata.",
            ]
        )
        + "\n",
    )
    log.info(
        "optimization_completed",
        experiment_id=experiment_id,
        status=solver_solution.status,
        objective=solver_solution.objective_value,
        gap=solver_solution.optimality_gap,
        feasible=solver_solution.feasible,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an optimization experiment.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)
    paths = ProjectPaths(Path(args.project).resolve())
    result = run_optimization(paths, args.experiment_id)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
