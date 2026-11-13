import argparse
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
    load_optimization_spec,
    save_optimization_result,
)
from mmos.schemas.optimization import (
    EnumerationResult,
    OptimizationModel,
    OptimizationResult,
    OptimizationSolution,
    VariableType,
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
    if status == "optimal":
        gap = 0.0
    elif best_bound is not None and math.isfinite(best_bound):
        gap = abs(best_bound - reported_objective) / max(abs(reported_objective), 1e-9)
    return OptimizationSolution(
        method="ortools.cbc",
        status=status,
        objective_value=reported_objective,
        best_bound=best_bound,
        optimality_gap=gap,
        variable_values=values,
        feasible=feasible,
        constraint_violations=violations,
        runtime_seconds=runtime,
        solver="CBC",
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

        def score(variable_name: str) -> float:
            coefficient = model.objective.get(variable_name, 0.0)
            if model.sense == "minimize":
                coefficient = -coefficient
            return coefficient / consumption(variable_name)

        ordered = sorted(
            (variable for variable in model.variables if variable.type is VariableType.BINARY),
            key=lambda variable: score(variable.name),
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


def run_optimization(paths: ProjectPaths, experiment_id: str) -> OptimizationResult:
    spec = load_optimization_spec(paths, experiment_id)
    solver_solution = solve_ortools(spec.model, spec.time_limit_seconds)
    solutions = [solver_solution]
    if spec.baseline == "greedy_feasible":
        solutions.append(solve_greedy(spec.model))
    selected = solver_solution.method if solver_solution.feasible else None
    enumeration = enumerate_binary_model(spec.model)
    result = OptimizationResult(
        experiment_id=experiment_id,
        problem_id=spec.problem_id,
        model_name=spec.model.name,
        sense=spec.model.sense,
        solutions=solutions,
        selected_method=selected,
        enumeration=enumeration,
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
