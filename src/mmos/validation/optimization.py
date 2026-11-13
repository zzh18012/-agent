import json

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_optimization_result,
    load_optimization_spec,
)
from mmos.experiments.optimization_runner import check_solution
from mmos.schemas.data import Severity
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-6


def validate_optimization(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = load_optimization_spec(paths, experiment_id)
    result = load_optimization_result(paths, experiment_id)
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, severity=Severity.WARNING, message=message))

    methods = {solution.method for solution in result.solutions}
    if result.selected_method is not None and result.selected_method not in methods:
        error("selected_method_unknown", "selected_method does not match any recorded solution")

    any_feasible = False
    for solution in result.solutions:
        if solution.status in {"optimal", "feasible"}:
            feasible, violations, recomputed = check_solution(spec.model, solution.variable_values)
            if not solution.feasible or not feasible:
                error(
                    "solution_infeasible",
                    f"{solution.method}: recorded feasible={solution.feasible}, "
                    f"recomputed feasible={feasible}",
                )
            if violations:
                error(
                    "constraint_violations",
                    f"{solution.method}: {'; '.join(violations[:5])}",
                )
            if solution.objective_value is None:
                error("missing_objective", f"{solution.method}: no objective value")
            elif abs(solution.objective_value - recomputed) > _TOLERANCE:
                error(
                    "objective_mismatch",
                    f"{solution.method}: reported {solution.objective_value}, "
                    f"recomputed {recomputed}",
                )
            if feasible:
                any_feasible = True
            if (
                solution.status == "optimal"
                and solution.optimality_gap not in (None, 0.0)
                and abs(solution.optimality_gap) > _TOLERANCE
            ):
                error(
                    "optimal_gap_nonzero",
                    f"{solution.method}: optimal status with gap {solution.optimality_gap}",
                )
            elif solution.status != "optimal" and solution.optimality_gap is not None:
                warning(
                    "nonzero_gap",
                    f"{solution.method}: feasible solution has optimality gap "
                    f"{solution.optimality_gap}",
                )
        elif solution.feasible:
            error(
                "status_feasible_mismatch",
                f"{solution.method}: status {solution.status} but marked feasible",
            )

    solver_solution = next(
        (solution for solution in result.solutions if solution.method == "ortools.cbc"),
        None,
    )
    enumeration = result.enumeration
    enumeration_optimum = enumeration.optimum_value if enumeration is not None else None
    if enumeration is not None and enumeration.status == "complete" and enumeration_optimum is None:
        error("enumeration_missing_optimum", "enumeration did not produce an optimum")
    elif (
        enumeration is not None
        and enumeration.status == "complete"
        and solver_solution is not None
        and solver_solution.objective_value is not None
        and enumeration_optimum is not None
        and abs(solver_solution.objective_value - enumeration_optimum) > _TOLERANCE
    ):
        error(
            "enumeration_mismatch",
            f"solver objective {solver_solution.objective_value} differs from "
            f"enumerated optimum {enumeration_optimum}",
        )

    if not any_feasible:
        error("no_feasible_solution", "no feasible solution was produced")
    if result.selected_method is None:
        error("no_selected_solution", "no feasible solver solution was selected")
    solver_solutions = [
        solution for solution in result.solutions if solution.method == "ortools.cbc"
    ]
    if not solver_solutions:
        error("missing_solver", "no OR-Tools solution was recorded")
    if spec.baseline == "greedy_feasible" and not any(
        solution.method == "greedy_feasible" for solution in result.solutions
    ):
        error("missing_baseline", "no greedy baseline was recorded")

    return build_report(
        "optimization",
        experiment_id,
        issues,
        evidence=[
            str(paths.experiment_dir(experiment_id) / "optimization_spec.json"),
            str(paths.experiment_dir(experiment_id) / "artifacts" / "optimization_result.json"),
        ],
        recommendations=[
            "recompute every constraint from the frozen variable values",
            "report the optimality gap for non-optimal solver statuses",
        ],
    )


def validate_optimization_experiment(
    paths: ProjectPaths, experiment_id: str
) -> list[ValidationReport]:
    report = validate_optimization(paths, experiment_id)
    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "optimization.json", report.model_dump_json(indent=2) + "\n")
    summary = {
        "experiment_id": experiment_id,
        "blocking": report.blocking,
        "reports": {
            report.validator: {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
        },
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    return [report]
