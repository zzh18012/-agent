import itertools
import json
from pathlib import Path

from mmos.core.project import create_project, save_optimization_spec
from mmos.evidence import freeze_optimization, load_evidence_index
from mmos.experiments.optimization_runner import run_optimization, solve_greedy, solve_ortools
from mmos.paper import render_markdown
from mmos.paper.optimization_builder import build_optimization_paper
from mmos.schemas.optimization import OptimizationSpec
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
