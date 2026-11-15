import json

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    epsilon_solutions_file,
    load_multi_objective_record,
    load_multi_objective_result,
    load_multi_objective_spec,
    load_pareto_oracle,
    multi_objective_result_file,
    multi_objective_spec_file,
    pareto_oracle_file,
)
from mmos.experiments.multi_objective_oracle import enumerate_pareto_oracle
from mmos.experiments.pareto_utils import hypervolume_2d
from mmos.schemas.data import Severity
from mmos.schemas.multi_objective import MultiObjectiveSpec, ParetoVector
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-9


def _error(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.ERROR, message=message)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE + 1e-9 * abs(right)


def _dominates(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return (
        left[0] >= right[0]
        and left[1] >= right[1]
        and (left[0] > right[0] or left[1] > right[1])
    )


def _vector_key(vector: ParetoVector) -> tuple[int, int]:
    return (round(vector.f1), round(vector.f2))


def _validate_decision(
    spec: MultiObjectiveSpec, decision: dict[str, int]
) -> tuple[float, float] | None:
    expected_ids = {item.project_id for item in spec.items}
    if set(decision) != expected_ids:
        return None
    if any(value not in (0, 1) for value in decision.values()):
        return None
    cost = sum(
        item.cost * decision[item.project_id] for item in spec.items
    )
    if cost > spec.budget + _TOLERANCE:
        return None
    f1 = sum(
        item.benefits[spec.primary_objective_id] * decision[item.project_id]
        for item in spec.items
    )
    f2 = sum(
        item.benefits[spec.epsilon_objective_id] * decision[item.project_id]
        for item in spec.items
    )
    return float(f1), float(f2)


def validate_multi_objective_experiment(
    paths: ProjectPaths, experiment_id: str
) -> list[ValidationReport]:
    spec = load_multi_objective_spec(paths, experiment_id)
    result = load_multi_objective_result(paths, experiment_id)
    oracle = load_pareto_oracle(paths, experiment_id)
    record = load_multi_objective_record(paths, experiment_id)
    issues: list[ValidationIssue] = []

    if spec.instance_hash != spec.computed_instance_hash():
        issues.append(_error("instance_hash_mismatch", "instance hash does not match"))
    if spec.spec_hash != spec.computed_spec_hash():
        issues.append(_error("spec_hash_mismatch", "spec hash does not match"))
    if result.instance_hash != spec.instance_hash:
        issues.append(_error("result_instance_hash_mismatch", "result instance hash differs"))
    if result.spec_hash != spec.spec_hash:
        issues.append(_error("result_spec_hash_mismatch", "result spec hash differs"))
    if record.instance_hash != spec.instance_hash or record.spec_hash != spec.spec_hash:
        issues.append(_error("record_hash_mismatch", "record hashes differ"))
    if record.task_family != "multi_objective":
        issues.append(_error("task_family_mismatch", f"record task is {record.task_family}"))
    if not record.mlflow_run_id:
        issues.append(_error("missing_mlflow_run", "record has no MLflow run id"))
    if result.solver != spec.solver or result.method != spec.method:
        issues.append(_error("solver_method_mismatch", "result solver/method differs"))
    if result.solver_version != record.solver_version:
        issues.append(_error("solver_version_mismatch", "solver version differs"))
    if result.primary_objective_id != spec.primary_objective_id or (
        result.epsilon_objective_id != spec.epsilon_objective_id
    ):
        issues.append(_error("objective_mismatch", "result objectives differ from spec"))

    expected_levels = [float(level) for level in spec.epsilon_levels()]
    actual_levels = [solution.epsilon for solution in result.raw_epsilon_solutions]
    if actual_levels != expected_levels:
        issues.append(
            _error(
                "epsilon_schedule_mismatch",
                "epsilon solve schedule does not match the frozen instance/policy",
            )
        )
    if result.epsilon_coverage.epsilon_count != len(expected_levels):
        issues.append(_error("epsilon_coverage_mismatch", "epsilon count differs"))
    successful = sum(
        1 for solve in result.raw_epsilon_solutions if solve.status == "optimal"
    )
    infeasible = sum(
        1 for solve in result.raw_epsilon_solutions if solve.status == "infeasible"
    )
    errors = sum(
        1 for solve in result.raw_epsilon_solutions if solve.status == "error"
    )
    if (
        result.epsilon_coverage.successful_solves != successful
        or result.epsilon_coverage.infeasible_levels != infeasible
        or result.epsilon_coverage.error_levels != errors
    ):
        issues.append(_error("epsilon_coverage_mismatch", "epsilon outcome counts differ"))

    seen_decisions: set[tuple[tuple[str, int], ...]] = set()
    candidate_vectors: set[tuple[float, float]] = set()
    for solution in result.candidate_pareto_set:
        label = f"candidate decision {solution.decision_vector}"
        recomputed = _validate_decision(spec, solution.decision_vector)
        if recomputed is None:
            issues.append(_error("candidate_infeasible", label))
            continue
        if not _close(recomputed[0], solution.f1) or not _close(
            recomputed[1], solution.f2
        ):
            issues.append(_error("objective_recompute_mismatch", label))
        candidate_vectors.add((solution.f1, solution.f2))
        key = tuple(sorted(solution.decision_vector.items()))
        if key in seen_decisions:
            issues.append(_error("duplicate_candidate_decision", label))
        seen_decisions.add(key)

    declared_front = {
        _vector_key(vector) for vector in result.candidate_pareto_front
    }
    if len(declared_front) != len(result.candidate_pareto_front):
        issues.append(_error("duplicate_front_vector", "candidate front has duplicates"))
    if declared_front != {
        (round(f1), round(f2)) for f1, f2 in candidate_vectors
    }:
        issues.append(_error("front_set_mismatch", "front and candidate decisions differ"))
    if len(result.candidate_pareto_front) != result.candidate_front_size:
        issues.append(_error("front_size_mismatch", "candidate front size differs"))
    if len(result.candidate_pareto_set) != result.candidate_set_size:
        issues.append(_error("set_size_mismatch", "candidate set size differs"))

    vectors = [(vector.f1, vector.f2) for vector in result.candidate_pareto_front]
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            if _dominates(left, right) or _dominates(right, left):
                issues.append(
                    _error("candidate_front_dominated", f"candidate vectors {left}, {right}")
                )

    recomputed_oracle = enumerate_pareto_oracle(spec)
    if oracle.instance_hash != spec.instance_hash:
        issues.append(_error("oracle_instance_hash_mismatch", "oracle instance hash differs"))
    if oracle.search_space_size != 2 ** len(spec.items):
        issues.append(_error("oracle_search_space", "oracle search space is not 2^n"))
    if oracle.search_space_size != recomputed_oracle.search_space_size:
        issues.append(_error("oracle_search_space", "search space differs"))
    if oracle.feasible_solution_count != recomputed_oracle.feasible_solution_count:
        issues.append(_error("oracle_mismatch", "feasible solution count differs"))
    if oracle.pareto_objective_vectors != recomputed_oracle.pareto_objective_vectors:
        issues.append(_error("oracle_mismatch", "oracle front differs"))
    if oracle.pareto_decision_vectors != recomputed_oracle.pareto_decision_vectors:
        issues.append(_error("oracle_mismatch", "oracle Pareto set differs"))
    if oracle.front_size != len(oracle.pareto_objective_vectors):
        issues.append(_error("oracle_mismatch", "oracle front size differs"))
    if oracle.set_size != len(oracle.pareto_decision_vectors):
        issues.append(_error("oracle_mismatch", "oracle set size differs"))

    oracle_vectors = {
        _vector_key(vector) for vector in oracle.pareto_objective_vectors
    }
    candidate_keys = declared_front
    missing = oracle_vectors - candidate_keys
    extra = candidate_keys - oracle_vectors
    if missing:
        issues.append(
            _error(
                "missing_pareto_point",
                f"candidate front misses {sorted(missing)}",
            )
        )
    if extra:
        issues.append(
            _error(
                "false_pareto_point",
                f"candidate front has non-oracle vectors {sorted(extra)}",
            )
        )
    for other in oracle.pareto_objective_vectors:
        for f1, f2 in vectors:
            if _dominates((other.f1, other.f2), (f1, f2)):
                issues.append(
                    _error(
                        "dominated_candidate",
                        f"candidate vector {(f1, f2)} is dominated by oracle "
                        f"{(other.f1, other.f2)}",
                    )
                )
    comparison = result.comparison
    precision = (
        len(candidate_keys & oracle_vectors) / len(candidate_keys)
        if candidate_keys
        else 0.0
    )
    recall = (
        len(candidate_keys & oracle_vectors) / len(oracle_vectors)
        if oracle_vectors
        else 1.0
    )
    exact = candidate_keys == oracle_vectors
    if comparison is None:
        issues.append(_error("missing_comparison", "result comparison is missing"))
    else:
        if not _close(comparison.precision, precision) or not _close(
            comparison.recall, recall
        ):
            issues.append(_error("comparison_mismatch", "precision/recall differs"))
        if comparison.exact_match != exact:
            issues.append(_error("comparison_mismatch", "exact-match flag differs"))
        if {_vector_key(vector) for vector in comparison.missing_vectors} != missing:
            issues.append(_error("comparison_mismatch", "missing vectors differ"))
        if {_vector_key(vector) for vector in comparison.extra_vectors} != extra:
            issues.append(_error("comparison_mismatch", "extra vectors differ"))
    if not exact:
        issues.append(_error("pareto_front_mismatch", "candidate front != oracle front"))
    if [item.step for item in result.epsilon_sensitivity] != [2, 5]:
        issues.append(_error("sensitivity_missing", "step-2/step-5 sensitivity is missing"))
    for item in result.epsilon_sensitivity:
        maximum = int(
            sum(problem_item.benefits[spec.epsilon_objective_id] for problem_item in spec.items)
        )
        expected_levels = list(range(0, maximum + 1, item.step))
        if not expected_levels or expected_levels[-1] != maximum:
            expected_levels.append(maximum)
        if item.levels != len(expected_levels):
            issues.append(_error("sensitivity_mismatch", "sensitivity levels differ"))
        if item.candidate_front_size > oracle.front_size:
            issues.append(
                _error("sensitivity_mismatch", "sensitivity front exceeds oracle front")
            )
        if not 0.0 <= item.precision <= 1.0 or not 0.0 <= item.recall <= 1.0:
            issues.append(_error("sensitivity_mismatch", "sensitivity precision/recall"))
        missing_sensitivity = {
            _vector_key(vector) for vector in item.missing_vectors
        }
        if not missing_sensitivity <= oracle_vectors:
            issues.append(
                _error("sensitivity_mismatch", "sensitivity missing vectors not in oracle")
            )
        if item.exact_match != (not missing_sensitivity):
            issues.append(_error("sensitivity_mismatch", "sensitivity exact flag differs"))

    reference = (float(spec.reference_point[0]), float(spec.reference_point[1]))
    candidate_hv = hypervolume_2d(vectors, reference)
    oracle_hv = hypervolume_2d(
        [(vector.f1, vector.f2) for vector in oracle.pareto_objective_vectors],
        reference,
    )
    if not _close(result.hypervolume, candidate_hv):
        issues.append(_error("hypervolume_mismatch", "candidate hypervolume differs"))
    if not _close(oracle.hypervolume, oracle_hv):
        issues.append(_error("hypervolume_mismatch", "oracle hypervolume differs"))
    if result.hypervolume_gap is None or not _close(
        result.hypervolume_gap, abs(candidate_hv - oracle_hv)
    ):
        issues.append(_error("hypervolume_mismatch", "hypervolume gap differs"))
    if list(result.reference_point) != list(spec.reference_point):
        issues.append(_error("reference_point_mismatch", "reference point differs"))

    report = build_report(
        "multi_objective",
        experiment_id,
        issues,
        evidence=[
            str(multi_objective_spec_file(paths, experiment_id)),
            str(multi_objective_result_file(paths, experiment_id)),
            str(pareto_oracle_file(paths, experiment_id)),
            str(epsilon_solutions_file(paths, experiment_id)),
        ],
    )
    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "multi_objective.json", report.model_dump_json(indent=2) + "\n"
    )
    summary = {
        "experiment_id": experiment_id,
        "blocking": report.blocking,
        "candidate_front_size": len(candidate_keys),
        "oracle_front_size": len(oracle_vectors),
        "front_precision": precision,
        "front_recall": recall,
        "exact_match": exact,
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


__all__ = ["validate_multi_objective_experiment"]
