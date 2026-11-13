import json
from pathlib import Path

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_record,
    load_optimization_result,
    load_optimization_spec,
)
from mmos.experiments.knapsack_benchmark import (
    enumerate_knapsack,
    generator_code_hash,
    load_manifest,
    methods_code_hash,
)
from mmos.experiments.knapsack_suite import _aggregate, load_benchmark_result
from mmos.experiments.optimization_tournament import load_optimization_tournament
from mmos.schemas.data import Severity
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-9


def _error(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.ERROR, message=message)


def _warning(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.WARNING, message=message)


def _close(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= _TOLERANCE + 1e-9 * abs(right)


def validate_benchmark_suite(
    paths: ProjectPaths,
    benchmark_run_id: str,
    suite_dir: Path,
) -> list[ValidationReport]:
    artifact_dir = paths.experiments_dir / "benchmarks" / benchmark_run_id
    result = load_benchmark_result(artifact_dir / "benchmark_result.json")
    manifest = load_manifest(suite_dir)

    integrity_issues: list[ValidationIssue] = []
    oracle_issues: list[ValidationIssue] = []
    aggregate_issues: list[ValidationIssue] = []

    if manifest.manifest_hash != result.manifest_hash:
        integrity_issues.append(
            _error("manifest_hash_mismatch", "suite result and manifest hashes differ")
        )
    if result.suite_id != manifest.suite_id:
        integrity_issues.append(
            _error(
                "suite_id_mismatch",
                f"result suite {result.suite_id}, manifest suite {manifest.suite_id}",
            )
        )
    if result.suite_seed != manifest.suite_seed:
        integrity_issues.append(
            _error(
                "suite_seed_mismatch",
                f"result seed {result.suite_seed}, manifest seed {manifest.suite_seed}",
            )
        )
    if result.methods_code_hash != methods_code_hash():
        integrity_issues.append(
            _error(
                "methods_code_hash_mismatch",
                "stored methods code hash differs from the current implementation",
            )
        )
    if result.generator_code_hash != generator_code_hash():
        integrity_issues.append(
            _error(
                "generator_code_hash_mismatch",
                "stored generator code hash differs from the current implementation",
            )
        )
    required_runtime = {"python_version", "platform", "ortools_version", "cbc_version"}
    missing_runtime = sorted(required_runtime - set(result.runtime))
    if missing_runtime:
        integrity_issues.append(
            _error(
                "missing_runtime_identifiers",
                f"recorded runtime is missing {missing_runtime}",
            )
        )
    if (
        result.n_instances != len(manifest.instances)
        or len(result.per_instance) != len(manifest.instances)
    ):
        integrity_issues.append(
            _error(
                "instance_count_mismatch",
                f"manifest has {len(manifest.instances)} instances, result declares "
                f"{result.n_instances} and carries {len(result.per_instance)} rows",
            )
        )
    if result.status != "complete" or result.missing_instances:
        integrity_issues.append(
            _error(
                "suite_incomplete",
                f"suite status {result.status}; missing {result.missing_instances}",
            )
        )

    manifest_by_id = {instance.instance_id: instance for instance in manifest.instances}
    per_instance = {instance.instance_id: instance for instance in result.per_instance}
    extra_rows = sorted(set(per_instance) - set(manifest_by_id))
    if extra_rows:
        integrity_issues.append(
            _error("unexpected_instance_result", f"result rows not in manifest: {extra_rows}")
        )
    rows_complete = all(instance_id in per_instance for instance_id in manifest_by_id)
    expected_experiments = len(manifest.instances) * 3
    expected_tournaments = len(manifest.instances)
    if len(result.experiment_ids) != expected_tournaments:
        integrity_issues.append(
            _error(
                "experiment_index_incomplete",
                f"manifest expects {expected_tournaments} experiment groups, result "
                f"indexes {len(result.experiment_ids)}",
            )
        )
    indexed_experiments = sum(len(ids) for ids in result.experiment_ids.values())
    if indexed_experiments != expected_experiments:
        integrity_issues.append(
            _error(
                "experiment_count_mismatch",
                f"manifest expects {expected_experiments} experiments, result indexes "
                f"{indexed_experiments}",
            )
        )
    if len(result.tournament_ids) != expected_tournaments:
        integrity_issues.append(
            _error(
                "tournament_count_mismatch",
                f"manifest expects {expected_tournaments} tournaments, result indexes "
                f"{len(result.tournament_ids)}",
            )
        )
    for instance_id in sorted(manifest_by_id):
        row = per_instance.get(instance_id)
        if row is None:
            integrity_issues.append(
                _error("missing_instance_result", f"no result row for {instance_id}")
            )
            continue
        expected_fields = {
            "candidate_experiment_id": f"exp-{manifest.suite_id}-{instance_id}-cbc",
            "greedy_experiment_id": f"exp-{manifest.suite_id}-{instance_id}-greedy",
            "enhanced_greedy_experiment_id": (
                f"exp-{manifest.suite_id}-{instance_id}-enhanced-greedy"
            ),
        }
        for field in (
            "candidate_experiment_id",
            "greedy_experiment_id",
            "enhanced_greedy_experiment_id",
            "tournament_id",
        ):
            if getattr(row, field) is None:
                integrity_issues.append(
                    _error(f"missing_{field}", f"{instance_id}: {field} is missing")
                )
        for field, expected in expected_fields.items():
            actual = getattr(row, field)
            if actual is not None and actual != expected:
                integrity_issues.append(
                    _error(
                        "experiment_id_mismatch",
                        f"{instance_id}: {field} is {actual}, expected {expected}",
                    )
                )
        indexed = result.experiment_ids.get(instance_id, [])
        if sorted(indexed) != sorted(expected_fields.values()):
            integrity_issues.append(
                _error(
                    "experiment_index_mismatch",
                    f"{instance_id}: indexed experiments {indexed} do not match the manifest",
                )
            )
        if row.tournament_id is not None and (
            result.tournament_ids.get(instance_id) != row.tournament_id
        ):
            integrity_issues.append(
                _error("tournament_index_mismatch", f"{instance_id}: tournament not indexed")
            )
        if row.instance_hash != manifest_by_id[instance_id].instance_hash:
            integrity_issues.append(
                _error("instance_hash_mismatch", f"{instance_id}: result hash differs")
            )
        candidate_available = row.candidate_experiment_id is not None
        if row.candidate_experiment_id is not None:
            try:
                spec = load_optimization_spec(paths, row.candidate_experiment_id)
                load_optimization_result(paths, row.candidate_experiment_id)
                record = load_experiment_record(paths, row.candidate_experiment_id)
            except FileNotFoundError:
                candidate_available = False
                integrity_issues.append(
                    _error("missing_experiment_artifact", f"{instance_id}: candidate artifacts")
                )
            else:
                if spec.instance_hash != row.instance_hash:
                    integrity_issues.append(
                        _error("candidate_instance_mismatch", f"{instance_id}")
                    )
                if record.validation_status.get("optimization") != "passed":
                    integrity_issues.append(
                        _error("candidate_not_validated", f"{instance_id}")
                    )
        if not candidate_available or row.candidate_experiment_id is None:
            continue

        oracle = enumerate_knapsack(manifest_by_id[instance_id])
        if not _close(oracle.verified_optimum, row.oracle_optimum):
            oracle_issues.append(
                _error(
                    "oracle_recompute_mismatch",
                    f"{instance_id}: stored {row.oracle_optimum}, recomputed "
                    f"{oracle.verified_optimum}",
                )
            )
        cbc_result = load_optimization_result(paths, row.candidate_experiment_id)
        cbc_solution = cbc_result.solutions[0]
        if not cbc_solution.feasible or cbc_solution.objective_value is None:
            oracle_issues.append(
                _error("candidate_infeasible", f"{instance_id}: CBC infeasible")
            )
        elif not _close(cbc_solution.objective_value, oracle.verified_optimum):
            oracle_issues.append(
                _error(
                    "oracle_disagreement",
                    f"{instance_id}: CBC {cbc_solution.objective_value}, oracle "
                    f"{oracle.verified_optimum}",
                )
            )
        instance = manifest_by_id[instance_id]
        assignment = cbc_solution.variable_values
        xs = [assignment.get(f"x_{index}") for index in range(len(instance.weights))]
        if any(value is None for value in xs):
            oracle_issues.append(
                _error("cbc_assignment_missing", f"{instance_id}: CBC assignment incomplete")
            )
        else:
            raw_values = [float(item) for item in xs if item is not None]
            rounded = [round(item) for item in raw_values]
            if any(
                abs(item - bit) > 1e-9 or bit not in (0, 1)
                for item, bit in zip(raw_values, rounded, strict=True)
            ):
                oracle_issues.append(
                    _error(
                        "cbc_assignment_not_binary",
                        f"{instance_id}: CBC assignment is not binary",
                    )
                )
            else:
                weight = sum(
                    weight * bit
                    for weight, bit in zip(instance.weights, rounded, strict=True)
                )
                value = sum(
                    value * bit
                    for value, bit in zip(instance.values, rounded, strict=True)
                )
                if weight > instance.capacity:
                    oracle_issues.append(
                        _error(
                            "cbc_assignment_over_capacity",
                            f"{instance_id}: CBC weight {weight} > capacity "
                            f"{instance.capacity}",
                        )
                    )
                if (
                    cbc_solution.objective_value is not None
                    and abs(value - cbc_solution.objective_value) > 1e-6
                ):
                    oracle_issues.append(
                        _error(
                            "cbc_assignment_objective_mismatch",
                            f"{instance_id}: assignment value {value}, reported "
                            f"{cbc_solution.objective_value}",
                        )
                    )
        if not row.cbc_verified:
            oracle_issues.append(
                _error("cbc_not_verified", f"{instance_id}: result marks CBC unverified")
            )

    recomputed_verified = sum(1 for row in per_instance.values() if row.cbc_verified)
    if result.candidate_verified_optimum_count != recomputed_verified:
        oracle_issues.append(
            _error(
                "verified_count_mismatch",
                f"stored {result.candidate_verified_optimum_count}, recomputed "
                f"{recomputed_verified}",
            )
        )
    expected_agreement = rows_complete and recomputed_verified == len(manifest.instances)
    if result.oracle_agreement != expected_agreement:
        oracle_issues.append(
            _error(
                "oracle_agreement_mismatch",
                f"stored {result.oracle_agreement}, recomputed {expected_agreement}",
            )
        )

    candidate_rows = []
    greedy_rows = []
    enhanced_rows = []
    wins = 0
    ties = 0
    losses = 0
    for instance_id in sorted(manifest_by_id):
        row = per_instance.get(instance_id)
        if row is None or not (
            row.candidate_experiment_id
            and row.greedy_experiment_id
            and row.enhanced_greedy_experiment_id
            and row.tournament_id
        ):
            continue
        oracle = enumerate_knapsack(manifest_by_id[instance_id]).verified_optimum
        cbc = load_optimization_result(paths, row.candidate_experiment_id).solutions[0]
        greedy = load_optimization_result(
            paths, row.greedy_experiment_id
        ).solutions[0]
        enhanced = load_optimization_result(
            paths, row.enhanced_greedy_experiment_id
        ).solutions[0]
        candidate_rows.append(
            (instance_id, cbc.objective_value, cbc.feasible, oracle)
        )
        greedy_rows.append(
            (instance_id, greedy.objective_value, greedy.feasible, oracle)
        )
        enhanced_rows.append(
            (instance_id, enhanced.objective_value, enhanced.feasible, oracle)
        )
        tournament = load_optimization_tournament(paths, row.tournament_id)
        if tournament.winner_status.value == "winner" and (
            tournament.winner_experiment_id == row.candidate_experiment_id
        ):
            wins += 1
        elif tournament.winner_status.value == "equivalent":
            ties += 1
        else:
            losses += 1

    recomputed = {
        "candidate": _aggregate("ortools.cbc", "candidate", candidate_rows),
        "greedy": _aggregate("greedy_feasible", "sanity_baseline", greedy_rows),
        "enhanced_greedy": _aggregate(
            "enhanced_greedy_feasible", "heuristic_baseline", enhanced_rows
        ),
    }
    for name in ("candidate", "greedy", "enhanced_greedy"):
        actual = getattr(result, name)
        expected = recomputed[name]
        for field in (
            "optimum_hits",
            "failures",
            "mean_absolute_regret",
            "median_absolute_regret",
            "max_absolute_regret",
            "mean_relative_regret",
            "median_relative_regret",
            "max_relative_regret",
            "worst_instance_id",
            "worst_relative_regret",
        ):
            left = getattr(actual, field)
            right = getattr(expected, field)
            if isinstance(left, float) or isinstance(right, float):
                matches = _close(left, right)
            else:
                matches = left == right
            if not matches:
                aggregate_issues.append(
                    _error(
                        "aggregate_mismatch",
                        f"{name}.{field}: stored {left}, recomputed {right}",
                    )
                )
    if result.candidate_win_count != wins:
        aggregate_issues.append(
            _error("win_count_mismatch", f"stored {result.candidate_win_count}, recomputed {wins}")
        )
    if result.candidate_tie_count != ties:
        aggregate_issues.append(
            _error("tie_count_mismatch", f"stored {result.candidate_tie_count}, recomputed {ties}")
        )
    if result.candidate_loss_count != losses:
        aggregate_issues.append(
            _error(
                "loss_count_mismatch",
                f"stored {result.candidate_loss_count}, recomputed {losses}",
            )
        )
    feasible_count = sum(1 for row in candidate_rows if row[2])
    if result.candidate_feasible_count != feasible_count:
        aggregate_issues.append(
            _error(
                "feasible_count_mismatch",
                f"stored {result.candidate_feasible_count}, recomputed {feasible_count}",
            )
        )
    if result.worst_case_instance != recomputed["greedy"].worst_instance_id:
        aggregate_issues.append(
            _error(
                "worst_case_mismatch",
                f"stored {result.worst_case_instance}, recomputed "
                f"{recomputed['greedy'].worst_instance_id}",
            )
        )

    reports = [
        build_report(
            "benchmark_integrity",
            benchmark_run_id,
            integrity_issues,
            evidence=[str(artifact_dir / "benchmark_result.json")],
        ),
        build_report(
            "benchmark_oracle",
            benchmark_run_id,
            oracle_issues,
            evidence=[str(artifact_dir / "oracle")],
        ),
        build_report(
            "benchmark_aggregate",
            benchmark_run_id,
            aggregate_issues,
            evidence=[str(artifact_dir / "benchmark_result.json")],
        ),
    ]
    output_dir = artifact_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        atomic_write_text(
            output_dir / f"{report.validator}.json",
            report.model_dump_json(indent=2) + "\n",
        )
    summary = {
        "benchmark_run_id": benchmark_run_id,
        "blocking": any(report.blocking for report in reports),
        "reports": {
            report.validator: {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
            for report in reports
        },
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    return reports
