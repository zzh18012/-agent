import secrets
import statistics
from datetime import UTC, datetime
from pathlib import Path

import structlog

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
    save_experiment_record,
    save_optimization_spec,
)
from mmos.experiments.knapsack_benchmark import (
    enumerate_knapsack,
    instance_model,
    load_manifest,
    write_manifest_copy,
)
from mmos.experiments.optimization_runner import (
    build_experiment_record,
    run_optimization,
)
from mmos.experiments.optimization_tournament import run_optimization_tournament
from mmos.schemas.benchmark import (
    BenchmarkSuiteResult,
    InstanceSuiteResult,
    KnapsackInstance,
    MethodAggregate,
)
from mmos.schemas.optimization import (
    ExperimentStatus,
    OptimizationResult,
    OptimizationSpec,
)
from mmos.tracking import (
    log_benchmark_suite,
    log_optimization_run,
    log_optimization_tournament,
)
from mmos.validation.optimization import validate_optimization_experiment

log = structlog.get_logger("mmos.benchmark")

_TOLERANCE = 1e-6


def _spec(
    experiment_id: str,
    instance: KnapsackInstance,
    model,
    solver: str,
    time_limit_seconds: float,
) -> OptimizationSpec:
    return OptimizationSpec(
        experiment_id=experiment_id,
        problem_id=instance.instance_id,
        model=model,
        solver=solver,
        baseline="none",
        time_limit_seconds=time_limit_seconds,
        instance_hash=instance.instance_hash,
    )


def _solution(result: OptimizationResult):
    return result.solutions[0] if result.solutions else None


def _run(paths: ProjectPaths, spec: OptimizationSpec):
    save_optimization_spec(paths, spec)
    result = run_optimization(paths, spec.experiment_id)
    record = build_experiment_record(paths, spec, result, status=ExperimentStatus.SUCCEEDED)
    mlflow = log_optimization_run(paths, spec, result, record)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_experiment_record(paths, record)
    return result, record, mlflow


def _relative_regret(oracle: float, objective: float | None, feasible: bool) -> float | None:
    if not feasible or objective is None or oracle <= 0:
        return None
    return max(0.0, oracle - objective) / oracle


def _aggregate(
    method: str,
    role: str,
    rows: list[tuple[str, float | None, bool, float]],
) -> MethodAggregate:
    hits = 0
    failures = 0
    absolute_regrets: list[float] = []
    relative_regrets: list[float] = []
    worst_instance: str | None = None
    worst_value: float | None = None
    for instance_id, objective, feasible, oracle in rows:
        hit = feasible and objective is not None and abs(objective - oracle) <= _TOLERANCE
        if hit:
            hits += 1
        else:
            failures += 1
        absolute = max(0.0, oracle - objective) if objective is not None else oracle
        absolute_regrets.append(absolute)
        relative = _relative_regret(oracle, objective, feasible)
        if relative is not None:
            relative_regrets.append(relative)
            if worst_value is None or relative > worst_value:
                worst_value = relative
                worst_instance = instance_id
    return MethodAggregate(
        method=method,
        role=role,
        optimum_hits=hits,
        failures=failures,
        mean_absolute_regret=statistics.fmean(absolute_regrets),
        median_absolute_regret=statistics.median(absolute_regrets),
        max_absolute_regret=max(absolute_regrets),
        mean_relative_regret=(statistics.fmean(relative_regrets) if relative_regrets else None),
        median_relative_regret=(statistics.median(relative_regrets) if relative_regrets else None),
        max_relative_regret=max(relative_regrets) if relative_regrets else None,
        worst_instance_id=worst_instance,
        worst_relative_regret=worst_value,
        hit_rate=hits / len(rows) if rows else 0.0,
    )


def run_knapsack_suite(
    paths: ProjectPaths,
    suite_dir: Path,
    *,
    time_limit_seconds: float = 10.0,
) -> BenchmarkSuiteResult:
    manifest = load_manifest(suite_dir)
    started = datetime.now(UTC).replace(microsecond=0)
    run_id = f"benchmark-{secrets.token_hex(4)}"
    artifact_dir = paths.experiments_dir / "benchmarks" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "oracle").mkdir(parents=True, exist_ok=True)
    write_manifest_copy(manifest, artifact_dir / "benchmark_manifest.json")

    per_instance: list[InstanceSuiteResult] = []
    missing: list[str] = []
    candidate_rows: list[tuple[str, float | None, bool, float]] = []
    greedy_rows: list[tuple[str, float | None, bool, float]] = []
    enhanced_rows: list[tuple[str, float | None, bool, float]] = []
    experiment_ids: dict[str, list[str]] = {}
    tournament_ids: dict[str, str] = {}
    mlflow_run_ids: dict[str, str] = {}
    candidate_wins = 0
    candidate_ties = 0
    candidate_losses = 0

    for instance in manifest.instances:
        oracle = enumerate_knapsack(instance)
        model = instance_model(instance)
        cbc_id = f"exp-{manifest.suite_id}-{instance.instance_id}-cbc"
        greedy_id = f"exp-{manifest.suite_id}-{instance.instance_id}-greedy"
        enhanced_id = f"exp-{manifest.suite_id}-{instance.instance_id}-enhanced-greedy"
        instance_result = InstanceSuiteResult(
            instance_id=instance.instance_id,
            instance_hash=instance.instance_hash,
            tags=instance.tags,
            n_items=len(instance.weights),
            capacity=instance.capacity,
            oracle_optimum=oracle.verified_optimum,
        )
        try:
            cbc_result, cbc_record, cbc_mlflow = _run(
                paths, _spec(cbc_id, instance, model, "ortools.cbc", time_limit_seconds)
            )
            greedy_result, _, greedy_mlflow = _run(
                paths, _spec(greedy_id, instance, model, "greedy", time_limit_seconds)
            )
            enhanced_result, _, enhanced_mlflow = _run(
                paths,
                _spec(enhanced_id, instance, model, "enhanced_greedy", time_limit_seconds),
            )

            cbc_solution = _solution(cbc_result)
            greedy_solution = _solution(greedy_result)
            enhanced_solution = _solution(enhanced_result)
            cbc_objective = cbc_solution.objective_value if cbc_solution else None
            greedy_objective = greedy_solution.objective_value if greedy_solution else None
            enhanced_objective = enhanced_solution.objective_value if enhanced_solution else None
            cbc_feasible = bool(cbc_solution and cbc_solution.feasible)
            greedy_feasible = bool(greedy_solution and greedy_solution.feasible)
            enhanced_feasible = bool(enhanced_solution and enhanced_solution.feasible)
            cbc_verified = (
                cbc_feasible
                and cbc_objective is not None
                and abs(cbc_objective - oracle.verified_optimum) <= _TOLERANCE
            )
            oracle = oracle.model_copy(update={"matches_candidate": cbc_verified})

            reports = validate_optimization_experiment(paths, cbc_id)
            blocking = any(report.blocking for report in reports)
            cbc_record = cbc_record.model_copy(
                update={
                    "status": (
                        ExperimentStatus.INVALID if blocking else ExperimentStatus.VALIDATED
                    ),
                    "validation_status": {
                        report.validator: report.status.value for report in reports
                    },
                    "linked_experiment_ids": [greedy_id, enhanced_id],
                }
            )
            save_experiment_record(paths, cbc_record)

            tournament = run_optimization_tournament(paths, cbc_id, greedy_id)
            tournament_mlflow = log_optimization_tournament(paths, tournament)
            if tournament.winner_status.value == "winner" and (
                tournament.winner_experiment_id == cbc_id
            ):
                candidate_wins += 1
            elif tournament.winner_status.value == "equivalent":
                candidate_ties += 1
            else:
                candidate_losses += 1

            if blocking and instance.instance_id not in missing:
                missing.append(instance.instance_id)
            instance_result = instance_result.model_copy(
                update={
                    "cbc_objective": cbc_objective,
                    "greedy_objective": greedy_objective,
                    "enhanced_greedy_objective": enhanced_objective,
                    "greedy_relative_regret": _relative_regret(
                        oracle.verified_optimum, greedy_objective, greedy_feasible
                    ),
                    "enhanced_relative_regret": _relative_regret(
                        oracle.verified_optimum, enhanced_objective, enhanced_feasible
                    ),
                    "cbc_verified": cbc_verified,
                    "tournament_id": tournament.tournament_id,
                    "candidate_experiment_id": cbc_id,
                    "greedy_experiment_id": greedy_id,
                    "enhanced_greedy_experiment_id": enhanced_id,
                    "mlflow_run_ids": {
                        "cbc": cbc_mlflow.run_id,
                        "greedy": greedy_mlflow.run_id,
                        "enhanced_greedy": enhanced_mlflow.run_id,
                        "tournament": tournament_mlflow.run_id,
                    },
                    "notes": [] if not blocking else ["candidate validation failed"],
                }
            )
            experiment_ids[instance.instance_id] = [cbc_id, greedy_id, enhanced_id]
            tournament_ids[instance.instance_id] = tournament.tournament_id
            mlflow_run_ids[instance.instance_id] = cbc_mlflow.run_id
            candidate_rows.append(
                (instance.instance_id, cbc_objective, cbc_feasible, oracle.verified_optimum)
            )
            greedy_rows.append(
                (
                    instance.instance_id,
                    greedy_objective,
                    greedy_feasible,
                    oracle.verified_optimum,
                )
            )
            enhanced_rows.append(
                (
                    instance.instance_id,
                    enhanced_objective,
                    enhanced_feasible,
                    oracle.verified_optimum,
                )
            )
        except Exception as exc:
            missing.append(instance.instance_id)
            instance_result = instance_result.model_copy(
                update={"notes": [f"instance execution failed: {exc}"]}
            )
            candidate_rows.append((instance.instance_id, None, False, oracle.verified_optimum))
            greedy_rows.append((instance.instance_id, None, False, oracle.verified_optimum))
            enhanced_rows.append((instance.instance_id, None, False, oracle.verified_optimum))
        atomic_write_text(
            artifact_dir / "oracle" / f"{instance.instance_id}.json",
            oracle.model_dump_json(indent=2) + "\n",
        ) if (artifact_dir / "oracle").exists() else (
            (artifact_dir / "oracle").mkdir(parents=True, exist_ok=True),
            atomic_write_text(
                artifact_dir / "oracle" / f"{instance.instance_id}.json",
                oracle.model_dump_json(indent=2) + "\n",
            ),
        )
        per_instance.append(instance_result)

    candidate = _aggregate("ortools.cbc", "candidate", candidate_rows)
    greedy = _aggregate("greedy_feasible", "sanity_baseline", greedy_rows)
    enhanced = _aggregate("enhanced_greedy_feasible", "heuristic_baseline", enhanced_rows)
    candidate_feasible_count = sum(row[2] for row in candidate_rows)
    candidate_verified_count = sum(1 for instance in per_instance if instance.cbc_verified)
    oracle_agreement = all(instance.cbc_verified for instance in per_instance)
    completed = datetime.now(UTC).replace(microsecond=0)
    status = "complete" if not missing else "incomplete"
    result = BenchmarkSuiteResult(
        benchmark_run_id=run_id,
        suite_id=manifest.suite_id,
        manifest_hash=manifest.manifest_hash,
        suite_seed=manifest.suite_seed,
        n_instances=len(manifest.instances),
        candidate=candidate,
        greedy=greedy,
        enhanced_greedy=enhanced,
        candidate_feasible_count=candidate_feasible_count,
        candidate_verified_optimum_count=candidate_verified_count,
        candidate_win_count=candidate_wins,
        candidate_tie_count=candidate_ties,
        candidate_loss_count=candidate_losses,
        oracle_agreement=oracle_agreement,
        worst_case_instance=greedy.worst_instance_id,
        per_instance=per_instance,
        status=status,
        missing_instances=missing,
        started_at=started,
        completed_at=completed,
        git_commit=current_git_commit(),
        experiment_ids=experiment_ids,
        tournament_ids=tournament_ids,
        mlflow_run_ids=mlflow_run_ids,
        notes=[] if status == "complete" else ["suite incomplete; see missing_instances"],
    )
    atomic_write_text(
        artifact_dir / "benchmark_result.json", result.model_dump_json(indent=2) + "\n"
    )
    suite_mlflow = log_benchmark_suite(paths, result, artifact_dir)
    result = result.model_copy(update={"suite_mlflow_run_id": suite_mlflow.run_id})
    atomic_write_text(
        artifact_dir / "benchmark_result.json", result.model_dump_json(indent=2) + "\n"
    )
    log.info(
        "benchmark_suite_completed",
        suite_id=manifest.suite_id,
        benchmark_run_id=run_id,
        status=status,
        oracle_agreement=oracle_agreement,
    )
    return result


def load_benchmark_result(path: Path) -> BenchmarkSuiteResult:
    return BenchmarkSuiteResult.model_validate_json(path.read_text(encoding="utf-8"))
