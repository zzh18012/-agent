import hashlib
import json
import statistics

from scipy import stats as scipy_stats

from mmos.core.project import (
    ProjectPaths,
    analytic_oracle_file,
    atomic_write_text,
    load_analytic_oracle,
    load_simulation_record,
    load_simulation_result,
    load_simulation_spec,
    simulation_result_file,
    simulation_spec_file,
)
from mmos.schemas.data import Severity
from mmos.schemas.simulation import SIMULATION_METRICS, SimulationSpec
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-9


def _error(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.ERROR, message=message)


def _warning(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.WARNING, message=message)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE + 1e-9 * abs(right)


def _ci(values: list[float], confidence: float) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    standard_error = statistics.stdev(values) / (len(values) ** 0.5)
    critical = float(scipy_stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def _expected_schedule(master_seed: int, n: int) -> list[dict[str, int]]:
    schedule = []
    for replication_id in range(n):
        digest = hashlib.sha256(
            f"{master_seed}:replication:{replication_id}".encode()
        ).digest()
        schedule.append(
            {
                "replication_id": replication_id,
                "replication_seed": int.from_bytes(digest[0:8], "big"),
                "arrival_seed": int.from_bytes(digest[8:16], "big"),
                "service_seed": int.from_bytes(digest[16:24], "big"),
            }
        )
    return schedule


def _schedule_hash(schedule: list[dict[str, int]]) -> str:
    encoded = json.dumps(schedule, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _oracle_values(spec: SimulationSpec) -> dict[str, float]:
    rho = spec.arrival_rate / spec.service_rate
    W = 1.0 / (spec.service_rate - spec.arrival_rate)
    Wq = rho / (spec.service_rate - spec.arrival_rate)
    L = spec.arrival_rate * W
    Lq = spec.arrival_rate * Wq
    return {
        "rho": rho,
        "W": W,
        "Wq": Wq,
        "L": L,
        "Lq": Lq,
        "utilization": rho,
        "throughput": spec.arrival_rate,
    }


def validate_simulation_experiment(
    paths: ProjectPaths, experiment_id: str
) -> list[ValidationReport]:
    spec = load_simulation_spec(paths, experiment_id)
    result = load_simulation_result(paths, experiment_id)
    record = load_simulation_record(paths, experiment_id)
    issues: list[ValidationIssue] = []

    if spec.spec_hash != spec.computed_hash():
        issues.append(_error("spec_hash_mismatch", "simulation spec hash does not match content"))
    if result.spec_hash != spec.spec_hash:
        issues.append(_error("result_spec_hash_mismatch", "result spec hash differs"))
    if record.spec_hash != spec.spec_hash:
        issues.append(_error("record_spec_hash_mismatch", "experiment record spec hash differs"))
    if record.task_family != "simulation":
        issues.append(_error("task_family_mismatch", f"record task is {record.task_family}"))
    if not record.mlflow_run_id:
        issues.append(_error("missing_mlflow_run", "experiment record has no MLflow run id"))
    if record.git_commit in ("", "unknown"):
        issues.append(_warning("missing_git_commit", "experiment record has no git commit"))
    runtime = result.runtime
    for key in ("python_version", "platform", "simpy_version"):
        if not runtime.get(key):
            issues.append(_error("missing_runtime_identifier", f"runtime is missing {key}"))

    if spec.arrival_rate <= 0 or spec.service_rate <= 0:
        issues.append(_error("invalid_rates", "arrival and service rates must be positive"))
    if spec.warmup_time <= 0:
        issues.append(_error("invalid_warmup", "warmup_time must be positive"))
    if spec.measurement_time <= 0:
        issues.append(_error("invalid_measurement", "measurement_time must be positive"))
    if spec.replications < 2:
        issues.append(_error("invalid_replications", "at least two replications are required"))
    if spec.rho >= 1:
        issues.append(
            _error(
                "unstable_queue",
                f"rho={spec.rho:.6g} >= 1; steady-state M/M/1 theory is undefined",
            )
        )
    if result.n_replications != spec.replications:
        issues.append(
            _error(
                "replication_count_mismatch",
                f"spec requires {spec.replications}, result has {result.n_replications}",
            )
        )
    if len(result.replications) != spec.replications:
        issues.append(
            _error(
                "missing_replication",
                f"spec requires {spec.replications}, found {len(result.replications)} rows",
            )
        )
    if result.master_seed != spec.master_seed:
        issues.append(_error("master_seed_mismatch", "result master seed differs from spec"))
    if result.determinism.get("runs") != 2:
        issues.append(_error("determinism_missing", "two-run determinism check is missing"))
    if result.determinism.get("match") is not True:
        issues.append(_error("determinism_mismatch", "determinism check did not match"))
    if result.determinism.get("first_run_digest") != result.determinism.get(
        "second_run_digest"
    ):
        issues.append(_error("determinism_mismatch", "run digests differ"))

    schedule = _expected_schedule(spec.master_seed, spec.replications)
    if result.replication_seed_hash != _schedule_hash(schedule):
        issues.append(_error("seed_schedule_mismatch", "replication seed hash differs"))
    by_id = {replication.replication_id: replication for replication in result.replications}
    if set(by_id) != set(range(spec.replications)):
        issues.append(_error("replication_index_mismatch", "replication ids are incomplete"))
    replication_seeds = [row.replication_seed for row in result.replications]
    arrival_seeds = [row.arrival_seed for row in result.replications]
    service_seeds = [row.service_seed for row in result.replications]
    if len(set(replication_seeds)) != len(replication_seeds):
        issues.append(_error("seed_reuse", "replication seeds are not unique"))
    if len(set(arrival_seeds)) != len(arrival_seeds):
        issues.append(_error("arrival_seed_reuse", "arrival seeds are not unique"))
    if len(set(service_seeds)) != len(service_seeds):
        issues.append(_error("service_seed_reuse", "service seeds are not unique"))

    for expected in schedule:
        replication = by_id.get(expected["replication_id"])
        if replication is None:
            continue
        for key in ("replication_seed", "arrival_seed", "service_seed"):
            if getattr(replication, key) != expected[key]:
                issues.append(
                    _error(
                        "seed_schedule_mismatch",
                        f"replication {replication.replication_id} has wrong {key}",
                    )
                )
    for replication in result.replications:
        label = f"replication {replication.replication_id}"
        if not _close(replication.warmup_time, spec.warmup_time):
            issues.append(_error("warmup_mismatch", f"{label} warmup differs"))
        if not _close(replication.measurement_time, spec.measurement_time):
            issues.append(_error("measurement_mismatch", f"{label} measurement differs"))
        if replication.Wq < 0 or replication.W < 0:
            issues.append(_error("negative_time", f"{label} has a negative sojourn/wait"))
        if replication.Wq - _TOLERANCE > replication.W:
            issues.append(_error("time_invariant", f"{label} has W < Wq"))
        if replication.Lq < 0 or replication.L < 0:
            issues.append(_error("negative_number", f"{label} has a negative count"))
        if replication.Lq - _TOLERANCE > replication.L:
            issues.append(_error("count_invariant", f"{label} has L < Lq"))
        if not 0.0 <= replication.utilization <= 1.0 + _TOLERANCE:
            issues.append(_error("utilization_bounds", f"{label} utilization outside [0,1]"))
        if replication.throughput < 0:
            issues.append(_error("negative_throughput", f"{label} throughput is negative"))
        if replication.eligible_customers <= 0:
            issues.append(_error("no_eligible_customers", f"{label} has no eligible customers"))
        if replication.completed_customers != replication.eligible_customers:
            issues.append(
                _error(
                    "drain_incomplete",
                    f"{label} completed {replication.completed_customers} of "
                    f"{replication.eligible_customers} eligible customers",
                )
            )
        if replication.arrivals < replication.eligible_customers:
            issues.append(_error("arrival_count", f"{label} eligible exceeds total arrivals"))
        if replication.post_window_completions < 0 or (
            replication.post_window_completions > replication.completed_customers
        ):
            issues.append(
                _error(
                    "post_window_count",
                    f"{label} post-window completions are out of range",
                )
            )
        if replication.idle_tail < 0:
            issues.append(_error("negative_idle_tail", f"{label} idle tail is negative"))
        expected_idle = max(0.0, spec.measurement_end - replication.drain_end_time)
        if not _close(replication.idle_tail, expected_idle):
            issues.append(_error("idle_tail_mismatch", f"{label} idle tail is inconsistent"))
        duration = replication.measurement_time
        if not _close(replication.L, replication.system_area / duration):
            issues.append(_error("time_average_mismatch", f"{label} L is not area/duration"))
        if not _close(replication.Lq, replication.queue_area / duration):
            issues.append(_error("time_average_mismatch", f"{label} Lq is not area/duration"))
        if not _close(replication.utilization, replication.busy_area / duration):
            issues.append(
                _error("time_average_mismatch", f"{label} utilization is not busy/duration")
            )
        expected_rate = replication.eligible_customers / duration
        if not _close(replication.effective_arrival_rate, expected_rate):
            issues.append(_error("flow_rate_mismatch", f"{label} effective arrival rate differs"))
        if not _close(replication.throughput, replication.completed_customers / duration):
            issues.append(_error("throughput_mismatch", f"{label} throughput differs"))

    for metric in spec.metrics:
        summary = result.metrics.get(metric)
        if summary is None:
            issues.append(_error("missing_metric", f"summary for {metric} is missing"))
            continue
        values = [float(getattr(row, metric)) for row in result.replications]
        if summary.n != len(values) or summary.n != spec.replications:
            issues.append(_error("summary_n_mismatch", f"{metric} summary n differs"))
        if len(summary.values) != len(values) or any(
            not _close(left, right) for left, right in zip(summary.values, values, strict=True)
        ):
            issues.append(_error("summary_values_mismatch", f"{metric} values differ"))
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        median = statistics.median(values)
        low, high = _ci(values, spec.confidence_level)
        for name, expected in (
            ("mean", mean),
            ("std", std),
            ("median", median),
            ("ci_low", low),
            ("ci_high", high),
            ("ci_width", high - low),
        ):
            if not _close(float(getattr(summary, name)), expected):
                issues.append(
                    _error("summary_mismatch", f"{metric} {name} does not match replications")
                )
        if not summary.ci_low <= summary.mean <= summary.ci_high:
            issues.append(_error("ci_order", f"{metric} mean lies outside its CI"))

    oracle = load_analytic_oracle(paths, experiment_id) if spec.rho < 1 else None
    if spec.rho < 1 and oracle is None:
        issues.append(_error("missing_oracle", "analytic oracle artifact is missing"))
    if oracle is not None and spec.rho < 1:
        theory = _oracle_values(spec)
        for field, expected in theory.items():
            if not _close(float(getattr(oracle, field)), expected):
                issues.append(
                    _error("oracle_mismatch", f"oracle {field} does not match closed form")
                )
        if not _close(oracle.L, oracle.arrival_rate * oracle.W):
            issues.append(_error("theory_invariant", "analytic L != lambda * W"))
        if not _close(oracle.Lq, oracle.arrival_rate * oracle.Wq):
            issues.append(_error("theory_invariant", "analytic Lq != lambda * Wq"))
        if not _close(oracle.L, oracle.rho / (1 - oracle.rho)):
            issues.append(_error("theory_invariant", "analytic L != rho/(1-rho)"))
        if not _close(oracle.Lq, oracle.rho**2 / (1 - oracle.rho)):
            issues.append(_error("theory_invariant", "analytic Lq != rho^2/(1-rho)"))

        for metric in spec.metrics:
            summary = result.metrics.get(metric)
            if summary is None or metric not in theory:
                continue
            comparison = result.analytic_comparison.get(metric)
            if comparison is None:
                issues.append(_error("missing_comparison", f"comparison for {metric} is missing"))
                continue
            value = theory[metric]
            absolute = abs(summary.mean - value)
            relative = absolute / abs(value) if value != 0 else absolute
            for name, expected in (
                ("simulation_mean", summary.mean),
                ("analytic_value", value),
                ("absolute_error", absolute),
                ("relative_error", relative),
            ):
                if not _close(float(getattr(comparison, name)), expected):
                    issues.append(
                        _error("comparison_mismatch", f"{metric} {name} differs")
                    )
            covers = summary.ci_low <= value <= summary.ci_high
            if comparison.ci_covers_analytic != covers:
                issues.append(_error("comparison_mismatch", f"{metric} CI coverage differs"))
            if relative > spec.analytic_relative_tolerance:
                issues.append(
                    _error(
                        "analytic_tolerance_exceeded",
                        f"{metric} relative error {relative:.6f} exceeds "
                        f"{spec.analytic_relative_tolerance}",
                    )
                )
            if relative > 0 and summary.ci_width > 0 and relative > 0.5:
                issues.append(
                    _warning("wide_evidence", f"{metric} relative error is large; CI width "
                            f"{summary.ci_width:.6f}")
                )

        mean_rate = statistics.fmean(
            row.effective_arrival_rate for row in result.replications
        )
        for metric, time_metric in (("L", "W"), ("Lq", "Wq")):
            simulated = statistics.fmean(
                float(getattr(row, metric)) for row in result.replications
            )
            dwell = statistics.fmean(
                float(getattr(row, time_metric)) for row in result.replications
            )
            predicted = mean_rate * dwell
            absolute = abs(simulated - predicted)
            relative = absolute / max(abs(simulated), 1e-12)
            check = result.little_law.get(metric)
            if check is None:
                issues.append(_error("missing_little_law", f"Little's law for {metric} missing"))
                continue
            for name, expected in (
                ("effective_arrival_rate", mean_rate),
                ("simulated_mean", simulated),
                ("rate_times_mean", predicted),
                ("absolute_discrepancy", absolute),
                ("relative_discrepancy", relative),
            ):
                if not _close(float(getattr(check, name)), expected):
                    issues.append(
                        _error("little_law_mismatch", f"{metric} {name} differs")
                    )
            if relative > spec.little_law_relative_tolerance:
                issues.append(
                    _error(
                        "little_law_tolerance_exceeded",
                        f"{metric} relative discrepancy {relative:.6f} exceeds "
                        f"{spec.little_law_relative_tolerance}",
                    )
                )
            expected_pass = relative <= spec.little_law_relative_tolerance
            if check.passed != expected_pass:
                issues.append(_error("little_law_mismatch", f"{metric} stored pass is wrong"))
            if not _close(check.tolerance, spec.little_law_relative_tolerance):
                issues.append(_error("little_law_mismatch", f"{metric} tolerance differs"))

    unknown_metrics = sorted(set(result.metrics) - set(SIMULATION_METRICS))
    if unknown_metrics:
        issues.append(_error("unknown_metric", f"result has unknown metrics {unknown_metrics}"))

    report = build_report(
        "simulation",
        experiment_id,
        issues,
        evidence=[
            str(simulation_spec_file(paths, experiment_id)),
            str(simulation_result_file(paths, experiment_id)),
            str(analytic_oracle_file(paths, experiment_id)),
        ],
    )
    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "simulation.json", report.model_dump_json(indent=2) + "\n"
    )
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


__all__ = ["validate_simulation_experiment"]
