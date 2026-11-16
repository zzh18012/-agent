import csv
import hashlib
import io
import json
import platform
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

import simpy
from scipy import stats as scipy_stats

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
    replications_file,
    save_analytic_oracle,
    save_simulation_result,
)
from mmos.experiments.simulation_oracle import compute_analytic_oracle
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.simulation import (
    SIMULATION_METRICS,
    AnalyticComparison,
    LittleLawCheck,
    MetricSummary,
    ReplicationResult,
    SimulationExperimentRecord,
    SimulationResult,
    SimulationSpec,
)


class SimulationError(RuntimeError):
    pass


class _TimeWeightedAccumulator:
    """Integrates a piecewise-constant state over [start, end)."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end
        self.last = start
        self.system = 0
        self.queue = 0
        self.busy = 0
        self.system_area = 0.0
        self.queue_area = 0.0
        self.busy_area = 0.0

    def _touch(self, now: float) -> None:
        moment = min(max(now, self.start), self.end)
        if moment > self.last:
            duration = moment - self.last
            self.system_area += self.system * duration
            self.queue_area += self.queue * duration
            self.busy_area += self.busy * duration
            self.last = moment

    def transition(
        self, now: float, *, system: int = 0, queue: int = 0, busy: int = 0
    ) -> None:
        self._touch(now)
        self.system += system
        self.queue += queue
        self.busy += busy

    def finalize(self) -> None:
        self._touch(self.end)


def _seed_schedule(master_seed: int, replications: int) -> list[dict[str, int]]:
    schedule: list[dict[str, int]] = []
    for replication_id in range(replications):
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
    for key in ("replication_seed", "arrival_seed", "service_seed"):
        values = [entry[key] for entry in schedule]
        if len(set(values)) != len(values):
            raise SimulationError(f"derived {key} values are not unique")
    return schedule


def _seed_hash(schedule: list[dict[str, int]]) -> str:
    encoded = json.dumps(schedule, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def derive_seed_schedule(master_seed: int, replications: int) -> list[dict[str, int]]:
    """Public deterministic seed schedule shared by formal simulation experiments."""
    return _seed_schedule(master_seed, replications)


def seed_schedule_hash(schedule: list[dict[str, int]]) -> str:
    return _seed_hash(schedule)


def confidence_interval(
    values: list[float], confidence_level: float
) -> tuple[float, float]:
    return _confidence_interval(values, confidence_level)


class _MM1Run:
    def __init__(
        self,
        spec: SimulationSpec,
        seeds: dict[str, int],
    ) -> None:
        self.spec = spec
        self.seeds = seeds
        self.env = simpy.Environment()
        self.server = simpy.Resource(self.env, capacity=1)
        self.accumulator = _TimeWeightedAccumulator(
            spec.warmup_time, spec.measurement_end
        )
        self.total_arrivals = 0
        self.eligible = 0
        self.completed = 0
        self.post_window_completions = 0
        self.departures: list[float] = []
        self.waiting_times: list[float] = []
        self.system_times: list[float] = []

    def run(self) -> ReplicationResult:
        started = time.perf_counter()
        self.env.process(self._arrivals())
        self.env.run()
        self.accumulator.finalize()
        if self.eligible == 0:
            raise SimulationError("no eligible customers arrived in the measurement window")
        if self.completed != self.eligible:
            raise SimulationError(
                f"{self.eligible - self.completed} eligible customers did not drain"
            )
        duration = self.spec.measurement_time
        last_departure = max(self.departures, default=self.spec.measurement_end)
        idle_tail = max(0.0, self.spec.measurement_end - last_departure)
        return ReplicationResult(
            replication_id=self.seeds["replication_id"],
            replication_seed=self.seeds["replication_seed"],
            arrival_seed=self.seeds["arrival_seed"],
            service_seed=self.seeds["service_seed"],
            warmup_time=self.spec.warmup_time,
            measurement_time=duration,
            W=statistics.fmean(self.system_times),
            Wq=statistics.fmean(self.waiting_times),
            L=self.accumulator.system_area / duration,
            Lq=self.accumulator.queue_area / duration,
            utilization=self.accumulator.busy_area / duration,
            throughput=self.completed / duration,
            effective_arrival_rate=self.eligible / duration,
            arrivals=self.total_arrivals,
            eligible_customers=self.eligible,
            completed_customers=self.completed,
            post_window_completions=self.post_window_completions,
            system_area=self.accumulator.system_area,
            queue_area=self.accumulator.queue_area,
            busy_area=self.accumulator.busy_area,
            drain_end_time=max(self.departures, default=0.0),
            idle_tail=idle_tail,
            runtime_seconds=time.perf_counter() - started,
        )

    def _arrivals(self):
        arrival_rng = random.Random(self.seeds["arrival_seed"])
        end = self.spec.measurement_end
        while True:
            yield self.env.timeout(arrival_rng.expovariate(self.spec.arrival_rate))
            if self.env.now >= end:
                return
            self.total_arrivals += 1
            customer_id = self.total_arrivals
            arrival = self.env.now
            eligible = self.spec.warmup_time <= arrival < end
            if eligible:
                self.eligible += 1
            self.accumulator.transition(arrival, system=1, queue=1)
            self.env.process(self._customer(arrival, eligible, customer_id))

    def _customer(self, arrival: float, eligible: bool, customer_id: int):
        service_rng = random.Random(self.seeds["service_seed"] + customer_id)
        with self.server.request() as request:
            yield request
            service_start = self.env.now
            self.accumulator.transition(service_start, queue=-1, busy=1)
            yield self.env.timeout(service_rng.expovariate(self.spec.service_rate))
            departure = self.env.now
            self.accumulator.transition(departure, system=-1, busy=-1)
            self.departures.append(departure)
            if eligible:
                self.completed += 1
                if departure > self.spec.measurement_end:
                    self.post_window_completions += 1
                self.waiting_times.append(service_start - arrival)
                self.system_times.append(departure - arrival)


def _confidence_interval(
    values: list[float], confidence_level: float
) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    standard_error = statistics.stdev(values) / (len(values) ** 0.5)
    critical = float(scipy_stats.t.ppf((1 + confidence_level) / 2, len(values) - 1))
    half_width = critical * standard_error
    return mean - half_width, mean + half_width


def _summaries(
    spec: SimulationSpec, replications: list[ReplicationResult]
) -> dict[str, MetricSummary]:
    summaries: dict[str, MetricSummary] = {}
    for metric in spec.metrics:
        values = [float(getattr(replication, metric)) for replication in replications]
        low, high = _confidence_interval(values, spec.confidence_level)
        summaries[metric] = MetricSummary(
            metric=metric,
            values=values,
            mean=statistics.fmean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            median=statistics.median(values),
            ci_low=low,
            ci_high=high,
            ci_width=high - low,
            confidence_level=spec.confidence_level,
            n=len(values),
        )
    return summaries


def _comparisons(
    spec: SimulationSpec,
    oracle,
    summaries: dict[str, MetricSummary],
) -> dict[str, AnalyticComparison]:
    theory = {
        "W": oracle.W,
        "Wq": oracle.Wq,
        "L": oracle.L,
        "Lq": oracle.Lq,
        "utilization": oracle.utilization,
        "throughput": oracle.throughput,
    }
    comparisons: dict[str, AnalyticComparison] = {}
    for metric, summary in summaries.items():
        if metric not in theory:
            continue
        value = theory[metric]
        absolute = abs(summary.mean - value)
        relative = absolute / abs(value) if value != 0 else absolute
        comparisons[metric] = AnalyticComparison(
            metric=metric,
            simulation_mean=summary.mean,
            analytic_value=value,
            absolute_error=absolute,
            relative_error=relative,
            ci_low=summary.ci_low,
            ci_high=summary.ci_high,
            ci_covers_analytic=summary.ci_low <= value <= summary.ci_high,
        )
    return comparisons


def _little_law(
    spec: SimulationSpec, replications: list[ReplicationResult]
) -> dict[str, LittleLawCheck]:
    mean_rate = statistics.fmean(
        replication.effective_arrival_rate for replication in replications
    )
    checks: dict[str, LittleLawCheck] = {}
    pairs: tuple[tuple[Literal["L", "Lq"], Literal["W", "Wq"]], ...] = (
        ("L", "W"),
        ("Lq", "Wq"),
    )
    for metric, time_metric in pairs:
        simulated = statistics.fmean(
            float(getattr(replication, metric)) for replication in replications
        )
        dwell = statistics.fmean(
            float(getattr(replication, time_metric)) for replication in replications
        )
        predicted = mean_rate * dwell
        absolute = abs(simulated - predicted)
        relative = absolute / max(abs(simulated), 1e-12)
        checks[metric] = LittleLawCheck(
            metric=metric,
            effective_arrival_rate=mean_rate,
            simulated_mean=simulated,
            rate_times_mean=predicted,
            absolute_discrepancy=absolute,
            relative_discrepancy=relative,
            tolerance=spec.little_law_relative_tolerance,
            passed=relative <= spec.little_law_relative_tolerance,
        )
    return checks


def _replication_digest(replications: list[ReplicationResult]) -> str:
    payload = [
        replication.model_dump(mode="json", exclude={"runtime_seconds"})
        for replication in replications
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _runtime_identifiers() -> dict[str, str]:
    try:
        simpy_version = metadata.version("simpy")
    except metadata.PackageNotFoundError:
        simpy_version = "not found"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "simpy_version": simpy_version,
    }


def _write_replications(paths: ProjectPaths, result: SimulationResult) -> Path:
    target = replications_file(paths, result.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    fieldnames = list(ReplicationResult.model_fields)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for replication in result.replications:
        writer.writerow(replication.model_dump())
    atomic_write_text(target, buffer.getvalue())
    return target


def run_simulation(paths: ProjectPaths, spec: SimulationSpec) -> SimulationResult:
    if spec.spec_hash != spec.computed_hash():
        raise SimulationError("simulation spec hash does not match its content")
    oracle = compute_analytic_oracle(spec)
    started = datetime.now(UTC).replace(microsecond=0)
    started_clock = time.perf_counter()
    schedule = _seed_schedule(spec.master_seed, spec.replications)
    replications = [_MM1Run(spec, seeds).run() for seeds in schedule]
    second_run = [_MM1Run(spec, seeds).run() for seeds in schedule]
    first_digest = _replication_digest(replications)
    second_digest = _replication_digest(second_run)
    if first_digest != second_digest:
        raise SimulationError("two independent runs with the same seed schedule differ")
    summaries = _summaries(spec, replications)
    comparisons = _comparisons(spec, oracle, summaries)
    little_law = _little_law(spec, replications)
    completed = datetime.now(UTC).replace(microsecond=0)
    result = SimulationResult(
        experiment_id=spec.experiment_id,
        simulation_id=spec.simulation_id,
        spec_hash=spec.spec_hash,
        git_commit=current_git_commit(),
        master_seed=spec.master_seed,
        replication_seed_hash=_seed_hash(schedule),
        n_replications=len(replications),
        metrics=summaries,
        analytic_comparison=comparisons,
        little_law=little_law,
        replications=replications,
        runtime=_runtime_identifiers(),
        determinism={
            "first_run_digest": first_digest,
            "second_run_digest": second_digest,
            "match": True,
            "runs": 2,
            "note": (
                "Two independent in-process executions with fresh SimPy environments and "
                "RNG streams; same process/platform, not a cross-platform check."
            ),
        },
        runtime_seconds=time.perf_counter() - started_clock,
        started_at=started,
        completed_at=completed,
        notes=[
            "Warm-up runs normally but is excluded from measurement statistics.",
            "Arrivals stop at measurement end; eligible customers drain to completion.",
            "L, Lq, and utilization are time-weighted over the measurement window.",
        ],
    )
    save_analytic_oracle(paths, spec.experiment_id, oracle)
    save_simulation_result(paths, result)
    _write_replications(paths, result)
    return result


def build_simulation_record(
    paths: ProjectPaths,
    spec: SimulationSpec,
    result: SimulationResult,
    *,
    status: ExperimentStatus,
) -> SimulationExperimentRecord:
    return SimulationExperimentRecord(
        experiment_id=spec.experiment_id,
        project_id=paths.root.name,
        spec_hash=spec.spec_hash,
        git_commit=result.git_commit,
        master_seed=spec.master_seed,
        replication_seed_hash=result.replication_seed_hash,
        status=status,
    )


__all__ = [
    "SIMULATION_METRICS",
    "SimulationError",
    "_TimeWeightedAccumulator",
    "_confidence_interval",
    "_seed_schedule",
    "build_simulation_record",
    "confidence_interval",
    "derive_seed_schedule",
    "run_simulation",
    "seed_schedule_hash",
]
