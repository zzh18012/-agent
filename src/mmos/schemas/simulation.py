import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.optimization import ExperimentStatus

SIMULATION_METRICS = ("W", "Wq", "L", "Lq", "utilization", "throughput")


class SimulationSpec(MMOSModel):
    schema_version: int = 1
    simulation_id: str
    experiment_id: str
    model_family: Literal["mm1"] = "mm1"
    arrival_process: Literal["poisson"] = "poisson"
    service_process: Literal["exponential"] = "exponential"
    arrival_rate: float = Field(gt=0)
    service_rate: float = Field(gt=0)
    queue_discipline: Literal["fifo"] = "fifo"
    initial_state: Literal["empty"] = "empty"
    simulation_time_units: str = "abstract_time_units"
    warmup_time: float = Field(gt=0)
    measurement_time: float = Field(gt=0)
    replications: int = Field(ge=2)
    master_seed: int = Field(ge=0)
    seed_derivation: Literal["sha256"] = "sha256"
    metrics: list[str] = Field(default_factory=lambda: list(SIMULATION_METRICS))
    analytic_validation: bool = True
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    analytic_relative_tolerance: float = Field(default=0.10, gt=0)
    little_law_relative_tolerance: float = Field(default=0.15, gt=0)
    spec_hash: str = ""

    @model_validator(mode="after")
    def _valid_metrics(self) -> "SimulationSpec":
        unknown = sorted(set(self.metrics) - set(SIMULATION_METRICS))
        if unknown:
            raise ValueError(f"unsupported simulation metrics: {unknown}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("simulation metrics must be unique")
        if not self.metrics:
            raise ValueError("at least one simulation metric is required")
        return self

    @property
    def rho(self) -> float:
        return self.arrival_rate / self.service_rate

    @property
    def stable(self) -> bool:
        return self.rho < 1.0

    @property
    def measurement_end(self) -> float:
        return self.warmup_time + self.measurement_time

    def computed_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def with_hash(self) -> "SimulationSpec":
        return self.model_copy(update={"spec_hash": self.computed_hash()})


class AnalyticOracle(MMOSModel):
    method: Literal["closed_form_mm1"] = "closed_form_mm1"
    arrival_rate: float
    service_rate: float
    rho: float
    W: float
    Wq: float
    L: float
    Lq: float
    utilization: float
    throughput: float


class ReplicationResult(MMOSModel):
    replication_id: int
    replication_seed: int
    arrival_seed: int
    service_seed: int
    warmup_time: float
    measurement_time: float
    W: float
    Wq: float
    L: float
    Lq: float
    utilization: float
    throughput: float
    effective_arrival_rate: float
    arrivals: int
    eligible_customers: int
    completed_customers: int
    post_window_completions: int
    system_area: float
    queue_area: float
    busy_area: float
    drain_end_time: float
    idle_tail: float
    runtime_seconds: float


class MetricSummary(MMOSModel):
    metric: str
    values: list[float]
    mean: float
    std: float
    median: float
    ci_low: float
    ci_high: float
    ci_width: float
    confidence_level: float
    ci_method: Literal["student_t"] = "student_t"
    n: int


class AnalyticComparison(MMOSModel):
    metric: str
    simulation_mean: float
    analytic_value: float
    absolute_error: float
    relative_error: float
    ci_low: float
    ci_high: float
    ci_covers_analytic: bool


class LittleLawCheck(MMOSModel):
    metric: Literal["L", "Lq"]
    effective_arrival_rate: float
    simulated_mean: float
    rate_times_mean: float
    absolute_discrepancy: float
    relative_discrepancy: float
    tolerance: float
    passed: bool


class SimulationResult(MMOSModel):
    experiment_id: str
    simulation_id: str
    spec_hash: str
    git_commit: str = "unknown"
    master_seed: int
    replication_seed_hash: str
    n_replications: int
    metrics: dict[str, MetricSummary]
    analytic_comparison: dict[str, AnalyticComparison]
    little_law: dict[str, LittleLawCheck]
    replications: list[ReplicationResult]
    runtime: dict[str, str] = Field(default_factory=dict)
    determinism: dict[str, str | int | bool] = Field(default_factory=dict)
    runtime_seconds: float
    started_at: datetime
    completed_at: datetime
    notes: list[str] = Field(default_factory=list)


class SimulationExperimentRecord(MMOSModel):
    schema_version: int = 1
    experiment_id: str
    project_id: str
    task_family: Literal["simulation"] = "simulation"
    spec_hash: str
    git_commit: str = "unknown"
    master_seed: int
    replication_seed_hash: str
    mlflow_run_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    validation_status: dict[str, str] = Field(default_factory=dict)
