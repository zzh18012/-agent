import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.optimization import ExperimentStatus

MIXED_SIMULATION_METRICS = (
    "total_shortage",
    "shortage_period_count",
    "fill_rate",
    "ending_inventory",
    "average_inventory",
    "max_shortage",
)


class MixedSimulationSpec(MMOSModel):
    schema_version: int = 1
    simulation_experiment_id: str
    simulation_id: str
    problem_id: str
    project_id: str
    unit: str
    planning_horizon: int = Field(ge=1)
    timestamps: list[str] = Field(min_length=1)
    production_plan_artifact_id: str
    production_plan_artifact_hash: str
    forecast_artifact_id: str
    forecast_artifact_hash: str
    forecast_error_artifact_id: str
    forecast_error_artifact_hash: str
    replications: int = Field(ge=2)
    master_seed: int = Field(ge=0)
    seed_derivation: Literal["sha256"] = "sha256"
    residual_sampling_method: Literal["empirical_iid_bootstrap"] = (
        "empirical_iid_bootstrap"
    )
    demand_floor_policy: Literal["max_zero"] = "max_zero"
    metrics: list[str] = Field(
        default_factory=lambda: list(MIXED_SIMULATION_METRICS)
    )
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    spec_hash: str = ""

    @model_validator(mode="after")
    def _shape(self) -> "MixedSimulationSpec":
        if len(self.timestamps) != self.planning_horizon:
            raise ValueError("timestamps must match the planning horizon")
        if not self.unit:
            raise ValueError("unit is required")
        unknown = sorted(set(self.metrics) - set(MIXED_SIMULATION_METRICS))
        if unknown:
            raise ValueError(f"unsupported mixed simulation metrics: {unknown}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("simulation metrics must be unique")
        if not self.metrics:
            raise ValueError("at least one simulation metric is required")
        return self

    def computed_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def with_hash(self) -> "MixedSimulationSpec":
        return self.model_copy(update={"spec_hash": self.computed_hash()})


class SimulationPeriodResult(MMOSModel):
    timestamp: str
    point_forecast: float
    sampled_residual: float
    simulated_demand: float
    production: float
    available: float
    served: float
    shortage: float
    ending_inventory: float


class MixedSimulationReplication(MMOSModel):
    replication_id: int = Field(ge=0)
    replication_seed: int = Field(ge=0)
    periods: list[SimulationPeriodResult] = Field(min_length=1)
    total_demand: float
    total_served: float
    total_shortage: float
    shortage_period_count: int
    fill_rate: float
    ending_inventory: float
    average_inventory: float
    max_shortage: float

    @model_validator(mode="after")
    def _arithmetic(self) -> "MixedSimulationReplication":
        demand = sum(period.simulated_demand for period in self.periods)
        served = sum(period.served for period in self.periods)
        shortage = sum(period.shortage for period in self.periods)
        if abs(self.total_demand - demand) > 1e-9:
            raise ValueError("total_demand does not equal the period demands")
        if abs(self.total_served - served) > 1e-9:
            raise ValueError("total_served does not equal the period served values")
        if abs(self.total_shortage - shortage) > 1e-9:
            raise ValueError("total_shortage does not equal the period shortages")
        if self.shortage_period_count != sum(
            1 for period in self.periods if period.shortage > 1e-9
        ):
            raise ValueError("shortage_period_count does not match the periods")
        expected_fill = served / demand if demand > 0 else 1.0
        if abs(self.fill_rate - expected_fill) > 1e-9:
            raise ValueError("fill_rate must equal total_served / total_demand")
        if abs(self.ending_inventory - self.periods[-1].ending_inventory) > 1e-9:
            raise ValueError("ending_inventory must equal the last period inventory")
        average = sum(period.ending_inventory for period in self.periods) / len(
            self.periods
        )
        if abs(self.average_inventory - average) > 1e-9:
            raise ValueError("average_inventory does not match the periods")
        if abs(self.max_shortage - max(period.shortage for period in self.periods)) > 1e-9:
            raise ValueError("max_shortage does not match the periods")
        return self


class MixedSimulationMetricSummary(MMOSModel):
    metric: str
    mean: float
    std: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    ci_low: float
    ci_high: float
    confidence_level: float
    ci_method: Literal["student_t"] = "student_t"
    n: int


class MixedSimulationResult(MMOSModel):
    experiment_id: str
    simulation_id: str
    spec_hash: str
    unit: str
    planning_horizon: int
    timestamps: list[str]
    production_plan_artifact_id: str
    production_plan_artifact_hash: str
    forecast_artifact_id: str
    forecast_artifact_hash: str
    forecast_error_artifact_id: str
    forecast_error_artifact_hash: str
    master_seed: int
    replication_seed_hash: str
    n_replications: int
    metrics: dict[str, MixedSimulationMetricSummary]
    any_shortage_probability: float
    distinct_residual_paths: int = Field(ge=1)
    demand_floor_activations: int = Field(ge=0)
    replications: list[MixedSimulationReplication] = Field(min_length=1)
    determinism: dict[str, str | int | bool] = Field(default_factory=dict)
    runtime_seconds: float
    started_at: datetime
    completed_at: datetime
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _summary(self) -> "MixedSimulationResult":
        if self.n_replications != len(self.replications):
            raise ValueError("n_replications must equal the replication count")
        with_shortage = sum(
            1 for replication in self.replications if replication.total_shortage > 1e-9
        )
        expected = with_shortage / self.n_replications
        if abs(self.any_shortage_probability - expected) > 1e-9:
            raise ValueError("any_shortage_probability does not match the replications")
        if self.distinct_residual_paths > self.n_replications:
            raise ValueError("distinct path count exceeds the replication count")
        return self


class MixedSimulationRecord(MMOSModel):
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
    consumed_artifact_ids: list[str] = Field(default_factory=list)
    consumed_artifact_hashes: list[str] = Field(default_factory=list)
    dependency_contract_status: str | None = None
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "MIXED_SIMULATION_METRICS",
    "MixedSimulationMetricSummary",
    "MixedSimulationRecord",
    "MixedSimulationReplication",
    "MixedSimulationResult",
    "MixedSimulationSpec",
    "SimulationPeriodResult",
]
