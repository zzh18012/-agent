import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class KnapsackInstance(MMOSModel):
    instance_id: str
    problem_family: Literal["binary_knapsack"] = "binary_knapsack"
    weights: list[int] = Field(min_length=1)
    values: list[int] = Field(min_length=1)
    capacity: int = Field(gt=0)
    generator: str
    generator_version: str
    seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    instance_hash: str

    @model_validator(mode="after")
    def _consistent(self) -> "KnapsackInstance":
        if len(self.weights) != len(self.values):
            raise ValueError("weights and values must have the same length")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("weights must be positive")
        if any(value < 0 for value in self.values):
            raise ValueError("values must be non-negative")
        expected = knapsack_instance_hash(self.weights, self.values, self.capacity)
        if self.instance_hash != expected:
            raise ValueError(f"instance {self.instance_id!r} hash does not match its data")
        return self


def knapsack_instance_hash(weights: list[int], values: list[int], capacity: int) -> str:
    payload = {"weights": weights, "values": values, "capacity": capacity}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class BenchmarkManifest(MMOSModel):
    suite_id: str
    suite_version: str
    task_family: Literal["optimization"] = "optimization"
    problem_family: Literal["binary_knapsack"] = "binary_knapsack"
    created_at: datetime
    generator_version: str
    suite_seed: int
    instances: list[KnapsackInstance] = Field(min_length=1)
    methods: list[str]
    oracle_method: str = "exhaustive_enumeration"
    manifest_hash: str
    frozen: bool = True

    def computed_hash(self) -> str:
        payload = {
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "generator_version": self.generator_version,
            "suite_seed": self.suite_seed,
            "instances": [
                instance.model_dump(mode="json")
                for instance in sorted(self.instances, key=lambda item: item.instance_id)
            ],
            "methods": self.methods,
            "oracle_method": self.oracle_method,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class OracleResult(MMOSModel):
    instance_id: str
    method: str = "exhaustive_enumeration"
    verified_optimum: float
    optimal_subset: list[int]
    search_space_size: int
    feasible_subset_count: int
    runtime_seconds: float
    matches_candidate: bool | None = None


class MethodAggregate(MMOSModel):
    method: str
    role: str
    optimum_hits: int
    failures: int
    mean_absolute_regret: float | None = None
    median_absolute_regret: float | None = None
    max_absolute_regret: float | None = None
    mean_relative_regret: float | None = None
    median_relative_regret: float | None = None
    max_relative_regret: float | None = None
    worst_instance_id: str | None = None
    worst_relative_regret: float | None = None
    hit_rate: float


class InstanceSuiteResult(MMOSModel):
    instance_id: str
    instance_hash: str
    tags: list[str]
    n_items: int
    capacity: int
    oracle_optimum: float
    cbc_objective: float | None = None
    greedy_objective: float | None = None
    enhanced_greedy_objective: float | None = None
    greedy_relative_regret: float | None = None
    enhanced_relative_regret: float | None = None
    cbc_verified: bool = False
    tournament_id: str | None = None
    candidate_experiment_id: str | None = None
    greedy_experiment_id: str | None = None
    enhanced_greedy_experiment_id: str | None = None
    mlflow_run_ids: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class BenchmarkSuiteResult(MMOSModel):
    benchmark_run_id: str
    suite_id: str
    manifest_hash: str
    suite_seed: int
    n_instances: int
    candidate: MethodAggregate
    greedy: MethodAggregate
    enhanced_greedy: MethodAggregate
    candidate_feasible_count: int
    candidate_verified_optimum_count: int
    candidate_win_count: int
    candidate_tie_count: int
    candidate_loss_count: int
    oracle_agreement: bool
    worst_case_instance: str | None = None
    per_instance: list[InstanceSuiteResult]
    status: Literal["complete", "incomplete"]
    missing_instances: list[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime
    methods_code_hash: str = "unknown"
    generator_code_hash: str = "unknown"
    runtime: dict[str, str] = Field(default_factory=dict)
    git_commit: str = "unknown"
    suite_mlflow_run_id: str | None = None
    experiment_ids: dict[str, list[str]] = Field(default_factory=dict)
    tournament_ids: dict[str, str] = Field(default_factory=dict)
    mlflow_run_ids: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
