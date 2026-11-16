import hashlib
import json
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.states import TaskFamily


class NodeStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM"
    STALE = "STALE"


class DAGStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    STALE = "STALE"


class MixedSpec(MMOSModel):
    schema_version: int = 1
    mixed_id: str
    problem_id: str
    target_column: str
    time_column: str
    feature_set: list[str] = Field(default_factory=list)
    forecast_horizon: int = Field(ge=1)
    forecast_timestamps: list[str] = Field(min_length=1)
    forecast_model_id: str = "naive.drift"
    n_splits: int = Field(default=4, ge=2)
    seasonal_period: int | None = Field(default=None, ge=2)
    unit: str
    random_seed: int = 0
    planning_periods: list[str] = Field(min_length=1)
    capacities: list[float] = Field(min_length=1)
    production_costs: list[float] = Field(min_length=1)
    holding_costs: list[float] = Field(min_length=1)
    initial_inventory: float = Field(default=0.0, ge=0)
    solver_time_limit_seconds: float = Field(default=10.0, gt=0)
    spec_hash: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "MixedSpec":
        if self.forecast_model_id not in {
            "naive.last_value",
            "naive.drift",
            "naive.seasonal",
        }:
            raise ValueError("first mixed slice supports temporal baselines only")
        if self.forecast_model_id == "naive.seasonal" and self.seasonal_period is None:
            raise ValueError("seasonal baseline requires seasonal_period")
        horizon = self.forecast_horizon
        if len(self.forecast_timestamps) != horizon:
            raise ValueError("forecast timestamps must match the horizon")
        if len(self.planning_periods) != horizon:
            raise ValueError("planning periods must match the forecast horizon")
        if self.forecast_timestamps != self.planning_periods:
            raise ValueError("forecast timestamps must align with planning periods")
        for values, name in (
            (self.capacities, "capacities"),
            (self.production_costs, "production costs"),
            (self.holding_costs, "holding costs"),
        ):
            if len(values) != horizon:
                raise ValueError(f"{name} must match the horizon")
            if any(value < 0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
        if not self.unit:
            raise ValueError("unit is required")
        return self

    def computed_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def with_hash(self) -> "MixedSpec":
        return self.model_copy(update={"spec_hash": self.computed_hash()})


class TaskNode(MMOSModel):
    node_id: str
    subproblem_id: str
    task_family: TaskFamily
    required_inputs: list[str] = Field(default_factory=list)
    produced_outputs: list[str] = Field(default_factory=list)
    execution_status: NodeStatus = NodeStatus.PENDING
    status_reason: str | None = None
    experiment_id: str | None = None


class ArtifactEdge(MMOSModel):
    producer_node: str
    producer_output: str
    consumer_node: str
    consumer_input: str
    required_artifact_type: str
    required_schema_version: str


class MixedExecutionPlan(MMOSModel):
    plan_id: str
    project_id: str
    problem_hash: str
    problem_spec_hash: str
    routing_id: str
    nodes: list[TaskNode]
    edges: list[ArtifactEdge]
    entry_nodes: list[str]
    terminal_nodes: list[str]
    spec: MixedSpec
    created_at: datetime
    plan_hash: str = ""

    @model_validator(mode="after")
    def _graph_shape(self) -> "MixedExecutionPlan":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node ids must be unique")
        if not set(self.entry_nodes) <= set(node_ids):
            raise ValueError("entry nodes must exist")
        if not set(self.terminal_nodes) <= set(node_ids):
            raise ValueError("terminal nodes must exist")
        for edge in self.edges:
            if edge.producer_node not in node_ids or edge.consumer_node not in node_ids:
                raise ValueError("edge references an unknown node")
        return self

    def computed_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"plan_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def with_hash(self) -> "MixedExecutionPlan":
        return self.model_copy(update={"plan_hash": self.computed_hash()})

    def topological_order(self) -> list[str]:
        incoming = {node.node_id: 0 for node in self.nodes}
        outgoing: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            incoming[edge.consumer_node] += 1
            outgoing[edge.producer_node].append(edge.consumer_node)
        ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for consumer in outgoing[node_id]:
                incoming[consumer] -= 1
                if incoming[consumer] == 0:
                    ready.append(consumer)
            ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("DAG contains a cycle")
        return order


class ArtifactEnvelope(MMOSModel):
    artifact_id: str
    artifact_type: str
    schema_version: str
    producer_project_id: str
    producer_node_id: str
    producer_task_family: TaskFamily
    producer_experiment_id: str
    content_hash: str
    validation_status: str = "verified"
    created_at: datetime
    payload_path: str
    payload_hash: str
    provenance: dict[str, str] = Field(default_factory=dict)


class ForecastArtifact(MMOSModel):
    forecast_horizon: int = Field(ge=1)
    timestamps: list[str] = Field(min_length=1)
    target_name: str
    unit: str
    point_forecasts: list[float] = Field(min_length=1)
    model_id: str
    source_dataset_hash: str
    forecast_experiment_id: str
    validation_strategy: str
    validation_status: str = "verified"

    @model_validator(mode="after")
    def _shape(self) -> "ForecastArtifact":
        if not (
            len(self.timestamps)
            == len(self.point_forecasts)
            == self.forecast_horizon
        ):
            raise ValueError("forecast horizon, timestamps, and values must match")
        if self.validation_status != "verified":
            raise ValueError("only verified forecast payloads may be published")
        return self

    def computed_content_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


class ArtifactContract(MMOSModel):
    artifact_type: str
    schema_version: str
    target_name: str
    unit: str
    horizon: int
    timestamps: list[str]


class DAGRunRecord(MMOSModel):
    run_id: str
    plan_id: str
    plan_hash: str
    project_id: str
    status: DAGStatus
    nodes: list[TaskNode]
    forecast_artifact_id: str | None = None
    forecast_artifact_hash: str | None = None
    optimization_experiment_id: str | None = None
    dependency_validation: dict[str, str] = Field(default_factory=dict)
    final_objective: float | None = None
    final_feasible: bool | None = None
    provenance_complete: bool = False
    started_at: datetime
    completed_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "ArtifactContract",
    "ArtifactEdge",
    "ArtifactEnvelope",
    "DAGRunRecord",
    "DAGStatus",
    "ForecastArtifact",
    "MixedExecutionPlan",
    "MixedSpec",
    "NodeStatus",
    "TaskNode",
]
