import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.optimization import ExperimentStatus


class GraphEdge(MMOSModel):
    u: str
    v: str
    capacity: float = Field(ge=0, allow_inf_nan=False)


class GraphProblemSpec(MMOSModel):
    schema_version: int = 1
    graph_id: str
    experiment_id: str
    problem_type: Literal["maximum_flow"] = "maximum_flow"
    directed: Literal[True] = True
    nodes: list[str] = Field(min_length=2)
    edges: list[GraphEdge] = Field(min_length=1)
    source: str
    sink: str
    capacity_attribute: str = "capacity"
    algorithm: Literal["edmonds_karp"] = "edmonds_karp"
    graph_hash: str = ""
    spec_hash: str = ""

    @model_validator(mode="after")
    def _valid_graph(self) -> "GraphProblemSpec":
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("graph nodes must be unique")
        node_set = set(self.nodes)
        if self.source not in node_set or self.sink not in node_set:
            raise ValueError("source and sink must be nodes")
        if self.source == self.sink:
            raise ValueError("source and sink must differ")
        seen: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.u not in node_set or edge.v not in node_set:
                raise ValueError(f"edge endpoint {edge.u}->{edge.v} is not a node")
            if edge.u == edge.v:
                raise ValueError(f"self-loop {edge.u}->{edge.v} is unsupported")
            if (edge.u, edge.v) in seen:
                raise ValueError(f"parallel edge {edge.u}->{edge.v} is unsupported")
            seen.add((edge.u, edge.v))
        if not self.capacity_attribute:
            raise ValueError("capacity_attribute must be non-empty")
        return self

    def computed_hash(self) -> str:
        payload = {
            "problem_type": self.problem_type,
            "directed": self.directed,
            "nodes": sorted(self.nodes),
            "edges": sorted(
                (
                    {
                        "u": edge.u,
                        "v": edge.v,
                        "capacity": float(edge.capacity),
                    }
                    for edge in self.edges
                ),
                key=lambda item: (item["u"], item["v"]),
            ),
            "source": self.source,
            "sink": self.sink,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def computed_spec_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def with_hash(self) -> "GraphProblemSpec":
        graph_hash = self.computed_hash()
        with_graph = self.model_copy(update={"graph_hash": graph_hash})
        return with_graph.model_copy(update={"spec_hash": with_graph.computed_spec_hash()})


class EdgeFlow(MMOSModel):
    u: str
    v: str
    capacity: float
    flow: float


class NodeFlowAudit(MMOSModel):
    node: str
    inflow: float
    outflow: float
    residual: float


class FlowAuditResult(MMOSModel):
    graph_hash: str
    capacity_ok: bool
    conservation_ok: bool
    source_flow_value: float
    sink_flow_value: float
    reported_max_flow: float
    max_absolute_residual: float
    tolerance: float
    capacity_violations: list[str] = Field(default_factory=list)
    conservation_violations: list[str] = Field(default_factory=list)
    node_audits: list[NodeFlowAudit] = Field(default_factory=list)


class MinCutOracle(MMOSModel):
    method: Literal["exhaustive_partition_enumeration"] = (
        "exhaustive_partition_enumeration"
    )
    graph_hash: str
    node_count: int
    enumerated_cut_count: int
    expected_cut_count: int
    minimum_cut_capacity: float
    minimum_cut_count: int
    minimum_cut_partitions: list[list[str]]
    minimum_cut_partition_capacities: list[float]
    all_cut_partitions: list[list[str]]
    all_cut_capacities: list[float]
    canonical_partition: list[str]


class GraphResult(MMOSModel):
    experiment_id: str
    graph_id: str
    graph_hash: str
    spec_hash: str
    status: Literal["succeeded", "failed"]
    max_flow_value: float
    edge_flows: list[EdgeFlow]
    source: str
    sink: str
    algorithm: str
    library: str
    library_version: str
    determinism: dict[str, str | int | bool] = Field(default_factory=dict)
    runtime_seconds: float
    started_at: datetime
    completed_at: datetime
    notes: list[str] = Field(default_factory=list)


class GraphExperimentRecord(MMOSModel):
    schema_version: int = 1
    experiment_id: str
    project_id: str
    task_family: Literal["graph"] = "graph"
    graph_hash: str
    spec_hash: str
    algorithm: str
    library: str
    library_version: str
    git_commit: str = "unknown"
    mlflow_run_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    validation_status: dict[str, str] = Field(default_factory=dict)
