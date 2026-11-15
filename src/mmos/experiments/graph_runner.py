import csv
import hashlib
import io
import json
import time
from datetime import UTC, datetime

import networkx as nx
from networkx.algorithms.flow import edmonds_karp

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
    edge_flows_file,
    save_graph_result,
    save_min_cut_oracle,
)
from mmos.experiments.graph_oracle import enumerate_min_cuts
from mmos.schemas.graph import (
    EdgeFlow,
    GraphExperimentRecord,
    GraphProblemSpec,
    GraphResult,
)
from mmos.schemas.optimization import ExperimentStatus


class GraphError(RuntimeError):
    pass


def _networkx_graph(spec: GraphProblemSpec) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(spec.nodes)
    for edge in spec.edges:
        graph.add_edge(edge.u, edge.v, capacity=float(edge.capacity))
    return graph


def _solve(spec: GraphProblemSpec) -> tuple[float, list[EdgeFlow]]:
    graph = _networkx_graph(spec)
    value, flow = nx.maximum_flow(
        graph,
        spec.source,
        spec.sink,
        capacity=spec.capacity_attribute,
        flow_func=edmonds_karp,
    )
    edges = [
        EdgeFlow(
            u=edge.u,
            v=edge.v,
            capacity=float(edge.capacity),
            flow=float(flow.get(edge.u, {}).get(edge.v, 0.0)),
        )
        for edge in spec.edges
    ]
    return float(value), edges


def _digest(value: float, edges: list[EdgeFlow]) -> str:
    payload = {
        "max_flow_value": value,
        "edge_flows": sorted(
            (
                {"u": edge.u, "v": edge.v, "flow": edge.flow}
                for edge in edges
            ),
            key=lambda item: (item["u"], item["v"]),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _write_edge_flows(paths: ProjectPaths, result: GraphResult) -> None:
    target = edge_flows_file(paths, result.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EdgeFlow.model_fields))
    writer.writeheader()
    for edge in result.edge_flows:
        writer.writerow(edge.model_dump())
    atomic_write_text(target, buffer.getvalue())


def run_graph_experiment(paths: ProjectPaths, spec: GraphProblemSpec) -> GraphResult:
    if spec.graph_hash != spec.computed_hash():
        raise GraphError("graph hash does not match graph content")
    if spec.spec_hash != spec.computed_spec_hash():
        raise GraphError("graph spec hash does not match spec content")
    started_at = datetime.now(UTC).replace(microsecond=0)
    started = time.perf_counter()
    value, edges = _solve(spec)
    second_value, second_edges = _solve(spec)
    first_digest = _digest(value, edges)
    second_digest = _digest(second_value, second_edges)
    if first_digest != second_digest:
        raise GraphError("two independent solver runs produced different results")
    completed_at = datetime.now(UTC).replace(microsecond=0)
    oracle = enumerate_min_cuts(spec)
    result = GraphResult(
        experiment_id=spec.experiment_id,
        graph_id=spec.graph_id,
        graph_hash=spec.graph_hash,
        spec_hash=spec.spec_hash,
        status="succeeded",
        max_flow_value=value,
        edge_flows=edges,
        source=spec.source,
        sink=spec.sink,
        algorithm=spec.algorithm,
        library="networkx",
        library_version=nx.__version__,
        determinism={
            "first_run_digest": first_digest,
            "second_run_digest": second_digest,
            "match": True,
            "runs": 2,
            "note": "Two independent in-process NetworkX runs.",
        },
        runtime_seconds=time.perf_counter() - started,
        started_at=started_at,
        completed_at=completed_at,
        notes=[
            "Directed capacitated graph; capacity_attribute="
            f"{spec.capacity_attribute}.",
            "Minimum cut is computed by exhaustive partition enumeration, not NetworkX.",
        ],
    )
    save_graph_result(paths, result)
    save_min_cut_oracle(paths, spec.experiment_id, oracle)
    _write_edge_flows(paths, result)
    return result


def build_graph_record(
    paths: ProjectPaths,
    spec: GraphProblemSpec,
    result: GraphResult,
    *,
    status: ExperimentStatus,
) -> GraphExperimentRecord:
    return GraphExperimentRecord(
        experiment_id=spec.experiment_id,
        project_id=paths.root.name,
        graph_hash=spec.graph_hash,
        spec_hash=spec.spec_hash,
        algorithm=result.algorithm,
        library=result.library,
        library_version=result.library_version,
        git_commit=current_git_commit(),
        status=status,
    )


__all__ = ["GraphError", "build_graph_record", "run_graph_experiment"]
