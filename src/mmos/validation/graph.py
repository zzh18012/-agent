import json

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    edge_flows_file,
    flow_audit_file,
    graph_result_file,
    graph_spec_file,
    load_graph_record,
    load_graph_result,
    load_graph_spec,
    load_min_cut_oracle,
    min_cut_oracle_file,
    save_flow_audit,
)
from mmos.experiments.graph_oracle import enumerate_min_cuts
from mmos.schemas.data import Severity
from mmos.schemas.graph import FlowAuditResult, NodeFlowAudit
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOLERANCE = 1e-6


def _error(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.ERROR, message=message)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE + 1e-9 * abs(right)


def validate_graph_experiment(
    paths: ProjectPaths, experiment_id: str
) -> list[ValidationReport]:
    spec = load_graph_spec(paths, experiment_id)
    result = load_graph_result(paths, experiment_id)
    record = load_graph_record(paths, experiment_id)
    issues: list[ValidationIssue] = []

    if spec.graph_hash != spec.computed_hash():
        issues.append(_error("graph_hash_mismatch", "graph hash does not match content"))
    if spec.spec_hash != spec.computed_spec_hash():
        issues.append(_error("spec_hash_mismatch", "spec hash does not match content"))
    if result.graph_hash != spec.graph_hash:
        issues.append(_error("result_graph_hash_mismatch", "result graph hash differs"))
    if result.spec_hash != spec.spec_hash:
        issues.append(_error("result_spec_hash_mismatch", "result spec hash differs"))
    if record.graph_hash != spec.graph_hash or record.spec_hash != spec.spec_hash:
        issues.append(_error("record_hash_mismatch", "experiment record hashes differ"))
    if record.task_family != "graph":
        issues.append(_error("task_family_mismatch", f"record task is {record.task_family}"))
    if not record.mlflow_run_id:
        issues.append(_error("missing_mlflow_run", "experiment record has no MLflow run id"))
    if record.algorithm != result.algorithm or record.library != result.library:
        issues.append(_error("record_algorithm_mismatch", "record algorithm/library differs"))
    if result.library_version != record.library_version:
        issues.append(_error("record_version_mismatch", "record library version differs"))
    if result.algorithm != "edmonds_karp":
        issues.append(_error("unexpected_algorithm", f"algorithm is {result.algorithm}"))
    if not result.determinism.get("match") or result.determinism.get("runs") != 2:
        issues.append(_error("determinism_missing", "two-run determinism check failed"))

    spec_edges = {(edge.u, edge.v): float(edge.capacity) for edge in spec.edges}
    result_edges = {(edge.u, edge.v): edge for edge in result.edge_flows}
    if set(result_edges) != set(spec_edges):
        issues.append(
            _error(
                "edge_flow_set_mismatch",
                f"edge flows {sorted(set(result_edges) ^ set(spec_edges))} differ from spec",
            )
        )
    for key, edge in result_edges.items():
        capacity = spec_edges.get(key)
        if capacity is None:
            continue
        if edge.capacity != capacity:
            issues.append(_error("capacity_mismatch", f"edge {key} capacity differs"))
        if edge.flow < -_TOLERANCE:
            issues.append(_error("negative_flow", f"edge {key} has negative flow"))
        if edge.flow > capacity + _TOLERANCE:
            issues.append(_error("capacity_violation", f"edge {key} flow exceeds capacity"))

    flows = {key: edge.flow for key, edge in result_edges.items()}
    node_audits: list[NodeFlowAudit] = []
    capacity_violations = [
        issue.message
        for issue in issues
        if issue.code in ("negative_flow", "capacity_violation", "capacity_mismatch")
    ]
    conservation_violations: list[str] = []
    max_residual = 0.0
    for node in spec.nodes:
        inflow = sum(
            flow for (u, v), flow in flows.items() if v == node
        )
        outflow = sum(
            flow for (u, v), flow in flows.items() if u == node
        )
        residual = inflow - outflow
        node_audits.append(
            NodeFlowAudit(
                node=node,
                inflow=inflow,
                outflow=outflow,
                residual=residual,
            )
        )
        if node not in (spec.source, spec.sink):
            max_residual = max(max_residual, abs(residual))
            if not _close(inflow, outflow):
                conservation_violations.append(
                    f"{node}: inflow {inflow:.12g}, outflow {outflow:.12g}"
                )
                issues.append(
                    _error(
                        "flow_conservation_violation",
                        f"node {node} inflow {inflow:.12g} != outflow {outflow:.12g}",
                    )
                )

    by_node = {audit.node: audit for audit in node_audits}
    source_audit = by_node[spec.source]
    sink_audit = by_node[spec.sink]
    source_flow = source_audit.outflow - source_audit.inflow
    sink_flow = sink_audit.inflow - sink_audit.outflow
    if not _close(source_flow, result.max_flow_value):
        issues.append(_error("source_flow_mismatch", "source flow differs from reported value"))
    if not _close(sink_flow, result.max_flow_value):
        issues.append(_error("sink_flow_mismatch", "sink flow differs from reported value"))
    if not _close(source_flow, sink_flow):
        issues.append(_error("source_sink_mismatch", "source and sink flow values differ"))

    oracle = load_min_cut_oracle(paths, experiment_id)
    recomputed = enumerate_min_cuts(spec)
    if oracle.graph_hash != spec.graph_hash:
        issues.append(_error("oracle_hash_mismatch", "oracle graph hash differs"))
    if oracle.enumerated_cut_count != oracle.expected_cut_count:
        issues.append(_error("cut_enumeration_incomplete", "not all cuts were enumerated"))
    if oracle.expected_cut_count != 2 ** (len(spec.nodes) - 2):
        issues.append(_error("cut_count_formula", "expected cut count is not 2^(n-2)"))
    if oracle.minimum_cut_capacity != recomputed.minimum_cut_capacity:
        issues.append(_error("oracle_mismatch", "stored minimum cut capacity differs"))
    if oracle.minimum_cut_count != recomputed.minimum_cut_count:
        issues.append(_error("oracle_mismatch", "stored minimum-cut count differs"))
    if oracle.minimum_cut_partitions != recomputed.minimum_cut_partitions:
        issues.append(_error("oracle_mismatch", "stored minimum-cut partitions differ"))
    for partition in oracle.minimum_cut_partitions:
        if spec.source not in partition or spec.sink in partition:
            issues.append(_error("invalid_cut_partition", "cut partition is not a legal s-t cut"))
    if oracle.minimum_cut_count != len(oracle.minimum_cut_partitions):
        issues.append(_error("oracle_mismatch", "minimum-cut count does not match partitions"))
    if len(oracle.minimum_cut_partition_capacities) != len(
        oracle.minimum_cut_partitions
    ):
        issues.append(_error("oracle_mismatch", "minimum-cut capacities are incomplete"))
    for partition, capacity in zip(
        oracle.minimum_cut_partitions,
        oracle.minimum_cut_partition_capacities,
        strict=False,
    ):
        recomputed_capacity = sum(
            float(edge.capacity)
            for edge in spec.edges
            if edge.u in set(partition) and edge.v not in set(partition)
        )
        if not _close(capacity, recomputed_capacity) or not _close(
            capacity, oracle.minimum_cut_capacity
        ):
            issues.append(
                _error(
                    "oracle_mismatch",
                    f"partition {partition} has inconsistent cut capacity",
                )
            )
    if (
        oracle.minimum_cut_partitions
        != recomputed.minimum_cut_partitions
        or oracle.minimum_cut_partition_capacities
        != recomputed.minimum_cut_partition_capacities
    ):
        issues.append(_error("oracle_mismatch", "minimum-cut capacity list differs"))
    if (
        len(oracle.all_cut_partitions) != oracle.expected_cut_count
        or len(oracle.all_cut_capacities) != oracle.expected_cut_count
    ):
        issues.append(_error("oracle_mismatch", "full cut enumeration is incomplete"))
    if (
        oracle.all_cut_partitions != recomputed.all_cut_partitions
        or oracle.all_cut_capacities != recomputed.all_cut_capacities
    ):
        issues.append(_error("oracle_mismatch", "full cut enumeration differs"))
    if oracle.all_cut_capacities and min(oracle.all_cut_capacities) != (
        oracle.minimum_cut_capacity
    ):
        issues.append(_error("oracle_mismatch", "minimum of all cuts differs"))
    if oracle.canonical_partition not in oracle.minimum_cut_partitions:
        issues.append(_error("oracle_mismatch", "canonical partition is not a minimum cut"))
    flow_cut_gap = abs(result.max_flow_value - oracle.minimum_cut_capacity)
    if not _close(result.max_flow_value, oracle.minimum_cut_capacity):
        issues.append(
            _error(
                "max_flow_min_cut_mismatch",
                f"max flow {result.max_flow_value:.12g} != min cut "
                f"{oracle.minimum_cut_capacity:.12g}",
            )
        )

    audit = FlowAuditResult(
        graph_hash=spec.graph_hash,
        capacity_ok=not capacity_violations,
        conservation_ok=not conservation_violations,
        source_flow_value=source_flow,
        sink_flow_value=sink_flow,
        reported_max_flow=result.max_flow_value,
        max_absolute_residual=max_residual,
        tolerance=_TOLERANCE,
        capacity_violations=capacity_violations,
        conservation_violations=conservation_violations,
        node_audits=node_audits,
    )
    save_flow_audit(paths, experiment_id, audit)
    report = build_report(
        "graph",
        experiment_id,
        issues,
        evidence=[
            str(graph_spec_file(paths, experiment_id)),
            str(graph_result_file(paths, experiment_id)),
            str(min_cut_oracle_file(paths, experiment_id)),
            str(flow_audit_file(paths, experiment_id)),
            str(edge_flows_file(paths, experiment_id)),
        ],
    )
    output_dir = paths.experiment_dir(experiment_id) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "graph.json", report.model_dump_json(indent=2) + "\n")
    summary = {
        "experiment_id": experiment_id,
        "blocking": report.blocking,
        "flow_cut_gap": flow_cut_gap,
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


__all__ = ["validate_graph_experiment"]
