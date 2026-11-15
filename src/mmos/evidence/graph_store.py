import json
from datetime import UTC, datetime

import yaml

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    edge_flows_file,
    flow_audit_file,
    graph_record_file,
    graph_result_file,
    graph_spec_file,
    load_flow_audit,
    load_graph_record,
    load_graph_result,
    load_graph_spec,
    load_min_cut_oracle,
    min_cut_oracle_file,
    sha256_file,
)
from mmos.evidence.store import EvidenceError, load_evidence_index
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.validation import ValidationReport, ValidationStatus


def freeze_graph(paths: ProjectPaths, experiment_id: str) -> list[EvidenceRecord]:
    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    summary_path = validation_dir / "summary.json"
    report_path = validation_dir / "graph.json"
    if not summary_path.exists() or not report_path.exists():
        raise EvidenceError(f"graph experiment {experiment_id!r} has not been validated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("blocking"):
        raise EvidenceError("graph validation contains blocking issues")
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.blocking or report.status is not ValidationStatus.PASSED:
        raise EvidenceError("graph validator did not pass")

    spec = load_graph_spec(paths, experiment_id)
    result = load_graph_result(paths, experiment_id)
    oracle = load_min_cut_oracle(paths, experiment_id)
    audit = load_flow_audit(paths, experiment_id)
    record = load_graph_record(paths, experiment_id)
    spec_path = graph_spec_file(paths, experiment_id)
    result_path = graph_result_file(paths, experiment_id)
    oracle_path = min_cut_oracle_file(paths, experiment_id)
    audit_path = flow_audit_file(paths, experiment_id)
    edge_path = edge_flows_file(paths, experiment_id)
    hashes = {
        "spec": sha256_file(spec_path),
        "result": sha256_file(result_path),
        "oracle": sha256_file(oracle_path),
        "audit": sha256_file(audit_path),
        "edge_flows": sha256_file(edge_path) if edge_path.exists() else "",
        "record": sha256_file(graph_record_file(paths, experiment_id)),
    }
    now = datetime.now(UTC).replace(microsecond=0)
    gap = abs(result.max_flow_value - oracle.minimum_cut_capacity)
    provenance = {
        "kind": "graph",
        "spec_path": spec_path.relative_to(paths.root).as_posix(),
        "spec_sha256": hashes["spec"],
        "result_path": result_path.relative_to(paths.root).as_posix(),
        "result_sha256": hashes["result"],
        "oracle_path": oracle_path.relative_to(paths.root).as_posix(),
        "oracle_sha256": hashes["oracle"],
        "audit_path": audit_path.relative_to(paths.root).as_posix(),
        "audit_sha256": hashes["audit"],
        "edge_flows_path": (
            edge_path.relative_to(paths.root).as_posix() if edge_path.exists() else ""
        ),
        "edge_flows_sha256": hashes["edge_flows"],
        "record_path": graph_record_file(paths, experiment_id)
        .relative_to(paths.root)
        .as_posix(),
        "record_sha256": hashes["record"],
        "graph_hash": spec.graph_hash,
        "spec_hash": spec.spec_hash,
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id or "",
    }

    def evidence(evidence_id: str, key: str, value: float | int | str) -> EvidenceRecord:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            experiment_id=experiment_id,
            artifact_path=provenance["result_path"],
            artifact_hash=hashes["result"],
            source_field=key,
            validation_reports=[report.validator],
            verified_at=now,
            provenance=provenance,
        )

    records: list[EvidenceRecord] = [
        evidence(
            f"ev-{experiment_id}-graph-node-count",
            "graph.node_count",
            len(spec.nodes),
        ),
        evidence(
            f"ev-{experiment_id}-graph-edge-count",
            "graph.edge_count",
            len(spec.edges),
        ),
        evidence(f"ev-{experiment_id}-graph-source", "graph.source", spec.source),
        evidence(f"ev-{experiment_id}-graph-sink", "graph.sink", spec.sink),
        evidence(f"ev-{experiment_id}-graph-hash", "graph.hash", spec.graph_hash),
        evidence(
            f"ev-{experiment_id}-graph-max-flow",
            "graph.max_flow",
            result.max_flow_value,
        ),
        evidence(
            f"ev-{experiment_id}-graph-min-cut",
            "graph.min_cut_capacity",
            oracle.minimum_cut_capacity,
        ),
        evidence(f"ev-{experiment_id}-graph-flow-cut-gap", "graph.flow_cut_gap", gap),
        evidence(
            f"ev-{experiment_id}-graph-min-cut-partition",
            "graph.minimum_cut_partition",
            json.dumps(oracle.canonical_partition),
        ),
        evidence(
            f"ev-{experiment_id}-graph-min-cut-count",
            "graph.minimum_cut_count",
            oracle.minimum_cut_count,
        ),
        evidence(
            f"ev-{experiment_id}-graph-min-cuts",
            "graph.minimum_cut_partitions",
            json.dumps(oracle.minimum_cut_partitions),
        ),
        evidence(
            f"ev-{experiment_id}-graph-min-cut-capacities",
            "graph.minimum_cut_partition_capacities",
            json.dumps(oracle.minimum_cut_partition_capacities),
        ),
        evidence(
            f"ev-{experiment_id}-graph-cut-enumeration",
            "graph.enumerated_cut_count",
            oracle.enumerated_cut_count,
        ),
        evidence(
            f"ev-{experiment_id}-graph-capacity-audit",
            "graph.capacity_audit",
            str(audit.capacity_ok).lower(),
        ),
        evidence(
            f"ev-{experiment_id}-graph-conservation-audit",
            "graph.conservation_audit",
            str(audit.conservation_ok).lower(),
        ),
        evidence(
            f"ev-{experiment_id}-graph-conservation-residual",
            "graph.conservation_max_residual",
            audit.max_absolute_residual,
        ),
        evidence(
            f"ev-{experiment_id}-graph-source-flow",
            "graph.source_flow_value",
            audit.source_flow_value,
        ),
        evidence(
            f"ev-{experiment_id}-graph-sink-flow",
            "graph.sink_flow_value",
            audit.sink_flow_value,
        ),
        evidence(f"ev-{experiment_id}-graph-algorithm", "graph.algorithm", result.algorithm),
        evidence(
            f"ev-{experiment_id}-graph-library-version",
            "graph.library_version",
            result.library_version,
        ),
        evidence(
            f"ev-{experiment_id}-graph-runtime",
            "graph.runtime_seconds",
            result.runtime_seconds,
        ),
    ]

    index = load_evidence_index(paths)
    for item in records:
        index.records[item.evidence_id] = item
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    graph_dir = paths.verified_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        graph_dir / f"{experiment_id}.json",
        json.dumps(
            {
                "experiment_id": experiment_id,
                "artifact_hash": hashes["result"],
                "records": [item.model_dump(mode="json") for item in records],
            },
            indent=2,
        )
        + "\n",
    )
    provenance_path = paths.verified_dir / "provenance.json"
    provenance_index = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {"schema_version": 1, "experiments": {}}
    )
    provenance_index["experiments"][f"graph:{experiment_id}"] = {
        **provenance,
        "validation_reports": {"graph": sha256_file(report_path)},
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")
    if not paths.verified_claims_file.exists():
        atomic_write_text(
            paths.verified_claims_file,
            yaml.safe_dump({"claims": []}, sort_keys=False, allow_unicode=True),
        )
    return records


__all__ = ["freeze_graph"]
