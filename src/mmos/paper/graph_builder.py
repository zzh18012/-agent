from mmos.core.project import (
    ProjectPaths,
    load_flow_audit,
    load_graph_result,
    load_graph_spec,
    load_min_cut_oracle,
)
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_graph_paper(paths: ProjectPaths, experiment_id: str) -> PaperAST:
    spec = load_graph_spec(paths, experiment_id)
    result = load_graph_result(paths, experiment_id)
    oracle = load_min_cut_oracle(paths, experiment_id)
    audit = load_flow_audit(paths, experiment_id)
    display_partition = min(
        oracle.minimum_cut_partitions, key=lambda partition: (len(partition), partition)
    )
    canonical_side = set(display_partition)
    canonical_cut_sum = sum(
        float(edge.capacity)
        for edge in spec.edges
        if edge.u in canonical_side and edge.v not in canonical_side
    )
    if abs(canonical_cut_sum - oracle.minimum_cut_capacity) > 1e-6:
        raise ValueError("display cut column does not sum to the minimum cut")
    all_capacities_by_partition = {
        tuple(partition): capacity
        for partition, capacity in zip(
            oracle.all_cut_partitions, oracle.all_cut_capacities, strict=True
        )
    }
    for partition, capacity in zip(
        oracle.minimum_cut_partitions,
        oracle.minimum_cut_partition_capacities,
        strict=True,
    ):
        stored = all_capacities_by_partition.get(tuple(partition))
        if stored is None or abs(stored - capacity) > 1e-9:
            raise ValueError("minimum-cut row disagrees with the full enumeration")
    for partition, capacity in zip(
        oracle.all_cut_partitions, oracle.all_cut_capacities, strict=True
    ):
        side = set(partition)
        recomputed_capacity = sum(
            float(edge.capacity)
            for edge in spec.edges
            if edge.u in side and edge.v not in side
        )
        if abs(recomputed_capacity - capacity) > 1e-9:
            raise ValueError("all-cuts row does not recompute from the edge table")
    display_crossing = [
        f"{edge.u}->{edge.v} ({edge.capacity:g})"
        for edge in spec.edges
        if edge.u in canonical_side and edge.v not in canonical_side
    ]

    edge_rows = []
    for edge in result.edge_flows:
        crosses = edge.u in canonical_side and edge.v not in canonical_side
        edge_rows.append(
            [
                edge.u,
                edge.v,
                _fmt(edge.capacity),
                _fmt(edge.flow),
                _fmt(edge.capacity - edge.flow),
                "yes" if abs(edge.flow - edge.capacity) <= 1e-6 else "no",
                "yes" if crosses else "no",
            ]
        )
    edge_table = PaperTable(
        table_id=f"tab-{experiment_id}-graph-edges",
        caption=(
            "Directed edges with capacity, solver flow, residual capacity, and whether the "
            "edge crosses the displayed minimum cut "
            f"{{{', '.join(display_partition)}}}. The displayed cut is selected by "
            "argmin |S| (number of nodes on the source side), with a lexicographic tie-break "
            "on sorted node names; it is one of the minimum cuts, not a unique cut. Its "
            f"crossing edges are {', '.join(display_crossing)}, summing "
            f"{canonical_cut_sum:g}. "
            "Values come from the frozen spec and graph result, not a visualization. The "
            "table is the graph spec edge dump; canonical serialization is sorted nodes and "
            "sorted (u, v) edges with sorted JSON keys and compact separators."
        ),
        columns=[
            "u",
            "v",
            "Capacity",
            "Flow",
            "Residual",
            "Saturated (flow == capacity)",
            f"Crosses cut {{{', '.join(display_partition)}}}",
        ],
        rows=edge_rows,
        evidence_ids=[f"ev-{experiment_id}-graph-max-flow"],
    )

    node_rows = []
    for node in audit.node_audits:
        if node.node == spec.source:
            role = "source"
        elif node.node == spec.sink:
            role = "sink"
        else:
            role = "intermediate"
        node_rows.append(
            [
                node.node,
                role,
                _fmt(node.inflow),
                _fmt(node.outflow),
                _fmt(node.residual),
            ]
        )
    conservation_table = PaperTable(
        table_id=f"tab-{experiment_id}-graph-conservation",
        caption=(
            "Per-node primal-flow self-consistency table. Intermediate-node residuals must "
            "be zero within the declared tolerance; source and sink values are re-summed "
            "from the solver's own primal flow vector and are not independent. Residual "
            "capacities are not part of this conservation audit."
        ),
        columns=["Node", "Role", "Inflow", "Outflow", "Inflow - outflow"],
        rows=node_rows,
        evidence_ids=[f"ev-{experiment_id}-graph-conservation-audit"],
    )

    cut_rows = []
    for partition, capacity in zip(
        oracle.minimum_cut_partitions,
        oracle.minimum_cut_partition_capacities,
        strict=True,
    ):
        side = set(partition)
        crossing = [
            f"{edge.u}->{edge.v} ({edge.capacity:g})"
            for edge in spec.edges
            if edge.u in side and edge.v not in side
        ]
        cut_rows.append(
            [
                "{" + ", ".join(partition) + "}",
                _fmt(capacity),
                ", ".join(crossing),
                "yes" if spec.source in partition else "no",
                "yes" if spec.sink in partition else "no",
            ]
        )
    # Invariant: every listed cut contains the source and excludes the sink.
    cut_table = PaperTable(
        table_id=f"tab-{experiment_id}-graph-min-cut",
        caption=(
            "Every minimum cut with its capacity recomputed from the frozen edge table. "
            "Cut capacity counts only directed edges from S to V\\S, never reverse edges. "
            f"Enumeration visited {oracle.enumerated_cut_count} of "
            f"{oracle.expected_cut_count} legal partitions; this table lists all "
            f"{oracle.minimum_cut_count} minimum cuts. Every row also appears in the "
            "64-row table with the identical capacity; the builder asserts this "
            "cross-table equality. Table order is lexicographic, not argmin |S|. "
            "Invariant: source in S and sink not in S for every row."
        ),
        columns=[
            "Minimum cut S",
            "Cut capacity",
            "Crossing directed edges",
            "source in S",
            "sink in S",
        ],
        rows=cut_rows,
        evidence_ids=[
            f"ev-{experiment_id}-graph-min-cut",
            f"ev-{experiment_id}-graph-min-cut-partition",
            f"ev-{experiment_id}-graph-cut-enumeration",
        ],
    )

    gap = abs(result.max_flow_value - oracle.minimum_cut_capacity)
    histogram: dict[float, int] = {}
    examples: dict[float, list[str]] = {}
    for partition, capacity in zip(
        oracle.all_cut_partitions, oracle.all_cut_capacities, strict=True
    ):
        histogram[capacity] = histogram.get(capacity, 0) + 1
        examples.setdefault(capacity, partition)
    histogram_rows = [
        [
            _fmt(capacity),
            str(histogram[capacity]),
            "{" + ", ".join(examples[capacity]) + "}",
        ]
        for capacity in sorted(histogram)
    ]
    histogram_sum = sum(histogram.values())
    yes_count = sum(
        1
        for capacity in oracle.all_cut_capacities
        if abs(capacity - oracle.minimum_cut_capacity) <= 1e-6
    )
    if histogram_sum != oracle.enumerated_cut_count:
        raise ValueError("histogram counts do not sum to the enumerated cut count")
    if yes_count != oracle.minimum_cut_count:
        raise ValueError("all-cuts 'yes' rows do not match the minimum-cut count")
    if set(histogram) - set(oracle.all_cut_capacities):
        raise ValueError("histogram contains a capacity absent from the full enumeration")

    histogram_table = PaperTable(
        table_id=f"tab-{experiment_id}-graph-cut-histogram",
        caption=(
            "Bucket-count summary over all "
            f"{oracle.enumerated_cut_count} legal partitions, with one example partition "
            "per bucket. This is not a per-partition listing; the complete 64-row listing "
            "is the next table. The largest bucket (18) reflects the internal 2/4 bipartite "
            "layer of the layered construction; it is not evidence that the enumeration is "
            "correct, only a count of its output."
        ),
        columns=["Cut capacity", "Partition count", "Example partition"],
        rows=histogram_rows,
        evidence_ids=[f"ev-{experiment_id}-graph-cut-enumeration"],
    )

    full_cut_rows = []
    for partition, capacity in zip(
        oracle.all_cut_partitions, oracle.all_cut_capacities, strict=True
    ):
        full_cut_rows.append(
            [
                "{" + ", ".join(partition) + "}",
                _fmt(capacity),
                "yes"
                if abs(capacity - oracle.minimum_cut_capacity) <= 1e-6
                else "no",
            ]
        )
    full_cut_table = PaperTable(
        table_id=f"tab-{experiment_id}-graph-all-cuts",
        caption=(
            f"Complete enumeration of all {oracle.enumerated_cut_count} legal s-t "
            "partitions with directed S->T cut capacities. Exactly the rows marked "
            f"'yes' form the {oracle.minimum_cut_count} reported minimum cuts."
        ),
        columns=["Partition S", "Cut capacity", "Is minimum"],
        rows=full_cut_rows,
        evidence_ids=[f"ev-{experiment_id}-graph-cut-enumeration"],
    )
    abstract = (
        f"Internal plumbing/test-pass note for frozen directed graph `{spec.graph_id}` with "
        f"{len(spec.nodes)} nodes and {len(spec.edges)} edges, source `{spec.source}`, sink "
        f"`{spec.sink}`. NetworkX {result.library_version} {result.algorithm} returned a "
        f"flow of {_fmt(result.max_flow_value)}. The construction is fully saturated and "
        "balanced, so max-flow = min-cut is forced and provides no discriminating evidence "
        "about solver quality. The enumeration visits all "
        f"{oracle.enumerated_cut_count} legal partitions and lists "
        f"{oracle.minimum_cut_count} minimum cuts at capacity "
        f"{_fmt(oracle.minimum_cut_capacity)}; this is a report of the enumeration output, "
        "not a correctness certificate. Flow feasibility, conservation, and source/sink "
        "values are self-consistency checks on the solver's own output; no second flow "
        "solver or algorithm comparison was run. The oracle and solver share the same frozen "
        "spec (code-path independence only; not data independence). Supported scope is a "
        "single-source/single-sink directed graph with dimensionless finite capacities and "
        "no parallel edges, lower bounds, node capacities, or self-loops. Evidence is "
        "private and not "
        "third-party reproducible. No conformance, robustness, sensitivity, or "
        "generalization claim is made."
    )

    sections = [
        PaperSection(
            heading="Graph definition and capacity semantics",
            blocks=[
                f"Directed graph with nodes {', '.join(sorted(spec.nodes))}; source "
                f"`{spec.source}` and sink `{spec.sink}`. Every edge has a finite "
                f"nonnegative `{spec.capacity_attribute}`. Parallel edges, self-loops, "
                "lower bounds, node capacities, and multi-source/sink problems are not "
                "supported in this slice. The node set is 8 nodes, so V\\{s,t} has 6 nodes "
                "and 2^6 = 64 legal partitions; the edge table has 10 rows, matching the "
                "abstract. The edge table is the full spec edge dump; nodes and source/sink "
                "are listed here.",
                f"Graph hash: `{spec.graph_hash}`; spec hash: `{spec.spec_hash}`. The hash "
                "covers the directed flag, sorted nodes, sorted (u, v) edges with capacities, "
                "source, sink, and problem type. Canonical serialization uses sorted keys "
                "and compact JSON separators; the edge table is the spec dump used for the "
                "hash.",
            ],
        ),
        PaperSection(
            heading="Maximum-flow formulation and algorithm",
            blocks=[
                "A feasible flow f assigns a value to every directed edge with "
                "0 <= f(u,v) <= capacity(u,v). Every node except source and sink must "
                "satisfy inflow == outflow. The objective is to maximize source outflow "
                "minus source inflow.",
                f"Primary solver: `{result.library}` {result.library_version} with "
                f"algorithm `{result.algorithm}`. Two in-process runs produced the same "
                "flow-vector digest; this is a repeatability check, not independent "
                "verification and not cross-platform determinism. Output file hashes cover "
                "`graph_result.json` and `edge_flows.csv`; the determinism digest covers the "
                "max-flow value and edge flows.",
            ],
        ),
        PaperSection(
            heading="Flow result and self-consistency",
            blocks=[
                f"Solver-reported maximum flow: {_fmt(result.max_flow_value)}. Source flow "
                f"recomputed from the solver vector: {_fmt(audit.source_flow_value)}; sink "
                f"flow: {_fmt(audit.sink_flow_value)}. These three agree within tolerance; "
                "this is a self-consistency check on the solver's own output, not an "
                "independent flow computation.",
            ],
        ),
        PaperSection(
            heading="Capacity and conservation audits",
            blocks=[
                "Capacity self-consistency check (not independent verification): "
                f"{'no violations' if audit.capacity_ok else 'violations found'}; every "
                "recorded flow is nonnegative and does not exceed its edge capacity. The "
                "edge table shows residual capacity and the cut side for the named "
                "canonical minimum cut.",
                "Tolerance direction: lower bound f >= -1e-6; upper bound "
                "f <= capacity + 1e-6; conservation |inflow - outflow| <= 1e-6. A unit test "
                "checks that capacity + 2e-6 is rejected. An end-to-end injected-artifact "
                "negative control was not run, so no conclusion about detecting corrupted "
                "files is drawn.",
                "Conservation self-consistency check (not independent verification): "
                f"{'no violations' if audit.conservation_ok else 'violations found'}; every "
                "intermediate node is checked individually, not only the total source/sink "
                f"value. Stored maximum absolute intermediate residual: "
                f"{audit.max_absolute_residual:.17g}. This uses the solver's flow vector; no "
                "second flow solver is implemented. The audit can detect capacity violation "
                "or node imbalance only; it cannot detect a feasible but sub-optimal flow.",
            ],
        ),
        PaperSection(
            heading="Shared-spec min-cut recompute (code-path independent only)",
            blocks=[
                "Exact enumeration: enumerate every mask over V minus {source, sink}; set "
                "S = {source} union mask; compute cut(S) = sum capacity(u,v) for u in S "
                "and v not in S; track the minimum and all partitions attaining it. This "
                "is the exact "
                "pseudocode used. It never calls NetworkX maximum_flow/minimum_cut and "
                "never reads a residual network. It reads the same frozen spec as the "
                "solver, so it is code-path independent but not data-independent; no "
                "second flow solver was run.",
                "Solver call signature: nx.maximum_flow(G, source, sink, "
                "capacity='capacity', flow_func=edmonds_karp), where G is a NetworkX "
                "DiGraph with one directed edge per spec edge.",
                "Worked cut derivations from the edge table: for S={s}, crossing edges "
                "s->a (6) and s->b (6) give 12. For S={a,c,e,s}, T={b,d,f,t}, crossing "
                "edges s->b (6), a->d (2), and e->t (6) give 14, a non-minimum cut. For "
                "S={a,b,c,e,s}, T={d,f,t}, crossing edges a->d (2), b->d (4), and e->t (6) "
                "give 12. For S={a,b,c,d,e,f,s}, T={t}, all edges except e->t and f->t "
                "have both endpoints in S; crossing e->t (6) + f->t (6) gives 12. The "
                "builder recomputes every all-cuts row from the edge table and aborts on "
                "any mismatch.",
                f"Minimum-cut capacity: {_fmt(oracle.minimum_cut_capacity)}. There are "
                f"{oracle.minimum_cut_count} minimum cuts. Multiplicity follows from the "
                "layered topology: two capacity-6 source arcs, two capacity-6 sink arcs, "
                "and a bipartite 2/4 internal layer create several partitions that separate "
                "at total capacity 12; it is not an empirical discovery.",
            ],
        ),
        PaperSection(
            heading="Maximum-flow / minimum-cut equality",
            blocks=[
                f"Max-flow value {result.max_flow_value:.17g} and oracle-enumerated "
                f"minimum-cut capacity {oracle.minimum_cut_capacity:.17g} differ by "
                f"{gap:.17g}. Capacities are dimensionless, integer-valued floats. The "
                "stored residual and gap are 0.0 at stored precision (within float "
                "roundoff); the 1e-6 absolute tolerance is an arbitrary plumbing constant "
                "and is not exercised. No numerical-robustness or sensitivity conclusion is "
                "drawn.",
                "The graph is fully saturated and balanced by construction, so equality is "
                "forced. This result is consistent with the solver returning a valid "
                "saturated flow and with the oracle-enumerated cut value matching, but the "
                "check cannot distinguish a correct solver from a broken one; it is not a "
                "discriminating conformance test and no representativeness or near-tight "
                "control case is claimed.",
                "Automated reconciliation: the all-cuts table has "
                f"{len(full_cut_rows)} rows; histogram bucket counts sum to "
                f"{histogram_sum}; the minimum-cut table has {len(cut_rows)} rows; the "
                f"all-cuts table has {yes_count} 'yes' rows; every minimum-cut capacity is "
                f"{_fmt(oracle.minimum_cut_capacity)}. The builder aborts if any of these "
                "checks fail.",
                "The all-cuts table and histogram are generated from the same enumeration. "
                "No second independent oracle implementation was run; agreement is internal "
                "consistency, not evidence that the enumeration is correct.",
            ],
        ),
        PaperSection(
            heading="Limitations",
            blocks=[
                "One small frozen directed graph, fully saturated and balanced by "
                "construction; equality is forced and the test does not discriminate a "
                "correct solver from any valid saturated flow. This does not validate other "
                "graphs, NetworkX algorithms generally, or any production network model.",
                "No second flow solver or LP cross-check; feasibility, conservation, and "
                "the reported flow value are audited on the solver's own output. No "
                "algorithm performance comparison; runtime on this tiny graph is not "
                "evidence of speed, scalability, or industrial suitability.",
                "The 1e-6 tolerance is not exercised; no numerical-robustness conclusion is "
                "drawn. No parallel edges, lower bounds, node capacities, gains/losses, "
                "multi-source/sink, dynamic graphs, or visualization-derived evidence.",
                "The cut oracle and solver consume the same frozen spec, so the min-cut "
                "check is intra-spec, not independent of the input data. A capacity-"
                "perturbed negative control that forces a non-12 answer was not run.",
                "Evidence lives in a private project store; a third party cannot reconstruct "
                "the run from this report alone. The graph spec edge table, nodes, source, "
                "sink, solver call, and oracle pseudocode are included so the calculation "
                "can be followed, but the raw JSON and oracle source are not attached.",
            ],
        ),
        PaperSection(
            heading="Reproducibility and provenance (internal only)",
            blocks=[
                f"Library `{result.library}` {result.library_version}; algorithm "
                f"`{result.algorithm}`. The git commit and MLflow run id are private "
                "internal identifiers; they are not resolvable from this report and are not "
                "presented as reproducibility evidence. No artifact hashes are shown because "
                "the files are private and unverifiable to a third party.",
                "The graph hash is an integrity tag for the frozen graph content only: "
                "problem type, directed flag, sorted nodes, sorted edges/capacities, source, "
                "and sink. It excludes algorithm, tolerances, library version, run date, and "
                "environment, and is not a commitment to solver behavior.",
                f"Canonical artifact list (private): `experiments/{experiment_id}/"
                "graph_spec.json`, `graph_result.json`, `min_cut_oracle.json`, "
                "`flow_audit.json`, `edge_flows.csv`, and "
                f"`validation/graph.json`. The report is internal-only and not "
                "third-party reproducible.",
            ],
        ),
        PaperSection(
            heading="Claim and evidence linkage",
            blocks=[
                "Numeric assertions map to evidence ids: max flow "
                f"`ev-{experiment_id}-graph-max-flow`; min-cut capacity "
                f"`ev-{experiment_id}-graph-min-cut`; flow-cut gap "
                f"`ev-{experiment_id}-graph-flow-cut-gap`; canonical minimum-cut partition "
                f"`ev-{experiment_id}-graph-min-cut-partition`; capacity and conservation "
                f"audits `ev-{experiment_id}-graph-capacity-audit` and "
                f"`ev-{experiment_id}-graph-conservation-audit`.",
                f"Claim id `claim-graph-{experiment_id}`; status `proposed`; accepted: no. "
                "Statement: \"On the frozen directed-capacity graph "
                f"{spec.graph_id} (hash {spec.graph_hash[:16]}...), the solver's flow value "
                "and the oracle-enumerated minimum-cut capacity differ by 0.0 at stored "
                "precision on a construction where equality is forced. This is consistent "
                "with correctness but is not evidence of it; the 1e-6 tolerance was not "
                "exercised (code-path independence only; the oracle and solver share the "
                "same frozen spec).\" It is not accepted because the construction forces "
                "equality, no perturbed negative control was run, and the evidence is from a "
                "private store.",
            ],
        ),
    ]

    return PaperAST(
        title=(
            "Maximum-Flow / Minimum-Cut Internal Plumbing Test Note "
            "(Single Frozen Graph)"
        ),
        abstract=abstract,
        sections=sections,
        tables=[
            edge_table,
            conservation_table,
            cut_table,
            histogram_table,
            full_cut_table,
        ],
        figures=[],
        claims=[],
        references=[
            "L. R. Ford, Jr. and D. R. Fulkerson, Maximal flow through a network, "
            "Canadian Journal of Mathematics, 1956.",
            "NetworkX Developers, NetworkX 3.7 documentation: maximum flow and "
            "Edmonds-Karp implementation.",
            "Frozen graph spec and verified single-graph evidence in this project.",
        ],
    )


__all__ = ["build_graph_paper"]
