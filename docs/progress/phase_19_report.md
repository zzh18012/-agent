# Phase 19 — Graph / Network Vertical Slice

Date: 2026-11-15

## Goal

Make `TaskFamily.GRAPH` a first-class modeling paradigm using the shared lifecycle and
a maximum-flow / minimum-cut vertical slice on a frozen directed capacitated graph.

No new TaskFamily enum was added. Prediction, Optimization, and Simulation paths are
unchanged.

## 19.1 — Maximum Flow / Minimum Cut on a Frozen Directed Graph

### Lifecycle and CLI

`mmos init --task graph` and `mmos graph-run <project> <graph_spec.json>` drive the
shared Orchestrator:

INIT -> PROJECT_INGESTED -> PROBLEM_UNDERSTOOD -> PROTOTYPING -> PROTOTYPED ->
EXPERIMENT_COMPLETED -> VALIDATING -> VERIFIED -> EVIDENCE_FROZEN -> PAPER_DRAFTED ->
RED_TEAM_REVIEW -> READY_FOR_RELEASE -> RELEASED only on Judge PASS.

The real run used DeepSeek `analyze-problem`; the ProblemSpec correctly identified a
directed network, source/sink, capacities, maximum-flow objective, conservation, and
independent min-cut validation. It did not collapse the task into generic LP or
prediction semantics.

### Graph identity and schema

- `GraphProblemSpec`, `GraphEdge`, `EdgeFlow`, `NodeFlowAudit`, `FlowAuditResult`,
  `MinCutOracle`, `GraphResult`, and `GraphExperimentRecord`.
- Directed finite-capacity single-source/single-sink graph; parallel edges, self-loops,
  lower bounds, node capacities, and multi-source/sink are rejected.
- `graph_hash` covers problem type, directed flag, sorted nodes, sorted edges/capacities,
  source, and sink. `spec_hash` covers the full spec. Changing capacity, edge, or
  source/sink changes the hash.

### Solver and independent cut oracle

- NetworkX 3.7 `nx.maximum_flow(..., capacity='capacity', flow_func=edmonds_karp)`.
- Two in-process runs produce the same canonical flow digest; this is a repeatability
  check, not independent verification.
- The min-cut oracle is a pure-Python exhaustive enumeration over all 2^(n-2) legal s-t
  partitions. For each partition it sums only directed S -> V\S capacities. It does not
  call NetworkX maximum_flow/minimum_cut and does not read a residual network.
- The oracle records every partition and capacity, the minimum, all minimum cuts, and
  canonical/display partitions. The solver and oracle share the frozen spec, so this is
  code-path independence only, not data independence.

### Auditor

`mmos/validation/graph.py` independently checks hashes and experiment identity, edge-flow
set, capacities, negative/overshoot flows, per-node primal conservation, source/sink flow
values, full cut enumeration, min-cut partition capacities, all-cut recomputation,
max-flow/min-cut equality, and determinism metadata. It writes `flow_audit.json`,
`validation/graph.json`, and a blocking summary. No weighted score is used.

### Tracking, evidence, paper

- One MLflow run per graph experiment with graph/node/edge/source/sink/algorithm params,
  max-flow/min-cut/gap/runtime metrics, and artifacts.
- Verified evidence includes node/edge counts, source/sink, graph hash, max flow, min cut,
  flow-cut gap, minimum-cut partition(s)/count, capacity/conservation audits, algorithm,
  library version, and runtime.
- The paper is deliberately scoped as an internal plumbing/test-pass note, not a
  conformance report. It publishes the full 64-cut enumeration, per-cut crossing edges,
  canonical/display cut derivation, solver call, oracle pseudocode, and automated
  reconciliation checks. It states that the fully saturated symmetric construction forces
  max-flow = min-cut, that flow/conservation audits are self-consistency checks on the
  solver vector, and that evidence is private and not third-party reproducible.

### Live run

Project `graph-maxflow-release-20261115-20261115-bfb16f`:

- Experiment `exp-maxflow-benchmark-v1`; graph `maxflow-benchmark-v1`; task `graph`.
- 8 nodes, 10 edges; source `s`, sink `t`; capacities dimensionless, integer-valued.
- Graph hash `765204d4ec29148ff6484debd59147a527447cfd53f2708553e9da0cf908ae05`.
- NetworkX 3.7 `edmonds_karp`; MLflow run `bf0ae770a9a347b59419fa87aa0f31a3`.
- Max flow 12; independent min-cut capacity 12; gap 0.0 at stored precision.
- Full enumeration: 64/64 legal partitions; 12 minimum cuts, each capacity 12.
- Capacity audit passed (no violations); per-node conservation passed.
- Validation: `graph` passed. Evidence frozen.
- Claim `claim-graph-exp-maxflow-benchmark-v1` remains `proposed`; it is explicitly
  consistent-with, not evidence-of, correctness.
- Red Team final: 10 attacks (2 critical, 6 major, 2 minor), all pass, overall `pass`,
  blocking false. Release written to `releases/graph_exp-maxflow-benchmark-v1.{json,md}`;
  project `RELEASED`.
- Earlier blocked runs remain as audit history; the final run superseded them after
  paper/claim corrections.

## Known limitations / not done

- One small frozen graph, fully saturated and balanced by construction; equality is
  forced and cannot discriminate a correct solver from any valid saturated flow.
- No perturbed-capacity or asymmetric negative control; no non-forced min-cut instance.
- No second flow solver or LP cross-check; no algorithm comparison.
- No sensitivity analysis; the 1e-6 tolerance is not exercised.
- Cut oracle and solver share the frozen spec; independence is code-path only.
- Evidence is private and not third-party reproducible.
- Out of scope: PageRank, community detection, TSP/VRP, graph neural networks, dynamic
  graphs, multi-layer networks, graph databases, universal graph DSLs, and visualization
  as an evidence source.
