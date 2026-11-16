# Phase 21 — Automatic Problem Router and Mixed Modeling DAG

Date: 2026-11-15 (21.1), 2026-11-16 (21.2)

## Goal

Classify an unfamiliar mathematical-modeling problem into an existing TaskFamily,
MULTI_TASK, AMBIGUOUS, or UNSUPPORTED without executing a model. The Router is a
control-plane component: it produces routing artifacts, not scientific Verified
Evidence.

No new TaskFamily was added. Prediction, Forecasting, Optimization, Simulation, Graph,
and Multi-objective implementations were left unchanged.

## 21.1 — Structured Signals + Deterministic Routing Policy

### Architecture

Raw problem -> existing Problem Analyst -> ProblemSpec -> structured LLM
`ProblemSignalExtraction` -> deterministic routing policy -> `RoutingDecision` ->
Routing Auditor.

The LLM produces semantic signals only. It never names a TaskFamily directly. The
deterministic policy maps signals to routes; the auditor checks policy invariants and
signal consistency.

### Signals

Per subproblem, `ProblemSignals` records supervised target, future prediction, temporal
order, decision variables, hard constraints, objective count, simultaneous trade-off
objectives, preference availability, stochastic process evolution, replication/
distributional requirements, graph-native/primary structure, evaluation metrics,
insufficient information, missing information, and unsupported requirements.

The extraction prompt explicitly distinguishes:
- evaluation metrics from objectives;
- a threshold/constraint from a trade-off objective;
- future temporal prediction from ordinary time-featured prediction;
- native graph structure from graph keywords in "neural network";
- historical simulation data from an actual simulation study;
- simulation configuration from optimization decision variables;
- ordinary neural forecasting from unsupported graph neural networks/deep RL;
- deep reinforcement learning/policy training as an unsupported paradigm.

The extractor uses temperature 0 and an exact JSON output template to prevent schema
drift.

### Deterministic routing policy

- Native graph problem -> GRAPH, without adding optimization merely because flows/
  capacities exist.
- Decision + hard constraints + at least one objective:
  - two or more simultaneous trade-off objectives with no preference -> MULTI_OBJECTIVE;
  - otherwise -> OPTIMIZATION (a threshold objective stays a constraint).
- Stochastic process evolution + replication/distributional output -> SIMULATION.
- Future prediction with chronological order -> FORECASTING; otherwise supervised target
  -> PREDICTION.
- Multiple required families -> MULTI_TASK, with dependency hints; execution blocked
  pending a Mixed DAG.
- No supported signals with missing information -> AMBIGUOUS; no task is selected.
- Unsupported core requirement -> UNSUPPORTED; no nearest-family fallback.
- Insufficient information takes precedence over guessed single-task routing.
- No confidence probabilities are fabricated.

Capability is separate from routing. Capability reports list each family's current
implemented scope and FULL/PARTIAL readiness. MULTI_TASK is blocked pending Phase 21.2.

### CLI and artifacts

- `mmos route <project-or-problem-file>` runs Problem Analyst + signal extraction +
  deterministic routing + audit and writes `routing/problem_signals.json`,
  `routing/routing_decision.json`, and `routing/routing_audit.json` (plus the analyzed
  ProblemSpec) under the project or a file-derived routing directory.
- `mmos route-benchmark <manifest>` runs the frozen suite twice.
- No model is auto-executed. SINGLE_TASK decisions provide a suggested entrypoint only.
- Routing hashes include problem content and ProblemSpec hash; provider/model/git/time
  are recorded. Routing artifacts are kept out of Verified Evidence.

### Routing benchmark

- Suite `routing-suite-v1`, version 1.0.0, 24 cases, manifest hash
  `63cd847101c7c736a084b4d943b4ce59ce1715cd436870e1451a0732ac40beae`.
- Composition: 18 canonical single-family cases (3 per family), 3 mixed cases, 2
  ambiguous cases, 1 unsupported case. Includes paraphrases and adversarial wording
  (neural network, historical simulation data, multiple metrics).
- Gold routes are frozen in the manifest; no gold label or case id is shown to the LLM.
  Case problems are analyzed with opaque ids.
- Real DeepSeek `deepseek-chat` structured outputs; ProblemSpec analyzed once per case and
  cached, then two independent signal+routing passes are run. Temperature 0.
- Run directory: `routing_benchmark/runs/run_20261115`.

### Final benchmark results

Both runs:

- Status accuracy: 1.000
- Single-task exact accuracy: 1.000
- Multi-task detection recall: 1.000
- Ambiguous detection: 1.000
- Unsupported detection: 1.000
- Critical unsafe routing count: 0
- Structured route stability: 1.000
- Confusion: AMBIGUOUS->AMBIGUOUS 2; SINGLE_TASK->SINGLE_TASK 18;
  MULTI_TASK->MULTI_TASK 3; UNSUPPORTED->UNSUPPORTED 1.
- Misroutes: none.

The benchmark iterations exposed and fixed real router issues rather than changing gold
labels:
1. schema drift on an ambiguous case -> exact output template + temperature 0;
2. native graph problems double-routed with optimization -> graph-native priority;
3. under-specified "best plan" guessed as optimization -> insufficient-information
   fail-safe to AMBIGUOUS;
4. queue simulation signals split across stochastic and replication subproblems ->
   global simulation signal combination;
5. ordinary neural forecasting marked unsupported -> supported neural predictor
   clarification;
6. deep RL policy training routed as optimization/simulation -> explicit unsupported
   paradigm.

## 21.2 — Mixed Modeling DAG: Forecasting -> Optimization

### Scope and architecture

The first formal mixed vertical slice is exactly one path: a forecasting node produces a
typed demand forecast artifact and an optimization node consumes it to solve a
production-planning model. No new TaskFamily was added. MULTI_TASK is an execution mode:
`ProjectConfig.execution_mode` defaults to `single_task`; `--mode multi_task` selects the
shared optimization-style lifecycle (`PROBLEM_UNDERSTOOD -> PROTOTYPING -> PROTOTYPED ->
EXPERIMENT_COMPLETED -> VALIDATING -> VERIFIED -> EVIDENCE_FROZEN -> PAPER_DRAFTED ->
RED_TEAM_REVIEW -> READY_FOR_RELEASE -> RELEASED`). All existing experiment, validation,
MLflow, evidence, paper, Red Team, and PASS-only release machinery is reused.

### Formal schemas and planner

`src/mmos/schemas/mixed.py` defines `NodeStatus`, `DAGStatus`, `MixedSpec`, `TaskNode`,
`ArtifactEdge`, `MixedExecutionPlan`, `ArtifactEnvelope`, `ForecastArtifact`,
`ArtifactContract`, and `DAGRunRecord`. `MixedSpec` enforces horizon = timestamps =
planning periods, aligned costs/capacities, nonnegative values, and temporal-baseline
only forecasts.

`src/mmos/mixed/plan.py` requires a MULTI_TASK routing decision with exactly
`{FORECASTING, OPTIMIZATION}`, a forecasting -> optimization dependency hint, and exactly
one subproblem route per family. It builds `forecast_node` and `optimization_node` with one
typed artifact edge, rejects cycles through Kahn topological order, and hashes the frozen
plan. The live run exposed a real planning bug: the plan hash was computed before the final
`plan_id` was assigned, so saved plans failed hash verification; the fix assigns the id
first and then hashes the final content.

### Typed artifacts and contract

The forecast node publishes a `ForecastArtifact` with point forecasts, target, unit,
timestamps, model id, source dataset hash, and producer experiment id. The envelope records
artifact type/schema version, producer identity, payload hash, content hash, validation
status, and provenance. The consumer accepts the artifact only after checking type,
version, verified status, payload hash, content hash, experiment identity, target, unit,
horizon, and timestamp alignment. Any mismatch raises `ArtifactContractError` and blocks
the downstream node; there is no fallback path.

The envelope provenance also records the exact history window length and a SHA-256 digest
of the canonical `(period, demand)` pairs. The paper recomputes that digest from the
displayed history and fails if it does not match.

### Executor

`src/mmos/mixed/executor.py` executes nodes in topological order and writes a
`DAGRunRecord` after every state change. The forecasting node:

- runs the data gate;
- builds a rolling-origin `ExperimentSpec` with `naive.drift` and the `naive.last_value`
  comparison baseline;
- runs the existing Snakemake workflow and validators;
- freezes forecast evidence;
- builds the future point forecast from the full validated history.

The optimization node re-verifies the artifact contract, builds a continuous
inventory-balance production-planning model with per-period production and inventory,
capacity upper bounds, and production + holding cost minimization, then reuses the
OR-Tools CBC runner, optimization validator, MLflow logging, and evidence freeze. The
optimization record stores consumed artifact ids/hashes and
`dependency_contract_status="passed"`.

Two executor bugs were found by the live run: `_update_node` accidentally applied the new
status and experiment id to every node, and the validator assumed node order rather than
looking up the forecasting node by family. Both were fixed; the run record now shows the
forecast experiment only on the forecast node.

Upstream failures, missing artifacts, and non-succeeded producers produce
`BLOCKED_UPSTREAM`/`FAILED`/`BLOCKED` records. Successful runs with a changed forecast
artifact hash mark earlier same-plan runs `STALE`.

### Validation, evidence, and paper

`src/mmos/mixed/validation.py` independently re-verifies the plan hash, DAG completion,
artifact contract and hashes, upstream validation and frozen evidence, consumed artifact
identity, optimization instance hash and status, every inventory-balance RHS against the
artifact point forecasts, capacity/nonnegativity bounds, the objective value, and
provenance completeness. The balance constraints must equal the artifact demand, which
detects a silent fallback. The report is regenerated from frozen artifacts only.

`src/mmos/evidence/mixed_store.py` writes mixed DAG evidence with plan, run, artifact,
forecast-spec, optimization-spec/result/record hashes and artifact consumption fields. The
paper builder emits a DAG figure, a forecast/capacity/production/inventory figure, input
history, forecast, fold-derivation, window-sensitivity, parameter, plan, balance, and
contract tables. It states that the forecast is a deterministic point forecast; that the
plan is conditional on the frozen artifact; that production cost and holding cost rates
apply per unit and per unit-period; that the validator is an internal arithmetic check;
and that no uncertainty propagation, robust planning, or operational recommendation is
claimed.

### CLI and review resume

- `mmos mixed-plan <project> <mixed_spec.json>` routes the problem with the real Router,
  freezes the plan, and advances PROBLEM_UNDERSTOOD -> PROTOTYPING -> PROTOTYPED.
- `mmos run-plan <project>` executes the frozen plan. There is no automatic
  Router -> execution path and no `mmos solve`.
- Review resume enters RED_TEAM_REVIEW, verifies the frozen evidence/artifact hashes and
  current plan, re-runs validation without re-executing nodes, and rejects changed
  evidence with REQUIRES_HUMAN_REVIEW.

### Frozen suite

`benchmarks/mixed/suite_v1/manifest.json` defines `mixed-suite-v1` with one case,
`forecast_to_production_plan`: monthly demand (24 rows), forecast horizon 3
(2026-01..03), `naive.drift`, 4 rolling-origin splits, unit `units`, capacities
170/140/170, production cost 12, holding cost 2, initial inventory 0. The case expects a
MULTI_TASK route with a forecasting -> optimization dependency. Manifest hashes cover the
problem text, data, and spec.

### Tests

`tests/test_mixed.py` covers:

1. `MixedSpec` horizon/timestamp/seasonal consistency rejection;
2. MULTI_TASK routing requirement in the planner;
3. topological order and cycle rejection before execution;
4. artifact contract mismatches (horizon, timestamps, unit, target, payload mutation,
   missing payload);
5. missing/invalid upstream artifact -> BLOCKED_UPSTREAM with no downstream execution;
6. MULTI_TASK lifecycle order;
7. end-to-end pipeline with a FAIL review then PASS resume (asserts no node rerun,
   forecast node experiment identity, inventory carry-over, evidence, claim, release);
8. validator detecting demand/spec tampering (no fallback);
9. resume rejecting changed evidence without re-executing nodes.

### Live run

Project: `projects/mixed-forecast-plan-20261116c-20261116-7a954e`.

- Real DeepSeek Problem Analyst and Router: MULTI_TASK.
- Plan: `plan-4b05fa090089`, hash
  `656ff9cb6462f1c23a6c212dbf4bbaa678b3bbafca20b0d27deae54bc3bdaa16`.
- Run: `mixed-656ff9cb-3d3219`; one DAG run only.
- Forecast artifact: `artifact-mixed-656ff9cb-3d3219-demand-forecast`, content hash
  `c184a004bef94731830ba5fa22740b34a23961cdd8a25c3f9108f839069a8bb0`.
- Forecast points: 148, 150, 152 units.
- Plan: production 158, 140, 152; ending inventory 10, 0, 0; objective 5420 (production
  5400 + holding 20), feasible; CBC optimal, zero gap.
- Validation: `mixed: passed`; evidence frozen; claim proposed (not accepted).
- Red Team: PASS after seven blocked review cycles. The reviews drove real scientific tightening:
  the first frozen instance made optimization a pass-through, so capacity 140 < 150 was
  introduced to force carry-over; the paper then added forecast error metrics and a
  baseline, explicit inventory recurrence, fold-level reconstruction with realized test
  demand and MASE scales, window-length sensitivity, provenance digest distinctions,
  constraint-enforcement and reproducibility scope limits, and figure annotations. Final
  release: `releases/mixed_mixed-656ff9cb-3d3219.md`.

### Limitations

- One small synthetic instance; no demand, cost, or capacity sweep, and no negative
  control (mutated artifact, schema bump, or infeasible companion solve).
- The pre-build is forced by the hard demand equality and the capacity shortfall; it is a
  feasibility identity, not a planning-mechanism result.
- Deterministic point forecast only; no uncertainty propagation, stock-out, backorder, or
  robust/stochastic planning.
- The declared drift model is not shown to beat the last-value baseline; fold metrics are
  not a forecasting evaluation.
- The artifact hashes establish internal contract provenance; the raw CSV and evaluation
  code are not attached, so external reproduction is not claimed.
- Simulation-aware or multi-node mixed DAGs remain future work; no automatic `mmos solve`.

### Verification

- `uv run pytest tests/test_mixed.py`: 9 passed.
- Full `uv run pytest`: passed.
- `uv run ruff check .`: passed.
- `uv run pyright`: 0 errors.

## 21.3 — Mixed Modeling DAG: Forecasting -> Optimization -> Simulation

### Scope and architecture

The DAG now has three nodes and three artifact edges: forecasting produces a point
forecast and a residual artifact; optimization consumes the forecast and produces a
`ProductionPlanArtifact`; simulation consumes the frozen plan and the residual artifact.
No new TaskFamily was added. The slice supports only this path; no branching, arbitrary DAG
engine, scenario-tree optimization, robust optimization, or `mmos solve` was introduced.
A two-node plan remains the default when `simulation_replications` is absent, preserving
the Phase 21.2 behavior.

### Artifacts and simulation

- `ProductionPlanArtifact` records planning horizon, timestamps, production, planned
  inventory, capacities, unit, objective, forecast identity, and optimization experiment.
- `ForecastErrorArtifact` records `residual = actual - forecast`, source experiment and
  forecast artifact identity, fold id, row index, timestamp, actual, forecast, residual,
  sample count, mean, and sample standard deviation. The artifact is built only from
  rolling-origin test rows; the builder rejects training rows.
- The mixed simulation spec freezes the plan/forecast/error artifact ids and hashes,
  horizon, unit, replications, master seed, empirical iid sampling method, demand floor
  policy, metrics, and confidence level.
- Simulation never re-optimizes production. Each replication samples H residuals with
  replacement, forms `max(0, point forecast + residual)` demand, and applies lost-sales
  dynamics: `available = opening + production`, `served = min(available, demand)`,
  `shortage = max(0, demand - available)`, `ending = available - served`.
- Seeds reuse the formal sha256 master-seed schedule; two in-process runs are compared by
  a replication digest. Metrics include total shortage, shortage-period count, fill rate,
  ending/average inventory, and maximum shortage. Implausible high-precision risk claims
  are deliberately not made; the paper reports finite-sample diagnostics.

### Validation, evidence, paper, and lifecycle

`validate_mixed_simulation_experiment` independently checks the plan and residual
contracts, cross-task unit/horizon/timestamp alignment, plan immutability, OOF residual
provenance against `predictions.parquet` and fold windows, seed schedule uniqueness and
hash, replication completeness, inventory/shortage arithmetic, metric recomputation,
consumed artifact identity, and provenance. `validate_mixed_run` additionally checks the
DAG-level links and run summary.

Mixed evidence now includes plan, residual, simulation, seed, shortage, fill, contract
chain, and path-diversity fields. Staleness is directional: a changed forecast stales
optimization and simulation; a changed residual or plan stales only simulation.
`paper/evidence_bundle` is written with the raw inputs, manifest, frozen spec, predictions,
residual pool, replication CSV, DAG evidence, code, and `SHA256SUMS`.

### Tests and checks

`tests/test_mixed_simulation.py` adds 11 tests: three-node topology, residual direction,
training-row rejection, manual inventory/service arithmetic, seed reproducibility, no
optimization call, unit/horizon blocking, residual-signature stale propagation (residual
change stales simulation only; forecast change stales both; plan change stales simulation),
three-node end-to-end lineage/release resume, and evidence-change resume rejection. The
Phase 21.2 tests remain green.

- Full `uv run pytest`: 282 passed, 3 integration deselected.
- `uv run ruff check .`: passed.
- `uv run pyright`: 0 errors.

### Live three-node run and release gate

Authoritative project: `projects/mixed-forecast-plan-sim-20261116d-20261116-828a5b`.

- Real DeepSeek Problem Analyst/Router and real Snakemake, CBC, MLflow, validation, and
  evidence execution.
- Plan `plan-b031a5dd1eaa`, hash `a5aad5c12219e843557dc9c7b245658d8f3863564b7954fe577fc6ead1629a3e`.
- Run `mixed-a5aad5c1-73b2e9`, status SUCCEEDED.
- Forecast artifact `artifact-mixed-a5aad5c1-73b2e9-demand-forecast`, content hash
  `1d735723db38f2c217a6c633961f62e8d2f4255a45ede32ffa9e7790719abfd2`.
- Residual artifact `artifact-mixed-a5aad5c1-73b2e9-forecast-error`, content hash
  `fe2c7c1015744377d5ea019f8b23d0e2f571761ca9be97e3f221ce2c088ffed7`;
  12 historical rolling-origin rows, mean residual -1.267157, sample std 6.026062.
- Plan artifact `artifact-mixed-a5aad5c1-73b2e9-production-plan`, content hash
  `766e9e1dd46b40bcd0821a860a73a4fe40fd6c0a25007b4dfb07969c6abf5557`;
  production 158/140/152, inventory 10/0/0, objective 5420, feasible.
- Experiments: `exp-plan-b031a5dd1eaa-forecast`, `…-optimization`, `…-simulation`.
- Simulation: 500 replications, seed hash
  `e183867d704b29e4e2e2ae5fdb8380393d719b16afd3fa545f79915410f84f19`, 430 distinct
  residual paths, zero demand-floor activations. Diagnostic mean total shortage 3.444971,
  median 0, p95 14; mean fill rate 0.992476; any-shortage fraction 0.48. These are
  in-sample residual-resampling diagnostics, not deployed-forecast risk estimates.
- Evidence: `verified/mixed/mixed-a5aad5c1-73b2e9.json`; bundle:
  `paper/evidence_bundle/`.
- Validation: `mixed: passed`, `mixed_simulation: passed`; contract chain complete;
  provenance complete.
- Red Team: **PASS**; project state **RELEASED**.
- Release: `releases/mixed_mixed-a5aad5c1-73b2e9.md`.

The Red Team repeatedly blocked release after the live run and drove real evidence
strengthening: explicit in-sample residual labeling, fixed-anchor slope/window sensitivity
tables, per-period error tables, centered-pool and mechanical-fill diagnostics, 3-decimal
precision, bundled raw inputs/predictions/replications/code, and a long list of scope
limits. The final review passed without weakening the gate: no robust-optimization,
service-guarantee, capacity-enforcement, out-of-sample-risk, or cross-model-superiority
claim is made. Remaining limitations below are disclosed rather than hidden.

### Bug fixes in scope

The three-node live run and tests exposed and fixed: missing-input guards for optimization
and simulation artifacts, a validator that assumed node order instead of the forecasting
node by family, and test construction errors around artifact content hashes when asserting
unit/timestamp mismatches. No re-optimization path was introduced.

### Limitations

- One hand-built synthetic instance; no demand/cost/capacity sweep, no comparator plan,
  no mode/fault-injection test, and no robust/stochastic optimization.
- Residual resampling is historical and in-sample with respect to the deployed forecast;
  iid drawing discards fold/time dependence and the nonzero pool mean biases simulated
  demand low. No pool bootstrap, centered rerun, or held-out fold is included in the
  frozen result (the paper derives a centered-pool arithmetic diagnostic from the same
  draws).
- The plan's pre-build and production-at-capacity equality are conditional on the
  fixed-anchor window and are not evidence of capacity enforcement.
- The simulation outcome is a single-plan diagnostic; fill rate has no evaluator scale.
- No external third-party reproduction was performed; hashes are internal provenance.

## Limitations / not done

- 24 synthetic routing cases; three per family, not a universal routing taxonomy.
- One gold route per case; no graded ambiguity or calibrated confidence.
- ProblemSpec is analyzed once and reused across the two signal/routing passes.
- LLM behavior can drift across models/versions; stability is demonstrated for two runs
  with the recorded configuration, not guaranteed.
- Phase 21.3 executes only forecasting -> optimization -> simulation. Arbitrary,
  branching, parallel, or longer mixed DAGs remain out of scope and were not started.
- No automatic `mmos solve`, competition paper agent, router council, embedding
  classifier, or external model training.
- Routing artifacts are control-plane and are not frozen scientific evidence.
