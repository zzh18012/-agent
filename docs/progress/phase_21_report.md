# Phase 21 — Automatic Problem Router

Date: 2026-11-15

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

## Limitations / not done

- 24 synthetic routing cases; three per family, not a universal routing taxonomy.
- One gold route per case; no graded ambiguity or calibrated confidence.
- ProblemSpec is analyzed once and reused across the two signal/routing passes.
- LLM behavior can drift across models/versions; stability is demonstrated for two runs
  with the recorded configuration, not guaranteed.
- MULTI_TASK is detected but not executed; Mixed Modeling DAG remains Phase 21.2.
- No automatic `mmos solve`, competition paper agent, router council, embedding
  classifier, or external model training.
- Routing artifacts are control-plane and are not frozen scientific evidence.
