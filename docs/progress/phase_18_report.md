# Phase 18 — Simulation Vertical Slice

Date: 2026-11-14

## Goal

Make `TaskFamily.SIMULATION` a first-class modeling paradigm with an independent
lifecycle, formal experiment identity, stochastic replications, and an analytic oracle.
The first slice is a stable M/M/1 discrete-event queue simulation using SimPy.

No new TaskFamily enum was added. Prediction and Optimization code paths are unchanged.

## 18.1 — M/M/1 Simulation Vertical Slice

### Lifecycle

`mmos init --task simulation` and `mmos simulate <project> <spec.json>` use the shared
Orchestrator lifecycle:

INIT -> PROJECT_INGESTED -> PROBLEM_UNDERSTOOD -> PROTOTYPING -> PROTOTYPED ->
EXPERIMENT_COMPLETED -> VALIDATING -> VERIFIED -> EVIDENCE_FROZEN -> PAPER_DRAFTED ->
RED_TEAM_REVIEW -> READY_FOR_RELEASE -> RELEASED only on Judge PASS.

No data-validation, research, council, or tournament stages are fabricated for this
benchmark. The real run used DeepSeek `analyze-problem`; the ProblemSpec correctly
identified stochastic arrivals/service, a single FIFO server, warm-up, time-weighted
measures, replications, confidence intervals, Little's Law, and analytic validation.

### Simulation schema and runner

- New `SimulationSpec`, `AnalyticOracle`, `ReplicationResult`, `MetricSummary`,
  `AnalyticComparison`, `LittleLawCheck`, `SimulationResult`, and
  `SimulationExperimentRecord`.
- SimPy discrete-event single-server FIFO queue with Poisson arrivals and exponential
  service.
- Warm-up [0, warmup) runs normally and is excluded from measurement. Measurement is
  [warmup, warmup+measurement).
- Arrivals stop at measurement end; eligible arrivals drain to completion, so W/Wq are
  not right-censored.
- L, Lq, and utilization use time-weighted areas, not event-observation averages.
- Customer-level W/Wq cover eligible arrivals; time-weighted metrics integrate to the
  measurement end.
- Master seed derives a SHA-256 seed schedule with unique replication, arrival, and
  service seeds. The full schedule and its hash are recorded.
- Two independent in-process executions with fresh SimPy environments and RNG streams
  are compared by a canonical replication digest; a mismatch fails the experiment.
- Per-replication artifacts record eligible/completed counts, post-window completions,
  idle-tail length, areas, flow rate, and runtime.

### Analytic oracle and auditor

- The closed-form M/M/1 oracle is independently implemented (no simulator calls):
  rho, W, Wq, L, Lq, utilization, throughput, plus internal identity checks.
- The oracle rejects rho >= 1; an unstable queue cannot enter VERIFIED.
- `mmos/validation/simulation.py` independently checks:
  spec hash, task identity, MLflow/git provenance, rates, stability, replication count,
  seed schedule and uniqueness, determinism digests, warm-up/measurement semantics,
  drain completion, post-window/idle-tail bounds, W>=Wq>=0, L>=Lq>=0,
  utilization in [0,1], area-derived L/Lq/utilization, flow-rate and throughput
  definitions, summary mean/std/median/Student-t CI from replication rows, oracle values
  and identities, analytic differences, Little's Law bookkeeping, and missing metrics.
- No weighted validation score is used.

### Tracking, evidence, paper

- One formal MLflow run per simulation experiment (not per replication), with spec
  params, metric summaries/CIs, relative differences, Little's Law discrepancies, and
  artifacts (`simulation_spec.json`, `simulation_result.json`, `analytic_oracle.json`,
  `replications.csv`, validation report).
- Verified evidence includes lambda/mu/rho, replications, warm-up/measurement, per-metric
  mean/std/CI/theory, Little's Law discrepancies, seed schedule, SimPy version, and
  provenance hashes.
- Paper is an M/M/1 Single-Operating-Point Simulation Discrepancy Report. It explicitly
  classifies W/Wq as descriptive finite-cohort estimates, L/Lq/utilization as descriptive
  time-weighted quantities, and throughput as a counting-path arithmetic check. It makes
  no conformance/accuracy claim, reports no tolerance-pass verdict, and documents that no
  warm-up, window, rate, or master-seed sensitivity sweep was performed.

### Live run

Project `simulation-mm1-release-20261114-20261114-bfadfd`:

- Experiment `exp-mm1-benchmark-v1`; simulation `mm1-benchmark-v1`; task `simulation`.
- lambda=2, mu=3, rho=2/3; warm-up 500; measurement 2000; 30 replications; master seed
  20261113; spec hash `5ec80401fcac8b438b3001b21549fa6da93399ff79fe29b710dc593c9c68b50f`.
- MLflow run `9fdf934fb0304d79bba6f153ad3a85d3`; seed schedule hash
  `96e752dded9076e33c843fdb21d72cc0a12cff56c868d1b97c95dffa08399fc0`.
- Determinism: two independent in-process runs produced identical canonical digest
  `e139992ff99a6ccbc9885bb923fe212198edea8bf32fd9d65704e4a1e2afb12e`.
- Descriptive results: W 0.994399 [0.966788, 1.022009]; Wq 0.661421 [0.634906,
  0.687936]; L 1.991525 [1.927112, 2.055937]; Lq 1.325151 [1.265940, 1.384362];
  utilization 0.666374 [0.660034, 0.672714]; throughput 2.001700 [1.986920, 2.016480].
- Analytic values: W 1, Wq 2/3, L 2, Lq 4/3, utilization 2/3, throughput 2.
- Post-window completions: 63 across replications; maximum idle tail 3.274325 time units.
- Validation: `simulation` passed. Evidence frozen.
- Claim `claim-simulation-exp-mm1-benchmark-v1` remains `proposed`; not auto-accepted.
- Red Team final: 10 attacks (6 major, 4 minor), all pass, overall `pass`, blocking false.
  Release written to `releases/simulation_exp-mm1-benchmark-v1.{json,md}`; project
  `RELEASED`.
- An earlier blocked run remains as audit history; it was superseded after the schema and
  paper claim reductions, not cherry-picked.

### Tests

- 214 tests pass; ruff and pyright clean.
- Coverage includes the analytic oracle, unstable queues, seed uniqueness/reproducibility,
  time-weighted statistics versus observation means, warm-up exclusion, end drain,
  metric invariants, Student-t CI, Little's Law arithmetic, missing/reused replications,
  tampered time averages, paper scope, lifecycle PASS/REVISE, evidence-preserving resume,
  and the determination that an unstable spec never reaches VERIFIED.

## Known limitations / not done

- One problem family (M/M/1), synthetic, single operating point, one master seed.
- No warm-up, measurement-length, rate, or master-seed sensitivity sweep.
- W/Wq are finite-cohort descriptors, not PASTA/unbiased stationary estimators.
- Little's Law is a bookkeeping identity; per-replication residuals are tabulated but the
  support mismatch and Jensen gap are not separately quantified.
- No bootstrap/QQ/autocorrelation/batch-means diagnostics.
- Throughput is a counting-path check, not a service-rate estimate.
- Multi-objective, VRP, graph, agent-based, inventory, traffic, simulation-optimization,
  and universal simulation DSLs are out of scope.
