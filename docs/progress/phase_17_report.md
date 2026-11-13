# Phase 17 — Generalization Benchmark Suite (MILP first slice)

Date: 2026-11-13

## Goal

Test whether the architecture generalizes beyond prediction. The first slice implements a
0/1 knapsack MILP end to end through the shared ProblemSpec, evidence, validation, paper,
and Red Team boundaries.

## Implemented

- `schemas/optimization.py`: `OptimizationModel` (variables, bounds, objective, linear
  constraints), `OptimizationSpec`, `OptimizationSolution`, `OptimizationResult`,
  `EnumerationResult`.
- `experiments/optimization_runner.py`: OR-Tools CBC solve, independent constraint and
  objective recomputation, greedy feasible baseline, independent binary enumeration
  (up to 16 binary variables), run record.
- `validation/optimization.py`: feasibility/objective/constraint audit, optimality-gap
  semantics, enumeration agreement, baseline presence; blocks on inconsistencies.
- `evidence/store.py`: `freeze_optimization` records objective, gap, best bound,
  feasibility, runtime, time limit, selected variable values, constraint activities,
  enumeration results, and relative improvement with artifact hashes.
- `paper/optimization_builder.py`: optimization PaperAST with formulation, results
  table, validation, reproducibility, and explicit limitations; rendered through the
  same renderer and reviewed by the same Red Team.
- CLI `mmos optimize <project> <optimization_spec.json>`.
- Benchmark: `benchmarks/milp_knapsack` (10 items, capacity 50). Brute-force optimum is
  360 (1024 subsets).

## Live run (project `milp-knapsack-final-20261113-ad1f2a`)

- Solver: CBC, status optimal, objective 360, gap 0, feasible true.
- Greedy baseline: feasible, objective 350.
- Relative improvement: +0.0285714 (sign bug found and fixed during this phase).
- Independent enumeration: 1024 subsets, optimum 360.
- Validation: optimization passed; evidence 29 records; claim `proposed` only.
- Red Team: **revise**, blocking false; release gate requires PASS, so no release was
  written. State remains `PROBLEM_UNDERSTOOD` because this slice does not yet integrate
  the task-specific lifecycle.
- Blocked review: `reviews/blocked_optimization_exp-milp-knapsack.md`.

Red Team findings (all legitimate or disclosure requests):

1. The first run exposed an inverted relative-improvement sign; fixed and regression
   tested.
2. Optimality gap 0 on a concluded-optimal run is tautological; the independent
   enumeration is now reported as the real correctness signal.
3. The greedy baseline is weak on a 10-item toy; the paper must not infer method
   superiority.
4. The instance is hand-crafted and near-greedy-optimal; an adversarial instance is
   still needed.
5. Time limit, runtime, node counts, solver alternatives, selected subset, used capacity,
   and instance hashes should be reported or disclosed; several are now in evidence,
   others remain limitations.

## Architecture findings

What generalized:

- ProblemSpec accepted the optimization problem (the analyst produced two subproblems).
- Evidence records, artifact hashes, validation reports, PaperAST, renderer, Red Team,
  and the PASS-only release gate all worked without prediction-specific hacks.
- A separate `OptimizationSpec` avoided forcing optimization into a target/feature/split
  schema.

What did **not** generalize yet:

- `mmos run` and the project state machine are prediction/forecasting-specific; the
  `optimize` command bypasses Orchestrator lifecycle transitions.
- Tournament has no objective/gap comparison for optimization.
- MLflow tracking is not yet wired into the optimization runner.
- Model Council is not used to generate optimization formulations; the model is supplied
  as a declarative spec.
- Multi-objective, simulation, and graph tasks are not implemented.

## Verification

```text
uv run pytest        -> 156 passed, 3 deselected
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors
live MILP run        -> OR-Tools CBC solved, enumeration agreed, Red Team blocked release
```

## Next

Phase 17.2 should make the lifecycle task-aware: a shared state machine/pipeline branch
for optimization, an optimization Tournament (objective/gap + feasibility), MLflow
tracking, and then a multi-objective or simulation slice. The current MILP slice is a
proven vertical slice, not a released optimization product.


## 17.2A — Task-aware shared lifecycle

Implemented a `TaskFamily` concept (`prediction`, `forecasting`, `optimization`,
`multi_objective`, `simulation`, `graph`) and stored it in `project.yaml`.

- Prediction/forecasting keep the original lifecycle.
- Optimization has its own legal state path:
  `INIT -> PROJECT_INGESTED -> PROBLEM_UNDERSTOOD -> PROTOTYPING -> PROTOTYPED ->
  EXPERIMENT_COMPLETED -> VALIDATING -> VERIFIED -> EVIDENCE_FROZEN -> PAPER_DRAFTED ->
  RED_TEAM_REVIEW -> READY_FOR_RELEASE -> RELEASED`.
- Data-validation and research/council states are not fabricated for optimization.
- `mmos init --task optimization` creates an optimization project.
- `mmos optimize` now enters the Orchestrator lifecycle instead of bypassing it:
  spec planned -> solve -> validate -> freeze -> paper -> Red Team -> PASS gate.
- Blocked optimization runs transition to `REQUIRES_HUMAN_REVIEW` with resume target
  `RED_TEAM_REVIEW`. A bug where the blocked run stayed at `RED_TEAM_REVIEW` was found
  and fixed with a regression test.
- Objective direction is centralized in `relative_improvement(sense, candidate,
  baseline)`; maximize/minimize sign semantics are unit tested.

### Live verification

Project `milp-lifecycle-live-20261113-0d233f` (task optimization):

- State history follows the optimization path through `RED_TEAM_REVIEW`.
- Red Team returned `revise`; release was blocked and the project was moved to
  `REQUIRES_HUMAN_REVIEW` (resume `RED_TEAM_REVIEW`), with a blocked-review artifact.
- Fake-PASS and fake-REVISE CLI tests verify both the RELEASED and human-review
  branches without network calls.
- Full tests: 161 passed, 3 deselected (see final run log).

### Still outstanding (17.2B–E)

- Optimization experiment identity in MLflow.
- Optimization Tournament (objective/gap/feasibility/runtime/baseline).
- Adversarial MILP instances where greedy fails materially.
- Re-run Paper -> Red Team -> Judge after those additions.


## 17.2B — Optimization experiment identity and MLflow

Optimization now has the same formal experiment identity as prediction:

- `ExperimentStatus` distinguishes CREATED/RUNNING/SUCCEEDED/FAILED/INVALID/VALIDATED
  at the experiment level; project lifecycle state remains separate.
- `optimization_instance_hash` is a canonical hash of variables, objective, and
  constraints; changing an objective value or capacity changes it.
- `OptimizationExperimentRecord` stores experiment_id, project_id, task_family,
  status, instance_hash, config_hash, git commit, solver identity, MLflow run id,
  artifacts, validation status, and linked experiment ids.
- The greedy baseline is a separate formal experiment
  (`<candidate>-baseline-greedy`) with its own spec, result, run record, solver log,
  experiment record, and MLflow run.
- `log_optimization_run` uses the existing per-project SQLite MLflow ledger, tags,
  params, metrics, and artifact root; there is no second tracking system.
- Solver identity records OR-Tools 9.15.6755 and backend CBC; CBC native console
  output is unavailable through pywraplp and is explicitly logged as such.
- Enumeration is recorded as `exhaustive_enumeration` with 1024 subsets, optimum 360,
  one optimal subset, and `matches_solver_result=true`.
- Evidence provenance now includes instance hash, MLflow run id, git commit,
  experiment record path/hash, solver backend/version, termination condition, and
  linked baseline experiment id.
- `comparison_diagnostics` enforces same instance/objective/sense/constraints and
  discloses differing time limits without treating them as incomparable.

### Live run

Project `milp-identity-final-2-20261113-0d276d`:

- Candidate experiment: `exp-milp-knapsack`, status validated,
  MLflow run `4e69333994b24179933bd80f483ef359`.
- Baseline experiment: `exp-milp-knapsack-baseline-greedy`, status succeeded,
  MLflow run `46cce9b45f22491db0225e5311f1851a`.
- Instance hash: `5b6c0a6b73925766eccd54d72b4d4b792b23d61ba2e9e648bd9cd0eed2a9cebd`
  (shared by candidate and baseline).
- CBC: status optimal, objective 360, best bound 360, absolute/relative gap 0,
  runtime 0.00249 s, feasible, no constraint violations.
- Enumeration: 1024 subsets, optimum 360, matches solver.
- Evidence: 44 records; optimization validation passed.
- Red Team: revise, blocking false; PASS-only gate blocked release and moved the
  project to REQUIRES_HUMAN_REVIEW.
- Both solver logs and both experiment records are present; artifact lists include
  spec, result, run record, and solver log.

### Constraints / disclosures

- Node counts are not exposed by pywraplp and are recorded as unavailable.
- Solver-native CBC logs are not exposed; solver_log.txt records returned metadata.
- TIME_LIMIT vs feasible-incumbent semantics are represented by status and
  termination_condition but were not forced in this phase's small benchmark.
- Optimization Tournament, adversarial instances, and multi-objective remain 17.2C–E.


## 17.2C — Optimization Tournament

Implements a formal lexicographic optimization tournament over Optimization
Experiments only.

- New schemas: `OptimalityClass`, `BaselineRole`, `MethodFamily`, `WinnerStatus`,
  `OptimizationTournamentEntry`, `ComparisonMatrixEntry`,
  `OptimizationTournamentResult`.
- Tournament gates: comparison compatibility → validity → feasibility → objective →
  optimality evidence → gap → independent verification → runtime/simplicity.
- Comparison uses `comparison_diagnostics`; differing time limits are allowed but
  recorded. No weighted score is used.
- Infeasible experiments cannot win even with attractive objective values. Invalid
  experiments are retained in the result but excluded from selection.
- Optimality classes distinguish `proven_optimal` (solver optimal + independent
  enumeration agreement), `solver_optimal`, `feasible_not_proven_optimal`,
  `infeasible`, and `invalid`.
- Objective improvement reuses `relative_improvement(sense, candidate, baseline)`;
  maximize/minimize regression tests are present.
- Equivalent objectives within tolerance produce `EQUIVALENT` with no forced winner.
- Tournament artifacts: `tournament.json` and `decision_trace.md`; a tournament
  MLflow run is logged in the same project ledger.
- Evidence freeze accepts `tournament_id` and records status, winner, objectives,
  feasibility, optimality classes, verification, baseline role, improvements,
  decision trace, warnings, MLflow id, and links to both experiments.
- Paper gains an `Optimization comparison` section restricted to the evaluated
  instance; no general method-superiority language.
- Auto-claims are `proposed` only and limited to instance-level comparison.

### Live run

Project `milp-tournament-live-20261113-aa5fb7`:

- Tournament `tournament-8862ebdd`, status winner, winner `exp-milp-knapsack`.
- Candidate objective 360 (maximize), baseline 350; absolute +10, relative +2.857%.
- Both feasible. Candidate `proven_optimal` via exhaustive enumeration of 1024
  subsets; baseline `sanity_baseline`, `feasible_not_proven_optimal`, no bound/gap.
- MLflow: candidate `b029b3f2aaa143488eef8ce7b357ac24`, baseline
  `d436143368294d08914d9d5e00160f77`, tournament
  `81d8209107fd450391c256312a5c18e4`.
- Instance hash: `5b6c0a6b73925766eccd54d72b4d4b792b23d61ba2e9e648bd9cd0eed2a9cebd`.
- Evidence: 63 records, including 19 `tournament.*` keys.
- Claim remains `proposed`: candidate achieved a higher objective than the greedy
  sanity baseline on the evaluated benchmark instance.
- Red Team returned revise (5 major, 4 minor); PASS-only gate blocked release;
  project state `REQUIRES_HUMAN_REVIEW`. No release summary.
- Tests: 173 passed, 3 deselected; ruff and pyright clean.

### Deliberately not done (17.2D/E)

Adversarial MILP instances, multi-objective/Pareto/hypervolume, VRP/simulation/graph,
and a post-tournament Paper -> Red Team -> Judge revision cycle.


## 17.2D — Adversarial knapsack benchmark suite v1

Adds a first-class, freeze-before-run benchmark suite.

- New schemas: `KnapsackInstance`, `BenchmarkManifest`, `OracleResult`,
  `MethodAggregate`, `InstanceSuiteResult`, `BenchmarkSuiteResult`.
- Frozen suite: `benchmarks/knapsack/suite_v1/benchmark_manifest.json` plus
  per-instance JSON files. Manifest hash is computed over suite metadata, methods,
  oracle, and all instance hashes; editing a frozen instance invalidates the hash.
- Nine instances: one legacy sanity instance, one ratio-greedy trap, one capacity
  fragmentation adversarial case, one tie-stress case, two seeded uniform, two
  correlated, and one weakly correlated random instance. All seeds derive from the
  suite seed; curated instances do not change with seed.
- Independent oracle: in-process exhaustive enumeration, limited to 20 items, with
  search space, feasible subset count, optimum, and optimal subset. It does not call
  CBC or read solver values.
- Methods: CBC candidate, deterministic ratio-greedy sanity baseline, and a
  best-of(ratio greedy, best single feasible item) enhanced heuristic baseline.
  Greedy tie-breaking is deterministic (ratio, value, weight, index).
- Per instance: three formal experiments (CBC, greedy, enhanced greedy), each with
  spec/result/run record/solver log/MLflow run; candidate validation; and a formal
  Phase 17.2C tournament (CBC vs greedy). Enhanced greedy is aggregated but not part
  of the binary tournament.
- Suite aggregate: candidate feasible/verified counts, win/tie/loss, oracle
  agreement; greedy/enhanced hit rate, failures, mean/median/max absolute and
  relative regret, worst instance, plus per-instance table; no weighted score.
- Suite evidence, suite MLflow run, suite PaperAST with all instances, proposed
  claims, and Red Team review.

### Live suite run

Project `knapsack-benchmark-live-20261113-c3dbc6`:

- Suite `knapsack-suite-v1` v1.0.0, generator `knapsack-generator-v1`, suite seed
  20261113, 9 instances, status `complete`.
- Manifest hash: `ecedb6c403c4d2951130c68044151c3ff7cff61e6ee5f60365bca64123515779`.
- CBC: 9/9 feasible, 9/9 verified against the independent oracle; oracle agreement
  true; candidate wins/ties/losses = 5/4/0.
- CBC runtime: min 0.00092 s, median 0.00391 s, max 0.04076 s (tiny instances).
- Greedy: hits 4/9, failures 5; mean relative regret 0.05352, median 0.01739,
  max 0.33333, worst `greedy_ratio_trap`.
- Enhanced greedy: hits 4/9, failures 5; mean relative regret 0.05106, median
  0.01739, max 0.33333, worst `greedy_ratio_trap`.
- Adversarial: `greedy_ratio_trap` oracle 18 vs greedy 12 (33.3% regret);
  `capacity_fragmentation` oracle 115 vs greedy 113 (1.74%).
- Evidence: 81 verified records plus two proposed claims (CBC oracle agreement;
  greedy failure rate/worst case). No claim was auto-accepted.
- Red Team: revise (3 major, 3 minor); PASS-only gate blocked release and wrote a
  blocked-review artifact. Project lifecycle state remains `INIT` because the suite
  runner is artifact-level; this is documented as a remaining integration gap.
- Tests: 181 passed, 3 deselected; ruff and pyright clean.

### Known limitations / not done

- One problem family (0/1 knapsack), small instances, synthetic data only.
- Enhanced greedy is not in the per-instance tournament; it is aggregated.
- Suite-level run is not yet integrated with the project lifecycle state machine.
- No scale sensitivity, multi-objective, Pareto, VRP, simulation, or graph tasks.
- Phase 17.2E (post-tournament Paper -> Red Team -> Judge revision cycle) was not
  started.
