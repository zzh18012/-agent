# Phase 20 — Multi-objective Vertical Slice

Date: 2026-11-15

## Goal

Make `TaskFamily.MULTI_OBJECTIVE` a first-class modeling paradigm with a bi-objective
binary project-portfolio Pareto front, exact epsilon-constraint candidate generation,
and independent exhaustive enumeration. No single winner is selected without preference
information.

No new TaskFamily enum was added. Prediction, Optimization, Simulation, and Graph paths
are unchanged.

## 20.1 — Bi-objective Portfolio Pareto Front

### Lifecycle and CLI

`mmos init --task multi_objective` and
`mmos multi-optimize <project> <spec.json>` drive the shared Orchestrator:

INIT -> PROJECT_INGESTED -> PROBLEM_UNDERSTOOD -> PROTOTYPING -> PROTOTYPED ->
EXPERIMENT_COMPLETED -> VALIDATING -> VERIFIED -> EVIDENCE_FROZEN -> PAPER_DRAFTED ->
RED_TEAM_REVIEW -> READY_FOR_RELEASE -> RELEASED only on Judge PASS.

The real run used DeepSeek `analyze-problem`; the ProblemSpec identified binary
decisions, the budget constraint, two conflicting maximization objectives, Pareto
trade-offs, and the absence of a unique best portfolio. It did not collapse objectives
into a weighted score.

### Schema and instance identity

- `MultiObjectiveSpec`, `ObjectiveSpec`, `ProjectItem`, `ParetoVector`, `ParetoSolution`,
  `EpsilonSolve`, `EpsilonCoverage`, `EpsilonSensitivity`, `FrontComparison`,
  `ParetoOracle`, `MultiObjectiveResult`, and `MultiObjectiveExperimentRecord`.
- The frozen instance has 10 binary projects, budget 12, integer costs/benefits,
  objective A and B both maximized, reference point (0,0), and step-1 epsilon policy.
- `instance_hash` covers project ids/costs/benefits, budget, objective definitions and
  senses. `spec_hash` covers the full spec. Hash changes on any cost, benefit, budget,
  objective definition, or sense change.

### Candidate method

- Exact MILP with OR-Tools CBC, one epsilon constraint `B >= epsilon`.
- Two-stage lexicographic solve: maximize A, fix A exactly, then maximize B. No weighted
  sum or Big-M surrogate.
- The step-1 epsilon schedule is 0..sum(all B)=0..53, a loose a-priori bound from the
  frozen instance; it does not read the oracle front.
- Infeasible levels are high thresholds no budget-feasible portfolio can meet.
- Candidate front extraction removes infeasible solves, deduplicates objective vectors,
  and applies maximize dominance independently from the oracle implementation.
- The candidate returns one tie-broken decision per objective vector by construction;
  decision-space multiplicity is not claimed.

### Independent oracle and auditor

- The oracle enumerates all 2^n binary subsets, filters budget feasibility, computes
  both objectives, and applies an independently written maximize-dominance rule. It
  never calls CBC, the epsilon runner, or candidate fronts.
- The auditor checks instance/spec hashes, experiment identity, epsilon schedule and
  coverage, candidate binary/budget/objective feasibility, duplicate vectors, candidate
  front nondominance, exact objective-vector set comparison against a recomputed oracle,
  missing/false/dominated Pareto points, Pareto front vs decision set sizes,
  hypervolume, and reference point. Exact objective-vector set equality is the primary
  criterion; hypervolume is secondary.
- Sensitivity steps 2 and 5 are run as secondary coarse-grid checks; missed vectors and
  whether they are off-grid are reported.

### Tracking, evidence, paper

- One MLflow run per experiment with budget/objective/method/epsilon/reference params,
  front sizes/precision/recall, hypervolume, solver calls/runtime, and artifacts.
- Verified evidence includes instance hash, objectives, candidate/oracle fronts, front
  sizes, precision/recall, search space, feasible count, epsilon policy, reference point,
  hypervolume, feasibility audit, solver, and solver calls.
- The paper is a Bi-objective Pareto-front Conformance Report with a formal 2D Pareto
  figure generated from artifacts. It explicitly states no unique winner, separates
  objective-space front from decision-space set, reports decision-space recall, and
  frames exact step-1 recovery as a grid-coverage/implementation check rather than
  general method superiority.

### Live run

Project `portfolio-pareto-release-20261115-20261115-d9a559`:

- Experiment `exp-portfolio-benchmark-v1`; problem `portfolio-benchmark-v1`; task
  `multi_objective`.
- Instance hash `96fa6385da73850b506db0caf8f91062cb53e847360dfc93c248ce4628166839`;
  spec hash `6c404027b6142c77a0280f7804902d5aa3039484b0ede1f8e414ff3f59f0f0fd`.
- 10 projects, budget 12; objectives A and B both maximize; reference point [0,0].
- Method `epsilon_constraint`; solver `ortools.cbc` 9.15.6755; 54 step-1 levels
  (0..53), 29 feasible / 25 infeasible; 83 CBC calls (29*2 + 25*1).
- Candidate front 10 objective vectors; oracle front 10; decision set candidate 10,
  oracle 11 (one objective vector has two oracle decisions; candidate returns one).
- Front precision/recall 1.0/1.0 and exact objective-vector match True at step 1;
  decision-space recall 10/11 = 0.909 is reported and not claimed as 1.0.
- Hypervolume 699.0 vs oracle 699.0 (secondary); gap 0.0.
- Coarse-grid sensitivity: step 2 recall 0.7 (misses 3 oracle vectors, all off-grid);
  step 5 recall 0.4 (misses 6, all off-grid).
- MLflow run `0f6f0cef92154c29a854542c653edfc8`.
- Validation `multi_objective` passed; evidence frozen.
- Claim `claim-multiobjective-exp-portfolio-benchmark-v1` remains `proposed`, scoped to
  internal implementation/grid-coverage with no decision-space 1.0 claim.
- Red Team final: 12 attacks (3 critical, 7 major, 2 minor), all pass, overall `pass`,
  blocking false. Release written to
  `releases/multi_objective_exp-portfolio-benchmark-v1.{json,md}`; project `RELEASED`.
- Earlier blocked multi-objective runs remain as audit history; the final run superseded
  them after disclosure and sensitivity additions.

## Known limitations / not done

- One small synthetic integer instance in the most favourable antagonistic, binding-budget
  regime. Non-binding budgets, non-integer objectives, larger n, and 3+ objectives are
  untested.
- Step-1 exact recovery is grid-coverage-contingent and not a general method guarantee;
  coarse steps lose vectors.
- Candidate and oracle share the frozen spec (implementation-path independence only);
  no perturbed-input negative control or second instance encoding was run.
- No decision-space completeness claim; candidate lexicographic tie-break returns one
  decision per vector.
- No alternative-reference hypervolume sensitivity; hypervolume is secondary.
- Evidence is private; oracle/runner code, epsilon_solutions.csv, and front lists are not
  published with the report.
- No NSGA-II/III, MOEA/D, pymoo, stochastic metaheuristics, preference elicitation,
  knee/compromise recommendation, many-objective, or multi-objective benchmark suite.
