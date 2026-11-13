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
