# Phase 9 Report

Date: 2026-11-12

## Scope

A tournament that ranks models from completed experiments and selects a candidate,
strictly from executed metrics.

## Created / modified

- `src/mmos/schemas/tournament.py` — `TournamentEntry`, `TournamentResult`
- `src/mmos/experiments/tournament.py` — comparability checks, ranking, baseline gate
- `src/mmos/schemas/experiment.py` — results now record feature set and split
  strategy so comparability is checkable
- `src/mmos/experiments/runner.py` — populates those fields
- `src/mmos/core/project.py` — `experiments/tournaments/` persistence
- `src/mmos/cli/main.py` — `tournament` command
- Tests: 7 tournament tests, 1 CLI integration test

## Rules enforced

1. **Comparability is mandatory.** Experiments must share dataset hash, task type,
   feature set, split strategy, validation strategy, seed, fold count, and metric set.
   Any difference raises before ranking.
2. **A baseline must be present.** Registered baselines are identified via
   `is_baseline`; without one the tournament is blocked, not decided.
3. **Candidates must beat the best baseline** on the primary metric. If no candidate
   does, the tournament is blocked and the baseline becomes the challenger.
4. **No duplicate models across experiments** in one tournament; otherwise the same
   model could be counted twice.
5. **Ties are deterministic** via model id as the secondary sort key.
6. **Selection is provisional.** The result records that robustness, sensitivity, and
   validation gates have not run yet.
7. **The tournament never uses language-model judgment.** It only reads
   `ExperimentResult` metrics produced by the runner.

## Verification (actually executed)

```text
uv run pytest        -> 107 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

The CLI test runs a real experiment (dummy + linear regression), then invokes
`mmos tournament` and checks that the linear model is selected over the baseline.

## Deferred

- Staged elimination by robustness/sensitivity (needs Phases 11–12 outputs).
- Multi-metric Pareto selection.

## Next phase (Phase 10)

Independent validation framework: leakage, numerical, statistical, optimization,
robustness, sensitivity, reproducibility, and evidence auditors returning uniform
`ValidationReport`s, with blocking issues preventing verification.
