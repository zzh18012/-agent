# Phase 7 Report

Date: 2026-11-12

## Scope

A real end-to-end compute pipeline for tabular CSV projects:

```text
registered CSV → data gate → cross-validated baseline + candidates
               → out-of-fold metrics → predictions + run record
```

executed as a Snakemake DAG through `LocalExecutor`.

## Created / modified

- `src/mmos/schemas/experiment.py` — `ExperimentSpec.target_column`,
  `ExperimentResult`
- `src/mmos/experiments/planner.py` — requires `target_column`; rejects a target that
  is also a feature
- `src/mmos/experiments/engines.py` — executable sklearn engines for dummy, OLS,
  ridge, random forest, histogram gradient boosting, logistic regression, random
  forest classifier
- `src/mmos/experiments/metrics.py` — rmse/mae/r2 and accuracy/f1_macro/roc_auc with
  explicit rejection of unknown metrics and non-finite values
- `src/mmos/experiments/runner.py` — verifies the dataset fingerprint, refuses nulls,
  runs out-of-fold CV for all models on the same folds, writes
  `artifacts/metrics.json`, `artifacts/predictions.parquet`, `run_record.json`
- `src/mmos/experiments/data_gate.py` — workflow stage that runs the data-quality and
  leakage gates for the experiment's target
- `workflow/Snakefile` — `data_gate → run_experiment`
- `src/mmos/cli/main.py` — `run-experiment` (state guard, dry-run, timeout, execution
  records)
- Tests: runner (7), workflow integration (1), CLI dry-run (1)

## Key decisions

1. **One split per experiment, shared by every model.** Folds are generated once and
   reused, so baseline and candidates are compared on identical data partitions.
2. **Out-of-fold predictions, not in-fold scores.** Every reported metric is computed
   on held-out predictions; predictions are persisted for later auditing.
3. **Dataset identity is re-verified at run time.** If the manifest hash differs from
   the spec, the run aborts. A stale experiment cannot silently use new data.
4. **Nulls are refused, not imputed.** Imputation must be an explicit, recorded
   transformation step later; silent imputation would corrupt the provenance.
5. **No winner selection.** The runner reports metrics for all models in the spec.
6. **Engine coverage is explicit.** Registered models without an executable engine
   (forecasting, optimization) raise `NotImplementedError` instead of being skipped.
7. **The workflow runs in the project directory** with absolute interpreter and
   Snakefile paths, and each run is captured by `LocalExecutor`.
8. **Only one data table is supported in the runner** for now; multi-table experiments
   raise rather than guessing a join.

## Verification (actually executed)

```text
uv run pytest        -> 98 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

The workflow test executes real Snakemake: data gate, training, metrics, predictions,
and run record. It passed alone and in three consecutive full-suite runs.

Honest note: during one full-suite run, the Snakemake test exited with code 1 before
its logs could reveal a cause; three subsequent full runs passed. The failure is not
explained and is recorded here rather than hidden. If it recurs, the captured
stdout/stderr paths in the assertion will be the starting point.

## Deferred

- Forecasting, optimization, simulation execution.
- Multi-table datasets and merges.
- Example project data (will accompany the automated demo command in a later phase).

## Next phase (Phase 8)

MLflow experiment ledger: one local tracking store per project, logging parameters,
dataset hash, metrics, and artifacts from every formal run, with a thin helper instead
of scattered tracking calls.
