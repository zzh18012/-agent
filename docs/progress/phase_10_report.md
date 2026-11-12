# Phase 10 Report

Date: 2026-11-12

## Scope

A uniform validation layer with independent checks over completed experiments.

## Created / modified

- `src/mmos/schemas/validation.py` — `ValidationReport`, `ValidationIssue`,
  `ValidationStatus`
- `src/mmos/validation/base.py` — shared report builder (blocking = any ERROR)
- `src/mmos/validation/numerical.py` — re-computes rmse/mae/r2/accuracy/f1 from the
  persisted out-of-fold predictions and compares against `metrics.json`; range checks
  r2, accuracy, f1, roc_auc, and non-negative errors
- `src/mmos/validation/leakage.py` — target-in-features, stale dataset, target
  presence, and leakage heuristics on the actual data
- `src/mmos/validation/statistical.py` — fold-count, sample-size, and fold-variance
  warnings from per-fold metrics
- `src/mmos/validation/reproducibility.py` — required artifacts, hash agreement, MLflow
  run status/tags, prediction coverage, and row counts
- `src/mmos/validation/__init__.py` — validator registry and
  `experiments/<id>/validation/` report persistence + summary
- `src/mmos/experiments/runner.py` — now persists per-fold metrics and `actual` values
  alongside predictions
- `src/mmos/cli/main.py` — `validate` command
- Tests: tampered metric detection, target leakage, missing MLflow record, clean run via
  API and CLI

## Key decisions

1. **Numerical audit is a real recomputation, not a range review.** Reported metrics are
   recomputed from persisted out-of-fold predictions; mismatch blocks. roc_auc cannot be
   recomputed from labels and is explicitly range-checked only, with that limitation in
   the report's recommendations.
2. **Leakage audit re-reads the data.** It does not trust the earlier data-gate file;
   target-in-features, stale dataset hash, and leakage heuristics are checked again.
3. **Reproducibility audit cross-checks the ledger.** It verifies the MLflow run is
   FINISHED and tagged with the same dataset hash, and that prediction coverage is
   exactly one prediction per row per model.
4. **Warnings do not block.** Only ERROR issues make a report blocking, and any blocking
   report fails validation and the CLI exit code.
5. **Reports are persisted** under `experiments/<id>/validation/` with a summary.
6. **No fake validators.** Optimization/robustness/sensitivity auditors are not
   implemented and are absent from the registry rather than returning meaningless
   passes.

## Verification (actually executed)

```text
uv run pytest        -> 111 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

Tests include a real tampering case (reported rmse changed after the run) that the
numerical auditor catches, and a missing-MLflow-record case that blocks reproducibility.

## Deferred

- Optimization/robustness/sensitivity auditors (need their experiment types).
- Blind review with an independent implementation.
- Evidence auditing (Phase 11).

## Next phase (Phase 11)

Verified evidence: an evidence store keyed by experiment and artifact hashes, claim
records that only reference verified evidence, and a freeze step that refuses to run
when any required validation is missing or blocking.
