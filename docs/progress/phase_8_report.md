# Phase 8 Report

Date: 2026-11-12

## Scope

A per-project MLflow ledger for every formal experiment run.

## Created / modified

- `src/mmos/schemas/tracking.py` — `MLflowRunRecord`
- `src/mmos/tracking/mlflow_tracker.py` — `tracking_uri()`, `log_experiment_run()`
- `src/mmos/experiments/runner.py` — logs each completed run before returning
- `pyproject.toml` / `uv.lock` — MLflow dependency
- Tests: `test_mlflow_tracker.py`; shared experiment fixtures moved to `tests/helpers.py`

## Key decisions

1. **Project-local SQLite tracking**, `exercises/…/mlflow.db`, instead of a global
   `mlruns/` directory. Nothing leaks into the repo root, and each project can be
   archived independently.
2. **Artifact root inside the project** (`experiments/mlflow_artifacts`), set at
   experiment creation, so MLflow does not write artifacts relative to the current
   working directory.
3. **One thin helper owns all MLflow API calls.** The runner does not scatter tracking
   logic.
4. **Tracking failure is fatal.** A completed computation without a ledger entry would
   break provenance, so the run fails rather than continuing untracked.
5. **Local pointer file.** `experiments/<id>/mlflow_run.json` records the run id and
   tracking URI so the ledger can be audited without scanning the database.
6. **What is logged:** project/problem/proposal ids and dataset hash as tags; seed,
   fold count, model list, metrics list, target, feature count as params; every
   model×metric as `model_id.metric`; the whole experiment directory as artifacts.

## Verification (actually executed)

```text
uv run pytest        -> 99 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

The MLflow test performs a real runner execution, then queries the SQLite store with
`MlflowClient` and checks run status, tags, params, metrics, and logged artifacts.

## Next phase (Phase 9)

Tournament: compare completed experiments that used the same dataset, split, seed, and
metrics; require a baseline; eliminate models that fail to beat it; and record the
selected model plus a baseline challenger. The tournament never replaces experiments
with LLM judgment.
