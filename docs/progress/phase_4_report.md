# Phase 4 Report

Date: 2026-11-12

## Scope

Data profiling, declarative schema validation, data-quality and leakage gates for
tabular CSV data, persisted as auditable artifacts and enforced by project state.

## Created / modified

- `src/mmos/schemas/data.py` — `Issue`, `ColumnProfile`, `TableProfile`,
  `DatasetProfile`, `ColumnSpec`, `TableSchema`, `DataSchema`, `DataQualityReport`,
  `LeakageFinding`, `LeakageReport`, `Severity`, `QualityStatus`
- `src/mmos/data/profiling.py` — CSV loading, column statistics, inferred schema,
  candidate keys
- `src/mmos/data/quality.py` — explicit-threshold quality checks and gate report
- `src/mmos/data/leakage.py` — heuristic leakage checks with stated limitations
- `src/mmos/data/agent.py` — `DataAgent` orchestrates profiling, Pandera validation,
  quality, leakage, and artifact persistence
- `src/mmos/core/project.py` — public `atomic_write_text`, artifact path properties
- `src/mmos/cli/main.py` — `analyze-data` command with `--target`, prints profile,
  issues, findings, and refuses to advance on a blocking gate
- Tests: profiling, quality, leakage, data agent, CLI flow

## Key decisions

1. **Polars 2.1 for computation, Pandera 0.36 as an independent schema check.** The
   inferred nullability contract is validated by Pandera before the artifacts are saved.
2. **Explicit thresholds.** High null fraction = 0.5 (warning), critical = 0.9 (error),
   numeric-text ratio = 0.9. Constants are named and documented so they can be argued
   with rather than hidden.
3. **ERROR blocks the gate; WARNING/INFO do not.** Blocking issues keep the project at
   `PROBLEM_UNDERSTOOD`; only a passing run advances to `DATA_VALIDATED`. Later phases
   will require `DATA_VALIDATED`, so the gate cannot be bypassed.
4. **Leakage detection is honest about its limits.** Only ID-like unique columns,
   duplicate columns, exact target duplicates, and temporal columns are detected. The
   report carries a `limitations` list; train/test overlap requires the split design
   that does not exist yet.
5. **DuckDB is not added yet.** There is no multi-table SQL/join requirement in the
   MVP; adding it would be speculative. Polars + Pandera cover the current need. DuckDB
   will be added when a real query requirement appears.
6. **CSV only in this phase.** Other formats raise a clear error instead of being
   skipped.

## Verification (actually executed)

```text
uv run pytest        -> 60 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

End-to-end CLI smoke (temp project, deliberately messy CSV with a null target and a
null feature): profile table printed, warnings reported, leakage warning for the id
column, quality/leakage passed, artifacts written, project advanced
`PROBLEM_UNDERSTOOD -> DATA_VALIDATED`.

## Deferred

- Excel/JSON/Parquet ingestion, multi-table relational checks, train/validation/test
  split leakage analysis, target-relationship leakage analysis.
- DuckDB-backed querying.

## Next phase (Phase 5)

Research Agent and Model Council: `MethodCard`, `ModelProposal`, `ModelRegistry`,
`ExperimentSpec`, and the Experiment Planner. LLM-dependent parts will follow the same
rule as Phase 3: real schemas and live-tested-by-request, no fabricated proposals.
