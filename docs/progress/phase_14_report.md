# Phase 14 Report

Date: 2026-11-12

## Scope

An automated `mmos run` pipeline with resume support, explicit target/feature inputs,
auto-generated **proposed** claims, release summaries, and human-review handoff when the
red team blocks release.

## Implementation

- `src/mmos/pipeline.py` — staged pipeline with idempotent resume:
  problem analysis → data gate → research → council → experiment planning → Snakemake →
  tournament → validation → evidence freeze → proposed claim → paper → red team →
  release summary.
- `src/mmos/schemas/pipeline.py` — `PipelineState` (options + ids), `ReleaseSummary`.
- `Orchestrator.advance_to()` — walks the linear state machine one legal step at a time.
- `experiments/workflow.py` — shared Snakemake invocation used by CLI and pipeline.
- `core/problem_text.py` — shared problem-text extraction.
- CLI `mmos run <project> --target ... --feature ... [--primary-metric ...]`.
- Red-team blocking writes `reviews/blocked_release_<experiment>.md` with attacks,
  verdicts, and required fixes, then moves the project to `REQUIRES_HUMAN_REVIEW`.
- Pipeline state records target/features/metrics/seed; reruns with different options are
  rejected to avoid mixing incompatible artifacts.
- Claims created by the pipeline are always `proposed`; human acceptance is required.

## Live end-to-end demo (demo project 2)

A 60-row synthetic demand dataset with a Chinese problem statement. All steps executed
for real:

- ProblemSpec produced; data gate passed.
- Research produced method cards; council produced proposals; non-executable models
  filtered out; experiment planned with the registered baseline.
- Snakemake ran the data gate and cross-validation; tournament selected
  `sklearn.linear_regression` (rmse 0.294223, mae 0.263225, r2 0.999022) over
  `sklearn.dummy_regressor` (rmse 9.62298).
- All four validators passed; 6 evidence records were frozen; one proposed claim was
  created; paper and figure were rendered.
- Red team produced 2 critical, 4 major, and 3 minor attacks. The judge marked the
  review blocking (`revise`). The pipeline refused release and stopped at
  `REQUIRES_HUMAN_REVIEW`. No release summary was written, by design.

The blocking attacks are substantive, not noise:

1. **Critical — metric interpretability**: target scale/units and descriptive statistics
   are absent, so RMSE 0.294 cannot be interpreted; the auditor's numbers are not shown.
2. **Critical — unfair baseline**: asking whether the model is "worth using" while the
   only challenger is a constant mean predictor is insufficient; naive/seasonal/AR
   baselines and a time-ordered split are required for a demand series.
3. **Major — synthetic/leakage provenance**: near-perfect r2 suggests synthetic data or
   leakage; feature provenance/measurement timing is undocumented.
4. **Major — random KFold** on a 60-period demand series may leak across time.
5. **Major — contradiction**: the paper presents a selection while calling it
   provisional.
6. **Major — no uncertainty**: one seed, one fold count, no per-fold or repeated-CV
   variance.

This is the evidence and review planes working as designed: the system prevented an
unsubstantiated release. It is not treated as a passing release demo.

## Bugs found and fixed during this phase

1. Council could propose registered-but-non-executable models (e.g., `statsmodels.ols`),
   causing experiment failure. Council catalogs and normalization now allow only
   executable engines; the pipeline filters again before planning.
2. The LLM timeout change passed an `AsyncOpenAI` client with a default OpenAI base URL
   to `DeepSeekProvider`, breaking DeepSeek calls. Fixed with an explicit DeepSeek base
   URL and a regression test.
3. A stalled provider call hung for hours. Requests now use a configurable timeout
   (default 120s).
4. The intermittent Snakemake test failure was diagnosed as Windows `WinError 8`
   (process-spawn resource error). The workflow test retries once only for that exact OS
   error; matplotlib figures are explicitly cleared after saving.

## Verification

```text
uv run pytest        -> 135 passed, 3 deselected (run twice)
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
uv run pytest -m integration  -> live DeepSeek problem analysis, research/council, red team
```

## Next decision

The critical attack about baselines/time structure requires either a time-series
execution path (rolling-origin CV + naive baselines) or a differently framed
cross-sectional demo. Alternatively, implement an author-revision loop / human
resolution command so a reviewed paper can be revised and re-reviewed. This is a
scientific-design choice, not a mechanical fix.
