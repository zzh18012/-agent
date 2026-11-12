# Phase 15 Report — Scientific Hardening

Date: 2026-11-12

## Scope

Fixing the scientific-method failures exposed by the Red Team, as general system
capabilities rather than demo-specific rules.

## Implemented

- **Target metadata** (`TargetSummary`): unit status, dtype, mean, std, min, max,
  median, positivity, synthetic provenance (generator, parameters, seed, code path).
  Statistics are computed from the real data; the LLM never supplies them.
- **Dataset metadata sidecar** (`dataset_metadata.json`): target description/unit
  status, temporal structure, feature availability, synthetic provenance. Unit status
  distinguishes `known`, `unknown`, and `synthetic_units`; units are never invented.
- **Time structure**: explicit `--time-column` / `--horizon` or sidecar metadata;
  the pipeline refuses to guess a horizon and stops with human review if it is missing.
- **Rolling-origin validation** (`TemporalSplit` + `FoldWindow`): no random KFold for
  temporal data; schema enforces `train_start < train_end <= test_start < test_end`;
  temporal validator checks fold order, baselines, seasonal period, and feature
  availability at forecast time.
- **Baseline family**: `naive.last_value`, `naive.drift`, and `naive.seasonal` (only when
  a seasonal period is confirmed). Baselines are injected for temporal experiments and
  are evaluated on the same folds/metrics as candidates.
- **Metrics**: MASE with an explicit training-window naive scaling denominator, and
  `rmse_std_ratio = RMSE / full-sample target std`; denominators are documented.
- **Uncertainty**: per-fold metrics, `MetricSummary` (mean/std/median/min/max/n,
  percentile bootstrap range), and determinism flags. Deterministic models are not
  repeatedly seeded; stochastic models are flagged as single-seed in the paper.
- **Evidence**: target statistics/unit/synthetic provenance, temporal split and fold
  boundaries, feature availability, determinism, metric uncertainty and fold values,
  and evaluated-row count all enter `verified/` with artifact hashes.
- **Paper**: target context, temporal evaluation, metric definitions, information sets,
  fold values, uncertainty wording, synthetic-data sanity-check framing, and
  provisional/verified wording. A provisional result is never rendered as final.
- **Release gate**: release now requires judge verdict **PASS**. `revise` or `reject`
  keeps the project at `REQUIRES_HUMAN_REVIEW` and writes a blocked-release handoff.

## Final live run (authoritative)

Project: `temporal-final-gate-20261112-2ea8ee`
Experiment: `exp-prop-statistician-1-508574`

### Target context

| Field | Value |
| --- | --- |
| Target | y |
| Unit status | synthetic_units (no real-world interpretation) |
| Mean | 29.610897 |
| Std | 9.889438 |
| Min / Max | 11.782392 / 49.158370 |
| Median | 28.968343 |
| Synthetic | yes |
| Generator / seed | linear_demand_generator_v1 / 20261112 |
| Code | examples/linear_demand/generate.py |

### Evaluation

- Rolling-origin, horizon 1, 5 folds, time column `period`.
- Fold windows: train `[0,55)` test `[55,56)`; `[0,56)`/`[56,57)`;
  `[0,57)`/`[57,58)`; `[0,58)`/`[58,59)`; `[0,59)`/`[59,60)`.
- Every fold satisfies max(train) < min(test).

### Baselines and candidate (out-of-fold)

| Model | RMSE | MAE | R² | MASE | RMSE/std |
| --- | --- | --- | --- | --- | --- |
| sklearn.ridge (candidate) | 0.252623 | 0.237047 | 0.997385 | 0.017496 | 0.025545 |
| sklearn.linear_regression (candidate) | 0.253726 | 0.237779 | 0.997362 | 0.017550 | 0.025656 |
| sklearn.dummy_regressor | 5.231313 | 4.686615 | -0.121361 | 0.347244 | 0.528980 |
| naive.last_value | 7.999599 | 7.453009 | -1.622167 | 0.551987 | 0.808903 |
| naive.drift | 8.058531 | 7.510750 | -1.660944 | 0.556260 | 0.814862 |

The tournament selected `sklearn.ridge` with `sklearn.dummy_regressor` as the
lowest-RMSE reference. Candidates and naive baselines see different information sets
(candidates use known-future x1/x2; naive baselines use target history only); the paper
states this explicitly.

### Validation

All five validators passed: leakage, numerical, statistical, reproducibility, temporal.
Evidence: 114 verified records. Claim state: `proposed` only (human acceptance required).

### Red Team / Judge

- Overall: **revise**; blocking: yes. No release summary was written.
- Final state: `REQUIRES_HUMAN_REVIEW`; blocked-release handoff:
  `reviews/blocked_release_exp-prop-statistician-1-508574.md`.
- Three critical attacks remain:
  1. MASE scaling denominator/window needs a sharper operational definition.
  2. R² and RMSE/std denominators must be reconciled and the SST basis reported.
  3. The headline comparison lacks an information-matched baseline (features vs
     target-history-only baselines); the synthetic generator is exactly linear in x1/x2,
     so a linear candidate's advantage is structurally expected.
- Major attacks cover synthetic-data tautology, missing residual/coefficient diagnostics,
  single seed/split, alternative fold counts, seasonal-naive lags, reproducibility
  details, and uncertainty interpretation.

This is the correct outcome: the system did not force a release on a synthetic
sanity-check result that cannot support a model-selection claim.

## Earlier runs and gate correction

Two earlier runs reached release before the release rule was tightened. That rule was
wrong per this phase specification: absence of critical attacks is not sufficient;
judge PASS is required. Those releases are not treated as valid evidence of Phase 15
success. The authoritative final run above is blocked.

## Bugs fixed during the phase

1. Council could select registered-but-not-executable models; executable-only catalogs
   and filters were added.
2. The LLM timeout change broke the DeepSeek base URL; fixed with an explicit base URL
   and a regression test.
3. Provider calls had no timeout and could hang; bounded timeout added.
4. Windows subprocess failures (`WinError 8`, `0xC0000409`) were traced to memory
   pressure; BLAS/OMP thread defaults were limited to 1, and workflow rules retry once.
5. Release gate allowed `revise` (non-blocking) through; now requires judge PASS.

## Verification

```text
uv run pytest        -> 143 passed, 3 deselected
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
uv run pytest -m integration -> live DeepSeek problem analysis, research/council, red team
```

Real pipeline runs executed: data gate, Snakemake rolling-origin evaluation, MLflow,
validators, evidence freeze, paper, Red Team.

## Known limitations

- AutoReg was not implemented; naive and drift baselines are present, seasonal naive is
  available only with a confirmed period.
- ProblemSpec does not yet provide structured temporal metadata; temporal structure must
  be declared via CLI or the dataset sidecar.
- Synthetic-data residual/oracle decomposition, coefficient diagnostics, Fourier
  variants, alternative seeds, and alternative fold counts were not run; the paper
  discloses this.
- Red Team is adversarial and can continue to request new revisions; release remains
  blocked until a judge returns PASS.

## Phase 15 success criteria

Implemented: deterministic temporal splits, naive baseline family, target scale in
evidence/paper, synthetic provenance, interpretable metrics, determinism-aware
uncertainty, consistent paper wording, automated temporal/scale/availability checks,
full real rerun. Final release is correctly withheld under `revise`.
