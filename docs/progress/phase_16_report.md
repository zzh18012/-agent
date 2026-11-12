# Phase 16 Report — Metric Semantics & Information-Set Fairness

Date: 2026-11-12

## Implemented

- **Unified pooled evaluation** (`EvaluationRecord`): every regression metric is computed
  from one out-of-fold evaluation vector. Records persist n, SSE, SST, target mean,
  population evaluation std, RMSE, MAE, pooled R², RMSE/eval_std, MASE, and MASE fold
  details. R² is `1 - SSE/SST`; when SST = 0 it is undefined with an explicit reason.
- **Metric identity invariant**: `(RMSE / eval_std)^2 = 1 - R²` by construction and is
  tested. The Numerical Auditor recomputes SSE, SST, R², RMSE, std, ratio, and MASE
  scaled errors from the prediction artifact.
- **MASE semantics**: per-fold scaling denominator uses only that fold's training
  history (`mean |y_t - y_{t-lag}|`); lag is 1 unless a confirmed seasonal period exists.
  Zero or insufficient denominators yield `undefined` with a reason; no epsilon is added.
  Each fold's train bounds, lag, denominator, absolute error, and scaled error are saved.
- **Information sets** (`InformationSet`): target-history availability, feature columns,
  feature availability, horizon, and lookback policy are recorded per model. Temporal
  models use `historical_y_only`; regression candidates use the declared feature set;
  constant predictors use a separate information set.
- **Matched practical equivalence** (`PairedComparison`): paired squared-loss
  differences, relative RMSE improvement, fixed-seed bootstrap interval, optional
  configurable practical threshold, and statuses
  `equivalent | different | inconclusive | incomparable`. Different information sets
  are explicitly incomparable.
- **Tournament semantics**: candidates are compared against a matched-information
  comparator (preferring more parsimonious models); external naive baselines remain as
  context but are never labelled a matched baseline. Selection status supports
  `winner | equivalent | incomparable`; equivalence retains the simpler model.
- **Evidence**: evaluation components, MASE fold details, information sets, selection
  status, and paired comparisons are frozen with artifact hashes.
- **Paper**: metric definitions with SSE/SST/evaluation std, information-set column,
  matched-comparison results and intervals, equivalence/parsimony wording, and explicit
  information-access caveats.
- **Claims** remain automatically `proposed`. Equivalence produces a proposed claim
  that no meaningful improvement was demonstrated.

## Authoritative live run

Project: `temporal-semantics-check-20261112-69ea0c`
Experiment: `exp-prop-statistician-1-25feff`

### Evaluation set

| Field | Value |
| --- | --- |
| Initial training window | 40 periods |
| Evaluation points | 20 (one-step rolling origin) |
| Folds | 20, horizon 1, step 1 |
| Evaluation target mean | 29.3947 |
| Evaluation target std (sqrt(SST/n)) | 8.473866 |
| SSE (selected model) | 1.671882 |
| SST (selected model) | 1436.128169 |
| Pooled R² | 0.998836 |
| RMSE | 0.289126 |
| RMSE/eval_std | 0.034120 |

### MASE

Denominator per fold is the mean absolute one-step naive error on that fold's training
window only. All 20 folds had positive denominators; each fold's denominator, absolute
error, and scaled error are frozen in evidence. No future observation influences a
fold's denominator (tested).

### Information sets

| Model | Information set |
| --- | --- |
| sklearn.linear_regression | features x1, x2 (both known_future) |
| sklearn.ridge | features x1, x2 (both known_future) |
| naive.last_value | target history only |
| naive.drift | target history only |
| sklearn.dummy_regressor | constant, no inputs |

### Matched comparison

- Linear regression vs ridge: relative RMSE improvement 1.5175%, bootstrap interval
  [-1.9445%, 4.7186%], information-set match true, status **equivalent** (no threshold
  specified; interval includes zero).
- Tournament selection status: `equivalent`; parsimony retains
  `sklearn.linear_regression`, with ridge as reference.
- External comparison against the constant floor: relative improvement 96.6% but status
  **incomparable** (different information sets). The paper states that the gap partly
  reflects feature access, not modelling skill.

### Temporal baselines (out-of-fold)

| Model | RMSE | MAE | R² | MASE | RMSE/eval_std |
| --- | --- | --- | --- | --- | --- |
| naive.last_value | 13.417134 | 10.986038 | -1.507011 | 0.790052 | 1.583354 |
| naive.drift | 13.564786 | 11.126231 | -1.562493 | 0.800040 | 1.600779 |
| sklearn.dummy_regressor | 8.567242 | 6.614162 | -0.022160 | 0.475880 | 1.011019 |

### Validation

leakage, numerical, statistical, reproducibility, and temporal all passed with the
final auditor code (re-run after the semantics changes). Evidence: 363 records.

### Red Team and Judge

- 12 attacks (critical/major/minor) were reviewed; the Judge returned **PASS** with
  blocking false.
- Project state: **RELEASED**; release summary:
  `releases/release-exp-prop-statistician-1-25feff.md`.
- Claim: one `proposed` claim stating that linear and ridge are practically equivalent;
  no automatic acceptance.

### Synthetic diagnostic

The generator is explicitly recorded: `y = 10 + 3*x1 + 2*x2 + 0.4*sin(period)`, seed
20261112, code `examples/linear_demand/generate.py`. The paper labels the run a
synthetic pipeline sanity check, so near-unity R² is explained by structural agreement
with the generator rather than claimed forecasting skill. A separate oracle/noise-floor
diagnostic was **not** implemented; this remains a documented limitation.

## Tests (actually executed)

```text
uv run pytest        -> 152 passed, 3 deselected
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

New tests cover the R² SSE/SST definition and identity, undefined R² for constant
targets, MASE training-only denominators and zero-denominator handling, information-set
equivalence/incomparability, practical equivalence with and without thresholds, matched
comparison in a tournament, and the requirement that release needs Judge PASS.

## Known limitations

- AutoReg, seasonal/Fourier candidates, oracle noise-floor diagnostics, alternative
  generator seeds, and alternative fold counts were not implemented; the paper discloses
  the relevant gaps.
- Practical equivalence without a business threshold relies on the bootstrap interval
  including zero; a user-specified threshold can be supplied via
  `--improvement-threshold`.
- ProblemSpec still lacks structured temporal metadata; temporal structure comes from the
  CLI or the dataset sidecar.
