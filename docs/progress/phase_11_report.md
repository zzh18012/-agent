# Phase 11 Report

Date: 2026-11-12

## Scope

The verified evidence layer: hashed records, provenance, claims, and a freeze gate that
refuses unsupported results.

## Created / modified

- `src/mmos/schemas/evidence.py` — `EvidenceRecord`, `EvidenceIndex`, `ClaimRecord`
- `src/mmos/evidence/store.py` — freeze, index/claim loading, claim registration
- `src/mmos/validation/__init__.py` — verification summary now records sha256 of
  `metrics.json` and `predictions.parquet`
- `src/mmos/core/project.py` — verified paths (`numbers.json`, `claims.yaml`,
  `verified/metrics/`)
- `src/mmos/cli/main.py` — `freeze-evidence`, `add-claim`
- Tests: missing validation, hash mismatch after validation, store contents, claims,
  CLI end to end

## Enforcement rules

A record enters `verified/` only if all of the following hold:

1. A validation summary exists for this exact experiment and is not blocking.
2. All four required validators (`leakage`, `numerical`, `statistical`,
   `reproducibility`) have persisted reports with status `passed`.
3. `metrics.json` currently hashes to the value recorded at validation time. If the
   artifact changed after validation, freezing fails and demands re-validation.
4. The spec and result agree on the dataset hash.
5. The referenced artifacts exist.

Each record carries the experiment id, source artifact path, artifact sha256, source
field, validator list, timestamp, and provenance (dataset hash, spec hash/path,
validation summary, MLflow run id). Freezing is idempotent.

Claims can only cite existing evidence ids. Claim status starts as `proposed`; only an
explicit human acceptance (`--accept`, reviewer recorded) marks it `supported`. This
keeps an agent from asserting a conclusion on its own.

## Verification (actually executed)

```text
uv run pytest        -> 115 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

Tests cover: freezing without validation fails; tampering with metrics after validation
fails; a clean freeze writes numbers/metrics/provenance with matching hashes; unknown
evidence cannot be cited; duplicate claim ids are rejected; the CLI path works.

A real bug was found and fixed during this phase: the validation summary used the key
`metrics_json_sha256` while the freeze gate read `metrics_sha256`, so legitimate freezes
were rejected. The keys are now explicit.

## Deferred

- Evidence auditing validator (validates store hashes against artifacts on demand).
- Automatic claim generation from an LLM (must remain human/judge accepted).

## Next phase (Phase 12)

Paper layer: a structured paper AST, a figure/table pipeline grounded only in verified
evidence, and a Markdown renderer.
