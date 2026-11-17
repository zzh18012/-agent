# Phase 22.1 — Competition Paper Agent

Date: 2026-11-16

## Goal and confirmed scope

Turn a released, evidence-backed modeling project into a competition-style paper without
inventing results, and make missing experiments first-class artifacts.

Confirmed choices:

1. Paper Agent core + a structured experiment-request manifest. Requests are **not**
   auto-executed in 22.1.
2. Generic competition style (abstract, problem restatement, assumptions, notation, model,
   solution, results, sensitivity, evaluation, limitations, references), not bound to
   CUMCM or MCM/ICM.
3. Markdown + PNG figures, reusing `PaperAST` and the existing renderer.

Hard rules:

- Every numeric token in paper prose must come from an expanded evidence placeholder or an
  allowed problem/release metadata literal.
- The agents must identify gaps and emit structured experiment requests; they are not a
  prose polisher.
- Math audit, experiment analysis, and final Red Team are independent. Paper agents have
  no PASS/FAIL verdict field; the system computes release status.

## Architecture

The stage is a post-release sidecar. `RELEASED` remains terminal and `project.yaml` is not
advanced. The pipeline requires a released project, reads frozen evidence and release
metadata, writes its own paper/audit/request artifacts, and writes a competition release
only after the independent Red Team passes.

```
RELEASED project
  -> EvidenceContext (verified evidence, claims, release, derived plan/forecast evidence)
  -> PaperArchitect (plan; placeholders for all numbers)
  -> MathAuditor + ExperimentAnalyst + FigureDesigner (independent inputs)
  -> deterministic PaperAST + tables + figures
  -> fail-closed number audit
  -> experiment-request manifest
  -> independent RedTeam / Judge
  -> PASS: releases/competition_paper_<run>.md/json
     else: blocked report, no release
```

## Components

### Schemas

`src/mmos/schemas/competition.py` defines:

- `SectionKind`, `SectionPlan`, `ClaimPoint`, `NotationEntry`;
- `TablePlan` templates `metric_comparison`, `experiment_inventory`, `evidence_lineage`;
- `FigurePlan` templates `pipeline_schematic`, `metric_bar`, `uncertainty`;
- `CompetitionPaperPlan` with canonical section order and unique ids;
- `MathAuditReport` / `MathAuditIssue` with no overall verdict;
- `ExperimentAnalysisReport` / `ExperimentFinding` / `CapabilityNeed`;
- `CompetitionExperimentRequest` / `CompetitionExperimentManifest` (`executed=false`);
- `NumberAuditReport`;
- `CompetitionPaperRun`.

### Evidence context

`src/mmos/competition/context.py` loads the problem spec, manifest, verified evidence,
claims, latest release, experiment labels, and model facts. It rejects non-RELEASED
projects and computes a stable context hash.

For the mixed slice it derives additional evidence records directly from the frozen plan
spec, forecast artifact, and production-plan artifact:

- capacity, production cost, holding cost per period;
- initial inventory;
- point forecast, production, and planned inventory per period.

Each derived record carries the source file path and SHA-256, and is persisted in
`paper/competition/derived_evidence.json`. This lets the paper cite cost/capacity values
instead of treating them as unstated numbers.

### Agents

`src/mmos/competition/agents.py` implements four independent PydanticAI agents with
module-level prompts, `retries=2`, temperature 0, and `max_tokens=8192`:

- `PaperArchitect` -> `CompetitionPaperPlan`;
- `MathAuditor` -> `MathAuditReport`;
- `ExperimentAnalyst` -> `ExperimentAnalysisReport`;
- `FigureDesigner` -> up to two supported figures.

The architect receives system-provided `model_facts` (period indexing, drift rule,
lost-sales recursion, aggregation formulas, in-sample residual semantics, unit/capacity
conventions) and may not invent alternative formulas. Evidence ids are normalized for
common `ev:` prefix mistakes, and experiment ids are resolved to their evidence records.
If the math audit returns critical issues, the architect gets one revision attempt; the
auditor then re-audits independently.

### Number audit

`src/mmos/competition/number_audit.py` expands `[[ev:EVIDENCE_ID]]` placeholders from
`EvidenceRecord.display_value`. After expansion, every numeric token must match an
expanded evidence value (including numbers embedded in a cited evidence display string), a
section/table/figure-seeded evidence value, or an allowed problem/release metadata literal.
Structural constants 0–3 are allowed only in model/notation/solution and research-gap
text. Unknown citations and unverified numbers block the run before Red Team.

### Tables, figures, and requests

`src/mmos/paper/competition_builder.py` builds tables and figures deterministically from
evidence; the LLM cannot author table cells. Figures are written as PNG plus a JSON spec
and SHA-256. The paper always includes a "Research gaps and proposed experiments" appendix.

`src/mmos/competition/requests.py` normalizes and deduplicates capability needs, checks
cited evidence, and classifies capabilities against the implemented engines. Unsupported
requests are recorded with a reason and are not blocking by themselves; nothing is executed.

### Pipeline and CLI

`src/mmos/competition_pipeline.py` writes:

- `paper/competition/report.md`
- `paper/competition/plan.json`
- `math_audit.json`, `experiment_analysis.json`, `number_audit.json`,
  `experiment_requests.json`, `derived_evidence.json`, `run.json`
- `paper/competition/figures/*.png` and spec JSON
- `reviews/redteam_<run>.json` after the independent review
- `releases/competition_paper_<run>.json/.md` only on PASS

Previous competition artifacts are copied to `paper/competition/history/<old_run>/` before
a new run. No project state transition occurs.

CLI: `mmos competition-paper <released-project>`.

## Tests

`tests/test_competition.py` (16 tests) covers:

- number-audit placeholder expansion, fabricated-value rejection, unknown citations;
- request classification, deduplication, unknown-evidence rejection;
- RELEASED gating and context catalog construction;
- offline agent normalization and unavailable-figure rejection;
- full pipeline PASS release, unsupported-request non-blocking behavior;
- Red Team block with no release;
- number-audit failure before review;
- critical math-audit block;
- prior-run archiving;
- figure rendering and CLI RELEASED gating;
- schema section-order enforcement.

Full verification: **298 passed**, 3 integration deselected; `ruff check .` passed;
`pyright` 0 errors.

## Live run

- Source project: `projects/mixed-forecast-plan-sim-20261116d-20261116-828a5b` (RELEASED)
- Source release: `releases/mixed_mixed-a5aad5c1-73b2e9.json`
- Context hash: `a3d3fcded8d1e0685986999c2d2f830b1717a56d2bda2316d8efab7f80f1fd58`
- Competition run: `comp-a3d3fcde-f531`
- Paper: `paper/competition/report.md` (SHA-256
  `ab53d7b5fa28dffb6e3b0c8c86c3b9eab12ad47e4fee026f1f31a499a386f7fc`)
- Evidence records in context: 174 (derived + frozen), not truncated
- Number audit: 95 tokens checked, **0 unverified**
- Math audit: 8 issues, **0 critical** (majors/minors reported in the audit file)
- Experiment requests: 9 proposed, 9 supported by current engines, **0 executed**
- Red Team: **PASS**; independent Judge PASS
- Release: `releases/competition_paper_comp-a3d3fcde-f531.md`
- Project state remains **RELEASED**; no lifecycle change

The live loop required several fail-closed iterations: the first architect output exceeded
the token budget, later runs were blocked by critical math-audit findings (indexing,
residual semantics, aggregation, units), unknown evidence ids, and unverified literal
numbers. Each issue was fixed by system-provided model facts, evidence derivation,
seed-aware number auditing, and one bounded architect revision rather than by lowering the
gate.

## Limitations

- Markdown only; no LaTeX/PDF or CUMCM/MCM template binding.
- Requests are proposals only; no request→experiment execution/backfill loop in 22.1.
- The math audit and Red Team are procedurally independent but share the same LLM
  provider, so their errors can be correlated.
- Derived evidence is labelled `competition_context` and traceable to frozen source file
  hashes; it is not a separate scientific validator.
- The paper is a single-instance report. It does not claim robust planning, service
  guarantees, out-of-sample validity, or model superiority.
- Competition paper release is a separate sidecar release; it does not change the parent
  modeling project's release state.

## Commit

`feat(phase-22.1): add competition paper agent`

No push; no Competition Mode or `mmos solve` yet.

# Phase 22.2 — Competition Evidence Augmentation Loop

## Scope

One augmentation cycle was added on top of the Phase 22.1 competition paper sidecar:
resolve the source paper's request manifest deterministically, execute only requests mapped
to existing formal runners, validate and register the results as separate
`competition_experiment` evidence, regenerate the paper from the augmented context, and
re-run the independent number audit, math audit, and Red Team. The source project remains
`RELEASED`; its release, verified evidence index, and artifacts are not modified.

## Implementation

- `src/mmos/competition/augmentation.py`: resolution classes
  `EXECUTABLE_EXISTING_CAPABILITY`, `REQUIRES_NEW_CAPABILITY`,
  `REQUIRES_HUMAN_DECISION`, `NOT_JUSTIFIED`, `INVALID`; formal adapters for
  optimization sensitivity and simulation replication stability; system-owned
  `executed`/`validated`/`evidence_registered` flags; evidence store integrity checks;
  source-release/request-manifest staleness checks; resume without re-executing
  experiments.
- `src/mmos/competition/context.py`: source catalog plus derived competition evidence
  (frozen plan/forecast values, capacity/cost terms, objective recomputation, simulation
  diagnostics, fold means), with only the 54 appended augmentation records excluded from
  the LLM catalog and presented in the appendix.
- `src/mmos/paper/competition_builder.py`: system-owned table/figure captions, frozen
  plan/parameter/objective/simulation diagnostics tables, augmentation reconciliation,
  four-significant-digit rendering (simulation summaries at two), and strict separation
  of appended evidence from source claims.
- `src/mmos/competition_pipeline.py`: deterministic model closure and residual-bias
  statements, notation completion, stale-claim filtering, abstract caveat, and bounded
  math revision.
- `tests/test_competition_augmentation.py` (7 tests) plus expanded
  `tests/test_competition.py` cover resolution, eligible-only execution, failure retention,
  source immutability, manifest/artifact hash staleness, executed-flag rejection,
  caption/rounding behavior, and experiment-task mapping.

## Real run

- Source project: `projects/mixed-forecast-plan-sim-20261116d-20261116-828a5b`
  (remains `RELEASED`)
- Source release: `releases/mixed_mixed-a5aad5c1-73b2e9.json`, SHA-256
  `5165180adc3b8f56b39201566b0163a87fb27f90ab0024fddc7c493a578d0282` (unchanged
  before and after augmentation)
- Source verified evidence SHA-256:
  `13a135e616be9b3a5defdf4d31240713d7f8fb923d5eee8bcfbd12f2a527506e` (unchanged)
- Source context hash: `a3d3fcded8d1e0685986999c2d2f830b1717a56d2bda2316d8efab7f80f1fd58`
- Source competition run: `comp-a3d3fcde-f531`
- Augmentation: `aug-d0e04bfa-a1a9`
- Request manifest hash: `d0e04bfa7e8fddb632046290606463e1d29e50a4f46e4c6efaab6725263104ab`
- Resolution distribution: 3 `executable_existing_capability`, 5
  `requires_new_capability`, 1 `requires_human_decision`.
- Executed and evidence-registered:
  - request-3: 2 optimization experiments, 16 evidence records
  - request-7: 3 optimization experiments, 24 evidence records
  - request-9: 2 simulation experiments, 14 evidence records
  - Total: 7 experiments, 54 new `competition_experiment` evidence records
- Not executed, retained as failures: request-1, request-4, request-5, request-6,
  request-8 (`requires_new_capability`); request-2 (`requires_human_decision`).
- Evidence store: `paper/competition/augmentation/aug-d0e04bfa-a1a9/evidence.json`,
  SHA-256 `a1060145dbd57029afbfe9a44165c6f1d8bfe6a07188c83e1d9e533ef85d16a1`.
- Regenerated paper run: `comp-b0e30df9-8363` (context hash
  `b0e30df918894e2c4ab5367e0cdf6e8dc4d211ce2c3b3914fd638f37eb09b892`), paper SHA-256
  `0f3176411942008bf477e34493a702c54d368c139c040d48d2efb17c680d8e41`.
- Number audit: 59 tokens checked, 0 unverified.
- Math audit: 8 issues, 0 critical.
- Red Team / Judge: **REVISE** (blocking), with two critical findings: (a1) the plan's
  deterministic demand-satisfaction convention is not stated explicitly enough to make the
  capacity-driven build-ahead auditable; (a2) the proxy fold-model residual pool has an
  uncorrected negative mean and no de-meaned rerun exists, so the simulation summaries are
  not interpretable as service. No competition paper release was written.
- Source release and verified evidence hashes are byte-identical before and after the
  augmentation. The project state remains `RELEASED`.

## Outcome and limitations

The one-cycle augmentation loop is implemented and completed its execution/evidence stages:
three requests produced seven validated experiments and 54 registered evidence records,
and the regenerated paper reached the independent Red Team with a clean number audit and no
critical math-audit issue. The paper did not pass the Red Team because the remaining
scientific gaps require new capability or a human decision: de-meaned/alternative residual
resampling, a comparator plan, broader forecast families, a bootstrap/interval policy, and
raw-series/seed-draw publication. These are recorded as unresolved requests rather than
hidden or auto-upgraded. No source claim was revised by augmentation evidence, and no
release was fabricated.

## Verification

- `uv run pytest` (full suite)
- `uv run ruff check .`
- `uv run pyright`

## Commit

`feat(phase-22.2): close competition evidence augmentation loop`

No push; no Phase 23, Competition Mode, CUMCM/MCM binding, LaTeX, deadline, blind
benchmark, or `mmos solve`.
