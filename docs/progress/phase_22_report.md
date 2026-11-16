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
