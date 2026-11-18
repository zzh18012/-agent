# Phase 24.1 — Historical Competition Operational Blind Benchmark

## Scope and verdict boundary

Phase 24.1 evaluates whether MathModelingOS can independently run a real historical
competition problem under strict runtime blindness. It deliberately does **not** assess
PDF formatting, page limits, filename conventions, submission-site compliance, or external
competition quality.

This report records one protocol-valid blind attempt, reached after two infrastructure
invalidations exhausted A and C; the canonical reserve attempt used problem B. The attempt
ended at the existing Problem Router with a structured-output capability failure. No
modeling, experiment, evidence, or paper stage was fabricated. The final scientific result
is `BLOCKED`; external competition quality is `NOT_YET_ASSESSED`.

## Operational blindness definition

The benchmark is marked:

```text
operational_blind = true
training_contamination_verifiable = false
```

Operational blindness verifies runtime information isolation: no known solution artifact,
award paper, contest analysis, reflection article, third-party problem mirror, or search
result summary was accessed; no human supplied modeling/algorithm/paper hints; no
best-of-N answer selection occurred. It does **not** prove that relevant material was
absent from pretrained model parameters. No claim of zero contamination is made.

The first round is strict offline blind: after problem reveal, no external web retrieval
was performed. The official archive was downloaded from the official competition site
before reveal. DeepSeek API use was allowed only as the existing model provider.

## Official source provenance

- Competition: 2026 高教社杯全国大学生数学建模竞赛 (CUMCM)
- Official source page: `https://www.mcm.edu.cn/`
- Official archive URL: `https://www.mcm.edu.cn/upload_cn/CUMCM2026Problems.zip`
- Downloaded through the official-site browser session (Kimi WebBridge) to the local
  source directory before problem reveal.
- Archive SHA-256:
  `a54c0e6b552d31e2dbd41aba4a07769943433cb9317927514c2d39ec6442e241`
- Archive size: `4,362,100` bytes
- Source manifest hash (canonical run):
  `4167cc982e6bdf62126aad3d5c4e15c86fd5259b86192328f79a9737d717f169`
- Original archive is immutable; extracted files are hashed in `source_manifest.json`.
  Only label A/B/C are eligible; D/E were never selected or read.

## Deterministic problem selection

The archive hash and labels were used before opening problem bodies:

```text
digest = SHA256(archive_sha256 + ":" + label)
```

Digest order:

```text
A  51fa15c631f5bbfbf6046223c74d199982f004a4fe8db2e9cd621735d2739717  rank 1 (primary)
C  90751a5466fe1d69b96f3fae1cf52fcb42fed47a0816633d31c748afc4a66452  rank 2 (reserve_1)
B  d1e54b11b4e582dad070ab645f637913c4e241105628b8a4bf000b0115cfd837  rank 3 (reserve_2)
```

The canonical selection manifest (run 03) hash is
`e918dabce5c939ec3f02ec3884f4842d86e6ac7ba84e6ab85abd28010892d4e1`.
Selection was committed before any reveal. No human chose or switched a problem.

## Code freeze and invalidation history

Code-freeze records were created before reveal; only time/artifacts were subsequently
added under ignored benchmark runtime directories.

| Attempt | Code freeze commit | Problem | Outcome |
|---|---|---|---|
| `cumcm-2026-blind-01` | `ef184ab` | A (primary) | `INVALIDATED_INFRASTRUCTURE`: selected package contains xlsx attachments; post-reveal harness change required |
| `cumcm-2026-blind-02` | `6716aa4` | C (reserve_1) | `INVALIDATED_INFRASTRUCTURE`: pdftotext dropped Chinese characters; PDF backend fix required |
| `cumcm-2026-blind-03` | `8564092` | B (reserve_2) | Protocol-valid attempt; router structured-output failure; scientific result `BLOCKED` |

The invalidation and fix history is retained rather than squashed. The first two attempts
remain on disk as audit artifacts. Reserve use was authorized only by the preceding
infrastructure invalidation, never by a bad scientific result.

## Canonical run

- Run ID: `cumcm-2026-blind-03`
- Artifact root: `benchmarks/historical/cumcm-2026-blind-03/`
- Code freeze: `856409294036d535a35f40fe6285c5b18a077d56`
- Code freeze hash:
  `0d3ed5e8b7fc65380befde3aa2f3b102471b712ce22c1ab893ee7992e8305ceb`
- Worktree clean at freeze: `true`
- Revealed problem: **B**, hash `5007f89ff6ab9ae4a8e76593138c976c491891027801147cf4d3907f36361f92`
- Problem: B题 无线电干扰源的快速自动定位与清除
- Official files read after reveal: B题.pdf and appendices 1/2 (docx); no solution or
  third-party material.

## Router result

The existing Problem Router was run through `route_project` on the official statement
plus appendices. The `ProblemAnalyst`/`SignalExtractor` path failed structurally:

```text
RoutingOutputError: structured routing output failed
ProblemSignalExtraction: Invalid JSON: trailing characters
UnexpectedModelBehavior: Exceeded maximum output retries (3)
```

A second router invocation used only the main B题 statement (appendices retained for
requirement context) and failed with the same structured-output error. The retry artifacts
are under `router_retry/`; it was a router robustness retry, not an answer-selection run.

No TaskFamily route was produced. This is a capability/robustness finding, not a
retroactive implementation of a new engine. The router was not manually overridden, and
no alternative answer was selected. Consequently no model/experiment/evidence/paper stage
was entered.

Benchmark result hash:
`8149a4d67f91769a13b28c01e22d50c8bff5957af73f77fa5773711ec2f5a700`.

## Requirements

Four explicit requirements were recorded before finalization:

| ID | Requirement | Status |
|---|---|---|
| q1 | Bearing-line intersection polygon diameter and covering-circle question | UNRESOLVED |
| q2 | Second measurement-point selection strategy and candidate region | UNRESOLVED |
| q3 | Unknown 10–16 omnidirectional sources; search/clear strategy, simulation, three official tests | UNRESOLVED |
| q4 | Mixed omnidirectional/directional sources; search/clear strategy, simulation, three official tests | UNRESOLVED |

Coverage: `answered_with_evidence=0`, `answered_with_limitation=0`,
`unresolved=4`. No requirement was silently omitted; no paper claim was generated.

## Capability gaps

`benchmarks/historical/cumcm-2026-blind-03/capability_gaps.json` records:

- `gap-router-output`: structured routing extraction fails on a long official
  multi-appendix statement after retries.
- `gap-geometry`: no existing engine for bearing-line intersection polygons, polygon
  diameter, or measurement-point design.
- `gap-active-search`: no active search / belief update / multi-target robot path
  planning under uncertainty.
- `gap-simulator`: official simulator requires HTTP interaction and official three-run
  logs; strict offline blind execution has no such adapter.
- `gap-directional-model`: no directional-source coverage model.

These are findings only. No capability was implemented during Phase 24.1.

## Audits, evidence, and paper

- Experiments: none (router did not produce a route)
- Evidence: none
- Model / Evidence / Paper freeze: none
- Paper: none; `paper/competition/report.md` was not generated for this benchmark
- Number audit / Math audit / Red Team / Judge: not reached
- Scientific verdict: `BLOCKED`
- External competition quality: `NOT_YET_ASSESSED`

This is an honest scientific/capability block, not a fabricated PASS.

## Blindness audit

`benchmarks/historical/cumcm-2026-blind-03/blindness_audit.json`:

- official archive present and hash matches: true
- selection committed before reveal: true
- code freeze present and commit matches run: true
- no prohibited access: true
- no human scientific intervention: true
- one reveal: true
- reserve authorization: true

Prohibited access count: **0**. Human scientific intervention count: **0**.
The DeepSeek contamination caveat applies: this audit does not establish absence of
relevant training data.

## Source artifact immutability

The Phase 22 source release, augmentation evidence, and Phase 23 canonical controller
replay were not modified. Their known hashes were rechecked:

- Source release SHA-256:
  `5165180adc3b8f56b39201566b0163a87fb27f90ab0024fddc7c493a578d0282`
- Augmentation evidence store SHA-256:
  `a1060145dbd57029afbfe9a44165c6f1d8bfe6a07188c83e1d9e533ef85d16a1`
- Phase 23 controller run `ctrl-b0e30df9-phase231c` remains `closed`,
  `scientific_gate_failed=true`, unresolved `crit-a1`, `crit-a2`.

## Tests and checks

- `uv run pytest`: **333 passed, 3 deselected**
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings

Historical benchmark tests cover source hashing/tampering, deterministic selection,
selection-before-reveal, reserve authorization, prohibited access and human intervention
audit failure, package extraction/ingestion, Unicode PDF extraction, code-freeze/code-tree
tamper detection, requirement mapping, result hash mutation detection, and Phase 22/23
artifact immutability.

## Real bugs found and fixed

- Selected problem packages contain xlsx data attachments; the first harness attempted to
  parse them as problem text after reveal. Fixed by restricting statement extraction to
  problem documents; attempt 01 invalidated.
- pdftotext on the official PDFs dropped Chinese characters. Fixed with a PyMuPDF Unicode
  extraction backend and a regression test; attempt 02 invalidated.
- Reserve use across the invalidated run was not authorized; added explicit
  `reserve_authorization.json` backed by a prior infrastructure invalidation; regression
  tested.

## Limitations and next assessments

- The canonical run ends at the Problem Router; no modeling path, experiments, evidence,
  paper, or Red Team/Judge stage was reached.
- The harness still requires manual adaptation for non-PDF appendix formats in future
  runs; the docx conversion path emitted a Windows encoding warning in this run although
  the PDF statement was preserved.
- No reference solutions, official analyses, or award papers were opened. Phase 24.2
  post-hoc evaluation remains prohibited until the Phase 24.1 artifacts are frozen and a
  new explicit authorization is given.
- No award prediction or external competition-quality claim is made.

## Commits

- `ef184ab feat(phase-24.1): add historical blind benchmark harness`
- `6716aa4 fix(phase-24.1): route reserve benchmark after infrastructure invalidation`
- `8564092 fix(phase-24.1): preserve Unicode problem text with PDF backend`
- `07ab08d feat(phase-24.1): add historical blind benchmark` (progress report)

No push. Stop after Phase 24.1; do not start Phase 24.2.

# Phase 24.R1 — Router Robustness Remediation

## 1. Scope

This remediation only hardens the Problem Router structured-output path. It does not add
computational geometry, active search, directional-source modeling, a simulator adapter,
a new TaskFamily, solver, or DAG infrastructure. The burned B fixture is used only as a
regression fixture; it is not a new blind attempt.

## 2. Root cause

Phase 24.1 used PydanticAI `Agent[..., ProblemSignalExtraction]` over the DeepSeek
OpenAI-compatible Chat Completions transport. The model output was handled as a tool call;
on the official B statement the response contained a JSON object followed by extra
characters and failed with `Invalid JSON: trailing characters`, exhausting the agent
retries. The router did not distinguish syntax, schema, semantic, truncation, or empty
output failures.

## 3. New structured-output path

A minimal helper `mmos.agents.structured.request_structured` now:

- uses the existing OpenAI-compatible Chat Completions transport with
  `response_format={"type": "json_object"}` and `temperature=0`;
- constructs the JSON Schema from the existing Pydantic models (`ProblemSignals`,
  `ProblemSignalExtraction`, `ProblemSpec`, `RoutingDecision`) rather than maintaining
  handwritten schemas;
- performs syntax, schema, and semantic validation in that order;
- records every attempt to `structured_attempts.jsonl` and raw local diagnostics under
  `structured_raw/`;
- uses bounded, error-aware retries (maximum 3) with deterministic repair instructions;
- never guesses or repairs malformed JSON.

`ProblemAnalyst` and `SignalExtractor` use this path for real provider models; `TestModel`
continues to use the existing PydanticAI path for offline tests.

Transport after: `openai_chat_completions` (DeepSeek-compatible), structured mode
`json_object`. Provider: `deepseek`; model: `deepseek-chat`.

## 4. Validation and failure taxonomy

- Syntax: one JSON object after whitespace; explicit safe ```json fence unwrap only when
  the entire content is one fenced object.
- Reject trailing prose, multiple objects, empty content, truncated content, non-objects.
- Schema: full Pydantic validation; field-level errors are recorded.
- Semantic: signal subproblem ids must exactly match the `ProblemSpec`, with no duplicates
  or missing/extra ids.
- Failure codes: `EMPTY_OUTPUT`, `TRUNCATED_OUTPUT`, `NON_JSON_OUTPUT`,
  `TRAILING_CONTENT`, `MULTIPLE_OBJECTS`, `SCHEMA_INVALID`, `SEMANTIC_INVALID`,
  `PROVIDER_ERROR`, `RETRY_EXHAUSTED`.
- Problem text is wrapped as data and the signal prompt explicitly states that embedded
  instructions are not system instructions.

Schema hashes:

```text
ProblemSignals          7686e251a30dbb89461bf6cd83f3e0fdc31aa58d96ba0aec818829f734081250
ProblemSignalExtraction cdafc21982aba3413fd5c7fdbf6bb5c76d73d97a7419c2305c50ed860c47ee09
RoutingDecision         f3da6f47d1dace4be926fa01fd07c3f40dd6f0c0a9cd026c36d0f39529b72918
```

## 5. B burned-fixture regression (live DeepSeek)

- Source benchmark: `cumcm-2026-blind-03` (problem B, burned)
- Regression artifact: `benchmarks/remediation/router/r1-b-fixture-full-3/`
- Before: `RoutingOutputError: structured routing output failed; ProblemSignalExtraction:
  Invalid JSON: trailing characters`.
- After: structured extraction completed without transport/parser failure.
- Attempts: 2 total (`problem_spec` attempt 1 passed; `problem_signals` attempt 1 passed);
  no retries required.
- Provider/model: `deepseek` / `deepseek-chat`
- Finish reasons: both `stop`; prompt/completion tokens recorded in diagnostics.
- Final route: `SINGLE_TASK` / `optimization` / `READY_FOR_SINGLE_TASK`; routing audit
  passed. This is a Router transport result, not a new blind scientific result.
- Original `cumcm-2026-blind-03` result hash remains
  `8149a4d67f91769a13b28c01e22d50c8bff5957af73f77fa5773711ec2f5a700`; it was not
  rewritten.

The B route remains a capability question: the existing optimization scope is the
0/1-knapsack MILP slice, while B's explicit requirements include computational geometry,
active sensing, multi-target robot search, directional sources, and an official HTTP
simulator. Those gaps remain recorded and are not implemented here.

## 6. Frozen 24-case Router benchmark

The Phase 21.1 suite still passes with no regression:

```text
suite: routing-suite-v1
manifest_hash: 63cd847101c7c736a084b4d943b4ce59ce1715cd436870e1451a0732ac40beae
status_accuracy: 1.000 / 1.000
single_task_accuracy: 1.000 / 1.000
multi_task_recall: 1.000 / 1.000
ambiguous_detection: 1.000 / 1.000
unsupported_detection: 1.000 / 1.000
critical_unsafe: 0 / 0
stability: 1.000
```

Artifacts: `benchmarks/remediation/router/r1-24case/`.

## 7. Tests

New `tests/test_structured_output.py` covers valid JSON, safe fenced JSON, trailing prose,
multiple objects, blank output, truncation and increased retry budget, schema field errors,
semantic errors, retry exhaustion, diagnostics, and problem-text-as-data isolation.
`tests/test_routing.py` adds semantic validation for missing/duplicate subproblem ids.
Existing ProblemAnalyst and routing tests remain green.

Full verification:

- `uv run pytest`: **345 passed, 3 deselected**
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings

## 8. Bugs fixed

- DeepSeek base URL was omitted when `MMOS_LLM_BASE_URL` was empty; direct structured
  client now falls back to `https://api.deepseek.com`.
- Tool-call structured output was brittle on trailing content; replaced for real providers
  with JSON-object mode plus strict parsing and retry taxonomy.
- Signal semantics could accept missing/duplicate subproblem ids after schema validation;
  now rejected.
- Empty and truncated outputs are now classified and retried instead of collapsing into a
  generic JSON error.

## 9. Remaining capability gaps

Router transport robustness is repaired, but the scientific capability gaps from Phase
24.1 remain: computational geometry/bearing localization, active search/path planning,
multi-target sequencing, directional-source modeling, official simulator integration,
and the limited optimization scope. No new capability was added in R1.

## 10. Commit

Phase 24.R1 commit: `fix(phase-24.r1): harden structured problem routing`

No push; stop after R1. Do not start Phase 24.2 or a fresh blind benchmark.

# Phase 24.2 — Fresh Historical Blind Benchmark #2

## 1. Freshness preflight and burn registry

Before downloading or opening 2025 A/B/C, a freshness preflight scanned local historical
benchmark records, the burn registry, and runtime project problem specs. The pre-reveal
registry had no 2025 labels burned:

```text
competition: CUMCM
year: 2025
burned_labels: []
preflight_registry_hash: bcc23f268d53d185b54d0b3f66906d9ae53acae4309f7488db2004b4b43c549a
```

After the run, B is recorded as burned:

```text
burned_labels: ["B"]
post-run registry hash: 2700990ab755d40a44739f4c9e8f725b3f2959a897ddf4e1e2f1fb4df951b7d8
```

2026 A/B/C remain burned from Phase 24.1 and were not reused. No 2025 solution, lecture,
award paper, blog, or third-party problem mirror was accessed. Training-contamination
caveat remains:

```text
operational_blind = true
training_contamination_verifiable = false
```

## 2. Official source and selection

- Competition: 2025 CUMCM
- Official source page: `https://www.mcm.edu.cn/`
- Official archive URL:
  `https://www.mcm.edu.cn/upload_cn/node/759/SvpohSGacdffe718bcaa3b6e835c03ae3461cab1.zip`
- Archive SHA-256:
  `cef6262c24ee3017bdab4ca255299c7b47b2700ad89fd773addde7e241e7e4de`
- Archive size: `52,097,165` bytes
- Source manifest hash:
  `ab1c0df1e6a23e44d28b3e48ac2255981e9e267d41fc9bac8d45fb9a2d5fa258`

Selection was committed before reveal:

```text
B  20beb19ac0e46515d393a5ff44354b32ab366d089420079fd77704e1726d8e26  primary
C  ad096cd2a149baad448eab2633f27d26bea5a97d68236a780a49d4231b30afcd  reserve_1
A  f1c3c2e5c296b959ccfd59aff40d5c84f739a3494abdf2bcd18a085f16bf77b4  reserve_2
```

Selection manifest hash:
`875c8ac5b54eea21ff890d21fb9d0bce56e8cf923d98bb082c13ea9032394122`.
Revealed problem: **B**, reveal hash
`4df5b65919c4659e6fe20f96a85ff8cc8a842d571e88275997c65f290e0a594a`.
Problem: B题 碳化硅外延层厚度的确定.

Code freeze before reveal:

- commit `6fcfa0059f24864f8bb9fcc6746cdfc812d54cbe`
- clean worktree: `true`
- freeze hash: `7e912ab6ec10f5c288ca35601e4c9b3beb7a80833ca800b4fc934d46933bc833`
- tests: 349 passed, 3 deselected; ruff/pyright clean

No infrastructure invalidation occurred; primary B was used. No reserve was used.

## 3. Router result and diagnostics

The Phase 24.R1 DeepSeek structured path ran successfully:

- Provider/model: `deepseek` / `deepseek-chat`
- Transport: OpenAI-compatible Chat Completions, `json_object`
- `problem_spec` attempt 1: syntax/schema passed, finish `stop`, 8652 chars, 3378
  completion tokens
- `problem_signals` attempt 1: syntax/schema/semantic passed, finish `stop`, 2294 chars,
  520 completion tokens
- No retry or transport failure
- Router decision: `AMBIGUOUS` / `BLOCKED_NEEDS_CLARIFICATION`
- Router audit: passed
- No TaskFamily required; subproblem routes q1/q2/q3 had no family
- Clarifications included missing reliability metrics, unknown refractive-index handling,
  multi-beam interference criteria, and dispersion-model requirements

## 4. Requirements and Capability Fit

ProblemSpec produced three explicit requirements:

| ID | Requirement | Capability fit |
|---|---|---|
| q1 | Derive single-reflection interference thickness model | `UNSUPPORTED` |
| q2 | Design thickness algorithm for measured spectra and assess reliability | `UNSUPPORTED` |
| q3 | Derive multi-beam interference conditions and corrected thickness algorithm | `UNSUPPORTED` |

Capability Fit uses the formal capability descriptors: tabular prediction/forecasting,
0/1 knapsack MILP optimization, M/M/1 simulation, max-flow graph, and bi-objective binary
portfolio. None matches optical thin-film interference, spectral analysis, physical
parameter inversion, dispersion modeling, or multi-beam interference physics.

Capability Fit verdict: **UNSUPPORTED**. All three requirements are unresolved. No
modeling, experiments, evidence, paper, or Controller run was started. This is a
scientific/capability block, not an infrastructure invalidation.

Capability gaps (not implemented):

- optical thin-film interference and Fresnel reflection modeling
- spectral reflectance analysis and fringe-based thickness inversion
- refractive-index dispersion/parameter estimation
- multi-beam interference criteria and correction
- reliability assessment for the physical measurement model

## 5. Final verdict

```text
Protocol validity: VALID
Blindness audit: PASS
Prohibited access count: 0
Human scientific intervention count: 0
Router: PASS (structured transport)
Capability Fit: UNSUPPORTED
Scientific verdict: BLOCKED
External competition quality: NOT_YET_ASSESSED
```

Benchmark result hash:
`23170003a04ca75266d831634abf898d0ec6682da110d775355c89f77676a3ae`.
Artifact root: `benchmarks/historical/cumcm-2025-blind-01/`.

## 6. Artifacts and immutability

- `cumcm-2025-blind-01` is `VALID` / `BLOCKED`; no paper or competition release.
- 2026 `cumcm-2026-blind-03` result remains
  `8149a4d67f91769a13b28c01e22d50c8bff5957af73f77fa5773711ec2f5a700`.
- Phase 22 source release hash and augmentation evidence hash are unchanged.
- Phase 23 controller replay remains `closed` / `scientific_gate_failed=true`.
- No solution or post-contest analysis was opened.

## 7. Tests

- `uv run pytest`: **349 passed, 3 deselected**
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings

New tests cover capability-slice mismatch under a valid TaskFamily, partial/full
requirement coverage, capability descriptor hash mutation, freshness preflight/burn
registry, reserve restrictions, and failure-to-model behavior.

## 8. Commits

- `6881f8c feat(phase-24.2): add capability fit and freshness preflight`
- `6fcfa00 chore(phase-24.2): ignore local burn registry`
- Final Phase 24.2 commit after this report: `feat(phase-24.2): run second fresh historical blind benchmark`

No push. Stop after Phase 24.2; do not start Phase 25 or implement capability gaps.

# Phase 24.R2 — Execution Disposition Semantics

## 1. Problem

The Router and Capability Fit were independent but had no deterministic reconciliation
layer. The fresh 2025 B run exposed:

```text
Router: AMBIGUOUS / BLOCKED_NEEDS_CLARIFICATION
Capability Fit: UNSUPPORTED
```

The actual meaning is: the requirements were extracted, but the formal capability
registry has no executable/validatable path. It is not “ask the user to explain the
problem.” R2 adds a separate, deterministic `ExecutionDisposition` after Router and
Capability Fit.

## 2. New disposition layer

Added `mmos.historical_benchmark.disposition` and
`ExecutionDispositionRecord`:

```text
READY
NEEDS_CLARIFICATION
PARTIALLY_SUPPORTED
UNSUPPORTED_CAPABILITY
BLOCKED_INVALID_INPUT
BLOCKED_ROUTING_FAILURE
BLOCKED_CONTRACT_INCONSISTENCY
```

Reason codes are limited to:

```text
ALL_CORE_REQUIREMENTS_SUPPORTED
MULTIPLE_SUPPORTED_ROUTES_NEED_RESOLUTION
NO_CORE_REQUIREMENT_SUPPORTED
SOME_CORE_REQUIREMENTS_UNSUPPORTED
ROUTER_DECLARED_UNSUPPORTED
ROUTER_CAPABILITY_CONTRADICTION
CAPABILITY_ASSESSMENT_MISSING
INVALID_PROBLEM_INPUT
ROUTING_TRANSPORT_FAILED
```

The disposition is computed only from typed Router decision, typed Capability Fit, typed
ProblemRequirements and explicit error-path flags. No LLM decides it.

## 3. Decision precedence

- `Capability Fit = UNSUPPORTED` → `UNSUPPORTED_CAPABILITY`, even when Router is
  `AMBIGUOUS`; clarification cannot create a missing runner/validator.
- `SINGLE_TASK` / `MULTI_TASK`:
  - supported → `READY`
  - partially supported → `PARTIALLY_SUPPORTED`
  - unsupported → `UNSUPPORTED_CAPABILITY`
  - missing assessment → `BLOCKED_CONTRACT_INCONSISTENCY`
- `AMBIGUOUS`:
  - supported → `NEEDS_CLARIFICATION`
  - partially supported with a fully supported route candidate → `NEEDS_CLARIFICATION`
  - partially supported with no fully supported candidate → `PARTIALLY_SUPPORTED`
  - unsupported → `UNSUPPORTED_CAPABILITY`
- Router `UNSUPPORTED` + Fit `UNSUPPORTED` → `UNSUPPORTED_CAPABILITY`.
- Router `UNSUPPORTED` + Fit `SUPPORTED`/`PARTIALLY_SUPPORTED` →
  `BLOCKED_CONTRACT_INCONSISTENCY` (`ROUTER_CAPABILITY_CONTRADICTION`).
- Invalid input / routing transport failure → typed blocked states.

Only `READY` permits a formal modeling execution. `PARTIALLY_SUPPORTED` cannot claim full
problem completion. `NEEDS_CLARIFICATION` is emitted only when a fully supported route
candidate exists and route resolution is genuinely needed.

## 4. 2025 B burned regression

A deterministic replay read the frozen canonical artifacts, without a new LLM call:

- Router: `AMBIGUOUS` / `BLOCKED_NEEDS_CLARIFICATION`
- Capability Fit: `UNSUPPORTED`
- Requirements: q1/q2/q3 unsupported
- **ExecutionDisposition: `UNSUPPORTED_CAPABILITY`**
- Reason: `NO_CORE_REQUIREMENT_SUPPORTED`
- Affected requirements: `q1`, `q2`, `q3`
- Disposition record hash:
  `212aed5f19ae0b835724510778da839545685370690f478dfb883191318d208c`
- Artifacts: `benchmarks/remediation/disposition/cumcm-2025-blind-01/`
- Canonical 2025 result hash before/after:
  `23170003a04ca75266d831634abf898d0ec6682da110d775355c89f77676a3ae`
  (unchanged)
- 2026 result hash remains
  `8149a4d67f91769a13b28c01e22d50c8bff5957af73f77fa5773711ec2f5a700`.

This resolves the R2 semantic question without rewriting the canonical benchmark verdict:
the canonical run remains VALID / BLOCKED; it would now report
`UNSUPPORTED_CAPABILITY` at the disposition layer.

## 5. Staleness

Disposition records contain Router hash, Capability Fit hash and requirements hash.
`disposition_is_stale` invalidates a disposition when any input changes. Old dispositions
must not be reused after Router, Fit or Requirement changes.

## 6. Tests

New `tests/test_disposition.py` covers the full truth table:

- SINGLE/MULTI + SUPPORTED/PARTIAL/UNSUPPORTED
- AMBIGUOUS + SUPPORTED/UNSUPPORTED/PARTIAL (with and without a fully supported candidate)
- UNSUPPORTED + UNSUPPORTED/SUPPORTED contradiction
- invalid input, routing failure, missing fit
- modeling guard only for READY
- staleness for router/fit/requirements hash changes
- canonical 2025 B replay and result-hash immutability

Verification:

- `uv run pytest`: **357 passed, 3 deselected**
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings

## 7. Real bugs fixed

- Router `AMBIGUOUS` and capability `UNSUPPORTED` could be presented as a single
  “clarification” blocker; separated into independent Router, Capability Fit and
  ExecutionDisposition layers.
- Missing capability assessment could be treated as a route outcome; now fails closed as
  `BLOCKED_CONTRACT_INCONSISTENCY`.
- Router/Capability contradictions are now explicit instead of selecting one arbitrary
  result.
- Modeling entry is now gated on `ExecutionDisposition == READY`, not Router status alone.

## 8. Commit

`fix(phase-24.r2): reconcile routing and capability disposition`

No push. Stop after R2; do not open 2025 B solutions, implement optical capabilities, start
fresh blind #3, or start Phase 25.

# Phase 24.3 — Fresh Historical Blind Benchmark #3

## 1. Freshness preflight and burn state

Before opening any 2024 A/B/C content, a freshness preflight scanned historical run
metadata, the burn registry, remediation records and runtime project specs. The pre-reveal
2024 registry had no burned labels:

```text
pre-reveal burned_labels: []
pre-reveal registry hash: bd4de93b4d064d2d49a6fa706356c49cf99dcae5fa5eb5ecb85ca2f1c8337c8a
```

After reveal, B is burned:

```text
post-run burned_labels: ["B"]
post-run registry hash: a6d7f2c69deff38cec2823787f35b00af87638371444115550be6b3cf336068a
```

2026 A/B/C and 2025 B remain burned. Training-contamination caveat remains:

```text
operational_blind = true
training_contamination_verifiable = false
```

No 2024 solution, lecture, award paper, blog, or third-party problem mirror was accessed.

## 2. Source and selection

- Competition: 2024 CUMCM
- Official source page: `https://www.mcm.edu.cn/`
- Official archive URL:
  `https://www.mcm.edu.cn/upload_cn/node/725/pmkWxf8H9cfe9984c1a1a5b1263e5dd3b5596ed5.zip`
- Archive SHA-256:
  `38d9effcede947354f9e9a9c2b4fc68947d83a77c2ff75737e9a662888158726`
- Archive size: `105,537,539` bytes
- Source manifest hash:
  `8531c9582de8d74e51c121066bf0828f9bbe5b179dad7d5e78c083dab0ed39a6`

Selection before reveal:

```text
B  b4ea80d7413d618378362db6547c6f60286e4c2fc74ca8056489b8b493b362fd  primary
A  b8494e2c527016ad635262c8e58f70c9867c41408bd7965b6cea7b112eb3a9ec  reserve_1
C  bb7d6667ef4213aaa879830aa7c8e72b170187ec1d608b1fe28c4089203ebfb7  reserve_2
```

Selection manifest hash:
`663fbf17b7427546645ef377c34d78121984869b7ff556a30064a9340df4a8fc`.
Revealed problem: **2024 B 生产过程中的决策问题**, reveal hash
`3eb07449df68a38ea78dd6326867cc6c323a8a9f6d788692e6b955b4ef2fac37`.

Code freeze before reveal:

- commit `11c1fa23164c888be065e569395dcfbbc423bb85`
- clean worktree: `true`
- freeze hash: `fb9f27ec9113a6604d8e16acd8e72467d6c664ca5bb54993898f4e3762e3f743`
- tests: 360 passed, 3 deselected; ruff/pyright clean

No infrastructure invalidation occurred.

## 3. Router

- Provider/model: `deepseek` / `deepseek-chat`
- R1 OpenAI-compatible Chat Completions, `json_object` structured mode
- Attempts: 3 total
  - `problem_spec` attempt 1: schema invalid after a JSON parse, bounded retry
  - `problem_spec` attempt 2: syntax/schema passed, finish `stop`, 8703 chars / 4134
    completion tokens
  - `problem_signals` attempt 1: syntax/schema/semantic passed, finish `stop`, 3068 chars
    / 708 completion tokens
- Router decision: `SINGLE_TASK` / `optimization` / `READY_FOR_SINGLE_TASK`
- Routing audit: passed
- No LLM transport failure; the R1 schema/retry path handled the malformed first attempt.

## 4. Requirements and Capability Fit

Four explicit requirements were extracted:

| ID | Requirement | Capability Fit |
|---|---|---|
| q1 | Minimum-cost sampling inspection / acceptance decision with confidence levels | `UNSUPPORTED` |
| q2 | Per-stage inspection / assembly / disassembly / exchange decision for six cases | `UNSUPPORTED` |
| q3 | General m-stage, n-component production decision instance | `UNSUPPORTED` |
| q4 | Redo q2/q3 when defect rates are estimated through sampling | `UNSUPPORTED` |

Capability descriptors are limited to tabular prediction/forecasting, 0/1 knapsack MILP,
M/M/1 simulation, max-flow, and bi-objective binary portfolio. They do not cover:

- acceptance sampling / hypothesis-test design with error probabilities
- stochastic defect/yield propagation with sequential rework
- general multi-stage BOM inspection/disassembly optimization
- sampling-uncertainty propagation into production decisions

Capability Fit verdict: **UNSUPPORTED**.

## 5. Execution Disposition and coverage

- ExecutionDisposition: `UNSUPPORTED_CAPABILITY`
- Reason: `NO_CORE_REQUIREMENT_SUPPORTED`
- Disposition record hash:
  `16abf643534761bb9395ed2c16e7e00d6460c6ee451e03fab4b5ebb05d68ad5b`
- Coverage: 4 core requirements, 0 supported, 0 partial, 4 unsupported
- coverage_fraction: `0.0`
- First blocking stage: `capability_fit`
- Router reached: true; Capability Fit reached: true; Modeling: not reached
- Paper/audits/Judge: not reached

## 6. Capability gaps

Detailed gaps are under
`benchmarks/historical/cumcm-2024-blind-01/capability_gaps_detailed.json`:

- acceptance-sampling / confidence-controlled inspection design (potentially general)
- discrete production-stage inspection/assembly/disassembly/exchange optimization with
  defect propagation (potentially general)
- general m-stage/n-component BOM decision optimization (potentially general)
- sampling-uncertainty propagation into production decisions (potentially general)

No capability was implemented in this phase.

## 7. Final verdict

```text
Protocol validity: VALID
Blindness audit: PASS
Router: PASS
Capability Fit: UNSUPPORTED
Execution Disposition: UNSUPPORTED_CAPABILITY
Scientific verdict: BLOCKED
External competition quality: NOT_YET_ASSESSED
```

Benchmark result hash:
`01d604017896d561a27b5714e8e793135100c3c405f460557c20047288b80e75`.
Artifact root: `benchmarks/historical/cumcm-2024-blind-01/`.

## 8. Cross-run first blockers

```text
2026 blind: Router structured-output failure (later remediated by R1)
2025 blind: Capability Fit — optical/spectral/inverse requirements unsupported
2024 blind: Capability Fit — sampling / production-decision requirements unsupported
```

These are capability samples, not a success rate.

## 9. Immutability and verification

- 2026 canonical result remains
  `8149a4d67f91769a13b28c01e22d50c8bff5957af73f77fa5773711ec2f5a700`
- 2025 canonical result remains
  `23170003a04ca75266d831634abf898d0ec6682da110d775355c89f77676a3ae`
- Phase 22 source release / augmentation evidence hashes unchanged
- Phase 23 controller replay unchanged
- `uv run pytest`: **360 passed, 3 deselected**
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings
- No new infrastructure bug was exposed; the R1 retry path correctly recovered from a
  schema-invalid first structured attempt.

## 10. Commit

Phase 24.3 coverage code: `11c1fa2 feat(phase-24.3): add capability coverage snapshot`
Final Phase 24.3 report commit: `feat(phase-24.3): run third fresh historical blind benchmark`

No push. Stop after Phase 24.3.

# Phase 24.C1 — Statistical Inference & Uncertainty Foundation

## 1. Why this capability was added

Phase 24 blind evidence after Router remediation:

```text
2025 B: 3 core requirements, 0 supported, 0 partial, 3 unsupported
2024 B: 4 core requirements, 0 supported, 0 partial, 4 unsupported
```

Seven core requirements produced zero formally supported capability slices. C1 is the
first benchmark-driven capability expansion, limited to statistical inference and local
uncertainty.

## 2. New TaskFamily and capability slices

Added `TaskFamily.STATISTICAL_INFERENCE` with lifecycle support and Router signal fields:

```text
requires_parameter_inference
requires_hypothesis_test
requires_acceptance_sampling
requires_nonlinear_parameter_estimation
requires_parameter_uncertainty
```

Capability descriptors (hashes in `inference_capability_descriptors()`):

- `statistical_inference.binomial_exact` — `80ac999a8eff833a7cf0f2b71508f5475ce4fdba3feba8429c3e489a0dc50692`
- `statistical_inference.acceptance_single_binomial` — `a164f38eec840533b05a19f0e919873778a3ac7b2e0cf33544c566bb4524b7ca`
- `statistical_inference.nonlinear_least_squares_registered_model` — `9a082a9dfa343c25b09def0e93badbe6b4d1d2511cb5061918481b51bd3023e3`
- `statistical_inference.local_parameter_uncertainty` — `140d7dca33051bdf5844b87ea16b1966f519596cb0520984de38c1af942d1815`

The family is not treated as a blanket capability; Capability Fit matches specific slices
from requirement text and keeps unsupported production/optical requirements blocked.

## 3. Mathematical contracts

### Exact binomial inference

- Exact binomial test (`scipy.stats.binomtest`)
- Two-sided or directional alternative
- Two-sided Clopper–Pearson confidence interval via beta quantiles
- Evidence distinguishes `p_value` from CI; no “null proven true” language
- Edge cases k=0, k=n, p0=0, p0=1 tested

### Single-sampling acceptance plan

- Attribute single sampling `(n, c)` under `BINOMIAL_LARGE_LOT`
- Constraints: `P_accept(p_good) >= 1-alpha`, `P_accept(p_bad) <= beta`
- Deterministic selection: minimum `n`, then minimum `c`
- Fixed OC grid `0.00..1.00 step=0.05`
- If no plan exists within `max_sample_size`, `feasible=false`; search bound is a contract
- Validator independently sums the binomial terms and verifies minimum `n`/`c`

### Registered-forward-model NLS

- No `eval`, arbitrary formula, or LLM-generated code
- Registry currently contains only `registered.exponential_decay`:
  `y = amplitude * exp(-decay * x) + offset`
- `scipy.optimize.least_squares`, explicit multi-start, all starts retained
- Claims limited to validated local least-squares solution among declared starts
- Jacobian rank/condition recorded; no global optimum claim
- Local covariance only when loss is linear, `m > p`, and full column rank
- Approximate local-linearization intervals; robust loss gets no covariance

## 4. Live formal experiments

Project: `projects/inference-c1-20261118/inference-c1-20261118-4eeba4`

### Binomial

- Experiment: `exp-c1-binomial`
- k=18, n=100, p0=0.10, greater, 95%
- p-value `0.010007279262125073`
- exact CI `[0.11031122915326057, 0.26947708596681186]`
- Artifact file SHA-256:
  `ef9617d45f7e5782aeb3c87fe982a608409c3091485337517f9546922c7ee2c6`
- 7 Evidence records registered; validator PASS

### Acceptance sampling

- Experiment: `exp-c1-acceptance`
- p_good=0.02, p_bad=0.15, alpha=0.05, beta=0.10, max n=80
- Selected `(n, c) = (34, 2)`
- P_accept(good) `0.9698124693879533`; P_accept(bad) `0.09747379691192189`
- Artifact file SHA-256:
  `053e235bc1a97285bb4c04da2edda2ffdeb7cdeb4ed79318a613fc6a4ae53275`
- 9 Evidence records registered; independent validator PASS

### Nonlinear least squares

- Experiment: `exp-c1-nls`
- Forward model `registered.exponential_decay`
- True synthetic parameters `[2.0, 0.5, 1.0]`
- Estimated `[2.0, 0.5000000000000001, 1.0000000000000002]`
- SSE `1.4791141972893971e-31`; RMSE `1.1102230246251565e-16`
- 2 starts retained; selected start 0; Jacobian rank 3
- Local linearization uncertainty available; approximate=true
- Artifact file SHA-256:
  `2405b6682457e7fea2939e81eae8dfb3edb3541784d3e6a68a0a2eb10fbd6167`
- 20 Evidence records registered; validator PASS

All values originate from validated artifacts; no LLM-authored numbers.

## 5. Router regression and new inference suite

- Frozen v1 `routing-suite-v1` remains 1.000/1.000 on all metrics, stability 1.000,
  critical unsafe 0 (artifacts `benchmarks/remediation/capability-uplift/c1-routing-v1b/`).
- Initial v1 rerun with an over-broad inference prompt incorrectly sent one simulation
  case to inference; the prompt was tightened to exclude simulation/metric uncertainty.
  The clean rerun restored 1.000.
- New `routing-suite-v2-inference` has 8 cases, manifest hash
  `2b86509bfeecbb8c9dd6a89df417ea03a9faa35ef30012d6c4da3ecfc0bbe28b`.
  Live runs: status accuracy 0.875/0.875, single-task 0.833/0.833, multi-task recall
  1.0/1.0, ambiguous 1.0/1.0, unsupported 1.0/1.0, critical unsafe 0, stability 1.0.
  One acceptance-plan case was classified `MULTI_TASK` because the model saw both
  inference and a minimize-n decision; this is recorded, not hidden.

## 6. Burned-benchmark capability uplift replay

Re-routered frozen burned problem text (not new blind attempts) and re-ran Capability Fit
with current descriptors. Canonical results were not modified.

### 2024 B

- Before: `UNSUPPORTED`; supported 0, partial 0, unsupported q1–q4.
- After: `PARTIALLY_SUPPORTED`; q1/q4 partially supported via inference slices;
  q2/q3 remain unsupported (production decision / multi-stage BOM).
- Coverage by fully supported requirements: `0.0` before and after; partial requirements
  increased from 0 to 2.
- Disposition: `PARTIALLY_SUPPORTED`; first blocking stage: `capability_fit`.
- Remaining gaps: multi-stage production inspection/disassembly/rework optimization,
  sampling-uncertainty propagation into decisions, full acceptance-planning integration.

### 2025 B

- Before and after: `UNSUPPORTED`; q1/q2/q3 remain unsupported.
- Reason: generic NLS exists, but there is no optical thin-film forward model, no
  spectral/fringe inversion, and no multi-beam interference model.
- Disposition: `UNSUPPORTED_CAPABILITY`; first blocking stage: `capability_fit`.
- This is the correct boundary: generic parameter estimation does not fabricate a domain
  forward model.

## 7. Tests and verification

- `uv run pytest`: **372 passed, 3 deselected**
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings
- Existing v1, R1 malformed-output, R2 disposition, and Capability Fit tests remain green.
- New inference tests cover exact CI/test semantics, k=0/k=n boundaries, deterministic
  minimum acceptance plans, no-feasible bound preservation, OC curve validation,
  multi-start retention, local uncertainty eligibility, m<=p, robust loss, evidence
  registration, and inference-vs-prediction/optimization routing.

## 8. Real bugs fixed

- Binomial runner originally used scipy directional CI while the validator used a
  two-sided Clopper–Pearson interval; the contract now explicitly uses two-sided exact CI.
- Non-identifiable `ParameterInterval` objects omitted optional fields; now explicitly
  set to `None` and produce no CI.
- Inference routing prompt initially over-triggered on simulation/metric uncertainty,
  dropping frozen v1 accuracy; tightened and v1 rerun restored 1.000.
- Capability Fit was treating the new TaskFamily as one broad descriptor; it now maps
  requirements to four concrete inference slices.

## 9. Limitations

- Only four narrow inference slices; no Bayesian/MCMC, bootstrap framework, GLM, or
  arbitrary forward-model code.
- Acceptance sampling is binomial large-lot single sampling only.
- NLS is local, not global; robust loss gets no local covariance.
- Generic NLS does not provide an optical forward model or production-decision optimizer.
- Uplift is measured as capability coverage, not a re-evaluation of blind results.

## 10. Commit

`feat(phase-24.c1): add statistical inference capabilities`

No push. Stop after C1; do not open burned solutions, implement optical/BOM capabilities,
start fresh blind #4, or start Phase 25.
