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
- Final documentation commit: `docs(phase-24.1): record historical blind benchmark`

No push. Stop after Phase 24.1; do not start Phase 24.2.
