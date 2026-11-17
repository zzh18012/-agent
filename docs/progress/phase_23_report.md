# Phase 23.1 — Competition Controller

## Scope

Phase 23.1 adds a post-release, deterministic controller for time-bounded competition
work. It decides what may run now, when exploration closes, when the model / evidence /
paper freeze, when work must stop, and when the competition is closed. It does **not**
add modeling capability, does not alter `ProjectState`, does not regenerate or repair the
Phase 22 blocked paper, and does not implement submission automation.

The canonical exercise is the real Phase 22 blocked run
`comp-b0e30df9-8363` on source project
`mixed-forecast-plan-sim-20261116d-20261116-828a5b`. Its Red Team criticals and
`requires_new_capability` / `requires_human_decision` requests remain unresolved.

## Architecture

New modules:

- `src/mmos/competition/clock.py`: `CompetitionClock`, `SystemClock`, `FrozenClock`,
  timezone-aware validation and UTC conversion.
- `src/mmos/competition/controller_schemas.py`: timing profile, action and work-item
  categories, guard decisions, work queue, freeze records, checkpoints, budget ledger,
  scope-reduction and human-decision records.
- `src/mmos/competition/guard.py`: deterministic lexicographic priority, deadline /
  freeze / capability guards, readiness checks, optional-scope reduction.
- `src/mmos/competition/controller.py`: controller lifecycle, fixture loading from real
  Phase 22 artifacts, `tick()`, freeze revisions, queue/budget/timeline/checkpoint
  persistence, source immutability checks, human decisions.
- `src/mmos/cli/main.py`: `mmos competition init-controller|status|queue|tick|`
  `freeze-model|freeze-evidence|freeze-paper|checkpoint`; no `submit` command.

The controller is independent of the project lifecycle. It reads the `RELEASED` source
release and verified evidence read-only and keeps all controller state under
`competition/runs/<competition_run_id>/`.

## Timing model

`CompetitionTimingProfile` requires timezone-aware datetimes and validates:

```text
contest_start_at < exploration_end_at
                 <= model_freeze_at
                 <= evidence_freeze_at
                 <= paper_freeze_at
                 <= work_stop_at
                 <= submission_deadline_at
```

`work_stop_at == submission_deadline_at` is allowed. Naive datetimes, unknown timezones
and out-of-order milestones are rejected. The profile's `timezone` is used for display;
internal comparisons are UTC. A `FrozenClock` is used for tests and replay; production
commands use `SystemClock`. No LLM is asked for time or deadline information.

## State machine

Monotonic controller phases:

```text
CONFIGURED
→ ACTIVE_EXPLORATION
→ CANDIDATE_CONVERGENCE
→ MODEL_FROZEN
→ EVIDENCE_COMPLETION
→ EVIDENCE_FROZEN
→ PAPER_FINALIZATION
→ PAPER_FROZEN
→ READY_FOR_SUBMISSION_CHECKS
→ CLOSED
```

At a blocked run, `READY_FOR_SUBMISSION_CHECKS` is never entered. `tick()` is idempotent:
freeze records and checkpoint hashes are not regenerated at the same clock instant.
`work_stop_at` sets `work_stopped`, and a non-PASS scientific gate sets
`scientific_gate_failed` and `requires_human_review`. At `submission_deadline_at` the
controller moves directly to `CLOSED` (without passing through a false READY state).

## Work queue policy

`CompetitionWorkItem` retains source, kind, blocking release status, scientific severity,
claim criticality, capability status, runtime class, dependencies and action category.
Priority is a deterministic lexicographic tuple:

```text
blocking_release
→ scientific_severity
→ claim_criticality
→ capability_ready
→ dependency_unlock
→ expected_runtime_class
→ created_at
→ work_item_id
```

No LLM-generated numeric priority is accepted. The queue reports allowed, blocked,
deferred, cancelled and completed items with a reason code for every decision.

Scientific blockers remain first. `REQUIRES_NEW_CAPABILITY` is never auto-executed;
`REQUIRES_HUMAN_DECISION` is blocked. Optional improvements are cancelled after
`exploration_end_at`; critical blockers are never cancelled or reclassified as PASS.

## Freeze semantics

- `ModelFreezeRecord` captures primary/challenger/fallback refs, source project, experiment
  and artifact references, selection evidence ids and a content hash.
- `EvidenceFreezeRecord` captures source + augmentation evidence ids, a combined evidence
  catalog hash, source release hash, augmentation hashes, validation summary, unresolved
  issues and a content hash.
- `PaperFreezeRecord` captures paper run/report hash, evidence freeze id/hash, number and
  math audit status, Red Team/Judge status and a content hash.
- Records are files under `freezes/` and are never overwritten in place. A revision
  requires an explicit revision number and reason; the previous hash is retained and a
  `FreezeInvalidated` event is written.
- Freeze hashes include the controller run id and all referenced material, so changing a
  model reference, evidence id or paper report changes the corresponding hash. Tampered
  freeze files are rejected by hash verification; changed source/report artifacts are
  reported by `verify_artifacts()` without silently updating a freeze.

## Deadline and action guards

Actions are classified as `READ_ONLY`, `COMPUTE`, `MODEL_MUTATION`,
`EVIDENCE_MUTATION`, `PAPER_MUTATION`, `RELEASE_MUTATION` or `SUBMISSION_PREPARATION`.
`DeadlineGuardDecision` records decision, reason code, state, current time, relevant
deadline, work item and action.

Key rules:

- Before `contest_start_at`: mutations are blocked (`contest_not_started`).
- After model freeze: model mutation is blocked.
- After evidence freeze: ordinary new computation/evidence mutation is blocked; only a
  P0 existing-capability repair may create a new evidence revision.
- After paper freeze and before work stop: ordinary paper polish is blocked; only P0
  existing-capability repair is eligible.
- At/after `work_stop_at`: every mutation is blocked; read-only inspection remains allowed.
- At/after `submission_deadline_at`: the controller closes; no remediation is attempted.
- `RELEASE_MUTATION` is always blocked; only the existing Red Team/Judge can authorize a
  competition release.
- `SUBMISSION_PREPARATION` is out of scope in 23.1.

## Scope reduction and human decisions

`auto_scope_reductions` cancels optional improvements that were not started before
exploration ended. It never cancels a scientific blocker, deletes negative evidence,
alters validation status or lowers a gate. `ScopeReductionDecision` records claim
reduction / cancellation actions.

A minimal `HumanDecisionRecord` supports cancelling non-critical work items, recording
fallback choices, reducing a claim and deferring a new-capability request. A human cannot
cancel/promote a scientific critical, alter evidence/validation/number audit, bypass work
stop or the submission deadline, or create a competition release.

## Real replay

- Controller run: `ctrl-b0e30df9-phase231c`
- Source project: `mixed-forecast-plan-sim-20261116d-20261116-828a5b`
- Source paper run: `comp-b0e30df9-8363`
- Source release: `releases/mixed_mixed-a5aad5c1-73b2e9.json`
- Source release SHA-256: `5165180adc3b8f56b39201566b0163a87fb27f90ab0024fddc7c493a578d0282`
- Source verified evidence SHA-256:
  `13a135e616be9b3a5defdf4d31240713d7f8fb923d5eee8bcfbd12f2a527506e`
- Timing profile: `phase23-1-canonical`; timezone `Asia/Shanghai`; profile hash
  `78b0c657b81242649aefae6b3577b568908ce7ad91c52588168ee2f999cf169b`; source hash
  `589ea23f1655a5ff24ba9b72ba2b6af9660e87a3bae955707bf06abc7d262f8b`
- Replay uses `FrozenClock`; only time is simulated, all science/evidence/review artifacts
  are the real Phase 22 files.

State transition timeline:

```text
2026-11-15T00:00Z configured → active_exploration
2026-11-15T06:00Z active_exploration → candidate_convergence
2026-11-15T12:00Z model_frozen → evidence_completion (model-freeze-r1)
2026-11-16T12:00Z evidence_frozen → paper_finalization (evidence-freeze-r1)
2026-11-16T20:00Z paper_frozen (paper-freeze-r1)
2026-11-17T00:00Z work_stopped; scientific gate failed
2026-11-17T02:00Z closed; deadline expired
```

Freeze records:

- Model: `model-freeze-r1` /
  `27c1ec496238cc6121e0bf0ac97df97db95efec8a962dc0052083d40dfe515af`
- Evidence: `evidence-freeze-r1` /
  `8b19ed39c451fdf4029b9bd5d30dc2628fb494d72dcb8490f6f8ca5bc51ea7d3`
  (source 155 + augmentation 54 evidence ids; 2 augmentation hashes)
- Paper: `paper-freeze-r1` /
  `1f321cc25615f85c79fef7eb063cb9f315b178fd1bce3755666702965d8224bc`
- Checkpoint: `checkpoint-9c65059f` /
  `9c65059fe584de17b4b92e2e3170a9511f4085a690ec0fc2771dc059b4431deb`
- Queue hash: `c6e12e6f26eb5f728b7d47a5dff2a4e6ad514d73931e45025133c04030d3c79b`

Final queue counts: 23 planned, 3 completed, 15 blocked, 5 cancelled, 0 deferred, 0
allowed.

- Completed (already executed Phase 22 augmentation requests): `req-request-3`,
  `req-request-7`, `req-request-9`.
- Blocked: `crit-a1`, `crit-a2`, `review-a3`–`review-a9`, `req-request-1`,
  `req-request-2`, `req-request-4`, `req-request-5`, `req-request-6`, `req-request-8`.
- Cancelled by scope reduction: `review-a10`–`review-a14` (optional improvements).

The two unresolved scientific criticals remain `crit-a1` (demand-satisfaction /
capacity-build-ahead auditability) and `crit-a2` (proxy fold-model residual bias /
missing de-meaned rerun). The controller does not fix them and does not create a
competition release. Source release and verified evidence hashes are byte-identical
before and after the replay; `verify_artifacts()` is clean. The Phase 22 blocked paper
remains blocked.

## Verification

- `uv run pytest`: **322 passed**, 3 deselected
- `uv run ruff check .`: passed
- `uv run pyright`: 0 errors, 0 warnings
- New controller tests: 14, including FrozenClock replay, timezone/ordering validation,
  boundary guards, priority ordering, freeze tamper detection, source immutability,
  checkpoint determinism, optional scope cancellation, human-decision constraints and the
  real blocked-fixture critical ordering.

## Real bugs fixed

- Missing `contest_not_started` guard: pre-start mutation could be allowed; fixed.
- Readiness/unresolved tracking counted only currently blocked criticals, dropping an
  allowed-but-unfinished critical; fixed to retain all incomplete criticals.
- Evidence freeze initially excluded the 54 Phase 22 augmentation evidence records; fixed
  to include evidence ids, catalog hash and augmentation hashes.
- A blocked run could have advanced through a false READY state on the way to CLOSED;
  fixed to transition directly to CLOSED when the gate did not pass.
- Manual freeze commands were blocked after the automatic milestone freeze; fixed to be
  idempotent and to require an explicit revision + reason for invalidation.
- Freeze files were not verified before reuse; tampering and changed paper/evidence
  artifacts are now detected read-only.

## Limitations

- No real submission, login, upload or deadline-site automation.
- No CUMCM/MCM profile, PDF/LaTeX renderer, page-limit checker or AI disclosure generator.
- No team collaboration, cloud scheduler or resource scheduler.
- No new scientific engine, forecasting family, optimizer or simulator.
- Only the minimal timing profile; later competition profiles will add format and
  submission constraints.
- The controller uses procedural guard rules, not an LLM, for all time/deadline decisions.

## Commit

`feat(phase-23.1): add competition controller`

No push. Stop after Phase 23.1; do not start Phase 23.2.
