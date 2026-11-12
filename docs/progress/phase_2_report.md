# Phase 2 Report

Date: 2026-11-12

## Scope

Project intake, immutable input registration, rule-based state machine, and an
orchestrator skeleton that records every transition.

## Created / modified

- `src/mmos/schemas/states.py` — added `ESCAPE_STATES`
- `src/mmos/schemas/project.py` — `ProjectConfig.resume_state` plus a consistency validator
- `src/mmos/schemas/state.py` — `StateTransitionRecord`
- `src/mmos/core/state_machine.py` — linear transition table, escape states, terminal
  state, `allowed_targets()`, `check_transition()`, `InvalidTransitionError`
- `src/mmos/core/project.py` — project creation, project id generation, atomic YAML/JSON
  writes, config/manifest load/save, `resolve_project()`
- `src/mmos/core/intake.py` — safe ingestion into `problem/`, `rules/`, `data/raw/`,
  `data/raw/attachments/`
- `src/mmos/orchestration/orchestrator.py` — `Orchestrator.ingest()` and rule-checked
  `transition()`, appending to `manifests/state_history.jsonl`
- `src/mmos/cli/main.py` — `init`, `ingest`, `status`; console logging honors
  `MMOS_LOG_LEVEL`
- Tests: `test_state_machine.py`, `test_intake.py`, `test_orchestrator.py`, expanded
  `test_schemas.py` and `test_cli.py`
- `README.md` — CLI examples and phase status

## Key behavior decisions

1. **Copy, never move or modify.** Sources are copied with `shutil.copy2`; the original
   is untouched. Tested by comparing the source bytes after ingestion.
2. **Hash-based idempotency per (sha256, role).** Re-ingesting the same file adds no
   duplicate manifest entry.
3. **Name collisions are refused, not overwritten.** If a different file already
   occupies the destination path, ingestion raises instead of clobbering raw data.
4. **Manifest is saved after each file**, so an interrupted batch cannot leave copied
   files that are missing from the manifest.
5. **Escape states carry a resume target.** `BLOCKED`, `FAILED`,
   `REQUIRES_HUMAN_REVIEW` can be entered from any working state; leaving them is only
   legal back to the interrupted state. `RELEASED` is terminal. The invariant is
   enforced in the `ProjectConfig` validator, not only by callers.
6. **Transitions are rule-checked and audited.** Skipping a state is impossible through
   the orchestrator, and every accepted transition is appended to
   `manifests/state_history.jsonl` with a reason and UTC timestamp.
7. **Atomic writes** for `project.yaml` and the manifest avoid partially written state.

## Verification (actually executed)

```text
uv run pytest        -> 26 passed
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
end-to-end CLI smoke -> init, ingest, status in a temp directory:
                       INIT -> PROJECT_INGESTED, 1 data input, manifest + history written
```

## Deliberately deferred

- File parsing (PDF/Word/Excel) — Phase 4 data subsystem; intake only registers bytes.
- Event bus — the state history file is the auditable transition log for now; a typed
  event bus will be introduced when more than one consumer exists.
- Agent-driven automatic transitions — Phase 3+.

## Next phase (Phase 3)

Agent runtime: provider abstraction, PydanticAI integration, typed ProblemSpec output,
and a Problem Analyst agent. This is the first phase that needs a configured LLM API key
for a real end-to-end test; provider/model/key availability will be checked first.
