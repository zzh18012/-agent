# Phase 13 Report

Date: 2026-11-12

## Scope

Adversarial review of the generated paper: red-team attacks, author responses, and a
judge that must not let unresolved critical issues through.

## Created / modified

- `src/mmos/schemas/review.py` — `Attack`, `AttackReport`, `AuthorResponse`,
  `AuthorResponseReport`, `AttackVerdict`, `JudgeReport`, `RedTeamReview`
- `src/mmos/agents/red_team.py` — attacker, author responder, judge agents; deterministic
  normalization
- `src/mmos/core/project.py` — `reviews/redteam_<experiment>.json` persistence
- `src/mmos/cli/main.py` — `redteam` command
- Tests: 7 unit tests for normalization/aggregation, 1 live end-to-end review

## Enforcement and normalization

1. **Attack ids are system-owned** (`a1`, `a2`, …), so responses and verdicts can be
   matched reliably regardless of model output.
2. **Every attack gets a response.** Missing responses are filled with
   `revise_paper` and marked unresolved; unknown response ids are dropped.
3. **Every attack gets a verdict.** Missing verdicts default to `revise`; unknown
   verdicts are dropped.
4. **Aggregation is conservative and computed in code.** Any critical attack that is not
   `pass` makes the review blocking. A critical `reject` yields overall `reject`;
   otherwise blocking yields `revise`; only all-pass is `pass`.
5. **The judge cannot bypass the evidence layer.** It reviews the paper AST generated
   from verified evidence; it does not create numbers or evidence.
6. Reviews are persisted under `reviews/` with the experiment id.

## Verification (actually executed)

```text
uv run pytest                         -> 131 passed, 3 deselected (integration)
uv run ruff check .                   -> All checks passed
uv run pyright                        -> 0 errors, 0 warnings
uv run pytest -m integration -v       -> live research/council 220s, live red team 45s
```

The live red-team test builds a paper from a real experiment, validates and freezes it,
then runs the three real DeepSeek calls. It checks that responses and verdicts cover
exactly the attack ids.

## Deferred

- Automatic revision loop (author rewrites paper, then re-review).
- Release summary / release gate.
- Full `mmos run` automation.

## Next phase (Phase 14)

Automated pipeline and release summary. Design decision needed: how target/features and
the prototyping proposal are supplied, and whether the pipeline may auto-create
*proposed* (never auto-accepted) claims.
