# Phase 3 Report

Date: 2026-11-12

## Scope

Agent runtime, provider boundary, structured `ProblemSpec`, and the Problem Analyst
agent with a CLI entry point.

## Created / modified

- `src/mmos/schemas/problem.py` — `Variable`, `SubProblem`, `ProblemSpec` with a
  dependency-graph validator (unique ids, known references, no cycles)
- `src/mmos/agents/model_factory.py` — `build_model()` for DeepSeek and OpenAI
  (OpenAI-compatible), typed errors for missing key / unsupported provider
- `src/mmos/agents/problem_analyst.py` — PydanticAI `Agent` with
  `output_type=ProblemSpec`, extraction-only system prompt, system-assigned problem id
- `src/mmos/core/project.py` — `problem_spec.json` path plus save/load helpers
- `src/mmos/cli/main.py` — `analyze-problem` command (state guard, text collection,
  model build, save, transition)
- `pyproject.toml` — added `pydantic-ai-slim[openai]`, integration test marker
- Tests: `test_problem_spec.py`, `test_problem_analyst.py`, expanded `test_cli.py`

## Key decisions

1. **Two real providers, no aspirational framework.** DeepSeek (primary) and OpenAI
   (same OpenAI-compatible code path). `anthropic` remains a valid config value but
   fails with a clear "not installed in this build" error rather than silently
   switching providers.
2. **No silent fallback.** Every failure (missing key, unsupported provider, bad
   problem input) is reported and exits non-zero. No model is substituted.
3. **Problem id is system-owned.** The analyst may echo it, but the persisted value is
   always the project id, not the model's claim.
4. **ProblemSpec is validated structurally.** Duplicate ids, dangling dependencies,
   and dependency cycles are rejected before anything is saved.
5. **Text-only problem inputs in this phase.** `.md`, `.markdown`, `.txt`; PDF/Word
   parsing is deferred to the data subsystem phase. Unsupported files raise instead of
   being silently skipped.
6. **TestModel is used only to verify plumbing.** It tests text collection, schema
   validation, persistence, and the state transition. It is not treated as a model
   result and is never written into evidence.

## Verification (actually executed)

```text
uv run pytest        -> 38 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

The skipped test is `test_live_problem_analysis` (marked `integration`): it requires a
real `MMOS_LLM_API_KEY`. **No live provider call has been executed yet**, because no
project-scoped DeepSeek/OpenAI key exists in this environment. This is an unresolved
verification gap, not a passing result.

## Open issue

- Real DeepSeek tool-calling + structured output is unverified. When a key is
  configured, run `uv run pytest -m integration`. If DeepSeek rejects the tool call or
  returns invalid structured output, that becomes a blocking issue for this phase.

## Next phase (Phase 4)

Data subsystem: DuckDB/Polars ingestion profiling, Pandera schema checks, and the Data
Agent producing `DataProfile`, `DataQualityReport`, and `LeakageReport` behind a quality
gate that blocks the model tournament.
