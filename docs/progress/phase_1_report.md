# Phase 0 + Phase 1 Report

Date: 2026-11-12

## Phase 0 — environment

| Check | Result |
| --- | --- |
| Working directory | `C:\Users\zzhn2\Desktop\math good` (empty at start) |
| Git | 2.54.0.windows.1 |
| Python (system) | 3.14.5, plus 3.12.10 available via uv |
| uv | 0.11.11 |
| OS | Windows 11 (MINGW64 bash) |
| Docker | not used, by design |

Repository actions:

- `git init -b main`
- `git remote add origin https://github.com/zzh18012/-agent`

Known environment issue: `git ls-remote origin` failed with
`SSL certificate problem: unable to get local issuer certificate`. This does not block
local development or local commits, but it will block fetch/push until the local CA
configuration is fixed. Not touched in this phase.

## Phase 1 — skeleton

Created:

- `pyproject.toml` — package metadata, runtime and dev dependencies, ruff/pytest/pyright config
- `.python-version` — pinned to 3.12 for scientific-stack compatibility
- `uv.lock` — resolved lock file
- `.gitignore`, `.env.example`
- `README.md` — from-zero explanation of purpose, concepts, install, CLI, layout
- `docs/architecture.md` — four-layer architecture and the core invariant
- `src/mmos/__init__.py` — version
- `src/mmos/config/settings.py` — pydantic-settings `Settings` (LLM provider/model/key, paths, log level)
- `src/mmos/schemas/` — `MMOSModel` base, `ProjectState`, `SourceRole`, `InputFileRecord`, `ProjectConfig`, `ProjectManifest`
- `src/mmos/core/project.py` — project subdirectory contract, `ProjectPaths`, `sha256_file`
- `src/mmos/utils/logging.py` — structlog to console + optional JSONL file
- `src/mmos/cli/main.py` — Typer app with `--version` and `doctor`
- `tests/` — 5 test modules covering schemas, hashing/paths, settings, logging, CLI
- Top-level scaffolding: `configs/`, `workflow/`, `model_registry/`, `skills/`, `templates/`,
  `benchmarks/`, `examples/`, `knowledge/`, `projects/`

## Key design decisions

1. **Python 3.12, not the system 3.14.** The project will depend on DuckDB, Polars,
   MLflow, Snakemake, and ML libraries; 3.12 has the most reliable wheel coverage.
   `.python-version` lets uv select it without touching the system interpreter.
2. **`extra="forbid"` on all persisted schemas.** Typos and schema drift fail loudly.
3. **`ProjectState` exists now, transition rules do not.** The enum is data; the
   rule-based state machine belongs to Phase 2.
4. **Logging is dual-output.** Human-readable console for the developer, JSONL for the
   machine-readable audit trail required by the evidence protocol.
5. **`doctor` is real functionality**, not a stub; unimplemented commands are simply
   not registered yet.
6. **No speculative abstractions.** No provider factory, agent base class, or executor
   is being built before its first real consumer exists.

## Verification (actually executed)

```text
uv sync                 -> 27 packages installed, mmos 0.1.0 editable
uv run python --version -> Python 3.12.10
uv run pytest           -> 11 passed
uv run ruff check .     -> All checks passed
uv run ruff format .    -> 2 files reformatted, 18 unchanged
uv run pyright          -> 0 errors, 0 warnings
uv run mmos doctor      -> Python/uv/git/venv OK; .env absent; API key not configured
```

## Open issues

- Git remote TLS verification fails locally (see above). Needs local CA/proxy
  configuration before any push.
- No `.env` yet; the LLM key is not configured. Not needed until Phase 3.

## Next phase (Phase 2)

Project intake and state: create a project directory, copy/register input files without
modifying originals, compute sha256 into `manifests/project_manifest.json`, write
`project.yaml`, and implement the rule-based state machine plus orchestrator skeleton,
with tests.
