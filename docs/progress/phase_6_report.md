# Phase 6 Report

Date: 2026-11-12

## Scope

`LocalExecutor`: project-scoped, auditable command execution for the compute plane.

## Created / modified

- `src/mmos/schemas/execution.py` — `ExecutionResult`
- `src/mmos/execution/local_executor.py` — argument-list execution, safety checks,
  timeout, stdout/stderr capture, record persistence
- `src/mmos/core/project.py` — `logs/executions/<execution_id>/` location
- Tests: 15 executor tests

## Key decisions

1. **argv only, never a shell.** Commands are passed as a list with `shell=False`.
   Shells (`bash`, `sh`, `cmd`, `powershell`, …) are blocked entirely, so shell
   metacharacters can never be reinterpreted.
2. **Denylist of destructive executables** (`sudo`, `rm`, `dd`, `format`, `diskpart`,
   `schtasks`, …). Refusal raises `ExecutorSafetyError` before anything runs.
3. **Working directory must resolve inside the project.** This is the concrete safety
   boundary, since no container is used.
4. **Every attempt is recorded**, including timeouts and missing executables:
   `execution.json` plus `stdout.log` and `stderr.log` under
   `logs/executions/<execution_id>/`.
5. **Timeouts are enforced** and reported as `timed_out` with `exit_code=None`,
   never as a successful run.
6. **Dry-run mode** records the command without executing it.
7. **Failures are returned as data, not bare strings.** Missing executables yield an
   `ExecutionResult` with `error`; only policy violations raise.

## Verification (actually executed)

```text
uv run pytest        -> 89 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

Executor tests include a real subprocess: stdout/stderr capture, exit codes, a real
timeout, blocked executables, cwd violations, dry-run, and record round-trip.

## Next phase (Phase 7)

An end-to-end Snakemake workflow for a simple CSV regression project: data gate →
baseline → candidate models → cross-validated evaluation → metrics/predictions
artifacts, executed through `LocalExecutor`.
