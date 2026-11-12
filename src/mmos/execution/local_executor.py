import os
import secrets
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import structlog

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.execution import ExecutionResult

DEFAULT_TIMEOUT_SECONDS = 300.0

# Shells and destructive system utilities are never legitimate for this project.
# Blocking shells also means arguments can never be reinterpreted by a shell.
BLOCKED_EXECUTABLES = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "rm",
        "rmdir",
        "del",
        "erase",
        "format",
        "mkfs",
        "diskpart",
        "fdisk",
        "dd",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "taskkill",
        "reg",
        "regedit",
        "bcdedit",
        "schtasks",
        "sc",
        "powershell",
        "pwsh",
        "cmd",
        "bash",
        "sh",
        "zsh",
        "fish",
        "csh",
        "ksh",
        "wsl",
    }
)


class ExecutorSafetyError(RuntimeError):
    """Raised before execution when a command violates the project safety boundary."""


class LocalExecutor:
    """Runs argv commands inside a project directory and records every attempt."""

    def __init__(self, paths: ProjectPaths, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self.paths = paths
        self.timeout_seconds = timeout_seconds
        self._log = structlog.get_logger("mmos.executor")

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
        dry_run: bool = False,
        extra_env: Mapping[str, str] | None = None,
    ) -> ExecutionResult:
        command = [str(token) for token in argv]
        if not command:
            raise ExecutorSafetyError("empty command")

        executable = Path(command[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[: -len(".exe")]
        if executable in BLOCKED_EXECUTABLES:
            raise ExecutorSafetyError(f"executable is blocked by project policy: {command[0]}")

        project_root = self.paths.root.resolve()
        workdir = (cwd or self.paths.root).resolve()
        if not workdir.is_relative_to(project_root):
            raise ExecutorSafetyError(f"cwd is outside the project: {workdir}")
        if not workdir.is_dir():
            raise ExecutorSafetyError(f"cwd does not exist: {workdir}")

        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        if timeout <= 0:
            raise ExecutorSafetyError("timeout must be positive")

        started_at = datetime.now(UTC)
        execution_id = f"exec-{started_at:%Y%m%dT%H%M%S%f}-{secrets.token_hex(3)}"
        execution_dir = self.paths.executions_dir / execution_id
        execution_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = execution_dir / "stdout.log"
        stderr_path = execution_dir / "stderr.log"
        stdout_path.touch()
        stderr_path.touch()

        if dry_run:
            result = ExecutionResult(
                execution_id=execution_id,
                command=command,
                cwd=self._relative(workdir),
                started_at=started_at,
                finished_at=started_at,
                duration_seconds=0.0,
                exit_code=None,
                dry_run=True,
                stdout_path=self._relative(stdout_path),
                stderr_path=self._relative(stderr_path),
            )
            self._save(result)
            self._log.info("execution_dry_run", execution_id=execution_id, command=command)
            return result

        exit_code: int | None = None
        timed_out = False
        error: str | None = None
        started_clock = time.perf_counter()

        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                completed = subprocess.run(
                    command,
                    cwd=workdir,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env,
                    timeout=timeout,
                    shell=False,
                    check=False,
                )
                exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error = f"timed out after {timeout} seconds"
        except OSError as exc:
            error = f"failed to start command: {exc}"

        finished_at = datetime.now(UTC)
        result = ExecutionResult(
            execution_id=execution_id,
            command=command,
            cwd=self._relative(workdir),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - started_clock,
            exit_code=exit_code,
            timed_out=timed_out,
            dry_run=False,
            stdout_path=self._relative(stdout_path),
            stderr_path=self._relative(stderr_path),
            error=error,
        )
        self._save(result)
        self._log.info(
            "execution_finished",
            execution_id=execution_id,
            command=command,
            exit_code=exit_code,
            timed_out=timed_out,
            succeeded=result.succeeded(),
        )
        return result

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.paths.root.resolve()).as_posix()

    def _save(self, result: ExecutionResult) -> None:
        target = self.paths.executions_dir / result.execution_id / "execution.json"
        atomic_write_text(target, result.model_dump_json(indent=2) + "\n")
