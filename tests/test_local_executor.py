import sys
from pathlib import Path

import pytest

from mmos.core.project import create_project
from mmos.execution import ExecutorSafetyError, LocalExecutor
from mmos.schemas.execution import ExecutionResult


@pytest.fixture
def project_paths(tmp_path: Path):
    paths, _ = create_project(tmp_path / "projects", "demo")
    return paths


def test_successful_command_captures_stdout_and_record(project_paths) -> None:
    result = LocalExecutor(project_paths).run([sys.executable, "-c", "print('hello')"])

    assert result.succeeded()
    assert result.exit_code == 0
    assert result.cwd == "."
    assert result.timed_out is False
    assert result.error is None
    stdout = (project_paths.root / result.stdout_path).read_text(encoding="utf-8")
    assert stdout.strip() == "hello"

    saved_path = project_paths.executions_dir / result.execution_id / "execution.json"
    saved = ExecutionResult.model_validate_json(saved_path.read_text(encoding="utf-8"))
    assert saved == result


def test_nonzero_exit_code_is_recorded(project_paths) -> None:
    result = LocalExecutor(project_paths).run([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert result.exit_code == 3
    assert result.succeeded() is False


def test_stderr_is_captured(project_paths) -> None:
    result = LocalExecutor(project_paths).run(
        [sys.executable, "-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(1)"]
    )
    stderr = (project_paths.root / result.stderr_path).read_text(encoding="utf-8")
    assert "bad" in stderr


def test_timeout_is_reported(project_paths) -> None:
    result = LocalExecutor(project_paths).run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_seconds=0.2,
    )
    assert result.timed_out is True
    assert result.exit_code is None
    assert result.error is not None and "timed out" in result.error
    assert result.duration_seconds < 4.0


def test_missing_executable_is_reported_not_raised(project_paths) -> None:
    result = LocalExecutor(project_paths).run(["mmos-definitely-not-installed"])
    assert result.exit_code is None
    assert result.error is not None and "failed to start" in result.error


def test_empty_command_is_rejected(project_paths) -> None:
    with pytest.raises(ExecutorSafetyError, match="empty command"):
        LocalExecutor(project_paths).run([])


def test_cwd_outside_project_is_rejected(project_paths, tmp_path: Path) -> None:
    with pytest.raises(ExecutorSafetyError, match="outside the project"):
        LocalExecutor(project_paths).run([sys.executable, "-c", "print('x')"], cwd=tmp_path)


def test_cwd_inside_project_is_accepted(project_paths) -> None:
    result = LocalExecutor(project_paths).run(
        [sys.executable, "-c", "print('inside')"],
        cwd=project_paths.root / "data",
    )
    assert result.succeeded()
    assert result.cwd == "data"


@pytest.mark.parametrize(
    "executable", ["sudo", "rm", "bash", "CMD.EXE", r"C:\Windows\System32\cmd.exe"]
)
def test_blocked_executables_are_rejected(project_paths, executable: str) -> None:
    with pytest.raises(ExecutorSafetyError, match="blocked"):
        LocalExecutor(project_paths).run([executable, "-c", "echo hi"])


def test_non_positive_timeout_is_rejected(project_paths) -> None:
    with pytest.raises(ExecutorSafetyError, match="timeout must be positive"):
        LocalExecutor(project_paths).run([sys.executable, "-c", "print(1)"], timeout_seconds=0)


def test_dry_run_does_not_execute(project_paths) -> None:
    result = LocalExecutor(project_paths).run(
        [sys.executable, "-c", "print('should not run')"],
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.exit_code is None
    stdout = (project_paths.root / result.stdout_path).read_text(encoding="utf-8")
    assert stdout == ""
    assert (project_paths.executions_dir / result.execution_id / "execution.json").exists()
