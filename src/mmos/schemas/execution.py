from datetime import datetime

from pydantic import Field

from mmos.schemas.common import MMOSModel


class ExecutionResult(MMOSModel):
    """Auditable record of one local command execution."""

    execution_id: str
    command: list[str]
    cwd: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    exit_code: int | None
    timed_out: bool = False
    dry_run: bool = False
    stdout_path: str
    stderr_path: str
    error: str | None = None

    def succeeded(self) -> bool:
        return (
            not self.dry_run and not self.timed_out and self.exit_code == 0 and self.error is None
        )
