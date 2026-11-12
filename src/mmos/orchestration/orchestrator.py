from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import structlog

from mmos.core.intake import ingest_files
from mmos.core.project import (
    ProjectPaths,
    load_project_config,
    save_project_config,
)
from mmos.core.state_machine import LINEAR_ORDER, NEXT_STATE, check_transition
from mmos.schemas.project import InputFileRecord, ProjectConfig, SourceRole
from mmos.schemas.state import StateTransitionRecord
from mmos.schemas.states import ESCAPE_STATES, ProjectState


class Orchestrator:
    """Applies rule-based state transitions to one project and records them."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self._log = structlog.get_logger("mmos.orchestrator")

    def ingest(
        self, sources: Iterable[Path], role: SourceRole
    ) -> tuple[list[InputFileRecord], ProjectState]:
        records = ingest_files(self.paths, sources, role)
        config = load_project_config(self.paths)
        if records and config.state is ProjectState.INIT:
            config = self.transition(
                ProjectState.PROJECT_INGESTED,
                reason=f"ingested {len(records)} {role.value} file(s)",
            )
        return records, config.state

    def transition(self, target: ProjectState, *, reason: str) -> ProjectConfig:
        config = load_project_config(self.paths)
        check_transition(config.state, target, config.resume_state)

        resume_state = config.state if target in ESCAPE_STATES else None

        updated = config.model_copy(update={"state": target, "resume_state": resume_state})
        save_project_config(self.paths, updated)
        self._append_history(
            StateTransitionRecord(
                project_id=config.project_id,
                from_state=config.state,
                to_state=target,
                reason=reason,
                at=datetime.now(UTC).replace(microsecond=0),
            )
        )
        self._log.info(
            "state_transition",
            project_id=config.project_id,
            from_state=config.state.value,
            to_state=target.value,
            reason=reason,
        )
        return updated

    def advance_to(self, target: ProjectState, *, reason: str) -> ProjectConfig:
        """Bookkeeping helper: walk the linear lifecycle one legal step at a time.

        It never skips a state, never leaves an escape state, and never moves backwards.
        """
        config = load_project_config(self.paths)
        if config.state in ESCAPE_STATES or config.state not in LINEAR_ORDER:
            raise ValueError(f"cannot advance from state {config.state.value}")
        if target not in LINEAR_ORDER:
            raise ValueError(f"cannot advance to non-linear state {target.value}")
        if LINEAR_ORDER.index(target) <= LINEAR_ORDER.index(config.state):
            return config

        while config.state is not target:
            next_state = NEXT_STATE.get(config.state)
            if next_state is None:
                raise ValueError(f"no path from {config.state.value} to {target.value}")
            config = self.transition(next_state, reason=reason)
        return config

    def _append_history(self, record: StateTransitionRecord) -> None:
        self.paths.state_history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.state_history_file.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
