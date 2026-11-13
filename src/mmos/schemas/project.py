from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.states import ESCAPE_STATES, ProjectState, TaskFamily


class SourceRole(StrEnum):
    PROBLEM = "problem"
    RULES = "rules"
    DATA = "data"
    ATTACHMENT = "attachment"


class InputFileRecord(MMOSModel):
    """Immutable registration of one file ingested into a project."""

    path: str
    filename: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    source_role: SourceRole
    ingested_at: datetime


class ProjectConfig(MMOSModel):
    """Contents of project.yaml."""

    schema_version: Literal[1] = 1
    project_id: str
    name: str
    created_at: datetime
    state: ProjectState = ProjectState.INIT
    task: TaskFamily = TaskFamily.PREDICTION
    resume_state: ProjectState | None = None

    @model_validator(mode="after")
    def _resume_state_is_consistent(self) -> "ProjectConfig":
        if self.state in ESCAPE_STATES:
            if self.resume_state is None or self.resume_state in ESCAPE_STATES:
                raise ValueError("escape states need a resume_state outside the escape states")
        elif self.resume_state is not None:
            raise ValueError("resume_state is only meaningful in an escape state")
        return self


class ProjectManifest(MMOSModel):
    """Contents of manifests/project_manifest.json."""

    schema_version: Literal[1] = 1
    project_id: str
    name: str
    created_at: datetime
    inputs: list[InputFileRecord] = Field(default_factory=list)
