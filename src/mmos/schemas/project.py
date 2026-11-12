from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel
from mmos.schemas.states import ProjectState


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


class ProjectManifest(MMOSModel):
    """Contents of manifests/project_manifest.json."""

    schema_version: Literal[1] = 1
    project_id: str
    name: str
    created_at: datetime
    inputs: list[InputFileRecord] = Field(default_factory=list)
