from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mmos.schemas import InputFileRecord, ProjectConfig, ProjectManifest, ProjectState, SourceRole

NOW = datetime(2026, 11, 12, 11, 0, tzinfo=UTC)
SHA = "a" * 64


def make_input() -> InputFileRecord:
    return InputFileRecord(
        path="data/raw/problem.pdf",
        filename="problem.pdf",
        size_bytes=1024,
        sha256=SHA,
        media_type="application/pdf",
        source_role=SourceRole.PROBLEM,
        ingested_at=NOW,
    )


def test_manifest_roundtrip() -> None:
    manifest = ProjectManifest(project_id="p-1", name="demo", created_at=NOW, inputs=[make_input()])
    restored = ProjectManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_sha256_must_be_64_lowercase_hex() -> None:
    with pytest.raises(ValidationError):
        InputFileRecord(
            path="x",
            filename="x",
            size_bytes=1,
            sha256="not-a-hash",
            media_type="text/plain",
            source_role=SourceRole.DATA,
            ingested_at=NOW,
        )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(
            project_id="p-1",
            name="demo",
            created_at=NOW,
            state=ProjectState.INIT,
            unexpected=True,
        )


def test_project_config_defaults_to_init_state() -> None:
    config = ProjectConfig(project_id="p-1", name="demo", created_at=NOW)
    assert config.state is ProjectState.INIT
    assert config.schema_version == 1


def test_escape_state_requires_resume_state() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(
            project_id="p-1",
            name="demo",
            created_at=NOW,
            state=ProjectState.REQUIRES_HUMAN_REVIEW,
        )


def test_working_state_may_not_carry_resume_state() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(
            project_id="p-1",
            name="demo",
            created_at=NOW,
            resume_state=ProjectState.INIT,
        )
