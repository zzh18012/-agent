from pathlib import Path

import pytest

from mmos.core.intake import ingest_files
from mmos.core.project import create_project, load_manifest
from mmos.data import DataAgent
from mmos.schemas.data import DataSchema, DatasetProfile
from mmos.schemas.project import SourceRole


def make_project_with_csv(tmp_path: Path, content: str):
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = tmp_path / "data.csv"
    source.write_text(content, encoding="utf-8")
    ingest_files(paths, [source], SourceRole.DATA)
    return paths


def test_analyze_profiles_data_and_writes_artifacts(tmp_path: Path) -> None:
    paths = make_project_with_csv(tmp_path, "id,y,x\n1,10,1\n2,12,2\n3,14,3\n")

    result = DataAgent(paths).analyze(target="y")

    assert len(result.profile.tables) == 1
    assert result.profile.tables[0].row_count == 3
    assert result.quality.status.value == "passed"
    assert result.leakage.blocking is False

    for artifact in (
        paths.data_profile_file,
        paths.data_schema_file,
        paths.data_quality_file,
        paths.leakage_report_file,
    ):
        assert artifact.exists()

    DatasetProfile.model_validate_json(paths.data_profile_file.read_text(encoding="utf-8"))
    DataSchema.model_validate_json(paths.data_schema_file.read_text(encoding="utf-8"))


def test_missing_target_fails_the_gate(tmp_path: Path) -> None:
    paths = make_project_with_csv(tmp_path, "id,y\n1,10\n2,12\n")

    result = DataAgent(paths).analyze(target="not_a_column")

    assert result.quality.blocking is True
    assert result.quality.status.value == "failed"


def test_analysis_without_data_raises(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "empty")
    with pytest.raises(ValueError, match="no data files"):
        DataAgent(paths).analyze()


def test_unsupported_data_type_raises(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = tmp_path / "data.json"
    source.write_text("{}", encoding="utf-8")
    ingest_files(paths, [source], SourceRole.DATA)

    with pytest.raises(ValueError, match="CSV only"):
        DataAgent(paths).analyze()


def test_manifest_is_not_modified_by_analysis(tmp_path: Path) -> None:
    paths = make_project_with_csv(tmp_path, "x,y\n1,2\n3,4\n")
    before = load_manifest(paths)

    DataAgent(paths).analyze(target="y")

    assert load_manifest(paths) == before
