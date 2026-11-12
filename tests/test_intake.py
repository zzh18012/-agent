from pathlib import Path

import pytest

from mmos.core.intake import ingest_files
from mmos.core.project import create_project, load_manifest, sha256_file
from mmos.schemas.project import SourceRole


def make_source(directory: Path, name: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def test_data_file_is_copied_hashed_and_registered(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = make_source(tmp_path / "input", "train.csv", b"x,y\n1,2\n")
    original_bytes = source.read_bytes()

    records = ingest_files(paths, [source], SourceRole.DATA)

    assert len(records) == 1
    record = records[0]
    assert record.path == "data/raw/train.csv"
    assert record.sha256 == sha256_file(source)
    assert record.size_bytes == len(original_bytes)
    assert record.media_type == "text/csv"
    copied = paths.root / record.path
    assert copied.read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes

    manifest = load_manifest(paths)
    assert manifest.inputs == records


def test_reingesting_identical_file_is_idempotent(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = make_source(tmp_path / "input", "train.csv", b"x,y\n1,2\n")

    ingest_files(paths, [source], SourceRole.DATA)
    second = ingest_files(paths, [source], SourceRole.DATA)

    assert second == []
    assert len(load_manifest(paths).inputs) == 1


def test_name_collision_with_different_content_is_refused(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    first = make_source(tmp_path / "a", "train.csv", b"original")
    ingest_files(paths, [first], SourceRole.DATA)

    conflicting = make_source(tmp_path / "b", "train.csv", b"different")
    with pytest.raises(FileExistsError):
        ingest_files(paths, [conflicting], SourceRole.DATA)

    assert (paths.root / "data" / "raw" / "train.csv").read_bytes() == b"original"


def test_problem_role_is_stored_under_problem_dir(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = make_source(tmp_path / "input", "problem.md", b"# task")

    records = ingest_files(paths, [source], SourceRole.PROBLEM)

    assert records[0].path == "problem/problem.md"
