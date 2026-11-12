from datetime import UTC, datetime

from mmos.core.dataset import dataset_fingerprint
from mmos.schemas.project import InputFileRecord, ProjectManifest, SourceRole

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def record(path: str, digest: str, role: SourceRole = SourceRole.DATA) -> InputFileRecord:
    return InputFileRecord(
        path=path,
        filename=path.split("/")[-1],
        size_bytes=1,
        sha256=digest,
        media_type="text/csv",
        source_role=role,
        ingested_at=NOW,
    )


def manifest(*records: InputFileRecord) -> ProjectManifest:
    return ProjectManifest(project_id="p1", name="demo", created_at=NOW, inputs=list(records))


def test_fingerprint_is_independent_of_input_order() -> None:
    first = manifest(record("data/raw/a.csv", "a" * 64), record("data/raw/b.csv", "b" * 64))
    second = manifest(record("data/raw/b.csv", "b" * 64), record("data/raw/a.csv", "a" * 64))
    assert dataset_fingerprint(first) == dataset_fingerprint(second)


def test_fingerprint_changes_when_data_changes() -> None:
    before = manifest(record("data/raw/a.csv", "a" * 64))
    after = manifest(record("data/raw/a.csv", "b" * 64))
    assert dataset_fingerprint(before) != dataset_fingerprint(after)


def test_fingerprint_ignores_non_data_roles() -> None:
    data_only = manifest(record("data/raw/a.csv", "a" * 64))
    with_problem = manifest(
        record("data/raw/a.csv", "a" * 64),
        record("problem/problem.md", "c" * 64, SourceRole.PROBLEM),
    )
    assert dataset_fingerprint(data_only) == dataset_fingerprint(with_problem)
