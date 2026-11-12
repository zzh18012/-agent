import mimetypes
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import (
    ProjectPaths,
    load_manifest,
    save_manifest,
    sha256_file,
)
from mmos.schemas.project import InputFileRecord, SourceRole

ROLE_SUBDIR: dict[SourceRole, str] = {
    SourceRole.PROBLEM: "problem",
    SourceRole.RULES: "rules",
    SourceRole.DATA: "data/raw",
    SourceRole.ATTACHMENT: "data/raw/attachments",
}


def ingest_files(
    paths: ProjectPaths,
    sources: Iterable[Path],
    role: SourceRole,
    *,
    now: datetime | None = None,
) -> list[InputFileRecord]:
    """Copy sources into the project and register them in the manifest.

    Originals are never modified. An identical file is ingested at most once per
    role; a name collision with different content is refused rather than overwritten.
    """
    ingested_at = now or datetime.now(UTC).replace(microsecond=0)
    manifest = load_manifest(paths)
    known = {(record.sha256, record.source_role) for record in manifest.inputs}
    destination_dir = paths.root / ROLE_SUBDIR[role]
    created: list[InputFileRecord] = []

    for source in sources:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"input file not found: {source}")

        digest = sha256_file(source)
        if (digest, role) in known:
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists():
            if sha256_file(destination) != digest:
                raise FileExistsError(
                    f"refusing to overwrite a different file already at {destination}"
                )
        else:
            shutil.copy2(source, destination)

        record = InputFileRecord(
            path=destination.relative_to(paths.root).as_posix(),
            filename=source.name,
            size_bytes=destination.stat().st_size,
            sha256=digest,
            media_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
            source_role=role,
            ingested_at=ingested_at,
        )
        manifest.inputs.append(record)
        known.add((digest, role))
        created.append(record)
        save_manifest(paths, manifest)

    return created
