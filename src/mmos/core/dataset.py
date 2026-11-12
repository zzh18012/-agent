import hashlib
import json

from mmos.schemas.project import ProjectManifest, SourceRole


def dataset_fingerprint(manifest: ProjectManifest) -> str:
    """Hash of the registered data inputs, independent of ingestion order."""
    records = sorted(
        (
            {"path": record.path, "sha256": record.sha256}
            for record in manifest.inputs
            if record.source_role is SourceRole.DATA
        ),
        key=lambda record: record["path"],
    )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
