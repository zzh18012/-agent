import hashlib
from dataclasses import dataclass
from pathlib import Path

PROJECT_SUBDIRS: tuple[str, ...] = (
    "problem",
    "rules",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/features",
    "experiments",
    "artifacts",
    "verified/metrics",
    "verified/tables",
    "verified/figures",
    "verified/equations",
    "reviews",
    "paper",
    "logs",
    "manifests",
)

_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved locations inside one project directory."""

    root: Path

    @property
    def config_file(self) -> Path:
        return self.root / "project.yaml"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def manifest_file(self) -> Path:
        return self.manifest_dir / "project_manifest.json"

    @property
    def raw_data_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def verified_dir(self) -> Path:
        return self.root / "verified"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
