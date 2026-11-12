import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mmos.schemas.project import ProjectConfig, ProjectManifest

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
_MAX_SLUG_LENGTH = 40


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
    def state_history_file(self) -> Path:
        return self.manifest_dir / "state_history.jsonl"

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


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:_MAX_SLUG_LENGTH] or "project"


def new_project_id(name: str, now: datetime) -> str:
    return f"{_slugify(name)}-{now:%Y%m%d}-{secrets.token_hex(3)}"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_project_config(paths: ProjectPaths, config: ProjectConfig) -> None:
    payload = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    _atomic_write_text(paths.config_file, payload)


def load_project_config(paths: ProjectPaths) -> ProjectConfig:
    data = yaml.safe_load(paths.config_file.read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)


def save_manifest(paths: ProjectPaths, manifest: ProjectManifest) -> None:
    _atomic_write_text(paths.manifest_file, manifest.model_dump_json(indent=2) + "\n")


def load_manifest(paths: ProjectPaths) -> ProjectManifest:
    return ProjectManifest.model_validate_json(paths.manifest_file.read_text(encoding="utf-8"))


def create_project(
    projects_dir: Path, name: str, *, now: datetime | None = None
) -> tuple[ProjectPaths, ProjectConfig]:
    created_at = now or datetime.now(UTC).replace(microsecond=0)
    projects_dir.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(projects_dir / new_project_id(name, created_at))
    paths.root.mkdir()
    for subdir in PROJECT_SUBDIRS:
        (paths.root / subdir).mkdir(parents=True, exist_ok=True)

    config = ProjectConfig(project_id=paths.root.name, name=name, created_at=created_at)
    save_project_config(paths, config)
    save_manifest(
        paths,
        ProjectManifest(project_id=config.project_id, name=name, created_at=created_at),
    )
    return paths, config


def resolve_project(reference: str | Path, projects_dir: Path) -> ProjectPaths:
    """Accept either a project directory or a project id under projects_dir."""
    direct = Path(reference)
    if direct.is_dir() and (direct / "project.yaml").is_file():
        return ProjectPaths(direct.resolve())

    by_id = projects_dir / str(reference)
    if by_id.is_dir() and (by_id / "project.yaml").is_file():
        return ProjectPaths(by_id.resolve())

    raise FileNotFoundError(f"project not found: {reference}")
