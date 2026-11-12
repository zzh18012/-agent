import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mmos.schemas.experiment import ExperimentSpec
from mmos.schemas.problem import ProblemSpec
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
    def problem_spec_file(self) -> Path:
        return self.manifest_dir / "problem_spec.json"

    @property
    def data_profile_file(self) -> Path:
        return self.manifest_dir / "data_profile.json"

    @property
    def data_schema_file(self) -> Path:
        return self.manifest_dir / "data_schema.json"

    @property
    def data_quality_file(self) -> Path:
        return self.manifest_dir / "data_quality_report.json"

    @property
    def leakage_report_file(self) -> Path:
        return self.manifest_dir / "leakage_report.json"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

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


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_project_config(paths: ProjectPaths, config: ProjectConfig) -> None:
    payload = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    atomic_write_text(paths.config_file, payload)


def load_project_config(paths: ProjectPaths) -> ProjectConfig:
    data = yaml.safe_load(paths.config_file.read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)


def save_manifest(paths: ProjectPaths, manifest: ProjectManifest) -> None:
    atomic_write_text(paths.manifest_file, manifest.model_dump_json(indent=2) + "\n")


def load_manifest(paths: ProjectPaths) -> ProjectManifest:
    return ProjectManifest.model_validate_json(paths.manifest_file.read_text(encoding="utf-8"))


def save_problem_spec(paths: ProjectPaths, spec: ProblemSpec) -> None:
    atomic_write_text(paths.problem_spec_file, spec.model_dump_json(indent=2) + "\n")


def load_problem_spec(paths: ProjectPaths) -> ProblemSpec:
    return ProblemSpec.model_validate_json(paths.problem_spec_file.read_text(encoding="utf-8"))


def experiment_spec_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return paths.experiments_dir / experiment_id / "experiment_spec.json"


def save_experiment_spec(paths: ProjectPaths, spec: ExperimentSpec) -> Path:
    target = experiment_spec_file(paths, spec.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, spec.model_dump_json(indent=2) + "\n")
    return target


def load_experiment_spec(paths: ProjectPaths, experiment_id: str) -> ExperimentSpec:
    return ExperimentSpec.model_validate_json(
        experiment_spec_file(paths, experiment_id).read_text(encoding="utf-8")
    )


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
