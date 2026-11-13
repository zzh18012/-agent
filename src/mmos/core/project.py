import hashlib
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mmos.schemas.experiment import ExperimentSpec
from mmos.schemas.optimization import (
    OptimizationExperimentRecord,
    OptimizationResult,
    OptimizationSpec,
)
from mmos.schemas.pipeline import PipelineState, ReleaseSummary
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.project import ProjectConfig, ProjectManifest
from mmos.schemas.proposal import CouncilDecision, CritiqueReport, ModelProposal, ProposalList
from mmos.schemas.research import ResearchResult
from mmos.schemas.review import RedTeamReview
from mmos.schemas.states import TaskFamily
from mmos.schemas.tournament import TournamentResult

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
    "releases",
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
    def research_result_file(self) -> Path:
        return self.manifest_dir / "research_result.json"

    @property
    def proposals_file(self) -> Path:
        return self.manifest_dir / "model_proposals.json"

    @property
    def critiques_file(self) -> Path:
        return self.manifest_dir / "council_critiques.json"

    @property
    def council_decision_file(self) -> Path:
        return self.manifest_dir / "council_decision.json"

    @property
    def pipeline_state_file(self) -> Path:
        return self.manifest_dir / "pipeline_state.json"

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

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.experiments_dir / experiment_id

    @property
    def executions_dir(self) -> Path:
        return self.log_dir / "executions"

    @property
    def tournaments_dir(self) -> Path:
        return self.experiments_dir / "tournaments"

    @property
    def reviews_dir(self) -> Path:
        return self.root / "reviews"

    @property
    def releases_dir(self) -> Path:
        return self.root / "releases"

    @property
    def raw_data_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def verified_dir(self) -> Path:
        return self.root / "verified"

    @property
    def verified_numbers_file(self) -> Path:
        return self.verified_dir / "numbers.json"

    @property
    def verified_claims_file(self) -> Path:
        return self.verified_dir / "claims.yaml"

    @property
    def verified_metrics_dir(self) -> Path:
        return self.verified_dir / "metrics"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


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


def save_research_result(paths: ProjectPaths, result: ResearchResult) -> None:
    atomic_write_text(paths.research_result_file, result.model_dump_json(indent=2) + "\n")


def load_research_result(paths: ProjectPaths) -> ResearchResult:
    return ResearchResult.model_validate_json(
        paths.research_result_file.read_text(encoding="utf-8")
    )


def save_proposals(paths: ProjectPaths, proposals: list[ModelProposal]) -> None:
    payload = ProposalList(proposals=proposals)
    atomic_write_text(paths.proposals_file, payload.model_dump_json(indent=2) + "\n")


def load_proposals(paths: ProjectPaths) -> list[ModelProposal]:
    payload = ProposalList.model_validate_json(paths.proposals_file.read_text(encoding="utf-8"))
    return payload.proposals


def save_critiques(paths: ProjectPaths, report: CritiqueReport) -> None:
    atomic_write_text(paths.critiques_file, report.model_dump_json(indent=2) + "\n")


def load_critiques(paths: ProjectPaths) -> CritiqueReport:
    return CritiqueReport.model_validate_json(paths.critiques_file.read_text(encoding="utf-8"))


def save_council_decision(paths: ProjectPaths, decision: CouncilDecision) -> None:
    atomic_write_text(paths.council_decision_file, decision.model_dump_json(indent=2) + "\n")


def load_council_decision(paths: ProjectPaths) -> CouncilDecision:
    return CouncilDecision.model_validate_json(
        paths.council_decision_file.read_text(encoding="utf-8")
    )


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


def optimization_spec_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return paths.experiment_dir(experiment_id) / "optimization_spec.json"


def save_optimization_spec(paths: ProjectPaths, spec: OptimizationSpec) -> Path:
    target = optimization_spec_file(paths, spec.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, spec.model_dump_json(indent=2) + "\n")
    return target


def load_optimization_spec(paths: ProjectPaths, experiment_id: str) -> OptimizationSpec:
    return OptimizationSpec.model_validate_json(
        optimization_spec_file(paths, experiment_id).read_text(encoding="utf-8")
    )


def optimization_result_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return paths.experiment_dir(experiment_id) / "artifacts" / "optimization_result.json"


def save_optimization_result(paths: ProjectPaths, result: OptimizationResult) -> Path:
    target = optimization_result_file(paths, result.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, result.model_dump_json(indent=2) + "\n")
    return target


def load_optimization_result(paths: ProjectPaths, experiment_id: str) -> OptimizationResult:
    return OptimizationResult.model_validate_json(
        optimization_result_file(paths, experiment_id).read_text(encoding="utf-8")
    )


def experiment_record_file(paths: ProjectPaths, experiment_id: str) -> Path:
    return paths.experiment_dir(experiment_id) / "experiment_record.json"


def save_experiment_record(paths: ProjectPaths, record: OptimizationExperimentRecord) -> Path:
    target = experiment_record_file(paths, record.experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, record.model_dump_json(indent=2) + "\n")
    return target


def load_experiment_record(paths: ProjectPaths, experiment_id: str) -> OptimizationExperimentRecord:
    return OptimizationExperimentRecord.model_validate_json(
        experiment_record_file(paths, experiment_id).read_text(encoding="utf-8")
    )


def save_tournament_result(paths: ProjectPaths, result: TournamentResult) -> Path:
    paths.tournaments_dir.mkdir(parents=True, exist_ok=True)
    target = paths.tournaments_dir / f"{result.tournament_id}.json"
    atomic_write_text(target, result.model_dump_json(indent=2) + "\n")
    return target


def load_tournament_result(paths: ProjectPaths, tournament_id: str) -> TournamentResult:
    target = paths.tournaments_dir / f"{tournament_id}.json"
    if not target.exists():
        raise FileNotFoundError(f"tournament not found: {tournament_id}")
    return TournamentResult.model_validate_json(target.read_text(encoding="utf-8"))


def save_red_team_review(paths: ProjectPaths, review: RedTeamReview) -> Path:
    paths.reviews_dir.mkdir(parents=True, exist_ok=True)
    target = paths.reviews_dir / f"redteam_{review.experiment_id}.json"
    atomic_write_text(target, review.model_dump_json(indent=2) + "\n")
    return target


def load_red_team_review(paths: ProjectPaths, experiment_id: str) -> RedTeamReview:
    target = paths.reviews_dir / f"redteam_{experiment_id}.json"
    return RedTeamReview.model_validate_json(target.read_text(encoding="utf-8"))


def load_pipeline_state(paths: ProjectPaths) -> PipelineState | None:
    if not paths.pipeline_state_file.exists():
        return None
    return PipelineState.model_validate_json(paths.pipeline_state_file.read_text(encoding="utf-8"))


def save_pipeline_state(paths: ProjectPaths, state: PipelineState) -> None:
    atomic_write_text(paths.pipeline_state_file, state.model_dump_json(indent=2) + "\n")


def save_release_summary(paths: ProjectPaths, summary: ReleaseSummary) -> Path:
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    target = paths.releases_dir / f"{summary.release_id}.json"
    atomic_write_text(target, summary.model_dump_json(indent=2) + "\n")
    return target


def create_project(
    projects_dir: Path,
    name: str,
    *,
    now: datetime | None = None,
    task: TaskFamily = TaskFamily.PREDICTION,
) -> tuple[ProjectPaths, ProjectConfig]:
    created_at = now or datetime.now(UTC).replace(microsecond=0)
    projects_dir.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(projects_dir / new_project_id(name, created_at))
    paths.root.mkdir()
    for subdir in PROJECT_SUBDIRS:
        (paths.root / subdir).mkdir(parents=True, exist_ok=True)

    config = ProjectConfig(project_id=paths.root.name, name=name, created_at=created_at, task=task)
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
