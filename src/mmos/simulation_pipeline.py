import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mmos.agents import release_allowed
from mmos.core.project import (
    ProjectPaths,
    analytic_oracle_file,
    atomic_write_text,
    load_analytic_oracle,
    load_project_config,
    load_simulation_record,
    load_simulation_result,
    load_simulation_spec,
    replications_file,
    save_red_team_review,
    save_simulation_record,
    save_simulation_spec,
    sha256_file,
    simulation_result_file,
    simulation_spec_file,
)
from mmos.evidence import add_claim, freeze_simulation
from mmos.evidence.store import load_claims
from mmos.experiments.simulation_oracle import UnstableQueueError, compute_analytic_oracle
from mmos.experiments.simulation_runner import (
    build_simulation_record,
    run_simulation,
)
from mmos.orchestration import Orchestrator
from mmos.paper.render import render_paper
from mmos.paper.simulation_builder import build_simulation_paper
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.paper import PaperAST
from mmos.schemas.review import RedTeamReview
from mmos.schemas.simulation import SimulationSpec
from mmos.schemas.states import ProjectState, TaskFamily
from mmos.tracking import log_simulation_run
from mmos.validation.simulation import validate_simulation_experiment

Reviewer = Callable[[PaperAST, str], RedTeamReview]


class SimulationPipelineBlocked(RuntimeError):
    def __init__(self, experiment_id: str, state: ProjectState, reason: str) -> None:
        self.experiment_id = experiment_id
        self.state = state
        super().__init__(f"simulation {experiment_id} blocked in {state.value}: {reason}")


@dataclass(frozen=True)
class SimulationPipelineOptions:
    spec_file: Path


@dataclass(frozen=True)
class SimulationRunSummary:
    experiment_id: str
    simulation_id: str
    spec_hash: str
    n_replications: int
    validation: dict[str, str]
    claim_id: str | None
    red_team_overall: str
    paper_path: Path
    review_path: Path
    release_path: Path | None
    state: ProjectState


def _blocked_artifact(
    paths: ProjectPaths, experiment_id: str, title: str, lines: list[str]
) -> Path:
    target = paths.reviews_dir / f"blocked_simulation_{experiment_id}.md"
    atomic_write_text(target, "\n".join([f"# {title}", "", *lines, ""]))
    return target


def _review_lines(review: RedTeamReview) -> list[str]:
    lines: list[str] = []
    for attack in review.attacks.attacks:
        verdict = next(
            verdict
            for verdict in review.judge.verdicts
            if verdict.attack_id == attack.attack_id
        )
        lines.extend(
            [
                f"## {attack.attack_id}: {attack.severity.value} / "
                f"{attack.category.value} -> {verdict.verdict.value}",
                attack.argument,
                f"Required fix: {attack.required_fix}",
                f"Judge: {verdict.rationale}",
                "",
            ]
        )
    return lines


def _run_record(
    paths: ProjectPaths,
    spec: SimulationSpec,
    result,
    record,
    *,
    state: ProjectState,
    validation: dict[str, str] | None = None,
    red_team: str | None = None,
    release_path: Path | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "project_id": paths.root.name,
        "task_family": "simulation",
        "experiment_id": spec.experiment_id,
        "simulation_id": spec.simulation_id,
        "spec_hash": spec.spec_hash,
        "git_commit": result.git_commit,
        "mlflow_run_id": record.mlflow_run_id,
        "arrival_rate": spec.arrival_rate,
        "service_rate": spec.service_rate,
        "rho": spec.rho,
        "warmup_time": spec.warmup_time,
        "measurement_time": spec.measurement_time,
        "replications": spec.replications,
        "master_seed": spec.master_seed,
        "replication_seed_hash": result.replication_seed_hash,
        "runtime": result.runtime,
        "project_state": state.value,
        "validation": validation or {},
        "red_team": red_team,
        "release_path": release_path.relative_to(paths.root).as_posix()
        if release_path is not None
        else None,
        "evidence_path": (
            paths.verified_dir / "simulation" / f"{spec.experiment_id}.json"
        )
        .relative_to(paths.root)
        .as_posix(),
        "recorded_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    target = paths.experiment_dir(spec.experiment_id) / "experiment_run.json"
    atomic_write_text(target, json.dumps(payload, indent=2) + "\n")
    return target


def _register_claim(paths: ProjectPaths, spec: SimulationSpec) -> str:
    return add_claim(
        paths,
        statement=(
            f"Under the specified stable M/M/1 benchmark configuration "
            f"(lambda={spec.arrival_rate:g}, mu={spec.service_rate:g}, "
            f"rho={spec.rho:.6f}, warmup={spec.warmup_time:g}, measurement="
            f"{spec.measurement_time:g}, replications={spec.replications}), the simulated "
            "steady-state metrics were consistent with the closed-form M/M/1 values within "
            "the reported numerical and statistical tolerances."
        ),
        evidence_ids=[
            f"ev-{spec.experiment_id}-simulation-W-mean",
            f"ev-{spec.experiment_id}-simulation-W-theory",
            f"ev-{spec.experiment_id}-simulation-L-mean",
            f"ev-{spec.experiment_id}-simulation-L-theory",
            f"ev-{spec.experiment_id}-simulation-little-law",
            f"ev-{spec.experiment_id}-simulation-seed-schedule",
        ],
        claim_id=f"claim-simulation-{spec.experiment_id}",
        accept=False,
        notes="Auto-generated from the simulation conformance check; human acceptance is required.",
    ).claim_id


def _finish(
    paths: ProjectPaths,
    orchestrator: Orchestrator,
    spec: SimulationSpec,
    result,
    record,
    reviewer: Reviewer,
    validation: dict[str, str],
    claim_id: str | None,
) -> SimulationRunSummary:
    ast = build_simulation_paper(paths, spec.experiment_id)
    report_path = render_paper(paths, ast)
    orchestrator.advance_to(
        ProjectState.PAPER_DRAFTED, reason="simulation paper rendered"
    )
    orchestrator.advance_to(
        ProjectState.RED_TEAM_REVIEW, reason="simulation red team started"
    )
    review = reviewer(ast, spec.experiment_id)
    review_path = save_red_team_review(paths, review)

    if not release_allowed(review):
        blocker = _blocked_artifact(
            paths,
            spec.experiment_id,
            f"Simulation release blocked: {spec.experiment_id}",
            _review_lines(review),
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"simulation red team judged {review.judge.overall.value}; PASS required",
        )
        _run_record(
            paths,
            spec,
            result,
            record,
            state=config.state,
            validation=validation,
            red_team=review.judge.overall.value,
        )
        raise SimulationPipelineBlocked(
            spec.experiment_id,
            config.state,
            f"red team {review.judge.overall.value}; see {blocker}",
        )

    orchestrator.advance_to(
        ProjectState.READY_FOR_RELEASE, reason="simulation red team passed"
    )
    oracle = load_analytic_oracle(paths, spec.experiment_id)
    release = {
        "project_id": paths.root.name,
        "task_family": "simulation",
        "experiment_id": spec.experiment_id,
        "simulation_id": spec.simulation_id,
        "simulation_spec_hash": spec.spec_hash,
        "git_commit": result.git_commit,
        "mlflow_run_id": record.mlflow_run_id,
        "arrival_rate": spec.arrival_rate,
        "service_rate": spec.service_rate,
        "rho": spec.rho,
        "warmup_time": spec.warmup_time,
        "measurement_time": spec.measurement_time,
        "replications": spec.replications,
        "master_seed": spec.master_seed,
        "replication_seed_hash": result.replication_seed_hash,
        "runtime": result.runtime,
        "metrics": {
            metric: {
                "mean": summary.mean,
                "ci_low": summary.ci_low,
                "ci_high": summary.ci_high,
                "analytic": result.analytic_comparison[metric].analytic_value
                if metric in result.analytic_comparison
                else None,
                "relative_error": result.analytic_comparison[metric].relative_error
                if metric in result.analytic_comparison
                else None,
            }
            for metric, summary in result.metrics.items()
        },
        "analytic_values": {
            "W": oracle.W,
            "Wq": oracle.Wq,
            "L": oracle.L,
            "Lq": oracle.Lq,
            "utilization": oracle.utilization,
            "throughput": oracle.throughput,
        },
        "little_law": {
            metric: {
                "relative_discrepancy": check.relative_discrepancy,
                "tolerance": check.tolerance,
                "passed": check.passed,
            }
            for metric, check in result.little_law.items()
        },
        "validation": validation,
        "claim_id": claim_id,
        "paper": report_path.relative_to(paths.root).as_posix(),
        "evidence": (
            paths.verified_dir / "simulation" / f"{spec.experiment_id}.json"
        )
        .relative_to(paths.root)
        .as_posix(),
        "review": review_path.relative_to(paths.root).as_posix(),
        "judge": review.judge.overall.value,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    release_json = paths.releases_dir / f"simulation_{spec.experiment_id}.json"
    release_md = paths.releases_dir / f"simulation_{spec.experiment_id}.md"
    atomic_write_text(release_json, json.dumps(release, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Simulation release {spec.experiment_id}\n\n"
        f"- Project: `{paths.root.name}` (simulation)\n"
        f"- Simulation: `{spec.simulation_id}`\n"
        f"- Spec hash: `{spec.spec_hash}`\n"
        f"- Git commit: `{result.git_commit}`\n"
        f"- MLflow run: `{record.mlflow_run_id}`\n"
        f"- lambda / mu / rho: {spec.arrival_rate:g} / {spec.service_rate:g} / "
        f"{spec.rho:.6f}\n"
        f"- Warm-up / measurement: {spec.warmup_time:g} / {spec.measurement_time:g} "
        f"({spec.simulation_time_units})\n"
        f"- Replications: {spec.replications}; master seed {spec.master_seed}; seed "
        f"schedule `{result.replication_seed_hash}`\n"
        f"- Validation: {validation}\n"
        f"- Red team: `{review.judge.overall.value}`\n"
        f"- Claim: `{claim_id or 'none'}`\n"
        f"- Paper: `{report_path.relative_to(paths.root).as_posix()}`\n",
    )
    config = orchestrator.advance_to(
        ProjectState.RELEASED, reason="simulation release written"
    )
    _run_record(
        paths,
        spec,
        result,
        record,
        state=config.state,
        validation=validation,
        red_team=review.judge.overall.value,
        release_path=release_md,
    )
    return SimulationRunSummary(
        experiment_id=spec.experiment_id,
        simulation_id=spec.simulation_id,
        spec_hash=spec.spec_hash,
        n_replications=spec.replications,
        validation=validation,
        claim_id=claim_id,
        red_team_overall=review.judge.overall.value,
        paper_path=report_path,
        review_path=review_path,
        release_path=release_md,
        state=config.state,
    )


def _latest_experiment(paths: ProjectPaths) -> str:
    candidates = sorted(
        (paths.experiments_dir).glob("*/artifacts/simulation_result.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise ValueError("no simulation experiment artifact found for resume")
    return candidates[-1].parents[1].name


def _resume(
    paths: ProjectPaths, options: SimulationPipelineOptions, reviewer: Reviewer
) -> SimulationRunSummary:
    config = load_project_config(paths)
    orchestrator = Orchestrator(paths)
    if config.state is ProjectState.REQUIRES_HUMAN_REVIEW:
        orchestrator.transition(
            ProjectState.RED_TEAM_REVIEW, reason="resume simulation red team review"
        )
    experiment_id = _latest_experiment(paths)
    spec = load_simulation_spec(paths, experiment_id)
    result = load_simulation_result(paths, experiment_id)
    record = load_simulation_record(paths, experiment_id)

    evidence_path = paths.verified_dir / "simulation" / f"{experiment_id}.json"
    evidence_changed = not evidence_path.exists()
    if not evidence_changed:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        provenance = evidence["records"][0]["provenance"] if evidence["records"] else {}
        evidence_changed = (
            evidence.get("artifact_hash") != sha256_file(
                simulation_result_file(paths, experiment_id)
            )
            or provenance.get("spec_sha256")
            != sha256_file(simulation_spec_file(paths, experiment_id))
            or provenance.get("oracle_sha256")
            != sha256_file(analytic_oracle_file(paths, experiment_id))
            or provenance.get("replications_sha256")
            != sha256_file(replications_file(paths, experiment_id))
        )
    if evidence_changed:
        blocker = _blocked_artifact(
            paths,
            experiment_id,
            f"Simulation resume rejected: evidence changed for {experiment_id}",
            ["- frozen evidence hashes do not match current experiment inputs"],
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="simulation evidence changed; a new experiment is required",
        )
        _run_record(paths, spec, result, record, state=config.state)
        raise SimulationPipelineBlocked(
            experiment_id,
            config.state,
            f"evidence changed; rerun the simulation instead of resuming review; see {blocker}",
        )

    reports = validate_simulation_experiment(paths, experiment_id)
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked_artifact(
            paths,
            experiment_id,
            f"Simulation resume validation failed: {experiment_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="simulation resume validation failed",
        )
        _run_record(
            paths,
            spec,
            result,
            record,
            state=config.state,
            validation=validation,
        )
        raise SimulationPipelineBlocked(
            experiment_id, config.state, f"validation failed; see {blocker}"
        )
    existing = [
        claim.claim_id
        for claim in load_claims(paths)
        if claim.claim_id == f"claim-simulation-{experiment_id}"
    ]
    claim_id = existing[0] if existing else _register_claim(paths, spec)
    return _finish(
        paths, orchestrator, spec, result, record, reviewer, validation, claim_id
    )


def run_simulation_pipeline(
    paths: ProjectPaths,
    options: SimulationPipelineOptions,
    reviewer: Reviewer,
) -> SimulationRunSummary:
    config = load_project_config(paths)
    if config.task is not TaskFamily.SIMULATION:
        raise ValueError(
            f"simulation pipeline requires a simulation project; task is {config.task.value}"
        )
    if config.state is ProjectState.RED_TEAM_REVIEW or (
        config.state is ProjectState.REQUIRES_HUMAN_REVIEW
        and config.resume_state is ProjectState.RED_TEAM_REVIEW
    ):
        return _resume(paths, options, reviewer)
    if config.state is not ProjectState.PROBLEM_UNDERSTOOD:
        raise ValueError(
            "simulation requires PROBLEM_UNDERSTOOD (run analyze-problem first); "
            f"state is {config.state.value}"
        )
    if not options.spec_file.exists():
        raise FileNotFoundError(f"simulation spec not found: {options.spec_file}")
    spec = SimulationSpec.model_validate_json(
        options.spec_file.read_text(encoding="utf-8")
    ).with_hash()
    if paths.experiment_dir(spec.experiment_id).exists():
        raise ValueError(f"experiment already exists: {spec.experiment_id}")
    if not spec.stable:
        raise UnstableQueueError(
            f"rho={spec.rho:.6g} >= 1; steady-state M/M/1 benchmark is undefined"
        )
    compute_analytic_oracle(spec)

    orchestrator = Orchestrator(paths)
    orchestrator.advance_to(
        ProjectState.PROTOTYPING, reason="simulation spec planned"
    )
    orchestrator.advance_to(
        ProjectState.PROTOTYPED,
        reason=(
            f"preflight: stable M/M/1 rho={spec.rho:.6f}, "
            f"{spec.replications} derived replication seeds"
        ),
    )
    save_simulation_spec(paths, spec)
    result = run_simulation(paths, spec)
    record = build_simulation_record(
        paths, spec, result, status=ExperimentStatus.SUCCEEDED
    )
    mlflow = log_simulation_run(paths, spec, result)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_simulation_record(paths, record)
    orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED, reason="simulation experiment completed"
    )
    orchestrator.advance_to(ProjectState.VALIDATING, reason="simulation validation started")
    reports = validate_simulation_experiment(paths, spec.experiment_id)
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked_artifact(
            paths,
            spec.experiment_id,
            f"Simulation validation failed: {spec.experiment_id}",
            lines,
        )
        record = record.model_copy(update={"status": ExperimentStatus.INVALID})
        save_simulation_record(paths, record)
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="simulation validation failed",
        )
        _run_record(
            paths,
            spec,
            result,
            record,
            state=config.state,
            validation=validation,
        )
        raise SimulationPipelineBlocked(
            spec.experiment_id, config.state, f"validation failed; see {blocker}"
        )

    record = record.model_copy(
        update={"status": ExperimentStatus.VALIDATED, "validation_status": validation}
    )
    save_simulation_record(paths, record)
    orchestrator.advance_to(ProjectState.VERIFIED, reason="simulation auditor passed")
    freeze_simulation(paths, spec.experiment_id)
    orchestrator.advance_to(ProjectState.EVIDENCE_FROZEN, reason="simulation evidence frozen")
    claim_id = _register_claim(paths, spec)
    return _finish(
        paths, orchestrator, spec, result, record, reviewer, validation, claim_id
    )


__all__ = [
    "SimulationPipelineBlocked",
    "SimulationPipelineOptions",
    "SimulationRunSummary",
    "run_simulation_pipeline",
]
