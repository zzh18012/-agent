import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.models import Model

from mmos.agents import release_allowed
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    experiment_record_file,
    experiment_spec_file,
    load_optimization_result,
    load_optimization_spec,
    load_problem_spec,
    load_project_config,
    save_red_team_review,
    sha256_file,
)
from mmos.evidence import add_claim, freeze_mixed
from mmos.evidence.store import load_claims
from mmos.mixed.artifacts import load_envelope
from mmos.mixed.executor import run_mixed_execution
from mmos.mixed.plan import (
    build_mixed_plan,
    load_mixed_plan,
    save_mixed_plan,
)
from mmos.mixed.validation import run_record_path, validate_mixed_experiment
from mmos.orchestration import Orchestrator
from mmos.paper.mixed_builder import build_mixed_paper
from mmos.paper.render import render_paper
from mmos.routing.runner import route_project
from mmos.schemas.mixed import (
    DAGRunRecord,
    DAGStatus,
    ForecastArtifact,
    MixedExecutionPlan,
    MixedSpec,
)
from mmos.schemas.paper import PaperAST
from mmos.schemas.review import RedTeamReview
from mmos.schemas.routing import RoutingStatus
from mmos.schemas.states import ExecutionMode, ProjectState

Reviewer = Callable[[PaperAST, str], RedTeamReview]


class MixedPipelineError(RuntimeError):
    pass


class MixedPipelineBlocked(RuntimeError):
    def __init__(self, run_id: str, state: ProjectState, reason: str) -> None:
        self.run_id = run_id
        self.state = state
        super().__init__(f"mixed run {run_id} blocked in {state.value}: {reason}")


@dataclass(frozen=True)
class MixedPlanOptions:
    spec_file: Path


@dataclass(frozen=True)
class MixedPlanSummary:
    plan_id: str
    plan_hash: str
    status: str
    nodes: int
    edges: int
    state: ProjectState


@dataclass(frozen=True)
class MixedRunSummary:
    run_id: str
    plan_id: str
    plan_hash: str
    forecast_experiment_id: str
    optimization_experiment_id: str
    forecast_artifact_id: str
    forecast_artifact_hash: str
    objective: float | None
    feasible: bool
    validation: dict[str, str]
    claim_id: str | None
    red_team_overall: str
    paper_path: Path
    review_path: Path
    release_path: Path | None
    state: ProjectState


def _require_multi_task(paths: ProjectPaths) -> None:
    config = load_project_config(paths)
    if config.execution_mode is not ExecutionMode.MULTI_TASK:
        raise MixedPipelineError(
            "mixed planning requires a multi_task project; re-create the project with "
            "--mode multi_task"
        )


def _load_spec(path: Path) -> MixedSpec:
    if not path.exists():
        raise FileNotFoundError(f"mixed spec not found: {path}")
    return MixedSpec.model_validate_json(path.read_text(encoding="utf-8")).with_hash()


def plan_mixed_project(
    paths: ProjectPaths, options: MixedPlanOptions, model: Model | str
) -> MixedPlanSummary:
    _require_multi_task(paths)
    config = load_project_config(paths)
    if config.state not in {ProjectState.PROBLEM_UNDERSTOOD, ProjectState.PROTOTYPED}:
        raise MixedPipelineError(
            "mixed planning requires PROBLEM_UNDERSTOOD (run analyze-problem first); "
            f"state is {config.state.value}"
        )
    runs_dir = paths.mixed_dir / "runs"
    if runs_dir.exists() and any(runs_dir.glob("*/dag_run.json")):
        raise MixedPipelineError(
            "mixed runs already exist for this project; create a fresh project to replan"
        )
    spec = _load_spec(options.spec_file)
    outcome = route_project(paths, model)
    decision = outcome.decision
    if decision.status is not RoutingStatus.MULTI_TASK:
        raise MixedPipelineError(
            "router did not produce a MULTI_TASK decision; "
            f"status is {decision.status.value}"
        )
    problem = load_problem_spec(paths)
    plan = build_mixed_plan(
        paths,
        problem,
        decision,
        spec,
        problem_hash=outcome.problem_hash,
        problem_spec_hash=outcome.problem_spec_hash,
    )
    save_mixed_plan(paths, plan)
    orchestrator = Orchestrator(paths)
    orchestrator.advance_to(ProjectState.PROTOTYPING, reason="mixed plan drafted")
    config = orchestrator.advance_to(
        ProjectState.PROTOTYPED,
        reason=(
            f"frozen mixed plan {plan.plan_id}: "
            f"{len(plan.nodes)} nodes, {len(plan.edges)} artifact edge(s)"
        ),
    )
    return MixedPlanSummary(
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        status=decision.status.value,
        nodes=len(plan.nodes),
        edges=len(plan.edges),
        state=config.state,
    )


def _blocked_artifact(
    paths: ProjectPaths, run_id: str, title: str, lines: list[str]
) -> Path:
    target = paths.reviews_dir / f"blocked_mixed_{run_id}.md"
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


def _register_claim(paths: ProjectPaths, plan: MixedExecutionPlan, record: DAGRunRecord) -> str:
    if record.optimization_experiment_id is None:
        raise MixedPipelineError("mixed run has no optimization experiment")
    evidence_ids = [
        f"ev-mixed-{record.run_id}-artifact-id",
        f"ev-mixed-{record.run_id}-artifact-hash",
        f"ev-mixed-{record.run_id}-dependency-contract",
        f"ev-mixed-{record.run_id}-objective",
        f"ev-mixed-{record.run_id}-feasible",
    ]
    objective = (
        f"{record.final_objective:.6f}" if record.final_objective is not None else "unavailable"
    )
    return add_claim(
        paths,
        statement=(
            f"Under the frozen mixed plan {plan.plan_id} and the verified demand forecast "
            f"artifact {record.forecast_artifact_id}, the production plan reported feasible "
            f"with objective value {objective}; it is conditional on deterministic point "
            "forecasts and does not represent an optimal future production plan."
        ),
        evidence_ids=evidence_ids,
        claim_id=f"claim-mixed-{record.run_id}",
        accept=False,
        notes="Auto-generated from the mixed DAG audit; human acceptance is required.",
    ).claim_id


def _write_release(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    record: DAGRunRecord,
    artifact: ForecastArtifact,
    envelope_hash: str,
    objective: float | None,
    feasible: bool,
    validation: dict[str, str],
    claim_id: str | None,
    paper_path: Path,
    review_path: Path,
    review: RedTeamReview,
) -> Path:
    if record.optimization_experiment_id is None:
        raise MixedPipelineError("mixed run has no optimization experiment")
    optimization_spec = load_optimization_spec(paths, record.optimization_experiment_id)
    result = load_optimization_result(paths, record.optimization_experiment_id)
    payload = {
        "project_id": paths.root.name,
        "task_family": "mixed",
        "execution_mode": "multi_task",
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "spec_hash": plan.spec.spec_hash,
        "run_id": record.run_id,
        "forecast_artifact_id": record.forecast_artifact_id,
        "forecast_artifact_hash": envelope_hash,
        "forecast_experiment_id": artifact.forecast_experiment_id,
        "optimization_experiment_id": record.optimization_experiment_id,
        "instance_hash": optimization_spec.instance_hash,
        "dataset_hash": artifact.source_dataset_hash,
        "objective_value": objective,
        "feasible": feasible,
        "dependency_contract_status": record.dependency_validation.get("status", "missing"),
        "provenance_complete": record.provenance_complete,
        "solver": result.solver_backend,
        "solver_version": result.solver_version,
        "validation": validation,
        "claim_id": claim_id,
        "paper": paper_path.relative_to(paths.root).as_posix(),
        "evidence": (paths.verified_dir / "mixed" / f"{record.run_id}.json")
        .relative_to(paths.root)
        .as_posix(),
        "review": review_path.relative_to(paths.root).as_posix(),
        "judge": review.judge.overall.value,
        "point_forecast_only": True,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    release_json = paths.releases_dir / f"mixed_{record.run_id}.json"
    release_md = paths.releases_dir / f"mixed_{record.run_id}.md"
    atomic_write_text(release_json, json.dumps(payload, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Mixed release {record.run_id}\n\n"
        f"- Project: `{paths.root.name}` (multi_task)\n"
        f"- Plan: `{plan.plan_id}`\n"
        f"- Plan hash: `{plan.plan_hash}`\n"
        f"- Forecast artifact: `{record.forecast_artifact_id}` "
        f"(`{envelope_hash}`)\n"
        f"- Forecast experiment: `{artifact.forecast_experiment_id}`\n"
        f"- Optimization experiment: `{record.optimization_experiment_id}`\n"
        f"- Objective: {objective if objective is not None else 'unavailable'} "
        f"(feasible: {feasible})\n"
        f"- Dependency contract: "
        f"`{record.dependency_validation.get('status', 'missing')}`\n"
        f"- Validation: {validation}\n"
        f"- Red team: `{review.judge.overall.value}`\n"
        f"- Claim: `{claim_id or 'none'}`\n"
        f"- Paper: `{paper_path.relative_to(paths.root).as_posix()}`\n\n"
        "Point forecasts only: demand uncertainty is not propagated; this release does "
        "not claim an optimal future production plan.\n",
    )
    return release_md


def _finish(
    paths: ProjectPaths,
    orchestrator: Orchestrator,
    plan: MixedExecutionPlan,
    record: DAGRunRecord,
    artifact: ForecastArtifact,
    envelope_hash: str,
    objective: float | None,
    feasible: bool,
    reviewer: Reviewer,
    validation: dict[str, str],
    claim_id: str | None,
) -> MixedRunSummary:
    ast = build_mixed_paper(paths, plan, record.run_id)
    paper_path = render_paper(paths, ast)
    orchestrator.advance_to(ProjectState.PAPER_DRAFTED, reason="mixed paper rendered")
    orchestrator.advance_to(ProjectState.RED_TEAM_REVIEW, reason="mixed red team started")
    review = reviewer(ast, record.run_id)
    review_path = save_red_team_review(paths, review)

    if not release_allowed(review):
        blocker = _blocked_artifact(
            paths,
            record.run_id,
            f"Mixed release blocked: {record.run_id}",
            _review_lines(review),
        )
        orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"mixed red team judged {review.judge.overall.value}; PASS required",
        )
        raise MixedPipelineBlocked(
            record.run_id,
            ProjectState.REQUIRES_HUMAN_REVIEW,
            f"red team {review.judge.overall.value}; see {blocker}",
        )

    orchestrator.advance_to(ProjectState.READY_FOR_RELEASE, reason="mixed red team passed")
    release_path = _write_release(
        paths,
        plan,
        record,
        artifact,
        envelope_hash,
        objective,
        feasible,
        validation,
        claim_id,
        paper_path,
        review_path,
        review,
    )
    config = orchestrator.advance_to(ProjectState.RELEASED, reason="mixed release written")
    return MixedRunSummary(
        run_id=record.run_id,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        forecast_experiment_id=artifact.forecast_experiment_id,
        optimization_experiment_id=record.optimization_experiment_id or "",
        forecast_artifact_id=record.forecast_artifact_id or "",
        forecast_artifact_hash=envelope_hash,
        objective=objective,
        feasible=feasible,
        validation=validation,
        claim_id=claim_id,
        red_team_overall=review.judge.overall.value,
        paper_path=paper_path,
        review_path=review_path,
        release_path=release_path,
        state=config.state,
    )


def _load_run(paths: ProjectPaths, run_id: str) -> DAGRunRecord:
    return DAGRunRecord.model_validate_json(
        run_record_path(paths, run_id).read_text(encoding="utf-8")
    )


def _latest_run_id(paths: ProjectPaths) -> str:
    candidates = sorted(
        (paths.mixed_dir / "runs").glob("*/dag_run.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise ValueError("no mixed DAG run found for resume")
    return candidates[-1].parent.name


def _evidence_changed(
    paths: ProjectPaths, plan: MixedExecutionPlan, run_id: str
) -> bool:
    evidence_path = paths.verified_dir / "mixed" / f"{run_id}.json"
    if not evidence_path.exists():
        return True
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("artifact_hash") != sha256_file(run_record_path(paths, run_id)):
        return True
    records = evidence.get("records", [])
    provenance = records[0]["provenance"] if records else {}
    record = _load_run(paths, run_id)
    if record.optimization_experiment_id is None:
        return True
    envelope_path = paths.mixed_dir / "runs" / run_id / "artifacts" / "forecast_envelope.json"
    if not envelope_path.exists():
        return True
    envelope = load_envelope(envelope_path)
    return (
        provenance.get("plan_hash") != plan.plan_hash
        or provenance.get("forecast_envelope_sha256") != sha256_file(envelope_path)
        or provenance.get("forecast_payload_sha256")
        != sha256_file(paths.root / envelope.payload_path)
        or provenance.get("forecast_experiment_spec_sha256")
        != sha256_file(experiment_spec_file(paths, envelope.producer_experiment_id))
        or provenance.get("optimization_spec_sha256")
        != sha256_file(
            paths.experiment_dir(record.optimization_experiment_id) / "optimization_spec.json"
        )
        or provenance.get("optimization_result_sha256")
        != sha256_file(
            paths.experiment_dir(record.optimization_experiment_id)
            / "artifacts"
            / "optimization_result.json"
        )
        or provenance.get("optimization_record_sha256")
        != sha256_file(experiment_record_file(paths, record.optimization_experiment_id))
    )


def _resume(paths: ProjectPaths, plan: MixedExecutionPlan, reviewer: Reviewer) -> MixedRunSummary:
    config = load_project_config(paths)
    orchestrator = Orchestrator(paths)
    if config.state is ProjectState.REQUIRES_HUMAN_REVIEW:
        orchestrator.transition(
            ProjectState.RED_TEAM_REVIEW, reason="resume mixed red team review"
        )
    run_id = _latest_run_id(paths)
    record = _load_run(paths, run_id)
    if record.plan_hash != plan.plan_hash:
        raise MixedPipelineBlocked(
            run_id,
            ProjectState.RED_TEAM_REVIEW,
            "saved plan does not match the run record; a new plan is required",
        )
    if _evidence_changed(paths, plan, run_id):
        blocker = _blocked_artifact(
            paths,
            run_id,
            f"Mixed resume rejected: evidence changed for {run_id}",
            ["- frozen evidence or artifact hashes no longer match the run"],
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="mixed evidence changed; a new run is required",
        )
        raise MixedPipelineBlocked(
            run_id,
            config.state,
            f"evidence changed; rerun the mixed plan instead of resuming review; see {blocker}",
        )
    reports = validate_mixed_experiment(paths, plan, run_id)
    validation = {report.validator: report.status.value for report in reports}
    if any(report.blocking for report in reports):
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in reports
            for issue in report.issues
        ]
        blocker = _blocked_artifact(
            paths, run_id, f"Mixed resume validation failed: {run_id}", lines
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="mixed resume validation failed",
        )
        raise MixedPipelineBlocked(
            run_id, config.state, f"validation failed; see {blocker}"
        )
    envelope_path = paths.mixed_dir / "runs" / run_id / "artifacts" / "forecast_envelope.json"
    envelope = load_envelope(envelope_path)
    artifact = ForecastArtifact.model_validate_json(
        (paths.root / envelope.payload_path).read_text(encoding="utf-8")
    )
    existing = [
        claim.claim_id
        for claim in load_claims(paths)
        if claim.claim_id == f"claim-mixed-{run_id}"
    ]
    claim_id = existing[0] if existing else _register_claim(paths, plan, record)
    return _finish(
        paths,
        orchestrator,
        plan,
        record,
        artifact,
        envelope.content_hash,
        record.final_objective,
        bool(record.final_feasible),
        reviewer,
        validation,
        claim_id,
    )


def run_mixed_project(paths: ProjectPaths, reviewer: Reviewer) -> MixedRunSummary:
    _require_multi_task(paths)
    config = load_project_config(paths)
    if config.state is ProjectState.RED_TEAM_REVIEW or (
        config.state is ProjectState.REQUIRES_HUMAN_REVIEW
        and config.resume_state is ProjectState.RED_TEAM_REVIEW
    ):
        return _resume(paths, load_mixed_plan(paths), reviewer)
    if config.state is not ProjectState.PROTOTYPED:
        raise MixedPipelineError(
            "mixed execution requires PROTOTYPED (run mixed-plan first); "
            f"state is {config.state.value}"
        )
    plan = load_mixed_plan(paths)
    orchestrator = Orchestrator(paths)
    record = run_mixed_execution(paths, plan)
    if record.status is not DAGStatus.SUCCEEDED:
        lines = [f"- DAG run status: {record.status.value}", *record.notes]
        blocker = _blocked_artifact(
            paths,
            record.run_id,
            f"Mixed DAG failed: {record.run_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"mixed DAG ended {record.status.value}; human review required",
        )
        raise MixedPipelineBlocked(
            record.run_id, config.state, f"DAG ended {record.status.value}; see {blocker}"
        )
    orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED, reason="mixed DAG nodes completed"
    )
    orchestrator.advance_to(ProjectState.VALIDATING, reason="mixed validation started")
    reports = validate_mixed_experiment(paths, plan, record.run_id)
    validation = {report.validator: report.status.value for report in reports}
    if any(report.blocking for report in reports):
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in reports
            for issue in report.issues
        ]
        blocker = _blocked_artifact(
            paths,
            record.run_id,
            f"Mixed validation failed: {record.run_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="mixed validation failed",
        )
        raise MixedPipelineBlocked(
            record.run_id, config.state, f"validation failed; see {blocker}"
        )
    orchestrator.advance_to(ProjectState.VERIFIED, reason="mixed validator passed")
    freeze_mixed(paths, plan, record.run_id)
    orchestrator.advance_to(ProjectState.EVIDENCE_FROZEN, reason="mixed evidence frozen")
    envelope_path = (
        paths.mixed_dir / "runs" / record.run_id / "artifacts" / "forecast_envelope.json"
    )
    envelope = load_envelope(envelope_path)
    artifact = ForecastArtifact.model_validate_json(
        (paths.root / envelope.payload_path).read_text(encoding="utf-8")
    )
    claim_id = _register_claim(paths, plan, record)
    return _finish(
        paths,
        orchestrator,
        plan,
        record,
        artifact,
        envelope.content_hash,
        record.final_objective,
        bool(record.final_feasible),
        reviewer,
        validation,
        claim_id,
    )


__all__ = [
    "MixedPipelineBlocked",
    "MixedPipelineError",
    "MixedPlanOptions",
    "MixedPlanSummary",
    "MixedRunSummary",
    "plan_mixed_project",
    "run_mixed_project",
]
