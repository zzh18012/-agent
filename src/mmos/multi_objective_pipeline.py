import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mmos.agents import release_allowed
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    epsilon_solutions_file,
    load_multi_objective_record,
    load_multi_objective_result,
    load_multi_objective_spec,
    load_pareto_oracle,
    load_project_config,
    multi_objective_result_file,
    multi_objective_spec_file,
    pareto_oracle_file,
    save_multi_objective_record,
    save_multi_objective_spec,
    save_red_team_review,
    sha256_file,
)
from mmos.evidence import add_claim, freeze_multi_objective
from mmos.evidence.store import load_claims
from mmos.experiments.multi_objective_runner import (
    build_multi_objective_record,
    run_multi_objective,
)
from mmos.orchestration import Orchestrator
from mmos.paper.multi_objective_builder import build_multi_objective_paper
from mmos.paper.render import render_paper
from mmos.schemas.multi_objective import MultiObjectiveSpec
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.paper import PaperAST
from mmos.schemas.review import RedTeamReview
from mmos.schemas.states import ProjectState, TaskFamily
from mmos.tracking import log_multi_objective_run
from mmos.validation.multi_objective import validate_multi_objective_experiment

Reviewer = Callable[[PaperAST, str], RedTeamReview]


class MultiObjectivePipelineBlocked(RuntimeError):
    def __init__(self, experiment_id: str, state: ProjectState, reason: str) -> None:
        self.experiment_id = experiment_id
        self.state = state
        super().__init__(f"multi-objective {experiment_id} blocked in {state.value}: {reason}")


@dataclass(frozen=True)
class MultiObjectivePipelineOptions:
    spec_file: Path


@dataclass(frozen=True)
class MultiObjectiveRunSummary:
    experiment_id: str
    instance_hash: str
    spec_hash: str
    candidate_front_size: int
    oracle_front_size: int
    precision: float
    recall: float
    exact_match: bool
    validation: dict[str, str]
    claim_id: str | None
    red_team_overall: str
    paper_path: Path
    review_path: Path
    release_path: Path | None
    state: ProjectState


def _blocked(paths: ProjectPaths, experiment_id: str, title: str, lines: list[str]) -> Path:
    target = paths.reviews_dir / f"blocked_multi_objective_{experiment_id}.md"
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
    spec: MultiObjectiveSpec,
    result,
    oracle,
    record,
    *,
    state: ProjectState,
    validation: dict[str, str] | None = None,
    red_team: str | None = None,
    release_path: Path | None = None,
) -> Path:
    comparison = result.comparison
    payload = {
        "schema_version": 1,
        "project_id": paths.root.name,
        "task_family": "multi_objective",
        "experiment_id": spec.experiment_id,
        "problem_id": spec.problem_id,
        "instance_hash": spec.instance_hash,
        "spec_hash": spec.spec_hash,
        "objectives": [objective.model_dump(mode="json") for objective in spec.objectives],
        "budget": spec.budget,
        "method": result.method,
        "solver": result.solver,
        "solver_version": result.solver_version,
        "epsilon_policy": spec.epsilon_policy,
        "epsilon_coverage": result.epsilon_coverage.model_dump(mode="json"),
        "reference_point": spec.reference_point,
        "candidate_front_size": result.candidate_front_size,
        "oracle_front_size": oracle.front_size,
        "front_precision": comparison.precision if comparison else 0.0,
        "front_recall": comparison.recall if comparison else 0.0,
        "exact_match": comparison.exact_match if comparison else False,
        "hypervolume": result.hypervolume,
        "hypervolume_oracle": oracle.hypervolume,
        "hypervolume_gap": result.hypervolume_gap,
        "solver_calls": result.solver_calls,
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id,
        "project_state": state.value,
        "validation": validation or {},
        "red_team": red_team,
        "release_path": release_path.relative_to(paths.root).as_posix()
        if release_path is not None
        else None,
        "evidence_path": (
            paths.verified_dir / "multi_objective" / f"{spec.experiment_id}.json"
        )
        .relative_to(paths.root)
        .as_posix(),
        "recorded_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    target = paths.experiment_dir(spec.experiment_id) / "multi_objective_experiment_run.json"
    atomic_write_text(target, json.dumps(payload, indent=2) + "\n")
    return target


def _register_claim(paths: ProjectPaths, spec: MultiObjectiveSpec) -> str:
    return add_claim(
        paths,
        statement=(
            f"On the frozen bi-objective project-selection instance {spec.problem_id} "
            f"(instance hash {spec.instance_hash[:16]}...), at the step-1 epsilon grid "
            "covering every distinct secondary value, the epsilon-constraint procedure "
            "recovered the same Pareto objective-vector set as independent exhaustive "
            "enumeration. This is an internal implementation/grid-coverage check, not a "
            "general method-superiority claim, and decision-space recall 1.0 is not claimed."
        ),
        evidence_ids=[
            f"ev-{spec.experiment_id}-multiobjective-candidate-front",
            f"ev-{spec.experiment_id}-multiobjective-oracle-front",
            f"ev-{spec.experiment_id}-multiobjective-precision",
            f"ev-{spec.experiment_id}-multiobjective-recall",
        ],
        claim_id=f"claim-multiobjective-{spec.experiment_id}",
        accept=False,
        notes=(
            "Auto-generated from the Pareto-front conformance check; no preference or "
            "unique winner is claimed. Human acceptance is required."
        ),
    ).claim_id


def _finish(
    paths: ProjectPaths,
    orchestrator: Orchestrator,
    spec: MultiObjectiveSpec,
    result,
    oracle,
    record,
    reviewer: Reviewer,
    validation: dict[str, str],
    claim_id: str | None,
) -> MultiObjectiveRunSummary:
    ast = build_multi_objective_paper(paths, spec.experiment_id)
    report_path = render_paper(paths, ast)
    orchestrator.advance_to(
        ProjectState.PAPER_DRAFTED, reason="multi-objective paper rendered"
    )
    orchestrator.advance_to(
        ProjectState.RED_TEAM_REVIEW, reason="multi-objective red team started"
    )
    review = reviewer(ast, spec.experiment_id)
    review_path = save_red_team_review(paths, review)
    comparison = result.comparison

    if not release_allowed(review):
        blocker = _blocked(
            paths,
            spec.experiment_id,
            f"Multi-objective release blocked: {spec.experiment_id}",
            _review_lines(review),
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"multi-objective red team judged {review.judge.overall.value}; PASS required",
        )
        _run_record(
            paths,
            spec,
            result,
            oracle,
            record,
            state=config.state,
            validation=validation,
            red_team=review.judge.overall.value,
        )
        raise MultiObjectivePipelineBlocked(
            spec.experiment_id,
            config.state,
            f"red team {review.judge.overall.value}; see {blocker}",
        )

    orchestrator.advance_to(
        ProjectState.READY_FOR_RELEASE, reason="multi-objective red team passed"
    )
    release = {
        "project_id": paths.root.name,
        "task_family": "multi_objective",
        "experiment_id": spec.experiment_id,
        "problem_id": spec.problem_id,
        "instance_hash": spec.instance_hash,
        "spec_hash": spec.spec_hash,
        "objectives": [objective.model_dump(mode="json") for objective in spec.objectives],
        "budget": spec.budget,
        "method": result.method,
        "solver": result.solver,
        "solver_version": result.solver_version,
        "epsilon_policy": spec.epsilon_policy,
        "epsilon_coverage": result.epsilon_coverage.model_dump(mode="json"),
        "candidate_front_size": result.candidate_front_size,
        "oracle_front_size": oracle.front_size,
        "front_precision": comparison.precision if comparison else 0.0,
        "front_recall": comparison.recall if comparison else 0.0,
        "exact_match": comparison.exact_match if comparison else False,
        "reference_point": spec.reference_point,
        "hypervolume": result.hypervolume,
        "hypervolume_oracle": oracle.hypervolume,
        "hypervolume_gap": result.hypervolume_gap,
        "solver_calls": result.solver_calls,
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id,
        "validation": validation,
        "claim_id": claim_id,
        "paper": report_path.relative_to(paths.root).as_posix(),
        "evidence": (
            paths.verified_dir / "multi_objective" / f"{spec.experiment_id}.json"
        )
        .relative_to(paths.root)
        .as_posix(),
        "review": review_path.relative_to(paths.root).as_posix(),
        "judge": review.judge.overall.value,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    release_json = paths.releases_dir / f"multi_objective_{spec.experiment_id}.json"
    release_md = paths.releases_dir / f"multi_objective_{spec.experiment_id}.md"
    atomic_write_text(release_json, json.dumps(release, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Multi-objective release {spec.experiment_id}\n\n"
        f"- Project: `{paths.root.name}` (multi_objective)\n"
        f"- Problem: `{spec.problem_id}`\n"
        f"- Instance hash: `{spec.instance_hash}`\n"
        f"- Spec hash: `{spec.spec_hash}`\n"
        f"- Objectives: "
        f"{', '.join(objective.objective_id + ' maximize' for objective in spec.objectives)}\n"
        f"- Budget: {spec.budget:g}\n"
        f"- Method: `{result.method}`; solver `{result.solver}` "
        f"{result.solver_version}; {result.solver_calls} solver calls\n"
        f"- Epsilon coverage: {result.epsilon_coverage.epsilon_count} levels\n"
        f"- Candidate front: {result.candidate_front_size}; oracle front: "
        f"{oracle.front_size}; precision "
        f"{comparison.precision if comparison else 0.0:.6f}; recall "
        f"{comparison.recall if comparison else 0.0:.6f}; exact match "
        f"{comparison.exact_match if comparison else False}\n"
        f"- Reference point: {spec.reference_point}; hypervolume "
        f"{result.hypervolume:.6f} vs oracle {oracle.hypervolume:.6f}\n"
        f"- Validation: {validation}\n"
        f"- Red team: `{review.judge.overall.value}`\n"
        f"- Claim: `{claim_id or 'none'}`\n"
        f"- Paper: `{report_path.relative_to(paths.root).as_posix()}`\n"
        "- No unique winner is selected; the result is a Pareto set of alternatives.\n",
    )
    config = orchestrator.advance_to(
        ProjectState.RELEASED, reason="multi-objective release written"
    )
    _run_record(
        paths,
        spec,
        result,
        oracle,
        record,
        state=config.state,
        validation=validation,
        red_team=review.judge.overall.value,
        release_path=release_md,
    )
    return MultiObjectiveRunSummary(
        experiment_id=spec.experiment_id,
        instance_hash=spec.instance_hash,
        spec_hash=spec.spec_hash,
        candidate_front_size=result.candidate_front_size,
        oracle_front_size=oracle.front_size,
        precision=comparison.precision if comparison else 0.0,
        recall=comparison.recall if comparison else 0.0,
        exact_match=comparison.exact_match if comparison else False,
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
        paths.experiments_dir.glob("*/artifacts/multi_objective_result.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise ValueError("no multi-objective experiment artifact found for resume")
    return candidates[-1].parents[1].name


def _resume(
    paths: ProjectPaths,
    options: MultiObjectivePipelineOptions,
    reviewer: Reviewer,
) -> MultiObjectiveRunSummary:
    config = load_project_config(paths)
    orchestrator = Orchestrator(paths)
    if config.state is ProjectState.REQUIRES_HUMAN_REVIEW:
        orchestrator.transition(
            ProjectState.RED_TEAM_REVIEW, reason="resume multi-objective red team review"
        )
    experiment_id = _latest_experiment(paths)
    spec = load_multi_objective_spec(paths, experiment_id)
    result = load_multi_objective_result(paths, experiment_id)
    oracle = load_pareto_oracle(paths, experiment_id)
    record = load_multi_objective_record(paths, experiment_id)

    evidence_path = paths.verified_dir / "multi_objective" / f"{experiment_id}.json"
    evidence_changed = not evidence_path.exists()
    if not evidence_changed:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        provenance = evidence["records"][0]["provenance"] if evidence["records"] else {}
        evidence_changed = (
            provenance.get("spec_sha256")
            != sha256_file(multi_objective_spec_file(paths, experiment_id))
            or evidence.get("artifact_hash")
            != sha256_file(multi_objective_result_file(paths, experiment_id))
            or provenance.get("oracle_sha256")
            != sha256_file(pareto_oracle_file(paths, experiment_id))
            or provenance.get("epsilon_solutions_sha256")
            != sha256_file(epsilon_solutions_file(paths, experiment_id))
        )
    if evidence_changed:
        blocker = _blocked(
            paths,
            experiment_id,
            f"Multi-objective resume rejected: evidence changed for {experiment_id}",
            ["- frozen evidence hashes do not match current experiment inputs"],
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="multi-objective evidence changed; a new experiment is required",
        )
        _run_record(paths, spec, result, oracle, record, state=config.state)
        raise MultiObjectivePipelineBlocked(
            experiment_id,
            config.state,
            f"evidence changed; rerun the experiment; see {blocker}",
        )

    reports = validate_multi_objective_experiment(paths, experiment_id)
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked(
            paths,
            experiment_id,
            f"Multi-objective resume validation failed: {experiment_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="multi-objective resume validation failed",
        )
        _run_record(
            paths,
            spec,
            result,
            oracle,
            record,
            state=config.state,
            validation=validation,
        )
        raise MultiObjectivePipelineBlocked(
            experiment_id, config.state, f"validation failed; see {blocker}"
        )
    existing = [
        claim.claim_id
        for claim in load_claims(paths)
        if claim.claim_id == f"claim-multiobjective-{experiment_id}"
    ]
    claim_id = existing[0] if existing else _register_claim(paths, spec)
    return _finish(
        paths,
        orchestrator,
        spec,
        result,
        oracle,
        record,
        reviewer,
        validation,
        claim_id,
    )


def run_multi_objective_pipeline(
    paths: ProjectPaths,
    options: MultiObjectivePipelineOptions,
    reviewer: Reviewer,
) -> MultiObjectiveRunSummary:
    config = load_project_config(paths)
    if config.task is not TaskFamily.MULTI_OBJECTIVE:
        raise ValueError(
            "multi-objective pipeline requires a multi_objective project; "
            f"task is {config.task.value}"
        )
    if config.state is ProjectState.RED_TEAM_REVIEW or (
        config.state is ProjectState.REQUIRES_HUMAN_REVIEW
        and config.resume_state is ProjectState.RED_TEAM_REVIEW
    ):
        return _resume(paths, options, reviewer)
    if config.state is not ProjectState.PROBLEM_UNDERSTOOD:
        raise ValueError(
            "multi-objective requires PROBLEM_UNDERSTOOD (run analyze-problem first); "
            f"state is {config.state.value}"
        )
    if not options.spec_file.exists():
        raise FileNotFoundError(f"multi-objective spec not found: {options.spec_file}")
    spec = MultiObjectiveSpec.model_validate_json(
        options.spec_file.read_text(encoding="utf-8")
    ).with_hash()
    if paths.experiment_dir(spec.experiment_id).exists():
        raise ValueError(f"experiment already exists: {spec.experiment_id}")

    orchestrator = Orchestrator(paths)
    orchestrator.advance_to(ProjectState.PROTOTYPING, reason="multi-objective spec planned")
    orchestrator.advance_to(
        ProjectState.PROTOTYPED,
        reason=(
            f"preflight: {len(spec.items)} binary projects, budget {spec.budget:g}, "
            f"epsilon levels {len(spec.epsilon_levels())}"
        ),
    )
    save_multi_objective_spec(paths, spec)
    result = run_multi_objective(paths, spec)
    oracle = load_pareto_oracle(paths, spec.experiment_id)
    record = build_multi_objective_record(
        paths, spec, result, status=ExperimentStatus.SUCCEEDED
    )
    mlflow = log_multi_objective_run(paths, spec, result, oracle)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_multi_objective_record(paths, record)
    orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED, reason="multi-objective experiment completed"
    )
    orchestrator.advance_to(
        ProjectState.VALIDATING, reason="multi-objective validation started"
    )
    reports = validate_multi_objective_experiment(paths, spec.experiment_id)
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked(
            paths,
            spec.experiment_id,
            f"Multi-objective validation failed: {spec.experiment_id}",
            lines,
        )
        record = record.model_copy(update={"status": ExperimentStatus.INVALID})
        save_multi_objective_record(paths, record)
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="multi-objective validation failed",
        )
        _run_record(
            paths,
            spec,
            result,
            oracle,
            record,
            state=config.state,
            validation=validation,
        )
        raise MultiObjectivePipelineBlocked(
            spec.experiment_id, config.state, f"validation failed; see {blocker}"
        )

    record = record.model_copy(
        update={"status": ExperimentStatus.VALIDATED, "validation_status": validation}
    )
    save_multi_objective_record(paths, record)
    orchestrator.advance_to(
        ProjectState.VERIFIED, reason="multi-objective Pareto auditor passed"
    )
    freeze_multi_objective(paths, spec.experiment_id)
    orchestrator.advance_to(
        ProjectState.EVIDENCE_FROZEN, reason="multi-objective evidence frozen"
    )
    claim_id = _register_claim(paths, spec)
    return _finish(
        paths,
        orchestrator,
        spec,
        result,
        oracle,
        record,
        reviewer,
        validation,
        claim_id,
    )


__all__ = [
    "MultiObjectivePipelineBlocked",
    "MultiObjectivePipelineOptions",
    "MultiObjectiveRunSummary",
    "run_multi_objective_pipeline",
]
