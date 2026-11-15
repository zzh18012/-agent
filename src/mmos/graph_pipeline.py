import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mmos.agents import release_allowed
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    edge_flows_file,
    flow_audit_file,
    graph_result_file,
    graph_spec_file,
    load_flow_audit,
    load_graph_record,
    load_graph_result,
    load_graph_spec,
    load_min_cut_oracle,
    load_project_config,
    min_cut_oracle_file,
    save_graph_record,
    save_graph_spec,
    save_red_team_review,
    sha256_file,
)
from mmos.evidence import add_claim, freeze_graph
from mmos.evidence.store import load_claims
from mmos.experiments.graph_runner import build_graph_record, run_graph_experiment
from mmos.orchestration import Orchestrator
from mmos.paper.graph_builder import build_graph_paper
from mmos.paper.render import render_paper
from mmos.schemas.graph import GraphProblemSpec
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.paper import PaperAST
from mmos.schemas.review import RedTeamReview
from mmos.schemas.states import ProjectState, TaskFamily
from mmos.tracking import log_graph_run
from mmos.validation.graph import validate_graph_experiment

Reviewer = Callable[[PaperAST, str], RedTeamReview]


class GraphPipelineBlocked(RuntimeError):
    def __init__(self, experiment_id: str, state: ProjectState, reason: str) -> None:
        self.experiment_id = experiment_id
        self.state = state
        super().__init__(f"graph {experiment_id} blocked in {state.value}: {reason}")


@dataclass(frozen=True)
class GraphPipelineOptions:
    spec_file: Path


@dataclass(frozen=True)
class GraphRunSummary:
    experiment_id: str
    graph_id: str
    graph_hash: str
    max_flow_value: float
    min_cut_capacity: float
    validation: dict[str, str]
    claim_id: str | None
    red_team_overall: str
    paper_path: Path
    review_path: Path
    release_path: Path | None
    state: ProjectState


def _blocked(paths: ProjectPaths, experiment_id: str, title: str, lines: list[str]) -> Path:
    target = paths.reviews_dir / f"blocked_graph_{experiment_id}.md"
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
    spec: GraphProblemSpec,
    result,
    oracle,
    audit,
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
        "task_family": "graph",
        "experiment_id": spec.experiment_id,
        "graph_id": spec.graph_id,
        "graph_hash": spec.graph_hash,
        "spec_hash": spec.spec_hash,
        "node_count": len(spec.nodes),
        "edge_count": len(spec.edges),
        "source": spec.source,
        "sink": spec.sink,
        "algorithm": result.algorithm,
        "library": result.library,
        "library_version": result.library_version,
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id,
        "max_flow_value": result.max_flow_value,
        "min_cut_capacity": oracle.minimum_cut_capacity,
        "flow_cut_gap": abs(result.max_flow_value - oracle.minimum_cut_capacity),
        "minimum_cut_count": oracle.minimum_cut_count,
        "capacity_ok": audit.capacity_ok,
        "conservation_ok": audit.conservation_ok,
        "project_state": state.value,
        "validation": validation or {},
        "red_team": red_team,
        "release_path": release_path.relative_to(paths.root).as_posix()
        if release_path is not None
        else None,
        "evidence_path": (paths.verified_dir / "graph" / f"{spec.experiment_id}.json")
        .relative_to(paths.root)
        .as_posix(),
        "recorded_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    target = paths.experiment_dir(spec.experiment_id) / "graph_experiment_run.json"
    atomic_write_text(target, json.dumps(payload, indent=2) + "\n")
    return target


def _register_claim(paths: ProjectPaths, spec: GraphProblemSpec) -> str:
    return add_claim(
        paths,
        statement=(
            f"On the frozen directed-capacity graph {spec.graph_id} "
            f"(hash {spec.graph_hash[:16]}...), the solver's flow value and the "
            "oracle-enumerated minimum-cut capacity differ by 0.0 at stored precision on a "
            "construction where equality is forced. This is consistent with correctness but "
            "is not evidence of it; the 1e-6 tolerance was not exercised (code-path "
            "independence only; the oracle and solver share the same frozen spec)."
        ),
        evidence_ids=[
            f"ev-{spec.experiment_id}-graph-max-flow",
            f"ev-{spec.experiment_id}-graph-min-cut",
            f"ev-{spec.experiment_id}-graph-flow-cut-gap",
            f"ev-{spec.experiment_id}-graph-min-cut-partition",
        ],
        claim_id=f"claim-graph-{spec.experiment_id}",
        accept=False,
        notes="Auto-generated from the graph conformance check; human acceptance is required.",
    ).claim_id


def _finish(
    paths: ProjectPaths,
    orchestrator: Orchestrator,
    spec: GraphProblemSpec,
    result,
    oracle,
    audit,
    record,
    reviewer: Reviewer,
    validation: dict[str, str],
    claim_id: str | None,
) -> GraphRunSummary:
    ast = build_graph_paper(paths, spec.experiment_id)
    report_path = render_paper(paths, ast)
    orchestrator.advance_to(ProjectState.PAPER_DRAFTED, reason="graph paper rendered")
    orchestrator.advance_to(ProjectState.RED_TEAM_REVIEW, reason="graph red team started")
    review = reviewer(ast, spec.experiment_id)
    review_path = save_red_team_review(paths, review)

    if not release_allowed(review):
        blocker = _blocked(
            paths,
            spec.experiment_id,
            f"Graph release blocked: {spec.experiment_id}",
            _review_lines(review),
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"graph red team judged {review.judge.overall.value}; PASS required",
        )
        _run_record(
            paths,
            spec,
            result,
            oracle,
            audit,
            record,
            state=config.state,
            validation=validation,
            red_team=review.judge.overall.value,
        )
        raise GraphPipelineBlocked(
            spec.experiment_id,
            config.state,
            f"red team {review.judge.overall.value}; see {blocker}",
        )

    orchestrator.advance_to(
        ProjectState.READY_FOR_RELEASE, reason="graph red team passed"
    )
    release = {
        "project_id": paths.root.name,
        "task_family": "graph",
        "experiment_id": spec.experiment_id,
        "graph_id": spec.graph_id,
        "graph_hash": spec.graph_hash,
        "spec_hash": spec.spec_hash,
        "node_count": len(spec.nodes),
        "edge_count": len(spec.edges),
        "source": spec.source,
        "sink": spec.sink,
        "algorithm": result.algorithm,
        "library": result.library,
        "library_version": result.library_version,
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id,
        "max_flow_value": result.max_flow_value,
        "min_cut_capacity": oracle.minimum_cut_capacity,
        "flow_cut_gap": abs(result.max_flow_value - oracle.minimum_cut_capacity),
        "minimum_cut_count": oracle.minimum_cut_count,
        "minimum_cut_partition": oracle.canonical_partition,
        "capacity_ok": audit.capacity_ok,
        "conservation_ok": audit.conservation_ok,
        "max_conservation_residual": audit.max_absolute_residual,
        "validation": validation,
        "claim_id": claim_id,
        "paper": report_path.relative_to(paths.root).as_posix(),
        "evidence": (paths.verified_dir / "graph" / f"{spec.experiment_id}.json")
        .relative_to(paths.root)
        .as_posix(),
        "review": review_path.relative_to(paths.root).as_posix(),
        "judge": review.judge.overall.value,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    release_json = paths.releases_dir / f"graph_{spec.experiment_id}.json"
    release_md = paths.releases_dir / f"graph_{spec.experiment_id}.md"
    atomic_write_text(release_json, json.dumps(release, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Graph release {spec.experiment_id}\n\n"
        f"- Project: `{paths.root.name}` (graph)\n"
        f"- Graph: `{spec.graph_id}` ({len(spec.nodes)} nodes, {len(spec.edges)} edges)\n"
        f"- Source / sink: `{spec.source}` -> `{spec.sink}`\n"
        f"- Graph hash: `{spec.graph_hash}`\n"
        f"- Algorithm: `{result.algorithm}` via `{result.library}` "
        f"`{result.library_version}`\n"
        f"- Max flow: {result.max_flow_value:g}\n"
        f"- Min cut: {oracle.minimum_cut_capacity:g} "
        f"({oracle.minimum_cut_count} minimum cut(s))\n"
        f"- Flow-cut gap: {abs(result.max_flow_value - oracle.minimum_cut_capacity):g}\n"
        f"- Capacity audit: {audit.capacity_ok}; conservation audit: "
        f"{audit.conservation_ok}\n"
        f"- Validation: {validation}\n"
        f"- Red team: `{review.judge.overall.value}`\n"
        f"- Claim: `{claim_id or 'none'}`\n"
        f"- Paper: `{report_path.relative_to(paths.root).as_posix()}`\n",
    )
    config = orchestrator.advance_to(ProjectState.RELEASED, reason="graph release written")
    _run_record(
        paths,
        spec,
        result,
        oracle,
        audit,
        record,
        state=config.state,
        validation=validation,
        red_team=review.judge.overall.value,
        release_path=release_md,
    )
    return GraphRunSummary(
        experiment_id=spec.experiment_id,
        graph_id=spec.graph_id,
        graph_hash=spec.graph_hash,
        max_flow_value=result.max_flow_value,
        min_cut_capacity=oracle.minimum_cut_capacity,
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
        paths.experiments_dir.glob("*/artifacts/graph_result.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise ValueError("no graph experiment artifact found for resume")
    return candidates[-1].parents[1].name


def _resume(
    paths: ProjectPaths, options: GraphPipelineOptions, reviewer: Reviewer
) -> GraphRunSummary:
    config = load_project_config(paths)
    orchestrator = Orchestrator(paths)
    if config.state is ProjectState.REQUIRES_HUMAN_REVIEW:
        orchestrator.transition(
            ProjectState.RED_TEAM_REVIEW, reason="resume graph red team review"
        )
    experiment_id = _latest_experiment(paths)
    spec = load_graph_spec(paths, experiment_id)
    result = load_graph_result(paths, experiment_id)
    oracle = load_min_cut_oracle(paths, experiment_id)
    audit = load_flow_audit(paths, experiment_id)
    record = load_graph_record(paths, experiment_id)

    evidence_path = paths.verified_dir / "graph" / f"{experiment_id}.json"
    evidence_changed = not evidence_path.exists()
    if not evidence_changed:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        provenance = evidence["records"][0]["provenance"] if evidence["records"] else {}
        evidence_changed = (
            provenance.get("spec_sha256") != sha256_file(graph_spec_file(paths, experiment_id))
            or evidence.get("artifact_hash")
            != sha256_file(graph_result_file(paths, experiment_id))
            or provenance.get("oracle_sha256")
            != sha256_file(min_cut_oracle_file(paths, experiment_id))
            or provenance.get("audit_sha256")
            != sha256_file(flow_audit_file(paths, experiment_id))
            or provenance.get("edge_flows_sha256")
            != sha256_file(edge_flows_file(paths, experiment_id))
        )
    if evidence_changed:
        blocker = _blocked(
            paths,
            experiment_id,
            f"Graph resume rejected: evidence changed for {experiment_id}",
            ["- frozen evidence hashes do not match current experiment inputs"],
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="graph evidence changed; a new experiment is required",
        )
        _run_record(
            paths, spec, result, oracle, audit, record, state=config.state
        )
        raise GraphPipelineBlocked(
            experiment_id,
            config.state,
            f"evidence changed; rerun the graph experiment; see {blocker}",
        )

    reports = validate_graph_experiment(paths, experiment_id)
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
            f"Graph resume validation failed: {experiment_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="graph resume validation failed",
        )
        _run_record(
            paths,
            spec,
            result,
            oracle,
            audit,
            record,
            state=config.state,
            validation=validation,
        )
        raise GraphPipelineBlocked(
            experiment_id, config.state, f"validation failed; see {blocker}"
        )
    existing = [
        claim.claim_id
        for claim in load_claims(paths)
        if claim.claim_id == f"claim-graph-{experiment_id}"
    ]
    claim_id = existing[0] if existing else _register_claim(paths, spec)
    return _finish(
        paths,
        orchestrator,
        spec,
        result,
        oracle,
        audit,
        record,
        reviewer,
        validation,
        claim_id,
    )


def run_graph_pipeline(
    paths: ProjectPaths,
    options: GraphPipelineOptions,
    reviewer: Reviewer,
) -> GraphRunSummary:
    config = load_project_config(paths)
    if config.task is not TaskFamily.GRAPH:
        raise ValueError(
            f"graph pipeline requires a graph project; task is {config.task.value}"
        )
    if config.state is ProjectState.RED_TEAM_REVIEW or (
        config.state is ProjectState.REQUIRES_HUMAN_REVIEW
        and config.resume_state is ProjectState.RED_TEAM_REVIEW
    ):
        return _resume(paths, options, reviewer)
    if config.state is not ProjectState.PROBLEM_UNDERSTOOD:
        raise ValueError(
            "graph requires PROBLEM_UNDERSTOOD (run analyze-problem first); "
            f"state is {config.state.value}"
        )
    if not options.spec_file.exists():
        raise FileNotFoundError(f"graph spec not found: {options.spec_file}")
    spec = GraphProblemSpec.model_validate_json(
        options.spec_file.read_text(encoding="utf-8")
    ).with_hash()
    if paths.experiment_dir(spec.experiment_id).exists():
        raise ValueError(f"experiment already exists: {spec.experiment_id}")

    orchestrator = Orchestrator(paths)
    orchestrator.advance_to(ProjectState.PROTOTYPING, reason="graph spec planned")
    orchestrator.advance_to(
        ProjectState.PROTOTYPED,
        reason=(
            f"preflight: directed graph with {len(spec.nodes)} nodes, "
            f"{len(spec.edges)} edges, source {spec.source}, sink {spec.sink}"
        ),
    )
    save_graph_spec(paths, spec)
    result = run_graph_experiment(paths, spec)
    oracle = load_min_cut_oracle(paths, spec.experiment_id)
    record = build_graph_record(paths, spec, result, status=ExperimentStatus.SUCCEEDED)
    mlflow = log_graph_run(paths, spec, result, oracle)
    record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
    save_graph_record(paths, record)
    orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED, reason="graph experiment completed"
    )
    orchestrator.advance_to(ProjectState.VALIDATING, reason="graph validation started")
    reports = validate_graph_experiment(paths, spec.experiment_id)
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    audit = load_flow_audit(paths, spec.experiment_id)
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked(
            paths,
            spec.experiment_id,
            f"Graph validation failed: {spec.experiment_id}",
            lines,
        )
        record = record.model_copy(update={"status": ExperimentStatus.INVALID})
        save_graph_record(paths, record)
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="graph validation failed",
        )
        _run_record(
            paths,
            spec,
            result,
            oracle,
            audit,
            record,
            state=config.state,
            validation=validation,
        )
        raise GraphPipelineBlocked(
            spec.experiment_id, config.state, f"validation failed; see {blocker}"
        )

    record = record.model_copy(
        update={"status": ExperimentStatus.VALIDATED, "validation_status": validation}
    )
    save_graph_record(paths, record)
    orchestrator.advance_to(ProjectState.VERIFIED, reason="graph auditor passed")
    freeze_graph(paths, spec.experiment_id)
    orchestrator.advance_to(ProjectState.EVIDENCE_FROZEN, reason="graph evidence frozen")
    claim_id = _register_claim(paths, spec)
    return _finish(
        paths,
        orchestrator,
        spec,
        result,
        oracle,
        audit,
        record,
        reviewer,
        validation,
        claim_id,
    )


__all__ = [
    "GraphPipelineBlocked",
    "GraphPipelineOptions",
    "GraphRunSummary",
    "run_graph_pipeline",
]
