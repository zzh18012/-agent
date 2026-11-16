import hashlib
import json
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

from mmos.agents import release_allowed
from mmos.competition.agents import CompetitionAgents
from mmos.competition.context import (
    EvidenceContext,
    load_context,
    resolve_evidence_ids,
)
from mmos.competition.number_audit import NumberAuditError, audit_texts
from mmos.competition.requests import (
    CompetitionRequestError,
    build_manifest,
)
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_project_config,
    save_red_team_review,
)
from mmos.evidence.store import load_evidence_index
from mmos.paper.competition_builder import (
    CompetitionBuildError,
    build_competition_ast,
)
from mmos.paper.competition_figures import CompetitionFigureError
from mmos.paper.render import render_paper
from mmos.schemas.competition import (
    CompetitionPaperPlan,
    CompetitionPaperRun,
    MathSeverity,
)
from mmos.schemas.paper import PaperAST
from mmos.schemas.review import RedTeamReview
from mmos.schemas.states import ProjectState

Reviewer = Callable[[PaperAST, str], RedTeamReview]


class CompetitionPaperError(RuntimeError):
    pass


class CompetitionPaperBlocked(RuntimeError):
    def __init__(
        self,
        run_id: str,
        stage: str,
        reason: str,
        artifacts: Path,
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.reason = reason
        self.artifacts = artifacts
        super().__init__(
            f"competition paper {run_id} blocked at {stage}: {reason}; "
            f"artifacts: {artifacts}"
        )


@dataclass(frozen=True)
class CompetitionPaperSummary:
    run: CompetitionPaperRun
    request_count: int
    unsupported_request_count: int


def _timestamp() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _hash_model(model) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(target: Path, model) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, model.model_dump_json(indent=2) + "\n")
    return target


def _archive_previous(comp_dir: Path) -> None:
    run_path = comp_dir / "run.json"
    if not run_path.exists():
        return
    previous = json.loads(run_path.read_text(encoding="utf-8"))
    old_id = str(previous.get("run_id", "previous"))
    history = comp_dir / "history" / old_id
    history.mkdir(parents=True, exist_ok=True)
    for child in comp_dir.iterdir():
        if child.name == "history":
            continue
        if child.is_file():
            shutil.copy2(child, history / child.name)
        elif child.is_dir() and child.name == "figures":
            shutil.copytree(child, history / "figures", dirs_exist_ok=True)


def _augment_plan(
    plan: CompetitionPaperPlan, context: EvidenceContext
) -> CompetitionPaperPlan:
    if not context.model_facts:
        return plan
    assumptions = [
        "Rolling-origin residuals are out-of-sample for their own fold models but not "
        "out-of-sample for the deployed full-history forecast.",
        "Independent per-month residual draws do not reproduce horizon-dependent "
        "variance or within-fold correlations of the drift model.",
    ]
    limitations = [
        "The simulation has no residual pool held out from the deployed forecast.",
        "The iid per-month resampling ignores within-fold and horizon correlation.",
        "No comparator production plan was executed.",
    ]
    return plan.model_copy(
        update={
            "assumptions": list(dict.fromkeys([*assumptions, *plan.assumptions])),
            "limitations": list(dict.fromkeys([*limitations, *plan.limitations])),
        }
    )


def _resolve_plan(
    plan: CompetitionPaperPlan, context: EvidenceContext
) -> CompetitionPaperPlan:
    sections = [
        section.model_copy(
            update={
                "evidence_ids": resolve_evidence_ids(
                    context, section.evidence_ids
                ),
                "claim_points": [
                    point.model_copy(
                        update={
                            "evidence_ids": resolve_evidence_ids(
                                context, point.evidence_ids
                            )
                        }
                    )
                    for point in section.claim_points
                ],
            }
        )
        for section in plan.sections
    ]
    tables = [
        table.model_copy(
            update={
                "evidence_ids": resolve_evidence_ids(
                    context, table.evidence_ids
                )
            }
        )
        for table in plan.tables
    ]
    needs = [
        need.model_copy(
            update={
                "supporting_evidence_ids": resolve_evidence_ids(
                    context, need.supporting_evidence_ids
                )
            }
        )
        for need in plan.capability_needs
    ]
    abstract_points = [
        point.model_copy(
            update={
                "evidence_ids": resolve_evidence_ids(
                    context, point.evidence_ids
                )
            }
        )
        for point in plan.abstract_points
    ]
    return plan.model_copy(
        update={
            "sections": sections,
            "tables": tables,
            "capability_needs": needs,
            "abstract_points": abstract_points,
        }
    )


def _resolve_analysis(analysis, context: EvidenceContext):
    findings = [
        finding.model_copy(
            update={
                "evidence_ids": resolve_evidence_ids(
                    context, finding.evidence_ids
                )
            }
        )
        for finding in analysis.findings
    ]
    checks = [
        need.model_copy(
            update={
                "supporting_evidence_ids": resolve_evidence_ids(
                    context, need.supporting_evidence_ids
                )
            }
        )
        for need in analysis.requested_checks
    ]
    return analysis.model_copy(
        update={"findings": findings, "requested_checks": checks}
    )


def _resolve_figures(figures, context: EvidenceContext):
    return [
        figure.model_copy(
            update={
                "evidence_ids": resolve_evidence_ids(
                    context, figure.evidence_ids
                )
            }
        )
        for figure in figures
    ]


def _blocked_markdown(
    run_id: str,
    stage: str,
    reason: str,
    details: list[str],
) -> str:
    lines = [
        f"# Competition paper blocked: {run_id}",
        "",
        f"- Stage: `{stage}`",
        f"- Reason: {reason}",
        "",
        "## Details",
        "",
        *[f"- {line}" for line in details],
        "",
    ]
    return "\n".join(lines)


def _blocked(
    paths: ProjectPaths,
    comp_dir: Path,
    run_id: str,
    stage: str,
    reason: str,
    details: list[str],
) -> NoReturn:
    target = paths.reviews_dir / f"blocked_competition_{run_id}.md"
    atomic_write_text(target, _blocked_markdown(run_id, stage, reason, details))
    raise CompetitionPaperBlocked(run_id, stage, reason, comp_dir)


def _make_run(
    *,
    run_id: str,
    context: EvidenceContext,
    plan: CompetitionPaperPlan | None,
    math_audit_hash: str,
    analysis_hash: str,
    number_audit_hash: str,
    request_hash: str,
    status: str,
    review: RedTeamReview | None,
    paper_path: Path,
) -> CompetitionPaperRun:
    return CompetitionPaperRun(
        run_id=run_id,
        source_release=context.release_path.relative_to(
            context.paths.root
        ).as_posix(),
        source_release_hash=context.release_hash,
        context_hash=context.context_hash,
        plan_hash=_hash_model(plan) if plan is not None else "",
        math_audit_hash=math_audit_hash,
        experiment_analysis_hash=analysis_hash,
        number_audit_hash=number_audit_hash,
        request_manifest_hash=request_hash,
        status="released" if status == "released" else "blocked",
        red_team_overall=review.judge.overall.value if review is not None else None,
        paper_path=paper_path.relative_to(context.paths.root).as_posix(),
        review_path=None,
        release_path=None,
        created_at=_timestamp(),
    )


def run_competition_paper(
    paths: ProjectPaths,
    agents: CompetitionAgents,
    reviewer: Reviewer,
) -> CompetitionPaperSummary:
    config = load_project_config(paths)
    if config.state is not ProjectState.RELEASED:
        raise CompetitionPaperError(
            "competition paper requires a RELEASED project; "
            f"current state is {config.state.value}"
        )
    context = load_context(paths)
    records = {
        **load_evidence_index(paths).records,
        **{record.evidence_id: record for record in context.derived_records},
    }
    comp_dir = paths.root / "paper" / "competition"
    comp_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous(comp_dir)
    run_id = f"comp-{context.context_hash[:8]}-{secrets.token_hex(2)}"
    paper_path = comp_dir / "report.md"

    try:
        plan = _augment_plan(agents.architect.plan(context), context)
        math_audit = agents.math_auditor.audit(context, plan)
        critical = [
            issue
            for issue in math_audit.issues
            if issue.severity is MathSeverity.CRITICAL
        ]
        revise = getattr(agents.architect, "revise", None)
        if critical and callable(revise):
            plan = _augment_plan(
                cast(CompetitionPaperPlan, revise(context, plan, math_audit)),
                context,
            )
            math_audit = agents.math_auditor.audit(context, plan)
        plan = _resolve_plan(plan, context)
        analysis = _resolve_analysis(
            agents.experiment_analyst.analyze(context, plan), context
        )
        figure_plans = _resolve_figures(
            agents.figure_designer.design(context, plan, analysis), context
        )
        needs = [*plan.capability_needs, *analysis.requested_checks]
        manifest = build_manifest(needs, context.evidence_ids())
    except (CompetitionRequestError, NumberAuditError) as exc:
        _blocked(paths, comp_dir, run_id, "planning", str(exc), [str(exc)])

    _write_json(comp_dir / "plan.json", plan)
    _write_json(comp_dir / "math_audit.json", math_audit)
    atomic_write_text(
        comp_dir / "derived_evidence.json",
        json.dumps(
            [record.model_dump(mode="json") for record in context.derived_records],
            indent=2,
        )
        + "\n",
    )
    _write_json(comp_dir / "experiment_analysis.json", analysis)
    _write_json(comp_dir / "experiment_requests.json", manifest)
    math_hash = _hash_model(math_audit)
    analysis_hash = _hash_model(analysis)
    request_hash = _hash_model(manifest)

    critical = [
        issue
        for issue in math_audit.issues
        if issue.severity is MathSeverity.CRITICAL
    ]
    if critical:
        details = [
            f"{issue.issue_id}: {issue.finding} Required fix: {issue.required_fix}"
            for issue in critical
        ]
        _blocked(
            paths,
            comp_dir,
            run_id,
            "math_audit",
            "critical mathematical audit issues",
            details,
        )

    try:
        result = build_competition_ast(
            paths,
            context,
            plan,
            analysis,
            figure_plans,
            manifest,
            records,
        )
    except (CompetitionBuildError, CompetitionFigureError) as exc:
        _blocked(paths, comp_dir, run_id, "paper_build", str(exc), [str(exc)])

    number_audit = audit_texts(
        result.fragments, records, context.allowed_metadata_numbers
    )
    _write_json(comp_dir / "number_audit.json", number_audit)
    number_hash = _hash_model(number_audit)
    if number_audit.blocking:
        details = [
            f"{finding.location}: token {finding.token!r}: {finding.reason}"
            for finding in number_audit.unverified
        ]
        _blocked(
            paths,
            comp_dir,
            run_id,
            "number_audit",
            "unverified numeric tokens in paper text",
            details,
        )

    paper_written = render_paper(paths, result.ast, target=paper_path)
    review = reviewer(result.ast, run_id)
    review_path = save_red_team_review(paths, review)
    run = _make_run(
        run_id=run_id,
        context=context,
        plan=plan,
        math_audit_hash=math_hash,
        analysis_hash=analysis_hash,
        number_audit_hash=number_hash,
        request_hash=request_hash,
        status="released" if release_allowed(review) else "blocked",
        review=review,
        paper_path=paper_written,
    )
    run = run.model_copy(
        update={"review_path": review_path.relative_to(paths.root).as_posix()}
    )

    if not release_allowed(review):
        _write_json(comp_dir / "run.json", run)
        details = [
            f"{attack.attack_id}: {attack.severity.value}/{attack.category.value}; "
            f"{attack.argument}"
            for attack in review.attacks.attacks
        ]
        _blocked(
            paths,
            comp_dir,
            run_id,
            "red_team",
            f"red team judged {review.judge.overall.value}; PASS required",
            details,
        )

    release_json = paths.releases_dir / f"competition_paper_{run_id}.json"
    release_md = paths.releases_dir / f"competition_paper_{run_id}.md"
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    release_relative = release_md.relative_to(paths.root).as_posix()
    payload = {
        **run.model_dump(mode="json"),
        "release_path": release_relative,
        "judge": review.judge.overall.value,
        "request_count": len(manifest.requests),
        "unsupported_request_count": manifest.unsupported_count,
        "paper_sha256": hashlib.sha256(paper_written.read_bytes()).hexdigest(),
        "evidence_records": len(context.entries),
        "catalog_truncated": context.catalog_truncated,
    }
    atomic_write_text(release_json, json.dumps(payload, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Competition paper release {run_id}\n\n"
        f"- Source release: `{run.source_release}`\n"
        f"- Context hash: `{run.context_hash}`\n"
        f"- Paper: `{run.paper_path}`\n"
        f"- Paper sha256: `{payload['paper_sha256']}`\n"
        f"- Requests proposed: {len(manifest.requests)} "
        f"({manifest.unsupported_count} unsupported)\n"
        f"- Math audit issues: {len(math_audit.issues)} "
        f"({len(critical)} critical)\n"
        f"- Number audit: {len(number_audit.unverified)} unverified tokens\n"
        f"- Red team: `{review.judge.overall.value}`\n\n"
        "No requested experiment was executed. Only verified evidence was cited; "
        "unsupported research gaps remain proposed.\n",
    )
    run = run.model_copy(update={"release_path": release_relative})
    _write_json(comp_dir / "run.json", run)
    return CompetitionPaperSummary(
        run=run,
        request_count=len(manifest.requests),
        unsupported_request_count=manifest.unsupported_count,
    )


__all__ = [
    "CompetitionPaperBlocked",
    "CompetitionPaperError",
    "CompetitionPaperSummary",
    "Reviewer",
    "run_competition_paper",
]
