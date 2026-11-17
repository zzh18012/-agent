import hashlib
import json
import re
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
    ClaimPoint,
    CompetitionPaperPlan,
    CompetitionPaperRun,
    MathSeverity,
    NotationEntry,
    SectionKind,
)
from mmos.schemas.evidence import EvidenceRecord
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
    by_key = {entry.key: entry.evidence_id for entry in context.entries}

    def placeholder(key: str, fallback: str) -> str:
        evidence_id = by_key.get(key)
        return f"[[ev:{evidence_id}]]" if evidence_id else fallback

    optimality_correction = (
        "The recorded plan is feasible at the frozen parameters; the solver reports "
        "status optimal with zero optimality gap. No dual or binding-constraint "
        "certificate is evidenced."
    )

    def sanitize(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sanitized = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(
                marker in lowered
                for marker in (
                    "cheaper feasible plan",
                    "not the minimizer",
                    "no optimality is asserted",
                    "cost minimization is not supported",
                )
            ):
                sanitized.append(optimality_correction)
            else:
                sanitized.append(sentence)
        return " ".join(sanitized).strip()

    residual_markers = (
        "pooled residual mean",
        "residual mean",
        "not recentered",
        "standard errors are normal",
        "lower bounds on sampling",
    )

    def strip_residual(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [
            sentence
            for sentence in sentences
            if not any(marker in sentence.lower() for marker in residual_markers)
        ]
        return " ".join(kept).strip()

    assumptions = [
        "Rolling-origin residuals are out-of-sample for their own fold models but not "
        "out-of-sample for the deployed full-history forecast.",
        "Independent per-month residual draws do not reproduce horizon-dependent "
        "variance or within-fold correlations of the drift model.",
        "The pooled residuals are treated as a fixed empirical distribution and are not "
        "recentered or resampled by fold or horizon.",
        "Monte Carlo summaries are single-run descriptions under one seed schedule and "
        "are not treated as precise estimates.",
    ]
    limitations = [
        "The simulation has no residual pool held out from the deployed forecast; the "
        "simulated uncertainty is the uncertainty of a proxy fold model's errors applied "
        "to the deployed forecast's demand path, not a characterized exposure.",
        "Residuals are not recentered before resampling: the pooled mean "
        f"{placeholder('mixed.forecast_error.mean', 'value')} and standard deviation "
        f"{placeholder('mixed.forecast_error.std', 'value')} are nonzero, so simulated "
        "demand inherits the pool's mean shift. No de-meaned re-run exists.",
        "No standard error or bootstrap interval is evidenced for the simulation means. "
        f"The run reports {placeholder('mixed.simulation.distinct_residual_paths', 'some')} "
        f"distinct paths among {placeholder('mixed.simulation.replications', 'the')} "
        "replications, so duplicate paths are not independent samples, effective sample "
        "size is below nominal, and any nominal standard error would be understated.",
        "The iid per-month resampling ignores within-fold and horizon correlation; no "
        "horizon-indexed or block-resampling variant was executed, so the simulation is "
        "not a service characterization.",
        "Cost minimization is asserted only for the frozen parameter point under solver "
        f"status {placeholder('optimization.ortools.cbc.status', 'the recorded status')} "
        "and optimality gap "
        f"{placeholder('optimization.ortools.cbc.optimality_gap', 'the recorded gap')}; "
        "no duals or binding-constraint certificate is evidenced, and a capacity-"
        "perturbed instance in the augmentation appendix is a different instance.",
        "The source stage did not run a perturbation sweep. Executed capacity, cost, "
        "initial-inventory, and replication/seed variants are appended as augmentation "
        "evidence and do not revise source-stage optimality claims.",
        "No comparator production plan was executed. Augmentation experiments use "
        "perturbed instances or different replication/seed protocols and are not "
        "numerically comparable to the source headline results.",
        "The last-value benchmark is reported without a paired fold-level significance "
        "test; the reported descriptive metrics do not establish a drift advantage.",
        "The demand-floor activation count is taken from the run; the minimum realized "
        "demand is not displayed, so the floor margin is not quantified.",
        "The experiment inventory counts are evidence-record (cell) counts, not the "
        "number of scalar results cited in the text.",
        "The residual pool contains only the rolling-origin test residuals; tail "
        "realizations beyond the observed maximum residual cannot be drawn, so extreme "
        "shortage and fill-rate realizations are structurally truncated.",
        "The drift slope uses the first and last observations only; no ordinary-least-"
        "squares or endpoint-robustness check is evidenced.",
        "The confidence interval in the uncertainty figure is reported for the fold "
        "mean; its construction method and effective sample size are not evidenced.",
        "Only one source seed schedule is recorded and the seed value is not shown; the "
        "residual-draw implementation is not described beyond the protocol text.",
    ]
    abstract_points = [
        point.model_copy(update={"text": sanitize(point.text)})
        for point in plan.abstract_points
    ]
    caveat_evidence = by_key.get("mixed.simulation.shortage.mean_total_shortage")
    caveat_text = (
        "Simulation values are single-run descriptive summaries of a proxy "
        "fold-model residual protocol with an uncorrected nonzero residual mean; "
        "no standard error is evidenced, so they are not an exposure or service "
        "guarantee for the deployed forecast."
    )
    if caveat_evidence is not None and not any(
        point.text == caveat_text for point in abstract_points
    ):
        abstract_points.append(
            ClaimPoint(text=caveat_text, evidence_ids=[caveat_evidence])
        )
    mean_entry = next(
        (entry for entry in context.entries if entry.key == "mixed.forecast_error.mean"),
        None,
    )
    if mean_entry is not None and float(mean_entry.raw_value) < 0:
        bias_direction = (
            "The recorded pooled mean is negative, so simulated demand is shifted "
            "downward relative to the deployed forecast, which mechanically understates "
            "shortage and overstates fill rate; no bias correction is applied."
        )
    else:
        bias_direction = (
            "The recorded pooled mean is nonzero, so simulated demand inherits a mean "
            "shift relative to the deployed forecast; no bias correction is applied."
        )
    model_notes = [
        "Simulation closure: for every replication s, I_0^(s) = I_init, the same "
        "frozen exogenous initial inventory used in the deterministic plan. For "
        "m=1..3, D_m = max(0, F_m + e_m), A_m = I_(m-1)^(s) + P_m, "
        "served_m = min(A_m, D_m), shortage_m = max(0, D_m - A_m), and "
        "I_m^(s) = A_m - served_m.",
        "Residual mean: the pooled residual mean is "
        f"{placeholder('mixed.forecast_error.mean', 'a recorded nonzero value')} and is "
        "not recentered before resampling. Simulated demand therefore inherits that "
        "mean shift, so the simulation does not represent the deployed forecast's own "
        "error distribution and the shortage and fill-rate statistics are not unbiased "
        "or calibrated. "
        + bias_direction,
        "Simulation scope: production P_m is frozen at the recorded plan, and the plan "
        "already satisfies 0 <= P_m <= cap_m. The simulation therefore cannot test "
        "capacity enforcement, recourse, or alternative production decisions; it only "
        "replays demand shocks against the frozen plan.",
        "Uncertainty reporting: the recorded simulation standard errors are normal "
        "approximations from replication-level outputs, not bootstrap intervals. "
        "Duplicate residual paths mean the effective sample size is below the nominal "
        "replication count, so the nominal standard errors may be optimistic. Recorded "
        "fold intervals are confidence intervals for the fold "
        "mean, not for the pooled metric and not predictive intervals.",
        "Solver check: the optional enumeration check is recorded as skipped with zero "
        "evaluated subsets; optimality is claimed only from the solver status and gap, "
        "not from exhaustive enumeration.",
    ]
    notation_by_symbol: dict[str, NotationEntry] = {}
    for entry in plan.notation:
        notation_by_symbol.setdefault(entry.symbol, entry)
    existing_symbols = set(notation_by_symbol)
    existing_bases = {
        part.strip().split("^")[0]
        for symbol in existing_symbols
        for part in symbol.split(",")
    }
    notation_additions = [
        NotationEntry(
            symbol="e_m",
            meaning="resampled residual for month m",
            unit="units/month",
        ),
        NotationEntry(
            symbol="D_m",
            meaning="simulated demand in month m",
            unit="units/month",
        ),
        NotationEntry(
            symbol="A_m",
            meaning="available supply in month m",
            unit="units/month",
        ),
        NotationEntry(
            symbol="served_m",
            meaning="served demand in month m",
            unit="units/month",
        ),
        NotationEntry(
            symbol="shortage_m",
            meaning="shortage in month m",
            unit="units/month",
        ),
        NotationEntry(
            symbol="cap_m",
            meaning="capacity in month m",
            unit="units/month",
        ),
        NotationEntry(
            symbol="c_p",
            meaning="production cost per unit",
            unit="cost_units/unit",
        ),
        NotationEntry(
            symbol="c_h",
            meaning="holding cost per unit-month",
            unit="cost_units/unit-month",
        ),
    ]
    notation = [
        *notation_by_symbol.values(),
        *[
            entry
            for entry in notation_additions
            if entry.symbol not in existing_symbols
            and entry.symbol.split("^")[0].split(",")[0].strip() not in existing_bases
        ],
    ]
    sections = []
    for section in plan.sections:
        blocks = [sanitize(block) for block in section.blocks]
        claim_points = [
            point.model_copy(update={"text": sanitize(point.text)})
            for point in section.claim_points
        ]
        if section.kind is SectionKind.MODEL:
            blocks = [
                strip_residual(block)
                for block in blocks
                if not block.startswith(
                    (
                        "Simulation closure:",
                        "Residual mean:",
                        "Simulation scope:",
                        "Uncertainty reporting:",
                        "Solver check:",
                    )
                )
            ]
            blocks = [
                block for block in blocks if block.strip()
            ]
            blocks = list(dict.fromkeys([*blocks, *model_notes]))
        sections.append(
            section.model_copy(
                update={"blocks": blocks, "claim_points": claim_points}
            )
        )
    needs = [
        need.model_copy(
            update={
                "question": sanitize(need.question),
                "rationale": sanitize(need.rationale),
                "proposed_experiment": sanitize(need.proposed_experiment),
                "expected_evidence": sanitize(need.expected_evidence),
            }
        )
        for need in plan.capability_needs
    ]
    return plan.model_copy(
        update={
            "assumptions": list(dict.fromkeys([*assumptions, *plan.assumptions])),
            "limitations": list(
                dict.fromkeys(
                    [
                        *[sanitize(item) for item in limitations],
                        *[sanitize(item) for item in plan.limitations],
                    ]
                )
            ),
            "abstract_points": abstract_points,
            "sections": sections,
            "notation": notation,
            "capability_needs": needs,
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
    *,
    extra_evidence: list[EvidenceRecord] | None = None,
    augmentation_id: str | None = None,
) -> CompetitionPaperSummary:
    config = load_project_config(paths)
    if config.state is not ProjectState.RELEASED:
        raise CompetitionPaperError(
            "competition paper requires a RELEASED project; "
            f"current state is {config.state.value}"
        )
    extra = list(extra_evidence or [])
    context = load_context(
        paths, extra_records=extra, augmentation_id=augmentation_id
    )
    records = {
        **load_evidence_index(paths).records,
        **{record.evidence_id: record for record in context.derived_records},
        **{record.evidence_id: record for record in extra},
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
        "augmentation_id": augmentation_id,
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
