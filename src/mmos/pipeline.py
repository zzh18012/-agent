import json
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from pydantic_ai.models import Model

from mmos.agents import (
    ModelCouncil,
    ProblemAnalyst,
    RedTeam,
    ResearchAgent,
    build_model,
    release_allowed,
)
from mmos.agents.catalog import EXECUTABLE_MODEL_IDS
from mmos.config import get_settings
from mmos.core.dataset import dataset_fingerprint
from mmos.core.problem_text import collect_problem_text
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_council_decision,
    load_experiment_spec,
    load_manifest,
    load_pipeline_state,
    load_problem_spec,
    load_project_config,
    load_proposals,
    load_red_team_review,
    load_research_result,
    load_tournament_result,
    save_council_decision,
    save_critiques,
    save_experiment_spec,
    save_pipeline_state,
    save_problem_spec,
    save_proposals,
    save_red_team_review,
    save_release_summary,
    save_research_result,
    save_tournament_result,
)
from mmos.core.state_machine import LINEAR_ORDER
from mmos.data import DataAgent
from mmos.data.metadata import load_dataset_metadata
from mmos.data.table import load_single_table
from mmos.evidence import (
    add_claim,
    freeze_experiment,
    load_claims,
    load_evidence_index,
)
from mmos.experiments.planner import ExperimentPlanner
from mmos.experiments.tournament import Tournament
from mmos.experiments.workflow import run_experiment_workflow
from mmos.orchestration import Orchestrator
from mmos.paper import build_paper, render_paper
from mmos.registries import ModelRegistry
from mmos.schemas.experiment import ExperimentResult, TemporalSplit
from mmos.schemas.pipeline import PipelineState, ReleaseSummary
from mmos.schemas.proposal import CouncilDecision, CritiqueReport, ModelProposal
from mmos.schemas.states import ESCAPE_STATES, ProjectState
from mmos.validation import validate_experiment

log = structlog.get_logger("mmos.pipeline")


class PipelineBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineOptions:
    target: str
    features: list[str]
    primary_metric: str = "rmse"
    higher_is_better: bool = False
    metrics: list[str] | None = None
    n_splits: int = 5
    seed: int = 0
    timeout_seconds: float = 900.0
    time_column: str | None = None
    forecast_horizon: int | None = None
    seasonal_period: int | None = None


def select_model_ids(decision: CouncilDecision, proposals: list[ModelProposal]) -> list[str]:
    by_id = {proposal.proposal_id: proposal for proposal in proposals}
    selected: list[str] = []
    for proposal_id in decision.selected_for_prototype:
        proposal = by_id.get(proposal_id)
        if proposal is None:
            continue
        for model_id in proposal.registry_model_ids:
            if model_id not in selected:
                selected.append(model_id)
    return selected


def _rank(state: ProjectState) -> int:
    return LINEAR_ORDER.index(state) if state in LINEAR_ORDER else -1


def _escape(paths: ProjectPaths, reason: str) -> None:
    config = load_project_config(paths)
    if config.state not in ESCAPE_STATES:
        Orchestrator(paths).transition(ProjectState.REQUIRES_HUMAN_REVIEW, reason=reason)


def _load_or_create_state(paths: ProjectPaths, options: PipelineOptions) -> PipelineState:
    state = load_pipeline_state(paths)
    if state is None:
        state = PipelineState(
            target_column=options.target,
            feature_set=options.features,
            primary_metric=options.primary_metric,
            higher_is_better=options.higher_is_better,
            metrics=options.metrics,
            n_splits=options.n_splits,
            random_seed=options.seed,
            time_column=options.time_column,
            forecast_horizon=options.forecast_horizon,
            seasonal_period=options.seasonal_period,
        )
        save_pipeline_state(paths, state)
        return state

    expected = (
        options.target,
        options.features,
        options.primary_metric,
        options.higher_is_better,
        options.metrics,
        options.n_splits,
        options.seed,
        options.time_column,
        options.forecast_horizon,
        options.seasonal_period,
    )
    actual = (
        state.target_column,
        state.feature_set,
        state.primary_metric,
        state.higher_is_better,
        state.metrics,
        state.n_splits,
        state.random_seed,
        state.time_column,
        state.forecast_horizon,
        state.seasonal_period,
    )
    if expected != actual:
        raise ValueError(
            "this project's pipeline was already started with different options; "
            "use a fresh project or remove manifests/pipeline_state.json deliberately"
        )
    return state


@dataclass(frozen=True)
class TemporalPlan:
    time_column: str
    split: TemporalSplit
    baseline_model_ids: list[str]
    seasonal_period: int | None


def _resolve_temporal(paths: ProjectPaths, options: PipelineOptions) -> TemporalPlan | None:
    metadata = load_dataset_metadata(paths)
    declared = metadata.temporal if metadata is not None else None
    time_column = options.time_column or (
        declared.time_column if declared is not None and declared.is_temporal else None
    )
    if time_column is None:
        return None
    horizon = options.forecast_horizon or (
        declared.forecast_horizon if declared is not None and declared.is_temporal else None
    )
    seasonal_period = options.seasonal_period or (
        declared.seasonal_period if declared is not None else None
    )
    if horizon is None:
        raise PipelineBlocked(
            "temporal data requires an explicit forecast horizon; pass --horizon "
            "(the system will not guess it)"
        )
    n_rows = load_single_table(paths).height
    step = horizon
    min_train_size = n_rows - options.n_splits * step
    if min_train_size < 10:
        raise PipelineBlocked(
            f"not enough rows for {options.n_splits} rolling folds with horizon {horizon}"
        )
    split = TemporalSplit(
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
        n_splits=options.n_splits,
    )
    baselines = ["naive.last_value", "naive.drift"]
    if seasonal_period is not None and seasonal_period <= min_train_size // 2:
        baselines.append("naive.seasonal")
    return TemporalPlan(
        time_column=time_column,
        split=split,
        baseline_model_ids=baselines,
        seasonal_period=seasonal_period,
    )


def _release_markdown(summary: ReleaseSummary) -> str:
    validation_rows = "\n".join(
        f"| {name} | {status} |" for name, status in sorted(summary.validation.items())
    )
    notes = "\n".join(f"- {note}" for note in summary.notes) or "- none"
    return f"""# Release {summary.release_id}

- Experiment: `{summary.experiment_id}`
- Tournament: `{summary.tournament_id}`
- Selected model: `{summary.selected_model}`
- Challenger: `{summary.challenger_model or "-"}`
- Primary metric: `{summary.primary_metric} = {summary.metric_display_value}`
- Red team overall: `{summary.red_team_overall}` (blocking: {summary.red_team_blocking})
- Paper: `{summary.paper_path}`
- Claims: {len(summary.claim_ids)} registered, {summary.supported_claims} supported

## Validation

| Validator | Status |
| --- | --- |
{validation_rows}

## Notes

{notes}
"""


def run_pipeline(
    paths: ProjectPaths,
    options: PipelineOptions,
    *,
    model: Model | str | None = None,
) -> ReleaseSummary:
    settings = get_settings()
    registry_file = settings.model_registry_file
    if not registry_file.exists():
        raise FileNotFoundError(f"model registry not found: {registry_file}")
    registry = ModelRegistry.load(registry_file)

    cached_model: Model | str | None = model

    def get_model() -> Model | str:
        nonlocal cached_model
        if cached_model is None:
            cached_model = build_model(settings)
        return cached_model

    orchestrator = Orchestrator(paths)
    config = load_project_config(paths)
    if config.state in ESCAPE_STATES:
        raise PipelineBlocked(
            f"project is in {config.state.value}; resolve the human-review state first"
        )
    if config.state is ProjectState.INIT:
        raise ValueError("project has no ingested inputs; run mmos init and mmos ingest first")

    pipeline = _load_or_create_state(paths, options)
    temporal = _resolve_temporal(paths, options)

    if config.state is ProjectState.PROJECT_INGESTED:
        log.info("pipeline_stage", stage="problem_analysis")
        spec = ProblemAnalyst(get_model()).analyze(
            collect_problem_text(paths), problem_id=config.project_id
        )
        save_problem_spec(paths, spec)
        orchestrator.transition(
            ProjectState.PROBLEM_UNDERSTOOD, reason="pipeline: problem analyzed"
        )

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.DATA_VALIDATED):
        log.info("pipeline_stage", stage="data_gate")
        data_result = DataAgent(paths).analyze(target=options.target)
        if data_result.quality.blocking or data_result.leakage.blocking:
            _escape(paths, "pipeline: data quality gate failed")
            raise PipelineBlocked("data quality gate failed")
        orchestrator.advance_to(ProjectState.DATA_VALIDATED, reason="pipeline: data validated")

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.RESEARCH_COMPLETED):
        log.info("pipeline_stage", stage="research")
        research = ResearchAgent(get_model()).research(load_problem_spec(paths), registry=registry)
        save_research_result(paths, research)
        orchestrator.advance_to(
            ProjectState.RESEARCH_COMPLETED, reason="pipeline: research completed"
        )

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.CANDIDATES_PROPOSED):
        log.info("pipeline_stage", stage="model_council")
        outcome = ModelCouncil(get_model()).propose(
            load_problem_spec(paths),
            load_research_result(paths),
            registry=registry,
        )
        save_proposals(paths, outcome.proposals)
        save_council_decision(paths, outcome.decision)
        if outcome.critiques:
            save_critiques(paths, CritiqueReport(critiques=outcome.critiques))
        orchestrator.advance_to(
            ProjectState.CANDIDATES_PROPOSED, reason="pipeline: council proposals recorded"
        )

    if pipeline.experiment_id is None:
        log.info("pipeline_stage", stage="plan_experiment")
        proposals = load_proposals(paths)
        decision = load_council_decision(paths)
        model_ids = [
            model_id
            for model_id in select_model_ids(decision, proposals)
            if model_id in EXECUTABLE_MODEL_IDS
        ]
        if temporal is not None and temporal.seasonal_period is None:
            model_ids = [model_id for model_id in model_ids if model_id != "naive.seasonal"]
        if not model_ids and temporal is None:
            raise PipelineBlocked("council selected no executable model")
        base_proposal = next(
            proposal
            for proposal in proposals
            if proposal.proposal_id == decision.selected_for_prototype[0]
        ).model_copy(update={"registry_model_ids": model_ids})
        spec = ExperimentPlanner(registry).plan(
            base_proposal,
            dataset_hash=dataset_fingerprint(load_manifest(paths)),
            target_column=options.target,
            feature_set=options.features,
            metrics=options.metrics,
            n_splits=options.n_splits,
            random_seed=options.seed,
            include_baseline=temporal is None,
            baseline_model_ids=temporal.baseline_model_ids if temporal else None,
            temporal_split=temporal.split if temporal else None,
            time_column=temporal.time_column if temporal else None,
            seasonal_period=temporal.seasonal_period if temporal else None,
        )
        save_experiment_spec(paths, spec)
        pipeline = pipeline.model_copy(update={"experiment_id": spec.experiment_id})
        save_pipeline_state(paths, pipeline)

    experiment_id = pipeline.experiment_id
    if experiment_id is None:
        raise PipelineBlocked("pipeline state has no experiment id")
    load_experiment_spec(paths, experiment_id)

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.PROTOTYPING):
        orchestrator.advance_to(ProjectState.PROTOTYPING, reason="pipeline: experiment planned")

    metrics_file = paths.experiment_dir(experiment_id) / "artifacts" / "metrics.json"
    if not metrics_file.exists():
        log.info("pipeline_stage", stage="run_experiment", experiment_id=experiment_id)
        execution = run_experiment_workflow(
            paths, experiment_id, timeout_seconds=options.timeout_seconds
        )
        if not execution.succeeded():
            _escape(
                paths,
                "pipeline: workflow failed "
                f"(exit={execution.exit_code}, timed_out={execution.timed_out})",
            )
            raise PipelineBlocked(
                f"workflow failed; stdout={execution.stdout_path}, stderr={execution.stderr_path}"
            )

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.EXPERIMENT_COMPLETED):
        orchestrator.advance_to(
            ProjectState.EXPERIMENT_COMPLETED, reason="pipeline: experiment artifacts complete"
        )

    if pipeline.tournament_id is None:
        log.info("pipeline_stage", stage="tournament")
        result = ExperimentResult.model_validate_json(metrics_file.read_text(encoding="utf-8"))
        tournament = Tournament(registry).run(
            [result],
            primary_metric=options.primary_metric,
            higher_is_better=options.higher_is_better,
        )
        save_tournament_result(paths, tournament)
        pipeline = pipeline.model_copy(update={"tournament_id": tournament.tournament_id})
        save_pipeline_state(paths, pipeline)
        if tournament.blocked:
            _escape(paths, "pipeline: tournament blocked; human review required")
            raise PipelineBlocked("tournament is blocked: " + "; ".join(tournament.notes))

    tournament_id = pipeline.tournament_id
    if tournament_id is None:
        raise PipelineBlocked("pipeline state has no tournament id")

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.VALIDATING):
        orchestrator.advance_to(ProjectState.VALIDATING, reason="pipeline: validation started")
    log.info("pipeline_stage", stage="validation")
    reports = validate_experiment(paths, experiment_id)
    if any(report.blocking for report in reports):
        _escape(paths, "pipeline: validation blocking")
        raise PipelineBlocked("validation has blocking issues")

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.VERIFIED):
        orchestrator.advance_to(ProjectState.VERIFIED, reason="pipeline: validators passed")

    state = load_project_config(paths).state
    index = load_evidence_index(paths)
    if not any(record.experiment_id == experiment_id for record in index.records.values()):
        log.info("pipeline_stage", stage="freeze_evidence")
        freeze_experiment(paths, experiment_id)
        index = load_evidence_index(paths)
    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.EVIDENCE_FROZEN):
        orchestrator.advance_to(ProjectState.EVIDENCE_FROZEN, reason="pipeline: evidence frozen")

    tournament = load_tournament_result(paths, tournament_id)
    selected_model = tournament.selected_model
    if selected_model is None:
        raise PipelineBlocked("tournament has no selected model")
    evidence_id = f"ev-{experiment_id}-{selected_model}-{options.primary_metric}"
    evidence = index.records.get(evidence_id)
    if evidence is None:
        raise PipelineBlocked(f"verified evidence missing for selected model: {evidence_id}")

    claim_id = f"claim-auto-{experiment_id}"
    claims = load_claims(paths)
    if not any(claim.claim_id == claim_id for claim in claims):
        add_claim(
            paths,
            statement=(
                f"{selected_model} achieves {options.primary_metric} = "
                f"{evidence.display_value} in verified cross-validation."
            ),
            evidence_ids=[evidence_id],
            claim_id=claim_id,
            accept=False,
            notes="Auto-generated by mmos run; human acceptance is required.",
        )

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.PAPER_DRAFTED):
        log.info("pipeline_stage", stage="paper")
        paper_ast = build_paper(
            paths,
            experiment_id,
            tournament_id=tournament_id,
            primary_metric=options.primary_metric,
            higher_is_better=options.higher_is_better,
        )
        render_paper(paths, paper_ast)
        orchestrator.advance_to(ProjectState.PAPER_DRAFTED, reason="pipeline: paper rendered")

    review_file = paths.reviews_dir / f"redteam_{experiment_id}.json"
    if review_file.exists():
        review = load_red_team_review(paths, experiment_id)
    else:
        log.info("pipeline_stage", stage="red_team")
        paper_ast = build_paper(
            paths,
            experiment_id,
            tournament_id=tournament_id,
            primary_metric=options.primary_metric,
            higher_is_better=options.higher_is_better,
        )
        review = RedTeam(get_model()).run(paper_ast, experiment_id=experiment_id)
        save_red_team_review(paths, review)

    if not release_allowed(review):
        verdicts = {verdict.attack_id: verdict for verdict in review.judge.verdicts}
        lines = [
            "# Release blocked — human review required",
            "",
            f"Experiment: `{experiment_id}`",
            "",
        ]
        for attack in review.attacks.attacks:
            verdict = verdicts[attack.attack_id]
            lines.extend(
                [
                    f"## {attack.attack_id} — {attack.severity.value} / "
                    f"{attack.category.value} → {verdict.verdict.value}",
                    f"- Target: {attack.target}",
                    f"- Attack: {attack.argument}",
                    f"- Required fix: {attack.required_fix}",
                    f"- Judge: {verdict.rationale}",
                    "",
                ]
            )
        atomic_write_text(
            paths.reviews_dir / f"blocked_release_{experiment_id}.md",
            "\n".join(lines),
        )
        _escape(paths, "pipeline: red team did not pass")
        raise PipelineBlocked(
            "red team did not pass; release requires judge=PASS "
            f"(got overall={review.judge.overall.value})"
        )

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.READY_FOR_RELEASE):
        orchestrator.advance_to(ProjectState.READY_FOR_RELEASE, reason="pipeline: red team passed")

    validation_summary = json.loads(
        (paths.experiment_dir(experiment_id) / "validation" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    all_claims = load_claims(paths)
    supported = sum(
        1
        for claim in all_claims
        if claim.status == "supported"
        and all(evidence_id in index.records for evidence_id in claim.evidence_ids)
    )
    notes = ["Claims are proposed only unless a human has accepted them."]
    if not supported:
        notes.append("No supported claims exist yet; the paper contains no claim statements.")

    summary = ReleaseSummary(
        release_id=f"release-{experiment_id}",
        experiment_id=experiment_id,
        tournament_id=tournament_id,
        selected_model=selected_model,
        challenger_model=tournament.challenger_model,
        primary_metric=options.primary_metric,
        metric_display_value=evidence.display_value,
        validation={name: info["status"] for name, info in validation_summary["reports"].items()},
        red_team_overall=review.judge.overall.value,
        red_team_blocking=review.judge.blocking,
        claim_ids=[claim.claim_id for claim in all_claims],
        supported_claims=supported,
        paper_path=str((paths.root / "paper" / "report.md").relative_to(paths.root).as_posix()),
        created_at=datetime.now(UTC).replace(microsecond=0),
        notes=notes,
    )
    save_release_summary(paths, summary)
    atomic_write_text(paths.releases_dir / f"{summary.release_id}.md", _release_markdown(summary))

    state = load_project_config(paths).state
    if _rank(state) < _rank(ProjectState.RELEASED):
        orchestrator.advance_to(ProjectState.RELEASED, reason="pipeline: release summary written")
    return summary
