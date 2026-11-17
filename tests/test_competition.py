from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from mmos.cli.main import app
from mmos.competition.agents import (
    CompetitionAgentError,
    CompetitionAgents,
    FigureDesigner,
    MathAuditor,
    PaperArchitect,
)
from mmos.competition.context import (
    CompetitionContextError,
    _collect_experiment_ids,
    load_context,
)
from mmos.competition.number_audit import (
    NumberAuditError,
    audit_texts,
    expand_text,
    format_display_value,
)
from mmos.competition.requests import (
    CompetitionRequestError,
    build_manifest,
)
from mmos.competition_pipeline import (
    CompetitionPaperBlocked,
    run_competition_paper,
)
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    create_project,
    load_project_config,
    save_problem_spec,
    save_project_config,
)
from mmos.schemas.competition import (
    SECTION_ORDER,
    Capability,
    CapabilityNeed,
    ClaimPoint,
    CompetitionPaperPlan,
    ExperimentAnalysisReport,
    ExperimentFinding,
    FigurePlan,
    FigureTemplate,
    FindingKind,
    MathAuditIssue,
    MathAuditReport,
    MathSeverity,
    NotationEntry,
    Priority,
    SectionKind,
    SectionPlan,
    TablePlan,
    TableTemplate,
)
from mmos.schemas.evidence import EvidenceIndex, EvidenceRecord
from mmos.schemas.problem import ProblemSpec, SubProblem
from mmos.schemas.review import (
    AttackReport,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)
from mmos.schemas.states import ProjectState, TaskFamily

EVIDENCE_ID = "ev-test-objective"


def evidence_record() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=EVIDENCE_ID,
        key="optimization.ortools.cbc.objective_value",
        raw_value=5420.0,
        display_value="5420",
        unit="cost_units",
        experiment_id="exp-test",
        artifact_path="experiments/exp-test/artifacts/result.json",
        artifact_hash="a" * 64,
        source_field="objective_value",
        validation_reports=["optimization"],
        verified_at=datetime.now(UTC),
        provenance={"kind": "test"},
    )


def released_project(tmp_path: Path, name: str = "competition") -> ProjectPaths:
    paths, _ = create_project(
        tmp_path / "projects", name, task=TaskFamily.OPTIMIZATION
    )
    config = load_project_config(paths).model_copy(
        update={"state": ProjectState.RELEASED}
    )
    save_project_config(paths, config)
    save_problem_spec(
        paths,
        ProblemSpec(
            problem_id="competition-test",
            title="A frozen production planning problem",
            overall_goal="Plan production under a frozen demand forecast.",
            subproblems=[
                SubProblem(
                    subproblem_id="q1",
                    statement="Plan production with bounded capacity.",
                    goal="Minimize cost.",
                    outputs=["production plan"],
                    decision_variables=[],
                )
            ],
            global_assumptions=["Demand is represented by a frozen point forecast."],
        ),
    )
    record = evidence_record()
    index = EvidenceIndex(records={record.evidence_id: record})
    atomic_write_text(
        paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n"
    )
    atomic_write_text(paths.verified_claims_file, "claims: []\n")
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        paths.releases_dir / "release_test.json",
        '{"experiment_id": "exp-test", "objective_value": 5420, "judge": "pass"}\n',
    )
    return paths


def _section(kind: SectionKind, *, text: str | None = None) -> SectionPlan:
    return SectionPlan(
        heading=kind.value.replace("_", " ").title(),
        kind=kind,
        purpose=f"Write the {kind.value} section.",
        blocks=[
            text
            or (
                f"The frozen objective is [[ev:{EVIDENCE_ID}]], and the section "
                "describes only verified evidence."
            )
        ],
        claim_points=[
            ClaimPoint(
                text=f"Verified objective: [[ev:{EVIDENCE_ID}]]",
                evidence_ids=[EVIDENCE_ID],
            )
        ],
        evidence_ids=[EVIDENCE_ID],
    )


def full_plan(*, fabricated: bool = False) -> CompetitionPaperPlan:
    sections = []
    for kind in SECTION_ORDER:
        text = None
        if fabricated and kind is SectionKind.RESULTS:
            text = "The model achieved 12345 units, which is not in the evidence."
        sections.append(_section(kind, text=text))
    return CompetitionPaperPlan(
        title="Frozen Production Planning Report",
        abstract_points=[
            ClaimPoint(
                text=f"The frozen objective is [[ev:{EVIDENCE_ID}]].",
                evidence_ids=[EVIDENCE_ID],
            )
        ],
        sections=sections,
        tables=[
            TablePlan(
                table_id="tab-c1",
                caption="Experiment inventory from the verified project.",
                template=TableTemplate.EXPERIMENT_INVENTORY,
                evidence_ids=[EVIDENCE_ID],
            ),
            TablePlan(
                table_id="tab-c2",
                caption="Evidence lineage for the frozen release.",
                template=TableTemplate.EVIDENCE_LINEAGE,
                evidence_ids=[EVIDENCE_ID],
            ),
        ],
        figures=[],
        assumptions=["The demand forecast is frozen before optimization."],
        notation=[
            NotationEntry(
                symbol="x_t",
                meaning="production in period t",
                unit="units",
            )
        ],
        limitations=["No comparator plan was executed."],
        capability_needs=[
            CapabilityNeed(
                capability=Capability.OPTIMIZATION,
                question="How sensitive is the plan to holding cost?",
                rationale="The current evidence has a single cost instance.",
                proposed_experiment="Re-solve with holding cost variants.",
                expected_evidence="Objectives and plans for each cost level.",
                priority=Priority.HIGH,
                supporting_evidence_ids=[EVIDENCE_ID],
            ),
            CapabilityNeed(
                capability=Capability.UNSUPPORTED,
                question="How robust is the plan under a full stochastic program?",
                rationale="No robust optimization engine exists.",
                proposed_experiment="Build a stochastic programming extension.",
                expected_evidence="Robust objective and policy.",
                priority=Priority.MEDIUM,
                supporting_evidence_ids=[EVIDENCE_ID],
            ),
        ],
    )


class FakeArchitect:
    def __init__(self, plan: CompetitionPaperPlan) -> None:
        self._plan = plan

    def plan(self, context) -> CompetitionPaperPlan:
        return self._plan


class FakeAuditor:
    def __init__(self, report: MathAuditReport) -> None:
        self._report = report

    def audit(self, context, plan) -> MathAuditReport:
        return self._report


class FakeAnalyst:
    def __init__(self, report: ExperimentAnalysisReport) -> None:
        self._report = report

    def analyze(self, context, plan) -> ExperimentAnalysisReport:
        return self._report


class FakeFigureDesigner:
    def __init__(self, figures: list[FigurePlan]) -> None:
        self._figures = figures

    def design(self, context, plan, analysis) -> list[FigurePlan]:
        return self._figures


def agents(
    plan: CompetitionPaperPlan | None = None,
    *,
    math: MathAuditReport | None = None,
    analysis: ExperimentAnalysisReport | None = None,
    figures: list[FigurePlan] | None = None,
) -> CompetitionAgents:
    return CompetitionAgents(
        architect=FakeArchitect(plan or full_plan()),
        math_auditor=FakeAuditor(
            math or MathAuditReport(issues=[], summary="No issues found.")
        ),
        experiment_analyst=FakeAnalyst(
            analysis
            or ExperimentAnalysisReport(
                findings=[
                    ExperimentFinding(
                        finding_id="ef-1",
                        statement="The frozen objective is verified.",
                        evidence_ids=[EVIDENCE_ID],
                        kind=FindingKind.SUPPORTED,
                    )
                ],
                limitations=["No sensitivity sweep exists."],
                requested_checks=[],
            )
        ),
        figure_designer=FakeFigureDesigner(figures or []),
    )


def reviewer(verdict: Verdict, calls: list[str] | None = None):
    def run(ast, experiment_id: str) -> RedTeamReview:
        if calls is not None:
            calls.append(experiment_id)
        return RedTeamReview(
            experiment_id=experiment_id,
            attacks=AttackReport(attacks=[]),
            responses=AuthorResponseReport(responses=[]),
            judge=JudgeReport(verdicts=[], overall=verdict, blocking=False),
            created_at=datetime.now(UTC),
        )

    return run


def test_number_audit_accepts_evidence_and_metadata() -> None:
    records = {EVIDENCE_ID: evidence_record()}
    fragment = expand_text(
        f"Objective [[ev:{EVIDENCE_ID}]] with a horizon of 3.",
        records,
        location="Abstract",
    )
    report = audit_texts([fragment], records, ("3",))
    assert not report.blocking
    assert report.checked_tokens >= 2


def test_number_audit_rejects_fabricated_value() -> None:
    records = {EVIDENCE_ID: evidence_record()}
    fragment = expand_text(
        f"The objective is [[ev:{EVIDENCE_ID}]] and the gain is 12345.",
        records,
        location="Results",
    )
    report = audit_texts([fragment], records, ())
    assert report.blocking
    assert any(item.token == "12345" for item in report.unverified)


def test_number_audit_rejects_unknown_evidence() -> None:
    with pytest.raises(NumberAuditError):
        expand_text("Value [[ev:missing]]", {}, location="Abstract")


def test_number_audit_accepts_rounded_evidence_value() -> None:
    record = evidence_record().model_copy(
        update={"raw_value": 0.9924756193, "display_value": "0.9924756193"}
    )
    records = {EVIDENCE_ID: record}
    assert format_display_value(record.raw_value, record.display_value) == "0.9925"
    fragment = expand_text(
        f"Fill rate [[ev:{EVIDENCE_ID}]].",
        records,
        location="Results",
    )
    assert fragment.text == "Fill rate 0.9925."
    report = audit_texts([fragment], records, ())
    assert not report.blocking


def test_number_audit_formats_simulation_summary_at_two_digits() -> None:
    record = evidence_record().model_copy(
        update={
            "key": "mixed.simulation.shortage.mean_total_shortage",
            "raw_value": 3.444970588,
            "display_value": "3.444970588",
        }
    )
    records = {EVIDENCE_ID: record}
    fragment = expand_text(
        f"Shortage [[ev:{EVIDENCE_ID}]].",
        records,
        location="Results",
    )
    assert fragment.text == "Shortage 3.4."
    assert not audit_texts([fragment], records, ()).blocking


def test_experiment_task_mapping_is_not_first_prefix() -> None:
    rows = _collect_experiment_ids(
        {
            "forecast_experiment_id": "exp-forecast",
            "optimization_experiment_id": "exp-optimization",
            "simulation_experiment_id": "exp-simulation",
        }
    )
    assert rows == [
        ("exp-forecast", "forecast", "forecasting"),
        ("exp-optimization", "optimization", "optimization"),
        ("exp-simulation", "simulation", "simulation"),
    ]


def test_request_classification_and_unknown_evidence() -> None:
    needs = full_plan().capability_needs
    manifest = build_manifest(needs, {EVIDENCE_ID})
    assert manifest.supported_count == 1
    assert manifest.unsupported_count == 1
    assert all(request.executed is False for request in manifest.requests)
    bad = [
        CapabilityNeed(
            capability=Capability.OPTIMIZATION,
            question="Q",
            rationale="R",
            proposed_experiment="P",
            expected_evidence="E",
            supporting_evidence_ids=["missing"],
        )
    ]
    with pytest.raises(CompetitionRequestError):
        build_manifest(bad, {EVIDENCE_ID})


def test_context_requires_released_project(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "not-released")
    with pytest.raises(CompetitionContextError, match="RELEASED"):
        load_context(paths)


def test_context_builds_catalog(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    context = load_context(paths)
    assert context.evidence_ids() == {EVIDENCE_ID}
    assert context.available_tables[TableTemplate.EXPERIMENT_INVENTORY.value] == 1
    assert context.context_hash


def test_pipeline_releases_with_unsupported_requests(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    summary = run_competition_paper(
        paths, agents(), reviewer(Verdict.PASS)
    )
    assert summary.run.status == "released"
    assert summary.unsupported_request_count == 1
    assert summary.run.release_path is not None
    assert (paths.root / summary.run.release_path).exists()
    assert (paths.root / "paper" / "competition" / "report.md").exists()
    assert (paths.root / "paper" / "competition" / "experiment_requests.json").exists()
    assert load_project_config(paths).state is ProjectState.RELEASED


def test_pipeline_blocks_on_red_team_without_release(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    with pytest.raises(CompetitionPaperBlocked, match="red_team"):
        run_competition_paper(paths, agents(), reviewer(Verdict.REJECT))
    assert not list(paths.releases_dir.glob("competition_paper_*.json"))
    run_record = paths.root / "paper" / "competition" / "run.json"
    assert run_record.exists()


def test_pipeline_number_audit_blocks_before_review(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    calls: list[str] = []
    with pytest.raises(CompetitionPaperBlocked, match="number_audit"):
        run_competition_paper(
            paths,
            agents(full_plan(fabricated=True)),
            reviewer(Verdict.PASS, calls),
        )
    assert calls == []
    assert not list(paths.releases_dir.glob("competition_paper_*.json"))


def test_pipeline_math_critical_blocks(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    critical = MathAuditReport(
        issues=[
            MathAuditIssue(
                issue_id="ma-1",
                severity=MathSeverity.CRITICAL,
                category="constraint",
                target="Model",
                finding="A hard constraint is missing.",
                required_fix="Add the missing constraint and rerun.",
            )
        ],
        summary="Critical issue.",
    )
    with pytest.raises(CompetitionPaperBlocked, match="math_audit"):
        run_competition_paper(
            paths, agents(math=critical), reviewer(Verdict.PASS)
        )


def test_pipeline_archives_previous_run(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    first = run_competition_paper(paths, agents(), reviewer(Verdict.PASS))
    second = run_competition_paper(paths, agents(), reviewer(Verdict.PASS))
    history = paths.root / "paper" / "competition" / "history" / first.run.run_id
    assert history.exists()
    assert (history / "report.md").exists()
    assert second.run.run_id != first.run.run_id


def test_pipeline_figure_render(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    figure = FigurePlan(
        figure_id="fig-c1",
        caption="Frozen experiment lineage.",
        template=FigureTemplate.PIPELINE_SCHEMATIC,
        evidence_ids=[EVIDENCE_ID],
    )
    summary = run_competition_paper(
        paths, agents(figures=[figure]), reviewer(Verdict.PASS)
    )
    figure_path = paths.root / "paper" / "competition" / "figures" / "fig-c1.png"
    assert figure_path.exists()
    assert (figure_path.with_suffix(".json")).exists()
    assert summary.run.status == "released"


def test_real_agents_normalize_ids(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    context = load_context(paths)
    plan = full_plan()
    architect = PaperArchitect(TestModel(custom_output_args=plan.model_dump()))
    built = architect.plan(context)
    assert built.figures == []
    assert [table.table_id for table in built.tables] == ["tab-c1", "tab-c2"]

    audit = MathAuditor(
        TestModel(
            custom_output_args=MathAuditReport(
                issues=[
                    MathAuditIssue(
                        issue_id="wrong",
                        severity=MathSeverity.MINOR,
                        category="notation",
                        target="Model",
                        finding="Symbols overlap.",
                        required_fix="Rename one symbol.",
                    )
                ],
                summary="One minor issue.",
            ).model_dump()
        )
    ).audit(context, built)
    assert [issue.issue_id for issue in audit.issues] == ["ma-1"]


def test_figure_designer_rejects_unavailable_template(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    context = load_context(paths)
    plan = full_plan()
    analysis = ExperimentAnalysisReport(findings=[], limitations=[])
    unavailable = FigurePlan(
        figure_id="fig",
        caption="Metric bars.",
        template=FigureTemplate.METRIC_BAR,
        metric_prefix="metrics.",
        evidence_ids=[EVIDENCE_ID],
    )
    designer = FigureDesigner(
        TestModel(custom_output_args={"figures": [unavailable.model_dump()]})
    )
    with pytest.raises(CompetitionAgentError, match="no numeric evidence"):
        designer.design(context, plan, analysis)


def test_cli_requires_released_project(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "cli-not-released")
    result = CliRunner().invoke(
        app,
        [
            "competition-paper",
            paths.root.name,
            "--projects-dir",
            str(tmp_path / "projects"),
        ],
    )
    assert result.exit_code == 1
    assert "requires a RELEASED project" in result.stdout


def test_plan_schema_rejects_missing_sections() -> None:
    with pytest.raises(ValidationError):
        CompetitionPaperPlan(
            title="T",
            abstract_points=[ClaimPoint(text="A", evidence_ids=[EVIDENCE_ID])],
            sections=[_section(SectionKind.RESULTS)],
            tables=[],
            figures=[],
            assumptions=["A"],
            notation=[NotationEntry(symbol="x", meaning="y")],
            limitations=["L"],
        )
