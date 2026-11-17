from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mmos.competition.clock import FrozenClock, ensure_aware, zone
from mmos.competition.controller import (
    CompetitionController,
    ControllerBlocked,
    _hash_model,
    load_blocked_fixture,
)
from mmos.competition.controller_schemas import (
    ActionCategory,
    ClaimCriticality,
    CompetitionControllerRun,
    CompetitionControllerState,
    CompetitionFixtureRecord,
    CompetitionTimingProfile,
    CompetitionWorkItem,
    EvidenceFreezeRecord,
    GuardVerdict,
    WorkItemKind,
    WorkItemStatus,
    WorkSeverity,
)
from mmos.competition.guard import (
    auto_scope_reductions,
    evaluate_action,
    priority_key,
    readiness_reasons,
)
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    create_project,
    load_project_config,
    save_project_config,
    sha256_file,
)
from mmos.schemas.competition import (
    CompetitionPaperRun,
    ResolutionClass,
)
from mmos.schemas.evidence import EvidenceIndex, EvidenceRecord
from mmos.schemas.states import ProjectState, TaskFamily

START = datetime(2026, 1, 1, 0, tzinfo=UTC)


def profile(**overrides: Any) -> CompetitionTimingProfile:
    values = {
        "profile_id": "profile-test",
        "competition_name": "Test Competition",
        "timezone": "Asia/Shanghai",
        "contest_start_at": START,
        "exploration_end_at": START + timedelta(hours=6),
        "model_freeze_at": START + timedelta(hours=12),
        "evidence_freeze_at": START + timedelta(days=1, hours=12),
        "paper_freeze_at": START + timedelta(days=2, hours=12),
        "work_stop_at": START + timedelta(days=3),
        "submission_deadline_at": START + timedelta(days=3, hours=2),
        "source": "test",
        "source_hash": "a" * 64,
    }
    values.update(overrides)
    return CompetitionTimingProfile(**values)


def work_item(
    item_id: str,
    *,
    kind: WorkItemKind = WorkItemKind.REVIEW_ISSUE,
    blocking: bool = False,
    severity: WorkSeverity = WorkSeverity.MAJOR,
    claim: ClaimCriticality = ClaimCriticality.SUPPORTING,
    capability: ResolutionClass | None = ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY,
    action: ActionCategory = ActionCategory.PAPER_MUTATION,
    p0: bool = False,
    status: WorkItemStatus = WorkItemStatus.PENDING,
    dependencies: list[str] | None = None,
) -> CompetitionWorkItem:
    return CompetitionWorkItem(
        work_item_id=item_id,
        kind=kind,
        source_type="test",
        source_id=item_id,
        title=item_id,
        description=f"description for {item_id}",
        blocking_release=blocking,
        scientific_severity=severity,
        claim_criticality=claim,
        capability_status=capability,
        expected_runtime_class="seconds",
        dependencies=dependencies or [],
        created_at=START,
        action_category=action,
        status=status,
        p0_repair=p0,
    )


def controller_project(tmp_path: Path, *, judge: str = "revise") -> ProjectPaths:
    paths, _ = create_project(tmp_path / "projects", "controller", task=TaskFamily.OPTIMIZATION)
    config = load_project_config(paths).model_copy(
        update={"state": ProjectState.RELEASED}
    )
    save_project_config(paths, config)
    record = EvidenceRecord(
        evidence_id="ev-controller",
        key="optimization.objective",
        raw_value=5420.0,
        display_value="5420",
        unit="cost_units",
        experiment_id="exp-controller",
        artifact_path="experiments/exp-controller/artifacts/result.json",
        artifact_hash="b" * 64,
        source_field="objective",
        validation_reports=["optimization"],
        verified_at=START,
        provenance={"kind": "test"},
    )
    atomic_write_text(
        paths.verified_numbers_file,
        EvidenceIndex(records={record.evidence_id: record}).model_dump_json(indent=2)
        + "\n",
    )
    release = paths.releases_dir / "release_controller.json"
    atomic_write_text(
        release,
        '{"run_id":"run-test","forecast_experiment_id":"exp-forecast",'
        '"optimization_experiment_id":"exp-optimization",'
        '"simulation_experiment_id":"exp-simulation",'
        '"forecast_artifact_id":"artifact-forecast","forecast_artifact_hash":"c",'
        '"production_plan_artifact_id":"artifact-plan","production_plan_artifact_hash":"d",'
        '"forecast_error_artifact_id":"artifact-error","forecast_error_artifact_hash":"e"}\n',
    )
    paper_dir = paths.root / "paper" / "competition"
    paper_dir.mkdir(parents=True)
    report = paper_dir / "report.md"
    atomic_write_text(report, "# Frozen paper\n")
    run = CompetitionPaperRun(
        run_id="comp-controller",
        source_release="releases/release_controller.json",
        source_release_hash=sha256_file(release),
        context_hash="c" * 64,
        plan_hash="d" * 64,
        math_audit_hash="e" * 64,
        experiment_analysis_hash="f" * 64,
        number_audit_hash="1" * 64,
        request_manifest_hash="2" * 64,
        status="blocked",
        red_team_overall="revise" if judge != "pass" else "pass",
        paper_path="paper/competition/report.md",
        review_path="reviews/redteam_comp-controller.json",
        created_at=START,
    )
    atomic_write_text(paper_dir / "run.json", run.model_dump_json(indent=2) + "\n")
    atomic_write_text(
        paper_dir / "number_audit.json",
        '{"checked_fragments": 1, "checked_tokens": 1, "unverified": [], '
        '"allowed_metadata_numbers": []}\n',
    )
    atomic_write_text(
        paper_dir / "math_audit.json",
        '{"issues": [], "summary": "no issues"}\n',
    )
    return paths


def fixture(
    *, judge: str = "revise", include_optional: bool = True, include_blockers: bool = True
) -> CompetitionFixtureRecord:
    items = [
        work_item(
            "crit-a1",
            kind=WorkItemKind.SCIENTIFIC_CRITICAL,
            blocking=True,
            severity=WorkSeverity.CRITICAL,
            claim=ClaimCriticality.CORE,
            p0=True,
        ),
        work_item(
            "crit-a2",
            kind=WorkItemKind.SCIENTIFIC_CRITICAL,
            blocking=True,
            severity=WorkSeverity.CRITICAL,
            claim=ClaimCriticality.CORE,
            capability=ResolutionClass.REQUIRES_NEW_CAPABILITY,
            action=ActionCategory.COMPUTE,
            p0=True,
        ),
        work_item(
            "req-new",
            kind=WorkItemKind.EXPERIMENT_REQUEST,
            capability=ResolutionClass.REQUIRES_NEW_CAPABILITY,
            action=ActionCategory.COMPUTE,
        ),
        work_item(
            "req-human",
            kind=WorkItemKind.EXPERIMENT_REQUEST,
            capability=ResolutionClass.REQUIRES_HUMAN_DECISION,
            action=ActionCategory.COMPUTE,
        ),
        work_item("review-major"),
    ]
    if include_optional:
        items.append(
            work_item(
                "optional-polish",
                kind=WorkItemKind.OPTIONAL_IMPROVEMENT,
                severity=WorkSeverity.MINOR,
                claim=ClaimCriticality.OPTIONAL,
            )
        )
    if not include_blockers:
        items = [item for item in items if not item.blocking_release]
    return CompetitionFixtureRecord(
        source_paper_run_id="comp-controller",
        judge_status=judge,
        number_audit_ok=True,
        math_audit_ok=True,
        items=items,
    )


def controller(
    tmp_path: Path, *, judge: str = "revise", at: datetime = START
) -> CompetitionController:
    paths = controller_project(tmp_path, judge=judge)
    clock = FrozenClock(at)
    return CompetitionController.initialize(
        paths,
        profile(),
        fixture(judge=judge),
        clock=clock,
        run_id="ctrl-test",
    )


def test_timing_profile_rejects_naive_and_bad_order() -> None:
    with pytest.raises(ValidationError):
        profile(contest_start_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        profile(timezone="Mars/Olympus")
    with pytest.raises(ValidationError):
        profile(work_stop_at=START + timedelta(days=4), submission_deadline_at=START)
    with pytest.raises(ValidationError):
        profile(model_freeze_at=START + timedelta(hours=1))
    allowed = profile(
        work_stop_at=START + timedelta(days=3),
        submission_deadline_at=START + timedelta(days=3),
    )
    assert allowed.work_stop_at == allowed.submission_deadline_at


def test_clock_and_timezone_equivalence() -> None:
    other = START.astimezone(zone("Asia/Shanghai"))
    assert ensure_aware(other, "other") == START
    clock = FrozenClock(START)
    clock.set(other)
    assert clock.now() == START


def test_priority_and_scope_reduction_keep_blockers_first() -> None:
    items = fixture().items
    counts: dict[str, int] = {}
    ordered = sorted(items, key=lambda item: priority_key(item, counts))
    assert ordered[0].work_item_id.startswith("crit-")
    assert ordered[-1].work_item_id == "optional-polish"
    run = CompetitionControllerRun(
        competition_run_id="r",
        profile_id="p",
        profile_hash="h",
        source_project_id="s",
        source_paper_run_id="paper",
        source_release="release.json",
        source_release_hash="h",
        source_verified_numbers_hash="v",
        state=CompetitionControllerState.ACTIVE_EXPLORATION,
        state_history=[CompetitionControllerState.ACTIVE_EXPLORATION],
        unresolved_blockers=["crit-a1"],
        created_at=START,
        updated_at=START,
    )
    decisions = auto_scope_reductions(
        items=items,
        run=run,
        profile=profile(),
        now=START + timedelta(hours=7),
    )
    targets = {target for decision in decisions for target in decision.target_work_item_ids}
    assert "optional-polish" in targets
    assert "crit-a1" not in targets


def test_guard_boundaries_and_readiness() -> None:
    p = profile()
    run = CompetitionControllerRun(
        competition_run_id="r",
        profile_id="p",
        profile_hash="h",
        source_project_id="s",
        source_paper_run_id="paper",
        source_release="release.json",
        source_release_hash="h",
        source_verified_numbers_hash="v",
        state=CompetitionControllerState.ACTIVE_EXPLORATION,
        state_history=[CompetitionControllerState.ACTIVE_EXPLORATION],
        created_at=START,
        updated_at=START,
    )
    item = work_item("paper-repair", p0=True, blocking=True, severity=WorkSeverity.CRITICAL)
    before_start = evaluate_action(
        run=run,
        profile=p,
        now=p.contest_start_at - timedelta(seconds=1),
        action=ActionCategory.COMPUTE,
        item=item,
    )
    assert before_start.decision is GuardVerdict.BLOCK
    assert before_start.reason_code == "contest_not_started"
    before_model = evaluate_action(
        run=run, profile=p, now=p.model_freeze_at - timedelta(seconds=1),
        action=ActionCategory.MODEL_MUTATION, item=item,
    )
    assert before_model.decision is GuardVerdict.ALLOW
    at_model = evaluate_action(
        run=run, profile=p, now=p.model_freeze_at,
        action=ActionCategory.MODEL_MUTATION, item=item,
    )
    assert at_model.decision is GuardVerdict.BLOCK
    at_evidence = evaluate_action(
        run=run, profile=p, now=p.evidence_freeze_at,
        action=ActionCategory.COMPUTE, item=item,
    )
    assert at_evidence.decision is GuardVerdict.ALLOW  # P0 existing-capability repair
    ordinary = evaluate_action(
        run=run,
        profile=p,
        now=p.evidence_freeze_at,
        action=ActionCategory.COMPUTE,
        item=work_item(
            "optional",
            capability=ResolutionClass.REQUIRES_NEW_CAPABILITY,
            action=ActionCategory.COMPUTE,
        ),
    )
    assert ordinary.decision is GuardVerdict.BLOCK
    at_paper = evaluate_action(
        run=run, profile=p, now=p.paper_freeze_at,
        action=ActionCategory.PAPER_MUTATION, item=item,
    )
    assert at_paper.decision is GuardVerdict.ALLOW
    polish = evaluate_action(
        run=run, profile=p, now=p.paper_freeze_at,
        action=ActionCategory.PAPER_MUTATION, item=work_item("polish"),
    )
    assert polish.decision is GuardVerdict.BLOCK
    at_stop = evaluate_action(
        run=run, profile=p, now=p.work_stop_at,
        action=ActionCategory.PAPER_MUTATION, item=item,
    )
    assert at_stop.decision is GuardVerdict.BLOCK
    read_only = evaluate_action(
        run=run, profile=p, now=p.submission_deadline_at,
        action=ActionCategory.READ_ONLY,
    )
    assert read_only.decision is GuardVerdict.ALLOW
    reasons = readiness_reasons(
        run=run, profile=p, now=p.work_stop_at, judge_status="revise",
        number_audit_ok=True, math_audit_ok=True,
    )
    assert "scientific_gate_not_pass" in reasons
    assert "model_freeze_missing" in reasons


def test_blocked_replay_is_idempotent_and_stays_blocked(tmp_path: Path) -> None:
    paths = controller_project(tmp_path)
    clock = FrozenClock(START)
    c = CompetitionController.initialize(
        paths, profile(), fixture(), clock=clock, run_id="ctrl-test"
    )
    source_release = sha256_file(paths.root / c.run.source_release)
    source_evidence = sha256_file(paths.verified_numbers_file)
    times = [
        START,
        START + timedelta(hours=6),
        START + timedelta(hours=12),
        START + timedelta(days=1, hours=12),
        START + timedelta(days=2, hours=12),
        START + timedelta(days=3),
        START + timedelta(days=3, hours=2),
    ]
    queue = c.evaluate_queue(START)
    for moment in times:
        clock.set(moment)
        queue = c.tick()
        if moment == START + timedelta(hours=12):
            assert queue.blocked[0].startswith("crit-")
            assert "crit-a1" in queue.allowed
            assert "review-major" in queue.allowed
        if moment == START + timedelta(hours=6):
            assert "optional-polish" in queue.cancelled
        if moment == START + timedelta(days=2, hours=12):
            assert "crit-a1" in queue.allowed
        if moment >= START + timedelta(days=3):
            assert queue.allowed == []
            assert c.run.work_stopped
            assert c.run.scientific_gate_failed
            assert c.run.state is not CompetitionControllerState.READY_FOR_SUBMISSION_CHECKS
        if moment >= START + timedelta(days=3, hours=2):
            assert c.run.state is CompetitionControllerState.CLOSED
            assert c.run.deadline_expired
    assert sha256_file(paths.root / c.run.source_release) == source_release
    assert sha256_file(paths.verified_numbers_file) == source_evidence
    assert c.run.model_freeze_hash and c.run.evidence_freeze_hash and c.run.paper_freeze_hash
    assert c.run.state_history.index(CompetitionControllerState.CLOSED) > c.run.state_history.index(
        CompetitionControllerState.PAPER_FROZEN
    )
    decision_lines = [
        json.loads(line)
        for line in c.decisions_file.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        line["reason_code"] == "work_stopped"
        and str(line["now"]).startswith("2026-01-04T00:00:00")
        for line in decision_lines
    )
    assert any(
        line["reason_code"] == "scope_reduction"
        and line["work_item_id"] == "optional-polish"
        for line in decision_lines
    )
    for required in (
        CompetitionControllerState.MODEL_FROZEN,
        CompetitionControllerState.EVIDENCE_FROZEN,
        CompetitionControllerState.PAPER_FROZEN,
    ):
        assert required in c.run.state_history
    # A cannot-fix scientific blocker stays blocked, never promoted by scope reduction.
    assert "crit-a2" in queue.blocked
    assert "crit-a2" not in queue.cancelled
    # Rerunning the final tick must not duplicate freeze records or change hashes.
    model_hash = c.run.model_freeze_hash
    cache = c.run.state_history.copy()
    clock.set(START + timedelta(days=3, hours=2))
    c.tick()
    assert c.run.model_freeze_hash == model_hash
    assert c.run.state_history == cache
    checkpoint_a = c.checkpoint()
    checkpoint_b = c.checkpoint()
    assert checkpoint_a.checkpoint_hash == checkpoint_b.checkpoint_hash
    assert checkpoint_a.checkpoint_id == checkpoint_b.checkpoint_id


def test_pass_fixture_reaches_ready_then_closed(tmp_path: Path) -> None:
    paths = controller_project(tmp_path, judge="pass")
    clock = FrozenClock(START)
    c = CompetitionController.initialize(
        paths,
        profile(),
        fixture(judge="pass", include_optional=False, include_blockers=False),
        clock=clock,
        run_id="ctrl-pass",
    )
    for moment in [
        START + timedelta(hours=6),
        START + timedelta(hours=12),
        START + timedelta(days=1, hours=12),
        START + timedelta(days=2, hours=12),
        START + timedelta(days=3),
    ]:
        clock.set(moment)
        c.tick()
    assert c.run.state is CompetitionControllerState.READY_FOR_SUBMISSION_CHECKS
    clock.set(START + timedelta(days=3, hours=2))
    c.tick()
    assert c.run.state is CompetitionControllerState.CLOSED
    assert c.run.scientific_gate_failed is False


def test_freeze_tamper_and_artifact_change_are_detected(tmp_path: Path) -> None:
    paths = controller_project(tmp_path)
    clock = FrozenClock(START + timedelta(hours=12))
    c = CompetitionController.initialize(
        paths, profile(), fixture(), clock=clock, run_id="ctrl-tamper"
    )
    c.tick()
    model = c.freeze_model()
    assert model.freeze_hash
    freeze_file = c.freezes_dir / f"{model.freeze_id}.json"
    original = freeze_file.read_text(encoding="utf-8")
    freeze_file.write_text(original.replace("exp-forecast", "exp-tampered"), encoding="utf-8")
    with pytest.raises(ControllerBlocked):
        c._load_freeze(model.freeze_id)
    freeze_file.write_text(original, encoding="utf-8")
    # Evidence and paper artifacts are frozen; changing the report is detected read-only.
    clock.set(START + timedelta(days=1, hours=12))
    c.tick()
    assert isinstance(c._load_freeze(c.run.evidence_freeze_id or ""), EvidenceFreezeRecord)
    clock.set(START + timedelta(days=2, hours=12))
    c.tick()
    report = paths.root / "paper" / "competition" / "report.md"
    report.write_text("changed after freeze\n", encoding="utf-8")
    verification = c.verify_artifacts()
    assert verification["ok"] is False
    mismatches = verification["mismatches"]
    assert isinstance(mismatches, list)
    assert "paper_report_changed" in mismatches
    # Freeze hash itself has not been silently updated.
    paper = c._load_freeze(c.run.paper_freeze_id or "")
    assert paper.freeze_hash == c.run.paper_freeze_hash


def test_freeze_hash_sensitivity() -> None:
    from mmos.competition.controller_schemas import ModelFreezeRecord

    record = ModelFreezeRecord(
        freeze_id="model-freeze-r1",
        competition_run_id="r",
        frozen_at=START,
        primary_model_refs=["exp-a"],
        source_project_id="p",
        artifact_hashes=["a" * 64],
        freeze_hash="",
    )
    first = _hash_model(record, exclude=("freeze_hash",))
    changed = record.model_copy(update={"primary_model_refs": ["exp-b"]})
    second = _hash_model(changed, exclude=("freeze_hash",))
    assert first != second


def test_human_can_cancel_optional_but_not_scientific_blocker(tmp_path: Path) -> None:
    from mmos.competition.controller_schemas import HumanDecisionAction

    paths = controller_project(tmp_path)
    clock = FrozenClock(START + timedelta(hours=1))
    c = CompetitionController.initialize(
        paths, profile(), fixture(), clock=clock, run_id="ctrl-human"
    )
    c.tick()
    decision = c.record_human_decision(
        HumanDecisionAction.CANCEL_WORK_ITEM,
        "review-major",
        "optional review item is out of scope",
    )
    assert decision.status == "applied"
    queue = c.evaluate_queue(clock.now())
    assert "review-major" in queue.cancelled
    with pytest.raises(ControllerBlocked) as exc:
        c.record_human_decision(
            HumanDecisionAction.CANCEL_WORK_ITEM,
            "crit-a1",
            "try to cancel a scientific blocker",
        )
    assert exc.value.reason_code == "cannot_override_scientific_critical"


def test_freeze_hash_sensitivity_for_evidence_and_paper() -> None:
    from mmos.competition.controller_schemas import PaperFreezeRecord

    evidence = EvidenceFreezeRecord(
        freeze_id="evidence-freeze-r1",
        competition_run_id="r",
        frozen_at=START,
        evidence_ids=["ev-a"],
        evidence_catalog_hash="a" * 64,
        source_release_hash="b" * 64,
        validation_summary="ok",
        freeze_hash="",
    )
    evidence_hash = _hash_model(evidence, exclude=("freeze_hash",))
    assert evidence_hash != _hash_model(
        evidence.model_copy(update={"evidence_ids": ["ev-b"]}),
        exclude=("freeze_hash",),
    )
    paper = PaperFreezeRecord(
        freeze_id="paper-freeze-r1",
        competition_run_id="r",
        paper_run_id="p",
        report_path="paper/report.md",
        report_hash="c" * 64,
        evidence_freeze_id="evidence-freeze-r1",
        evidence_freeze_hash="d" * 64,
        number_audit_status="passed",
        math_audit_status="passed",
        red_team_status="pass",
        judge_status="pass",
        frozen_at=START,
        freeze_hash="",
    )
    paper_hash = _hash_model(paper, exclude=("freeze_hash",))
    assert paper_hash != _hash_model(
        paper.model_copy(update={"report_hash": "e" * 64}),
        exclude=("freeze_hash",),
    )


def test_checkpoint_hash_changes_with_freeze_state(tmp_path: Path) -> None:
    paths = controller_project(tmp_path)
    clock = FrozenClock(START)
    c = CompetitionController.initialize(
        paths, profile(), fixture(), clock=clock, run_id="ctrl-checkpoint"
    )
    c.tick()
    before = c.checkpoint()
    clock.set(START + timedelta(hours=12))
    c.tick()
    after = c.checkpoint()
    assert before.checkpoint_hash != after.checkpoint_hash
    assert after.model_freeze_hash is not None
    assert before.model_freeze_hash is None


def test_dependency_is_deferred_not_executed() -> None:
    p = profile()
    run = CompetitionControllerRun(
        competition_run_id="r",
        profile_id="p",
        profile_hash="h",
        source_project_id="s",
        source_paper_run_id="paper",
        source_release="release.json",
        source_release_hash="h",
        source_verified_numbers_hash="v",
        state=CompetitionControllerState.ACTIVE_EXPLORATION,
        state_history=[CompetitionControllerState.ACTIVE_EXPLORATION],
        created_at=START,
        updated_at=START,
    )
    item = work_item(
        "dependent",
        dependencies=["missing"],
        action=ActionCategory.COMPUTE,
    )
    decision = evaluate_action(
        run=run,
        profile=p,
        now=START,
        action=ActionCategory.COMPUTE,
        item=item,
        dependencies_met=False,
    )
    assert decision.decision is GuardVerdict.DEFER
    assert decision.reason_code == "dependency_pending"


def test_invalid_request_is_blocked() -> None:
    p = profile()
    run = CompetitionControllerRun(
        competition_run_id="r",
        profile_id="p",
        profile_hash="h",
        source_project_id="s",
        source_paper_run_id="paper",
        source_release="release.json",
        source_release_hash="h",
        source_verified_numbers_hash="v",
        state=CompetitionControllerState.ACTIVE_EXPLORATION,
        state_history=[CompetitionControllerState.ACTIVE_EXPLORATION],
        created_at=START,
        updated_at=START,
    )
    item = work_item(
        "invalid",
        capability=ResolutionClass.INVALID,
        action=ActionCategory.COMPUTE,
    )
    decision = evaluate_action(
        run=run, profile=p, now=START, action=ActionCategory.COMPUTE, item=item
    )
    assert decision.decision is GuardVerdict.BLOCK
    assert decision.reason_code == "invalid_or_unjustified_request"


def test_real_blocked_fixture_has_critical_first() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths(
        root=root
        / "projects"
        / "mixed-forecast-plan-sim-20261116d-20261116-828a5b"
    )
    if not paths.root.exists():
        pytest.skip("canonical Phase 22 fixture is not available")
    loaded = load_blocked_fixture(paths, "comp-b0e30df9-8363")
    assert loaded.judge_status == "revise"
    assert loaded.number_audit_ok and loaded.math_audit_ok
    blockers = [item for item in loaded.items if item.blocking_release]
    assert len(blockers) == 2
    ordered = sorted(loaded.items, key=lambda item: priority_key(item, {}))
    assert {item.work_item_id for item in ordered[:2]} == {item.work_item_id for item in blockers}
    assert any(
        item.capability_status is ResolutionClass.REQUIRES_NEW_CAPABILITY
        for item in loaded.items
    )
    assert any(
        item.capability_status is ResolutionClass.REQUIRES_HUMAN_DECISION
        for item in loaded.items
    )
