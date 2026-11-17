from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmos.competition.augmentation import load_registered_evidence
from mmos.competition.clock import (
    CompetitionClock,
    SystemClock,
    ensure_aware,
)
from mmos.competition.controller_schemas import (
    STATE_ORDER,
    ActionCategory,
    ClaimCriticality,
    CompetitionBudgetLedger,
    CompetitionCheckpoint,
    CompetitionControllerRun,
    CompetitionControllerState,
    CompetitionFixtureRecord,
    CompetitionTimingProfile,
    CompetitionWorkItem,
    DeadlineGuardDecision,
    EvidenceFreezeRecord,
    GuardVerdict,
    HumanDecisionAction,
    HumanDecisionRecord,
    ModelFreezeRecord,
    PaperFreezeRecord,
    ScopeReductionDecision,
    TimelineEvent,
    WorkItemKind,
    WorkItemStatus,
    WorkQueue,
    WorkQueueEntry,
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
    load_project_config,
    sha256_file,
)
from mmos.evidence.store import load_evidence_index
from mmos.schemas.competition import (
    CompetitionExperimentManifest,
    CompetitionPaperRun,
    MathSeverity,
    RequestExecution,
    RequestResolution,
    ResolutionClass,
)
from mmos.schemas.review import RedTeamReview
from mmos.schemas.states import ProjectState


class CompetitionControllerError(RuntimeError):
    pass


class ControllerBlocked(CompetitionControllerError):
    def __init__(self, reason_code: str, reason: str) -> None:
        super().__init__(f"{reason_code}: {reason}")
        self.reason_code = reason_code
        self.reason = reason


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_payload(payload: object, *, exclude: tuple[str, ...] = ()) -> str:
    data = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for key in exclude:
            data.pop(key, None)
    return hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()


def _hash_model(model: Any, *, exclude: tuple[str, ...] = ()) -> str:
    return _hash_payload(model.model_dump(mode="json"), exclude=exclude)


def _utc(value: datetime) -> datetime:
    return ensure_aware(value, "datetime").astimezone(UTC)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise CompetitionControllerError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_run_dir(paths: ProjectPaths, run_id: str) -> Path:
    current = paths.root / "paper" / "competition"
    run_file = current / "run.json"
    if run_file.exists():
        payload = _read_json(run_file)
        if payload.get("run_id") == run_id:
            return current
    history = current / "history" / run_id
    if (history / "run.json").exists():
        return history
    raise CompetitionControllerError(f"competition paper run not found: {run_id}")


def _find_augmentation(paths: ProjectPaths, paper_run_id: str) -> dict[str, Any] | None:
    root = paths.root / "paper" / "competition" / "augmentation"
    if not root.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for run_file in sorted(root.glob("*/augmentation_run.json")):
        payload = _read_json(run_file)
        error = str(payload.get("new_paper_error") or "")
        if (
            payload.get("new_competition_run_id") == paper_run_id
            or paper_run_id in error
            or payload.get("source_competition_run_id") == paper_run_id
        ):
            candidates.append(payload)
    return candidates[-1] if candidates else None


def _audit_ok(paper_dir: Path) -> tuple[bool, bool, str, str]:
    number_ok = True
    number_file = paper_dir / "number_audit.json"
    if number_file.exists():
        number_ok = not bool(_read_json(number_file).get("unverified"))
    else:
        number_ok = False
    math_ok = True
    math_file = paper_dir / "math_audit.json"
    if math_file.exists():
        issues = _read_json(math_file).get("issues", [])
        math_ok = not any(
            issue.get("severity") == MathSeverity.CRITICAL.value for issue in issues
        )
    else:
        math_ok = False
    return number_ok, math_ok, "passed" if number_ok else "failed", (
        "passed" if math_ok else "critical"
    )


def _evidence_catalog_hash(
    paths: ProjectPaths, source_paper_run_id: str | None = None
) -> str:
    index = load_evidence_index(paths)
    payload = [
        [record.evidence_id, record.key, record.display_value, record.unit]
        for record in sorted(index.records.values(), key=lambda item: item.evidence_id)
    ]
    if source_paper_run_id:
        augmentation = _find_augmentation(paths, source_paper_run_id)
        if augmentation:
            aug_id = str(augmentation.get("augmentation_id", ""))
            if aug_id:
                extra, _ = load_registered_evidence(paths, aug_id)
                payload.extend(
                    [record.evidence_id, record.key, record.display_value, record.unit]
                    for record in sorted(extra, key=lambda item: item.evidence_id)
                )
    return _hash_payload(payload)


def _capability_from_fix(fix: str) -> ResolutionClass:
    lowered = fix.lower()
    if lowered.startswith(("state explicitly", "delete ", "footnote ")):
        return ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
    if any(
        marker in lowered
        for marker in (
            "re-run",
            "rerun",
            "run and report",
            "execute",
            "bootstrap",
            "mean-cent",
            "mean cent",
            "centered",
            "comparator",
            "seasonal",
            "ordinary-least",
            "ols",
            "alternative",
            "sensitivity",
            "new model",
        )
    ):
        return ResolutionClass.REQUIRES_NEW_CAPABILITY
    return ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY


def load_blocked_fixture(paths: ProjectPaths, paper_run_id: str) -> CompetitionFixtureRecord:
    """Build the controller fixture from real Phase 22 artifacts; never rewrite them."""
    paper_dir = _paper_run_dir(paths, paper_run_id)
    run = CompetitionPaperRun.model_validate_json(
        (paper_dir / "run.json").read_text(encoding="utf-8")
    )
    review_path = (
        paths.root / run.review_path
        if run.review_path
        else paths.reviews_dir / f"redteam_{paper_run_id}.json"
    )
    review = RedTeamReview.model_validate_json(review_path.read_text(encoding="utf-8"))
    number_ok, math_ok, _, _ = _audit_ok(paper_dir)
    created_at = _utc(review.created_at)
    items: list[CompetitionWorkItem] = []

    for attack in review.attacks.attacks:
        critical = attack.severity.value == "critical"
        capability = (
            _capability_from_fix(attack.required_fix)
            if critical
            else ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
        )
        if critical:
            kind = WorkItemKind.SCIENTIFIC_CRITICAL
            action = (
                ActionCategory.COMPUTE
                if capability is ResolutionClass.REQUIRES_NEW_CAPABILITY
                else ActionCategory.PAPER_MUTATION
            )
            severity = WorkSeverity.CRITICAL
            claim = ClaimCriticality.CORE
            blocking = True
            p0 = True
        elif attack.severity.value == "major":
            kind = WorkItemKind.REVIEW_ISSUE
            action = ActionCategory.PAPER_MUTATION
            severity = WorkSeverity.MAJOR
            claim = ClaimCriticality.SUPPORTING
            blocking = False
            p0 = False
        else:
            kind = WorkItemKind.OPTIONAL_IMPROVEMENT
            action = ActionCategory.PAPER_MUTATION
            severity = WorkSeverity.MINOR
            claim = ClaimCriticality.OPTIONAL
            blocking = False
            p0 = False
        items.append(
            CompetitionWorkItem(
                work_item_id=f"crit-{attack.attack_id}"
                if critical
                else f"review-{attack.attack_id}",
                kind=kind,
                source_type="review_issue",
                source_id=attack.attack_id,
                title=f"{attack.category.value}: {attack.target[:120]}",
                description=f"{attack.argument} Required fix: {attack.required_fix}",
                blocking_release=blocking,
                scientific_severity=severity,
                claim_criticality=claim,
                capability_status=capability,
                expected_runtime_class="minutes" if action is ActionCategory.COMPUTE else None,
                created_at=created_at,
                action_category=action,
                p0_repair=p0,
            )
        )

    augmentation = _find_augmentation(paths, paper_run_id)
    if augmentation is not None:
        source_run_id = str(augmentation.get("source_competition_run_id") or "")
        source_dir = paper_dir
        if source_run_id and source_run_id != paper_run_id:
            source_dir = paths.root / "paper" / "competition" / "history" / source_run_id
        manifest_path = source_dir / "experiment_requests.json"
        resolutions: list[RequestResolution] = [
            RequestResolution.model_validate(item)
            for item in augmentation.get("resolutions", [])
        ]
        executions: list[RequestExecution] = [
            RequestExecution.model_validate(item)
            for item in augmentation.get("executions", [])
        ]
        if manifest_path.exists():
            manifest = CompetitionExperimentManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            resolution_by_id = {item.request_id: item for item in resolutions}
            execution_by_id = {item.request_id: item for item in executions}
            manifest_created = _utc(manifest.created_at)
            for request in manifest.requests:
                resolution = resolution_by_id.get(request.request_id)
                execution = execution_by_id.get(request.request_id)
                completed = bool(
                    execution
                    and execution.status.value in {"validated", "evidence_registered"}
                )
                items.append(
                    CompetitionWorkItem(
                        work_item_id=f"req-{request.request_id}",
                        kind=WorkItemKind.EXPERIMENT_REQUEST,
                        source_type="experiment_request",
                        source_id=request.request_id,
                        title=request.question[:140],
                        description=request.rationale,
                        blocking_release=False,
                        scientific_severity=WorkSeverity.MAJOR,
                        claim_criticality=ClaimCriticality.SUPPORTING,
                        capability_status=(
                            resolution.classification
                            if resolution
                            else ResolutionClass.REQUIRES_HUMAN_DECISION
                        ),
                        expected_runtime_class="minutes",
                        created_at=manifest_created,
                        action_category=ActionCategory.COMPUTE,
                        status=(
                            WorkItemStatus.COMPLETED
                            if completed
                            else WorkItemStatus.PENDING
                        ),
                    )
                )

    return CompetitionFixtureRecord(
        source_paper_run_id=paper_run_id,
        judge_status=review.judge.overall.value,
        number_audit_ok=number_ok,
        math_audit_ok=math_ok,
        items=items,
    )


@dataclass
class CompetitionController:
    paths: ProjectPaths
    profile: CompetitionTimingProfile
    run: CompetitionControllerRun
    fixture: CompetitionFixtureRecord
    clock: CompetitionClock
    run_dir: Path

    @property
    def controller_file(self) -> Path:
        return self.run_dir / "controller.json"

    @property
    def profile_file(self) -> Path:
        return self.run_dir / "profile.json"

    @property
    def fixture_file(self) -> Path:
        return self.run_dir / "fixture.json"

    @property
    def queue_file(self) -> Path:
        return self.run_dir / "work_queue.json"

    @property
    def budget_file(self) -> Path:
        return self.run_dir / "budget.json"

    @property
    def timeline_file(self) -> Path:
        return self.run_dir / "timeline.jsonl"

    @property
    def human_decisions_file(self) -> Path:
        return self.run_dir / "human_decisions.jsonl"

    @property
    def scope_reductions_file(self) -> Path:
        return self.run_dir / "scope_reductions.jsonl"

    @property
    def decisions_file(self) -> Path:
        return self.run_dir / "decisions.jsonl"

    @property
    def freezes_dir(self) -> Path:
        return self.run_dir / "freezes"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @classmethod
    def initialize(
        cls,
        paths: ProjectPaths,
        profile: CompetitionTimingProfile,
        fixture: CompetitionFixtureRecord,
        *,
        clock: CompetitionClock | None = None,
        run_id: str | None = None,
    ) -> CompetitionController:
        config = load_project_config(paths)
        if config.state is not ProjectState.RELEASED:
            raise CompetitionControllerError(
                "competition controller requires a RELEASED source project; "
                f"current state is {config.state.value}"
            )
        profile_hash = _hash_model(profile)
        chosen = run_id or f"ctrl-{profile_hash[:8]}-{secrets.token_hex(2)}"
        run_dir = paths.root / "competition" / "runs" / chosen
        if run_dir.exists():
            raise CompetitionControllerError(f"competition controller run exists: {chosen}")
        run_dir.mkdir(parents=True)
        (run_dir / "freezes").mkdir()
        (run_dir / "checkpoints").mkdir()
        now = _utc((clock or SystemClock()).now())

        paper_dir = _paper_run_dir(paths, fixture.source_paper_run_id)
        paper_run = CompetitionPaperRun.model_validate_json(
            (paper_dir / "run.json").read_text(encoding="utf-8")
        )
        release_path = paths.root / paper_run.source_release
        if not release_path.exists():
            raise CompetitionControllerError("source release is missing")
        release_hash = sha256_file(release_path)
        if release_hash != paper_run.source_release_hash:
            raise CompetitionControllerError(
                "source release hash does not match the competition paper run"
            )
        verified_hash = (
            sha256_file(paths.verified_numbers_file)
            if paths.verified_numbers_file.exists()
            else ""
        )
        run = CompetitionControllerRun(
            competition_run_id=chosen,
            profile_id=profile.profile_id,
            profile_hash=profile_hash,
            source_project_id=paths.root.name,
            source_paper_run_id=fixture.source_paper_run_id,
            source_release=paper_run.source_release,
            source_release_hash=release_hash,
            source_verified_numbers_hash=verified_hash,
            state=CompetitionControllerState.CONFIGURED,
            state_history=[CompetitionControllerState.CONFIGURED],
            unresolved_blockers=[
                item.work_item_id
                for item in fixture.items
                if item.blocking_release
                and item.scientific_severity is WorkSeverity.CRITICAL
            ],
            created_at=now,
            updated_at=now,
        )
        controller = cls(
            paths=paths,
            profile=profile,
            run=run,
            fixture=fixture,
            clock=clock or SystemClock(),
            run_dir=run_dir,
        )
        atomic_write_text(controller.profile_file, profile.model_dump_json(indent=2) + "\n")
        atomic_write_text(controller.fixture_file, fixture.model_dump_json(indent=2) + "\n")
        controller._write_run()
        controller._append_event(
            "CompetitionConfigured",
            now,
            {"profile_id": profile.profile_id, "source_paper_run_id": fixture.source_paper_run_id},
        )
        controller._write_queue(controller.evaluate_queue(now))
        return controller

    @classmethod
    def load(
        cls,
        paths: ProjectPaths,
        run_id: str,
        *,
        clock: CompetitionClock | None = None,
    ) -> CompetitionController:
        run_dir = paths.root / "competition" / "runs" / run_id
        if not run_dir.exists():
            raise CompetitionControllerError(f"competition controller run not found: {run_id}")
        profile = CompetitionTimingProfile.model_validate_json(
            (run_dir / "profile.json").read_text(encoding="utf-8")
        )
        fixture = CompetitionFixtureRecord.model_validate_json(
            (run_dir / "fixture.json").read_text(encoding="utf-8")
        )
        run = CompetitionControllerRun.model_validate_json(
            (run_dir / "controller.json").read_text(encoding="utf-8")
        )
        profile_hash = _hash_model(profile)
        if profile_hash != run.profile_hash:
            raise ControllerBlocked(
                "profile_changed", "timing profile does not match the controller run"
            )
        controller = cls(
            paths=paths,
            profile=profile,
            run=run,
            fixture=fixture,
            clock=clock or SystemClock(),
            run_dir=run_dir,
        )
        controller._verify_source()
        return controller

    def _write_run(self) -> None:
        atomic_write_text(
            self.controller_file, self.run.model_dump_json(indent=2) + "\n"
        )

    def _save(self) -> None:
        self._write_run()
        atomic_write_text(
            self.fixture_file, self.fixture.model_dump_json(indent=2) + "\n"
        )

    def _verify_source(self) -> None:
        release = self.paths.root / self.run.source_release
        if not release.exists():
            raise ControllerBlocked("source_release_missing", "source release is missing")
        if sha256_file(release) != self.run.source_release_hash:
            raise ControllerBlocked(
                "source_release_changed",
                "source release changed; controller state is stale",
            )
        verified_hash = (
            sha256_file(self.paths.verified_numbers_file)
            if self.paths.verified_numbers_file.exists()
            else ""
        )
        if verified_hash != self.run.source_verified_numbers_hash:
            raise ControllerBlocked(
                "source_evidence_changed",
                "verified evidence changed; controller state is stale",
            )

    def _advance(self, state: CompetitionControllerState, now: datetime) -> None:
        current_index = STATE_ORDER.index(self.run.state)
        target_index = STATE_ORDER.index(state)
        if target_index <= current_index:
            return
        while current_index < target_index:
            current_index += 1
            next_state = STATE_ORDER[current_index]
            self.run.state = next_state
            self.run.state_history.append(next_state)
            self._append_event("StateChanged", now, {"state": next_state.value})

    def _append_event(
        self, event_type: str, now: datetime, payload: dict[str, object]
    ) -> None:
        event = TimelineEvent(
            event_type=event_type,
            occurred_at=now,
            state=self.run.state,
            payload=payload,
        )
        line = event.model_dump_json()
        existing = (
            self.timeline_file.read_text(encoding="utf-8").splitlines()
            if self.timeline_file.exists()
            else []
        )
        if line in existing:
            return
        with self.timeline_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def _next_milestone(self, now: datetime) -> tuple[str | None, datetime | None]:
        for name, value in self.profile.milestones():
            if now < value:
                return name, value
        return None, None

    def evaluate_queue(self, now: datetime) -> WorkQueue:
        completed = {
            item.work_item_id
            for item in self.fixture.items
            if item.status is WorkItemStatus.COMPLETED
        }
        dependent_counts: dict[str, int] = {}
        for item in self.fixture.items:
            for dependency in item.dependencies:
                dependent_counts[dependency] = dependent_counts.get(dependency, 0) + 1
        ordered: list[CompetitionWorkItem] = sorted(
            self.fixture.items,
            key=lambda item: priority_key(item, dependent_counts),
        )
        scope_decisions = auto_scope_reductions(
            items=self.fixture.items,
            run=self.run,
            profile=self.profile,
            now=now,
        )
        scope_targets = {
            target
            for decision in scope_decisions
            for target in decision.target_work_item_ids
        }
        entries: list[WorkQueueEntry] = []
        allowed: list[str] = []
        blocked: list[str] = []
        deferred: list[str] = []
        cancelled: list[str] = []
        completed_ids: list[str] = []
        for item in ordered:
            if item.status is WorkItemStatus.COMPLETED or item.work_item_id in completed:
                decision = DeadlineGuardDecision(
                    decision=GuardVerdict.ALLOW,
                    reason_code="completed",
                    reason="work item is already complete",
                    competition_state=self.run.state,
                    now=now,
                    action_category=item.action_category,
                    work_item_id=item.work_item_id,
                )
                status = WorkItemStatus.COMPLETED
                completed_ids.append(item.work_item_id)
            elif item.status is WorkItemStatus.CANCELLED:
                decision = DeadlineGuardDecision(
                    decision=GuardVerdict.CANCEL,
                    reason_code="human_cancelled",
                    reason="cancelled by explicit human decision",
                    competition_state=self.run.state,
                    now=now,
                    action_category=item.action_category,
                    work_item_id=item.work_item_id,
                )
                status = WorkItemStatus.CANCELLED
                cancelled.append(item.work_item_id)
            elif item.work_item_id in scope_targets:
                decision = DeadlineGuardDecision(
                    decision=GuardVerdict.CANCEL,
                    reason_code="scope_reduction",
                    reason="cancelled by deterministic scope reduction",
                    competition_state=self.run.state,
                    now=now,
                    action_category=item.action_category,
                    work_item_id=item.work_item_id,
                )
                status = WorkItemStatus.CANCELLED
                cancelled.append(item.work_item_id)
            else:
                dependencies_met = all(
                    dependency in completed
                    for dependency in item.dependencies
                )
                decision = evaluate_action(
                    run=self.run,
                    profile=self.profile,
                    now=now,
                    action=item.action_category,
                    item=item,
                    dependencies_met=dependencies_met,
                )
                if decision.decision is GuardVerdict.ALLOW:
                    status = WorkItemStatus.ALLOWED
                    allowed.append(item.work_item_id)
                elif decision.decision is GuardVerdict.DEFER:
                    status = WorkItemStatus.DEFERRED
                    deferred.append(item.work_item_id)
                elif decision.decision is GuardVerdict.CANCEL:
                    status = WorkItemStatus.CANCELLED
                    cancelled.append(item.work_item_id)
                else:
                    status = WorkItemStatus.BLOCKED
                    blocked.append(item.work_item_id)
            entries.append(
                WorkQueueEntry(
                    work_item_id=item.work_item_id,
                    status=status,
                    decision=decision,
                    priority_key=[str(value) for value in priority_key(item, dependent_counts)],
                )
            )
        counts = {
            "planned": len(self.fixture.items),
            "allowed": len(allowed),
            "blocked": len(blocked),
            "deferred": len(deferred),
            "cancelled": len(cancelled),
            "completed": len(completed_ids),
        }
        next_name, _next_time = self._next_milestone(now)
        queue = WorkQueue(
            competition_run_id=self.run.competition_run_id,
            state=self.run.state,
            now=now,
            next_milestone=next_name,
            remaining_seconds=max(
                0.0, (self.profile.work_stop_at - now).total_seconds()
            ),
            ordered_work_items=entries,
            allowed=allowed,
            blocked=blocked,
            deferred=deferred,
            cancelled=cancelled,
            completed=completed_ids,
            counts=counts,
        )
        queue_hash = _hash_model(queue, exclude=("queue_hash",))
        queue = queue.model_copy(update={"queue_hash": queue_hash})
        self.run.queue_hash = queue_hash
        self.run.unresolved_blockers = [
            item.work_item_id
            for item in self.fixture.items
            if item.blocking_release
            and item.scientific_severity is WorkSeverity.CRITICAL
            and item.work_item_id not in completed_ids
            and item.work_item_id not in cancelled
        ]
        for decision in scope_decisions:
            self._append_event(
                "ScopeReduced",
                now,
                {
                    "decision_id": decision.decision_id,
                    "targets": decision.target_work_item_ids,
                    "action": decision.action,
                },
            )
        return queue

    def _write_queue(self, queue: WorkQueue) -> None:
        atomic_write_text(self.queue_file, queue.model_dump_json(indent=2) + "\n")
        with self.decisions_file.open("a", encoding="utf-8", newline="\n") as handle:
            for entry in queue.ordered_work_items:
                handle.write(
                    json.dumps(
                        {
                            "now": queue.now.isoformat(),
                            "state": queue.state.value,
                            "work_item_id": entry.work_item_id,
                            "status": entry.status.value,
                            "reason_code": entry.decision.reason_code,
                            "reason": entry.decision.reason,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        next_name, next_time = self._next_milestone(queue.now)
        ledger = CompetitionBudgetLedger(
            competition_run_id=self.run.competition_run_id,
            phase=self.run.state,
            planned_work_items=queue.counts.get("planned", 0),
            completed_work_items=queue.counts.get("completed", 0),
            blocked_work_items=queue.counts.get("blocked", 0),
            elapsed_seconds=max(
                0.0, (queue.now - self.profile.contest_start_at).total_seconds()
            ),
            remaining_seconds=queue.remaining_seconds,
            next_milestone=next_name,
            time_to_next_milestone_seconds=(
                (next_time - queue.now).total_seconds() if next_time else None
            ),
        )
        atomic_write_text(self.budget_file, ledger.model_dump_json(indent=2) + "\n")

    def _model_freeze(
        self, now: datetime, *, revision: int = 1, previous_hash: str | None = None
    ) -> ModelFreezeRecord:
        existing_path = self.freezes_dir / f"model-freeze-r{revision}.json"
        if existing_path.exists():
            existing = ModelFreezeRecord.model_validate_json(
                existing_path.read_text(encoding="utf-8")
            )
            if _hash_model(existing, exclude=("freeze_hash",)) != existing.freeze_hash:
                raise ControllerBlocked(
                    "freeze_hash_mismatch", "existing model freeze was modified"
                )
            return existing
        release = json.loads(
            (self.paths.root / self.run.source_release).read_text(encoding="utf-8")
        )
        experiment_ids = [
            str(value)
            for key, value in release.items()
            if key.endswith("_experiment_id") and isinstance(value, str)
        ]
        artifact_ids = [
            str(value)
            for key, value in release.items()
            if key.endswith("_artifact_id") and isinstance(value, str)
        ]
        artifact_hashes = [
            str(value)
            for key, value in release.items()
            if key.endswith("_artifact_hash") and isinstance(value, str)
        ]
        evidence_ids = sorted(load_evidence_index(self.paths).records)
        record = ModelFreezeRecord(
            freeze_id=f"model-freeze-r{revision}",
            revision=revision,
            competition_run_id=self.run.competition_run_id,
            frozen_at=now,
            primary_model_refs=[
                str(release.get("forecast_experiment_id", "")),
                str(release.get("optimization_experiment_id", "")),
                str(release.get("simulation_experiment_id", "")),
            ],
            source_project_id=self.run.source_project_id,
            experiment_ids=experiment_ids,
            artifact_ids=artifact_ids,
            artifact_hashes=artifact_hashes,
            selection_evidence_ids=evidence_ids,
            freeze_hash="",
        )
        freeze_hash = _hash_model(record, exclude=("freeze_hash",))
        record = record.model_copy(update={"freeze_hash": freeze_hash})
        target = self.freezes_dir / f"{record.freeze_id}.json"
        if previous_hash is not None:
            self._append_event(
                "FreezeRevision",
                now,
                {"kind": "model", "revision": revision, "previous_hash": previous_hash},
            )
        atomic_write_text(target, record.model_dump_json(indent=2) + "\n")
        return record

    def _evidence_freeze(
        self, now: datetime, *, revision: int = 1, previous_hash: str | None = None
    ) -> EvidenceFreezeRecord:
        existing_path = self.freezes_dir / f"evidence-freeze-r{revision}.json"
        if existing_path.exists():
            existing = EvidenceFreezeRecord.model_validate_json(
                existing_path.read_text(encoding="utf-8")
            )
            if _hash_model(existing, exclude=("freeze_hash",)) != existing.freeze_hash:
                raise ControllerBlocked(
                    "freeze_hash_mismatch", "existing evidence freeze was modified"
                )
            return existing
        index = load_evidence_index(self.paths)
        evidence_ids = sorted(index.records)
        augmentation = _find_augmentation(self.paths, self.run.source_paper_run_id)
        augmentation_hashes: list[str] = []
        augmentation_count = 0
        if augmentation:
            aug_id = str(augmentation.get("augmentation_id", ""))
            if aug_id:
                extra, _ = load_registered_evidence(self.paths, aug_id)
                augmentation_count = len(extra)
                evidence_ids = sorted(
                    {*evidence_ids, *(record.evidence_id for record in extra)}
                )
            store = self.paths.root / str(augmentation.get("evidence_store_path", ""))
            if store.exists():
                augmentation_hashes.append(sha256_file(store))
            aug_file = self.paths.root / "paper" / "competition" / "augmentation" / str(
                augmentation.get("augmentation_id", "")
            ) / "augmentation_run.json"
            if aug_file.exists():
                augmentation_hashes.append(sha256_file(aug_file))
        catalog_hash = _evidence_catalog_hash(
            self.paths, self.run.source_paper_run_id
        )
        validation_summary = (
            f"source evidence records: {len(index.records)}; "
            f"augmentation records: {augmentation_count}; "
            f"augmentation registered: {'yes' if augmentation else 'no'}"
        )
        record = EvidenceFreezeRecord(
            freeze_id=f"evidence-freeze-r{revision}",
            revision=revision,
            competition_run_id=self.run.competition_run_id,
            frozen_at=now,
            evidence_ids=evidence_ids,
            evidence_catalog_hash=catalog_hash,
            source_release_hash=self.run.source_release_hash,
            augmentation_hashes=augmentation_hashes,
            validation_summary=validation_summary,
            unresolved_issues=list(self.run.unresolved_blockers),
            freeze_hash="",
        )
        freeze_hash = _hash_model(record, exclude=("freeze_hash",))
        record = record.model_copy(update={"freeze_hash": freeze_hash})
        target = self.freezes_dir / f"{record.freeze_id}.json"
        if previous_hash is not None:
            self._append_event(
                "FreezeRevision",
                now,
                {"kind": "evidence", "revision": revision, "previous_hash": previous_hash},
            )
        atomic_write_text(target, record.model_dump_json(indent=2) + "\n")
        return record

    def _paper_freeze(
        self, now: datetime, *, revision: int = 1, previous_hash: str | None = None
    ) -> PaperFreezeRecord:
        existing_path = self.freezes_dir / f"paper-freeze-r{revision}.json"
        if existing_path.exists():
            existing = PaperFreezeRecord.model_validate_json(
                existing_path.read_text(encoding="utf-8")
            )
            if _hash_model(existing, exclude=("freeze_hash",)) != existing.freeze_hash:
                raise ControllerBlocked(
                    "freeze_hash_mismatch", "existing paper freeze was modified"
                )
            return existing
        paper_dir = _paper_run_dir(self.paths, self.run.source_paper_run_id)
        paper_run = CompetitionPaperRun.model_validate_json(
            (paper_dir / "run.json").read_text(encoding="utf-8")
        )
        report = self.paths.root / paper_run.paper_path
        _, _, number_status, math_status = _audit_ok(paper_dir)
        evidence_id = self.run.evidence_freeze_id or ""
        evidence_hash = self.run.evidence_freeze_hash or ""
        record = PaperFreezeRecord(
            freeze_id=f"paper-freeze-r{revision}",
            revision=revision,
            competition_run_id=self.run.competition_run_id,
            paper_run_id=paper_run.run_id,
            report_path=paper_run.paper_path,
            report_hash=sha256_file(report),
            evidence_freeze_id=evidence_id,
            evidence_freeze_hash=evidence_hash,
            number_audit_status=number_status,
            math_audit_status=math_status,
            red_team_status=paper_run.red_team_overall or "unknown",
            judge_status=self.fixture.judge_status,
            frozen_at=now,
            freeze_hash="",
        )
        freeze_hash = _hash_model(record, exclude=("freeze_hash",))
        record = record.model_copy(update={"freeze_hash": freeze_hash})
        target = self.freezes_dir / f"{record.freeze_id}.json"
        if previous_hash is not None:
            self._append_event(
                "FreezeRevision",
                now,
                {"kind": "paper", "revision": revision, "previous_hash": previous_hash},
            )
        atomic_write_text(target, record.model_dump_json(indent=2) + "\n")
        return record

    def _load_freeze(
        self, freeze_id: str
    ) -> ModelFreezeRecord | EvidenceFreezeRecord | PaperFreezeRecord:
        path = self.freezes_dir / f"{freeze_id}.json"
        if not path.exists():
            raise CompetitionControllerError(f"freeze not found: {freeze_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if freeze_id.startswith("model-freeze"):
            record = ModelFreezeRecord.model_validate(payload)
        elif freeze_id.startswith("evidence-freeze"):
            record = EvidenceFreezeRecord.model_validate(payload)
        else:
            record = PaperFreezeRecord.model_validate(payload)
        expected = _hash_model(record, exclude=("freeze_hash",))
        if expected != record.freeze_hash:
            raise ControllerBlocked(
                "freeze_hash_mismatch", f"freeze record changed: {freeze_id}"
            )
        return record

    def _sync(self, now: datetime) -> None:
        self._verify_source()
        if now < self.profile.contest_start_at:
            return
        if now < self.profile.exploration_end_at:
            self._advance(CompetitionControllerState.ACTIVE_EXPLORATION, now)
        elif now < self.profile.model_freeze_at:
            self._advance(CompetitionControllerState.CANDIDATE_CONVERGENCE, now)
        if now >= self.profile.model_freeze_at and self.run.model_freeze_hash is None:
            self._advance(CompetitionControllerState.MODEL_FROZEN, now)
            record = self._model_freeze(now)
            self.run.model_freeze_id = record.freeze_id
            self.run.model_freeze_hash = record.freeze_hash
            self._append_event(
                "ModelFrozen", now, {"freeze_id": record.freeze_id, "hash": record.freeze_hash}
            )
            self._advance(CompetitionControllerState.EVIDENCE_COMPLETION, now)
        if now >= self.profile.evidence_freeze_at and self.run.evidence_freeze_hash is None:
            self._advance(CompetitionControllerState.EVIDENCE_FROZEN, now)
            record = self._evidence_freeze(now)
            self.run.evidence_freeze_id = record.freeze_id
            self.run.evidence_freeze_hash = record.freeze_hash
            self._append_event(
                "EvidenceFrozen",
                now,
                {"freeze_id": record.freeze_id, "hash": record.freeze_hash},
            )
            self._advance(CompetitionControllerState.PAPER_FINALIZATION, now)
        if now >= self.profile.paper_freeze_at and self.run.paper_freeze_hash is None:
            self._advance(CompetitionControllerState.PAPER_FROZEN, now)
            record = self._paper_freeze(now)
            self.run.paper_freeze_id = record.freeze_id
            self.run.paper_freeze_hash = record.freeze_hash
            self._append_event(
                "PaperFrozen",
                now,
                {"freeze_id": record.freeze_id, "hash": record.freeze_hash},
            )
        if now >= self.profile.work_stop_at and not self.run.work_stopped:
            self.run.work_stopped = True
            reasons = readiness_reasons(
                run=self.run,
                profile=self.profile,
                now=now,
                judge_status=self.fixture.judge_status,
                number_audit_ok=self.fixture.number_audit_ok,
                math_audit_ok=self.fixture.math_audit_ok,
            )
            self._append_event(
                "WorkStopped", now, {"reasons": reasons}
            )
            if reasons:
                self.run.scientific_gate_failed = True
                self.run.requires_human_review = True
            else:
                self._advance(
                    CompetitionControllerState.READY_FOR_SUBMISSION_CHECKS, now
                )
        if now >= self.profile.submission_deadline_at:
            self.run.deadline_expired = True
            if self.run.state is not CompetitionControllerState.CLOSED:
                self.run.state = CompetitionControllerState.CLOSED
                self.run.state_history.append(CompetitionControllerState.CLOSED)
                self._append_event(
                    "StateChanged", now, {"state": CompetitionControllerState.CLOSED.value}
                )
            self._append_event("ControllerClosed", now, {"deadline_expired": True})
        self.run.updated_at = now

    def tick(self) -> WorkQueue:
        now = _utc(self.clock.now())
        self._sync(now)
        queue = self.evaluate_queue(now)
        self._write_queue(queue)
        self._save()
        return queue

    def status(self) -> dict[str, object]:
        now = _utc(self.clock.now())
        queue = self.evaluate_queue(now)
        return {
            "competition_run_id": self.run.competition_run_id,
            "state": self.run.state.value,
            "work_stopped": self.run.work_stopped,
            "scientific_gate_failed": self.run.scientific_gate_failed,
            "deadline_expired": self.run.deadline_expired,
            "requires_human_review": self.run.requires_human_review,
            "judge_status": self.fixture.judge_status,
            "model_freeze_id": self.run.model_freeze_id,
            "model_freeze_hash": self.run.model_freeze_hash,
            "evidence_freeze_id": self.run.evidence_freeze_id,
            "evidence_freeze_hash": self.run.evidence_freeze_hash,
            "paper_freeze_id": self.run.paper_freeze_id,
            "paper_freeze_hash": self.run.paper_freeze_hash,
            "queue": queue.model_dump(mode="json"),
        }

    def freeze_model(
        self, *, revision: int | None = None, reason: str | None = None
    ) -> ModelFreezeRecord:
        now = _utc(self.clock.now())
        self._sync(now)
        if now < self.profile.model_freeze_at:
            raise ControllerBlocked(
                "model_freeze_not_reached", "model_freeze_at has not been reached"
            )
        if now >= self.profile.work_stop_at:
            raise ControllerBlocked(
                "work_stopped", "work_stop_at has passed; no freeze revision is allowed"
            )
        if self.run.model_freeze_hash is not None:
            previous = self._load_freeze(self.run.model_freeze_id or "")
            if revision is None:
                return previous  # type: ignore[return-value]
            if not reason:
                raise ControllerBlocked(
                    "freeze_revision_reason_required",
                    "a freeze revision requires an explicit reason",
                )
            new_revision = revision
            if new_revision <= int(previous.revision):
                raise ControllerBlocked(
                    "freeze_revision_not_increasing",
                    "freeze revision must increase monotonically",
                )
            record = self._model_freeze(
                now, revision=new_revision, previous_hash=previous.freeze_hash
            )
            self._append_event(
                "FreezeInvalidated",
                now,
                {
                    "kind": "model",
                    "previous": previous.freeze_id,
                    "revision": record.freeze_id,
                    "reason": reason,
                },
            )
        else:
            record = self._model_freeze(now, revision=revision or 1)
            self._append_event(
                "ModelFrozen",
                now,
                {"freeze_id": record.freeze_id, "hash": record.freeze_hash},
            )
        self.run.model_freeze_id = record.freeze_id
        self.run.model_freeze_hash = record.freeze_hash
        self._save()
        return record

    def freeze_evidence(
        self, *, revision: int | None = None, reason: str | None = None
    ) -> EvidenceFreezeRecord:
        now = _utc(self.clock.now())
        self._sync(now)
        if now < self.profile.evidence_freeze_at:
            raise ControllerBlocked(
                "evidence_freeze_not_reached", "evidence_freeze_at has not been reached"
            )
        if self.run.model_freeze_hash is None:
            raise ControllerBlocked("model_not_frozen", "freeze the model first")
        if now >= self.profile.work_stop_at:
            raise ControllerBlocked(
                "work_stopped", "work_stop_at has passed; no freeze revision is allowed"
            )
        if self.run.evidence_freeze_hash is not None:
            previous = self._load_freeze(self.run.evidence_freeze_id or "")
            if revision is None:
                return previous  # type: ignore[return-value]
            if not reason:
                raise ControllerBlocked(
                    "freeze_revision_reason_required",
                    "a freeze revision requires an explicit reason",
                )
            new_revision = revision
            if new_revision <= int(previous.revision):
                raise ControllerBlocked(
                    "freeze_revision_not_increasing",
                    "freeze revision must increase monotonically",
                )
            record = self._evidence_freeze(
                now, revision=new_revision, previous_hash=previous.freeze_hash
            )
            self._append_event(
                "FreezeInvalidated",
                now,
                {
                    "kind": "evidence",
                    "previous": previous.freeze_id,
                    "revision": record.freeze_id,
                    "reason": reason,
                },
            )
        else:
            record = self._evidence_freeze(now, revision=revision or 1)
            self._append_event(
                "EvidenceFrozen",
                now,
                {"freeze_id": record.freeze_id, "hash": record.freeze_hash},
            )
        self.run.evidence_freeze_id = record.freeze_id
        self.run.evidence_freeze_hash = record.freeze_hash
        self._save()
        return record

    def freeze_paper(
        self, *, revision: int | None = None, reason: str | None = None
    ) -> PaperFreezeRecord:
        now = _utc(self.clock.now())
        self._sync(now)
        if now < self.profile.paper_freeze_at:
            raise ControllerBlocked(
                "paper_freeze_not_reached", "paper_freeze_at has not been reached"
            )
        if self.run.evidence_freeze_hash is None:
            raise ControllerBlocked("evidence_not_frozen", "freeze evidence first")
        if now >= self.profile.work_stop_at:
            raise ControllerBlocked(
                "work_stopped", "work_stop_at has passed; no freeze revision is allowed"
            )
        if self.run.paper_freeze_hash is not None:
            previous = self._load_freeze(self.run.paper_freeze_id or "")
            if revision is None:
                return previous  # type: ignore[return-value]
            if not reason:
                raise ControllerBlocked(
                    "freeze_revision_reason_required",
                    "a freeze revision requires an explicit reason",
                )
            new_revision = revision
            if new_revision <= int(previous.revision):
                raise ControllerBlocked(
                    "freeze_revision_not_increasing",
                    "freeze revision must increase monotonically",
                )
            record = self._paper_freeze(
                now, revision=new_revision, previous_hash=previous.freeze_hash
            )
            self._append_event(
                "FreezeInvalidated",
                now,
                {
                    "kind": "paper",
                    "previous": previous.freeze_id,
                    "revision": record.freeze_id,
                    "reason": reason,
                },
            )
        else:
            record = self._paper_freeze(now, revision=revision or 1)
            self._append_event(
                "PaperFrozen",
                now,
                {"freeze_id": record.freeze_id, "hash": record.freeze_hash},
            )
        self.run.paper_freeze_id = record.freeze_id
        self.run.paper_freeze_hash = record.freeze_hash
        self._save()
        return record

    def record_human_decision(
        self,
        action: HumanDecisionAction,
        target_id: str,
        rationale: str,
    ) -> HumanDecisionRecord:
        now = _utc(self.clock.now())
        self._sync(now)
        if now >= self.profile.work_stop_at:
            raise ControllerBlocked(
                "work_stopped", "human decisions cannot mutate work after work_stop_at"
            )
        decision = HumanDecisionRecord(
            decision_id=f"human-{action.value}-{target_id}",
            competition_run_id=self.run.competition_run_id,
            created_at=now,
            action=action,
            target_id=target_id,
            rationale=rationale,
        )
        if action is HumanDecisionAction.CANCEL_WORK_ITEM:
            item = next(
                (
                    candidate
                    for candidate in self.fixture.items
                    if candidate.work_item_id == target_id
                ),
                None,
            )
            if item is None:
                raise ControllerBlocked(
                    "unknown_work_item", f"unknown work item: {target_id}"
                )
            if item.blocking_release or item.scientific_severity is WorkSeverity.CRITICAL:
                raise ControllerBlocked(
                    "cannot_override_scientific_critical",
                    "a human decision cannot cancel or promote a scientific blocker",
                )
            item.status = WorkItemStatus.CANCELLED
            self.fixture.items = [
                item if candidate.work_item_id == target_id else candidate
                for candidate in self.fixture.items
            ]
        elif action is HumanDecisionAction.REDUCE_CLAIM:
            reduction = ScopeReductionDecision(
                decision_id=f"scope-human-{target_id}",
                competition_run_id=self.run.competition_run_id,
                created_at=now,
                action="downgrade_claim",
                target_claims=[target_id],
                reason=rationale,
            )
            with self.scope_reductions_file.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(reduction.model_dump_json() + "\n")
        self._append_event(
            "HumanDecision",
            now,
            {"action": action.value, "target": target_id, "decision_id": decision.decision_id},
        )
        existing = (
            self.human_decisions_file.read_text(encoding="utf-8").splitlines()
            if self.human_decisions_file.exists()
            else []
        )
        line = decision.model_dump_json()
        if line not in existing:
            with self.human_decisions_file.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        self._save()
        return decision

    def verify_artifacts(self) -> dict[str, object]:
        self._verify_source()
        freezes: dict[str, object] = {}
        mismatches: list[str] = []
        for kind, freeze_id in (
            ("model", self.run.model_freeze_id),
            ("evidence", self.run.evidence_freeze_id),
            ("paper", self.run.paper_freeze_id),
        ):
            if not freeze_id:
                continue
            record = self._load_freeze(freeze_id)
            freezes[kind] = {"id": freeze_id, "hash": record.freeze_hash}
            if isinstance(record, EvidenceFreezeRecord):
                current = _evidence_catalog_hash(
                    self.paths, self.run.source_paper_run_id
                )
                if current != record.evidence_catalog_hash:
                    mismatches.append("evidence_catalog_changed")
            if isinstance(record, PaperFreezeRecord):
                report = self.paths.root / record.report_path
                if not report.exists():
                    mismatches.append("paper_report_missing")
                elif sha256_file(report) != record.report_hash:
                    mismatches.append("paper_report_changed")
        return {
            "ok": not mismatches,
            "mismatches": mismatches,
            "source_release_hash": self.run.source_release_hash,
            "freezes": freezes,
        }

    def checkpoint(self) -> CompetitionCheckpoint:
        now = _utc(self.clock.now())
        self._sync(now)
        queue = self.evaluate_queue(now)
        checkpoint = CompetitionCheckpoint(
            checkpoint_id="",
            competition_run_id=self.run.competition_run_id,
            controller_state=self.run.state,
            clock_snapshot=now,
            queue_hash=queue.queue_hash,
            model_freeze_hash=self.run.model_freeze_hash,
            evidence_freeze_hash=self.run.evidence_freeze_hash,
            paper_freeze_hash=self.run.paper_freeze_hash,
            unresolved_blockers=list(self.run.unresolved_blockers),
        )
        checkpoint_hash = _hash_model(checkpoint, exclude=("checkpoint_hash", "checkpoint_id"))
        checkpoint_id = f"checkpoint-{checkpoint_hash[:8]}"
        checkpoint = checkpoint.model_copy(
            update={"checkpoint_id": checkpoint_id, "checkpoint_hash": checkpoint_hash}
        )
        target = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not target.exists():
            atomic_write_text(target, checkpoint.model_dump_json(indent=2) + "\n")
            self._append_event(
                "CheckpointCreated", now, {"checkpoint_id": checkpoint_id, "hash": checkpoint_hash}
            )
        self._save()
        return checkpoint


__all__ = [
    "CompetitionController",
    "CompetitionControllerError",
    "ControllerBlocked",
    "load_blocked_fixture",
]
