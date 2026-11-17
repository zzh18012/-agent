from __future__ import annotations

from datetime import datetime

from mmos.competition.clock import ensure_aware
from mmos.competition.controller_schemas import (
    ActionCategory,
    CompetitionControllerRun,
    CompetitionControllerState,
    CompetitionTimingProfile,
    CompetitionWorkItem,
    DeadlineGuardDecision,
    GuardVerdict,
    ScopeReductionDecision,
    WorkItemKind,
    WorkItemStatus,
    WorkSeverity,
)
from mmos.schemas.competition import ResolutionClass

_CAPABILITY_RANK = {
    ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY: 0,
    None: 1,
    ResolutionClass.REQUIRES_HUMAN_DECISION: 2,
    ResolutionClass.REQUIRES_NEW_CAPABILITY: 3,
    ResolutionClass.NOT_JUSTIFIED: 4,
    ResolutionClass.INVALID: 5,
}

_RUNTIME_RANK = {"seconds": 0, "minutes": 1, "hours": 2, None: 3}

_P0_KINDS = frozenset(
    {
        WorkItemKind.SCIENTIFIC_CRITICAL,
        WorkItemKind.VALIDATION_REPAIR,
        WorkItemKind.PAPER_REPAIR,
        WorkItemKind.CLAIM_SCOPE_REDUCTION,
        WorkItemKind.REVIEW_ISSUE,
    }
)


def priority_key(
    item: CompetitionWorkItem, dependent_counts: dict[str, int]
) -> tuple[object, ...]:
    """Deterministic lexicographic order; no model-generated priority scores."""
    return (
        0 if item.blocking_release else 1,
        int(item.scientific_severity),
        int(item.claim_criticality),
        _CAPABILITY_RANK.get(item.capability_status, 6),
        -dependent_counts.get(item.work_item_id, 0),
        _RUNTIME_RANK[item.expected_runtime_class],
        item.created_at.isoformat(),
        item.work_item_id,
    )


def _decision(
    verdict: GuardVerdict,
    code: str,
    reason: str,
    run: CompetitionControllerRun,
    now: datetime,
    action: ActionCategory,
    deadline: datetime | None = None,
    work_item_id: str | None = None,
) -> DeadlineGuardDecision:
    return DeadlineGuardDecision(
        decision=verdict,
        reason_code=code,
        reason=reason,
        competition_state=run.state,
        now=now,
        relevant_deadline=deadline,
        work_item_id=work_item_id,
        action_category=action,
    )


def _p0_repair_allowed(item: CompetitionWorkItem | None) -> bool:
    return bool(
        item is not None
        and item.p0_repair
        and item.blocking_release
        and item.scientific_severity is WorkSeverity.CRITICAL
        and item.kind in _P0_KINDS
        and item.capability_status
        in {
            None,
            ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY,
        }
    )


def evaluate_action(
    *,
    run: CompetitionControllerRun,
    profile: CompetitionTimingProfile,
    now: datetime,
    action: ActionCategory,
    item: CompetitionWorkItem | None = None,
    dependencies_met: bool = True,
) -> DeadlineGuardDecision:
    ensure_aware(now, "now")
    if action is ActionCategory.READ_ONLY:
        return _decision(
            GuardVerdict.ALLOW,
            "read_only",
            "read-only inspection is always permitted",
            run,
            now,
            action,
            work_item_id=item.work_item_id if item else None,
        )
    if now < profile.contest_start_at:
        return _decision(
            GuardVerdict.BLOCK,
            "contest_not_started",
            "the competition contest has not started",
            run,
            now,
            action,
            profile.contest_start_at,
            item.work_item_id if item else None,
        )

    if now >= profile.submission_deadline_at:
        return _decision(
            GuardVerdict.BLOCK,
            "submission_deadline_expired",
            "submission deadline has passed; no mutation is permitted",
            run,
            now,
            action,
            profile.submission_deadline_at,
            item.work_item_id if item else None,
        )
    if now >= profile.work_stop_at:
        return _decision(
            GuardVerdict.BLOCK,
            "work_stopped",
            "work_stop_at has passed; only read-only actions are permitted",
            run,
            now,
            action,
            profile.work_stop_at,
            item.work_item_id if item else None,
        )

    if action is ActionCategory.RELEASE_MUTATION:
        return _decision(
            GuardVerdict.BLOCK,
            "release_owned_by_red_team",
            "competition releases require an independent Red Team/Judge PASS; "
            "the controller cannot write a release",
            run,
            now,
            action,
            work_item_id=item.work_item_id if item else None,
        )
    if action is ActionCategory.SUBMISSION_PREPARATION:
        return _decision(
            GuardVerdict.BLOCK,
            "submission_preparation_out_of_scope",
            "Phase 23.1 does not implement submission preparation",
            run,
            now,
            action,
            work_item_id=item.work_item_id if item else None,
        )

    if item is not None:
        capability = item.capability_status
        if capability is ResolutionClass.REQUIRES_NEW_CAPABILITY:
            return _decision(
                GuardVerdict.BLOCK,
                "requires_new_capability",
                "new capability is not auto-built during a competition run",
                run,
                now,
                action,
                work_item_id=item.work_item_id,
            )
        if capability is ResolutionClass.REQUIRES_HUMAN_DECISION:
            return _decision(
                GuardVerdict.BLOCK,
                "requires_human_decision",
                "a human decision is required before this item can run",
                run,
                now,
                action,
                work_item_id=item.work_item_id,
            )
        if capability in {
            ResolutionClass.INVALID,
            ResolutionClass.NOT_JUSTIFIED,
        }:
            return _decision(
                GuardVerdict.BLOCK,
                "invalid_or_unjustified_request",
                "the request is not valid or not justified",
                run,
                now,
                action,
                work_item_id=item.work_item_id,
            )
        if not dependencies_met:
            return _decision(
                GuardVerdict.DEFER,
                "dependency_pending",
                "one or more dependency work items is not complete",
                run,
                now,
                action,
                work_item_id=item.work_item_id,
            )
        if (
            item.kind is WorkItemKind.OPTIONAL_IMPROVEMENT
            and now >= profile.exploration_end_at
        ):
            return _decision(
                GuardVerdict.CANCEL,
                "optional_scope_closed",
                "optional improvement was not started before exploration ended",
                run,
                now,
                action,
                work_item_id=item.work_item_id,
            )

    if (
        action is ActionCategory.MODEL_MUTATION
        and now >= profile.model_freeze_at
    ):
        return _decision(
            GuardVerdict.BLOCK,
            "model_frozen",
            "the model is frozen; model mutation requires a new freeze revision "
            "and is not performed by this controller",
            run,
            now,
            action,
            profile.model_freeze_at,
            item.work_item_id if item else None,
        )

    if (
        action in {ActionCategory.COMPUTE, ActionCategory.EVIDENCE_MUTATION}
        and now >= profile.evidence_freeze_at
        and not _p0_repair_allowed(item)
    ):
        return _decision(
            GuardVerdict.BLOCK,
            "evidence_frozen",
            "evidence is frozen; only an existing-capability P0 repair may "
            "create a new evidence revision",
            run,
            now,
            action,
            profile.evidence_freeze_at,
            item.work_item_id if item else None,
        )

    if (
        action is ActionCategory.PAPER_MUTATION
        and now >= profile.paper_freeze_at
        and not _p0_repair_allowed(item)
    ):
        return _decision(
            GuardVerdict.BLOCK,
            "paper_frozen",
            "paper is frozen; only an existing-capability P0 repair may create a "
            "new paper freeze revision",
            run,
            now,
            action,
            profile.paper_freeze_at,
            item.work_item_id if item else None,
        )

    if run.state is CompetitionControllerState.CLOSED:
        return _decision(
            GuardVerdict.BLOCK,
            "controller_closed",
            "controller is closed; only read-only inspection is permitted",
            run,
            now,
            action,
            work_item_id=item.work_item_id if item else None,
        )

    return _decision(
        GuardVerdict.ALLOW,
        "allowed",
        "action is permitted by the current phase and deadlines",
        run,
        now,
        action,
        work_item_id=item.work_item_id if item else None,
    )


def readiness_reasons(
    *,
    run: CompetitionControllerRun,
    profile: CompetitionTimingProfile,
    now: datetime,
    judge_status: str,
    number_audit_ok: bool,
    math_audit_ok: bool,
) -> list[str]:
    ensure_aware(now, "now")
    reasons: list[str] = []
    if now < profile.paper_freeze_at:
        reasons.append("paper_freeze_at_not_reached")
    if run.model_freeze_hash is None:
        reasons.append("model_freeze_missing")
    if run.evidence_freeze_hash is None:
        reasons.append("evidence_freeze_missing")
    if run.paper_freeze_hash is None:
        reasons.append("paper_freeze_missing")
    if judge_status != "pass":
        reasons.append("scientific_gate_not_pass")
    if not number_audit_ok:
        reasons.append("number_audit_failed")
    if not math_audit_ok:
        reasons.append("math_audit_critical")
    reasons.extend(run.unresolved_blockers)
    if now >= profile.submission_deadline_at:
        reasons.append("submission_deadline_expired")
    return list(dict.fromkeys(reasons))


def auto_scope_reductions(
    *,
    items: list[CompetitionWorkItem],
    run: CompetitionControllerRun,
    profile: CompetitionTimingProfile,
    now: datetime,
) -> list[ScopeReductionDecision]:
    ensure_aware(now, "now")
    decisions: list[ScopeReductionDecision] = []
    for item in items:
        if item.status is not WorkItemStatus.PENDING:
            continue
        if item.blocking_release or item.scientific_severity is WorkSeverity.CRITICAL:
            continue
        if item.kind is WorkItemKind.OPTIONAL_IMPROVEMENT and now >= profile.exploration_end_at:
            decisions.append(
                ScopeReductionDecision(
                    decision_id=f"scope-{item.work_item_id}",
                    competition_run_id=run.competition_run_id,
                    created_at=now,
                    action="cancel_optional_improvement",
                    target_work_item_ids=[item.work_item_id],
                    reason="optional improvement was not started before exploration_end_at",
                )
            )
    return decisions


__all__ = [
    "auto_scope_reductions",
    "evaluate_action",
    "priority_key",
    "readiness_reasons",
]
