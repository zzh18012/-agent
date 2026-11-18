from __future__ import annotations

from datetime import UTC, datetime

from mmos.historical_benchmark.coverage import build_coverage_snapshot
from mmos.historical_benchmark.disposition import compute_disposition
from mmos.historical_benchmark.schemas import (
    CapabilityFitAssessment,
    CapabilityVerdict,
    ProblemRequirement,
)
from mmos.schemas.routing import (
    ExecutionStatus,
    RoutingDecision,
    RoutingStatus,
    SubproblemRoute,
)
from mmos.schemas.states import TaskFamily

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)


def _req(requirement_id: str, text: str) -> ProblemRequirement:
    import hashlib

    return ProblemRequirement(
        requirement_id=requirement_id,
        source_location=requirement_id,
        question_text=text,
        question_hash=hashlib.sha256(text.encode()).hexdigest(),
        required_output="output",
    )


def _router(status: RoutingStatus, families: list[TaskFamily] | None = None):
    families = families or []
    return RoutingDecision(
        routing_id="route-test",
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        status=status,
        primary_task=families[0] if status is RoutingStatus.SINGLE_TASK and families else None,
        required_task_families=families,
        clarifications_needed=["clarify"] if status is RoutingStatus.AMBIGUOUS else [],
        unsupported_requirements=["unsupported"] if status is RoutingStatus.UNSUPPORTED else [],
        subproblem_routes=[
            SubproblemRoute(subproblem_id="q1", task_families=families)
        ],
        capabilities=[
            {
                "task_family": family,
                "current_supported_scope": "test",
                "execution_readiness": "FULL",
            }
            for family in families
        ],
        execution_status=(
            ExecutionStatus.READY_FOR_SINGLE_TASK
            if status is RoutingStatus.SINGLE_TASK
            else ExecutionStatus.BLOCKED_NEEDS_CLARIFICATION
            if status is RoutingStatus.AMBIGUOUS
            else ExecutionStatus.BLOCKED_UNSUPPORTED
        ),
        provider="test",
        model="test",
        created_at=NOW,
    )


def _fit(verdict: CapabilityVerdict, requirement_ids: list[str]) -> CapabilityFitAssessment:
    return CapabilityFitAssessment(
        assessment_id="fit",
        problem_requirement_ids=requirement_ids,
        supported=requirement_ids if verdict is CapabilityVerdict.SUPPORTED else [],
        partially_supported=requirement_ids
        if verdict is CapabilityVerdict.PARTIALLY_SUPPORTED
        else [],
        unsupported=requirement_ids
        if verdict is CapabilityVerdict.UNSUPPORTED
        else [],
        verdict=verdict,
        core_requirement_count=len(requirement_ids),
        supported_core_requirement_count=len(requirement_ids)
        if verdict is CapabilityVerdict.SUPPORTED
        else 0,
        partial_core_requirement_count=len(requirement_ids)
        if verdict is CapabilityVerdict.PARTIALLY_SUPPORTED
        else 0,
        unsupported_core_requirement_count=len(requirement_ids)
        if verdict is CapabilityVerdict.UNSUPPORTED
        else 0,
        assessment_complete=True,
        assessment_hash="c" * 64,
    )


def test_coverage_and_first_blocking_stage() -> None:
    reqs = [
        _req("q1", "select binary items under capacity"),
        _req("q2", "select binary items under capacity"),
        _req("q3", "select binary items under capacity"),
    ]
    fit = _fit(CapabilityVerdict.UNSUPPORTED, ["q1", "q2", "q3"])
    router = _router(RoutingStatus.SINGLE_TASK, [TaskFamily.OPTIMIZATION])
    disposition = compute_disposition(
        router=router, capability_fit=fit, requirements=reqs, now=NOW
    )
    snapshot = build_coverage_snapshot(
        benchmark_run_id="b",
        requirements=reqs,
        fit=fit,
        disposition=disposition,
        router_reached=True,
        capability_fit_reached=True,
        now=NOW,
    )
    assert snapshot.core_requirement_count == 3
    assert snapshot.unsupported_count == 3
    assert snapshot.coverage_fraction == 0.0
    assert snapshot.first_blocking_stage == "capability_fit"


def test_coverage_ready_and_partial_guard() -> None:
    req_supported = [_req("q1", "select binary items under capacity")]
    supported = _fit(CapabilityVerdict.SUPPORTED, ["q1"])
    router = _router(RoutingStatus.SINGLE_TASK, [TaskFamily.OPTIMIZATION])
    disposition = compute_disposition(
        router=router, capability_fit=supported, requirements=req_supported, now=NOW
    )
    ready = build_coverage_snapshot(
        benchmark_run_id="b",
        requirements=req_supported,
        fit=supported,
        disposition=disposition,
        router_reached=True,
        capability_fit_reached=True,
        modeling_reached=False,
        now=NOW,
    )
    assert ready.coverage_fraction == 1.0
    assert ready.first_blocking_stage == "modeling"
    partial = _fit(CapabilityVerdict.PARTIALLY_SUPPORTED, ["q1"])
    partial_disposition = compute_disposition(
        router=router, capability_fit=partial, requirements=req_supported, now=NOW
    )
    partial_snapshot = build_coverage_snapshot(
        benchmark_run_id="b",
        requirements=req_supported,
        fit=partial,
        disposition=partial_disposition,
        router_reached=True,
        capability_fit_reached=True,
        now=NOW,
    )
    assert partial_snapshot.coverage_fraction == 0.0
    assert partial_snapshot.partial_count == 1
    assert partial_snapshot.first_blocking_stage == "capability_fit"


def test_coverage_router_not_reached_and_clarification_guard() -> None:
    reqs = [_req("q1", "not enough information")]
    snapshot = build_coverage_snapshot(
        benchmark_run_id="b",
        requirements=reqs,
        fit=None,
        disposition=None,
        router_reached=False,
        capability_fit_reached=False,
        now=NOW,
    )
    assert snapshot.first_blocking_stage == "router"
    assert snapshot.coverage_fraction == 0.0
    fit = _fit(CapabilityVerdict.SUPPORTED, ["q1"])
    router = _router(RoutingStatus.AMBIGUOUS)
    disposition = compute_disposition(
        router=router, capability_fit=fit, requirements=reqs, now=NOW
    )
    clarification = build_coverage_snapshot(
        benchmark_run_id="b",
        requirements=reqs,
        fit=fit,
        disposition=disposition,
        router_reached=True,
        capability_fit_reached=True,
        now=NOW,
    )
    assert disposition.disposition.value == "needs_clarification"
    assert clarification.first_blocking_stage == "execution_disposition"
