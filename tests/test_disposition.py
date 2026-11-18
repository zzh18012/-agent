from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from mmos.historical_benchmark.capabilities import assess_capability_fit
from mmos.historical_benchmark.disposition import (
    compute_disposition,
    disposition_is_stale,
    modeling_execution_allowed,
    requirements_hash,
)
from mmos.historical_benchmark.schemas import (
    CapabilityFitAssessment,
    CapabilityVerdict,
    DispositionReason,
    ExecutionDisposition,
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


def _requirement(requirement_id: str = "q1") -> ProblemRequirement:
    import hashlib

    text = "select binary items under one capacity constraint"
    return ProblemRequirement(
        requirement_id=requirement_id,
        source_location=requirement_id,
        question_text=text,
        question_hash=hashlib.sha256(text.encode()).hexdigest(),
        required_output="output",
    )


def _router(status: RoutingStatus, families: list[TaskFamily] | None = None) -> RoutingDecision:
    families = families or []
    return RoutingDecision(
        routing_id="route-test",
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        status=status,
        primary_task=families[0] if status is RoutingStatus.SINGLE_TASK and families else None,
        required_task_families=families,
        clarifications_needed=["need clarification"] if status is RoutingStatus.AMBIGUOUS else [],
        unsupported_requirements=["no supported route"]
        if status is RoutingStatus.UNSUPPORTED
        else [],
        subproblem_routes=[SubproblemRoute(subproblem_id="q1", task_families=families)],
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


def _fit(
    verdict: CapabilityVerdict,
    *,
    candidates: list[str] | None = None,
) -> CapabilityFitAssessment:
    supported = ["q1"] if verdict is CapabilityVerdict.SUPPORTED else []
    partial = ["q1"] if verdict is CapabilityVerdict.PARTIALLY_SUPPORTED else []
    unsupported = ["q1"] if verdict is CapabilityVerdict.UNSUPPORTED else []
    return CapabilityFitAssessment(
        assessment_id="fit-test",
        problem_requirement_ids=["q1"],
        route_task_families=["optimization"],
        required_capabilities=["optimization"],
        available_capabilities=["optimization.milp.knapsack_cbc"] if supported else [],
        supported=supported,
        partially_supported=partial,
        unsupported=unsupported,
        fully_supported_route_candidates=candidates or [],
        core_requirement_count=1,
        supported_core_requirement_count=len(supported),
        partial_core_requirement_count=len(partial),
        unsupported_core_requirement_count=len(unsupported),
        verdict=verdict,
        assessment_complete=True,
        assessment_hash="c" * 64,
    )


def _dispose(router, fit=None, **kwargs):
    return compute_disposition(
        router=router,
        capability_fit=fit,
        requirements=[_requirement()],
        now=NOW,
        **kwargs,
    )


def test_single_task_truth_table() -> None:
    router = _router(RoutingStatus.SINGLE_TASK, [TaskFamily.OPTIMIZATION])
    assert (
        _dispose(router, _fit(CapabilityVerdict.SUPPORTED)).disposition
        is ExecutionDisposition.READY
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.PARTIALLY_SUPPORTED)).disposition
        is ExecutionDisposition.PARTIALLY_SUPPORTED
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.UNSUPPORTED)).disposition
        is ExecutionDisposition.UNSUPPORTED_CAPABILITY
    )


def test_multi_task_truth_table() -> None:
    router = _router(
        RoutingStatus.MULTI_TASK,
        [TaskFamily.FORECASTING, TaskFamily.OPTIMIZATION],
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.SUPPORTED)).disposition
        is ExecutionDisposition.READY
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.UNSUPPORTED)).disposition
        is ExecutionDisposition.UNSUPPORTED_CAPABILITY
    )


def test_ambiguous_truth_table() -> None:
    router = _router(RoutingStatus.AMBIGUOUS)
    assert (
        _dispose(router, _fit(CapabilityVerdict.SUPPORTED)).disposition
        is ExecutionDisposition.NEEDS_CLARIFICATION
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.UNSUPPORTED)).disposition
        is ExecutionDisposition.UNSUPPORTED_CAPABILITY
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.PARTIALLY_SUPPORTED)).disposition
        is ExecutionDisposition.PARTIALLY_SUPPORTED
    )
    assert (
        _dispose(
            router,
            _fit(CapabilityVerdict.PARTIALLY_SUPPORTED, candidates=["optimization"]),
        ).disposition
        is ExecutionDisposition.NEEDS_CLARIFICATION
    )


def test_unsupported_router_contract_cases() -> None:
    router = _router(RoutingStatus.UNSUPPORTED)
    unsupported = _dispose(router, _fit(CapabilityVerdict.UNSUPPORTED))
    assert unsupported.disposition is ExecutionDisposition.UNSUPPORTED_CAPABILITY
    assert unsupported.reason_code is DispositionReason.ROUTER_DECLARED_UNSUPPORTED
    contradiction = _dispose(router, _fit(CapabilityVerdict.SUPPORTED))
    assert contradiction.disposition is ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
    assert contradiction.reason_code is DispositionReason.ROUTER_CAPABILITY_CONTRADICTION


def test_error_paths_fail_closed() -> None:
    router = _router(RoutingStatus.SINGLE_TASK, [TaskFamily.OPTIMIZATION])
    assert _dispose(router, None).disposition is ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
    assert (
        _dispose(router, _fit(CapabilityVerdict.SUPPORTED), invalid_input=True).disposition
        is ExecutionDisposition.BLOCKED_INVALID_INPUT
    )
    assert (
        _dispose(router, _fit(CapabilityVerdict.SUPPORTED), routing_failure=True).disposition
        is ExecutionDisposition.BLOCKED_ROUTING_FAILURE
    )


def test_modeling_guard_and_staleness() -> None:
    router = _router(RoutingStatus.SINGLE_TASK, [TaskFamily.OPTIMIZATION])
    ready = _dispose(router, _fit(CapabilityVerdict.SUPPORTED))
    assert modeling_execution_allowed(ready)
    partial = _dispose(router, _fit(CapabilityVerdict.PARTIALLY_SUPPORTED))
    assert not modeling_execution_allowed(partial)
    req_hash = requirements_hash([_requirement()])
    assert not disposition_is_stale(
        ready,
        router_hash=ready.router_result_hash,
        capability_fit_hash=ready.capability_fit_hash,
        requirements_sha256=req_hash,
    )
    assert disposition_is_stale(
        ready,
        router_hash="changed",
        capability_fit_hash=ready.capability_fit_hash,
        requirements_sha256=req_hash,
    )
    assert disposition_is_stale(
        ready,
        router_hash=ready.router_result_hash,
        capability_fit_hash="changed",
        requirements_sha256=req_hash,
    )
    assert disposition_is_stale(
        ready,
        router_hash=ready.router_result_hash,
        capability_fit_hash=ready.capability_fit_hash,
        requirements_sha256="changed",
    )


def test_capability_fit_exposes_core_counts() -> None:
    router = _router(RoutingStatus.SINGLE_TASK, [TaskFamily.OPTIMIZATION])
    req = _requirement("q1")
    fit = assess_capability_fit(
        router,
        [req],
    )
    assert fit.core_requirement_count == 1
    assert (
        fit.supported_core_requirement_count
        + fit.partial_core_requirement_count
        + fit.unsupported_core_requirement_count
        == 1
    )


def test_canonical_2025_replay_is_unsupported_capability() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "benchmarks" / "historical" / "cumcm-2025-blind-01"
    result_file = run / "benchmark_result.json"
    if not result_file.exists():
        import pytest

        pytest.skip("canonical 2025 benchmark is not available")
    before = json.loads(result_file.read_text(encoding="utf-8"))["result_hash"]
    router = RoutingDecision.model_validate_json(
        (run / "router_result.json").read_text(encoding="utf-8")
    )
    fit = CapabilityFitAssessment.model_validate_json(
        (run / "capability_fit.json").read_text(encoding="utf-8")
    )
    requirements = [
        ProblemRequirement.model_validate(item)
        for item in json.loads((run / "requirements.json").read_text(encoding="utf-8"))
    ]
    record = compute_disposition(
        router=router,
        capability_fit=fit,
        requirements=requirements,
        now=NOW,
    )
    assert record.disposition is ExecutionDisposition.UNSUPPORTED_CAPABILITY
    assert record.reason_code is DispositionReason.NO_CORE_REQUIREMENT_SUPPORTED
    after = json.loads(result_file.read_text(encoding="utf-8"))["result_hash"]
    assert before == after == "23170003a04ca75266d831634abf898d0ec6682da110d775355c89f77676a3ae"
