from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from mmos.core.project import atomic_write_text
from mmos.historical_benchmark.schemas import (
    CapabilityFitAssessment,
    CapabilityVerdict,
    DispositionReason,
    ExecutionDisposition,
    ExecutionDispositionRecord,
    ProblemRequirement,
)
from mmos.schemas.routing import RoutingDecision, RoutingStatus


class DispositionContractError(RuntimeError):
    pass


def _hash_payload(payload: object, *, exclude: tuple[str, ...] = ()) -> str:
    data = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for key in exclude:
            data.pop(key, None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _model_hash(model: object) -> str:
    return _hash_payload(model.model_dump(mode="json"))  # type: ignore[attr-defined]


def requirements_hash(requirements: list[ProblemRequirement]) -> str:
    return _hash_payload(
        [
            [item.requirement_id, item.question_hash, item.required_output]
            for item in sorted(requirements, key=lambda item: item.requirement_id)
        ]
    )


def compute_disposition(
    *,
    router: RoutingDecision | None,
    capability_fit: CapabilityFitAssessment | None,
    requirements: list[ProblemRequirement],
    invalid_input: bool = False,
    routing_failure: bool = False,
    now: datetime | None = None,
) -> ExecutionDispositionRecord:
    problem_hash = router.problem_hash if router is not None else ""
    router_hash = _model_hash(router) if router is not None else ""
    fit_hash = capability_fit.assessment_hash if capability_fit is not None else ""
    req_hash = requirements_hash(requirements)
    supported_ids = capability_fit.supported if capability_fit is not None else []
    unsupported_ids = capability_fit.unsupported if capability_fit is not None else []
    affected = [item.requirement_id for item in requirements]

    if invalid_input:
        disposition = ExecutionDisposition.BLOCKED_INVALID_INPUT
        reason = DispositionReason.INVALID_PROBLEM_INPUT
    elif routing_failure or router is None:
        disposition = ExecutionDisposition.BLOCKED_ROUTING_FAILURE
        reason = DispositionReason.ROUTING_TRANSPORT_FAILED
    elif capability_fit is None or not capability_fit.assessment_complete:
        disposition = ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
        reason = DispositionReason.CAPABILITY_ASSESSMENT_MISSING
    else:
        verdict = capability_fit.verdict
        if router.status is RoutingStatus.UNSUPPORTED:
            if verdict is CapabilityVerdict.UNSUPPORTED:
                disposition = ExecutionDisposition.UNSUPPORTED_CAPABILITY
                reason = DispositionReason.ROUTER_DECLARED_UNSUPPORTED
            else:
                disposition = ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
                reason = DispositionReason.ROUTER_CAPABILITY_CONTRADICTION
        elif verdict is CapabilityVerdict.UNSUPPORTED:
            disposition = ExecutionDisposition.UNSUPPORTED_CAPABILITY
            reason = DispositionReason.NO_CORE_REQUIREMENT_SUPPORTED
        elif router.status in {RoutingStatus.SINGLE_TASK, RoutingStatus.MULTI_TASK}:
            if verdict is CapabilityVerdict.SUPPORTED:
                disposition = ExecutionDisposition.READY
                reason = DispositionReason.ALL_CORE_REQUIREMENTS_SUPPORTED
            elif verdict is CapabilityVerdict.PARTIALLY_SUPPORTED:
                disposition = ExecutionDisposition.PARTIALLY_SUPPORTED
                reason = DispositionReason.SOME_CORE_REQUIREMENTS_UNSUPPORTED
            else:
                disposition = ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
                reason = DispositionReason.CAPABILITY_ASSESSMENT_MISSING
        elif router.status is RoutingStatus.AMBIGUOUS:
            if verdict is CapabilityVerdict.SUPPORTED:
                disposition = ExecutionDisposition.NEEDS_CLARIFICATION
                reason = DispositionReason.MULTIPLE_SUPPORTED_ROUTES_NEED_RESOLUTION
            elif verdict is CapabilityVerdict.PARTIALLY_SUPPORTED:
                if capability_fit.fully_supported_route_candidates:
                    disposition = ExecutionDisposition.NEEDS_CLARIFICATION
                    reason = DispositionReason.MULTIPLE_SUPPORTED_ROUTES_NEED_RESOLUTION
                else:
                    disposition = ExecutionDisposition.PARTIALLY_SUPPORTED
                    reason = DispositionReason.SOME_CORE_REQUIREMENTS_UNSUPPORTED
            else:
                disposition = ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
                reason = DispositionReason.CAPABILITY_ASSESSMENT_MISSING
        else:
            disposition = ExecutionDisposition.BLOCKED_CONTRACT_INCONSISTENCY
            reason = DispositionReason.ROUTER_CAPABILITY_CONTRADICTION

    record = ExecutionDispositionRecord(
        disposition_id=f"disp-{router.routing_id if router is not None else 'missing-router'}",
        source_problem_hash=problem_hash,
        router_result_id=router.routing_id if router is not None else "",
        router_result_hash=router_hash,
        capability_fit_id=capability_fit.assessment_id if capability_fit is not None else "",
        capability_fit_hash=fit_hash,
        requirements_hash=req_hash,
        disposition=disposition,
        reason_code=reason,
        affected_requirement_ids=affected,
        supported_requirement_ids=list(supported_ids),
        unsupported_requirement_ids=list(unsupported_ids),
        created_at=(now or datetime.now(UTC)),
    )
    return record.model_copy(
        update={
            "record_hash": _hash_payload(record.model_dump(mode="json"), exclude=("record_hash",))
        }
    )


def modeling_execution_allowed(record: ExecutionDispositionRecord) -> bool:
    return record.disposition is ExecutionDisposition.READY


def disposition_is_stale(
    record: ExecutionDispositionRecord,
    *,
    router_hash: str,
    capability_fit_hash: str,
    requirements_sha256: str,
) -> bool:
    return (
        record.router_result_hash != router_hash
        or record.capability_fit_hash != capability_fit_hash
        or record.requirements_hash != requirements_sha256
    )


def write_disposition(path, record: ExecutionDispositionRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, record.model_dump_json(indent=2) + "\n")


__all__ = [
    "DispositionContractError",
    "compute_disposition",
    "disposition_is_stale",
    "modeling_execution_allowed",
    "requirements_hash",
    "write_disposition",
]
