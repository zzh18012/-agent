from datetime import UTC, datetime

from mmos.schemas.competition import (
    Capability,
    CapabilityNeed,
    CompetitionExperimentManifest,
    CompetitionExperimentRequest,
)

SUPPORTED_CAPABILITIES = frozenset(
    {
        Capability.PREDICTION,
        Capability.FORECASTING,
        Capability.OPTIMIZATION,
        Capability.SIMULATION,
        Capability.GRAPH,
        Capability.MULTI_OBJECTIVE,
        Capability.MIXED,
    }
)


class CompetitionRequestError(RuntimeError):
    pass


def build_manifest(
    needs: list[CapabilityNeed], known_evidence_ids: set[str]
) -> CompetitionExperimentManifest:
    requests: list[CompetitionExperimentRequest] = []
    seen: set[tuple[str, str]] = set()
    for need in needs:
        unknown = sorted(set(need.supporting_evidence_ids) - known_evidence_ids)
        if unknown:
            raise CompetitionRequestError(
                "experiment request cites unknown evidence ids: " + ", ".join(unknown)
            )
        key = (need.capability.value, need.question.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        supported = need.capability in SUPPORTED_CAPABILITIES
        requests.append(
            CompetitionExperimentRequest(
                request_id=f"request-{len(requests) + 1}",
                question=need.question,
                rationale=need.rationale,
                capability=need.capability,
                proposed_experiment=need.proposed_experiment,
                expected_evidence=need.expected_evidence,
                priority=need.priority,
                supporting_evidence_ids=need.supporting_evidence_ids,
                supported=supported,
                reason=None
                if supported
                else "capability is not implemented by the current engines",
            )
        )
    supported_count = sum(1 for request in requests if request.supported)
    return CompetitionExperimentManifest(
        requests=requests,
        supported_count=supported_count,
        unsupported_count=len(requests) - supported_count,
        created_at=datetime.now(UTC).replace(microsecond=0),
    )


__all__ = [
    "SUPPORTED_CAPABILITIES",
    "CompetitionRequestError",
    "build_manifest",
]
