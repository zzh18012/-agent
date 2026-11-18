from __future__ import annotations

from datetime import UTC, datetime

from mmos.historical_benchmark.schemas import (
    CapabilityCoverageSnapshot,
    CapabilityFitAssessment,
    CapabilityVerdict,
    ExecutionDisposition,
    ExecutionDispositionRecord,
    ProblemRequirement,
)


def _first_blocking_stage(
    *,
    fit: CapabilityFitAssessment | None,
    disposition: ExecutionDispositionRecord | None,
    router_reached: bool,
    capability_fit_reached: bool,
    modeling_reached: bool,
    experiment_reached: bool,
    evidence_reached: bool,
    paper_reached: bool,
    judge_reached: bool,
) -> str:
    if not router_reached:
        return "router"
    if not capability_fit_reached or fit is None:
        return "capability_fit"
    if fit.verdict is not CapabilityVerdict.SUPPORTED:
        return "capability_fit"
    if disposition is None or disposition.disposition is not ExecutionDisposition.READY:
        return "execution_disposition"
    if not modeling_reached:
        return "modeling"
    if not experiment_reached:
        return "experiment"
    if not evidence_reached:
        return "evidence"
    if not paper_reached:
        return "paper"
    if not judge_reached:
        return "judge"
    return "none"


def build_coverage_snapshot(
    *,
    benchmark_run_id: str,
    requirements: list[ProblemRequirement],
    fit: CapabilityFitAssessment | None,
    disposition: ExecutionDispositionRecord | None,
    router_reached: bool,
    capability_fit_reached: bool,
    modeling_reached: bool = False,
    experiment_reached: bool = False,
    evidence_reached: bool = False,
    paper_reached: bool = False,
    judge_reached: bool = False,
    now: datetime | None = None,
) -> CapabilityCoverageSnapshot:
    total = len(requirements)
    supported = len(fit.supported) if fit is not None else 0
    partial = len(fit.partially_supported) if fit is not None else 0
    unsupported = len(fit.unsupported) if fit is not None else total
    return CapabilityCoverageSnapshot(
        benchmark_run_id=benchmark_run_id,
        core_requirement_count=total,
        supported_count=supported,
        partial_count=partial,
        unsupported_count=unsupported,
        coverage_fraction=(supported / total) if total else 0.0,
        router_reached=router_reached,
        capability_fit_reached=capability_fit_reached,
        modeling_reached=modeling_reached,
        experiment_reached=experiment_reached,
        evidence_reached=evidence_reached,
        paper_reached=paper_reached,
        judge_reached=judge_reached,
        first_blocking_stage=_first_blocking_stage(
            fit=fit,
            disposition=disposition,
            router_reached=router_reached,
            capability_fit_reached=capability_fit_reached,
            modeling_reached=modeling_reached,
            experiment_reached=experiment_reached,
            evidence_reached=evidence_reached,
            paper_reached=paper_reached,
            judge_reached=judge_reached,
        ),
        created_at=(now or datetime.now(UTC)),
    )


__all__ = ["build_coverage_snapshot"]
