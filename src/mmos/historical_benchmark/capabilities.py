from __future__ import annotations

import hashlib
import json
from typing import Any

from mmos.historical_benchmark.schemas import (
    CapabilityDescriptor,
    CapabilityFitAssessment,
    CapabilityGap,
    CapabilityVerdict,
    ProblemRequirement,
    RequirementCapabilityFit,
)
from mmos.schemas.routing import RoutingDecision, RoutingStatus
from mmos.schemas.states import TaskFamily

_SHAPE_KEYWORDS: dict[TaskFamily, tuple[str, ...]] = {
    TaskFamily.PREDICTION: (
        "predict",
        "classify",
        "classification",
        "regression",
        "label",
        "target",
    ),
    TaskFamily.FORECASTING: (
        "forecast",
        "future",
        "time series",
        "temporal",
        "next period",
        "horizon",
    ),
    TaskFamily.OPTIMIZATION: (
        "knapsack",
        "0/1",
        "binary selection",
        "select item",
        "item selection",
        "capacity",
    ),
    TaskFamily.SIMULATION: (
        "queue",
        "m/m/1",
        "waiting line",
        "service system",
        "replication",
    ),
    TaskFamily.GRAPH: (
        "max-flow",
        "max flow",
        "source",
        "sink",
        "flow network",
    ),
    TaskFamily.MULTI_OBJECTIVE: (
        "portfolio",
        "bi-objective",
        "trade-off",
        "tradeoff",
        "budget",
    ),
}

_LIMIT_KEYWORDS = (
    "geometry",
    "geometric",
    "polygon",
    "bearing",
    "intersection",
    "robot",
    "path planning",
    "search strategy",
    "active search",
    "directional",
    "simulator",
    "http",
    "pde",
    "partial differential",
    "optimal control",
    "continuous nonlinear",
    "continuous optimisation",
    "continuous optimization",
)


def _hash_payload(payload: object, *, exclude: tuple[str, ...] = ()) -> str:
    data = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for key in exclude:
            data.pop(key, None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def descriptor_hash(descriptor: CapabilityDescriptor) -> str:
    return _hash_payload(descriptor.model_dump(mode="json"), exclude=("descriptor_hash",))


def _descriptor(**kwargs: Any) -> CapabilityDescriptor:
    descriptor = CapabilityDescriptor(**kwargs)
    return descriptor.model_copy(update={"descriptor_hash": descriptor_hash(descriptor)})


def capability_registry() -> dict[TaskFamily, CapabilityDescriptor]:
    return {
        TaskFamily.PREDICTION: _descriptor(
            capability_id="prediction.tabular.supervised",
            task_family=TaskFamily.PREDICTION.value,
            supported_problem_shapes=["tabular supervised regression/classification"],
            supported_input_types=["csv", "tabular"],
            supported_decision_types=[],
            supported_constraint_types=[],
            runner="mmos.experiments.workflow.run_experiment_workflow",
            validator="mmos.validation.validate_experiment",
            regression_evidence=["tests/test_paper.py", "tests/test_experiment_workflow.py"],
            known_limits=["no geometry/path/active-sensing/simulator tasks"],
        ),
        TaskFamily.FORECASTING: _descriptor(
            capability_id="forecasting.tabular.rolling_origin",
            task_family=TaskFamily.FORECASTING.value,
            supported_problem_shapes=["tabular temporal forecasting with chronological validation"],
            supported_input_types=["csv", "temporal tabular"],
            supported_decision_types=[],
            supported_constraint_types=[],
            runner="mmos.experiments.workflow.run_experiment_workflow",
            validator="mmos.validation.validate_experiment",
            regression_evidence=["tests/test_forecasting.py"],
            known_limits=["one target series; no geometric/robot/simulator tasks"],
        ),
        TaskFamily.OPTIMIZATION: _descriptor(
            capability_id="optimization.milp.knapsack_cbc",
            task_family=TaskFamily.OPTIMIZATION.value,
            supported_problem_shapes=["0/1 knapsack MILP with CBC"],
            supported_input_types=["json spec", "tabular item values"],
            supported_decision_types=["binary item selection"],
            supported_constraint_types=["single capacity/budget constraint"],
            runner="mmos.experiments.optimization_runner.run_optimization",
            validator="mmos.validation.optimization.validate_optimization_experiment",
            regression_evidence=["tests/test_routing.py", "tests/test_optimization.py"],
            known_limits=[
                "only 0/1 knapsack MILP with one capacity constraint",
                "no continuous nonlinear, geometry, path planning, optimal control, or "
                "multi-vehicle planning",
            ],
        ),
        TaskFamily.SIMULATION: _descriptor(
            capability_id="simulation.queue.mm1",
            task_family=TaskFamily.SIMULATION.value,
            supported_problem_shapes=["stable M/M/1 discrete-event queueing simulation"],
            supported_input_types=["json spec"],
            supported_decision_types=[],
            supported_constraint_types=[],
            runner="mmos.experiments.simulation_runner",
            validator="mmos.validation.simulation.validate_simulation_experiment",
            regression_evidence=["tests/test_simulation.py"],
            known_limits=["one M/M/1 queue only; no external HTTP simulator or robot model"],
        ),
        TaskFamily.GRAPH: _descriptor(
            capability_id="graph.max_flow",
            task_family=TaskFamily.GRAPH.value,
            supported_problem_shapes=["single-source/single-sink directed max-flow"],
            supported_input_types=["json graph"],
            supported_decision_types=["flow assignment"],
            supported_constraint_types=["edge capacities"],
            runner="mmos.experiments.graph_runner.run_graph_experiment",
            validator="mmos.validation.graph.validate_graph_experiment",
            regression_evidence=["tests/test_graph.py"],
            known_limits=[
                "max-flow only; no shortest path, matching, centrality, dynamic graphs, or geometry"
            ],
        ),
        TaskFamily.MULTI_OBJECTIVE: _descriptor(
            capability_id="multi_objective.portfolio.epsilon",
            task_family=TaskFamily.MULTI_OBJECTIVE.value,
            supported_problem_shapes=["bi-objective binary project-portfolio epsilon-constraint"],
            supported_input_types=["json portfolio"],
            supported_decision_types=["binary project selection"],
            supported_constraint_types=["budget constraint"],
            runner="mmos.experiments.multi_objective_runner.run_multi_objective",
            validator="mmos.validation.multi_objective.validate_multi_objective_experiment",
            regression_evidence=["tests/test_multi_objective.py"],
            known_limits=["two maximize objectives, binary portfolio, one budget constraint only"],
        ),
    }


def _text_for(requirement: ProblemRequirement) -> str:
    return f"{requirement.question_text} {requirement.required_output}".lower()


def _requirement_supported(requirement: ProblemRequirement, family: TaskFamily) -> tuple[bool, str]:
    text = _text_for(requirement)
    if any(keyword in text for keyword in _LIMIT_KEYWORDS):
        return False, "requirement contains a known capability limit"
    keywords = _SHAPE_KEYWORDS[family]
    if not any(keyword in text for keyword in keywords):
        return False, (f"requirement does not match the formal {family.value} capability slice")
    return True, f"matches {family.value} supported slice"


def assess_capability_fit(
    decision: RoutingDecision,
    requirements: list[ProblemRequirement],
    *,
    descriptors: dict[TaskFamily, CapabilityDescriptor] | None = None,
) -> CapabilityFitAssessment:
    registry = descriptors or capability_registry()
    route_by_requirement = {
        route.subproblem_id: route.task_families for route in decision.subproblem_routes
    }
    fits: list[RequirementCapabilityFit] = []
    gaps: list[CapabilityGap] = []
    supported: list[str] = []
    partially_supported: list[str] = []
    unsupported: list[str] = []
    evidence: set[str] = set()
    for requirement in requirements:
        families = route_by_requirement.get(requirement.requirement_id, [])
        available: list[str] = []
        failures: list[str] = []
        if decision.status is RoutingStatus.UNSUPPORTED:
            failures.append("router marked the problem UNSUPPORTED")
        if not families:
            failures.append("router produced no TaskFamily route")
        for family in families:
            descriptor = registry.get(family)
            if descriptor is None:
                failures.append(f"no descriptor for {family.value}")
                continue
            ok, reason = _requirement_supported(requirement, family)
            if ok:
                available.append(descriptor.capability_id)
                evidence.update(
                    [descriptor.runner, descriptor.validator, *descriptor.regression_evidence]
                )
            else:
                failures.append(f"{descriptor.capability_id}: {reason}")
        verdict = (
            CapabilityVerdict.SUPPORTED
            if available and not failures
            else CapabilityVerdict.PARTIALLY_SUPPORTED
            if available
            else CapabilityVerdict.UNSUPPORTED
        )
        fits.append(
            RequirementCapabilityFit(
                requirement_id=requirement.requirement_id,
                route_task_families=[family.value for family in families],
                required_capabilities=[family.value for family in families],
                available_capabilities=available,
                verdict=verdict,
                reason="; ".join(failures) or "formal capability slice available",
            )
        )
        if verdict is CapabilityVerdict.SUPPORTED:
            supported.append(requirement.requirement_id)
        else:
            if verdict is CapabilityVerdict.PARTIALLY_SUPPORTED:
                partially_supported.append(requirement.requirement_id)
            else:
                unsupported.append(requirement.requirement_id)
            gaps.append(
                CapabilityGap(
                    gap_id=f"gap-{requirement.requirement_id}",
                    problem_requirement=requirement.requirement_id,
                    requirement_ids=[requirement.requirement_id],
                    failure_mode="capability_slice_unsupported",
                    existing_capability_attempted=", ".join(
                        registry[family].capability_id for family in families if family in registry
                    )
                    or "problem router",
                    why_insufficient="; ".join(failures)
                    or "no formal capability matches the requirement",
                    scientific_consequence="requirement remains unresolved",
                    severity="critical",
                    possible_future_capability="remediation candidate after benchmark",
                )
            )
    if requirements and len(supported) == len(requirements):
        overall = CapabilityVerdict.SUPPORTED
    elif supported or partially_supported:
        overall = CapabilityVerdict.PARTIALLY_SUPPORTED
    else:
        overall = CapabilityVerdict.UNSUPPORTED
    assessment = CapabilityFitAssessment(
        assessment_id=f"capfit-{decision.routing_id}",
        problem_requirement_ids=[requirement.requirement_id for requirement in requirements],
        route_task_families=[family.value for family in decision.required_task_families],
        required_capabilities=sorted({family.value for family in decision.required_task_families}),
        available_capabilities=sorted(
            {capability_id for fit in fits for capability_id in fit.available_capabilities}
        ),
        supported=supported,
        partially_supported=partially_supported,
        unsupported=unsupported,
        requirement_fits=fits,
        blocking_gaps=gaps,
        evidence_for_capability=sorted(evidence),
        verdict=overall,
        assessment_complete=True,
        core_requirement_count=len(requirements),
        supported_core_requirement_count=len(supported),
        partial_core_requirement_count=len(partially_supported),
        unsupported_core_requirement_count=len(unsupported),
    )
    return assessment.model_copy(
        update={
            "assessment_hash": _hash_payload(
                assessment.model_dump(mode="json"), exclude=("assessment_hash",)
            )
        }
    )


__all__ = [
    "assess_capability_fit",
    "capability_registry",
    "descriptor_hash",
]
