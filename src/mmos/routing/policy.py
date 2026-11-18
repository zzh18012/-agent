import hashlib
import itertools
import json
from datetime import UTC, datetime

from mmos.schemas.problem import ProblemSpec
from mmos.schemas.routing import (
    CapabilityLevel,
    CapabilityReport,
    DependencyHint,
    ExecutionStatus,
    ProblemSignalExtraction,
    ProblemSignals,
    RoutingDecision,
    RoutingStatus,
    SubproblemRoute,
)
from mmos.schemas.states import TaskFamily

_FAMILY_RANK = {
    TaskFamily.PREDICTION: 0,
    TaskFamily.FORECASTING: 0,
    TaskFamily.SIMULATION: 1,
    TaskFamily.GRAPH: 1,
    TaskFamily.OPTIMIZATION: 2,
    TaskFamily.MULTI_OBJECTIVE: 2,
    TaskFamily.STATISTICAL_INFERENCE: 0,
}

_CAPABILITY_SCOPE = {
    TaskFamily.PREDICTION: (
        CapabilityLevel.FULL,
        "tabular supervised regression/classification",
        [],
    ),
    TaskFamily.FORECASTING: (
        CapabilityLevel.FULL,
        "tabular temporal forecasting with chronological validation",
        [],
    ),
    TaskFamily.OPTIMIZATION: (
        CapabilityLevel.PARTIAL,
        "0/1 knapsack MILP with one objective and CBC",
        ["single-objective MILP slice only"],
    ),
    TaskFamily.SIMULATION: (
        CapabilityLevel.PARTIAL,
        "stable M/M/1 discrete-event simulation",
        ["one queueing model only"],
    ),
    TaskFamily.GRAPH: (
        CapabilityLevel.PARTIAL,
        "single-source/single-sink directed max-flow",
        ["max-flow only; no shortest path, matching, centrality, or dynamic graphs"],
    ),
    TaskFamily.MULTI_OBJECTIVE: (
        CapabilityLevel.PARTIAL,
        "bi-objective binary project-portfolio epsilon-constraint",
        ["two maximize objectives, binary portfolio, one budget constraint"],
    ),
    TaskFamily.STATISTICAL_INFERENCE: (
        CapabilityLevel.PARTIAL,
        "exact binomial inference, single-sampling plans, registered forward-model NLS",
        ["four narrow inference slices only"],
    ),
}


def _subproblem_families(
    signals: ProblemSignals,
) -> tuple[list[TaskFamily], list[str]]:
    reasons: list[str] = []
    if signals.unsupported_requirements:
        return [], ["unsupported_requirement"]
    families: list[TaskFamily] = []
    graph_native = signals.graph_native_problem and signals.graph_structure_is_primary
    if graph_native:
        families.append(TaskFamily.GRAPH)
        reasons.extend(["graph_native_problem", "graph_structure_primary"])
    decision_problem = (
        not graph_native
        and signals.has_decision_variables
        and signals.has_hard_constraints
        and signals.objective_count >= 1
    )
    if decision_problem:
        if (
            signals.objectives_simultaneously_optimized
            and signals.objective_count >= 2
            and not signals.objective_preference_given
        ):
            families.append(TaskFamily.MULTI_OBJECTIVE)
            reasons.extend(["decision_variables", "hard_constraints", "tradeoff_objectives"])
        else:
            families.append(TaskFamily.OPTIMIZATION)
            reasons.extend(["decision_variables", "hard_constraints", "single_objective"])
            if signals.objective_count >= 2 and not (signals.objectives_simultaneously_optimized):
                reasons.append("threshold_objective_as_constraint")
    if (
        signals.stochastic_process_evolution
        and signals.requires_replications_or_distributional_output
    ):
        families.append(TaskFamily.SIMULATION)
        reasons.extend(["stochastic_process", "replications_or_distribution"])
    if signals.requires_future_prediction and signals.temporal_order_matters:
        families.append(TaskFamily.FORECASTING)
        reasons.extend(["future_prediction", "temporal_order_matters"])
    elif signals.has_supervised_target:
        families.append(TaskFamily.PREDICTION)
        reasons.append("supervised_target")
    if (
        signals.requires_parameter_inference
        or signals.requires_hypothesis_test
        or signals.requires_acceptance_sampling
        or signals.requires_nonlinear_parameter_estimation
        or signals.requires_parameter_uncertainty
    ):
        families.append(TaskFamily.STATISTICAL_INFERENCE)
        reasons.append("statistical_inference_requested")
    if signals.reported_evaluation_metrics and families == [TaskFamily.PREDICTION]:
        reasons.append("evaluation_metrics_not_objectives")
    families = sorted(set(families), key=lambda family: (_FAMILY_RANK[family], family.value))
    return families, reasons


def _capability(task: TaskFamily) -> CapabilityReport:
    level, scope, limitations = _CAPABILITY_SCOPE[task]
    return CapabilityReport(
        task_family=task,
        current_supported_scope=scope,
        execution_readiness=level,
        limitations=limitations,
    )


def _suggested_command(task: TaskFamily) -> str:
    mapping = {
        TaskFamily.PREDICTION: "mmos run <project> --target ... --feature ...",
        TaskFamily.FORECASTING: (
            "mmos run <project> --target ... --feature ... --time-column ... --horizon ..."
        ),
        TaskFamily.OPTIMIZATION: "mmos optimize <project> <optimization_spec.json>",
        TaskFamily.SIMULATION: "mmos simulate <project> <simulation_spec.json>",
        TaskFamily.GRAPH: "mmos graph-run <project> <graph_spec.json>",
        TaskFamily.MULTI_OBJECTIVE: "mmos multi-optimize <project> <instance_spec.json>",
        TaskFamily.STATISTICAL_INFERENCE: "mmos infer <project> <inference_spec.json>",
    }
    return mapping[task]


def _ordered_families(families: set[TaskFamily]) -> list[TaskFamily]:
    return sorted(families, key=lambda family: (_FAMILY_RANK[family], family.value))


def route_problem_spec(
    problem: ProblemSpec,
    extraction: ProblemSignalExtraction,
    *,
    problem_hash: str,
    problem_spec_hash: str,
    provider: str,
    model: str,
    git_commit: str = "unknown",
) -> RoutingDecision:
    signals_by_id = {item.subproblem_id: item.signals for item in extraction.subproblems}
    subproblem_routes: list[SubproblemRoute] = []
    required: set[TaskFamily] = set()
    reasons: set[str] = set()
    clarifications: list[str] = []
    unsupported: list[str] = []
    dependency_hints: set[tuple[TaskFamily, TaskFamily]] = set()
    family_by_subproblem: dict[str, list[TaskFamily]] = {}
    seen_ids = set()

    for subproblem in problem.subproblems:
        signals = signals_by_id.get(subproblem.subproblem_id)
        if signals is None:
            signals = ProblemSignals(
                insufficient_information=True,
                missing_information=[f"no structured signals for {subproblem.subproblem_id}"],
            )
        families, sub_reasons = _subproblem_families(signals)
        family_by_subproblem[subproblem.subproblem_id] = families
        required.update(families)
        reasons.update(sub_reasons)
        clarifications.extend(signals.missing_information)
        unsupported.extend(signals.unsupported_requirements)
        seen_ids.add(subproblem.subproblem_id)
        subproblem_routes.append(
            SubproblemRoute(
                subproblem_id=subproblem.subproblem_id,
                task_families=families,
                reason_codes=sub_reasons,
                required_inputs=subproblem.inputs,
                expected_outputs=subproblem.outputs,
            )
        )
        if len(families) > 1:
            ordered = _ordered_families(set(families))
            for left, right in itertools.pairwise(ordered):
                if _FAMILY_RANK[left] < _FAMILY_RANK[right]:
                    dependency_hints.add((left, right))
                    reasons.add("mixed_task_dependency")

    for subproblem in problem.subproblems:
        for dependency in subproblem.depends_on:
            for from_family in family_by_subproblem.get(dependency, []):
                for to_family in family_by_subproblem.get(subproblem.subproblem_id, []):
                    if from_family != to_family:
                        dependency_hints.add((from_family, to_family))
                        reasons.add("mixed_task_dependency")

    global_simulation = any(
        item.stochastic_process_evolution for item in signals_by_id.values()
    ) and any(
        item.requires_replications_or_distributional_output for item in signals_by_id.values()
    )
    if global_simulation and TaskFamily.SIMULATION not in required:
        required.add(TaskFamily.SIMULATION)
        reasons.update(["stochastic_process", "replications_or_distribution"])
        for index, route in enumerate(subproblem_routes):
            source = signals_by_id.get(route.subproblem_id)
            if source is not None and source.stochastic_process_evolution:
                families = sorted(
                    set(route.task_families) | {TaskFamily.SIMULATION},
                    key=lambda family: (_FAMILY_RANK[family], family.value),
                )
                subproblem_routes[index] = route.model_copy(
                    update={
                        "task_families": families,
                        "reason_codes": sorted(
                            set(route.reason_codes)
                            | {"stochastic_process", "replications_or_distribution"}
                        ),
                    }
                )
                break

    extra_ids = sorted(set(signals_by_id) - seen_ids)
    if extra_ids:
        clarifications.append(f"unmatched signal subproblems: {', '.join(extra_ids)}")
    if not required:
        clarifications.extend(problem.ambiguities)
    clarifications = sorted(set(item for item in clarifications if item))
    unsupported = sorted(set(item for item in unsupported if item))
    required_families = _ordered_families(required)
    insufficient = any(item.insufficient_information for item in signals_by_id.values())

    if unsupported:
        status = RoutingStatus.UNSUPPORTED
        primary = None
        execution = ExecutionStatus.BLOCKED_UNSUPPORTED
        suggested = None
        required_families = []
        capabilities = []
        ordered_hints: list[tuple[TaskFamily, TaskFamily]] = []
        reasons.add("unsupported_requirement")
    elif insufficient:
        status = RoutingStatus.AMBIGUOUS
        primary = None
        execution = ExecutionStatus.BLOCKED_NEEDS_CLARIFICATION
        suggested = None
        required_families = []
        capabilities = []
        ordered_hints = []
        if not clarifications:
            clarifications = [
                "the problem does not specify a supported prediction, decision, "
                "simulation, or graph objective"
            ]
        reasons.add("insufficient_information")
    elif len(required_families) >= 2:
        status = RoutingStatus.MULTI_TASK
        primary = None
        execution = ExecutionStatus.BLOCKED_PENDING_MIXED_DAG
        suggested = None
        capabilities = [_capability(family) for family in required_families]
        ordered_hints = sorted(dependency_hints, key=lambda pair: (pair[0].value, pair[1].value))
        reasons.add("mixed_task_dependency")
    elif len(required_families) == 1:
        status = RoutingStatus.SINGLE_TASK
        primary = required_families[0]
        execution = ExecutionStatus.READY_FOR_SINGLE_TASK
        suggested = _suggested_command(primary)
        capabilities = [_capability(primary)]
        ordered_hints = []
    else:
        status = RoutingStatus.AMBIGUOUS
        primary = None
        execution = ExecutionStatus.BLOCKED_NEEDS_CLARIFICATION
        suggested = None
        required_families = []
        capabilities = []
        ordered_hints = []
        if not clarifications:
            clarifications = [
                "the problem does not specify a supported prediction, decision, "
                "simulation, or graph objective"
            ]
        reasons.add("insufficient_information")

    if status is RoutingStatus.MULTI_TASK and clarifications:
        reasons.add("clarification_needed")
    payload = {
        "problem_hash": problem_hash,
        "extraction": extraction.model_dump(mode="json"),
    }
    routing_id = (
        "route-"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
    )
    return RoutingDecision(
        routing_id=routing_id,
        problem_hash=problem_hash,
        problem_spec_hash=problem_spec_hash,
        status=status,
        primary_task=primary,
        required_task_families=required_families,
        subproblem_routes=subproblem_routes,
        dependency_hints=[
            DependencyHint(from_task=left, to_task=right) for left, right in ordered_hints
        ],
        reason_codes=sorted(reasons),
        clarifications_needed=clarifications,
        unsupported_requirements=unsupported,
        capabilities=capabilities,
        execution_status=execution,
        suggested_command=suggested,
        provider=provider,
        model=model,
        git_commit=git_commit,
        created_at=datetime.now(UTC).replace(microsecond=0),
    )


__all__ = ["route_problem_spec"]
