from mmos.schemas.problem import ProblemSpec
from mmos.schemas.routing import (
    ExecutionStatus,
    ProblemSignalExtraction,
    RoutingAuditResult,
    RoutingDecision,
    RoutingIssue,
    RoutingStatus,
)
from mmos.schemas.states import TaskFamily


def _issue(code: str, message: str) -> RoutingIssue:
    return RoutingIssue(code=code, message=message)


def audit_routing(
    decision: RoutingDecision,
    problem: ProblemSpec,
    extraction: ProblemSignalExtraction,
) -> RoutingAuditResult:
    issues: list[RoutingIssue] = []
    signals = [item.signals for item in extraction.subproblems]
    families = set(decision.required_task_families)
    critical = False

    if not decision.routing_id or not decision.problem_hash or not decision.problem_spec_hash:
        issues.append(_issue("incomplete_identity", "routing identity hashes are incomplete"))
    if decision.status is RoutingStatus.SINGLE_TASK:
        if len(decision.required_task_families) != 1 or decision.primary_task is None:
            issues.append(_issue("single_task_shape", "SINGLE_TASK must name one primary task"))
        route_families = {
            family
            for route in decision.subproblem_routes
            for family in route.task_families
        }
        if any(len(route.task_families) > 1 for route in decision.subproblem_routes) or (
            len(route_families) > 1
        ):
            issues.append(
                _issue(
                    "critical_mixed_as_single",
                    "subproblem routes require multiple families",
                )
            )
            critical = True
        if decision.dependency_hints:
            issues.append(
                _issue(
                    "critical_dependency_as_single",
                    "dependency hints exist but status is single",
                )
            )
            critical = True
        if any(item.unsupported_requirements for item in signals):
            issues.append(
                _issue("critical_unsupported_as_single", "unsupported requirements present")
            )
            critical = True
        if any(item.insufficient_information for item in signals):
            issues.append(
                _issue("critical_insufficient_as_single", "insufficient information present")
            )
            critical = True
    elif decision.status is RoutingStatus.MULTI_TASK:
        if len(set(decision.required_task_families)) < 2:
            issues.append(_issue("multi_task_shape", "MULTI_TASK needs at least two families"))
        if decision.execution_status is not ExecutionStatus.BLOCKED_PENDING_MIXED_DAG:
            issues.append(_issue("mixed_execution", "MULTI_TASK must not be executable yet"))
        if decision.suggested_command is not None:
            issues.append(_issue("mixed_execution", "MULTI_TASK must not suggest execution"))
    elif decision.status is RoutingStatus.AMBIGUOUS:
        if not decision.clarifications_needed:
            issues.append(_issue("ambiguous_clarification", "AMBIGUOUS needs clarifications"))
        if decision.required_task_families:
            issues.append(_issue("ambiguous_task", "AMBIGUOUS must not select a task"))
    elif decision.status is RoutingStatus.UNSUPPORTED:
        if not decision.unsupported_requirements:
            issues.append(_issue("unsupported_detail", "UNSUPPORTED needs a requirement"))
        if decision.required_task_families:
            issues.append(_issue("unsupported_task", "UNSUPPORTED must not select a task"))

    if decision.status in (RoutingStatus.AMBIGUOUS, RoutingStatus.UNSUPPORTED) and (
        decision.suggested_command is not None
    ):
        issues.append(_issue("unsafe_suggestion", "blocked routing must not suggest execution"))

    if (
        any(item.insufficient_information for item in signals)
        and decision.status not in (RoutingStatus.AMBIGUOUS, RoutingStatus.UNSUPPORTED)
        and not decision.clarifications_needed
    ):
        issues.append(
            _issue("missing_clarification", "insufficient signals need clarification")
        )

    if any(item.unsupported_requirements for item in signals) and decision.status is not (
        RoutingStatus.UNSUPPORTED
    ):
        issues.append(_issue("unsupported_ignored", "unsupported requirements were ignored"))

    tradeoff = any(
        item.objectives_simultaneously_optimized
        and item.objective_count >= 2
        and not item.objective_preference_given
        for item in signals
    )
    if tradeoff and TaskFamily.MULTI_OBJECTIVE not in families and decision.status is not (
        RoutingStatus.UNSUPPORTED
    ):
        issues.append(
            _issue(
                "multiobjective_downgraded",
                "trade-off objectives not routed to multi_objective",
            )
        )

    threshold = any(
        item.objective_count >= 2 and not item.objectives_simultaneously_optimized
        for item in signals
    )
    if (
        threshold
        and TaskFamily.OPTIMIZATION not in families
        and TaskFamily.MULTI_OBJECTIVE in families
    ):
        issues.append(
            _issue("constraint_as_objective", "threshold objective treated as trade-off")
        )

    future = any(
        item.requires_future_prediction and item.temporal_order_matters for item in signals
    )
    if future and TaskFamily.FORECASTING not in families and TaskFamily.PREDICTION in families:
        issues.append(_issue("forecasting_downgraded", "future prediction routed as prediction"))

    graph_native = any(
        item.graph_native_problem and item.graph_structure_is_primary for item in signals
    )
    if graph_native and TaskFamily.GRAPH not in families:
        issues.append(_issue("graph_downgraded", "native graph problem not routed to graph"))
    if TaskFamily.GRAPH in families and not graph_native:
        issues.append(_issue("graph_false_positive", "graph selected without graph-native signals"))

    simulation = any(
        item.stochastic_process_evolution
        and item.requires_replications_or_distributional_output
        for item in signals
    )
    if simulation and TaskFamily.SIMULATION not in families:
        issues.append(_issue("simulation_downgraded", "simulation signals not routed"))

    supervised = any(item.has_supervised_target for item in signals)
    if TaskFamily.PREDICTION in families and not supervised:
        issues.append(_issue("prediction_false_positive", "prediction lacks supervised target"))
    if TaskFamily.FORECASTING in families and not future:
        issues.append(_issue("forecasting_false_positive", "forecasting lacks future prediction"))
    if TaskFamily.MULTI_OBJECTIVE in families and not tradeoff:
        issues.append(_issue("multiobjective_false_positive", "multi-objective lacks trade-off"))
    if TaskFamily.OPTIMIZATION in families and not any(
        item.has_decision_variables and item.has_hard_constraints and item.objective_count >= 1
        for item in signals
    ):
        issues.append(_issue("optimization_false_positive", "optimization lacks decision signals"))
    if TaskFamily.SIMULATION in families and not simulation:
        issues.append(_issue("simulation_false_positive", "simulation lacks stochastic signals"))

    audit = RoutingAuditResult(
        routing_id=decision.routing_id,
        passed=not issues and not critical,
        critical_unsafe=critical,
        issues=issues,
    )
    return audit


__all__ = ["audit_routing"]
