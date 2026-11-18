from pathlib import Path

import pytest

from mmos.routing.audit import audit_routing
from mmos.routing.benchmark import load_routing_manifest
from mmos.routing.policy import route_problem_spec
from mmos.routing.signals import validate_signal_extraction
from mmos.schemas.problem import ProblemSpec, SubProblem
from mmos.schemas.routing import (
    DependencyHint,
    ExecutionStatus,
    ProblemSignalExtraction,
    ProblemSignals,
    RoutingStatus,
    SubproblemSignals,
)
from mmos.schemas.states import ProjectState, TaskFamily

MANIFEST = Path("benchmarks/routing/suite_v1/manifest.json")


def make_problem(
    *subproblems: tuple[str, list[str]]
) -> ProblemSpec:
    return ProblemSpec(
        problem_id="route-test",
        title="Routing test",
        overall_goal="Test problem",
        subproblems=[
            SubProblem(
                subproblem_id=subproblem_id,
                statement=f"Statement for {subproblem_id}",
                goal=f"Goal for {subproblem_id}",
                inputs=["input"],
                outputs=["output"],
                depends_on=depends_on,
            )
            for subproblem_id, depends_on in subproblems
        ],
    )


def make_extraction(
    *signals: tuple[str, ProblemSignals]
) -> ProblemSignalExtraction:
    return ProblemSignalExtraction(
        subproblems=[
            SubproblemSignals(subproblem_id=subproblem_id, signals=signal)
            for subproblem_id, signal in signals
        ]
    )


def route(problem: ProblemSpec, extraction: ProblemSignalExtraction):
    return route_problem_spec(
        problem,
        extraction,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        provider="test",
        model="test",
    )


def test_prediction_boundary() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_supervised_target=True,
                    target_description="churn label",
                    reported_evaluation_metrics=["accuracy", "F1"],
                ),
            )
        ),
    )
    assert decision.status is RoutingStatus.SINGLE_TASK
    assert decision.primary_task is TaskFamily.PREDICTION
    assert decision.execution_status is ExecutionStatus.READY_FOR_SINGLE_TASK


def test_forecasting_boundary() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_supervised_target=True,
                    requires_future_prediction=True,
                    temporal_order_matters=True,
                ),
            )
        ),
    )
    assert decision.primary_task is TaskFamily.FORECASTING


def test_optimization_and_threshold_boundary() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_decision_variables=True,
                    has_hard_constraints=True,
                    objective_count=2,
                    objectives_simultaneously_optimized=False,
                ),
            )
        ),
    )
    assert decision.primary_task is TaskFamily.OPTIMIZATION
    assert "threshold_objective_as_constraint" in decision.reason_codes


def test_multiobjective_boundary() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_decision_variables=True,
                    has_hard_constraints=True,
                    objective_count=2,
                    objectives_simultaneously_optimized=True,
                    objective_preference_given=False,
                ),
            )
        ),
    )
    assert decision.primary_task is TaskFamily.MULTI_OBJECTIVE


def test_simulation_and_graph_boundaries() -> None:
    simulation = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    stochastic_process_evolution=True,
                    requires_replications_or_distributional_output=True,
                ),
            )
        ),
    )
    assert simulation.primary_task is TaskFamily.SIMULATION

    graph = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    graph_native_problem=True,
                    graph_structure_is_primary=True,
                ),
            )
        ),
    )
    assert graph.primary_task is TaskFamily.GRAPH


def test_neural_network_is_not_graph() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_supervised_target=True,
                    requires_future_prediction=True,
                    temporal_order_matters=True,
                    graph_native_problem=False,
                    graph_structure_is_primary=False,
                ),
            )
        ),
    )
    assert decision.primary_task is TaskFamily.FORECASTING
    assert TaskFamily.GRAPH not in decision.required_task_families


def test_split_simulation_signals_are_combined() -> None:
    problem = make_problem(("q1", []), ("q2", ["q1"]))
    extraction = make_extraction(
        ("q1", ProblemSignals(stochastic_process_evolution=True)),
        ("q2", ProblemSignals(requires_replications_or_distributional_output=True)),
    )
    decision = route(problem, extraction)

    assert decision.status is RoutingStatus.SINGLE_TASK
    assert decision.primary_task is TaskFamily.SIMULATION
    assert TaskFamily.SIMULATION in decision.subproblem_routes[0].task_families


def test_multiple_metrics_are_not_multiobjective() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_supervised_target=True,
                    reported_evaluation_metrics=["RMSE", "MAE", "R2"],
                ),
            )
        ),
    )
    assert decision.primary_task is TaskFamily.PREDICTION


def test_mixed_task_is_not_forced_single() -> None:
    problem = make_problem(("q1", []), ("q2", ["q1"]))
    extraction = make_extraction(
        (
            "q1",
            ProblemSignals(
                has_supervised_target=True,
                requires_future_prediction=True,
                temporal_order_matters=True,
            ),
        ),
        (
            "q2",
            ProblemSignals(
                has_decision_variables=True,
                has_hard_constraints=True,
                objective_count=1,
            ),
        ),
    )
    decision = route(problem, extraction)

    assert decision.status is RoutingStatus.MULTI_TASK
    assert decision.required_task_families == [
        TaskFamily.FORECASTING,
        TaskFamily.OPTIMIZATION,
    ]
    assert decision.dependency_hints == [
        DependencyHint(
            from_task=TaskFamily.FORECASTING, to_task=TaskFamily.OPTIMIZATION
        )
    ]
    assert decision.execution_status is ExecutionStatus.BLOCKED_PENDING_MIXED_DAG
    assert decision.suggested_command is None


def test_ambiguous_and_unsupported_are_explicit() -> None:
    ambiguous = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    insufficient_information=True,
                    missing_information=["no target or decision specified"],
                ),
            )
        ),
    )
    assert ambiguous.status is RoutingStatus.AMBIGUOUS
    assert ambiguous.clarifications_needed
    assert ambiguous.primary_task is None

    unsupported = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    unsupported_requirements=["deep reinforcement learning control"]
                ),
            )
        ),
    )
    assert unsupported.status is RoutingStatus.UNSUPPORTED
    assert unsupported.unsupported_requirements
    assert unsupported.primary_task is None


def test_auditor_flags_mixed_and_downgrades() -> None:
    problem = make_problem(("q1", []), ("q2", ["q1"]))
    extraction = make_extraction(
        (
            "q1",
            ProblemSignals(
                has_supervised_target=True,
                requires_future_prediction=True,
                temporal_order_matters=True,
            ),
        ),
        (
            "q2",
            ProblemSignals(
                has_decision_variables=True,
                has_hard_constraints=True,
                objective_count=1,
            ),
        ),
    )
    decision = route(problem, extraction)
    tampered = decision.model_copy(
        update={
            "status": RoutingStatus.SINGLE_TASK,
            "primary_task": TaskFamily.FORECASTING,
            "required_task_families": [TaskFamily.FORECASTING],
            "dependency_hints": [],
            "execution_status": ExecutionStatus.READY_FOR_SINGLE_TASK,
            "suggested_command": "mmos run ...",
        }
    )
    audit = audit_routing(tampered, problem, extraction)

    assert audit.passed is False
    assert audit.critical_unsafe is True
    assert any("critical_mixed_as_single" in issue.code for issue in audit.issues)


def test_auditor_passes_valid_routes() -> None:
    problem = make_problem(("q1", []))
    extraction = make_extraction(
        (
            "q1",
            ProblemSignals(
                has_decision_variables=True,
                has_hard_constraints=True,
                objective_count=2,
                objectives_simultaneously_optimized=True,
            ),
        )
    )
    decision = route(problem, extraction)
    audit = audit_routing(decision, problem, extraction)

    assert decision.primary_task is TaskFamily.MULTI_OBJECTIVE
    assert audit.passed is True
    assert audit.critical_unsafe is False


def test_routing_manifest_is_frozen_and_well_formed() -> None:
    manifest = load_routing_manifest(MANIFEST)

    assert manifest.suite_id == "routing-suite-v1"
    assert len(manifest.cases) == 24
    assert len(manifest.manifest_hash) == 64
    expected_counts = {
        "prediction": 3,
        "forecasting": 3,
        "optimization": 3,
        "multi_objective": 3,
        "simulation": 3,
        "graph": 3,
        "multi": 3,
        "ambiguous": 2,
        "unsupported": 1,
    }
    statuses = [case.expected_status for case in manifest.cases]
    assert statuses.count(RoutingStatus.SINGLE_TASK) == 18
    assert statuses.count(RoutingStatus.MULTI_TASK) == 3
    assert statuses.count(RoutingStatus.AMBIGUOUS) == 2
    assert statuses.count(RoutingStatus.UNSUPPORTED) == 1
    assert sum(expected_counts.values()) == 24


def test_task_family_enum_still_supported() -> None:
    assert TaskFamily.MULTI_OBJECTIVE in TaskFamily
    assert ProjectState.RELEASED.value == "RELEASED"


def test_signal_semantics_reject_missing_and_duplicate_ids() -> None:
    problem = make_problem(("q1", []), ("q2", []))
    missing = make_extraction(("q1", ProblemSignals()))
    with pytest.raises(ValueError, match="missing"):
        validate_signal_extraction(missing, problem)
    duplicate = make_extraction(
        ("q1", ProblemSignals()),
        ("q1", ProblemSignals()),
        ("q2", ProblemSignals()),
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_signal_extraction(duplicate, problem)
    valid = make_extraction(("q1", ProblemSignals()), ("q2", ProblemSignals()))
    validate_signal_extraction(valid, problem)


def test_statistical_inference_route() -> None:
    decision = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    requires_parameter_inference=True,
                    requires_hypothesis_test=True,
                ),
            )
        ),
    )
    assert decision.status is RoutingStatus.SINGLE_TASK
    assert decision.primary_task is TaskFamily.STATISTICAL_INFERENCE
    assert decision.execution_status is ExecutionStatus.READY_FOR_SINGLE_TASK


def test_inference_does_not_absorb_prediction_or_optimization() -> None:
    prediction = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_supervised_target=True,
                    target_description="future class label",
                ),
            )
        ),
    )
    assert prediction.primary_task is TaskFamily.PREDICTION
    optimization = route(
        make_problem(("q1", [])),
        make_extraction(
            (
                "q1",
                ProblemSignals(
                    has_decision_variables=True,
                    has_hard_constraints=True,
                    objective_count=1,
                ),
            )
        ),
    )
    assert optimization.primary_task is TaskFamily.OPTIMIZATION


def test_inference_and_optimization_multi_task() -> None:
    decision = route(
        make_problem(("q1", []), ("q2", ["q1"])),
        make_extraction(
            ("q1", ProblemSignals(requires_parameter_inference=True)),
            (
                "q2",
                ProblemSignals(
                    has_decision_variables=True,
                    has_hard_constraints=True,
                    objective_count=1,
                ),
            ),
        ),
    )
    assert decision.status is RoutingStatus.MULTI_TASK
    assert set(decision.required_task_families) == {
        TaskFamily.STATISTICAL_INFERENCE,
        TaskFamily.OPTIMIZATION,
    }
