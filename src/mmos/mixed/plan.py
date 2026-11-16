from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.mixed import ArtifactEdge, MixedExecutionPlan, MixedSpec, TaskNode
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.routing import RoutingDecision, RoutingStatus
from mmos.schemas.states import TaskFamily


class MixedPlanError(RuntimeError):
    pass


def build_mixed_plan(
    paths: ProjectPaths,
    problem: ProblemSpec,
    routing: RoutingDecision,
    spec: MixedSpec,
    *,
    problem_hash: str,
    problem_spec_hash: str,
) -> MixedExecutionPlan:
    if routing.status is not RoutingStatus.MULTI_TASK:
        raise MixedPlanError(
            f"mixed plan requires MULTI_TASK routing, got {routing.status.value}"
        )
    simulation_enabled = spec.simulation_replications is not None
    families = set(routing.required_task_families)
    expected = {TaskFamily.FORECASTING, TaskFamily.OPTIMIZATION}
    if simulation_enabled:
        expected.add(TaskFamily.SIMULATION)
    if families != expected:
        raise MixedPlanError(
            "mixed slice supports forecasting -> optimization "
            "(+ simulation when requested); "
            f"got {sorted(task.value for task in families)}"
        )
    hints = {
        (hint.from_task, hint.to_task) for hint in routing.dependency_hints
    }
    required_hints = {(TaskFamily.FORECASTING, TaskFamily.OPTIMIZATION)}
    if simulation_enabled:
        required_hints.add((TaskFamily.FORECASTING, TaskFamily.SIMULATION))
        required_hints.add((TaskFamily.OPTIMIZATION, TaskFamily.SIMULATION))
    missing_hints = required_hints - hints
    if missing_hints:
        raise MixedPlanError(
            "routing is missing dependency hints: "
            + ", ".join(
                f"{source.value}->{target.value}"
                for source, target in sorted(missing_hints)
            )
        )
    required_families = [TaskFamily.FORECASTING, TaskFamily.OPTIMIZATION]
    if simulation_enabled:
        required_families.append(TaskFamily.SIMULATION)
    subproblem_ids: dict[TaskFamily, str] = {}
    for family in required_families:
        routes = [
            route
            for route in routing.subproblem_routes
            if family in route.task_families
        ]
        if len(routes) != 1:
            raise MixedPlanError(
                f"mixed slice requires exactly one {family.value} subproblem route"
            )
        subproblem_ids[family] = routes[0].subproblem_id
    known_subproblems = {item.subproblem_id for item in problem.subproblems}
    for family, subproblem_id in subproblem_ids.items():
        if subproblem_id not in known_subproblems:
            raise MixedPlanError(
                f"{family.value} subproblem is not in the ProblemSpec"
            )
    forecast_subproblem = subproblem_ids[TaskFamily.FORECASTING]
    optimization_subproblem = subproblem_ids[TaskFamily.OPTIMIZATION]

    forecast_outputs = ["demand_forecast"]
    if simulation_enabled:
        forecast_outputs.append("forecast_error")
    nodes = [
        TaskNode(
            node_id="forecast_node",
            subproblem_id=forecast_subproblem,
            task_family=TaskFamily.FORECASTING,
            required_inputs=["historical_demand"],
            produced_outputs=forecast_outputs,
        ),
        TaskNode(
            node_id="optimization_node",
            subproblem_id=optimization_subproblem,
            task_family=TaskFamily.OPTIMIZATION,
            required_inputs=["demand_forecast"],
            produced_outputs=["production_plan"],
        ),
    ]
    edges = [
        ArtifactEdge(
            producer_node="forecast_node",
            producer_output="demand_forecast",
            consumer_node="optimization_node",
            consumer_input="demand_forecast",
            required_artifact_type="demand_forecast",
            required_schema_version="1.0",
        )
    ]
    terminal_nodes = ["optimization_node"]
    if simulation_enabled:
        nodes.append(
            TaskNode(
                node_id="simulation_node",
                subproblem_id=subproblem_ids[TaskFamily.SIMULATION],
                task_family=TaskFamily.SIMULATION,
                required_inputs=["production_plan", "forecast_error"],
                produced_outputs=["risk_evidence"],
            )
        )
        edges.extend(
            [
                ArtifactEdge(
                    producer_node="forecast_node",
                    producer_output="forecast_error",
                    consumer_node="simulation_node",
                    consumer_input="forecast_error",
                    required_artifact_type="forecast_error",
                    required_schema_version="1.0",
                ),
                ArtifactEdge(
                    producer_node="optimization_node",
                    producer_output="production_plan",
                    consumer_node="simulation_node",
                    consumer_input="production_plan",
                    required_artifact_type="production_plan",
                    required_schema_version="1.0",
                ),
            ]
        )
        terminal_nodes = ["simulation_node"]
    draft = MixedExecutionPlan(
        plan_id="plan-pending",
        project_id=paths.root.name,
        problem_hash=problem_hash,
        problem_spec_hash=problem_spec_hash,
        routing_id=routing.routing_id,
        nodes=nodes,
        edges=edges,
        entry_nodes=["forecast_node"],
        terminal_nodes=terminal_nodes,
        spec=spec.with_hash(),
        created_at=datetime.now(UTC).replace(microsecond=0),
    )
    draft = draft.model_copy(
        update={"plan_id": f"plan-{draft.computed_hash()[:12]}"}
    )
    plan = draft.with_hash()
    plan.topological_order()
    return plan


def save_mixed_plan(paths: ProjectPaths, plan: MixedExecutionPlan) -> Path:
    paths.mixed_dir.mkdir(parents=True, exist_ok=True)
    target = paths.mixed_dir / "plan.json"
    atomic_write_text(target, plan.model_dump_json(indent=2) + "\n")
    return target


def load_mixed_plan(paths: ProjectPaths) -> MixedExecutionPlan:
    target = paths.mixed_dir / "plan.json"
    if not target.exists():
        raise FileNotFoundError(f"mixed plan not found: {target}")
    plan = MixedExecutionPlan.model_validate_json(target.read_text(encoding="utf-8"))
    if plan.plan_hash != plan.computed_hash():
        raise MixedPlanError("mixed plan hash does not match its content")
    return plan


__all__ = [
    "MixedPlanError",
    "build_mixed_plan",
    "load_mixed_plan",
    "save_mixed_plan",
]
