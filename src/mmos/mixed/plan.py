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
    families = set(routing.required_task_families)
    if families != {TaskFamily.FORECASTING, TaskFamily.OPTIMIZATION}:
        raise MixedPlanError(
            "first mixed slice supports exactly forecasting -> optimization; "
            f"got {sorted(task.value for task in families)}"
        )
    hints = {
        (hint.from_task, hint.to_task) for hint in routing.dependency_hints
    }
    if (TaskFamily.FORECASTING, TaskFamily.OPTIMIZATION) not in hints:
        raise MixedPlanError(
            "routing must declare a forecasting -> optimization dependency hint"
        )
    forecast_routes = [
        route
        for route in routing.subproblem_routes
        if TaskFamily.FORECASTING in route.task_families
    ]
    optimization_routes = [
        route
        for route in routing.subproblem_routes
        if TaskFamily.OPTIMIZATION in route.task_families
    ]
    if len(forecast_routes) != 1 or len(optimization_routes) != 1:
        raise MixedPlanError(
            "first mixed slice requires exactly one forecasting and one optimization "
            "subproblem route"
        )
    forecast_subproblem = forecast_routes[0].subproblem_id
    optimization_subproblem = optimization_routes[0].subproblem_id
    if forecast_subproblem not in {item.subproblem_id for item in problem.subproblems}:
        raise MixedPlanError("forecast subproblem is not in the ProblemSpec")
    if optimization_subproblem not in {item.subproblem_id for item in problem.subproblems}:
        raise MixedPlanError("optimization subproblem is not in the ProblemSpec")

    nodes = [
        TaskNode(
            node_id="forecast_node",
            subproblem_id=forecast_subproblem,
            task_family=TaskFamily.FORECASTING,
            required_inputs=["historical_demand"],
            produced_outputs=["demand_forecast"],
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
    draft = MixedExecutionPlan(
        plan_id="plan-pending",
        project_id=paths.root.name,
        problem_hash=problem_hash,
        problem_spec_hash=problem_spec_hash,
        routing_id=routing.routing_id,
        nodes=nodes,
        edges=edges,
        entry_nodes=["forecast_node"],
        terminal_nodes=["optimization_node"],
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
