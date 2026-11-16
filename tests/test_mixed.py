import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from mmos.core.project import (
    ProjectPaths,
    create_project,
    load_optimization_result,
    load_project_config,
)
from mmos.core.state_machine import task_order
from mmos.evidence.store import load_claims
from mmos.mixed.artifacts import (
    ArtifactContractError,
    build_forecast_artifact,
    contract_for,
    load_envelope,
    save_forecast_artifact,
    verify_forecast_artifact,
)
from mmos.mixed.executor import run_mixed_execution
from mmos.mixed.plan import (
    MixedPlanError,
    build_mixed_plan,
    save_mixed_plan,
)
from mmos.mixed.validation import run_record_path, validate_mixed_experiment
from mmos.mixed_pipeline import MixedPipelineBlocked, run_mixed_project
from mmos.orchestration import Orchestrator
from mmos.routing.policy import route_problem_spec
from mmos.schemas.mixed import (
    ArtifactEdge,
    DAGStatus,
    MixedExecutionPlan,
    MixedSpec,
    NodeStatus,
    TaskNode,
)
from mmos.schemas.problem import ProblemSpec, SubProblem
from mmos.schemas.project import SourceRole
from mmos.schemas.review import (
    AttackReport,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)
from mmos.schemas.routing import ProblemSignalExtraction, ProblemSignals, SubproblemSignals
from mmos.schemas.states import ExecutionMode, ProjectState, TaskFamily

DATA_CSV = """period,demand
2024-01,100
2024-02,106
2024-03,111
2024-04,109
2024-05,106
2024-06,112
2024-07,118
2024-08,123
2024-09,121
2024-10,119
2024-11,117
2024-12,122
2025-01,124
2025-02,130
2025-03,135
2025-04,133
2025-05,130
2025-06,136
2025-07,142
2025-08,147
2025-09,145
2025-10,143
2025-11,141
2025-12,146
"""

PROBLEM_MD = """# Demand forecast then production plan

Forecast monthly demand for the next three months from the historical series, then use that
forecast to plan production under monthly capacities and minimize production plus holding
cost.
"""


def mixed_spec() -> MixedSpec:
    return MixedSpec(
        mixed_id="mixed-test",
        problem_id="mixed-test",
        target_column="demand",
        time_column="period",
        forecast_horizon=3,
        forecast_timestamps=["2026-01", "2026-02", "2026-03"],
        forecast_model_id="naive.drift",
        n_splits=4,
        unit="units",
        planning_periods=["2026-01", "2026-02", "2026-03"],
        capacities=[170.0, 140.0, 170.0],
        production_costs=[12.0, 12.0, 12.0],
        holding_costs=[2.0, 2.0, 2.0],
    ).with_hash()


def mixed_problem() -> ProblemSpec:
    return ProblemSpec(
        problem_id="mixed-test",
        title="Demand forecast then production plan",
        overall_goal="Forecast demand and plan production",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="Forecast demand",
                goal="Forecast",
                inputs=["history"],
                outputs=["demand forecast"],
                depends_on=[],
            ),
            SubProblem(
                subproblem_id="q2",
                statement="Plan production",
                goal="Plan",
                inputs=["demand forecast"],
                outputs=["production plan"],
                depends_on=["q1"],
            ),
        ],
    )


def multi_task_decision():
    extraction = ProblemSignalExtraction(
        subproblems=[
            SubproblemSignals(
                subproblem_id="q1",
                signals=ProblemSignals(
                    has_supervised_target=True,
                    requires_future_prediction=True,
                    temporal_order_matters=True,
                ),
            ),
            SubproblemSignals(
                subproblem_id="q2",
                signals=ProblemSignals(
                    has_decision_variables=True,
                    has_hard_constraints=True,
                    objective_count=1,
                ),
            ),
        ]
    )
    return route_problem_spec(
        mixed_problem(),
        extraction,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        provider="test",
        model="test",
    )


def make_plan(paths: ProjectPaths) -> MixedExecutionPlan:
    return build_mixed_plan(
        paths,
        mixed_problem(),
        multi_task_decision(),
        mixed_spec(),
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
    )


def project(tmp_path: Path, name: str = "mixed") -> ProjectPaths:
    paths, _ = create_project(
        tmp_path / "projects",
        name,
        task=TaskFamily.FORECASTING,
        execution_mode=ExecutionMode.MULTI_TASK,
    )
    return paths


def prepared_project(
    tmp_path: Path, name: str = "mixed-run"
) -> tuple[ProjectPaths, MixedExecutionPlan]:
    paths = project(tmp_path, name)
    problem = tmp_path / f"{name}-problem.md"
    problem.write_text(PROBLEM_MD, encoding="utf-8")
    data = tmp_path / f"{name}-data.csv"
    data.write_text(DATA_CSV, encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.ingest([data], SourceRole.DATA)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")
    plan = make_plan(paths)
    save_mixed_plan(paths, plan)
    orchestrator.advance_to(ProjectState.PROTOTYPING, reason="test plan")
    orchestrator.advance_to(ProjectState.PROTOTYPED, reason="test plan frozen")
    return paths, plan


def reviewer(verdict: Verdict):
    def run(ast, experiment_id: str) -> RedTeamReview:
        return RedTeamReview(
            experiment_id=experiment_id,
            attacks=AttackReport(attacks=[]),
            responses=AuthorResponseReport(responses=[]),
            judge=JudgeReport(verdicts=[], overall=verdict, blocking=False),
            created_at=datetime.now(UTC),
        )

    return run


def test_mixed_spec_rejects_inconsistent_horizon() -> None:
    with pytest.raises(ValueError):
        MixedSpec(
            mixed_id="bad",
            problem_id="bad",
            target_column="demand",
            time_column="period",
            forecast_horizon=3,
            forecast_timestamps=["2026-01", "2026-02"],
            unit="units",
            planning_periods=["2026-01", "2026-02", "2026-03"],
            capacities=[1.0, 1.0, 1.0],
            production_costs=[1.0, 1.0, 1.0],
            holding_costs=[1.0, 1.0, 1.0],
        )
    with pytest.raises(ValueError):
        MixedSpec(
            mixed_id="bad",
            problem_id="bad",
            target_column="demand",
            time_column="period",
            forecast_horizon=2,
            forecast_timestamps=["2026-01", "2026-02"],
            forecast_model_id="naive.seasonal",
            unit="units",
            planning_periods=["2026-01", "2026-02"],
            capacities=[1.0, 1.0],
            production_costs=[1.0, 1.0],
            holding_costs=[1.0, 1.0],
        )


def test_plan_requires_multi_task_routing(tmp_path: Path) -> None:
    paths = project(tmp_path, "single-route")
    problem = ProblemSpec(
        problem_id="single",
        title="Forecast only",
        overall_goal="Forecast",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="Forecast demand",
                goal="Forecast",
                inputs=["history"],
                outputs=["forecast"],
                depends_on=[],
            )
        ],
    )
    extraction = ProblemSignalExtraction(
        subproblems=[
            SubproblemSignals(
                subproblem_id="q1",
                signals=ProblemSignals(
                    has_supervised_target=True,
                    requires_future_prediction=True,
                    temporal_order_matters=True,
                ),
            )
        ]
    )
    decision = route_problem_spec(
        problem,
        extraction,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        provider="test",
        model="test",
    )
    with pytest.raises(MixedPlanError):
        build_mixed_plan(
            paths,
            problem,
            decision,
            mixed_spec(),
            problem_hash="a" * 64,
            problem_spec_hash="b" * 64,
        )


def test_plan_topological_order_and_cycle_rejection(tmp_path: Path) -> None:
    paths = project(tmp_path, "topology")
    plan = make_plan(paths)
    assert plan.topological_order() == ["forecast_node", "optimization_node"]

    spec = mixed_spec()
    nodes = [
        TaskNode(
            node_id="a",
            subproblem_id="q1",
            task_family=TaskFamily.FORECASTING,
        ),
        TaskNode(
            node_id="b",
            subproblem_id="q2",
            task_family=TaskFamily.OPTIMIZATION,
        ),
    ]
    edge = ArtifactEdge(
        producer_node="a",
        producer_output="demand_forecast",
        consumer_node="b",
        consumer_input="demand_forecast",
        required_artifact_type="demand_forecast",
        required_schema_version="1.0",
    )
    reverse = ArtifactEdge(
        producer_node="b",
        producer_output="plan",
        consumer_node="a",
        consumer_input="plan",
        required_artifact_type="plan",
        required_schema_version="1.0",
    )
    cyclic = MixedExecutionPlan(
        plan_id="plan-cycle",
        project_id=paths.root.name,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        routing_id="route-cycle",
        nodes=nodes,
        edges=[edge, reverse],
        entry_nodes=[],
        terminal_nodes=[],
        spec=spec,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="cycle"):
        cyclic.topological_order()


def test_forecast_artifact_contract_mismatches(tmp_path: Path) -> None:
    paths = project(tmp_path, "artifact")
    plan = make_plan(paths)
    history = np.array([100.0 + index for index in range(24)])
    artifact = build_forecast_artifact(
        plan.spec,
        history,
        forecast_experiment_id="exp-artifact-forecast",
        source_dataset_hash="c" * 64,
    )
    payload_path, envelope_path, envelope = save_forecast_artifact(
        paths, plan, "artifact-run", artifact, provenance={}
    )
    contract = contract_for(plan.spec)
    assert verify_forecast_artifact(paths, envelope, contract) == artifact

    for mutated in (
        contract.model_copy(update={"horizon": 4}),
        contract.model_copy(update={"timestamps": ["2026-02", "2026-03", "2026-04"]}),
        contract.model_copy(update={"unit": "kg"}),
        contract.model_copy(update={"target_name": "sales"}),
    ):
        with pytest.raises(ArtifactContractError):
            verify_forecast_artifact(paths, envelope, mutated)

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["point_forecasts"][0] += 1.0
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="payload hash"):
        verify_forecast_artifact(paths, envelope, contract)

    payload_path.unlink()
    with pytest.raises(ArtifactContractError, match="payload missing"):
        verify_forecast_artifact(paths, envelope, contract)
    assert load_envelope(envelope_path).artifact_id == envelope.artifact_id


def test_missing_invalid_artifact_blocks_downstream(tmp_path: Path) -> None:
    paths = project(tmp_path, "blocked")
    spec = mixed_spec()

    def planned_nodes(task_family: TaskFamily) -> list[TaskNode]:
        return [
            TaskNode(
                node_id="forecast_node",
                subproblem_id="q1",
                task_family=task_family,
            ),
            TaskNode(
                node_id="optimization_node",
                subproblem_id="q2",
                task_family=TaskFamily.OPTIMIZATION,
            ),
        ]

    edge = ArtifactEdge(
        producer_node="forecast_node",
        producer_output="demand_forecast",
        consumer_node="optimization_node",
        consumer_input="demand_forecast",
        required_artifact_type="demand_forecast",
        required_schema_version="1.0",
    )
    plan = MixedExecutionPlan(
        plan_id="plan-blocked",
        project_id=paths.root.name,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        routing_id="route-blocked",
        nodes=planned_nodes(TaskFamily.GRAPH),
        edges=[edge],
        entry_nodes=["forecast_node"],
        terminal_nodes=["optimization_node"],
        spec=spec,
        created_at=datetime.now(UTC),
    ).with_hash()

    record = run_mixed_execution(paths, plan)
    assert record.status is DAGStatus.BLOCKED
    downstream = next(
        node for node in record.nodes if node.node_id == "optimization_node"
    )
    assert downstream.execution_status is NodeStatus.BLOCKED_UPSTREAM
    assert not (paths.mixed_dir / "runs" / record.run_id / "artifacts").exists()


def test_multi_task_execution_mode_uses_optimization_lifecycle() -> None:
    order = task_order(TaskFamily.FORECASTING, ExecutionMode.MULTI_TASK)
    assert ProjectState.PROTOTYPED in order
    assert ProjectState.FULL_EXPERIMENT not in order
    assert order[order.index(ProjectState.PROTOTYPED) + 1] is ProjectState.EXPERIMENT_COMPLETED


@pytest.fixture(scope="module")
def released_run(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("mixed-released")
    paths, plan = prepared_project(base)
    with pytest.raises(MixedPipelineBlocked):
        run_mixed_project(paths, reviewer(Verdict.REJECT))
    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.RED_TEAM_REVIEW
    before = sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))
    summary = run_mixed_project(paths, reviewer(Verdict.PASS))
    after = sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))
    return paths, plan, summary, before, after


def test_mixed_pipeline_end_to_end(released_run) -> None:
    paths, plan, summary, before, after = released_run
    assert summary.state is ProjectState.RELEASED
    assert summary.red_team_overall == Verdict.PASS.value
    assert before == after
    assert summary.release_path is not None and summary.release_path.exists()
    assert summary.paper_path.exists()
    assert summary.objective is not None and summary.feasible is True

    record = json.loads(run_record_path(paths, summary.run_id).read_text(encoding="utf-8"))
    assert record["status"] == DAGStatus.SUCCEEDED.value
    assert record["provenance_complete"] is True
    assert record["forecast_artifact_hash"] == summary.forecast_artifact_hash
    assert all(node["execution_status"] == NodeStatus.SUCCEEDED.value for node in record["nodes"])
    forecast_node = next(
        node for node in record["nodes"] if node["task_family"] == TaskFamily.FORECASTING
    )
    assert forecast_node["experiment_id"] == summary.forecast_experiment_id

    optimization = load_optimization_result(paths, summary.optimization_experiment_id)
    plan_values = next(
        solution for solution in optimization.solutions if solution.method == "ortools.cbc"
    ).variable_values
    forecast_values = [
        float(value)
        for value in json.loads(
            (
                paths.root
                / json.loads(
                    (
                        paths.mixed_dir
                        / "runs"
                        / summary.run_id
                        / "artifacts"
                        / "forecast_envelope.json"
                    ).read_text(encoding="utf-8")
                )["payload_path"]
            ).read_text(encoding="utf-8")
        )["point_forecasts"]
    ]
    capacity_1 = 140.0
    carry = max(0.0, forecast_values[1] - capacity_1)
    assert plan_values["inventory_0"] == pytest.approx(carry, abs=1e-6)
    assert plan_values["production_0"] == pytest.approx(forecast_values[0] + carry, abs=1e-6)
    assert plan_values["production_1"] == pytest.approx(capacity_1, abs=1e-6)

    envelope = load_envelope(
        paths.mixed_dir
        / "runs"
        / summary.run_id
        / "artifacts"
        / "forecast_envelope.json"
    )
    assert envelope.content_hash == summary.forecast_artifact_hash
    assert envelope.provenance["forecast_experiment_id"] == summary.forecast_experiment_id
    assert envelope.provenance["plan_hash"] == plan.plan_hash

    evidence = paths.verified_dir / "mixed" / f"{summary.run_id}.json"
    assert evidence.exists()
    keys = {
        item["key"]
        for item in json.loads(evidence.read_text(encoding="utf-8"))["records"]
    }
    assert {
        "mixed.artifact.id",
        "mixed.artifact.content_hash",
        "mixed.dependency_contract_status",
        "mixed.optimization.objective_value",
        "mixed.provenance_complete",
    } <= keys

    claims = load_claims(paths)
    assert any(claim.claim_id == f"claim-mixed-{summary.run_id}" for claim in claims)
    assert all(claim.status == "proposed" for claim in claims)


def test_validator_recomputes_demand_without_fallback(released_run) -> None:
    paths, plan, summary, _, _ = released_run
    records = validate_mixed_experiment(paths, plan, summary.run_id)
    assert not any(report.blocking for report in records)

    spec_path = (
        paths.experiment_dir(summary.optimization_experiment_id) / "optimization_spec.json"
    )
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    for constraint in payload["model"]["constraints"]:
        if constraint["name"] == "inventory_balance_0":
            constraint["lower"] += 1.0
            constraint["upper"] += 1.0
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    reports = validate_mixed_experiment(paths, plan, summary.run_id)
    assert any(report.blocking for report in reports)
    codes = {issue.code for report in reports for issue in report.issues}
    assert "balance_demand_mismatch" in codes


def test_resume_rejects_changed_evidence(tmp_path: Path) -> None:
    paths, _plan = prepared_project(tmp_path, "mixed-resume")
    with pytest.raises(MixedPipelineBlocked):
        run_mixed_project(paths, reviewer(Verdict.REJECT))
    run_files = sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))
    assert len(run_files) == 1
    run_files[0].write_text(
        run_files[0].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(MixedPipelineBlocked, match="evidence changed"):
        run_mixed_project(paths, reviewer(Verdict.PASS))
    assert len(sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))) == 1
    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
