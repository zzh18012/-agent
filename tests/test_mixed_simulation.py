import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmos.core.project import ProjectPaths, load_project_config
from mmos.mixed.artifacts import (
    ArtifactContractError,
    build_forecast_error_artifact,
    contract_for_forecast_error,
    contract_for_production_plan,
    save_forecast_error_artifact,
    verify_forecast_error_artifact,
    verify_production_plan_artifact,
)
from mmos.mixed.plan import build_mixed_plan, save_mixed_plan
from mmos.mixed.simulation import (
    MixedSimulationError,
    run_mixed_simulation,
    save_mixed_simulation_spec,
)
from mmos.mixed.staleness import apply_stale_propagation
from mmos.mixed.validation import run_record_path
from mmos.mixed_pipeline import MixedPipelineBlocked, run_mixed_project
from mmos.orchestration import Orchestrator
from mmos.routing.policy import route_problem_spec
from mmos.schemas.experiment import FoldWindow
from mmos.schemas.mixed import (
    DAGRunRecord,
    DAGStatus,
    ForecastArtifact,
    ForecastErrorArtifact,
    ForecastResidual,
    MixedSpec,
    NodeStatus,
    ProductionPlanArtifact,
    TaskNode,
)
from mmos.schemas.mixed_simulation import MixedSimulationSpec
from mmos.schemas.problem import ProblemSpec, SubProblem
from mmos.schemas.project import SourceRole
from mmos.schemas.routing import (
    ProblemSignalExtraction,
    ProblemSignals,
    SubproblemSignals,
)
from mmos.schemas.states import ProjectState, TaskFamily
from tests.test_mixed import DATA_CSV, PROBLEM_MD, project, reviewer

REPLICATIONS = 40


def simulation_spec() -> MixedSpec:
    return MixedSpec(
        mixed_id="mixed-sim-test",
        problem_id="mixed-sim-test",
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
        simulation_replications=REPLICATIONS,
        simulation_master_seed=20261116,
        confidence_level=0.95,
    ).with_hash()


def three_node_problem() -> ProblemSpec:
    return ProblemSpec(
        problem_id="mixed-sim-test",
        title="Forecast, plan, stress test",
        overall_goal="Forecast demand, plan production, and stress test the plan",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="Forecast demand",
                goal="Forecast",
                inputs=["history"],
                outputs=["demand forecast", "residuals"],
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
            SubProblem(
                subproblem_id="q3",
                statement="Simulate demand uncertainty around the frozen plan",
                goal="Stress test",
                inputs=["production plan", "residuals"],
                outputs=["risk evidence"],
                depends_on=["q1", "q2"],
            ),
        ],
    )


def three_node_decision():
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
            SubproblemSignals(
                subproblem_id="q3",
                signals=ProblemSignals(
                    stochastic_process_evolution=True,
                    requires_replications_or_distributional_output=True,
                ),
            ),
        ]
    )
    return route_problem_spec(
        three_node_problem(),
        extraction,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        provider="test",
        model="test",
    )


def three_node_plan(paths: ProjectPaths):
    return build_mixed_plan(
        paths,
        three_node_problem(),
        three_node_decision(),
        simulation_spec(),
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
    )


def test_three_node_topology(tmp_path: Path) -> None:
    paths = project(tmp_path, "three-node")
    plan = three_node_plan(paths)
    assert plan.topological_order() == [
        "forecast_node",
        "optimization_node",
        "simulation_node",
    ]
    edge_types = {
        (edge.producer_output, edge.consumer_input): edge.required_artifact_type
        for edge in plan.edges
    }
    assert edge_types == {
        ("demand_forecast", "demand_forecast"): "demand_forecast",
        ("forecast_error", "forecast_error"): "forecast_error",
        ("production_plan", "production_plan"): "production_plan",
    }
    assert plan.terminal_nodes == ["simulation_node"]
    assert plan.has_simulation_node()


def test_residual_direction_is_frozen() -> None:
    valid = ForecastResidual(
        fold_id=0, row_index=12, timestamp="2024-01", actual=110.0, forecast=100.0,
        residual=10.0,
    )
    assert valid.residual == valid.actual - valid.forecast
    with pytest.raises(ValidationError):
        ForecastResidual(
            fold_id=0,
            row_index=12,
            timestamp="2024-01",
            actual=110.0,
            forecast=100.0,
            residual=-10.0,
        )


def test_forecast_error_builder_rejects_training_rows(tmp_path: Path) -> None:
    import polars as pl

    spec = simulation_spec()
    predictions = pl.DataFrame(
        {
            "row_index": [0, 12, 13],
            "model": ["naive.drift"] * 3,
            "prediction": [100.0, 101.0, 102.0],
            "actual": [100.0, 110.0, 112.0],
        }
    )
    windows = [
        FoldWindow(train_start=0, train_end=12, test_start=12, test_end=15)
    ]
    with pytest.raises(ArtifactContractError, match="test window"):
        build_forecast_error_artifact(
            spec,
            predictions,
            [f"p{index}" for index in range(24)],
            windows,
            forecast_experiment_id="exp-f",
            forecast_artifact_id="artifact-f",
            forecast_artifact_hash="a" * 64,
        )


def _manual_artifacts(
    production: list[float], residuals: list[float], timestamps: list[str]
) -> tuple[ForecastArtifact, ForecastErrorArtifact, ProductionPlanArtifact]:
    horizon = len(timestamps)
    forecast = ForecastArtifact(
        forecast_horizon=horizon,
        timestamps=timestamps,
        target_name="demand",
        unit="units",
        point_forecasts=[100.0] * horizon,
        model_id="naive.drift",
        source_dataset_hash="d" * 64,
        forecast_experiment_id="exp-manual-forecast",
        validation_strategy="rolling_origin",
        validation_status="verified",
    )
    forecast_hash = forecast.computed_content_hash()
    residual_records = [
        ForecastResidual(
            fold_id=index,
            row_index=12 + index,
            timestamp=f"hist-{index}",
            actual=100.0 + value,
            forecast=100.0,
            residual=value,
        )
        for index, value in enumerate(residuals)
    ]
    error = ForecastErrorArtifact(
        forecast_horizon=horizon,
        unit="units",
        source_forecast_experiment_id=forecast.forecast_experiment_id,
        source_forecast_artifact_id="artifact-manual-forecast",
        source_forecast_artifact_hash=forecast_hash,
        residuals=residual_records,
        mean=sum(residuals) / len(residuals),
        std=0.0 if len(residuals) == 1 else statistics.stdev(residuals),
        sample_count=len(residuals),
    )
    plan = ProductionPlanArtifact(
        planning_horizon=horizon,
        timestamps=timestamps,
        unit="units",
        production=production,
        planned_inventory=[0.0] * horizon,
        capacity=[500.0] * horizon,
        objective_value=1.0,
        forecast_artifact_id="artifact-manual-forecast",
        forecast_artifact_hash=forecast_hash,
        optimization_experiment_id="exp-manual-optimization",
    )
    return forecast, error, plan


def _manual_spec(
    forecast: ForecastArtifact,
    error: ForecastErrorArtifact,
    plan: ProductionPlanArtifact,
    timestamps: list[str],
) -> MixedSimulationSpec:
    return MixedSimulationSpec(
        simulation_experiment_id="exp-manual-simulation",
        simulation_id="sim-manual",
        problem_id="manual",
        project_id="manual",
        unit="units",
        planning_horizon=len(timestamps),
        timestamps=timestamps,
        production_plan_artifact_id="artifact-manual-plan",
        production_plan_artifact_hash=plan.computed_content_hash(),
        forecast_artifact_id="artifact-manual-forecast",
        forecast_artifact_hash=forecast.computed_content_hash(),
        forecast_error_artifact_id="artifact-manual-error",
        forecast_error_artifact_hash=error.computed_content_hash(),
        replications=5,
        master_seed=7,
    ).with_hash()


def test_simulation_arithmetic_and_service_metric(tmp_path: Path) -> None:
    paths = project(tmp_path, "manual-sim")
    timestamps = ["2026-01", "2026-02"]
    forecast, error, plan = _manual_artifacts(
        production=[80.0, 90.0], residuals=[0.0, 0.0], timestamps=timestamps
    )
    spec = _manual_spec(forecast, error, plan, timestamps)
    save_mixed_simulation_spec(paths, spec)
    result = run_mixed_simulation(paths, spec, plan, forecast, error)
    assert result.metrics["total_shortage"].mean == pytest.approx(30.0)
    assert result.metrics["shortage_period_count"].mean == pytest.approx(2.0)
    assert result.metrics["fill_rate"].mean == pytest.approx(170.0 / 200.0)
    assert result.metrics["ending_inventory"].mean == pytest.approx(0.0)
    assert result.metrics["max_shortage"].mean == pytest.approx(20.0)
    assert result.any_shortage_probability == 1.0
    assert result.distinct_residual_paths == 1
    assert result.demand_floor_activations == 0
    for replication in result.replications:
        assert replication.periods[0].shortage == pytest.approx(20.0)
        assert replication.periods[1].shortage == pytest.approx(10.0)
        assert replication.periods[0].served + replication.periods[0].shortage == 100.0


def test_simulation_is_seed_reproducible(tmp_path: Path) -> None:
    paths = project(tmp_path, "sim-repro")
    timestamps = ["2026-01"]
    forecast, error, plan = _manual_artifacts(
        production=[50.0], residuals=[-10.0, 5.0, 20.0], timestamps=timestamps
    )
    spec = _manual_spec(forecast, error, plan, timestamps)
    save_mixed_simulation_spec(paths, spec)
    first = run_mixed_simulation(paths, spec, plan, forecast, error)
    second = run_mixed_simulation(paths, spec, plan, forecast, error)
    assert first.replication_seed_hash == second.replication_seed_hash
    assert [item.model_dump() for item in first.replications] == [
        item.model_dump() for item in second.replications
    ]


def test_demand_floor_policy_is_recorded(tmp_path: Path) -> None:
    paths = project(tmp_path, "floor")
    timestamps = ["2026-01"]
    forecast, error, plan = _manual_artifacts(
        production=[10.0], residuals=[-120.0, -120.0], timestamps=timestamps
    )
    spec = _manual_spec(forecast, error, plan, timestamps)
    save_mixed_simulation_spec(paths, spec)
    result = run_mixed_simulation(paths, spec, plan, forecast, error)
    assert result.demand_floor_activations == spec.replications
    for replication in result.replications:
        assert replication.periods[0].simulated_demand == 0.0
        assert replication.periods[0].shortage == 0.0


def test_simulation_never_calls_optimization(tmp_path: Path, monkeypatch) -> None:
    import mmos.experiments.optimization_runner as runner

    def forbidden(*args, **kwargs):
        raise AssertionError("simulation must not re-optimize the plan")

    monkeypatch.setattr(runner, "run_optimization", forbidden)
    monkeypatch.setattr(runner, "solve_ortools", forbidden)
    paths = project(tmp_path, "no-reopt")
    timestamps = ["2026-01"]
    forecast, error, plan = _manual_artifacts(
        production=[90.0], residuals=[0.0, 0.0], timestamps=timestamps
    )
    spec = _manual_spec(forecast, error, plan, timestamps)
    save_mixed_simulation_spec(paths, spec)
    result = run_mixed_simulation(paths, spec, plan, forecast, error)
    assert result.replications[0].periods[0].production == 90.0


def test_cross_task_unit_and_horizon_blocking(tmp_path: Path) -> None:
    paths = project(tmp_path, "contract-sim")
    timestamps = ["2026-01", "2026-02"]
    forecast, error, plan = _manual_artifacts(
        production=[100.0, 100.0], residuals=[0.0, 0.0], timestamps=timestamps
    )
    spec = _manual_spec(forecast, error, plan, timestamps)
    save_mixed_simulation_spec(paths, spec)
    mismatched_plan = plan.model_copy(update={"unit": "kg"})
    unit_spec = spec.model_copy(
        update={
            "production_plan_artifact_hash": mismatched_plan.computed_content_hash()
        }
    ).with_hash()
    with pytest.raises(MixedSimulationError, match="unit"):
        run_mixed_simulation(paths, unit_spec, mismatched_plan, forecast, error)
    shifted_forecast = forecast.model_copy(
        update={"timestamps": ["2026-02", "2026-03"]}
    )
    timestamp_spec = spec.model_copy(
        update={"forecast_artifact_hash": shifted_forecast.computed_content_hash()}
    ).with_hash()
    with pytest.raises(MixedSimulationError, match="timestamps"):
        run_mixed_simulation(paths, timestamp_spec, plan, shifted_forecast, error)

    run_artifacts = paths.mixed_dir / "runs" / "contract-run" / "artifacts"
    _, error_envelope_path, _ = save_forecast_error_artifact(
        paths, three_node_plan(paths), "contract-run", error, provenance={}
    )
    from mmos.mixed.artifacts import load_envelope

    error_envelope = load_envelope(error_envelope_path)
    bad_error_contract = contract_for_forecast_error(
        simulation_spec(), forecast.forecast_experiment_id,
        forecast.computed_content_hash(),
    ).model_copy(update={"unit": "kg"})
    with pytest.raises(ArtifactContractError, match="unit"):
        verify_forecast_error_artifact(paths, error_envelope, bad_error_contract)
    assert run_artifacts.exists()

    plan_contract = contract_for_production_plan(
        simulation_spec(), "artifact-manual-forecast", forecast.computed_content_hash()
    ).model_copy(
        update={"horizon": 2, "timestamps": ["2026-02", "2026-03"]}
    )
    from mmos.mixed.artifacts import save_production_plan_artifact

    _, plan_envelope_path, _ = save_production_plan_artifact(
        paths, three_node_plan(paths), "contract-run", plan, provenance={}
    )
    plan_envelope = load_envelope(plan_envelope_path)
    with pytest.raises(ArtifactContractError, match="timestamp"):
        verify_production_plan_artifact(paths, plan_envelope, plan_contract)


def _write_prior_run(
    paths: ProjectPaths,
    *,
    run_id: str = "prior-run",
    plan_id: str = "plan-prior",
    forecast_hash: str = "f" * 64,
    error_hash: str = "e" * 64,
    plan_hash: str = "p" * 64,
) -> Path:
    record = DAGRunRecord(
        run_id=run_id,
        plan_id=plan_id,
        plan_hash="h" * 64,
        project_id=paths.root.name,
        status=DAGStatus.SUCCEEDED,
        nodes=[
            TaskNode(
                node_id="forecast_node",
                subproblem_id="q1",
                task_family=TaskFamily.FORECASTING,
                execution_status=NodeStatus.SUCCEEDED,
            ),
            TaskNode(
                node_id="optimization_node",
                subproblem_id="q2",
                task_family=TaskFamily.OPTIMIZATION,
                execution_status=NodeStatus.SUCCEEDED,
            ),
            TaskNode(
                node_id="simulation_node",
                subproblem_id="q3",
                task_family=TaskFamily.SIMULATION,
                execution_status=NodeStatus.SUCCEEDED,
            ),
        ],
        forecast_artifact_id="artifact-forecast",
        forecast_artifact_hash=forecast_hash,
        forecast_error_artifact_id="artifact-error",
        forecast_error_artifact_hash=error_hash,
        optimization_experiment_id="exp-o",
        production_plan_artifact_id="artifact-plan",
        production_plan_artifact_hash=plan_hash,
        simulation_experiment_id="exp-s",
        provenance_complete=True,
        started_at=datetime.now(UTC),
    )
    target = run_record_path(paths, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def _statuses(path: Path) -> tuple[DAGStatus, dict[str, NodeStatus]]:
    record = DAGRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
    return record.status, {
        node.node_id: node.execution_status for node in record.nodes
    }


def test_stale_propagation_by_artifact_lineage(tmp_path: Path) -> None:
    paths = project(tmp_path, "stale")

    target = _write_prior_run(paths)
    updated = apply_stale_propagation(
        paths,
        "plan-prior",
        forecast_hash="f" * 64,
        forecast_error_hash="n" * 64,
        production_plan_hash="p" * 64,
    )
    assert updated == ["prior-run"]
    status, nodes = _statuses(target)
    assert status is DAGStatus.STALE
    assert nodes["simulation_node"] is NodeStatus.STALE
    assert nodes["optimization_node"] is NodeStatus.SUCCEEDED
    assert nodes["forecast_node"] is NodeStatus.SUCCEEDED

    target = _write_prior_run(paths, run_id="prior-forecast")
    apply_stale_propagation(
        paths,
        "plan-prior",
        forecast_hash="x" * 64,
        forecast_error_hash="e" * 64,
        production_plan_hash="p" * 64,
    )
    status, nodes = _statuses(target)
    assert status is DAGStatus.STALE
    assert nodes["optimization_node"] is NodeStatus.STALE
    assert nodes["simulation_node"] is NodeStatus.STALE
    assert nodes["forecast_node"] is NodeStatus.SUCCEEDED

    target = _write_prior_run(paths, run_id="prior-plan")
    apply_stale_propagation(
        paths,
        "plan-prior",
        forecast_hash="f" * 64,
        forecast_error_hash="e" * 64,
        production_plan_hash="z" * 64,
    )
    status, nodes = _statuses(target)
    assert status is DAGStatus.STALE
    assert nodes["simulation_node"] is NodeStatus.STALE
    assert nodes["optimization_node"] is NodeStatus.SUCCEEDED


def prepared_three_node(tmp_path: Path, name: str = "three-node") -> ProjectPaths:
    paths = project(tmp_path, name)
    problem = tmp_path / f"{name}-problem.md"
    problem.write_text(PROBLEM_MD, encoding="utf-8")
    data = tmp_path / f"{name}-data.csv"
    data.write_text(DATA_CSV, encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.ingest([data], SourceRole.DATA)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")
    plan = three_node_plan(paths)
    save_mixed_plan(paths, plan)
    orchestrator.advance_to(ProjectState.PROTOTYPING, reason="test plan")
    orchestrator.advance_to(ProjectState.PROTOTYPED, reason="test plan frozen")
    return paths


@pytest.fixture(scope="module")
def three_node_release(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("mixed-sim-release")
    paths = prepared_three_node(base)
    from mmos.schemas.review import Verdict

    with pytest.raises(MixedPipelineBlocked):
        run_mixed_project(paths, reviewer(Verdict.REJECT))
    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    before = sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))
    summary = run_mixed_project(paths, reviewer(Verdict.PASS))
    after = sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))
    return paths, summary, before, after


def test_three_node_pipeline_and_lineage(three_node_release) -> None:
    paths, summary, before, after = three_node_release
    assert summary.state is ProjectState.RELEASED
    assert before == after
    assert summary.simulation_experiment_id is not None
    assert summary.simulation_replications == REPLICATIONS
    assert summary.forecast_error_artifact_id is not None
    assert summary.production_plan_artifact_id is not None
    assert summary.release_path is not None and summary.release_path.exists()
    record = DAGRunRecord.model_validate_json(
        run_record_path(paths, summary.run_id).read_text(encoding="utf-8")
    )
    assert record.status is DAGStatus.SUCCEEDED
    assert record.contract_chain_complete is True
    assert record.simulation_experiment_id == summary.simulation_experiment_id
    assert record.production_plan_artifact_hash == summary.production_plan_artifact_hash
    assert record.forecast_error_artifact_hash == summary.forecast_error_artifact_hash
    assert record.shortage_summary
    assert record.service_summary

    evidence_path = paths.verified_dir / "mixed" / f"{summary.run_id}.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    keys = {item["key"] for item in evidence["records"]}
    assert {
        "mixed.production_plan.artifact_hash",
        "mixed.forecast_error.artifact_hash",
        "mixed.forecast_error.residual_source",
        "mixed.simulation.experiment_id",
        "mixed.simulation.seed_hash",
        "mixed.simulation.shortage.mean_total_shortage",
        "mixed.simulation.service.mean_fill_rate",
        "mixed.contract_chain_complete",
    } <= keys
    provenance = evidence["records"][0]["provenance"]
    for key in (
        "plan_hash",
        "forecast_artifact_hash",
        "forecast_error_artifact_hash",
        "production_plan_artifact_hash",
        "simulation_experiment_id",
        "simulation_seed_hash",
        "dataset_hash",
    ):
        assert provenance.get(key)

    paper = (paths.root / "paper" / "report.md").read_text(encoding="utf-8")
    assert "Fixed-anchor historical residual pool" in paper
    assert "Conditional simulation protocol" in paper
    assert "fill rate" in paper
    assert (paths.root / "paper" / "simulation_outcomes.png").exists()
    assert (paths.root / "paper" / "forecast_planning_dag.png").exists()
    bundle = paths.root / "paper" / "evidence_bundle"
    assert (bundle / "SHA256SUMS").exists()
    assert (bundle / "inputs" / "mixed_spec.json").exists()
    assert (bundle / "experiments" / "simulation" / "replications.csv").exists()


def test_three_node_resume_rejects_changed_error_artifact(tmp_path: Path) -> None:
    paths = prepared_three_node(tmp_path, "three-node-tamper")
    from mmos.schemas.review import Verdict

    with pytest.raises(MixedPipelineBlocked):
        run_mixed_project(paths, reviewer(Verdict.REJECT))
    run_files = sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))
    assert len(run_files) == 1
    run_id = run_files[0].parent.name
    error_path = (
        paths.mixed_dir / "runs" / run_id / "artifacts" / "forecast_error_artifact.json"
    )
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    payload["residuals"][0]["actual"] += 1.0
    error_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MixedPipelineBlocked, match="evidence changed"):
        run_mixed_project(paths, reviewer(Verdict.PASS))
    assert len(sorted((paths.mixed_dir / "runs").glob("*/dag_run.json"))) == 1
