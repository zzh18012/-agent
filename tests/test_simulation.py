import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scipy import stats as scipy_stats

from mmos.core.project import (
    ProjectPaths,
    create_project,
    load_analytic_oracle,
    load_project_config,
    load_simulation_result,
    save_simulation_record,
    save_simulation_spec,
    simulation_result_file,
)
from mmos.core.state_machine import allowed_targets, task_order
from mmos.evidence.store import load_claims
from mmos.experiments.simulation_oracle import (
    UnstableQueueError,
    compute_analytic_oracle,
)
from mmos.experiments.simulation_runner import (
    SimulationError,
    _confidence_interval,
    _MM1Run,
    _seed_schedule,
    _TimeWeightedAccumulator,
    build_simulation_record,
    run_simulation,
)
from mmos.orchestration import Orchestrator
from mmos.paper.simulation_builder import build_simulation_paper
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.project import SourceRole
from mmos.schemas.review import (
    AttackReport,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)
from mmos.schemas.simulation import SimulationSpec
from mmos.schemas.states import SUPPORTED_TASKS, ProjectState, TaskFamily
from mmos.simulation_pipeline import (
    SimulationPipelineBlocked,
    SimulationPipelineOptions,
    run_simulation_pipeline,
)
from mmos.validation.simulation import validate_simulation_experiment


def make_spec(**overrides) -> SimulationSpec:
    data = {
        "simulation_id": "mm1-test",
        "experiment_id": "exp-mm1-test",
        "arrival_rate": 2.0,
        "service_rate": 3.0,
        "warmup_time": 20.0,
        "measurement_time": 100.0,
        "replications": 6,
        "master_seed": 12345,
        "analytic_relative_tolerance": 0.5,
        "little_law_relative_tolerance": 0.5,
    }
    data.update(overrides)
    return SimulationSpec(**data).with_hash()


def write_spec(path: Path, spec: SimulationSpec) -> Path:
    target = path / "spec.json"
    target.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return target


def simulation_project(tmp_path: Path, name: str = "sim") -> ProjectPaths:
    paths, _ = create_project(tmp_path / "projects", name, task=TaskFamily.SIMULATION)
    problem = tmp_path / f"{name}-problem.md"
    problem.write_text("# M/M/1 queue with Poisson arrivals\n", encoding="utf-8")
    orchestrator = Orchestrator(paths)
    orchestrator.ingest([problem], SourceRole.PROBLEM)
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test")
    return paths


def review(verdict: Verdict):
    def run(ast, experiment_id: str) -> RedTeamReview:
        return RedTeamReview(
            experiment_id=experiment_id,
            attacks=AttackReport(attacks=[]),
            responses=AuthorResponseReport(responses=[]),
            judge=JudgeReport(verdicts=[], overall=verdict, blocking=False),
            created_at=datetime.now(UTC),
        )

    return run


def prepared_experiment(tmp_path: Path):
    paths, _ = create_project(tmp_path / "projects", "sim-unit", task=TaskFamily.SIMULATION)
    spec = make_spec()
    save_simulation_spec(paths, spec)
    result = run_simulation(paths, spec)
    record = build_simulation_record(
        paths, spec, result, status=ExperimentStatus.SUCCEEDED
    ).model_copy(update={"mlflow_run_id": "mlflow-test-run"})
    save_simulation_record(paths, record)
    return paths, spec, result


def test_simulation_is_a_supported_task_family() -> None:
    assert TaskFamily.SIMULATION in SUPPORTED_TASKS
    assert task_order(TaskFamily.SIMULATION)[:3] == (
        ProjectState.INIT,
        ProjectState.PROJECT_INGESTED,
        ProjectState.PROBLEM_UNDERSTOOD,
    )
    assert ProjectState.PROTOTYPING in allowed_targets(
        ProjectState.PROBLEM_UNDERSTOOD, task=TaskFamily.SIMULATION
    )


def test_analytic_oracle_matches_mm1_formulas() -> None:
    spec = make_spec(arrival_rate=2.0, service_rate=3.0)
    oracle = compute_analytic_oracle(spec)

    assert oracle.rho == pytest.approx(2 / 3)
    assert pytest.approx(1.0) == oracle.W
    assert oracle.Wq == pytest.approx(2 / 3)
    assert pytest.approx(2.0) == oracle.L
    assert oracle.Lq == pytest.approx(4 / 3)
    assert oracle.utilization == pytest.approx(2 / 3)
    assert oracle.throughput == pytest.approx(2.0)
    assert pytest.approx(oracle.arrival_rate * oracle.W) == oracle.L
    assert oracle.Lq == pytest.approx(oracle.arrival_rate * oracle.Wq)
    assert pytest.approx(oracle.rho / (1 - oracle.rho)) == oracle.L
    assert oracle.Lq == pytest.approx(oracle.rho**2 / (1 - oracle.rho))


def test_unstable_queue_has_no_steady_state_oracle() -> None:
    with pytest.raises(UnstableQueueError):
        compute_analytic_oracle(make_spec(arrival_rate=3.0, service_rate=3.0))
    with pytest.raises(UnstableQueueError):
        compute_analytic_oracle(make_spec(arrival_rate=4.0, service_rate=3.0))


def test_spec_hash_is_deterministic_and_tracks_changes() -> None:
    first = make_spec()
    second = make_spec()
    changed = make_spec(arrival_rate=1.5)

    assert first.spec_hash == second.spec_hash
    assert first.spec_hash != changed.spec_hash
    assert first.stable is True
    assert make_spec(arrival_rate=3.0).stable is False


def test_seed_schedule_is_unique_and_reproducible() -> None:
    first = _seed_schedule(2026, 8)
    second = _seed_schedule(2026, 8)
    changed = _seed_schedule(2027, 8)

    assert first == second
    assert first != changed
    for key in ("replication_seed", "arrival_seed", "service_seed"):
        values = [entry[key] for entry in first]
        assert len(set(values)) == len(values)


def test_time_weighted_statistics_use_durations_not_observations() -> None:
    accumulator = _TimeWeightedAccumulator(0.0, 10.0)
    accumulator.transition(0.0, system=0, queue=0, busy=0)
    accumulator.transition(9.0, system=10, queue=0, busy=0)
    accumulator.finalize()

    assert accumulator.system_area == pytest.approx(10.0)
    assert accumulator.system_area / 10.0 == pytest.approx(1.0)
    observation_mean = (0.0 + 10.0) / 2
    assert observation_mean != pytest.approx(1.0)


def test_pre_warmup_state_is_not_integrated() -> None:
    accumulator = _TimeWeightedAccumulator(5.0, 10.0)
    accumulator.transition(0.0, system=1)
    accumulator.transition(6.0, system=-1)
    accumulator.finalize()

    assert accumulator.system_area == pytest.approx(1.0)


def test_replication_results_are_reproducible_with_same_master_seed(
    tmp_path: Path,
) -> None:
    spec = make_spec(replications=4)
    first_paths, _ = create_project(
        tmp_path / "a", "sim-a", task=TaskFamily.SIMULATION
    )
    second_paths, _ = create_project(
        tmp_path / "b", "sim-b", task=TaskFamily.SIMULATION
    )
    save_simulation_spec(first_paths, spec)
    save_simulation_spec(second_paths, spec)

    first = run_simulation(first_paths, spec)
    second = run_simulation(second_paths, spec)

    assert first.replication_seed_hash == second.replication_seed_hash
    fields = ("W", "Wq", "L", "Lq", "utilization", "throughput", "eligible_customers")
    for left, right in zip(first.replications, second.replications, strict=True):
        for field in fields:
            assert getattr(left, field) == pytest.approx(getattr(right, field))
    changed = run_simulation(
        create_project(tmp_path / "c", "sim-c", task=TaskFamily.SIMULATION)[0],
        make_spec(replications=4, master_seed=999).with_hash(),
    )
    assert any(
        pytest.approx(right.W) != left.W
        for left, right in zip(first.replications, changed.replications, strict=True)
    )


def test_metrics_use_multiple_replications_and_time_weighting(tmp_path: Path) -> None:
    _paths, _, result = prepared_experiment(tmp_path)

    assert result.n_replications == 6
    assert all(summary.n == 6 for summary in result.metrics.values())
    for replication in result.replications:
        assert replication.completed_customers == replication.eligible_customers
        assert replication.arrivals > replication.eligible_customers
        assert replication.W >= replication.Wq >= 0
        assert replication.L >= replication.Lq >= 0
        assert 0 <= replication.utilization <= 1
        assert pytest.approx(
            replication.system_area / replication.measurement_time
        ) == replication.L
        assert replication.Lq == pytest.approx(
            replication.queue_area / replication.measurement_time
        )
        assert replication.utilization == pytest.approx(
            replication.busy_area / replication.measurement_time
        )


def test_eligible_customers_drain_after_measurement_end() -> None:
    spec = make_spec(
        arrival_rate=0.9,
        service_rate=1.0,
        warmup_time=10.0,
        measurement_time=20.0,
        replications=3,
        master_seed=7,
        analytic_relative_tolerance=0.9,
        little_law_relative_tolerance=0.9,
    )
    rows = [_MM1Run(spec, seeds).run() for seeds in _seed_schedule(spec.master_seed, 3)]

    assert all(row.completed_customers == row.eligible_customers for row in rows)
    assert any(row.drain_end_time > spec.measurement_end for row in rows)


def test_student_t_confidence_interval_matches_manual_formula() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    mean = sum(values) / len(values)
    standard_error = (sum((value - mean) ** 2 for value in values) / 3) ** 0.5 / 2
    critical = scipy_stats.t.ppf(0.975, 3)
    low, high = _confidence_interval(values, 0.95)

    assert low == pytest.approx(mean - critical * standard_error)
    assert high == pytest.approx(mean + critical * standard_error)


def test_little_law_summary_is_computed_from_replication_means(tmp_path: Path) -> None:
    _, _, result = prepared_experiment(tmp_path)

    mean_rate = sum(row.effective_arrival_rate for row in result.replications) / 6
    mean_l = sum(row.L for row in result.replications) / 6
    mean_w = sum(row.W for row in result.replications) / 6
    check = result.little_law["L"]

    assert check.effective_arrival_rate == pytest.approx(mean_rate)
    assert check.simulated_mean == pytest.approx(mean_l)
    assert check.rate_times_mean == pytest.approx(mean_rate * mean_w)
    assert check.relative_discrepancy == pytest.approx(
        abs(mean_l - mean_rate * mean_w) / max(abs(mean_l), 1e-12)
    )


def test_validation_passes_for_a_complete_experiment(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    reports = validate_simulation_experiment(paths, spec.experiment_id)

    assert len(reports) == 1
    assert reports[0].validator == "simulation"
    assert reports[0].blocking is False
    summary = paths.experiment_dir(spec.experiment_id) / "validation" / "summary.json"
    assert json.loads(summary.read_text(encoding="utf-8"))["blocking"] is False


def test_missing_replication_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)
    target = simulation_result_file(paths, spec.experiment_id)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["replications"] = payload["replications"][:-1]
    target.write_text(json.dumps(payload), encoding="utf-8")

    reports = validate_simulation_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "missing_replication" for issue in reports[0].issues)


def test_reused_replication_seed_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)
    target = simulation_result_file(paths, spec.experiment_id)
    payload = json.loads(target.read_text(encoding="utf-8"))
    for replication in payload["replications"]:
        replication["replication_seed"] = 7
    target.write_text(json.dumps(payload), encoding="utf-8")

    reports = validate_simulation_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "seed_reuse" for issue in reports[0].issues)


def test_tampered_time_average_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)
    target = simulation_result_file(paths, spec.experiment_id)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["replications"][0]["L"] += 1.0
    target.write_text(json.dumps(payload), encoding="utf-8")

    reports = validate_simulation_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "time_average_mismatch" for issue in reports[0].issues)


def test_simulation_paper_is_scoped_and_auditable(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    paper = build_simulation_paper(paths, spec.experiment_id)

    text = paper.abstract + " " + " ".join(
        block for section in paper.sections for block in section.blocks
    )
    assert "M/M/1 Single-Operating-Point Simulation Discrepancy Report" in paper.title
    assert "time-weighted" in text
    assert "right-censored" in text
    assert "not ranges containing 95%" in text
    assert "does not show that the simulator is generally correct" in text
    assert "Student-t interval" in text
    assert "unbounded and unquantified" in text
    metric_table = paper.tables[0]
    assert "Simulation Mean" in metric_table.columns
    assert "CI Low" in metric_table.columns
    assert "Analytic Value" in metric_table.columns
    assert "Claim role" in metric_table.columns
    assert "Std Error" in metric_table.columns
    assert "Difference / SE" in metric_table.columns
    roles = {row[0]: row[1] for row in metric_table.rows}
    assert roles["W"] == "descriptive (cohort)"
    assert roles["L"] == "descriptive (time-weighted)"
    assert "throughput" not in roles
    input_sanity = paper.tables[3]
    assert "lambda target" in input_sanity.columns
    assert "Per-rep z range" in input_sanity.columns
    assert "counting-path arithmetic check" in input_sanity.rows[0][8]
    evidence_table = paper.tables[4]
    assert evidence_table.rows
    little_law = paper.tables[2]
    relations = {row[0] for row in little_law.rows}
    assert "L vs lambda_eff x W" in relations
    assert "Lq vs lambda_eff x Wq" in relations
    assert "Estimator" in little_law.columns
    assert "Mean per-rep residual" in little_law.columns
    assert "Jensen gap" in little_law.columns
    assert "Passed" not in little_law.columns
    assert "bookkeeping" in text
    assert "Jensen" in text
    assert "simulated time units" in text


def test_simulation_pipeline_reaches_released_on_pass(tmp_path: Path) -> None:
    paths = simulation_project(tmp_path, "sim-pass")
    spec = make_spec(experiment_id="exp-sim-pass")
    spec_file = write_spec(tmp_path, spec)

    summary = run_simulation_pipeline(
        paths,
        SimulationPipelineOptions(spec_file=spec_file),
        review(Verdict.PASS),
    )

    assert summary.state is ProjectState.RELEASED
    assert load_project_config(paths).state is ProjectState.RELEASED
    assert summary.release_path is not None and summary.release_path.exists()
    assert summary.validation == {"simulation": "passed"}
    result = load_simulation_result(paths, spec.experiment_id)
    assert result.n_replications == spec.replications
    assert pytest.approx(1.0) == load_analytic_oracle(paths, spec.experiment_id).W
    claims = load_claims(paths)
    assert len(claims) == 1
    assert claims[0].status == "proposed"
    history = paths.state_history_file.read_text(encoding="utf-8")
    for state in (
        "PROBLEM_UNDERSTOOD",
        "PROTOTYPING",
        "PROTOTYPED",
        "EXPERIMENT_COMPLETED",
        "VALIDATING",
        "VERIFIED",
        "EVIDENCE_FROZEN",
        "PAPER_DRAFTED",
        "RED_TEAM_REVIEW",
        "READY_FOR_RELEASE",
        "RELEASED",
    ):
        assert state in history


def test_simulation_revise_blocks_then_resume_reuses_results(tmp_path: Path) -> None:
    paths = simulation_project(tmp_path, "sim-resume")
    spec = make_spec(experiment_id="exp-sim-resume")
    spec_file = write_spec(tmp_path, spec)
    options = SimulationPipelineOptions(spec_file=spec_file)

    with pytest.raises(SimulationPipelineBlocked):
        run_simulation_pipeline(paths, options, review(Verdict.REVISE))

    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.RED_TEAM_REVIEW
    result_path = simulation_result_file(paths, spec.experiment_id)
    completed_at = load_simulation_result(paths, spec.experiment_id).completed_at
    mtime = result_path.stat().st_mtime_ns

    summary = run_simulation_pipeline(paths, options, review(Verdict.PASS))

    assert summary.state is ProjectState.RELEASED
    assert summary.release_path is not None and summary.release_path.exists()
    assert load_simulation_result(paths, spec.experiment_id).completed_at == completed_at
    assert result_path.stat().st_mtime_ns == mtime


def test_resume_rejects_changed_frozen_evidence(tmp_path: Path) -> None:
    paths = simulation_project(tmp_path, "sim-evidence")
    spec = make_spec(experiment_id="exp-sim-evidence")
    spec_file = write_spec(tmp_path, spec)
    options = SimulationPipelineOptions(spec_file=spec_file)

    with pytest.raises(SimulationPipelineBlocked):
        run_simulation_pipeline(paths, options, review(Verdict.REVISE))
    result_path = simulation_result_file(paths, spec.experiment_id)
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(SimulationPipelineBlocked, match="evidence changed"):
        run_simulation_pipeline(paths, options, review(Verdict.PASS))

    assert load_project_config(paths).state is ProjectState.REQUIRES_HUMAN_REVIEW


def test_unstable_simulation_spec_never_enters_verified(tmp_path: Path) -> None:
    paths = simulation_project(tmp_path, "sim-unstable")
    spec = make_spec(
        experiment_id="exp-sim-unstable",
        arrival_rate=3.0,
        service_rate=3.0,
    )
    spec_file = write_spec(tmp_path, spec)

    with pytest.raises(UnstableQueueError):
        run_simulation_pipeline(
            paths, SimulationPipelineOptions(spec_file=spec_file), review(Verdict.PASS)
        )

    assert load_project_config(paths).state is ProjectState.PROBLEM_UNDERSTOOD
    assert not simulation_result_file(paths, spec.experiment_id).exists()
    assert not paths.releases_dir.exists() or not list(paths.releases_dir.iterdir())


def test_run_simulation_rejects_an_unhashed_spec(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "sim-hash", task=TaskFamily.SIMULATION)
    spec = make_spec(experiment_id="exp-sim-hash").model_copy(update={"spec_hash": "bad"})

    with pytest.raises(SimulationError, match="hash"):
        run_simulation(paths, spec)
