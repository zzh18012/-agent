import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmos.core.project import (
    ProjectPaths,
    create_project,
    graph_result_file,
    load_flow_audit,
    load_graph_result,
    load_min_cut_oracle,
    load_project_config,
    save_graph_record,
    save_graph_spec,
)
from mmos.core.state_machine import allowed_targets, task_order
from mmos.evidence.store import load_claims
from mmos.experiments.graph_oracle import enumerate_min_cuts
from mmos.experiments.graph_runner import build_graph_record, run_graph_experiment
from mmos.graph_pipeline import (
    GraphPipelineBlocked,
    GraphPipelineOptions,
    run_graph_pipeline,
)
from mmos.orchestration import Orchestrator
from mmos.paper.graph_builder import build_graph_paper
from mmos.schemas.graph import GraphEdge, GraphProblemSpec
from mmos.schemas.optimization import ExperimentStatus
from mmos.schemas.project import SourceRole
from mmos.schemas.review import (
    AttackReport,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)
from mmos.schemas.states import SUPPORTED_TASKS, ProjectState, TaskFamily
from mmos.validation.graph import validate_graph_experiment

BENCHMARK_SPEC = Path("benchmarks/graph/maxflow_v1/graph_spec.json")


def make_spec(
    nodes: list[str],
    edges: list[tuple[str, str, float]],
    *,
    source: str = "s",
    sink: str = "t",
    graph_id: str = "graph-test",
    experiment_id: str = "exp-graph-test",
) -> GraphProblemSpec:
    return GraphProblemSpec(
        graph_id=graph_id,
        experiment_id=experiment_id,
        nodes=nodes,
        edges=[
            GraphEdge(u=u, v=v, capacity=capacity) for u, v, capacity in edges
        ],
        source=source,
        sink=sink,
    ).with_hash()


def benchmark_spec() -> GraphProblemSpec:
    return GraphProblemSpec.model_validate_json(
        BENCHMARK_SPEC.read_text(encoding="utf-8")
    ).with_hash()


def prepared_experiment(tmp_path: Path, spec: GraphProblemSpec | None = None):
    paths, _ = create_project(tmp_path / "projects", "graph-unit", task=TaskFamily.GRAPH)
    spec = spec or benchmark_spec()
    save_graph_spec(paths, spec)
    result = run_graph_experiment(paths, spec)
    record = build_graph_record(
        paths, spec, result, status=ExperimentStatus.SUCCEEDED
    ).model_copy(update={"mlflow_run_id": "mlflow-test-run"})
    save_graph_record(paths, record)
    validate_graph_experiment(paths, spec.experiment_id)
    return paths, spec, result


def graph_project(tmp_path: Path, name: str = "graph") -> ProjectPaths:
    paths, _ = create_project(tmp_path / "projects", name, task=TaskFamily.GRAPH)
    problem = tmp_path / f"{name}-problem.md"
    problem.write_text("# Maximum flow network\n", encoding="utf-8")
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


def _tamper(paths: ProjectPaths, experiment_id: str, mutate) -> None:
    target = graph_result_file(paths, experiment_id)
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_graph_is_a_supported_task_family() -> None:
    assert TaskFamily.GRAPH in SUPPORTED_TASKS
    assert task_order(TaskFamily.GRAPH)[:3] == (
        ProjectState.INIT,
        ProjectState.PROJECT_INGESTED,
        ProjectState.PROBLEM_UNDERSTOOD,
    )
    assert ProjectState.PROTOTYPING in allowed_targets(
        ProjectState.PROBLEM_UNDERSTOOD, task=TaskFamily.GRAPH
    )


def test_known_benchmark_max_flow_and_min_cut(tmp_path: Path) -> None:
    paths, spec, result = prepared_experiment(tmp_path)
    oracle = load_min_cut_oracle(paths, spec.experiment_id)
    audit = load_flow_audit(paths, spec.experiment_id)

    assert result.max_flow_value == pytest.approx(12.0)
    assert oracle.minimum_cut_capacity == pytest.approx(12.0)
    assert oracle.enumerated_cut_count == 2 ** (len(spec.nodes) - 2)
    assert oracle.minimum_cut_count == len(oracle.minimum_cut_partitions)
    assert len(oracle.minimum_cut_partition_capacities) == oracle.minimum_cut_count
    assert all(
        capacity == pytest.approx(oracle.minimum_cut_capacity)
        for capacity in oracle.minimum_cut_partition_capacities
    )
    assert oracle.minimum_cut_count > 1
    assert len(oracle.all_cut_capacities) == oracle.expected_cut_count
    assert min(oracle.all_cut_capacities) == pytest.approx(
        oracle.minimum_cut_capacity
    )
    assert any(
        capacity > oracle.minimum_cut_capacity
        for capacity in oracle.all_cut_capacities
    )
    assert audit.capacity_ok is True
    assert audit.conservation_ok is True
    assert audit.source_flow_value == pytest.approx(12.0)
    assert audit.sink_flow_value == pytest.approx(12.0)


def test_validation_passes_for_known_graph(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    reports = validate_graph_experiment(paths, spec.experiment_id)

    assert len(reports) == 1
    assert reports[0].validator == "graph"
    assert reports[0].blocking is False


def test_capacity_violation_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        payload["edge_flows"][0]["flow"] = payload["edge_flows"][0]["capacity"] + 1

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_graph_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "capacity_violation" for issue in reports[0].issues)


def test_capacity_tolerance_direction_rejects_overshoot(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        edge = payload["edge_flows"][0]
        edge["flow"] = edge["capacity"] + 2e-6

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_graph_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "capacity_violation" for issue in reports[0].issues)


def test_negative_flow_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        payload["edge_flows"][0]["flow"] = -1

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_graph_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "negative_flow" for issue in reports[0].issues)


def test_intermediate_conservation_violation_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        for edge in payload["edge_flows"]:
            if (edge["u"], edge["v"]) == ("c", "e"):
                edge["flow"] -= 1
            if (edge["u"], edge["v"]) == ("e", "t"):
                edge["flow"] -= 1

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_graph_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(
        issue.code == "flow_conservation_violation" for issue in reports[0].issues
    )


def test_directed_cut_ignores_reverse_edges() -> None:
    spec = make_spec(
        ["s", "a", "t"],
        [("s", "a", 5), ("a", "t", 5), ("t", "a", 100)],
    )
    oracle = enumerate_min_cuts(spec)

    assert oracle.minimum_cut_capacity == pytest.approx(5.0)
    assert oracle.minimum_cut_count >= 1
    assert all(
        spec.source in partition and spec.sink not in partition
        for partition in oracle.minimum_cut_partitions
    )


def test_cut_enumeration_completeness_and_multiple_minima() -> None:
    spec = make_spec(
        ["s", "a", "b", "t"],
        [("s", "a", 5), ("s", "b", 5), ("a", "t", 5), ("b", "t", 5)],
    )
    oracle = enumerate_min_cuts(spec)

    assert oracle.enumerated_cut_count == 2 ** (len(spec.nodes) - 2)
    assert oracle.enumerated_cut_count == oracle.expected_cut_count
    assert oracle.minimum_cut_capacity == pytest.approx(10.0)
    assert oracle.minimum_cut_count >= 2
    assert ["s"] in oracle.minimum_cut_partitions
    assert ["a", "b", "s"] in oracle.minimum_cut_partitions


def test_disconnected_graph_has_zero_flow_and_cut(tmp_path: Path) -> None:
    spec = make_spec(
        ["s", "a", "b", "t"],
        [("s", "a", 5), ("b", "t", 5)],
    )
    paths, spec, result = prepared_experiment(tmp_path, spec)
    oracle = load_min_cut_oracle(paths, spec.experiment_id)

    assert result.max_flow_value == pytest.approx(0.0)
    assert oracle.minimum_cut_capacity == pytest.approx(0.0)
    reports = validate_graph_experiment(paths, spec.experiment_id)
    assert reports[0].blocking is False


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GraphEdge(u="s", v="a", capacity=-1.0)


def test_graph_hash_tracks_graph_changes() -> None:
    base = make_spec(["s", "a", "t"], [("s", "a", 5), ("a", "t", 5)])
    assert base.graph_hash == make_spec(
        ["s", "a", "t"], [("s", "a", 5), ("a", "t", 5)]
    ).graph_hash
    assert base.graph_hash != make_spec(
        ["s", "a", "t"], [("s", "a", 6), ("a", "t", 5)]
    ).graph_hash
    assert base.graph_hash != make_spec(
        ["s", "a", "t"], [("s", "a", 5)]
    ).graph_hash
    assert base.graph_hash != make_spec(
        ["s", "a", "t"],
        [("s", "a", 5), ("a", "t", 5)],
        source="a",
        sink="t",
    ).graph_hash


def test_graph_result_is_deterministic(tmp_path: Path) -> None:
    paths, spec, result = prepared_experiment(tmp_path)
    second = run_graph_experiment(paths, spec)
    oracle = load_min_cut_oracle(paths, spec.experiment_id)

    assert result.determinism["match"] is True
    assert result.determinism["first_run_digest"] == result.determinism["second_run_digest"]
    assert second.max_flow_value == result.max_flow_value
    assert [edge.flow for edge in second.edge_flows] == [
        edge.flow for edge in result.edge_flows
    ]
    assert oracle.minimum_cut_capacity == result.max_flow_value


def test_graph_paper_is_scoped_and_auditable(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    paper = build_graph_paper(paths, spec.experiment_id)

    text = paper.abstract + " " + " ".join(
        block for section in paper.sections for block in section.blocks
    )
    assert "Internal Plumbing Test Note" in paper.title
    assert "does not validate other graphs" in text
    assert "not" in text and "performance" in text
    assert "NetworkX" in text
    assert "all 64" in text
    assert len(paper.tables[2].rows) == 12
    assert "Cut capacity" in paper.tables[2].columns
    assert all(row[1] == "12.000000" for row in paper.tables[2].rows)
    assert "64 of 64 legal partitions" in paper.tables[2].caption
    assert "Residual" in paper.tables[0].columns
    assert any(
        column.startswith("Crosses cut {") for column in paper.tables[0].columns
    )
    assert "Saturated (flow == capacity)" in paper.tables[0].columns
    assert "Worked cut derivations" in text
    assert paper.tables[1].rows
    histogram = paper.tables[3]
    capacities = {row[0] for row in histogram.rows}
    assert "12.000000" in capacities
    assert any(float(capacity) > 12.0 for capacity in capacities)
    assert sum(int(row[1]) for row in histogram.rows) == 64
    full_cuts = paper.tables[4]
    assert len(full_cuts.rows) == 64
    assert sum(1 for row in full_cuts.rows if row[2] == "yes") == 12


def test_graph_pipeline_reaches_released_on_pass(tmp_path: Path) -> None:
    paths = graph_project(tmp_path, "graph-pass")
    summary = run_graph_pipeline(
        paths,
        GraphPipelineOptions(spec_file=BENCHMARK_SPEC),
        review(Verdict.PASS),
    )

    assert summary.state is ProjectState.RELEASED
    assert summary.max_flow_value == pytest.approx(12.0)
    assert summary.min_cut_capacity == pytest.approx(12.0)
    assert summary.validation == {"graph": "passed"}
    assert summary.release_path is not None and summary.release_path.exists()
    assert load_project_config(paths).state is ProjectState.RELEASED
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


def test_graph_revise_blocks_then_resume_reuses_result(tmp_path: Path) -> None:
    paths = graph_project(tmp_path, "graph-resume")
    options = GraphPipelineOptions(spec_file=BENCHMARK_SPEC)

    with pytest.raises(GraphPipelineBlocked):
        run_graph_pipeline(paths, options, review(Verdict.REVISE))

    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.RED_TEAM_REVIEW
    result_path = graph_result_file(paths, "exp-maxflow-benchmark-v1")
    mtime = result_path.stat().st_mtime_ns
    completed_at = load_graph_result(paths, "exp-maxflow-benchmark-v1").completed_at

    summary = run_graph_pipeline(paths, options, review(Verdict.PASS))

    assert summary.state is ProjectState.RELEASED
    assert load_graph_result(paths, "exp-maxflow-benchmark-v1").completed_at == completed_at
    assert result_path.stat().st_mtime_ns == mtime


def test_graph_resume_rejects_changed_evidence(tmp_path: Path) -> None:
    paths = graph_project(tmp_path, "graph-evidence")
    options = GraphPipelineOptions(spec_file=BENCHMARK_SPEC)

    with pytest.raises(GraphPipelineBlocked):
        run_graph_pipeline(paths, options, review(Verdict.REVISE))
    result_path = graph_result_file(paths, "exp-maxflow-benchmark-v1")
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(GraphPipelineBlocked, match="evidence changed"):
        run_graph_pipeline(paths, options, review(Verdict.PASS))

    assert load_project_config(paths).state is ProjectState.REQUIRES_HUMAN_REVIEW
