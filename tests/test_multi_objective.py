import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmos.core.project import (
    ProjectPaths,
    create_project,
    load_multi_objective_result,
    load_pareto_oracle,
    load_project_config,
    multi_objective_result_file,
    save_multi_objective_record,
    save_multi_objective_spec,
)
from mmos.core.state_machine import allowed_targets, task_order
from mmos.evidence.store import load_claims
from mmos.experiments.multi_objective_oracle import enumerate_pareto_oracle
from mmos.experiments.multi_objective_runner import (
    _candidate_dominates,
    build_multi_objective_record,
    run_multi_objective,
)
from mmos.experiments.pareto_utils import hypervolume_2d
from mmos.multi_objective_pipeline import (
    MultiObjectivePipelineBlocked,
    MultiObjectivePipelineOptions,
    run_multi_objective_pipeline,
)
from mmos.orchestration import Orchestrator
from mmos.paper.multi_objective_builder import build_multi_objective_paper
from mmos.schemas.multi_objective import MultiObjectiveSpec, ObjectiveSpec, ProjectItem
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
from mmos.validation.multi_objective import validate_multi_objective_experiment

BENCHMARK_SPEC = Path("benchmarks/multi_objective/portfolio_v1/instance_spec.json")


def make_spec(
    items: list[tuple[str, float, float, float]],
    *,
    budget: float,
    problem_id: str = "mo-test",
    experiment_id: str = "exp-mo-test",
    reference_point: list[float] | None = None,
) -> MultiObjectiveSpec:
    return MultiObjectiveSpec(
        problem_id=problem_id,
        experiment_id=experiment_id,
        items=[
            ProjectItem(
                project_id=project_id,
                cost=cost,
                benefits={"A": benefit_a, "B": benefit_b},
            )
            for project_id, cost, benefit_a, benefit_b in items
        ],
        budget=budget,
        objectives=[
            ObjectiveSpec(objective_id="A", definition="total benefit A"),
            ObjectiveSpec(objective_id="B", definition="total benefit B"),
        ],
        reference_point=reference_point or [0, 0],
    ).with_hash()


def benchmark_spec() -> MultiObjectiveSpec:
    return MultiObjectiveSpec.model_validate_json(
        BENCHMARK_SPEC.read_text(encoding="utf-8")
    ).with_hash()


def prepared_experiment(tmp_path: Path, spec: MultiObjectiveSpec | None = None):
    paths, _ = create_project(
        tmp_path / "projects", "mo-unit", task=TaskFamily.MULTI_OBJECTIVE
    )
    spec = spec or benchmark_spec()
    save_multi_objective_spec(paths, spec)
    result = run_multi_objective(paths, spec)
    record = build_multi_objective_record(
        paths, spec, result, status=ExperimentStatus.SUCCEEDED
    ).model_copy(update={"mlflow_run_id": "mlflow-test-run"})
    save_multi_objective_record(paths, record)
    validate_multi_objective_experiment(paths, spec.experiment_id)
    return paths, spec, result


def mo_project(tmp_path: Path, name: str = "mo") -> ProjectPaths:
    paths, _ = create_project(
        tmp_path / "projects", name, task=TaskFamily.MULTI_OBJECTIVE
    )
    problem = tmp_path / f"{name}-problem.md"
    problem.write_text("# bi-objective portfolio\n", encoding="utf-8")
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
    target = multi_objective_result_file(paths, experiment_id)
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_multi_objective_is_a_supported_task_family() -> None:
    assert TaskFamily.MULTI_OBJECTIVE in SUPPORTED_TASKS
    assert task_order(TaskFamily.MULTI_OBJECTIVE)[:3] == (
        ProjectState.INIT,
        ProjectState.PROJECT_INGESTED,
        ProjectState.PROBLEM_UNDERSTOOD,
    )
    assert ProjectState.PROTOTYPING in allowed_targets(
        ProjectState.PROBLEM_UNDERSTOOD, task=TaskFamily.MULTI_OBJECTIVE
    )


def test_maximize_dominance_semantics() -> None:
    assert _candidate_dominates((10.0, 10.0), (9.0, 10.0)) is True
    assert _candidate_dominates((10.0, 10.0), (10.0, 10.0)) is False
    assert _candidate_dominates((10.0, 5.0), (5.0, 10.0)) is False
    assert _candidate_dominates((5.0, 10.0), (10.0, 5.0)) is False
    assert _candidate_dominates((9.0, 9.0), (10.0, 5.0)) is False


def test_hypervolume_manual_examples() -> None:
    assert hypervolume_2d([(1.0, 4.0), (2.0, 3.0)], (0.0, 0.0)) == pytest.approx(7.0)
    assert hypervolume_2d(
        [(5.0, 10.0), (10.0, 5.0)], (0.0, 0.0)
    ) == pytest.approx(75.0)


def test_oracle_front_and_set_distinguish_duplicate_objective_vectors() -> None:
    spec = make_spec(
        [
            ("p1", 1, 1, 0),
            ("p2", 2, 1, 0),
        ],
        budget=2,
    )
    oracle = enumerate_pareto_oracle(spec)

    assert oracle.search_space_size == 4
    assert oracle.front_size == 1
    assert [(vector.f1, vector.f2) for vector in oracle.pareto_objective_vectors] == [
        (1.0, 0.0)
    ]
    assert oracle.set_size == 2
    decisions = {
        tuple(sorted(solution.decision_vector.items()))
        for solution in oracle.pareto_decision_vectors
    }
    assert decisions == {
        (("p1", 1), ("p2", 0)),
        (("p1", 0), ("p2", 1)),
    }


def test_epsilon_schedule_is_independent_of_oracle() -> None:
    spec = benchmark_spec()
    levels = spec.epsilon_levels()

    assert levels[0] == 0
    assert levels[-1] == int(sum(item.benefits["B"] for item in spec.items))
    assert len(levels) == levels[-1] + 1
    # The schedule is a pure function of the frozen spec and policy.
    assert benchmark_spec().epsilon_levels() == levels


def test_positive_reference_point_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_spec([("p1", 1, 1, 1)], budget=1, reference_point=[1, 0])


def test_benchmark_candidate_matches_oracle_front(tmp_path: Path) -> None:
    paths, spec, result = prepared_experiment(tmp_path)
    oracle = load_pareto_oracle(paths, spec.experiment_id)

    assert result.candidate_front_size == oracle.front_size
    assert result.comparison is not None
    assert result.comparison.exact_match is True
    assert result.comparison.precision == pytest.approx(1.0)
    assert result.comparison.recall == pytest.approx(1.0)
    assert result.comparison.missing_vectors == []
    assert result.comparison.extra_vectors == []
    assert result.candidate_front_size == 10
    assert oracle.set_size > oracle.front_size


def test_validation_passes_for_exact_front(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    reports = validate_multi_objective_experiment(paths, spec.experiment_id)

    assert len(reports) == 1
    assert reports[0].validator == "multi_objective"
    assert reports[0].blocking is False


def test_missing_pareto_point_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        payload["candidate_pareto_front"] = payload["candidate_pareto_front"][1:]

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_multi_objective_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "missing_pareto_point" for issue in reports[0].issues)


def test_false_pareto_point_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        payload["candidate_pareto_front"].append({"f1": 0.0, "f2": 0.0})

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_multi_objective_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "false_pareto_point" for issue in reports[0].issues)


def test_budget_violation_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        solution = payload["candidate_pareto_set"][0]
        solution["decision_vector"] = {
            item.project_id: 1 for item in spec.items
        }

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_multi_objective_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "candidate_infeasible" for issue in reports[0].issues)


def test_non_binary_decision_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        solution = payload["candidate_pareto_set"][0]
        first = sorted(solution["decision_vector"])[0]
        solution["decision_vector"][first] = 2

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_multi_objective_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(issue.code == "candidate_infeasible" for issue in reports[0].issues)


def test_objective_recompute_mismatch_blocks_validation(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    def mutate(payload: dict) -> None:
        payload["candidate_pareto_set"][0]["f1"] += 1

    _tamper(paths, spec.experiment_id, mutate)
    reports = validate_multi_objective_experiment(paths, spec.experiment_id)

    assert reports[0].blocking is True
    assert any(
        issue.code == "objective_recompute_mismatch" for issue in reports[0].issues
    )


def test_multi_objective_paper_has_no_winner(tmp_path: Path) -> None:
    paths, spec, _ = prepared_experiment(tmp_path)

    paper = build_multi_objective_paper(paths, spec.experiment_id)

    text = paper.abstract + " " + " ".join(
        block for section in paper.sections for block in section.blocks
    )
    assert "Bi-objective Pareto-front Conformance Report" in paper.title
    assert "no unique" in text.lower()
    assert "not used to select" in text
    assert "weighted" in text and "No weighted sum" in text
    assert "precision" in text.lower() or "front precision" in text.lower()
    assert len(paper.figures) == 1
    assert paper.figures[0].path.endswith(".png")
    assert len(paper.tables) == 5
    assert any(row[0] == "Exact objective-vector match" for row in paper.tables[3].rows)
    assert any(
        row[0] == "Decision-space recall (candidate/oracle)"
        for row in paper.tables[3].rows
    )
    sensitivity = paper.tables[4]
    assert [row[0] for row in sensitivity.rows] == ["1 (primary)", "2", "5"]
    assert "recall" in sensitivity.caption.lower()
    assert "Coarse-grid misses" in text


def test_multi_objective_pipeline_reaches_released_on_pass(tmp_path: Path) -> None:
    paths = mo_project(tmp_path, "mo-pass")
    summary = run_multi_objective_pipeline(
        paths,
        MultiObjectivePipelineOptions(spec_file=BENCHMARK_SPEC),
        review(Verdict.PASS),
    )

    assert summary.state is ProjectState.RELEASED
    assert summary.exact_match is True
    assert summary.candidate_front_size == summary.oracle_front_size
    assert summary.validation == {"multi_objective": "passed"}
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


def test_multi_objective_revise_blocks_then_resume_reuses_result(
    tmp_path: Path,
) -> None:
    paths = mo_project(tmp_path, "mo-resume")
    options = MultiObjectivePipelineOptions(spec_file=BENCHMARK_SPEC)

    with pytest.raises(MultiObjectivePipelineBlocked):
        run_multi_objective_pipeline(paths, options, review(Verdict.REVISE))

    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.RED_TEAM_REVIEW
    result_path = multi_objective_result_file(paths, "exp-portfolio-benchmark-v1")
    mtime = result_path.stat().st_mtime_ns
    completed_at = load_multi_objective_result(
        paths, "exp-portfolio-benchmark-v1"
    ).completed_at

    summary = run_multi_objective_pipeline(paths, options, review(Verdict.PASS))

    assert summary.state is ProjectState.RELEASED
    assert result_path.stat().st_mtime_ns == mtime
    assert (
        load_multi_objective_result(paths, "exp-portfolio-benchmark-v1").completed_at
        == completed_at
    )


def test_multi_objective_resume_rejects_changed_evidence(tmp_path: Path) -> None:
    paths = mo_project(tmp_path, "mo-evidence")
    options = MultiObjectivePipelineOptions(spec_file=BENCHMARK_SPEC)

    with pytest.raises(MultiObjectivePipelineBlocked):
        run_multi_objective_pipeline(paths, options, review(Verdict.REVISE))
    result_path = multi_objective_result_file(paths, "exp-portfolio-benchmark-v1")
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(MultiObjectivePipelineBlocked, match="evidence changed"):
        run_multi_objective_pipeline(paths, options, review(Verdict.PASS))

    assert load_project_config(paths).state is ProjectState.REQUIRES_HUMAN_REVIEW
