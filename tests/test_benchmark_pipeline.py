import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mmos.benchmark_pipeline import (
    BenchmarkPipelineBlocked,
    BenchmarkPipelineOptions,
    run_benchmark_pipeline,
)
from mmos.core.project import (
    ProjectPaths,
    create_project,
    load_project_config,
    sha256_file,
)
from mmos.evidence.store import load_claims
from mmos.experiments.knapsack_benchmark import DEFAULT_SUITE_DIR, ManifestError
from mmos.experiments.knapsack_suite import load_benchmark_result
from mmos.paper.benchmark_builder import build_benchmark_paper
from mmos.schemas.benchmark import knapsack_instance_hash
from mmos.schemas.review import (
    AttackReport,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)
from mmos.schemas.states import ProjectState, TaskFamily


def _clone(template_root: Path, tmp_path: Path, name: str) -> ProjectPaths:
    target = tmp_path / name
    shutil.copytree(template_root, target)
    return ProjectPaths(target.resolve())


def _existing_suite(paths: ProjectPaths, suite_dir: Path, *, time_limit_seconds: float = 10.0):
    artifacts = sorted(
        (paths.experiments_dir / "benchmarks").glob("*/benchmark_result.json")
    )
    assert artifacts
    return load_benchmark_result(artifacts[0])


def _review(verdict: Verdict) -> RedTeamReview:
    return RedTeamReview(
        experiment_id="benchmark",
        attacks=AttackReport(attacks=[]),
        responses=AuthorResponseReport(responses=[]),
        judge=JudgeReport(verdicts=[], overall=verdict, blocking=False),
        created_at=datetime.now(UTC),
    )


def _history_states(paths: ProjectPaths) -> list[str]:
    lines = paths.state_history_file.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["to_state"] for line in lines if line.strip()]


def test_benchmark_pipeline_reaches_released_on_pass(
    benchmark_template: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_root, _ = benchmark_template
    paths = _clone(template_root, tmp_path, "pipeline-pass")
    monkeypatch.setattr(
        "mmos.benchmark_pipeline.run_knapsack_suite", _existing_suite
    )
    manifest_hash_before = sha256_file(DEFAULT_SUITE_DIR / "benchmark_manifest.json")
    instance_before = sha256_file(DEFAULT_SUITE_DIR / "instances" / "greedy_ratio_trap.json")

    summary = run_benchmark_pipeline(
        paths,
        BenchmarkPipelineOptions(suite_dir=DEFAULT_SUITE_DIR),
        lambda ast, run_id: _review(Verdict.PASS),
    )

    assert summary.state is ProjectState.RELEASED
    assert summary.validation == {
        "benchmark_integrity": "passed",
        "benchmark_oracle": "passed",
        "benchmark_aggregate": "passed",
    }
    assert summary.expected_experiment_count == 27
    assert summary.expected_tournament_count == 9
    assert summary.release_path is not None and summary.release_path.exists()
    assert load_project_config(paths).state is ProjectState.RELEASED
    assert _history_states(paths) == [
        "PROJECT_INGESTED",
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
    ]

    run = summary.benchmark_run_id
    suite_record = json.loads(
        (
            paths.experiments_dir / "benchmarks" / run / "suite_run.json"
        ).read_text(encoding="utf-8")
    )
    assert suite_record["project_state"] == "RELEASED"
    assert suite_record["actual_experiment_count"] == 27
    assert suite_record["actual_tournament_count"] == 9
    assert len(suite_record["levels"]["experiments"]) == 27
    assert len(suite_record["levels"]["tournaments"]) == 9
    assert suite_record["red_team"] == "pass"

    release = json.loads(
        (paths.releases_dir / f"benchmark_{run}.json").read_text(encoding="utf-8")
    )
    assert release["red_team"] == "pass"
    assert release["actual_experiment_count"] == 27
    assert release["actual_tournament_count"] == 9
    assert set(release["validation"].values()) == {"passed"}

    claims = load_claims(paths)
    assert len(claims) == 2
    assert all(claim.status == "proposed" for claim in claims)
    assert all("frozen suite" in claim.statement for claim in claims)

    assert sha256_file(DEFAULT_SUITE_DIR / "benchmark_manifest.json") == manifest_hash_before
    assert (
        sha256_file(DEFAULT_SUITE_DIR / "instances" / "greedy_ratio_trap.json")
        == instance_before
    )


def test_non_pass_review_enters_human_review_without_release(
    benchmark_template: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_root, _ = benchmark_template
    paths = _clone(template_root, tmp_path, "pipeline-revise")
    monkeypatch.setattr(
        "mmos.benchmark_pipeline.run_knapsack_suite", _existing_suite
    )

    with pytest.raises(BenchmarkPipelineBlocked) as excinfo:
        run_benchmark_pipeline(
            paths,
            BenchmarkPipelineOptions(suite_dir=DEFAULT_SUITE_DIR),
            lambda ast, run_id: _review(Verdict.REVISE),
        )

    run_id = excinfo.value.benchmark_run_id
    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.RED_TEAM_REVIEW
    assert not (paths.releases_dir / f"benchmark_{run_id}.md").exists()
    assert (paths.reviews_dir / f"blocked_benchmark_{run_id}.md").exists()
    suite_record = json.loads(
        (
            paths.experiments_dir / "benchmarks" / run_id / "suite_run.json"
        ).read_text(encoding="utf-8")
    )
    assert suite_record["project_state"] == "REQUIRES_HUMAN_REVIEW"
    assert suite_record["red_team"] == "revise"


def test_blocked_review_can_resume_to_release(
    benchmark_template: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_root, _ = benchmark_template
    paths = _clone(template_root, tmp_path, "pipeline-resume")
    monkeypatch.setattr("mmos.benchmark_pipeline.run_knapsack_suite", _existing_suite)

    with pytest.raises(BenchmarkPipelineBlocked):
        run_benchmark_pipeline(
            paths,
            BenchmarkPipelineOptions(suite_dir=DEFAULT_SUITE_DIR),
            lambda ast, run_id: _review(Verdict.REVISE),
        )
    assert load_project_config(paths).state is ProjectState.REQUIRES_HUMAN_REVIEW

    summary = run_benchmark_pipeline(
        paths,
        BenchmarkPipelineOptions(suite_dir=DEFAULT_SUITE_DIR),
        lambda ast, run_id: _review(Verdict.PASS),
    )

    assert summary.state is ProjectState.RELEASED
    assert summary.release_path is not None and summary.release_path.exists()
    assert load_project_config(paths).state is ProjectState.RELEASED
    history = _history_states(paths)
    assert history.count("RED_TEAM_REVIEW") >= 2
    assert history[-1] == "RELEASED"


def test_missing_manifest_instance_blocks_before_experiment_completed(
    benchmark_template: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_root, _ = benchmark_template
    paths = _clone(template_root, tmp_path, "pipeline-missing")

    def tampered(paths: ProjectPaths, suite_dir: Path, *, time_limit_seconds: float = 10.0):
        result = _existing_suite(paths, suite_dir)
        rows = [
            row for row in result.per_instance if row.instance_id != "greedy_ratio_trap"
        ]
        return result.model_copy(
            update={"per_instance": rows, "n_instances": len(rows), "status": "complete"}
        )

    monkeypatch.setattr("mmos.benchmark_pipeline.run_knapsack_suite", tampered)

    with pytest.raises(BenchmarkPipelineBlocked):
        run_benchmark_pipeline(
            paths,
            BenchmarkPipelineOptions(suite_dir=DEFAULT_SUITE_DIR),
            lambda ast, run_id: _review(Verdict.PASS),
        )

    config = load_project_config(paths)
    assert config.state is ProjectState.REQUIRES_HUMAN_REVIEW
    assert config.resume_state is ProjectState.PROTOTYPED
    history = _history_states(paths)
    assert "EXPERIMENT_COMPLETED" not in history
    suite_records = list(
        (paths.experiments_dir / "benchmarks").glob("*/suite_run.json")
    )
    assert suite_records
    record = json.loads(suite_records[0].read_text(encoding="utf-8"))
    assert record["completion_issues"]


def test_mutated_frozen_manifest_is_rejected_before_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = create_project(
        tmp_path / "projects", "manifest-mutation", task=TaskFamily.OPTIMIZATION
    )
    suite_dir = tmp_path / "suite"
    shutil.copytree(DEFAULT_SUITE_DIR, suite_dir)
    manifest_path = suite_dir / "benchmark_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["instances"][0]["weights"][0] += 1
    payload["instances"][0]["instance_hash"] = knapsack_instance_hash(
        payload["instances"][0]["weights"],
        payload["instances"][0]["values"],
        payload["instances"][0]["capacity"],
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="hash mismatch"):
        run_benchmark_pipeline(
            paths,
            BenchmarkPipelineOptions(suite_dir=suite_dir),
            lambda ast, run_id: _review(Verdict.PASS),
        )

    assert load_project_config(paths).state is ProjectState.INIT
    assert not paths.releases_dir.exists() or not list(paths.releases_dir.iterdir())


def test_paper_is_scoped_and_auditable(
    benchmark_template: tuple[Path, str], tmp_path: Path
) -> None:
    template_root, run_id = benchmark_template
    paths = _clone(template_root, tmp_path, "paper-scope")

    paper = build_benchmark_paper(paths, run_id)

    text = paper.abstract + " " + " ".join(
        block for section in paper.sections for block in section.blocks
    )
    headings = [section.heading for section in paper.sections]
    assert "Adversarial findings" not in headings
    assert "Per-instance descriptive results" in headings
    assert "do not establish general superiority" in text
    assert "not evidence of heuristic improvement" in text
    assert "n=3-10" in text
    assert "random_correlated_2" in text
    assert "general greedy failure rates" in text
    assert all(table.evidence_ids for table in paper.tables)
    reconciliation = paper.tables[0]
    assert {row[0] for row in reconciliation.rows} == {
        row.instance_id
        for row in load_benchmark_result(
            paths.experiments_dir / "benchmarks" / run_id / "benchmark_result.json"
        ).per_instance
    }
    instance_table = paper.tables[1]
    assert "CBC - greedy" in instance_table.columns
    assert "Enhanced - greedy" in instance_table.columns
    assert "Independent feasible" in instance_table.columns
    greedy_trap = next(row for row in instance_table.rows if row[0] == "greedy_ratio_trap")
    assert greedy_trap[4] == "18"
    assert greedy_trap[6] == "12"
    assert all(len(cell.split(".")[1]) == 4 for row in instance_table.rows for cell in row[10:12])
    reconciliation = paper.tables[0]
    assert "Partition" in reconciliation.columns
    assert all(len(row[6]) == 64 for row in reconciliation.rows)
    appendix = paper.tables[4]
    assert len(appendix.rows) == len(reconciliation.rows)
    evidence = paper.tables[5]
    assert any(row[1] == "benchmark.manifest_hash" for row in evidence.rows)
