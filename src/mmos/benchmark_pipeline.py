import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mmos.agents import release_allowed
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_record,
    load_optimization_result,
    load_project_config,
    save_red_team_review,
)
from mmos.evidence import EvidenceError, add_claim, freeze_benchmark_suite
from mmos.evidence.store import load_claims
from mmos.experiments.knapsack_benchmark import (
    freeze_manifest,
    generator_code_hash,
    load_manifest,
    methods_code_hash,
)
from mmos.experiments.knapsack_suite import load_benchmark_result, run_knapsack_suite
from mmos.experiments.optimization_tournament import load_optimization_tournament
from mmos.orchestration import Orchestrator
from mmos.paper.benchmark_builder import build_benchmark_paper
from mmos.paper.render import render_paper
from mmos.schemas.benchmark import BenchmarkManifest, BenchmarkSuiteResult
from mmos.schemas.paper import PaperAST
from mmos.schemas.review import RedTeamReview
from mmos.schemas.states import ProjectState, TaskFamily
from mmos.validation.benchmark import validate_benchmark_suite

Reviewer = Callable[[PaperAST, str], RedTeamReview]


class BenchmarkPipelineBlocked(RuntimeError):
    def __init__(self, run_id: str, state: ProjectState, reason: str) -> None:
        self.benchmark_run_id = run_id
        self.state = state
        super().__init__(f"benchmark {run_id} blocked in {state.value}: {reason}")


@dataclass(frozen=True)
class BenchmarkPipelineOptions:
    suite_dir: Path
    suite_seed: int = 20261113
    time_limit_seconds: float = 10.0


@dataclass(frozen=True)
class BenchmarkRunSummary:
    benchmark_run_id: str
    suite_id: str
    manifest_hash: str
    n_instances: int
    expected_experiment_count: int
    expected_tournament_count: int
    validation: dict[str, str] = field(default_factory=dict)
    claim_ids: list[str] = field(default_factory=list)
    red_team_overall: str = ""
    paper_path: Path | None = None
    review_path: Path | None = None
    release_path: Path | None = None
    state: ProjectState = ProjectState.INIT


def _expected_experiment_ids(suite_id: str, instance_id: str) -> list[str]:
    return [
        f"exp-{suite_id}-{instance_id}-cbc",
        f"exp-{suite_id}-{instance_id}-greedy",
        f"exp-{suite_id}-{instance_id}-enhanced-greedy",
    ]


def _completion_issues(
    paths: ProjectPaths, manifest: BenchmarkManifest, result: BenchmarkSuiteResult
) -> list[str]:
    issues: list[str] = []
    manifest_ids = [instance.instance_id for instance in manifest.instances]
    expected_experiments = 3 * len(manifest_ids)
    expected_tournaments = len(manifest_ids)

    if result.suite_id != manifest.suite_id:
        issues.append(f"suite id {result.suite_id} != manifest {manifest.suite_id}")
    if result.manifest_hash != manifest.manifest_hash:
        issues.append("manifest hash differs from the frozen manifest")
    if result.n_instances != len(manifest_ids):
        issues.append(f"n_instances {result.n_instances} != {len(manifest_ids)}")
    if result.status != "complete" or result.missing_instances:
        issues.append(
            f"suite status {result.status}; missing {result.missing_instances}"
        )
    if result.methods_code_hash != methods_code_hash():
        issues.append("methods code hash differs from the current implementation")
    if result.generator_code_hash != generator_code_hash():
        issues.append("generator code hash differs from the current implementation")

    rows = {row.instance_id: row for row in result.per_instance}
    if len(rows) != len(result.per_instance):
        issues.append("duplicate per-instance result rows")
    if set(rows) != set(manifest_ids):
        issues.append(
            f"result instances {sorted(set(rows) ^ set(manifest_ids))} differ from manifest"
        )

    indexed_experiments = sum(len(ids) for ids in result.experiment_ids.values())
    if indexed_experiments != expected_experiments:
        issues.append(
            f"indexed experiments {indexed_experiments} != expected {expected_experiments}"
        )
    if len(result.tournament_ids) != expected_tournaments:
        issues.append(
            f"indexed tournaments {len(result.tournament_ids)} != expected "
            f"{expected_tournaments}"
        )

    for instance_id in manifest_ids:
        row = rows.get(instance_id)
        if row is None:
            issues.append(f"{instance_id}: no result row")
            continue
        expected = _expected_experiment_ids(manifest.suite_id, instance_id)
        actual = [
            row.candidate_experiment_id,
            row.greedy_experiment_id,
            row.enhanced_greedy_experiment_id,
        ]
        if actual != expected:
            issues.append(f"{instance_id}: experiment ids {actual} != {expected}")
            continue
        if row.tournament_id is None:
            issues.append(f"{instance_id}: missing tournament")
            continue
        for experiment_id in expected:
            try:
                load_optimization_result(paths, experiment_id)
                load_experiment_record(paths, experiment_id)
            except FileNotFoundError:
                issues.append(f"{instance_id}: missing artifacts for {experiment_id}")
        try:
            load_optimization_tournament(paths, row.tournament_id)
        except FileNotFoundError:
            issues.append(f"{instance_id}: missing tournament {row.tournament_id}")
    return issues


def _suite_run_record(
    paths: ProjectPaths,
    manifest: BenchmarkManifest,
    result: BenchmarkSuiteResult,
    *,
    state: ProjectState,
    validation: dict[str, str] | None = None,
    red_team: str | None = None,
    release_path: Path | None = None,
    completion_issues: list[str] | None = None,
) -> Path:
    artifact_dir = paths.experiments_dir / "benchmarks" / result.benchmark_run_id
    record = {
        "schema_version": 1,
        "benchmark_run_id": result.benchmark_run_id,
        "project_id": paths.root.name,
        "suite_id": result.suite_id,
        "suite_version": manifest.suite_version,
        "manifest_hash": result.manifest_hash,
        "suite_seed": result.suite_seed,
        "n_instances": result.n_instances,
        "expected_experiment_count": 3 * len(manifest.instances),
        "expected_tournament_count": len(manifest.instances),
        "actual_experiment_count": sum(len(ids) for ids in result.experiment_ids.values()),
        "actual_tournament_count": len(result.tournament_ids),
        "methods_code_hash": result.methods_code_hash,
        "runtime": result.runtime,
        "git_commit": result.git_commit,
        "benchmark_run_at": result.completed_at.isoformat(),
        "project_state": state.value,
        "validation": validation or {},
        "red_team": red_team,
        "completion_issues": completion_issues or [],
        "release_path": release_path.relative_to(paths.root).as_posix()
        if release_path is not None
        else None,
        "levels": {
            "project": paths.root.name,
            "benchmark_run": result.benchmark_run_id,
            "experiments": sorted(
                experiment_id
                for ids in result.experiment_ids.values()
                for experiment_id in ids
            ),
            "tournaments": sorted(result.tournament_ids.values()),
        },
        "recorded_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    target = artifact_dir / "suite_run.json"
    atomic_write_text(target, json.dumps(record, indent=2) + "\n")
    return target


def _blocked_artifact(paths: ProjectPaths, run_id: str, title: str, lines: list[str]) -> Path:
    target = paths.reviews_dir / f"blocked_benchmark_{run_id}.md"
    atomic_write_text(target, "\n".join([f"# {title}", "", *lines, ""]))
    return target


def _review_blocked_lines(review: RedTeamReview) -> list[str]:
    lines: list[str] = []
    for attack in review.attacks.attacks:
        verdict = next(
            verdict
            for verdict in review.judge.verdicts
            if verdict.attack_id == attack.attack_id
        )
        lines.extend(
            [
                f"## {attack.attack_id}: {attack.severity.value} / "
                f"{attack.category.value} -> {verdict.verdict.value}",
                attack.argument,
                f"Required fix: {attack.required_fix}",
                f"Judge: {verdict.rationale}",
                "",
            ]
        )
    return lines


def _register_claims(paths: ProjectPaths, result: BenchmarkSuiteResult, manifest) -> list[str]:
    note = "Auto-generated from the frozen benchmark suite; human acceptance is required."
    claim_ids: list[str] = []
    run = result.benchmark_run_id
    try:
        claim = add_claim(
            paths,
            statement=(
                f"In frozen suite {result.suite_id} v{manifest.suite_version} "
                f"(manifest {result.manifest_hash[:12]}...), CBC matched the independent "
                f"exhaustive oracle on {result.candidate_verified_optimum_count}/"
                f"{result.n_instances} instances; this is a solver-conformance check for "
                "these instances only."
            ),
            evidence_ids=[
                f"ev-{run}-candidate-verified",
                f"ev-{run}-oracle-agreement",
                f"ev-{run}-manifest-hash",
            ],
            claim_id=f"claim-benchmark-{run}-optimum",
            accept=False,
            notes=note,
        )
        claim_ids.append(claim.claim_id)
    except EvidenceError:
        pass
    max_regret = result.greedy.max_relative_regret
    worst = result.greedy.worst_instance_id
    if max_regret is not None and worst is not None:
        try:
            claim = add_claim(
                paths,
                statement=(
                    f"In frozen suite {result.suite_id} v{manifest.suite_version}, ratio "
                    f"greedy reached the optimum on {result.greedy.optimum_hits}/"
                    f"{result.n_instances} instances; the largest observed relative regret "
                    f"was {max_regret:.4f} on {worst}. This is a descriptive result at "
                    "n=3-10 for this suite only."
                ),
                evidence_ids=[
                    f"ev-{run}-greedy-hit-rate",
                    f"ev-{run}-greedy-failures",
                    f"ev-{run}-greedy-worst",
                ],
                claim_id=f"claim-benchmark-{run}-greedy",
                accept=False,
                notes=note,
            )
            claim_ids.append(claim.claim_id)
        except EvidenceError:
            pass
    return claim_ids


def run_benchmark_pipeline(
    paths: ProjectPaths,
    options: BenchmarkPipelineOptions,
    reviewer: Reviewer,
) -> BenchmarkRunSummary:
    config = load_project_config(paths)
    if config.task is not TaskFamily.OPTIMIZATION:
        raise ValueError(
            f"benchmark suite requires an optimization project; task is {config.task.value}"
        )
    if config.state is ProjectState.INIT:
        return _run_fresh(paths, options, reviewer)
    if config.state is ProjectState.RED_TEAM_REVIEW or (
        config.state is ProjectState.REQUIRES_HUMAN_REVIEW
        and config.resume_state is ProjectState.RED_TEAM_REVIEW
    ):
        return _resume_review(paths, options, reviewer)
    raise ValueError(
        "benchmark suite requires INIT or a resume point at RED_TEAM_REVIEW; "
        f"state is {config.state.value}"
    )


def _run_fresh(
    paths: ProjectPaths,
    options: BenchmarkPipelineOptions,
    reviewer: Reviewer,
) -> BenchmarkRunSummary:
    manifest, manifest_path = freeze_manifest(options.suite_dir, suite_seed=options.suite_seed)
    if manifest.suite_seed != options.suite_seed:
        raise ValueError(
            f"frozen suite seed {manifest.suite_seed} differs from requested "
            f"{options.suite_seed}; create a new suite version instead of editing it"
        )

    orchestrator = Orchestrator(paths)
    orchestrator.advance_to(
        ProjectState.PROJECT_INGESTED,
        reason=f"frozen suite reference loaded: {manifest_path.as_posix()}",
    )
    orchestrator.advance_to(
        ProjectState.PROBLEM_UNDERSTOOD,
        reason=(
            f"suite identity, methods, and oracle confirmed "
            f"({manifest.suite_id} v{manifest.suite_version})"
        ),
    )
    orchestrator.advance_to(
        ProjectState.PROTOTYPING,
        reason=f"benchmark suite execution planned for {len(manifest.instances)} instances",
    )
    orchestrator.advance_to(
        ProjectState.PROTOTYPED,
        reason=f"preflight: manifest {manifest.manifest_hash[:12]}... frozen and verified",
    )

    result = run_knapsack_suite(
        paths, options.suite_dir, time_limit_seconds=options.time_limit_seconds
    )
    completion_issues = _completion_issues(paths, manifest, result)
    if completion_issues:
        blocker = _blocked_artifact(
            paths,
            result.benchmark_run_id,
            f"Benchmark suite not complete: {result.benchmark_run_id}",
            [f"- {issue}" for issue in completion_issues],
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"manifest-driven completion gate failed ({len(completion_issues)} issues)",
        )
        _suite_run_record(
            paths,
            manifest,
            result,
            state=config.state,
            completion_issues=completion_issues,
        )
        raise BenchmarkPipelineBlocked(
            result.benchmark_run_id,
            config.state,
            f"completion gate failed; see {blocker}",
        )

    orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED,
        reason=(
            f"all {len(manifest.instances)} manifest instances produced 3 experiments "
            "and 1 tournament"
        ),
    )
    orchestrator.advance_to(ProjectState.VALIDATING, reason="benchmark suite validation started")
    reports = validate_benchmark_suite(
        paths, result.benchmark_run_id, options.suite_dir
    )
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked_artifact(
            paths,
            result.benchmark_run_id,
            f"Benchmark suite validation failed: {result.benchmark_run_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="benchmark suite validation failed",
        )
        _suite_run_record(
            paths,
            manifest,
            result,
            state=config.state,
            validation=validation,
        )
        raise BenchmarkPipelineBlocked(
            result.benchmark_run_id,
            config.state,
            f"validation failed; see {blocker}",
        )

    orchestrator.advance_to(
        ProjectState.VERIFIED, reason="benchmark integrity, oracle, and aggregate checks passed"
    )
    artifact_path = (
        paths.experiments_dir / "benchmarks" / result.benchmark_run_id / "benchmark_result.json"
    )
    freeze_benchmark_suite(paths, result, artifact_path)
    orchestrator.advance_to(ProjectState.EVIDENCE_FROZEN, reason="benchmark evidence frozen")
    registered = _register_claims(paths, result, manifest)
    claims = _claims_for_run(paths, result.benchmark_run_id) or registered
    orchestrator.advance_to(ProjectState.PAPER_DRAFTED, reason="benchmark paper rendered")
    orchestrator.advance_to(ProjectState.RED_TEAM_REVIEW, reason="benchmark red team started")
    return _review_and_release(
        paths, orchestrator, manifest, result, reviewer, validation, claims
    )


def _latest_benchmark_result(
    paths: ProjectPaths,
) -> tuple[BenchmarkSuiteResult, Path]:
    artifacts = sorted(
        (paths.experiments_dir / "benchmarks").glob("*/benchmark_result.json"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not artifacts:
        raise ValueError("no benchmark run artifact found for resume")
    artifact_path = artifacts[-1]
    return load_benchmark_result(artifact_path), artifact_path.parent


def _claims_for_run(paths: ProjectPaths, run_id: str) -> list[str]:
    return [claim.claim_id for claim in load_claims(paths) if run_id in claim.claim_id]


def _resume_review(
    paths: ProjectPaths,
    options: BenchmarkPipelineOptions,
    reviewer: Reviewer,
) -> BenchmarkRunSummary:
    config = load_project_config(paths)
    orchestrator = Orchestrator(paths)
    if config.state is ProjectState.REQUIRES_HUMAN_REVIEW:
        orchestrator.transition(
            ProjectState.RED_TEAM_REVIEW, reason="resume benchmark red team review"
        )

    result, artifact_dir = _latest_benchmark_result(paths)
    manifest = load_manifest(artifact_dir)
    completion_issues = _completion_issues(paths, manifest, result)
    if completion_issues:
        blocker = _blocked_artifact(
            paths,
            result.benchmark_run_id,
            f"Benchmark resume failed completion: {result.benchmark_run_id}",
            [f"- {issue}" for issue in completion_issues],
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="benchmark resume completion gate failed",
        )
        _suite_run_record(
            paths,
            manifest,
            result,
            state=config.state,
            completion_issues=completion_issues,
        )
        raise BenchmarkPipelineBlocked(
            result.benchmark_run_id,
            config.state,
            f"completion gate failed; see {blocker}",
        )

    reports = validate_benchmark_suite(paths, result.benchmark_run_id, options.suite_dir)
    validation = {report.validator: report.status.value for report in reports}
    blocking = [report for report in reports if report.blocking]
    if blocking:
        lines = [
            f"- {report.validator}/{issue.code}: {issue.message}"
            for report in blocking
            for issue in report.issues
        ]
        blocker = _blocked_artifact(
            paths,
            result.benchmark_run_id,
            f"Benchmark resume validation failed: {result.benchmark_run_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="benchmark resume validation failed",
        )
        _suite_run_record(
            paths, manifest, result, state=config.state, validation=validation
        )
        raise BenchmarkPipelineBlocked(
            result.benchmark_run_id,
            config.state,
            f"validation failed; see {blocker}",
        )

    claim_ids = _claims_for_run(paths, result.benchmark_run_id)
    if not claim_ids:
        claim_ids = _register_claims(paths, result, manifest)
    return _review_and_release(
        paths, orchestrator, manifest, result, reviewer, validation, claim_ids
    )


def _review_and_release(
    paths: ProjectPaths,
    orchestrator: Orchestrator,
    manifest: BenchmarkManifest,
    result: BenchmarkSuiteResult,
    reviewer: Reviewer,
    validation: dict[str, str],
    claim_ids: list[str],
) -> BenchmarkRunSummary:
    ast = build_benchmark_paper(paths, result.benchmark_run_id)
    report_path = render_paper(paths, ast)
    review = reviewer(ast, result.benchmark_run_id)
    review_path = save_red_team_review(paths, review)

    if not release_allowed(review):
        lines = _review_blocked_lines(review)
        blocker = _blocked_artifact(
            paths,
            result.benchmark_run_id,
            f"Benchmark release blocked: {result.benchmark_run_id}",
            lines,
        )
        config = orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason=f"benchmark red team judged {review.judge.overall.value}; PASS required",
        )
        _suite_run_record(
            paths,
            manifest,
            result,
            state=config.state,
            validation=validation,
            red_team=review.judge.overall.value,
        )
        raise BenchmarkPipelineBlocked(
            result.benchmark_run_id,
            config.state,
            f"red team {review.judge.overall.value}; see {blocker}",
        )

    orchestrator.advance_to(
        ProjectState.READY_FOR_RELEASE, reason="benchmark red team passed"
    )
    release = {
        "benchmark_run_id": result.benchmark_run_id,
        "project_id": paths.root.name,
        "suite_id": result.suite_id,
        "suite_version": manifest.suite_version,
        "manifest_hash": result.manifest_hash,
        "suite_seed": result.suite_seed,
        "n_instances": result.n_instances,
        "expected_experiment_count": 3 * len(manifest.instances),
        "expected_tournament_count": len(manifest.instances),
        "actual_experiment_count": sum(len(ids) for ids in result.experiment_ids.values()),
        "actual_tournament_count": len(result.tournament_ids),
        "methods_code_hash": result.methods_code_hash,
        "generator_code_hash": result.generator_code_hash,
        "runtime": result.runtime,
        "candidate_verified_optimum_count": result.candidate_verified_optimum_count,
        "candidate_win_count": result.candidate_win_count,
        "candidate_tie_count": result.candidate_tie_count,
        "candidate_loss_count": result.candidate_loss_count,
        "greedy_optimum_hits": result.greedy.optimum_hits,
        "greedy_max_relative_regret": result.greedy.max_relative_regret,
        "greedy_worst_instance": result.greedy.worst_instance_id,
        "enhanced_greedy_optimum_hits": result.enhanced_greedy.optimum_hits,
        "validation": validation,
        "red_team": review.judge.overall.value,
        "claims": claim_ids,
        "paper": report_path.relative_to(paths.root).as_posix(),
        "review": review_path.relative_to(paths.root).as_posix(),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    run = result.benchmark_run_id
    greedy_regret_text = (
        f"{result.greedy.max_relative_regret:.4f} on `{result.greedy.worst_instance_id}`"
        if result.greedy.max_relative_regret is not None
        and result.greedy.worst_instance_id is not None
        else "unavailable"
    )
    release_json = paths.releases_dir / f"benchmark_{run}.json"
    release_md = paths.releases_dir / f"benchmark_{run}.md"
    atomic_write_text(release_json, json.dumps(release, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Benchmark release {run}\n\n"
        f"- Project: `{paths.root.name}`\n"
        f"- Suite: `{result.suite_id}` v{manifest.suite_version}\n"
        f"- Manifest: `{result.manifest_hash}`\n"
        f"- Generator code hash: `{result.generator_code_hash}`\n"
        f"- Methods code hash: `{result.methods_code_hash}`\n"
        f"- Instances: {result.n_instances} (expected experiments "
        f"{3 * len(manifest.instances)}, tournaments {len(manifest.instances)})\n"
        f"- CBC objective matches: {result.candidate_verified_optimum_count}/"
        f"{result.n_instances}\n"
        f"- CBC vs greedy: wins {result.candidate_win_count}, ties "
        f"{result.candidate_tie_count}, losses {result.candidate_loss_count}\n"
        f"- Greedy hits: {result.greedy.optimum_hits}/{result.n_instances}; max relative "
        f"regret {greedy_regret_text}\n"
        f"- Validation: {validation}\n"
        f"- Red team: `{review.judge.overall.value}`\n"
        f"- Proposed claims: {', '.join(claim_ids) or 'none'}\n"
        f"- Paper: `{report_path.relative_to(paths.root).as_posix()}`\n",
    )
    config = orchestrator.advance_to(
        ProjectState.RELEASED, reason="benchmark release written"
    )
    _suite_run_record(
        paths,
        manifest,
        result,
        state=config.state,
        validation=validation,
        red_team=review.judge.overall.value,
        release_path=release_md,
    )
    return BenchmarkRunSummary(
        benchmark_run_id=run,
        suite_id=result.suite_id,
        manifest_hash=result.manifest_hash,
        n_instances=result.n_instances,
        expected_experiment_count=3 * len(manifest.instances),
        expected_tournament_count=len(manifest.instances),
        validation=validation,
        claim_ids=claim_ids,
        red_team_overall=review.judge.overall.value,
        paper_path=report_path,
        review_path=review_path,
        release_path=release_md,
        state=config.state,
    )
