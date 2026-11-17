from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mmos.historical_benchmark.capabilities import (
    assess_capability_fit,
    capability_registry,
    descriptor_hash,
)
from mmos.historical_benchmark.harness import (
    BlindnessViolation,
    HistoricalBenchmarkError,
    HistoricalBenchmarkHarness,
    create_code_freeze,
)
from mmos.historical_benchmark.preflight import FreshnessError, freshness_preflight
from mmos.historical_benchmark.schemas import (
    CapabilityVerdict,
    HistoricalBenchmarkRun,
    HistoricalBenchmarkState,
    LeakageEvent,
    ProblemRequirement,
    ProtocolValidity,
    RequirementStatus,
)
from mmos.schemas.problem import ProblemSpec, SubProblem
from mmos.schemas.routing import RoutingDecision, RoutingStatus, SubproblemRoute
from mmos.schemas.states import TaskFamily

NOW = datetime(2026, 11, 17, 12, tzinfo=UTC)


def make_archive(root: Path) -> Path:
    archive = root / "official.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("A题/problem.txt", "A problem statement\n")
        handle.writestr("B题/problem.txt", "B problem statement\n")
        handle.writestr("C题/problem.txt", "C problem statement\n")
        handle.writestr("A题/README.txt", "A readme\n")
        handle.writestr("B题/附件/data.xlsx", b"not really an xlsx")
    return archive


def harness(tmp_path: Path) -> HistoricalBenchmarkHarness:
    h = HistoricalBenchmarkHarness(tmp_path, "bench-test", competition="CUMCM", year=2026)
    h.start(now=NOW)
    h.seal_archive(
        make_archive(tmp_path),
        official_source="https://www.mcm.edu.cn/",
        downloaded_at=NOW,
        benchmark_source_id="cumcm-2026",
    )
    return h


def freeze(h: HistoricalBenchmarkHarness, tmp_path: Path) -> None:
    h.write_code_freeze(
        create_code_freeze(
            tmp_path,
            h.benchmark_run_id,
            test_status="passed",
            ruff_status="passed",
            pyright_status="passed",
            now=NOW,
            git_commit="testcommit",
        )
    )


def test_source_manifest_selection_before_reveal(tmp_path: Path) -> None:
    h = harness(tmp_path)
    source = h.load_source_manifest()
    assert source.archive_sha256
    assert source.archive_size > 0
    assert source.official_source.startswith("https://")
    selection = h.commit_selection(now=NOW)
    assert set(selection.labels) == {"A", "B", "C"}
    assert selection.primary_label in selection.labels
    assert len(selection.reserve_labels) == 2
    assert h.load_run().selection_manifest_hash == selection.selection_hash
    with pytest.raises(BlindnessViolation):
        h.reveal(attempt_id="a1", now=NOW)
    freeze(h, tmp_path)
    reveal = h.reveal(attempt_id="a1", now=NOW)
    assert reveal.problem_label == selection.primary_label
    assert h.load_run().state is HistoricalBenchmarkState.PROBLEM_REVEALED
    with pytest.raises(BlindnessViolation):
        h.reveal(attempt_id="a2", now=NOW)


def test_selection_order_is_deterministic() -> None:
    archive_hash = "a" * 64
    labels = ["A", "B", "C"]
    import hashlib

    ordered = sorted(
        labels,
        key=lambda label: (
            hashlib.sha256(f"{archive_hash}:{label}".encode()).hexdigest(),
            label,
        ),
    )
    assert ordered == sorted(
        labels,
        key=lambda label: (
            hashlib.sha256(f"{archive_hash}:{label}".encode()).hexdigest(),
            label,
        ),
    )


def test_reserve_requires_infrastructure_invalidation(tmp_path: Path) -> None:
    h = harness(tmp_path)
    selection = h.commit_selection(now=NOW)
    freeze(h, tmp_path)
    reserve = selection.reserve_labels[0]
    with pytest.raises(BlindnessViolation):
        h.reveal(label=reserve, attempt_id="reserve", now=NOW)
    h.invalidate("harness bug", now=NOW)
    assert h.load_run().protocol_validity is ProtocolValidity.INVALIDATED_INFRASTRUCTURE
    reveal = h.reveal(label=reserve, attempt_id="reserve", now=NOW)
    assert reveal.problem_label == reserve


def test_archive_tamper_and_package_hash(tmp_path: Path) -> None:
    h = harness(tmp_path)
    h.commit_selection(now=NOW)
    archive = h.archive_path()
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(HistoricalBenchmarkError):
        h.extract_package()


def test_leakage_and_human_intervention_fail_blindness_audit(tmp_path: Path) -> None:
    h = harness(tmp_path)
    selection = h.commit_selection(now=NOW)
    freeze(h, tmp_path)
    h.reveal(attempt_id="a1", now=NOW)
    h.log_leakage(LeakageEvent.WEB_ACCESS_ATTEMPT, "attempted search", now=NOW)
    h.log_leakage(
        LeakageEvent.HUMAN_INTERVENTION,
        "human suggested a model",
        now=NOW,
    )
    audit = h.blindness_audit(now=NOW)
    assert audit.passed is False
    assert audit.prohibited_access_count >= 1
    assert audit.human_scientific_intervention_count == 1
    assert selection.primary_label == h.load_reveal().problem_label


def test_package_extraction_and_project_ingestion(tmp_path: Path) -> None:
    h = harness(tmp_path)
    h.commit_selection(now=NOW)
    source = h.extract_package()
    assert len(source.extracted_files) == 5
    assert all(entry.sha256 for entry in source.extracted_files)
    freeze(h, tmp_path)
    reveal = h.reveal(attempt_id="a1", now=NOW)
    files = h.selected_files()
    assert files and all(reveal.problem_label in path.as_posix() for path in files)
    text = h.problem_text()
    assert f"{reveal.problem_label} problem statement" in text
    paths = h.create_project()
    assert paths.problem_spec_file.parent.exists()
    assert (paths.root / "problem").exists()


def test_pdf_text_backend_extracts_unicode(tmp_path: Path) -> None:
    import pymupdf

    pdf = tmp_path / "statement.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "A problem statement")
    document.save(pdf)
    document.close()
    h = harness(tmp_path)
    derived = h._derive_text(pdf)
    assert "A problem statement" in derived.read_text(encoding="utf-8")


def test_requirement_mapping_from_problem_spec(tmp_path: Path) -> None:
    h = harness(tmp_path)
    spec = ProblemSpec(
        problem_id="p",
        title="Test",
        overall_goal="Solve A.",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="Do the first task.",
                goal="Produce output.",
                outputs=["output"],
                decision_variables=[],
            )
        ],
        global_assumptions=[],
    )
    requirements = h.requirements_from_spec(spec)
    assert len(requirements) == 1
    assert requirements[0].status is RequirementStatus.UNRESOLVED
    h.write_requirements(requirements)
    assert h.load_requirements()[0].requirement_id == "q1"


def test_code_freeze_and_result_hash(tmp_path: Path) -> None:
    h = harness(tmp_path)
    h.commit_selection(now=NOW)
    h.extract_package()
    freeze(h, tmp_path)
    h.reveal(attempt_id="a1", now=NOW)
    h.write_requirements(
        h.requirements_from_spec(
            ProblemSpec(
                problem_id="p",
                title="Test",
                overall_goal="Solve A.",
                subproblems=[
                    SubProblem(
                        subproblem_id="q1",
                        statement="Do it.",
                        goal="Output.",
                        outputs=["output"],
                        decision_variables=[],
                    )
                ],
                global_assumptions=[],
            )
        )
    )
    (h.root / "router_result.json").write_text(
        json.dumps({"status": "unsupported"}), encoding="utf-8"
    )
    result = h.finalize_result(
        scientific_verdict="blocked",
        unresolved_issues=["q1 unresolved"],
    )
    assert result.result_hash
    assert result.protocol_validity is ProtocolValidity.VALID
    assert result.blindness_audit is not None
    assert result.blindness_audit.passed
    assert result.requirement_coverage[RequirementStatus.UNRESOLVED.value] == 1
    assert result.external_quality_assessment == "not_yet_assessed"
    assert h.verify_result()["ok"] is True
    result_file = h.root / "benchmark_result.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    payload["scientific_verdict"] = "tampered"
    result_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert h.verify_result()["ok"] is False


def test_code_freeze_detects_code_changes(tmp_path: Path) -> None:
    h = harness(tmp_path)
    freeze(h, tmp_path)
    assert h.verify_code_freeze()["ok"] is True
    changed = tmp_path / "src" / "mmos" / "changed.py"
    changed.parent.mkdir(parents=True)
    changed.write_text("changed = True\n", encoding="utf-8")
    assert h.verify_code_freeze()["ok"] is False


def test_phase22_phase23_artifacts_remain_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    project = root / "projects" / "mixed-forecast-plan-sim-20261116d-20261116-828a5b"
    if not project.exists():
        pytest.skip("canonical Phase 22 project is not available")
    release = project / "releases" / "mixed_mixed-a5aad5c1-73b2e9.json"
    augmentation = (
        project / "paper" / "competition" / "augmentation" / "aug-d0e04bfa-a1a9" / "evidence.json"
    )
    controller = project / "competition" / "runs" / "ctrl-b0e30df9-phase231c" / "controller.json"
    if not release.exists() or not augmentation.exists() or not controller.exists():
        pytest.skip("sealed Phase 22/23 artifacts are not present")
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest(release) == ("5165180adc3b8f56b39201566b0163a87fb27f90ab0024fddc7c493a578d0282")
    assert digest(augmentation) == (
        "a1060145dbd57029afbfe9a44165c6f1d8bfe6a07188c83e1d9e533ef85d16a1"
    )
    run = json.loads(controller.read_text(encoding="utf-8"))
    assert run["competition_run_id"] == "ctrl-b0e30df9-phase231c"
    assert run["scientific_gate_failed"] is True
    assert run["unresolved_blockers"] == ["crit-a1", "crit-a2"]


def _decision(*families: TaskFamily, requirement_ids: list[str]) -> RoutingDecision:
    from mmos.schemas.routing import ExecutionStatus

    return RoutingDecision(
        routing_id="route-test",
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        status=RoutingStatus.SINGLE_TASK if len(families) == 1 else RoutingStatus.MULTI_TASK,
        primary_task=families[0] if len(families) == 1 else None,
        required_task_families=list(families),
        subproblem_routes=[
            SubproblemRoute(
                subproblem_id=requirement_id,
                task_families=list(families),
            )
            for requirement_id in requirement_ids
        ],
        capabilities=[
            {
                "task_family": family,
                "current_supported_scope": "test",
                "execution_readiness": "PARTIAL",
            }
            for family in families
        ],
        execution_status=ExecutionStatus.READY_FOR_SINGLE_TASK
        if len(families) == 1
        else ExecutionStatus.BLOCKED_PENDING_MIXED_DAG,
        provider="test",
        model="test",
        created_at=NOW,
    )


def _requirement(requirement_id: str, text: str) -> ProblemRequirement:
    import hashlib

    return ProblemRequirement(
        requirement_id=requirement_id,
        source_location=requirement_id,
        question_text=text,
        question_hash=hashlib.sha256(text.encode()).hexdigest(),
        required_output="output",
    )


def test_capability_fit_rejects_taskfamily_without_slice() -> None:
    decision = _decision(TaskFamily.OPTIMIZATION, requirement_ids=["q1"])
    fit = assess_capability_fit(
        decision,
        [_requirement("q1", "plan a robot path through a geometric polygon field")],
    )
    assert fit.verdict is CapabilityVerdict.UNSUPPORTED
    assert fit.unsupported == ["q1"]
    assert fit.blocking_gaps


def test_capability_fit_partial_and_all_supported() -> None:
    decision = _decision(TaskFamily.OPTIMIZATION, requirement_ids=["q1", "q2"])
    partial = assess_capability_fit(
        decision,
        [
            _requirement("q1", "select binary items under one capacity constraint"),
            _requirement("q2", "plan a robot path through a polygon"),
        ],
    )
    assert partial.verdict is CapabilityVerdict.PARTIALLY_SUPPORTED
    assert partial.supported == ["q1"]
    assert partial.unsupported == ["q2"]
    supported = assess_capability_fit(
        decision,
        [_requirement("q1", "select binary items under one capacity constraint")],
    )
    assert supported.verdict is CapabilityVerdict.SUPPORTED


def test_capability_descriptor_hash_is_sensitive() -> None:
    descriptor = capability_registry()[TaskFamily.OPTIMIZATION]
    changed = descriptor.model_copy(update={"known_limits": ["changed"]})
    assert descriptor.descriptor_hash
    assert descriptor.descriptor_hash != descriptor_hash(changed)


def test_freshness_preflight_burns_revealed_labels_and_blocks_all_burned(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "benchmarks" / "historical" / "old"
    run_dir.mkdir(parents=True)
    run = HistoricalBenchmarkRun(
        benchmark_run_id="old",
        competition="CUMCM",
        year=2025,
        problem_label="A",
        state=HistoricalBenchmarkState.PROBLEM_REVEALED,
        created_at=NOW,
        updated_at=NOW,
    )
    run_dir.joinpath("benchmark_run.json").write_text(
        run.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    registry = freshness_preflight(
        tmp_path, competition="CUMCM", year=2025, labels=["A", "B", "C"], now=NOW
    )
    assert registry.burned_labels == ["A"]
    for label in ["B", "C"]:
        d = tmp_path / "benchmarks" / "historical" / f"old-{label}"
        d.mkdir(parents=True)
        r = run.model_copy(update={"benchmark_run_id": f"old-{label}", "problem_label": label})
        (d / "benchmark_run.json").write_text(r.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with pytest.raises(FreshnessError):
        freshness_preflight(
            tmp_path, competition="CUMCM", year=2025, labels=["A", "B", "C"], now=NOW
        )
