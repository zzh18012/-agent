import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmos.competition import augmentation as augmentation_module
from mmos.competition.augmentation import (
    AdapterOutcome,
    CompetitionAugmentationBlocked,
    load_registered_evidence,
    resolve_requests,
    run_augmentation,
)
from mmos.competition_pipeline import run_competition_paper
from mmos.schemas.competition import (
    CompetitionExperimentRequest,
    ExecutionStatus,
    Priority,
    ResolutionClass,
)
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.review import Verdict
from tests.test_competition import (
    EVIDENCE_ID,
    agents,
    released_project,
    reviewer,
)


def source_run(paths) -> str:
    summary = run_competition_paper(paths, agents(), reviewer(Verdict.PASS))
    assert summary.run.status == "released"
    return summary.run.run_id


def fake_adapter(paths, context, request, source_run, source_dir, aug_id):
    artifact = paths.root / "paper" / "competition" / "fake_result.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"result": "frozen"}\n', encoding="utf-8")
    record = EvidenceRecord(
        evidence_id=f"ev-aug-{aug_id}-{request.request_id}-fake",
        key="competition.fake.metric",
        raw_value=1.0,
        display_value="1",
        unit=None,
        experiment_id="exp-fake",
        artifact_path=artifact.relative_to(paths.root).as_posix(),
        artifact_hash=__import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
        source_field="fake",
        validation_reports=["optimization"],
        verified_at=datetime.now(UTC),
        provenance={"kind": "competition_experiment"},
    )
    return AdapterOutcome(
        ExecutionStatus.VALIDATED,
        ["exp-fake"],
        {"optimization": "passed"},
        [record],
    )


def test_resolver_classifies_source_manifest() -> None:
    from mmos.competition.requests import build_manifest
    from tests.test_competition import full_plan

    manifest = build_manifest(full_plan().capability_needs, {EVIDENCE_ID})
    resolutions = {item.request_id: item for item in resolve_requests(manifest)}
    assert (
        resolutions["request-1"].classification
        is ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
    )
    assert (
        resolutions["request-2"].classification
        is ResolutionClass.REQUIRES_NEW_CAPABILITY
    )


def test_executed_flag_cannot_be_set_by_llm() -> None:
    with pytest.raises(ValidationError):
        CompetitionExperimentRequest(
            request_id="request-x",
            question="Q",
            rationale="R",
            capability="optimization",
            proposed_experiment="P",
            expected_evidence="E",
            priority=Priority.HIGH,
            supporting_evidence_ids=[EVIDENCE_ID],
            supported=True,
            executed=True,
        )


def test_augmentation_executes_only_eligible_and_regenerates_paper(
    tmp_path: Path, monkeypatch
) -> None:
    paths = released_project(tmp_path)
    run_id = source_run(paths)
    release_path = paths.releases_dir / "release_test.json"
    release_hash = __import__("hashlib").sha256(release_path.read_bytes()).hexdigest()
    verified_path = paths.verified_numbers_file
    verified_hash = __import__("hashlib").sha256(
        verified_path.read_bytes()
    ).hexdigest()
    monkeypatch.setitem(
        augmentation_module.DEFAULT_ADAPTERS,
        "optimization_sensitivity",
        fake_adapter,
    )
    summary = run_augmentation(paths, agents(), reviewer(Verdict.PASS), run_id)
    assert summary.run.status == "completed"
    assert summary.run.source_release_hash_before == release_hash
    assert summary.run.source_release_hash_after == release_hash
    assert summary.run.source_verified_numbers_hash_before == verified_hash
    assert summary.run.source_verified_numbers_hash_after == verified_hash
    assert summary.run.new_competition_run_id not in {None, run_id}
    assert summary.run.new_release_path is not None
    assert (paths.root / summary.run.new_release_path).exists()
    assert summary.run.new_evidence_ids
    assert (paths.root / summary.run.evidence_store_path).exists()

    executions = {item.request_id: item for item in summary.run.executions}
    executable = [
        item
        for item in executions.values()
        if item.status is ExecutionStatus.EVIDENCE_REGISTERED
    ]
    blocked = [
        item for item in executions.values() if item.status is ExecutionStatus.BLOCKED
    ]
    assert len(executable) == 1
    assert len(blocked) == 1
    assert executable[0].experiment_ids == ["exp-fake"]
    assert executable[0].executed is True
    assert executable[0].validated is True
    assert executable[0].evidence_registered is True
    assert blocked[0].executed is False


def test_adapter_failure_produces_no_evidence(tmp_path: Path, monkeypatch) -> None:
    paths = released_project(tmp_path)
    run_id = source_run(paths)

    def failed_adapter(paths, context, request, source_run, source_dir, aug_id):
        return AdapterOutcome(
            ExecutionStatus.FAILED,
            ["exp-failed"],
            {},
            [],
            "runner failed",
        )

    monkeypatch.setitem(
        augmentation_module.DEFAULT_ADAPTERS,
        "optimization_sensitivity",
        failed_adapter,
    )
    summary = run_augmentation(paths, agents(), reviewer(Verdict.PASS), run_id)
    assert summary.run.status == "blocked"
    assert summary.run.new_evidence_ids == []
    assert summary.run.new_competition_run_id is None
    failed = next(
        item
        for item in summary.run.executions
        if item.status is ExecutionStatus.FAILED
    )
    assert failed.executed is True
    assert failed.validated is False
    assert failed.evidence_registered is False


def test_source_release_mutation_blocks(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    run_id = source_run(paths)
    release_path = paths.releases_dir / "release_test.json"
    release_path.write_text(
        release_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(CompetitionAugmentationBlocked, match="release hash changed"):
        run_augmentation(paths, agents(), reviewer(Verdict.PASS), run_id)


def test_request_manifest_mutation_blocks(tmp_path: Path) -> None:
    paths = released_project(tmp_path)
    run_id = source_run(paths)
    manifest_path = paths.root / "paper" / "competition" / "experiment_requests.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["requests"][0]["question"] = "Mutated question"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(CompetitionAugmentationBlocked, match="manifest hash changed"):
        run_augmentation(paths, agents(), reviewer(Verdict.PASS), run_id)


def test_registered_evidence_artifact_mutation_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    paths = released_project(tmp_path)
    run_id = source_run(paths)
    monkeypatch.setitem(
        augmentation_module.DEFAULT_ADAPTERS,
        "optimization_sensitivity",
        fake_adapter,
    )
    summary = run_augmentation(paths, agents(), reviewer(Verdict.PASS), run_id)
    evidence, _ = load_registered_evidence(
        paths, summary.run.augmentation_id
    )
    artifact = paths.root / evidence[0].artifact_path
    artifact.write_text('{"result": "mutated"}\n', encoding="utf-8")
    with pytest.raises(Exception, match="artifact hash changed"):
        load_registered_evidence(paths, summary.run.augmentation_id)
