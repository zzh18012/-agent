import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mmos.cli.main import app
from mmos.core.project import sha256_file
from mmos.evidence import (
    EvidenceError,
    add_claim,
    freeze_experiment,
    load_claims,
    load_evidence_index,
)
from mmos.experiments.runner import run_experiment
from mmos.orchestration import Orchestrator
from mmos.schemas.states import ProjectState
from mmos.validation import validate_experiment
from tests.helpers import make_experiment_project, regression_frame

runner = CliRunner()


def make_validated_experiment(tmp_path: Path):
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae"],
    )
    run_experiment(paths, spec.experiment_id)
    validate_experiment(paths, spec.experiment_id)
    return paths, spec


def test_freeze_requires_validation(tmp_path: Path) -> None:
    paths, spec = make_experiment_project(
        tmp_path,
        regression_frame(),
        models=["sklearn.linear_regression"],
        metrics=["rmse"],
    )
    run_experiment(paths, spec.experiment_id)

    with pytest.raises(EvidenceError, match="not been validated"):
        freeze_experiment(paths, spec.experiment_id)
    assert not paths.verified_numbers_file.exists()


def test_freeze_writes_hashed_evidence_and_claims(tmp_path: Path) -> None:
    paths, spec = make_validated_experiment(tmp_path)

    records = freeze_experiment(paths, spec.experiment_id)

    assert len(records) == 2 * 2
    metrics_hash = sha256_file(
        paths.experiment_dir(spec.experiment_id) / "artifacts" / "metrics.json"
    )
    assert all(record.artifact_hash == metrics_hash for record in records)

    index = load_evidence_index(paths)
    assert len(index.records) == len(records)
    assert paths.verified_metrics_dir.joinpath(f"{spec.experiment_id}.json").exists()

    provenance = json.loads((paths.verified_dir / "provenance.json").read_text(encoding="utf-8"))
    entry = provenance["experiments"][spec.experiment_id]
    assert entry["metrics_sha256"] == metrics_hash
    assert entry["dataset_hash"] == spec.dataset_hash

    freeze_experiment(paths, spec.experiment_id)
    assert len(load_evidence_index(paths).records) == len(records)

    with pytest.raises(EvidenceError, match="unknown evidence"):
        add_claim(paths, statement="Does not exist", evidence_ids=["ev-missing"])

    evidence_id = records[0].evidence_id
    claim = add_claim(
        paths,
        statement="The linear model beats the dummy baseline",
        evidence_ids=[evidence_id],
        accept=True,
        reviewer="test",
    )
    assert claim.status == "supported"
    assert load_claims(paths) == [claim]

    with pytest.raises(EvidenceError, match="already exists"):
        add_claim(paths, statement="duplicate", evidence_ids=[evidence_id], claim_id=claim.claim_id)


def test_freeze_rejects_metrics_changed_after_validation(tmp_path: Path) -> None:
    paths, spec = make_validated_experiment(tmp_path)
    metrics_file = paths.experiment_dir(spec.experiment_id) / "artifacts" / "metrics.json"
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    payload["models"]["sklearn.linear_regression"]["rmse"] += 1.0
    metrics_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError, match="changed after validation"):
        freeze_experiment(paths, spec.experiment_id)


def test_freeze_and_claim_cli(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    paths, spec = make_validated_experiment(tmp_path)
    orchestrator = Orchestrator(paths)
    orchestrator.transition(ProjectState.PROJECT_INGESTED, reason="test setup")
    orchestrator.transition(ProjectState.PROBLEM_UNDERSTOOD, reason="test setup")
    orchestrator.transition(ProjectState.DATA_VALIDATED, reason="test setup")

    freeze_result = runner.invoke(
        app,
        [
            "freeze-evidence",
            paths.root.name,
            spec.experiment_id,
            "--projects-dir",
            str(projects_dir),
        ],
    )
    assert freeze_result.exit_code == 0, freeze_result.output
    assert "Verified evidence" in freeze_result.output

    evidence_id = next(iter(load_evidence_index(paths).records))
    claim_result = runner.invoke(
        app,
        [
            "add-claim",
            paths.root.name,
            "--statement",
            "Linear regression outperforms the baseline",
            "--evidence",
            evidence_id,
            "--accept",
            "--projects-dir",
            str(projects_dir),
        ],
    )
    assert claim_result.exit_code == 0, claim_result.output
    assert "supported" in claim_result.output
    assert len(load_claims(paths)) == 1
