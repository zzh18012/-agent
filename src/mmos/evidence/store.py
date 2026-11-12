import json
import secrets
from datetime import UTC, datetime

import yaml

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    experiment_spec_file,
    load_experiment_spec,
    sha256_file,
)
from mmos.schemas.common import MMOSModel
from mmos.schemas.evidence import ClaimRecord, EvidenceIndex, EvidenceRecord
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.validation import ValidationReport, ValidationStatus

REQUIRED_VALIDATORS = ("leakage", "numerical", "statistical", "reproducibility")


class EvidenceError(RuntimeError):
    """Raised when evidence would be created or cited without sufficient support."""


class _ClaimFile(MMOSModel):
    claims: list[ClaimRecord]


def _verified_dir(paths: ProjectPaths):
    paths.verified_dir.mkdir(parents=True, exist_ok=True)
    return paths.verified_dir


def _load_passing_reports(
    paths: ProjectPaths, experiment_id: str
) -> tuple[list[ValidationReport], dict[str, str]]:
    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    summary_path = validation_dir / "summary.json"
    if not summary_path.exists():
        raise EvidenceError(
            f"experiment {experiment_id!r} has not been validated; run: mmos validate"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != experiment_id:
        raise EvidenceError("validation summary belongs to a different experiment")
    if summary.get("blocking"):
        raise EvidenceError("validation summary contains blocking issues")

    reports: list[ValidationReport] = []
    report_hashes: dict[str, str] = {}
    for name in REQUIRED_VALIDATORS:
        report_path = validation_dir / f"{name}.json"
        if not report_path.exists():
            raise EvidenceError(f"missing required validation report: {name}")
        report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if report.blocking or report.status is not ValidationStatus.PASSED:
            raise EvidenceError(f"validation report {name!r} did not pass")
        reports.append(report)
        report_hashes[name] = sha256_file(report_path)

    return reports, report_hashes


def load_evidence_index(paths: ProjectPaths) -> EvidenceIndex:
    target = paths.verified_numbers_file
    if not target.exists():
        return EvidenceIndex()
    return EvidenceIndex.model_validate_json(target.read_text(encoding="utf-8"))


def load_claims(paths: ProjectPaths) -> list[ClaimRecord]:
    target = paths.verified_claims_file
    if not target.exists():
        return []
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return _ClaimFile.model_validate(data).claims


def freeze_experiment(paths: ProjectPaths, experiment_id: str) -> list[EvidenceRecord]:
    reports, report_hashes = _load_passing_reports(paths, experiment_id)

    experiment_dir = paths.experiment_dir(experiment_id)
    metrics_path = experiment_dir / "artifacts" / "metrics.json"
    predictions_path = experiment_dir / "artifacts" / "predictions.parquet"
    spec_path = experiment_spec_file(paths, experiment_id)
    for required in (metrics_path, predictions_path, spec_path):
        if not required.exists():
            raise EvidenceError(f"missing experiment artifact: {required.name}")

    validation_summary = json.loads(
        (experiment_dir / "validation" / "summary.json").read_text(encoding="utf-8")
    )
    expected_hash = validation_summary.get("artifacts", {}).get("metrics_sha256")
    metrics_hash = sha256_file(metrics_path)
    if expected_hash != metrics_hash:
        raise EvidenceError(
            "metrics.json changed after validation; re-run validation before freezing"
        )

    spec = load_experiment_spec(paths, experiment_id)
    result = ExperimentResult.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    if result.dataset_hash != spec.dataset_hash:
        raise EvidenceError("result and spec disagree on the dataset hash")

    mlflow_run_id = ""
    mlflow_record = experiment_dir / "mlflow_run.json"
    if mlflow_record.exists():
        mlflow_run_id = json.loads(mlflow_record.read_text(encoding="utf-8")).get("run_id", "")

    now = datetime.now(UTC).replace(microsecond=0)
    relative_metrics = metrics_path.relative_to(paths.root).as_posix()
    provenance = {
        "dataset_hash": spec.dataset_hash,
        "metrics_path": relative_metrics,
        "metrics_sha256": metrics_hash,
        "experiment_spec_path": spec_path.relative_to(paths.root).as_posix(),
        "experiment_spec_sha256": sha256_file(spec_path),
        "validation_summary": str(
            (experiment_dir / "validation" / "summary.json").relative_to(paths.root).as_posix()
        ),
        "mlflow_run_id": mlflow_run_id,
    }

    records: list[EvidenceRecord] = []
    for model_id, metrics in result.models.items():
        for metric_name, value in metrics.items():
            records.append(
                EvidenceRecord(
                    evidence_id=f"ev-{experiment_id}-{model_id}-{metric_name}",
                    key=f"metrics.{model_id}.{metric_name}",
                    raw_value=value,
                    display_value=f"{value:.6g}",
                    experiment_id=experiment_id,
                    artifact_path=relative_metrics,
                    artifact_hash=metrics_hash,
                    source_field=f"models.{model_id}.{metric_name}",
                    validation_reports=[report.validator for report in reports],
                    verified_at=now,
                    provenance=provenance,
                )
            )

    verified = _verified_dir(paths)
    index = load_evidence_index(paths)
    for record in records:
        index.records[record.evidence_id] = record
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")

    paths.verified_metrics_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        paths.verified_metrics_dir / f"{experiment_id}.json",
        json.dumps(
            {
                "experiment_id": experiment_id,
                "artifact_hash": metrics_hash,
                "records": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
        )
        + "\n",
    )

    provenance_path = verified / "provenance.json"
    provenance_index = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {"schema_version": 1, "experiments": {}}
    )
    provenance_index["experiments"][experiment_id] = {
        **provenance,
        "validation_reports": report_hashes,
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")

    if not paths.verified_claims_file.exists():
        atomic_write_text(
            paths.verified_claims_file,
            yaml.safe_dump({"claims": []}, sort_keys=False, allow_unicode=True),
        )

    return records


def add_claim(
    paths: ProjectPaths,
    *,
    statement: str,
    evidence_ids: list[str],
    claim_id: str | None = None,
    accept: bool = False,
    reviewer: str | None = None,
    notes: str = "",
) -> ClaimRecord:
    index = load_evidence_index(paths)
    if not index.records:
        raise EvidenceError("no verified evidence exists; freeze an experiment first")
    missing = [evidence_id for evidence_id in evidence_ids if evidence_id not in index.records]
    if missing:
        raise EvidenceError(f"unknown evidence id(s): {', '.join(missing)}")

    claims = load_claims(paths)
    resolved_id = claim_id or f"claim-{secrets.token_hex(3)}"
    if any(claim.claim_id == resolved_id for claim in claims):
        raise EvidenceError(f"claim id already exists: {resolved_id}")

    claim = ClaimRecord(
        claim_id=resolved_id,
        statement=statement,
        evidence_ids=evidence_ids,
        status="supported" if accept else "proposed",
        reviewer=reviewer,
        notes=notes,
    )
    claims.append(claim)
    atomic_write_text(
        paths.verified_claims_file,
        yaml.safe_dump(
            {"claims": [claim.model_dump(mode="json") for claim in claims]},
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    return claim
