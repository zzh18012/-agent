import json
from datetime import UTC, datetime

import yaml

from mmos.core.project import (
    ProjectPaths,
    analytic_oracle_file,
    atomic_write_text,
    load_analytic_oracle,
    load_simulation_record,
    load_simulation_result,
    load_simulation_spec,
    replications_file,
    sha256_file,
    simulation_record_file,
    simulation_result_file,
    simulation_spec_file,
)
from mmos.evidence.store import EvidenceError, load_evidence_index
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.validation import ValidationReport, ValidationStatus


def freeze_simulation(paths: ProjectPaths, experiment_id: str) -> list[EvidenceRecord]:
    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    summary_path = validation_dir / "summary.json"
    report_path = validation_dir / "simulation.json"
    if not summary_path.exists() or not report_path.exists():
        raise EvidenceError(f"simulation experiment {experiment_id!r} has not been validated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("blocking"):
        raise EvidenceError("simulation validation contains blocking issues")
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.blocking or report.status is not ValidationStatus.PASSED:
        raise EvidenceError("simulation validator did not pass")

    spec = load_simulation_spec(paths, experiment_id)
    result = load_simulation_result(paths, experiment_id)
    oracle = load_analytic_oracle(paths, experiment_id)
    record = load_simulation_record(paths, experiment_id)
    spec_path = simulation_spec_file(paths, experiment_id)
    result_path = simulation_result_file(paths, experiment_id)
    oracle_path = analytic_oracle_file(paths, experiment_id)
    replication_path = replications_file(paths, experiment_id)
    spec_hash = sha256_file(spec_path)
    result_hash = sha256_file(result_path)
    oracle_hash = sha256_file(oracle_path)
    replication_hash = sha256_file(replication_path) if replication_path.exists() else ""
    now = datetime.now(UTC).replace(microsecond=0)
    record_path = simulation_record_file(paths, experiment_id)
    provenance = {
        "kind": "simulation",
        "spec_path": spec_path.relative_to(paths.root).as_posix(),
        "spec_sha256": spec_hash,
        "result_path": result_path.relative_to(paths.root).as_posix(),
        "result_sha256": result_hash,
        "oracle_path": oracle_path.relative_to(paths.root).as_posix(),
        "oracle_sha256": oracle_hash,
        "replications_path": (
            replication_path.relative_to(paths.root).as_posix()
            if replication_path.exists()
            else ""
        ),
        "replications_sha256": replication_hash,
        "experiment_record_path": record_path.relative_to(paths.root).as_posix(),
        "experiment_record_sha256": sha256_file(record_path),
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id or "",
        "spec_hash": spec.spec_hash,
        "replication_seed_hash": result.replication_seed_hash,
    }
    seed_schedule = [
        {
            "replication_id": replication.replication_id,
            "replication_seed": replication.replication_seed,
            "arrival_seed": replication.arrival_seed,
            "service_seed": replication.service_seed,
        }
        for replication in result.replications
    ]
    little_law = {
        metric: {
            "relative_discrepancy": check.relative_discrepancy,
            "tolerance": check.tolerance,
            "passed": check.passed,
        }
        for metric, check in result.little_law.items()
    }

    def evidence(
        evidence_id: str,
        key: str,
        value: float | int | str,
        *,
        artifact_path: str | None = None,
        artifact_hash: str | None = None,
    ) -> EvidenceRecord:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            experiment_id=experiment_id,
            artifact_path=artifact_path or provenance["result_path"],
            artifact_hash=artifact_hash or result_hash,
            source_field=key,
            validation_reports=[report.validator],
            verified_at=now,
            provenance=provenance,
        )

    records: list[EvidenceRecord] = [
        evidence(f"ev-{experiment_id}-simulation-lambda", "simulation.lambda", spec.arrival_rate),
        evidence(f"ev-{experiment_id}-simulation-mu", "simulation.mu", spec.service_rate),
        evidence(f"ev-{experiment_id}-simulation-rho", "simulation.rho", spec.rho),
        evidence(
            f"ev-{experiment_id}-simulation-replications",
            "simulation.replications",
            spec.replications,
        ),
        evidence(
            f"ev-{experiment_id}-simulation-warmup",
            "simulation.warmup",
            spec.warmup_time,
        ),
        evidence(
            f"ev-{experiment_id}-simulation-measurement",
            "simulation.measurement_time",
            spec.measurement_time,
        ),
        evidence(
            f"ev-{experiment_id}-simulation-seed-schedule",
            "simulation.seed_schedule",
            json.dumps(seed_schedule, sort_keys=True),
            artifact_path=provenance["replications_path"] or provenance["result_path"],
            artifact_hash=replication_hash or result_hash,
        ),
        evidence(
            f"ev-{experiment_id}-simulation-little-law",
            "simulation.little_law",
            json.dumps(little_law, sort_keys=True),
        ),
        evidence(
            f"ev-{experiment_id}-simulation-simpy-version",
            "simulation.simpy_version",
            result.runtime.get("simpy_version", "unknown"),
        ),
    ]
    theory = {
        "W": oracle.W,
        "Wq": oracle.Wq,
        "L": oracle.L,
        "Lq": oracle.Lq,
        "utilization": oracle.utilization,
        "throughput": oracle.throughput,
    }
    for metric, summary in result.metrics.items():
        records.append(
            evidence(
                f"ev-{experiment_id}-simulation-{metric}-mean",
                f"simulation.{metric}.mean",
                summary.mean,
            )
        )
        records.append(
            evidence(
                f"ev-{experiment_id}-simulation-{metric}-std",
                f"simulation.{metric}.std",
                summary.std,
            )
        )
        records.append(
            evidence(
                f"ev-{experiment_id}-simulation-{metric}-ci",
                f"simulation.{metric}.ci",
                f"[{summary.ci_low:.10g}, {summary.ci_high:.10g}]",
            )
        )
        records.append(
            evidence(
                f"ev-{experiment_id}-simulation-{metric}-ci-low",
                f"simulation.{metric}.ci_low",
                summary.ci_low,
            )
        )
        records.append(
            evidence(
                f"ev-{experiment_id}-simulation-{metric}-ci-high",
                f"simulation.{metric}.ci_high",
                summary.ci_high,
            )
        )
        if metric in theory:
            records.append(
                evidence(
                    f"ev-{experiment_id}-simulation-{metric}-theory",
                    f"simulation.{metric}.theory",
                    theory[metric],
                )
            )
    for metric, check in result.little_law.items():
        records.append(
            evidence(
                f"ev-{experiment_id}-simulation-little-law-{metric}",
                f"simulation.little_law.{metric}.relative_discrepancy",
                check.relative_discrepancy,
            )
        )

    index = load_evidence_index(paths)
    for item in records:
        index.records[item.evidence_id] = item
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    simulation_dir = paths.verified_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        simulation_dir / f"{experiment_id}.json",
        json.dumps(
            {
                "experiment_id": experiment_id,
                "artifact_hash": result_hash,
                "records": [item.model_dump(mode="json") for item in records],
            },
            indent=2,
        )
        + "\n",
    )
    provenance_path = paths.verified_dir / "provenance.json"
    provenance_index = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {"schema_version": 1, "experiments": {}}
    )
    provenance_index["experiments"][f"simulation:{experiment_id}"] = {
        **provenance,
        "validation_reports": {"simulation": sha256_file(report_path)},
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")
    if not paths.verified_claims_file.exists():
        atomic_write_text(
            paths.verified_claims_file,
            yaml.safe_dump({"claims": []}, sort_keys=False, allow_unicode=True),
        )
    return records


__all__ = ["freeze_simulation"]
