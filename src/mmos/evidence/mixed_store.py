import json
from datetime import UTC, datetime

import yaml

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    experiment_record_file,
    experiment_spec_file,
    load_experiment_record,
    load_experiment_spec,
    load_optimization_result,
    load_optimization_spec,
    optimization_result_file,
    optimization_spec_file,
    sha256_file,
)
from mmos.evidence.store import EvidenceError, load_evidence_index
from mmos.mixed.artifacts import load_envelope
from mmos.mixed.simulation import (
    load_mixed_simulation_record,
    load_mixed_simulation_result,
    load_mixed_simulation_spec,
    mixed_simulation_record_file,
    mixed_simulation_result_file,
    mixed_simulation_spec_file,
)
from mmos.mixed.validation import run_record_path
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.mixed import (
    DAGRunRecord,
    ForecastArtifact,
    ForecastErrorArtifact,
    MixedExecutionPlan,
    ProductionPlanArtifact,
)
from mmos.schemas.validation import ValidationReport, ValidationStatus


def freeze_mixed(
    paths: ProjectPaths, plan: MixedExecutionPlan, run_id: str
) -> list[EvidenceRecord]:
    validation_dir = paths.mixed_dir / "runs" / run_id / "validation"
    summary_path = validation_dir / "summary.json"
    report_path = validation_dir / "mixed.json"
    if not summary_path.exists() or not report_path.exists():
        raise EvidenceError(f"mixed run {run_id!r} has not been validated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("blocking"):
        raise EvidenceError("mixed validation contains blocking issues")
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.blocking or report.status is not ValidationStatus.PASSED:
        raise EvidenceError("mixed validator did not pass")
    simulation_report = None
    simulation_report_path = validation_dir / "mixed_simulation.json"
    if plan.has_simulation_node():
        if not simulation_report_path.exists():
            raise EvidenceError("mixed simulation validator report is missing")
        simulation_report = ValidationReport.model_validate_json(
            simulation_report_path.read_text(encoding="utf-8")
        )
        if (
            simulation_report.blocking
            or simulation_report.status is not ValidationStatus.PASSED
        ):
            raise EvidenceError("mixed simulation validator did not pass")

    record_path = run_record_path(paths, run_id)
    record = DAGRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    envelope_path = paths.mixed_dir / "runs" / run_id / "artifacts" / "forecast_envelope.json"
    envelope = load_envelope(envelope_path)
    artifact_path = paths.root / envelope.payload_path
    artifact = ForecastArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    forecast_spec = load_experiment_spec(paths, artifact.forecast_experiment_id)
    forecast_spec_path = experiment_spec_file(paths, artifact.forecast_experiment_id)
    if record.optimization_experiment_id is None:
        raise EvidenceError("mixed run has no optimization experiment")
    optimization_id = record.optimization_experiment_id
    optimization_spec = load_optimization_spec(paths, optimization_id)
    optimization_result = load_optimization_result(paths, optimization_id)
    optimization_record = load_experiment_record(paths, optimization_id)
    optimization_spec_path = optimization_spec_file(paths, optimization_id)
    optimization_result_path = optimization_result_file(paths, optimization_id)
    optimization_record_path = experiment_record_file(paths, optimization_id)

    hashes = {
        "run": sha256_file(record_path),
        "envelope": sha256_file(envelope_path),
        "artifact": sha256_file(artifact_path),
        "forecast_spec": sha256_file(forecast_spec_path),
        "optimization_spec": sha256_file(optimization_spec_path),
        "optimization_result": sha256_file(optimization_result_path),
        "optimization_record": sha256_file(optimization_record_path),
    }
    plan_envelope = None
    plan_artifact = None
    error_envelope = None
    error_artifact = None
    simulation_spec = None
    simulation_result = None
    if plan.has_simulation_node():
        run_artifacts = paths.mixed_dir / "runs" / run_id / "artifacts"
        plan_envelope_path = run_artifacts / "production_plan_envelope.json"
        error_envelope_path = run_artifacts / "forecast_error_envelope.json"
        plan_envelope = load_envelope(plan_envelope_path)
        plan_artifact = ProductionPlanArtifact.model_validate_json(
            (paths.root / plan_envelope.payload_path).read_text(encoding="utf-8")
        )
        error_envelope = load_envelope(error_envelope_path)
        error_artifact = ForecastErrorArtifact.model_validate_json(
            (paths.root / error_envelope.payload_path).read_text(encoding="utf-8")
        )
        if record.simulation_experiment_id is None:
            raise EvidenceError("mixed run has no simulation experiment")
        simulation_id = record.simulation_experiment_id
        simulation_spec = load_mixed_simulation_spec(paths, simulation_id)
        simulation_result = load_mixed_simulation_result(paths, simulation_id)
        load_mixed_simulation_record(paths, simulation_id)
        hashes.update(
            {
                "plan_envelope": sha256_file(plan_envelope_path),
                "plan_artifact": sha256_file(paths.root / plan_envelope.payload_path),
                "error_envelope": sha256_file(error_envelope_path),
                "error_artifact": sha256_file(paths.root / error_envelope.payload_path),
                "simulation_spec": sha256_file(
                    mixed_simulation_spec_file(paths, simulation_id)
                ),
                "simulation_result": sha256_file(
                    mixed_simulation_result_file(paths, simulation_id)
                ),
                "simulation_record": sha256_file(
                    mixed_simulation_record_file(paths, simulation_id)
                ),
            }
        )
    now = datetime.now(UTC).replace(microsecond=0)
    provenance = {
        "kind": "mixed",
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "spec_hash": plan.spec.spec_hash,
        "run_id": run_id,
        "run_record_path": record_path.relative_to(paths.root).as_posix(),
        "run_record_sha256": hashes["run"],
        "forecast_artifact_id": envelope.artifact_id,
        "forecast_artifact_hash": envelope.content_hash,
        "forecast_payload_path": envelope.payload_path,
        "forecast_payload_sha256": hashes["artifact"],
        "forecast_envelope_path": envelope_path.relative_to(paths.root).as_posix(),
        "forecast_envelope_sha256": hashes["envelope"],
        "forecast_experiment_id": artifact.forecast_experiment_id,
        "forecast_experiment_spec_sha256": hashes["forecast_spec"],
        "dataset_hash": forecast_spec.dataset_hash,
        "optimization_experiment_id": optimization_id,
        "optimization_spec_sha256": hashes["optimization_spec"],
        "optimization_result_sha256": hashes["optimization_result"],
        "optimization_record_sha256": hashes["optimization_record"],
        "instance_hash": optimization_spec.instance_hash or "",
        "dependency_contract_status": optimization_record.dependency_contract_status
        or "missing",
        "consumed_artifact_ids": ",".join(optimization_record.consumed_artifact_ids),
        "consumed_artifact_hashes": ",".join(optimization_record.consumed_artifact_hashes),
        "solver": optimization_result.solver_backend,
        "solver_version": optimization_result.solver_version,
        "git_commit": optimization_record.git_commit,
    }
    if simulation_spec is not None and simulation_result is not None:
        if plan_envelope is None or error_envelope is None:
            raise EvidenceError("simulation evidence is missing plan or error artifacts")
        provenance.update(
            {
                "production_plan_artifact_id": plan_envelope.artifact_id,
                "production_plan_artifact_hash": plan_envelope.content_hash,
                "production_plan_envelope_sha256": hashes["plan_envelope"],
                "production_plan_payload_sha256": hashes["plan_artifact"],
                "forecast_error_artifact_id": error_envelope.artifact_id,
                "forecast_error_artifact_hash": error_envelope.content_hash,
                "forecast_error_envelope_sha256": hashes["error_envelope"],
                "forecast_error_payload_sha256": hashes["error_artifact"],
                "simulation_experiment_id": simulation_spec.simulation_experiment_id,
                "simulation_spec_sha256": hashes["simulation_spec"],
                "simulation_result_sha256": hashes["simulation_result"],
                "simulation_record_sha256": hashes["simulation_record"],
                "simulation_seed_hash": simulation_result.replication_seed_hash,
                "simulation_replications": str(simulation_result.n_replications),
                "contract_chain_complete": str(record.contract_chain_complete).lower(),
            }
        )

    validation_reports = [report.validator]
    if simulation_report is not None:
        validation_reports.append(simulation_report.validator)

    def evidence(evidence_id: str, key: str, value: float | int | str) -> EvidenceRecord:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            experiment_id=optimization_id,
            artifact_path=record_path.relative_to(paths.root).as_posix(),
            artifact_hash=hashes["run"],
            source_field=key,
            validation_reports=validation_reports,
            verified_at=now,
            provenance=provenance,
        )

    records: list[EvidenceRecord] = [
        evidence(f"ev-mixed-{run_id}-plan-id", "mixed.plan_id", plan.plan_id),
        evidence(f"ev-mixed-{run_id}-plan-hash", "mixed.plan_hash", plan.plan_hash),
        evidence(f"ev-mixed-{run_id}-spec-hash", "mixed.spec_hash", plan.spec.spec_hash),
        evidence(f"ev-mixed-{run_id}-run-id", "mixed.run_id", run_id),
        evidence(f"ev-mixed-{run_id}-artifact-id", "mixed.artifact.id", envelope.artifact_id),
        evidence(
            f"ev-mixed-{run_id}-artifact-hash",
            "mixed.artifact.content_hash",
            envelope.content_hash,
        ),
        evidence(
            f"ev-mixed-{run_id}-artifact-payload-hash",
            "mixed.artifact.payload_hash",
            envelope.payload_hash,
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-experiment",
            "mixed.forecast.experiment_id",
            artifact.forecast_experiment_id,
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-model",
            "mixed.forecast.model_id",
            artifact.model_id,
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-horizon",
            "mixed.forecast.horizon",
            artifact.forecast_horizon,
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-history-rows",
            "mixed.forecast.history_rows",
            envelope.provenance.get("history_rows", "unavailable"),
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-history-digest",
            "mixed.forecast.history_digest",
            envelope.provenance.get("history_digest", "unavailable"),
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-dataset",
            "mixed.forecast.dataset_hash",
            artifact.source_dataset_hash,
        ),
        evidence(
            f"ev-mixed-{run_id}-forecast-points",
            "mixed.forecast.point_forecasts",
            json.dumps(
                {
                    timestamp: value
                    for timestamp, value in zip(
                        artifact.timestamps, artifact.point_forecasts, strict=True
                    )
                },
                sort_keys=True,
            ),
        ),
        evidence(
            f"ev-mixed-{run_id}-optimization-experiment",
            "mixed.optimization.experiment_id",
            optimization_id,
        ),
        evidence(
            f"ev-mixed-{run_id}-dependency-contract",
            "mixed.dependency_contract_status",
            optimization_record.dependency_contract_status or "missing",
        ),
        evidence(
            f"ev-mixed-{run_id}-provenance",
            "mixed.provenance_complete",
            str(record.provenance_complete).lower(),
        ),
    ]
    solver = next(
        (item for item in optimization_result.solutions if item.method == "ortools.cbc"),
        None,
    )
    if solver is None:
        raise EvidenceError("mixed run has no CBC solution")
    if solver.objective_value is not None:
        records.append(
            evidence(
                f"ev-mixed-{run_id}-objective",
                "mixed.optimization.objective_value",
                solver.objective_value,
            )
        )
    records.append(
        evidence(
            f"ev-mixed-{run_id}-feasible",
            "mixed.optimization.feasible",
            str(solver.feasible).lower(),
        )
    )
    for name, value in solver.variable_values.items():
        records.append(
            evidence(
                f"ev-mixed-{run_id}-plan-{name}",
                f"mixed.plan.{name}",
                value,
            )
        )
    if simulation_result is not None and simulation_spec is not None:
        if (
            plan_envelope is None
            or plan_artifact is None
            or error_envelope is None
            or error_artifact is None
        ):
            raise EvidenceError("simulation evidence is missing artifacts")
        records.extend(
            [
                evidence(
                    f"ev-mixed-{run_id}-production-plan-id",
                    "mixed.production_plan.artifact_id",
                    plan_envelope.artifact_id,
                ),
                evidence(
                    f"ev-mixed-{run_id}-production-plan-hash",
                    "mixed.production_plan.artifact_hash",
                    plan_envelope.content_hash,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-id",
                    "mixed.forecast_error.artifact_id",
                    error_envelope.artifact_id,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-hash",
                    "mixed.forecast_error.artifact_hash",
                    error_envelope.content_hash,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-definition",
                    "mixed.forecast_error.residual_definition",
                    error_artifact.residual_definition,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-source",
                    "mixed.forecast_error.residual_source",
                    error_artifact.residual_source,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-samples",
                    "mixed.forecast_error.sample_count",
                    error_artifact.sample_count,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-mean",
                    "mixed.forecast_error.mean",
                    error_artifact.mean,
                ),
                evidence(
                    f"ev-mixed-{run_id}-forecast-error-std",
                    "mixed.forecast_error.std",
                    error_artifact.std,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-experiment",
                    "mixed.simulation.experiment_id",
                    simulation_spec.simulation_experiment_id,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-seed-hash",
                    "mixed.simulation.seed_hash",
                    simulation_result.replication_seed_hash,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-replications",
                    "mixed.simulation.replications",
                    simulation_result.n_replications,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-shortage",
                    "mixed.simulation.shortage.mean_total_shortage",
                    simulation_result.metrics["total_shortage"].mean,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-fill-rate",
                    "mixed.simulation.service.mean_fill_rate",
                    simulation_result.metrics["fill_rate"].mean,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-any-shortage",
                    "mixed.simulation.any_shortage_probability",
                    simulation_result.any_shortage_probability,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-distinct-paths",
                    "mixed.simulation.distinct_residual_paths",
                    simulation_result.distinct_residual_paths,
                ),
                evidence(
                    f"ev-mixed-{run_id}-simulation-floor-activations",
                    "mixed.simulation.demand_floor_activations",
                    simulation_result.demand_floor_activations,
                ),
                evidence(
                    f"ev-mixed-{run_id}-contract-chain",
                    "mixed.contract_chain_complete",
                    str(record.contract_chain_complete).lower(),
                ),
            ]
        )

    index = load_evidence_index(paths)
    for item in records:
        index.records[item.evidence_id] = item
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    output_dir = paths.verified_dir / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / f"{run_id}.json",
        json.dumps(
            {
                "experiment_id": optimization_id,
                "run_id": run_id,
                "artifact_hash": hashes["run"],
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
    validation_report_hashes = {"mixed": sha256_file(report_path)}
    if simulation_report_path.exists():
        validation_report_hashes["mixed_simulation"] = sha256_file(
            simulation_report_path
        )
    provenance_index["experiments"][f"mixed:{run_id}"] = {
        **provenance,
        "validation_reports": validation_report_hashes,
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")
    if not paths.verified_claims_file.exists():
        atomic_write_text(
            paths.verified_claims_file,
            yaml.safe_dump({"claims": []}, sort_keys=False, allow_unicode=True),
        )
    return records


__all__ = ["freeze_mixed"]
