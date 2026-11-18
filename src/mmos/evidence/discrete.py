from __future__ import annotations

import json
import math
from datetime import UTC, datetime

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    finite_policy_result_file,
    finite_policy_spec_file,
    linear_milp_result_file,
    linear_milp_spec_file,
    load_finite_policy_result,
    load_finite_policy_spec,
    load_linear_milp_result,
    sha256_file,
)
from mmos.evidence.store import EvidenceError, load_evidence_index
from mmos.schemas.discrete import FinitePolicySpec, MILPSolverStatus
from mmos.schemas.evidence import EvidenceRecord
from mmos.validation.discrete import (
    validate_finite_policy_experiment,
    validate_linear_milp_experiment,
)


def _float_display(value: float) -> str:
    return f"{value:.10g}"


def verify_policy_provenance(
    paths: ProjectPaths, spec: FinitePolicySpec
) -> None:
    if not spec.parameter_provenance:
        return
    index = load_evidence_index(paths)
    for item in spec.parameter_provenance:
        record = index.records.get(item.source_evidence_id)
        if record is None:
            raise EvidenceError(
                f"provenance cites unknown evidence {item.source_evidence_id!r}"
            )
        if record.artifact_hash != item.source_artifact_hash:
            raise EvidenceError(
                f"provenance for {item.name!r} is stale: source artifact hash changed"
            )
        if isinstance(record.raw_value, bool) or not isinstance(
            record.raw_value, (int, float)
        ):
            raise EvidenceError(
                f"provenance for {item.name!r} does not cite a numeric point estimate"
            )
        if not math.isfinite(float(record.raw_value)):
            raise EvidenceError(f"provenance for {item.name!r} is not finite")
        if abs(float(record.raw_value) - item.value) > 1e-12:
            raise EvidenceError(
                f"provenance for {item.name!r} is stale: point estimate changed"
            )


def validate_and_register_linear_milp(
    paths: ProjectPaths, experiment_id: str
) -> list[EvidenceRecord]:
    report = validate_linear_milp_experiment(paths, experiment_id)[0]
    if report.blocking:
        raise EvidenceError(f"linear MILP validation failed for {experiment_id}")
    result = load_linear_milp_result(paths, experiment_id)
    spec_path = linear_milp_spec_file(paths, experiment_id)
    artifact_path = linear_milp_result_file(paths, experiment_id)
    artifact_hash = sha256_file(artifact_path)
    now = datetime.now(UTC).replace(microsecond=0)
    relative_artifact = artifact_path.relative_to(paths.root).as_posix()
    provenance = {
        "kind": "linear_milp",
        "spec_path": spec_path.relative_to(paths.root).as_posix(),
        "spec_sha256": sha256_file(spec_path),
        "artifact_path": relative_artifact,
        "artifact_sha256": artifact_hash,
        "instance_hash": result.instance_hash,
        "solver_status": result.solver_status.value,
        "validator": "linear_milp",
    }
    records: list[EvidenceRecord] = []

    def add(evidence_id: str, key: str, value: float | int | str) -> None:
        display = _float_display(value) if isinstance(value, float) else str(value)
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{experiment_id}-{evidence_id}",
                key=f"linear_milp.{key}",
                raw_value=value,
                display_value=display,
                experiment_id=experiment_id,
                artifact_path=relative_artifact,
                artifact_hash=artifact_hash,
                source_field=key,
                validation_reports=["linear_milp"],
                verified_at=now,
                provenance=provenance,
            )
        )

    add("instance-hash", "instance_hash", result.instance_hash)
    add("solver-status", "solver.status", result.solver_status.value)
    add("termination-reason", "solver.termination_reason", result.termination_reason)
    add("solver-name", "solver.name", result.solver_name)
    add("solver-version", "solver.version", result.solver_version)
    if result.solver_status in (
        MILPSolverStatus.OPTIMAL,
        MILPSolverStatus.FEASIBLE_NOT_PROVEN,
    ):
        if result.objective_value is not None:
            add("objective", "objective", result.objective_value)
        if result.best_bound is not None:
            add("best-bound", "best_bound", result.best_bound)
        if result.gap is not None:
            add("gap", "gap", result.gap)
        for name, value in result.variable_values.items():
            add(f"variable-{name}", f"variable.{name}", value)
        for activity in result.constraint_activities:
            add(
                f"constraint-{activity.constraint_id}-slack",
                f"constraint.{activity.constraint_id}.slack",
                activity.slack,
            )
            add(
                f"constraint-{activity.constraint_id}-satisfied",
                f"constraint.{activity.constraint_id}.satisfied",
                str(activity.satisfied).lower(),
            )
    _register(paths, records)
    return records


def validate_and_register_finite_policy(
    paths: ProjectPaths, experiment_id: str
) -> list[EvidenceRecord]:
    spec = load_finite_policy_spec(paths, experiment_id)
    verify_policy_provenance(paths, spec)
    report = validate_finite_policy_experiment(paths, experiment_id)[0]
    if report.blocking:
        raise EvidenceError(f"finite policy validation failed for {experiment_id}")
    result = load_finite_policy_result(paths, experiment_id)
    spec_path = finite_policy_spec_file(paths, experiment_id)
    artifact_path = finite_policy_result_file(paths, experiment_id)
    artifact_hash = sha256_file(artifact_path)
    now = datetime.now(UTC).replace(microsecond=0)
    relative_artifact = artifact_path.relative_to(paths.root).as_posix()
    provenance = {
        "kind": "finite_policy",
        "spec_path": spec_path.relative_to(paths.root).as_posix(),
        "spec_sha256": sha256_file(spec_path),
        "artifact_path": relative_artifact,
        "artifact_sha256": artifact_hash,
        "evaluator_id": result.evaluator_id,
        "evaluator_version": result.evaluator_version,
        "evaluator_hash": result.evaluator_hash,
        "parameter_hash": result.parameter_hash,
        "provenance_hash": result.provenance_hash,
        "point_estimate_conditioned": str(result.point_estimate_conditioned).lower(),
        "source_evidence": json.dumps(
            [
                {
                    "name": item.name,
                    "evidence_id": item.source_evidence_id,
                    "artifact_hash": item.source_artifact_hash,
                }
                for item in result.parameter_provenance
            ],
            sort_keys=True,
        ),
        "validator": "finite_policy",
    }
    records: list[EvidenceRecord] = []

    def add(evidence_id: str, key: str, value: float | int | str) -> None:
        display = _float_display(value) if isinstance(value, float) else str(value)
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{experiment_id}-{evidence_id}",
                key=f"finite_policy.{key}",
                raw_value=value,
                display_value=display,
                experiment_id=experiment_id,
                artifact_path=relative_artifact,
                artifact_hash=artifact_hash,
                source_field=key,
                validation_reports=["finite_policy"],
                verified_at=now,
                provenance=provenance,
            )
        )

    add("execution-status", "execution_status", result.execution_status.value)
    add("evaluator-id", "evaluator_id", result.evaluator_id)
    add("evaluator-version", "evaluator_version", result.evaluator_version)
    add("evaluator-hash", "evaluator_hash", result.evaluator_hash)
    add("parameter-hash", "parameter_hash", result.parameter_hash)
    add("state-space-size", "state_space_size", result.state_space_size)
    add("evaluated-state-count", "evaluated_state_count", result.evaluated_state_count)
    add("feasible-policy-count", "feasible_policy_count", result.feasible_policy_count)
    add("optimal-policy-count", "optimal_policy_count", result.optimal_policy_count)
    add("tie-break-rule", "tie_break_rule", result.tie_break_rule)
    if result.point_estimate_conditioned and result.uncertainty_disclosure is not None:
        add(
            "uncertainty-disclosure",
            "uncertainty_disclosure",
            result.uncertainty_disclosure,
        )
    if result.selected_policy is not None:
        for name, value in result.selected_policy.items():
            add(f"selected-{name}", f"selected_policy.{name}", str(value))
        if result.objective_value is not None:
            add("objective", "objective_value", result.objective_value)
        for key, value in result.metrics.items():
            add(f"metric-{key}", f"metrics.{key}", value)
        if result.objective_components is not None:
            for key, value in result.objective_components.items():
                add(f"component-{key}", f"objective_components.{key}", value)
    _register(paths, records)
    return records


def _register(paths: ProjectPaths, records: list[EvidenceRecord]) -> None:
    index = load_evidence_index(paths)
    for record in records:
        if record.evidence_id in index.records:
            raise EvidenceError(f"evidence id already exists: {record.evidence_id}")
        index.records[record.evidence_id] = record
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")


__all__ = [
    "validate_and_register_finite_policy",
    "validate_and_register_linear_milp",
    "verify_policy_provenance",
]
