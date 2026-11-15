import json
from datetime import UTC, datetime

import yaml

from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    epsilon_solutions_file,
    load_multi_objective_record,
    load_multi_objective_result,
    load_multi_objective_spec,
    load_pareto_oracle,
    multi_objective_record_file,
    multi_objective_result_file,
    multi_objective_spec_file,
    pareto_oracle_file,
    sha256_file,
)
from mmos.evidence.store import EvidenceError, load_evidence_index
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.validation import ValidationReport, ValidationStatus


def freeze_multi_objective(
    paths: ProjectPaths, experiment_id: str
) -> list[EvidenceRecord]:
    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    summary_path = validation_dir / "summary.json"
    report_path = validation_dir / "multi_objective.json"
    if not summary_path.exists() or not report_path.exists():
        raise EvidenceError(
            f"multi-objective experiment {experiment_id!r} has not been validated"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("blocking"):
        raise EvidenceError("multi-objective validation contains blocking issues")
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.blocking or report.status is not ValidationStatus.PASSED:
        raise EvidenceError("multi-objective validator did not pass")

    spec = load_multi_objective_spec(paths, experiment_id)
    result = load_multi_objective_result(paths, experiment_id)
    oracle = load_pareto_oracle(paths, experiment_id)
    record = load_multi_objective_record(paths, experiment_id)
    spec_path = multi_objective_spec_file(paths, experiment_id)
    result_path = multi_objective_result_file(paths, experiment_id)
    oracle_path = pareto_oracle_file(paths, experiment_id)
    epsilon_path = epsilon_solutions_file(paths, experiment_id)
    hashes = {
        "spec": sha256_file(spec_path),
        "result": sha256_file(result_path),
        "oracle": sha256_file(oracle_path),
        "epsilon": sha256_file(epsilon_path) if epsilon_path.exists() else "",
        "record": sha256_file(multi_objective_record_file(paths, experiment_id)),
    }
    now = datetime.now(UTC).replace(microsecond=0)
    comparison = result.comparison
    provenance = {
        "kind": "multi_objective",
        "spec_path": spec_path.relative_to(paths.root).as_posix(),
        "spec_sha256": hashes["spec"],
        "result_path": result_path.relative_to(paths.root).as_posix(),
        "result_sha256": hashes["result"],
        "oracle_path": oracle_path.relative_to(paths.root).as_posix(),
        "oracle_sha256": hashes["oracle"],
        "epsilon_solutions_path": (
            epsilon_path.relative_to(paths.root).as_posix() if epsilon_path.exists() else ""
        ),
        "epsilon_solutions_sha256": hashes["epsilon"],
        "record_path": multi_objective_record_file(paths, experiment_id)
        .relative_to(paths.root)
        .as_posix(),
        "record_sha256": hashes["record"],
        "instance_hash": spec.instance_hash,
        "spec_hash": spec.spec_hash,
        "git_commit": record.git_commit,
        "mlflow_run_id": record.mlflow_run_id or "",
    }

    def evidence(evidence_id: str, key: str, value: float | int | str) -> EvidenceRecord:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            experiment_id=experiment_id,
            artifact_path=provenance["result_path"],
            artifact_hash=hashes["result"],
            source_field=key,
            validation_reports=[report.validator],
            verified_at=now,
            provenance=provenance,
        )

    records: list[EvidenceRecord] = [
        evidence(
            f"ev-{experiment_id}-multiobjective-instance",
            "multiobjective.instance_hash",
            spec.instance_hash,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-objectives",
            "multiobjective.objectives",
            json.dumps(
                [objective.model_dump(mode="json") for objective in spec.objectives],
                sort_keys=True,
            ),
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-candidate-front",
            "multiobjective.pareto_front.candidate",
            json.dumps(
                [vector.model_dump(mode="json") for vector in result.candidate_pareto_front],
                sort_keys=True,
            ),
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-oracle-front",
            "multiobjective.pareto_front.oracle",
            json.dumps(
                [vector.model_dump(mode="json") for vector in oracle.pareto_objective_vectors],
                sort_keys=True,
            ),
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-candidate-front-size",
            "multiobjective.front_size",
            result.candidate_front_size,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-oracle-front-size",
            "multiobjective.oracle_front_size",
            oracle.front_size,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-precision",
            "multiobjective.front_precision",
            comparison.precision if comparison else 0.0,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-recall",
            "multiobjective.front_recall",
            comparison.recall if comparison else 0.0,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-search-space",
            "multiobjective.search_space_size",
            oracle.search_space_size,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-feasible-count",
            "multiobjective.feasible_solution_count",
            oracle.feasible_solution_count,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-epsilon-policy",
            "multiobjective.epsilon_policy",
            json.dumps(
                {
                    "mode": spec.epsilon_policy,
                    "step": spec.epsilon_step,
                    "levels": result.epsilon_coverage.model_dump(mode="json"),
                },
                sort_keys=True,
            ),
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-reference-point",
            "multiobjective.reference_point",
            json.dumps(spec.reference_point),
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-hypervolume",
            "multiobjective.hypervolume",
            result.hypervolume,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-hypervolume-oracle",
            "multiobjective.hypervolume_oracle",
            oracle.hypervolume,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-feasibility-audit",
            "multiobjective.feasibility_audit",
            "passed" if not report.blocking else "failed",
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-method",
            "multiobjective.method",
            result.method,
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-solver",
            "multiobjective.solver",
            f"{result.solver} {result.solver_version}",
        ),
        evidence(
            f"ev-{experiment_id}-multiobjective-solver-calls",
            "multiobjective.solver_calls",
            result.solver_calls,
        ),
    ]

    index = load_evidence_index(paths)
    for item in records:
        index.records[item.evidence_id] = item
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    output_dir = paths.verified_dir / "multi_objective"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / f"{experiment_id}.json",
        json.dumps(
            {
                "experiment_id": experiment_id,
                "artifact_hash": hashes["result"],
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
    provenance_index["experiments"][f"multi_objective:{experiment_id}"] = {
        **provenance,
        "validation_reports": {"multi_objective": sha256_file(report_path)},
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")
    if not paths.verified_claims_file.exists():
        atomic_write_text(
            paths.verified_claims_file,
            yaml.safe_dump({"claims": []}, sort_keys=False, allow_unicode=True),
        )
    return records


__all__ = ["freeze_multi_objective"]
