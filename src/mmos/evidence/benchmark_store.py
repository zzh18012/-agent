import json
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.evidence.store import load_evidence_index
from mmos.schemas.benchmark import BenchmarkSuiteResult
from mmos.schemas.evidence import EvidenceRecord


def freeze_benchmark_suite(
    paths: ProjectPaths,
    result: BenchmarkSuiteResult,
    artifact_path: Path,
) -> list[EvidenceRecord]:
    artifact_hash = sha256_file(artifact_path)
    now = datetime.now(UTC).replace(microsecond=0)
    provenance = {
        "kind": "benchmark",
        "benchmark_run_id": result.benchmark_run_id,
        "suite_id": result.suite_id,
        "manifest_hash": result.manifest_hash,
        "artifact_path": artifact_path.relative_to(paths.root).as_posix(),
        "artifact_sha256": artifact_hash,
        "git_commit": result.git_commit,
        "experiment_ids": json.dumps(result.experiment_ids, sort_keys=True),
        "tournament_ids": json.dumps(result.tournament_ids, sort_keys=True),
    }

    def record(evidence_id: str, key: str, value: float | int | str) -> EvidenceRecord:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            experiment_id=result.benchmark_run_id,
            artifact_path=provenance["artifact_path"],
            artifact_hash=artifact_hash,
            source_field=key,
            validation_reports=["benchmark_suite"],
            verified_at=now,
            provenance=provenance,
        )

    run = result.benchmark_run_id
    records: list[EvidenceRecord] = [
        record(f"ev-{run}-suite-id", "benchmark.suite_id", result.suite_id),
        record(f"ev-{run}-manifest-hash", "benchmark.manifest_hash", result.manifest_hash),
        record(f"ev-{run}-n-instances", "benchmark.n_instances", result.n_instances),
        record(f"ev-{run}-status", "benchmark.status", result.status),
        record(
            f"ev-{run}-oracle-agreement",
            "benchmark.oracle_agreement",
            str(result.oracle_agreement).lower(),
        ),
        record(
            f"ev-{run}-candidate-feasible",
            "benchmark.candidate.feasible_count",
            result.candidate_feasible_count,
        ),
        record(
            f"ev-{run}-candidate-verified",
            "benchmark.candidate.verified_optimum_count",
            result.candidate_verified_optimum_count,
        ),
        record(
            f"ev-{run}-candidate-wins",
            "benchmark.candidate.win_count",
            result.candidate_win_count,
        ),
        record(
            f"ev-{run}-candidate-ties",
            "benchmark.candidate.tie_count",
            result.candidate_tie_count,
        ),
        record(
            f"ev-{run}-candidate-losses",
            "benchmark.candidate.loss_count",
            result.candidate_loss_count,
        ),
        record(
            f"ev-{run}-worst-instance",
            "benchmark.worst_case_instance",
            result.worst_case_instance or "none",
        ),
    ]
    for label, aggregate in (
        ("greedy", result.greedy),
        ("enhanced_greedy", result.enhanced_greedy),
    ):
        records.append(
            record(
                f"ev-{run}-{label}-hit-rate",
                f"benchmark.{label}.hit_rate",
                aggregate.hit_rate,
            )
        )
        records.append(
            record(
                f"ev-{run}-{label}-failures",
                f"benchmark.{label}.failures",
                aggregate.failures,
            )
        )
        for field in (
            "mean_relative_regret",
            "median_relative_regret",
            "max_relative_regret",
            "mean_absolute_regret",
            "max_absolute_regret",
        ):
            value = getattr(aggregate, field)
            if value is not None:
                records.append(
                    record(
                        f"ev-{run}-{label}-{field}",
                        f"benchmark.{label}.{field}",
                        value,
                    )
                )
        if aggregate.worst_relative_regret is not None:
            records.append(
                record(
                    f"ev-{run}-{label}-worst",
                    f"benchmark.{label}.worst_relative_regret",
                    aggregate.worst_relative_regret,
                )
            )
    for instance in result.per_instance:
        records.extend(
            [
                record(
                    f"ev-{run}-{instance.instance_id}-hash",
                    f"benchmark.instance.{instance.instance_id}.hash",
                    instance.instance_hash,
                ),
                record(
                    f"ev-{run}-{instance.instance_id}-oracle",
                    f"benchmark.instance.{instance.instance_id}.oracle_optimum",
                    instance.oracle_optimum,
                ),
                record(
                    f"ev-{run}-{instance.instance_id}-verified",
                    f"benchmark.instance.{instance.instance_id}.cbc_verified",
                    str(instance.cbc_verified).lower(),
                ),
            ]
        )
        if instance.greedy_objective is not None:
            records.append(
                record(
                    f"ev-{run}-{instance.instance_id}-greedy",
                    f"benchmark.instance.{instance.instance_id}.greedy_objective",
                    instance.greedy_objective,
                )
            )
        if instance.enhanced_greedy_objective is not None:
            records.append(
                record(
                    f"ev-{run}-{instance.instance_id}-enhanced",
                    f"benchmark.instance.{instance.instance_id}.enhanced_greedy_objective",
                    instance.enhanced_greedy_objective,
                )
            )
        if instance.tournament_id is not None:
            records.append(
                record(
                    f"ev-{run}-{instance.instance_id}-tournament",
                    f"benchmark.instance.{instance.instance_id}.tournament_id",
                    instance.tournament_id,
                )
            )

    index = load_evidence_index(paths)
    for evidence in records:
        index.records[evidence.evidence_id] = evidence
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    benchmark_dir = paths.verified_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        benchmark_dir / f"{run}.json",
        json.dumps(
            {
                "benchmark_run_id": run,
                "artifact_hash": artifact_hash,
                "records": [evidence.model_dump(mode="json") for evidence in records],
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
    provenance_index["experiments"][f"benchmark:{run}"] = {
        **provenance,
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")
    return records
