from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.evidence.store import EvidenceError, load_evidence_index
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.inference import (
    AcceptanceSamplingDesignResult,
    BinomialInferenceResult,
    NonlinearLeastSquaresResult,
)
from mmos.validation.inference import (
    validate_acceptance_sampling,
    validate_binomial_inference,
    validate_nls,
)

InferenceKind = Literal["binomial", "acceptance", "nls"]


def _validator(kind: InferenceKind):
    return {
        "binomial": validate_binomial_inference,
        "acceptance": validate_acceptance_sampling,
        "nls": validate_nls,
    }[kind]


def validate_and_register_inference(
    paths: ProjectPaths, experiment_id: str, kind: InferenceKind
) -> list[EvidenceRecord]:
    report = _validator(kind)(paths, experiment_id)
    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(validation_dir / "inference.json", report.model_dump_json(indent=2) + "\n")
    summary = {
        "experiment_id": experiment_id,
        "kind": kind,
        "blocking": report.blocking,
        "reports": {
            "inference": {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
        },
    }
    atomic_write_text(validation_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    if report.blocking:
        raise EvidenceError(f"inference validation failed for {experiment_id}")
    artifact = paths.experiment_dir(experiment_id) / "artifacts" / "inference_result.json"
    spec = paths.experiment_dir(experiment_id) / "inference_spec.json"
    artifact_hash = sha256_file(artifact)
    now = datetime.now(UTC).replace(microsecond=0)
    relative_artifact = artifact.relative_to(paths.root).as_posix()
    provenance = {
        "kind": f"inference_{kind}",
        "spec_path": spec.relative_to(paths.root).as_posix(),
        "spec_sha256": sha256_file(spec),
        "artifact_sha256": artifact_hash,
        "validator": "inference",
    }
    records: list[EvidenceRecord] = []

    def add(evidence_id: str, key: str, value, unit: str | None = None) -> None:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{experiment_id}-{evidence_id}",
                key=f"inference.{kind}.{key}",
                raw_value=value,
                display_value=display,
                unit=unit,
                experiment_id=experiment_id,
                artifact_path=relative_artifact,
                artifact_hash=artifact_hash,
                source_field=key,
                validation_reports=["inference"],
                verified_at=now,
                provenance=provenance,
            )
        )

    if kind == "binomial":
        result = BinomialInferenceResult.model_validate_json(artifact.read_text(encoding="utf-8"))
        add("proportion", "observed_proportion", result.observed_proportion)
        add("p-value", "p_value", result.p_value)
        add("ci-lower", "confidence_interval_lower", result.confidence_interval_lower)
        add("ci-upper", "confidence_interval_upper", result.confidence_interval_upper)
        add("confidence-level", "confidence_level", result.confidence_level)
        add("alternative", "alternative", result.alternative.value)
        add("confidence-method", "confidence_method", result.confidence_method)
    elif kind == "acceptance":
        result = AcceptanceSamplingDesignResult.model_validate_json(
            artifact.read_text(encoding="utf-8")
        )
        add(
            "sample-size",
            "sample_size",
            result.sample_size if result.sample_size is not None else "none",
        )
        add(
            "acceptance-number",
            "acceptance_number",
            result.acceptance_number if result.acceptance_number is not None else "none",
        )
        add("p-accept-good", "probability_accept_good", result.probability_accept_good or 0.0)
        add("p-accept-bad", "probability_accept_bad", result.probability_accept_bad or 0.0)
        add("producer-risk-actual", "producer_risk_actual", result.producer_risk_actual or 0.0)
        add("consumer-risk-actual", "consumer_risk_actual", result.consumer_risk_actual or 0.0)
        add("search-bound", "search_bound", result.search_bound)
        add("feasible", "feasible", str(result.feasible).lower())
        add("oc-grid-hash", "oc_grid_hash", result.oc_grid_hash)
    else:
        result = NonlinearLeastSquaresResult.model_validate_json(
            artifact.read_text(encoding="utf-8")
        )
        for name, value in zip(result.parameter_names, result.parameter_estimates, strict=False):
            add(f"parameter-{name}", f"parameter_estimate.{name}", value)
        add("sse", "sse", result.sse)
        add("rmse", "rmse", result.rmse)
        add("jacobian-rank", "jacobian_rank", result.jacobian_rank)
        add(
            "condition-number",
            "jacobian_condition_number",
            result.jacobian_condition_number
            if result.jacobian_condition_number is not None
            else "none",
        )
        add("selected-start", "selected_start", result.selected_start)
        add("start-count", "start_count", len(result.all_start_results))
        add("uncertainty-status", "uncertainty_status", result.uncertainty_status)
        add("forward-model-hash", "forward_model_hash", result.forward_model_hash)
        for interval in result.parameter_intervals:
            if interval.standard_error is None:
                continue
            add(
                f"se-{interval.name}",
                f"standard_error.{interval.name}",
                interval.standard_error,
            )
            if interval.lower is not None and interval.upper is not None:
                add(
                    f"ci-lower-{interval.name}",
                    f"ci_lower.{interval.name}",
                    interval.lower,
                )
                add(
                    f"ci-upper-{interval.name}",
                    f"ci_upper.{interval.name}",
                    interval.upper,
                )

    index = load_evidence_index(paths)
    for record in records:
        if record.evidence_id in index.records:
            raise EvidenceError(f"evidence id already exists: {record.evidence_id}")
        index.records[record.evidence_id] = record
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    return records


__all__ = ["InferenceKind", "validate_and_register_inference"]
