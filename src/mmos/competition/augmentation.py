import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mmos.competition.context import load_context
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_spec,
    save_experiment_record,
    save_experiment_spec,
    save_optimization_spec,
    sha256_file,
)
from mmos.experiments.optimization_runner import (
    build_experiment_record,
    run_optimization,
)
from mmos.experiments.workflow import run_experiment_workflow
from mmos.mixed.artifacts import (
    contract_for,
    contract_for_forecast_error,
    contract_for_production_plan,
    load_envelope,
    verify_forecast_artifact,
    verify_forecast_error_artifact,
    verify_production_plan_artifact,
)
from mmos.mixed.simulation import (
    build_mixed_simulation_record,
    load_mixed_simulation_spec,
    run_mixed_simulation,
    save_mixed_simulation_record,
    save_mixed_simulation_spec,
)
from mmos.mixed.validation import (
    _check_simulation_metrics,
    _check_simulation_replications,
)
from mmos.schemas.competition import (
    CompetitionAugmentationRun,
    CompetitionExperimentManifest,
    CompetitionExperimentRequest,
    CompetitionPaperRun,
    ExecutionStatus,
    RequestExecution,
    RequestResolution,
    ResolutionClass,
)
from mmos.schemas.data import Severity
from mmos.schemas.evidence import EvidenceIndex, EvidenceRecord
from mmos.schemas.experiment import ExperimentResult, TemporalSplit
from mmos.schemas.mixed import MixedExecutionPlan
from mmos.schemas.optimization import (
    ExperimentStatus,
    OptimizationModel,
    optimization_instance_hash,
)
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.tracking import (
    log_experiment_run,
    log_mixed_simulation_run,
    log_optimization_run,
)
from mmos.validation import validate_experiment
from mmos.validation.base import build_report
from mmos.validation.optimization import validate_optimization_experiment


class CompetitionAugmentationError(RuntimeError):
    pass


class CompetitionAugmentationBlocked(CompetitionAugmentationError):
    pass


@dataclass(frozen=True)
class AdapterOutcome:
    status: ExecutionStatus
    experiment_ids: list[str]
    validation_status: dict[str, str]
    records: list[EvidenceRecord]
    error: str | None = None


@dataclass(frozen=True)
class AugmentationSummary:
    run: CompetitionAugmentationRun
    new_paper: object | None
    evidence_records: list[EvidenceRecord]


def _timestamp() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _hash_model(model) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _source_dir(paths: ProjectPaths, source_run_id: str) -> Path:
    comp = paths.root / "paper" / "competition"
    history = comp / "history" / source_run_id
    if (history / "run.json").exists():
        return history
    current = comp / "run.json"
    if current.exists():
        payload = json.loads(current.read_text(encoding="utf-8"))
        if payload.get("run_id") == source_run_id:
            return comp
    raise CompetitionAugmentationError(
        f"source competition run {source_run_id!r} not found"
    )


def _load_source(
    paths: ProjectPaths, source_run_id: str
) -> tuple[Path, CompetitionPaperRun, CompetitionExperimentManifest]:
    source_dir = _source_dir(paths, source_run_id)
    run = CompetitionPaperRun.model_validate_json(
        (source_dir / "run.json").read_text(encoding="utf-8")
    )
    manifest = CompetitionExperimentManifest.model_validate_json(
        (source_dir / "experiment_requests.json").read_text(encoding="utf-8")
    )
    release_path = paths.root / run.source_release
    if not release_path.exists():
        raise CompetitionAugmentationBlocked("source release is missing")
    if sha256_file(release_path) != run.source_release_hash:
        raise CompetitionAugmentationBlocked(
            "source release hash changed; augmentation is stale"
        )
    if _hash_model(manifest) != run.request_manifest_hash:
        raise CompetitionAugmentationBlocked(
            "request manifest hash changed; augmentation is stale"
        )
    return source_dir, run, manifest


def resolve_requests(
    manifest: CompetitionExperimentManifest,
) -> list[RequestResolution]:
    resolutions: list[RequestResolution] = []
    for request in manifest.requests:
        text = " ".join(
            [
                request.question,
                request.rationale,
                request.proposed_experiment,
                request.expected_evidence,
            ]
        ).lower()
        classification = ResolutionClass.REQUIRES_HUMAN_DECISION
        kind: str | None = None
        reason = "No deterministic mapping to an implemented experiment capability."
        if request.executed or not request.supporting_evidence_ids:
            classification = (
                ResolutionClass.INVALID
                if request.executed
                else ResolutionClass.NOT_JUSTIFIED
            )
            reason = (
                "The request claims execution or lacks evidence support."
            )
        elif request.capability.value == "unsupported":
            classification = ResolutionClass.REQUIRES_NEW_CAPABILITY
            reason = "The request names an unimplemented capability."
        elif request.capability.value == "forecasting":
            if any(word in text for word in ("damped", "exponential smoothing")):
                classification = ResolutionClass.REQUIRES_NEW_CAPABILITY
                reason = (
                    "Damped-trend and exponential-smoothing forecasters are not "
                    "implemented."
                )
            elif any(
                phrase in text
                for phrase in ("step 1", "overlapping", "step size")
            ):
                classification = ResolutionClass.REQUIRES_NEW_CAPABILITY
                reason = (
                    "Overlapping rolling-origin windows (step < horizon) are "
                    "explicitly disallowed by the current TemporalSplit validator; "
                    "no overlap-aware validation exists."
                )
            elif any(
                word in text for word in ("configuration", "minimum training")
            ):
                classification = (
                    ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
                )
                kind = "forecast_config_sweep"
                reason = (
                    "The existing ExperimentSpec/workflow supports alternative "
                    "non-overlapping rolling-origin settings."
                )
            else:
                classification = ResolutionClass.REQUIRES_NEW_CAPABILITY
                reason = "Requested forecast candidates exceed the current engines."
        elif request.capability.value == "optimization":
            if any(
                word in text
                for word in (
                    "sensitivity",
                    "perturb",
                    "capacity",
                    "cost",
                    "holding",
                    "initial inventory",
                )
            ):
                classification = (
                    ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
                )
                kind = "optimization_sensitivity"
                reason = (
                    "The existing optimization runner can re-solve variant LP "
                    "instances generated from the frozen source model."
                )
            else:
                classification = ResolutionClass.REQUIRES_HUMAN_DECISION
                reason = "Optimization request needs a human-selected experiment design."
        elif request.capability.value == "simulation":
            if any(word in text for word in ("bootstrap", "interval")):
                classification = ResolutionClass.REQUIRES_HUMAN_DECISION
                reason = (
                    "Bootstrap/interval policy requires a human-chosen statistical "
                    "procedure and seed policy."
                )
            elif any(
                word in text
                for word in (
                    "stratified",
                    "block",
                    "common-shock",
                    "resampling scheme",
                    "demand floor",
                )
            ):
                classification = ResolutionClass.REQUIRES_NEW_CAPABILITY
                reason = (
                    "The frozen mixed simulation supports empirical iid resampling "
                    "only; alternative schemes are not implemented."
                )
            elif any(
                word in text
                for word in ("replication count", "seed", "stable", "convergence")
            ):
                classification = (
                    ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
                )
                kind = "simulation_replication_stability"
                reason = (
                    "The existing mixed simulation runner supports new replication "
                    "counts and seed schedules on the frozen plan."
                )
            else:
                classification = ResolutionClass.REQUIRES_HUMAN_DECISION
                reason = "Simulation request needs a human-selected diagnostic design."
        elif request.capability.value in {
            "multi_objective",
            "graph",
            "pipeline",
        }:
            classification = ResolutionClass.REQUIRES_NEW_CAPABILITY
            reason = (
                "Requested analysis needs a capability outside the implemented "
                "single-slice engines."
            )
        resolutions.append(
            RequestResolution(
                request_id=request.request_id,
                classification=classification,
                mapped_kind=kind,
                rationale=reason,
                blocker_reason=None
                if classification
                is ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY
                else reason,
                executable=classification
                is ResolutionClass.EXECUTABLE_EXISTING_CAPABILITY,
            )
        )
    return resolutions


def _provenance(
    source_run: CompetitionPaperRun,
    request: CompetitionExperimentRequest,
    aug_id: str,
    experiment_id: str,
    *,
    spec_path: Path,
    validation_path: Path,
    artifact_path: Path,
    mlflow_run_id: str,
) -> dict[str, str]:
    return {
        "kind": "competition_experiment",
        "augmentation_id": aug_id,
        "request_id": request.request_id,
        "request_manifest_hash": source_run.request_manifest_hash,
        "source_release": source_run.source_release,
        "source_release_hash": source_run.source_release_hash,
        "source_context_hash": source_run.context_hash,
        "source_competition_run_id": source_run.run_id,
        "experiment_id": experiment_id,
        "spec_path": spec_path.name,
        "spec_sha256": sha256_file(spec_path),
        "validation_path": validation_path.name,
        "validation_sha256": sha256_file(validation_path)
        if validation_path.exists()
        else "",
        "artifact_path": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path)
        if artifact_path.exists()
        else "",
        "mlflow_run_id": mlflow_run_id,
    }


def _record(
    *,
    aug_id: str,
    request: CompetitionExperimentRequest,
    source_run: CompetitionPaperRun,
    experiment_id: str,
    project_root: Path,
    key: str,
    value: float | int | str,
    unit: str | None,
    artifact_path: Path,
    artifact_hash: str,
    source_field: str,
    validators: list[str],
    provenance: dict[str, str],
) -> EvidenceRecord:
    display = f"{value:.10g}" if isinstance(value, float) else str(value)
    return EvidenceRecord(
        evidence_id=(
            f"ev-aug-{aug_id}-{request.request_id}-"
            f"{_slug(experiment_id + ':' + key)}"
        ),
        key=key,
        raw_value=value,
        display_value=display,
        unit=unit,
        experiment_id=experiment_id,
        artifact_path=artifact_path.relative_to(project_root).as_posix(),
        artifact_hash=artifact_hash,
        source_field=source_field,
        validation_reports=validators,
        verified_at=_timestamp(),
        provenance=provenance,
    )


def _write_validation(
    experiment_dir: Path,
    report: ValidationReport,
) -> Path:
    output = experiment_dir / "validation"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{report.validator}.json"
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    summary = {
        "experiment_id": report.subject_id,
        "blocking": report.blocking,
        "reports": {
            report.validator: {
                "status": report.status.value,
                "severity": report.severity.value,
                "blocking": report.blocking,
            }
        },
    }
    atomic_write_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    return path


def _execute_forecast_sweep(
    paths: ProjectPaths,
    context,
    request: CompetitionExperimentRequest,
    source_run: CompetitionPaperRun,
    source_dir: Path,
    aug_id: str,
) -> AdapterOutcome:
    release = context.release
    parent_id = str(release["forecast_experiment_id"])
    base = load_experiment_spec(paths, parent_id)
    variants = [
        {"step": 1, "n_splits": 10},
        {"step": 2, "n_splits": 6},
    ]
    records: list[EvidenceRecord] = []
    experiment_ids: list[str] = []
    validation_status: dict[str, str] = {}
    for index, variant in enumerate(variants, start=1):
        exp_id = f"exp-{aug_id}-{request.request_id}-forecast-{index}"
        split = base.temporal_split
        if split is None:
            return AdapterOutcome(
                ExecutionStatus.REJECTED,
                experiment_ids,
                validation_status,
                records,
                "source experiment has no temporal split",
            )
        spec = base.model_copy(
            update={
                "experiment_id": exp_id,
                "proposal_id": "competition-augmentation",
                "n_splits": int(variant["n_splits"]),
                "temporal_split": TemporalSplit(
                    horizon=split.horizon,
                    min_train_size=split.min_train_size,
                    step=int(variant["step"]),
                    n_splits=int(variant["n_splits"]),
                ),
                "acceptance_criteria": [
                    "competition augmentation re-run under the frozen dataset"
                ],
            }
        )
        save_experiment_spec(paths, spec)
        execution = run_experiment_workflow(paths, exp_id)
        if not execution.succeeded():
            return AdapterOutcome(
                ExecutionStatus.FAILED,
                experiment_ids,
                validation_status,
                records,
                f"workflow failed for {exp_id}",
            )
        reports = validate_experiment(paths, exp_id)
        validation_status.update(
            {f"{exp_id}:{report.validator}": report.status.value for report in reports}
        )
        if any(report.blocking for report in reports):
            return AdapterOutcome(
                ExecutionStatus.REJECTED,
                experiment_ids,
                validation_status,
                records,
                f"validation failed for {exp_id}",
            )
        metrics_path = paths.experiment_dir(exp_id) / "artifacts" / "metrics.json"
        summary_path = paths.experiment_dir(exp_id) / "validation" / "summary.json"
        result = ExperimentResult.model_validate_json(
            metrics_path.read_text(encoding="utf-8")
        )
        mlflow = log_experiment_run(paths, spec, result)
        experiment_ids.append(exp_id)
        spec_path = paths.experiment_dir(exp_id) / "experiment_spec.json"
        provenance = _provenance(
            source_run,
            request,
            aug_id,
            exp_id,
            spec_path=spec_path,
            validation_path=summary_path,
            artifact_path=metrics_path,
            mlflow_run_id=mlflow.run_id,
        )
        for model_id, metrics in result.models.items():
            for metric, value in metrics.items():
                records.append(
                    _record(
                        aug_id=aug_id,
                        request=request,
                        source_run=source_run,
                        experiment_id=exp_id,
                        project_root=paths.root,
                        key=f"competition.forecast.{exp_id}.{model_id}.{metric}",
                        value=value,
                        unit=None,
                        artifact_path=metrics_path,
                        artifact_hash=sha256_file(metrics_path),
                        source_field=f"models.{model_id}.{metric}",
                        validators=[report.validator for report in reports],
                        provenance=provenance,
                    )
                )
    return AdapterOutcome(
        ExecutionStatus.VALIDATED,
        experiment_ids,
        validation_status,
        records,
    )


def _variant_model(
    model: OptimizationModel, variant: str
) -> OptimizationModel:
    variables = [variable.model_copy(deep=True) for variable in model.variables]
    objective = dict(model.objective)
    constraints = [constraint.model_copy(deep=True) for constraint in model.constraints]
    if variant in {"capacity_plus_10", "capacity_minus_5", "capacity_minus_10"}:
        factor = (
            1.1
            if variant == "capacity_plus_10"
            else 0.95
            if variant == "capacity_minus_5"
            else 0.9
        )
        for variable in variables:
            if variable.name.startswith("production_"):
                variable.upper *= factor
        for constraint in constraints:
            if constraint.name.startswith("capacity_") and constraint.upper is not None:
                constraint.upper *= factor
    elif variant == "holding_x2":
        objective = {
            name: value * 2 if name.startswith("inventory_") else value
            for name, value in objective.items()
        }
    elif variant == "production_cost_x2":
        objective = {
            name: value * 2 if name.startswith("production_") else value
            for name, value in objective.items()
        }
    elif variant == "initial_plus_5":
        for constraint in constraints:
            if constraint.name == "inventory_balance_0":
                if constraint.lower is not None:
                    constraint.lower -= 5.0
                if constraint.upper is not None:
                    constraint.upper -= 5.0
    return OptimizationModel(
        name=model.name + "_" + variant,
        sense=model.sense,
        variables=variables,
        objective=objective,
        constraints=constraints,
    )


def _execute_optimization_sensitivity(
    paths: ProjectPaths,
    context,
    request: CompetitionExperimentRequest,
    source_run: CompetitionPaperRun,
    source_dir: Path,
    aug_id: str,
) -> AdapterOutcome:
    from mmos.core.project import load_optimization_spec

    parent_id = str(context.release["optimization_experiment_id"])
    base = load_optimization_spec(paths, parent_id)
    variants = (
        ["capacity_plus_10", "capacity_minus_5", "holding_x2"]
        if request.request_id in {"request-3"}
        else ["capacity_minus_10", "production_cost_x2", "initial_plus_5"]
    )
    records: list[EvidenceRecord] = []
    experiment_ids: list[str] = []
    validation_status: dict[str, str] = {}
    errors: list[str] = []
    for index, variant in enumerate(variants, start=1):
        exp_id = f"exp-{aug_id}-{request.request_id}-optimization-{index}"
        model = _variant_model(base.model, variant)
        spec = base.model_copy(
            update={
                "experiment_id": exp_id,
                "model": model,
                "baseline": "none",
                "instance_hash": optimization_instance_hash(model),
            }
        )
        save_optimization_spec(paths, spec)
        result = run_optimization(paths, exp_id)
        solution = next(
            (item for item in result.solutions if item.method == "ortools.cbc"),
            None,
        )
        if solution is None or not solution.feasible:
            errors.append(f"{exp_id}: variant {variant} is infeasible; failure retained")
            continue
        record = build_experiment_record(
            paths,
            spec,
            result,
            status=ExperimentStatus.SUCCEEDED,
        )
        mlflow = log_optimization_run(paths, spec, result, record)
        record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
        save_experiment_record(paths, record)
        reports = validate_optimization_experiment(paths, exp_id)
        validation_status.update(
            {f"{exp_id}:{report.validator}": report.status.value for report in reports}
        )
        if any(report.blocking for report in reports):
            errors.append(f"{exp_id}: optimization validation failed")
            continue
        record = record.model_copy(
            update={
                "status": ExperimentStatus.VALIDATED,
                "validation_status": {
                    report.validator: report.status.value for report in reports
                },
            }
        )
        save_experiment_record(paths, record)
        experiment_ids.append(exp_id)
        result_path = paths.experiment_dir(exp_id) / "artifacts" / "optimization_result.json"
        spec_path = paths.experiment_dir(exp_id) / "optimization_spec.json"
        summary_path = paths.experiment_dir(exp_id) / "validation" / "summary.json"
        provenance = _provenance(
            source_run,
            request,
            aug_id,
            exp_id,
            spec_path=spec_path,
            validation_path=summary_path,
            artifact_path=result_path,
            mlflow_run_id=mlflow.run_id,
        )
        values = [
            ("objective", solution.objective_value),
            ("feasible", str(solution.feasible).lower()),
            *[(name, value) for name, value in solution.variable_values.items()],
        ]
        for name, value in values:
            if value is None:
                continue
            records.append(
                _record(
                    aug_id=aug_id,
                    request=request,
                    source_run=source_run,
                    experiment_id=exp_id,
                    project_root=paths.root,
                    key=f"competition.optimization.{exp_id}.{variant}.{name}",
                    value=value,
                    unit="cost_units" if name == "objective" else None,
                    artifact_path=result_path,
                    artifact_hash=sha256_file(result_path),
                    source_field=name,
                    validators=[report.validator for report in reports],
                    provenance=provenance,
                )
            )
    return AdapterOutcome(
        ExecutionStatus.VALIDATED if records else ExecutionStatus.REJECTED,
        experiment_ids,
        validation_status,
        records,
        "; ".join(errors) or None,
    )


def _execute_simulation_stability(
    paths: ProjectPaths,
    context,
    request: CompetitionExperimentRequest,
    source_run: CompetitionPaperRun,
    source_dir: Path,
    aug_id: str,
) -> AdapterOutcome:
    release = context.release
    parent_id = str(release["simulation_experiment_id"])
    parent_spec = load_mixed_simulation_spec(paths, parent_id)
    run_id = str(release["run_id"])
    artifact_dir = paths.mixed_dir / "runs" / run_id / "artifacts"
    plan = MixedExecutionPlan.model_validate_json(
        (paths.mixed_dir / "plan.json").read_text(encoding="utf-8")
    )
    forecast_env = load_envelope(artifact_dir / "forecast_envelope.json")
    error_env = load_envelope(artifact_dir / "forecast_error_envelope.json")
    plan_env = load_envelope(artifact_dir / "production_plan_envelope.json")
    forecast = verify_forecast_artifact(
        paths, forecast_env, contract_for(plan.spec)
    )
    error_artifact = verify_forecast_error_artifact(
        paths,
        error_env,
        contract_for_forecast_error(
            plan.spec,
            forecast.forecast_experiment_id,
            forecast_env.content_hash,
        ),
    )
    plan_artifact = verify_production_plan_artifact(
        paths,
        plan_env,
        contract_for_production_plan(
            plan.spec, forecast_env.artifact_id, forecast_env.content_hash
        ),
    )
    variants = [(200, 20261117), (1000, 20261118)]
    records: list[EvidenceRecord] = []
    experiment_ids: list[str] = []
    validation_status: dict[str, str] = {}
    for index, (replications, seed) in enumerate(variants, start=1):
        exp_id = f"exp-{aug_id}-{request.request_id}-simulation-{index}"
        spec = parent_spec.model_copy(
            update={
                "simulation_experiment_id": exp_id,
                "simulation_id": f"sim-{aug_id}-{request.request_id}-{index}",
                "replications": replications,
                "master_seed": seed,
                "spec_hash": "",
            }
        ).with_hash()
        save_mixed_simulation_spec(paths, spec)
        result = run_mixed_simulation(
            paths, spec, plan_artifact, forecast, error_artifact
        )
        record = build_mixed_simulation_record(
            paths, spec, result, status=ExperimentStatus.SUCCEEDED
        )
        mlflow = log_mixed_simulation_run(paths, spec, result, record)
        record = record.model_copy(update={"mlflow_run_id": mlflow.run_id})
        save_mixed_simulation_record(paths, record)
        issues: list[ValidationIssue] = []

        def error(
            code: str,
            message: str,
            target_issues: list[ValidationIssue] = issues,
        ) -> None:
            target_issues.append(
                ValidationIssue(code=code, severity=Severity.ERROR, message=message)
            )

        _check_simulation_replications(
            spec,
            result,
            forecast,
            plan_artifact,
            [item.residual for item in error_artifact.residuals],
            error,
        )
        _check_simulation_metrics(spec, result, error)
        report = build_report(
            "mixed_simulation", exp_id, issues,
            evidence=[str(paths.experiment_dir(exp_id))],
        )
        validation_path = _write_validation(paths.experiment_dir(exp_id), report)
        validation_status[f"{exp_id}:mixed_simulation"] = report.status.value
        if report.blocking:
            return AdapterOutcome(
                ExecutionStatus.REJECTED,
                experiment_ids,
                validation_status,
                records,
                f"simulation validation failed for {exp_id}",
            )
        experiment_ids.append(exp_id)
        result_path = (
            paths.experiment_dir(exp_id)
            / "artifacts"
            / "mixed_simulation_result.json"
        )
        spec_path = paths.experiment_dir(exp_id) / "mixed_simulation_spec.json"
        provenance = _provenance(
            source_run,
            request,
            aug_id,
            exp_id,
            spec_path=spec_path,
            validation_path=validation_path,
            artifact_path=result_path,
            mlflow_run_id=mlflow.run_id,
        )
        for metric, summary in result.metrics.items():
            records.append(
                _record(
                    aug_id=aug_id,
                    request=request,
                    source_run=source_run,
                    experiment_id=exp_id,
                    project_root=paths.root,
                    key=f"competition.simulation.{exp_id}.{metric}.mean",
                    value=summary.mean,
                    unit=result.unit,
                    artifact_path=result_path,
                    artifact_hash=sha256_file(result_path),
                    source_field=f"metrics.{metric}.mean",
                    validators=[report.validator],
                    provenance=provenance,
                )
            )
        records.append(
            _record(
                aug_id=aug_id,
                request=request,
                source_run=source_run,
                experiment_id=exp_id,
                project_root=paths.root,
                key=f"competition.simulation.{exp_id}.any_shortage_probability",
                value=result.any_shortage_probability,
                unit=None,
                artifact_path=result_path,
                artifact_hash=sha256_file(result_path),
                source_field="any_shortage_probability",
                validators=[report.validator],
                provenance=provenance,
            )
        )
    return AdapterOutcome(
        ExecutionStatus.VALIDATED,
        experiment_ids,
        validation_status,
        records,
    )


DEFAULT_ADAPTERS = {
    "forecast_config_sweep": _execute_forecast_sweep,
    "optimization_sensitivity": _execute_optimization_sensitivity,
    "simulation_replication_stability": _execute_simulation_stability,
}


def load_registered_evidence(
    paths: ProjectPaths, aug_id: str
) -> tuple[list[EvidenceRecord], Path]:
    store_path = paths.root / "paper" / "competition" / "augmentation" / aug_id / "evidence.json"
    if not store_path.exists():
        raise CompetitionAugmentationError("augmentation evidence store is missing")
    index = EvidenceIndex.model_validate_json(
        store_path.read_text(encoding="utf-8")
    )
    for record in index.records.values():
        artifact = paths.root / record.artifact_path
        if not artifact.exists():
            raise CompetitionAugmentationError(
                f"augmentation artifact is missing: {record.artifact_path}"
            )
        if sha256_file(artifact) != record.artifact_hash:
            raise CompetitionAugmentationError(
                f"augmentation artifact hash changed: {record.artifact_path}"
            )
    return list(index.records.values()), store_path


def run_augmentation(
    paths: ProjectPaths,
    agents,
    reviewer,
    source_run_id: str,
) -> AugmentationSummary:
    source_dir, source_run, manifest = _load_source(paths, source_run_id)
    context = load_context(paths)
    release_path = paths.root / source_run.source_release
    release_hash_before = sha256_file(release_path)
    verified_hash_before = (
        sha256_file(paths.verified_numbers_file)
        if paths.verified_numbers_file.exists()
        else ""
    )
    aug_id = f"aug-{_hash_model(manifest)[:8]}-{secrets.token_hex(2)}"
    out_dir = paths.root / "paper" / "competition" / "augmentation" / aug_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_dir / "request_manifest.json",
        manifest.model_dump_json(indent=2) + "\n",
    )
    resolutions = resolve_requests(manifest)
    resolution_by_id = {item.request_id: item for item in resolutions}
    executions: list[RequestExecution] = []
    evidence: list[EvidenceRecord] = []
    for request in manifest.requests:
        resolution = resolution_by_id[request.request_id]
        if not resolution.executable:
            executions.append(
                RequestExecution(
                    request_id=request.request_id,
                    status=ExecutionStatus.BLOCKED,
                    error=resolution.blocker_reason,
                )
            )
            continue
        adapter = DEFAULT_ADAPTERS.get(resolution.mapped_kind or "")
        if adapter is None:
            executions.append(
                RequestExecution(
                    request_id=request.request_id,
                    status=ExecutionStatus.BLOCKED,
                    error=f"no adapter for {resolution.mapped_kind}",
                )
            )
            continue
        try:
            outcome = adapter(
                paths, context, request, source_run, source_dir, aug_id
            )
        except Exception as exc:
            executions.append(
                RequestExecution(
                    request_id=request.request_id,
                    status=ExecutionStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        registered = outcome.status is ExecutionStatus.VALIDATED and bool(
            outcome.records
        )
        if registered:
            evidence.extend(outcome.records)
        executions.append(
            RequestExecution(
                request_id=request.request_id,
                status=ExecutionStatus.EVIDENCE_REGISTERED
                if registered
                else outcome.status,
                experiment_ids=outcome.experiment_ids,
                validation_status=outcome.validation_status,
                evidence_ids=[record.evidence_id for record in outcome.records],
                error=outcome.error,
                executed=bool(outcome.experiment_ids),
                validated=outcome.status is ExecutionStatus.VALIDATED,
                evidence_registered=registered,
            )
        )

    evidence_index = EvidenceIndex(
        records={record.evidence_id: record for record in evidence}
    )
    evidence_path = out_dir / "evidence.json"
    atomic_write_text(evidence_path, evidence_index.model_dump_json(indent=2) + "\n")
    evidence_hash = sha256_file(evidence_path)
    atomic_write_text(
        out_dir / "request_resolutions.json",
        json.dumps(
            [item.model_dump(mode="json") for item in resolutions], indent=2
        )
        + "\n",
    )
    atomic_write_text(
        out_dir / "executions.json",
        json.dumps(
            [item.model_dump(mode="json") for item in executions], indent=2
        )
        + "\n",
    )

    new_paper = None
    paper_error: str | None = None
    if evidence:
        extra, _ = load_registered_evidence(paths, aug_id)
        from mmos.competition_pipeline import (
            CompetitionPaperBlocked,
            run_competition_paper,
        )

        try:
            new_paper = run_competition_paper(
                paths,
                agents,
                reviewer,
                extra_evidence=extra,
                augmentation_id=aug_id,
            )
        except CompetitionPaperBlocked as exc:
            paper_error = str(exc)
        except Exception as exc:
            paper_error = f"{type(exc).__name__}: {exc}"

    release_hash_after = sha256_file(release_path)
    verified_hash_after = (
        sha256_file(paths.verified_numbers_file)
        if paths.verified_numbers_file.exists()
        else ""
    )
    if (
        release_hash_before != release_hash_after
        or verified_hash_before != verified_hash_after
    ):
        raise CompetitionAugmentationBlocked(
            "source release or verified evidence changed during augmentation"
        )
    registered = any(item.evidence_registered for item in executions)
    status = (
        "completed"
        if registered and new_paper is not None
        else "partial"
        if registered
        else "blocked"
    )
    run = CompetitionAugmentationRun(
        augmentation_id=aug_id,
        source_project_id=paths.root.name,
        source_release=source_run.source_release,
        source_release_hash_before=release_hash_before,
        source_release_hash_after=release_hash_after,
        source_verified_numbers_hash_before=verified_hash_before,
        source_verified_numbers_hash_after=verified_hash_after,
        source_context_hash=source_run.context_hash,
        source_competition_run_id=source_run.run_id,
        request_manifest_hash=source_run.request_manifest_hash,
        resolutions=resolutions,
        executions=executions,
        new_evidence_ids=[record.evidence_id for record in evidence],
        evidence_store_path=evidence_path.relative_to(paths.root).as_posix(),
        evidence_store_hash=evidence_hash,
        new_competition_run_id=new_paper.run.run_id if new_paper else None,
        new_paper_path=new_paper.run.paper_path if new_paper else None,
        new_release_path=new_paper.run.release_path if new_paper else None,
        new_paper_error=paper_error,
        status=status,
        created_at=_timestamp(),
    )
    atomic_write_text(
        out_dir / "augmentation_run.json", run.model_dump_json(indent=2) + "\n"
    )
    return AugmentationSummary(
        run=run, new_paper=new_paper, evidence_records=evidence
    )


def resume_augmentation(
    paths: ProjectPaths,
    agents,
    reviewer,
    aug_id: str,
    source_run_id: str,
) -> AugmentationSummary:
    out_dir = paths.root / "paper" / "competition" / "augmentation" / aug_id
    if not out_dir.exists():
        raise CompetitionAugmentationError(f"augmentation {aug_id!r} not found")
    resolutions = [
        RequestResolution.model_validate(item)
        for item in json.loads(
            (out_dir / "request_resolutions.json").read_text(encoding="utf-8")
        )
    ]
    executions = [
        RequestExecution.model_validate(item)
        for item in json.loads((out_dir / "executions.json").read_text(encoding="utf-8"))
    ]
    evidence, evidence_path = load_registered_evidence(paths, aug_id)
    if not evidence:
        raise CompetitionAugmentationError(
            "augmentation has no registered evidence to build a paper from"
        )
    provenance = evidence[0].provenance
    for record in evidence:
        if record.provenance.get("source_release_hash") != provenance.get(
            "source_release_hash"
        ):
            raise CompetitionAugmentationBlocked(
                "augmentation evidence mixes source releases"
            )
    source_release = str(provenance.get("source_release", ""))
    source_release_hash = str(provenance.get("source_release_hash", ""))
    source_context_hash = str(provenance.get("source_context_hash", ""))
    source_competition_run_id = str(
        provenance.get("source_competition_run_id", source_run_id)
    )
    request_manifest_hash = str(provenance.get("request_manifest_hash", ""))
    if not source_release:
        raise CompetitionAugmentationError("evidence has no source release metadata")
    release_path = paths.root / source_release
    if not release_path.exists():
        raise CompetitionAugmentationBlocked("source release is missing")
    if sha256_file(release_path) != source_release_hash:
        raise CompetitionAugmentationBlocked(
            "source release hash changed; augmentation is stale"
        )
    manifest_snapshot = out_dir / "request_manifest.json"
    if manifest_snapshot.exists():
        manifest = CompetitionExperimentManifest.model_validate_json(
            manifest_snapshot.read_text(encoding="utf-8")
        )
        if _hash_model(manifest) != request_manifest_hash:
            raise CompetitionAugmentationBlocked(
                "request manifest snapshot changed; augmentation is stale"
            )
    release_before = sha256_file(release_path)
    verified_before = (
        sha256_file(paths.verified_numbers_file)
        if paths.verified_numbers_file.exists()
        else ""
    )
    new_paper = None
    paper_error: str | None = None
    from mmos.competition_pipeline import (
        CompetitionPaperBlocked,
        run_competition_paper,
    )

    try:
        new_paper = run_competition_paper(
            paths,
            agents,
            reviewer,
            extra_evidence=evidence,
            augmentation_id=aug_id,
        )
    except CompetitionPaperBlocked as exc:
        paper_error = str(exc)
    except Exception as exc:
        paper_error = f"{type(exc).__name__}: {exc}"
    release_after = sha256_file(release_path)
    verified_after = (
        sha256_file(paths.verified_numbers_file)
        if paths.verified_numbers_file.exists()
        else ""
    )
    if release_before != release_after or verified_before != verified_after:
        raise CompetitionAugmentationBlocked(
            "source release or verified evidence changed during augmentation"
        )
    run = CompetitionAugmentationRun(
        augmentation_id=aug_id,
        source_project_id=paths.root.name,
        source_release=source_release,
        source_release_hash_before=release_before,
        source_release_hash_after=release_after,
        source_verified_numbers_hash_before=verified_before,
        source_verified_numbers_hash_after=verified_after,
        source_context_hash=source_context_hash,
        source_competition_run_id=source_competition_run_id,
        request_manifest_hash=request_manifest_hash,
        resolutions=resolutions,
        executions=executions,
        new_evidence_ids=[record.evidence_id for record in evidence],
        evidence_store_path=evidence_path.relative_to(paths.root).as_posix(),
        evidence_store_hash=sha256_file(evidence_path),
        new_competition_run_id=new_paper.run.run_id if new_paper else None,
        new_paper_path=new_paper.run.paper_path if new_paper else None,
        new_release_path=new_paper.run.release_path if new_paper else None,
        new_paper_error=paper_error,
        status="completed" if new_paper is not None else "partial",
        created_at=_timestamp(),
    )
    atomic_write_text(
        out_dir / "augmentation_run.json", run.model_dump_json(indent=2) + "\n"
    )
    return AugmentationSummary(
        run=run, new_paper=new_paper, evidence_records=evidence
    )


__all__ = [
    "DEFAULT_ADAPTERS",
    "AdapterOutcome",
    "AugmentationSummary",
    "CompetitionAugmentationBlocked",
    "CompetitionAugmentationError",
    "load_registered_evidence",
    "resolve_requests",
    "resume_augmentation",
    "run_augmentation",
]
