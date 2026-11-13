import json
import secrets
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
    load_tournament_result,
    optimization_result_file,
    optimization_spec_file,
    sha256_file,
)
from mmos.schemas.common import MMOSModel
from mmos.schemas.evidence import ClaimRecord, EvidenceIndex, EvidenceRecord
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.optimization import relative_improvement
from mmos.schemas.optimization_tournament import OptimizationTournamentResult
from mmos.schemas.validation import ValidationReport, ValidationStatus

REQUIRED_VALIDATORS = ("leakage", "numerical", "statistical", "reproducibility", "temporal")


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


def freeze_experiment(
    paths: ProjectPaths, experiment_id: str, *, tournament_id: str | None = None
) -> list[EvidenceRecord]:
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
    if result.dataset_metadata_sha256:
        provenance["dataset_metadata_sha256"] = result.dataset_metadata_sha256

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

    def extra_record(
        evidence_id: str,
        key: str,
        raw_value: float | int | str,
        display_value: str,
        source_field: str,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=raw_value,
            display_value=display_value,
            experiment_id=experiment_id,
            artifact_path=relative_metrics,
            artifact_hash=metrics_hash,
            source_field=source_field,
            validation_reports=[report.validator for report in reports],
            verified_at=now,
            provenance=provenance,
        )

    def display(value: object) -> str:
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    if result.target_summary is not None:
        target = result.target_summary
        for field in (
            "mean",
            "std",
            "min",
            "max",
            "median",
            "unit",
            "unit_status",
            "dtype",
            "description",
            "generator",
            "generation_seed",
            "generation_code_path",
            "generation_parameters",
            "is_synthetic",
            "is_positive",
        ):
            value = getattr(target, field)
            if value is None:
                continue
            raw = value.value if hasattr(value, "value") else value
            if isinstance(raw, bool):
                raw = str(raw).lower()
            records.append(
                extra_record(
                    f"ev-{experiment_id}-target-{field}",
                    f"target.{field}",
                    raw,
                    display(raw),
                    f"target_summary.{field}",
                )
            )

    if spec.temporal_split is not None:
        split = spec.temporal_split
        temporal_values: dict[str, float | int | str] = {
            "strategy": split.strategy,
            "horizon": split.horizon,
            "min_train_size": split.min_train_size,
            "step": split.step,
            "n_splits": split.n_splits,
        }
        if spec.time_column is not None:
            temporal_values["time_column"] = spec.time_column
        if spec.seasonal_period is not None:
            temporal_values["seasonal_period"] = spec.seasonal_period
        for name, value in temporal_values.items():
            records.append(
                extra_record(
                    f"ev-{experiment_id}-temporal-{name}",
                    f"temporal.{name}",
                    value,
                    display(value),
                    f"temporal_split.{name}",
                )
            )
        if result.fold_windows:
            windows = "; ".join(
                f"train[0,{window.train_end}) test[{window.test_start},{window.test_end})"
                for window in result.fold_windows
            )
            records.append(
                extra_record(
                    f"ev-{experiment_id}-temporal-fold-windows",
                    "temporal.fold_windows",
                    windows,
                    windows,
                    "fold_windows",
                )
            )

    for model_id, deterministic in result.model_determinism.items():
        value = "deterministic" if deterministic else "stochastic"
        records.append(
            extra_record(
                f"ev-{experiment_id}-determinism-{model_id}",
                f"determinism.{model_id}",
                value,
                value,
                f"model_determinism.{model_id}",
            )
        )

    for model_id, evaluation in result.evaluations.items():
        for field in (
            "n_observations",
            "sse",
            "sst",
            "r2",
            "r2_reason",
            "target_mean",
            "target_std_population",
            "rmse",
            "mae",
            "rmse_eval_std_ratio",
            "mase",
        ):
            value = getattr(evaluation, field)
            if value is None:
                continue
            records.append(
                extra_record(
                    f"ev-{experiment_id}-evaluation-{model_id}-{field}",
                    f"evaluation.{model_id}.{field}",
                    value,
                    display(value),
                    f"evaluations.{model_id}.{field}",
                )
            )
        if evaluation.mase_details:
            details = json.dumps(
                [detail.model_dump(mode="json") for detail in evaluation.mase_details],
                sort_keys=True,
            )
            records.append(
                extra_record(
                    f"ev-{experiment_id}-evaluation-{model_id}-mase-details",
                    f"evaluation.{model_id}.mase_details",
                    details,
                    details,
                    f"evaluations.{model_id}.mase_details",
                )
            )

    for model_id, information_set in result.information_sets.items():
        payload = information_set.model_dump_json()
        records.append(
            extra_record(
                f"ev-{experiment_id}-information-set-{model_id}",
                f"information_set.{model_id}",
                payload,
                payload,
                f"information_sets.{model_id}",
            )
        )

    if tournament_id is not None:
        tournament = load_tournament_result(paths, tournament_id)
        for field, value in (
            ("status", tournament.selection_status),
            ("selected_model", tournament.selected_model or ""),
            ("challenger_model", tournament.challenger_model or ""),
        ):
            records.append(
                extra_record(
                    f"ev-{experiment_id}-selection-{field}",
                    f"selection.{field}",
                    value,
                    value,
                    f"tournament.{field}",
                )
            )
        for comparison in tournament.equivalence:
            prefix = f"comparison.{comparison.candidate_model}_vs_{comparison.baseline_model}"
            for field in (
                "status",
                "information_set_match",
                "relative_improvement",
                "ci_low",
                "ci_high",
                "threshold",
                "n_pairs",
            ):
                value = getattr(comparison, field)
                if value is None:
                    continue
                raw = str(value).lower() if isinstance(value, bool) else value
                records.append(
                    extra_record(
                        f"ev-{experiment_id}-{prefix}-{field}",
                        f"{prefix}.{field}",
                        raw,
                        display(raw),
                        f"equivalence.{field}",
                    )
                )

    records.append(
        extra_record(
            f"ev-{experiment_id}-evaluation-rows",
            "evaluation.unique_rows",
            result.evaluated_rows,
            display(result.evaluated_rows),
            "evaluated_rows",
        )
    )

    for model_id, summaries in result.metric_summaries.items():
        for metric_name, summary in summaries.items():
            if summary.fold_values:
                fold_text = ", ".join(f"{value:.10g}" for value in summary.fold_values)
                records.append(
                    extra_record(
                        f"ev-{experiment_id}-uncertainty-{model_id}-{metric_name}-folds",
                        f"uncertainty.{model_id}.{metric_name}.folds",
                        fold_text,
                        fold_text,
                        f"metric_summaries.{model_id}.{metric_name}.fold_values",
                    )
                )
            for field, value in (
                ("std", summary.std),
                ("ci_low", summary.confidence_interval_low),
                ("ci_high", summary.confidence_interval_high),
            ):
                if value is None:
                    continue
                records.append(
                    extra_record(
                        f"ev-{experiment_id}-uncertainty-{model_id}-{metric_name}-{field}",
                        f"uncertainty.{model_id}.{metric_name}.{field}",
                        value,
                        display(value),
                        f"metric_summaries.{model_id}.{metric_name}.{field}",
                    )
                )

    for feature, availability in result.feature_availability.items():
        records.append(
            extra_record(
                f"ev-{experiment_id}-feature-{feature}-availability",
                f"feature.{feature}.availability",
                availability,
                availability,
                f"feature_availability.{feature}",
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


def freeze_optimization(
    paths: ProjectPaths,
    experiment_id: str,
    *,
    tournament_id: str | None = None,
) -> list[EvidenceRecord]:
    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    summary_path = validation_dir / "summary.json"
    report_path = validation_dir / "optimization.json"
    if not summary_path.exists() or not report_path.exists():
        raise EvidenceError(f"optimization experiment {experiment_id!r} has not been validated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("blocking"):
        raise EvidenceError("optimization validation contains blocking issues")
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.blocking or report.status is not ValidationStatus.PASSED:
        raise EvidenceError("optimization validator did not pass")

    spec = load_optimization_spec(paths, experiment_id)
    result = load_optimization_result(paths, experiment_id)
    result_path = optimization_result_file(paths, experiment_id)
    spec_path = optimization_spec_file(paths, experiment_id)
    result_hash = sha256_file(result_path)
    spec_hash = sha256_file(spec_path)
    now = datetime.now(UTC).replace(microsecond=0)
    mlflow_record_path = paths.experiment_dir(experiment_id) / "mlflow_run.json"
    mlflow_run_id = (
        json.loads(mlflow_record_path.read_text(encoding="utf-8")).get("run_id", "")
        if mlflow_record_path.exists()
        else ""
    )
    record_path = experiment_record_file(paths, experiment_id)
    experiment_record = (
        load_experiment_record(paths, experiment_id) if record_path.exists() else None
    )
    provenance = {
        "kind": "optimization",
        "result_path": result_path.relative_to(paths.root).as_posix(),
        "result_sha256": result_hash,
        "spec_path": spec_path.relative_to(paths.root).as_posix(),
        "spec_sha256": spec_hash,
        "instance_hash": result.instance_hash or spec.instance_hash or "",
        "mlflow_run_id": mlflow_run_id,
        "experiment_record_path": (
            record_path.relative_to(paths.root).as_posix() if record_path.exists() else ""
        ),
        "experiment_record_sha256": sha256_file(record_path) if record_path.exists() else "",
        "git_commit": experiment_record.git_commit if experiment_record else "unknown",
    }

    def record(evidence_id: str, key: str, value: float | int | str) -> EvidenceRecord:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        return EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            experiment_id=experiment_id,
            artifact_path=provenance["result_path"],
            artifact_hash=result_hash,
            source_field=key,
            validation_reports=[report.validator],
            verified_at=now,
            provenance=provenance,
        )

    records: list[EvidenceRecord] = []
    records.append(
        record(f"ev-{experiment_id}-optimization-model", "optimization.model", spec.model.name)
    )
    if result.selected_method:
        records.append(
            record(
                f"ev-{experiment_id}-optimization-selected",
                "optimization.selected_method",
                result.selected_method,
            )
        )
    solver_solution = next(
        (solution for solution in result.solutions if solution.method == "ortools.cbc"),
        None,
    )
    baseline_solution = next(
        (solution for solution in result.solutions if solution.method == "greedy_feasible"),
        None,
    )
    for solution in result.solutions:
        prefix = f"optimization.{solution.method}"
        records.append(
            record(f"ev-{experiment_id}-{prefix}-status", f"{prefix}.status", solution.status)
        )
        records.append(
            record(
                f"ev-{experiment_id}-{prefix}-feasible",
                f"{prefix}.feasible",
                str(solution.feasible).lower(),
            )
        )
        for field in (
            "objective_value",
            "best_bound",
            "optimality_gap",
            "absolute_gap",
            "node_count",
            "solver_version",
            "termination_condition",
        ):
            value = getattr(solution, field)
            if value is None:
                continue
            records.append(
                record(
                    f"ev-{experiment_id}-{prefix}-{field}",
                    f"{prefix}.{field}",
                    value,
                )
            )
        if solution.method == result.selected_method:
            for name, value in solution.variable_values.items():
                records.append(
                    record(
                        f"ev-{experiment_id}-optimization-variable-{name}",
                        f"optimization.variable.{name}",
                        value,
                    )
                )
    if (
        solver_solution is not None
        and baseline_solution is not None
        and solver_solution.objective_value is not None
        and baseline_solution.objective_value is not None
        and baseline_solution.objective_value != 0
    ):
        improvement = relative_improvement(
            result.sense,
            solver_solution.objective_value,
            baseline_solution.objective_value,
        )
        if improvement is not None:
            records.append(
                record(
                    f"ev-{experiment_id}-optimization-relative-improvement",
                    "optimization.relative_improvement",
                    improvement,
                )
            )

    if result.enumeration is not None:
        records.append(
            record(
                f"ev-{experiment_id}-optimization-enumeration-status",
                "optimization.enumeration.status",
                result.enumeration.status,
            )
        )
        records.append(
            record(
                f"ev-{experiment_id}-optimization-enumeration-method",
                "optimization.enumeration.method",
                result.enumeration.method,
            )
        )
        if result.enumeration.optimum_value is not None:
            records.append(
                record(
                    f"ev-{experiment_id}-optimization-enumeration-optimum",
                    "optimization.enumeration.optimum_value",
                    result.enumeration.optimum_value,
                )
            )
        if result.enumeration.matches_solver_result is not None:
            records.append(
                record(
                    f"ev-{experiment_id}-optimization-enumeration-matches",
                    "optimization.enumeration.matches_solver_result",
                    str(result.enumeration.matches_solver_result).lower(),
                )
            )
        records.append(
            record(
                f"ev-{experiment_id}-optimization-enumeration-evaluated",
                "optimization.enumeration.evaluated_subsets",
                result.enumeration.evaluated_subsets,
            )
        )
        if result.enumeration.optimal_subsets is not None:
            records.append(
                record(
                    f"ev-{experiment_id}-optimization-enumeration-optimal-count",
                    "optimization.enumeration.optimal_subsets",
                    result.enumeration.optimal_subsets,
                )
            )
    for solution in result.solutions:
        records.append(
            record(
                f"ev-{experiment_id}-optimization-{solution.method}-runtime",
                f"optimization.{solution.method}.runtime_seconds",
                solution.runtime_seconds,
            )
        )
        if solution.method == result.selected_method:
            for constraint in spec.model.constraints:
                activity = sum(
                    coefficient * solution.variable_values.get(name, 0.0)
                    for name, coefficient in constraint.coefficients.items()
                )
                records.append(
                    record(
                        f"ev-{experiment_id}-optimization-selected-constraint-{constraint.name}",
                        f"optimization.selected.constraint.{constraint.name}",
                        activity,
                    )
                )
    records.append(
        record(
            f"ev-{experiment_id}-optimization-time-limit",
            "optimization.time_limit_seconds",
            spec.time_limit_seconds,
        )
    )

    records.append(
        record(
            f"ev-{experiment_id}-experiment-instance-hash",
            "experiment.instance_hash",
            result.instance_hash or spec.instance_hash or "",
        )
    )
    records.append(
        record(
            f"ev-{experiment_id}-experiment-mlflow-run-id",
            "experiment.mlflow_run_id",
            mlflow_run_id or "unavailable",
        )
    )
    if experiment_record is not None:
        records.append(
            record(
                f"ev-{experiment_id}-experiment-status",
                "experiment.status",
                experiment_record.status.value,
            )
        )
        records.append(
            record(
                f"ev-{experiment_id}-experiment-git-commit",
                "experiment.git_commit",
                experiment_record.git_commit,
            )
        )
        for linked_id in experiment_record.linked_experiment_ids:
            records.append(
                record(
                    f"ev-{experiment_id}-experiment-linked-{linked_id}",
                    "experiment.linked_experiment_id",
                    linked_id,
                )
            )
    records.append(
        record(
            f"ev-{experiment_id}-experiment-solver-backend",
            "experiment.solver_backend",
            result.solver_backend,
        )
    )
    records.append(
        record(
            f"ev-{experiment_id}-experiment-solver-version",
            "experiment.solver_version",
            result.solver_version,
        )
    )
    records.append(
        record(
            f"ev-{experiment_id}-experiment-time-limit",
            "experiment.time_limit_seconds",
            spec.time_limit_seconds,
        )
    )

    tournament_dir = paths.tournaments_dir / tournament_id if tournament_id is not None else None
    if tournament_dir is not None and (tournament_dir / "tournament.json").exists():
        tournament_path = tournament_dir / "tournament.json"
        tournament = OptimizationTournamentResult.model_validate_json(
            tournament_path.read_text(encoding="utf-8")
        )
        tournament_hash = sha256_file(tournament_path)
        candidate = next(
            entry for entry in tournament.entries if entry.entry_role.value == "candidate"
        )
        baseline = next(
            entry for entry in tournament.entries if entry.entry_role.value == "baseline"
        )
        improvement = None
        if candidate.objective_value is not None and baseline.objective_value is not None:
            improvement = relative_improvement(
                candidate.objective_sense,
                candidate.objective_value,
                baseline.objective_value,
            )
        tournament_mlflow = tournament_dir / "mlflow_run.json"
        tournament_run_id = (
            json.loads(tournament_mlflow.read_text(encoding="utf-8")).get("run_id", "")
            if tournament_mlflow.exists()
            else ""
        )
        provenance["tournament_path"] = tournament_path.relative_to(paths.root).as_posix()
        provenance["tournament_sha256"] = tournament_hash
        provenance["tournament_id"] = tournament.tournament_id
        provenance["tournament_mlflow_run_id"] = tournament_run_id
        provenance["candidate_experiment_id"] = candidate.experiment_id
        provenance["baseline_experiment_id"] = baseline.experiment_id

        def tournament_record(
            evidence_id: str, key: str, value: float | int | str
        ) -> EvidenceRecord:
            display_value = f"{value:.10g}" if isinstance(value, float) else str(value)
            return EvidenceRecord(
                evidence_id=evidence_id,
                key=key,
                raw_value=value,
                display_value=display_value,
                experiment_id=experiment_id,
                artifact_path=tournament_path.relative_to(paths.root).as_posix(),
                artifact_hash=tournament_hash,
                source_field=key,
                validation_reports=[report.validator],
                verified_at=now,
                provenance=provenance,
            )

        tournament_values: list[tuple[str, str, float | int | str]] = [
            ("status", "tournament.status", tournament.winner_status.value),
            (
                "winner",
                "tournament.winner_experiment_id",
                tournament.winner_experiment_id or "none",
            ),
            ("candidate-id", "tournament.candidate_experiment_id", candidate.experiment_id),
            ("baseline-id", "tournament.baseline_experiment_id", baseline.experiment_id),
            ("sense", "tournament.objective_sense", candidate.objective_sense),
            ("comparison", "tournament.comparison_status", tournament.winner_status.value),
            (
                "candidate-feasible",
                "tournament.candidate_feasible",
                str(candidate.feasible).lower(),
            ),
            (
                "baseline-feasible",
                "tournament.baseline_feasible",
                str(baseline.feasible).lower(),
            ),
            (
                "candidate-optimality",
                "tournament.candidate_optimality",
                candidate.optimality_class.value,
            ),
            (
                "baseline-optimality",
                "tournament.baseline_optimality",
                baseline.optimality_class.value,
            ),
            (
                "verification",
                "tournament.independent_verification",
                candidate.independent_verification_method or "unavailable",
            ),
            (
                "baseline-role",
                "tournament.baseline_role",
                baseline.baseline_role.value if baseline.baseline_role else "none",
            ),
            ("decision-trace", "tournament.decision_trace", " | ".join(tournament.decision_trace)),
            ("warnings", "tournament.warnings", " | ".join(tournament.warnings) or "none"),
            ("mlflow", "tournament.mlflow_run_id", tournament_run_id or "unavailable"),
        ]
        if candidate.objective_value is not None:
            tournament_values.append(
                ("candidate-objective", "tournament.candidate_objective", candidate.objective_value)
            )
        if baseline.objective_value is not None:
            tournament_values.append(
                ("baseline-objective", "tournament.baseline_objective", baseline.objective_value)
            )
        if improvement is not None:
            tournament_values.append(
                ("relative-improvement", "tournament.relative_improvement", improvement)
            )
        if candidate.objective_value is not None and baseline.objective_value is not None:
            tournament_values.append(
                (
                    "absolute-improvement",
                    "tournament.absolute_improvement",
                    candidate.objective_value - baseline.objective_value,
                )
            )
        for suffix, key, value in tournament_values:
            records.append(tournament_record(f"ev-{experiment_id}-tournament-{suffix}", key, value))

    index = load_evidence_index(paths)
    for evidence in records:
        index.records[evidence.evidence_id] = evidence
    atomic_write_text(paths.verified_numbers_file, index.model_dump_json(indent=2) + "\n")
    optimization_dir = paths.verified_dir / "optimization"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        optimization_dir / f"{experiment_id}.json",
        json.dumps(
            {
                "experiment_id": experiment_id,
                "artifact_hash": result_hash,
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
    provenance_index["experiments"][f"optimization:{experiment_id}"] = {
        **provenance,
        "validation_reports": {"optimization": sha256_file(report_path)},
        "frozen_at": now.isoformat(),
    }
    atomic_write_text(provenance_path, json.dumps(provenance_index, indent=2) + "\n")
    if not paths.verified_claims_file.exists():
        atomic_write_text(
            paths.verified_claims_file,
            yaml.safe_dump({"claims": []}, sort_keys=False, allow_unicode=True),
        )
    return records
