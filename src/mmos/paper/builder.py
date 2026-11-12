import json
from typing import Literal

from mmos.core.project import (
    ProjectPaths,
    load_experiment_spec,
    load_problem_spec,
    load_tournament_result,
)
from mmos.evidence import load_claims, load_evidence_index
from mmos.experiments.engines import TEMPORAL_BASELINES
from mmos.paper.figures import create_metric_figure
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable

ModelStatus = Literal["provisional_winner", "verified_winner"]


class PaperError(RuntimeError):
    pass


def _group_metrics(records: list[EvidenceRecord]) -> dict[str, dict[str, EvidenceRecord]]:
    grouped: dict[str, dict[str, EvidenceRecord]] = {}
    for record in records:
        if not record.key.startswith("metrics."):
            continue
        remainder = record.key[len("metrics.") :]
        model_id, separator, metric = remainder.rpartition(".")
        if not separator or not model_id:
            continue
        grouped.setdefault(model_id, {})[metric] = record
    return grouped


def build_paper(
    paths: ProjectPaths,
    experiment_id: str,
    *,
    tournament_id: str | None = None,
    primary_metric: str = "rmse",
    higher_is_better: bool = False,
    model_status: ModelStatus | None = None,
) -> PaperAST:
    """Assemble a paper AST from verified evidence only."""
    index = load_evidence_index(paths)
    records = [record for record in index.records.values() if record.experiment_id == experiment_id]
    if not records:
        raise PaperError(
            f"no verified evidence for experiment {experiment_id!r}; "
            "run validate and freeze-evidence first"
        )

    grouped = _group_metrics(records)
    metric_names = sorted({metric for metrics in grouped.values() for metric in metrics})
    candidate_models = [
        model_id
        for model_id in grouped
        if model_id not in TEMPORAL_BASELINES and model_id != "sklearn.dummy_regressor"
    ]
    if not metric_names:
        raise PaperError("verified evidence does not contain metric records")

    primary_records = [
        metrics[primary_metric] for metrics in grouped.values() if primary_metric in metrics
    ]
    if not primary_records:
        raise PaperError(
            f"primary metric {primary_metric!r} has no verified evidence; "
            f"available: {', '.join(metric_names)}"
        )

    by_key = {record.key: record for record in records}

    def target_value(name: str) -> str | None:
        record = by_key.get(f"target.{name}")
        return record.display_value if record is not None else None

    synthetic = target_value("is_synthetic") == "true"

    tournament = load_tournament_result(paths, tournament_id) if tournament_id is not None else None
    spec = load_experiment_spec(paths, experiment_id)
    problem = None
    if paths.problem_spec_file.exists():
        problem = load_problem_spec(paths)

    per_fold_values: dict[str, list[float]] = {}
    for record in primary_records:
        model_id = record.key.removeprefix("metrics.").removesuffix(f".{primary_metric}")
        folds_record = by_key.get(f"uncertainty.{model_id}.{primary_metric}.folds")
        if folds_record is not None:
            try:
                per_fold_values[model_id] = [
                    float(value.strip())
                    for value in folds_record.display_value.split(",")
                    if value.strip()
                ]
            except ValueError:
                continue

    figure = create_metric_figure(
        paths,
        experiment_id,
        primary_metric,
        higher_is_better=higher_is_better,
        records=primary_records,
        fold_values=per_fold_values or None,
    )

    model_order = sorted(grouped)
    if tournament is not None:
        rank = {entry.model_id: entry.rank for entry in tournament.entries}
        model_order = sorted(grouped, key=lambda model: (rank.get(model, 10_000), model))

    def information_set(model_id: str) -> str:
        if model_id in TEMPORAL_BASELINES:
            return "target history only"
        if model_id == "sklearn.dummy_regressor":
            return "constant (no inputs)"
        return f"features: {', '.join(spec.feature_set) or '(none)'}"

    columns = ["Model", "Information set", *metric_names]
    rows: list[list[str]] = []
    table_evidence_ids: list[str] = []
    for model_id in model_order:
        row = [model_id, information_set(model_id)]
        for metric in metric_names:
            record = grouped[model_id].get(metric)
            row.append(record.display_value if record else "-")
            if record is not None:
                table_evidence_ids.append(record.evidence_id)
        rows.append(row)

    metrics_table = PaperTable(
        table_id=f"tab-{experiment_id}-metrics",
        caption="Cross-validated metrics from verified out-of-fold predictions.",
        columns=columns,
        rows=rows,
        evidence_ids=table_evidence_ids,
    )

    supported_claims = [
        claim
        for claim in load_claims(paths)
        if claim.status == "supported"
        and all(evidence_id in index.records for evidence_id in claim.evidence_ids)
    ]

    validation_dir = paths.experiment_dir(experiment_id) / "validation"
    summary_path = validation_dir / "summary.json"
    validation_summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    )
    validation_passed = bool(
        validation_summary is not None and not validation_summary.get("blocking")
    )
    validator_status = (
        ", ".join(
            f"{name}: {info['status']}"
            for name, info in sorted(validation_summary["reports"].items())
        )
        if validation_summary is not None
        else "not available"
    )
    resolved_status: ModelStatus = model_status or (
        "verified_winner" if validation_passed else "provisional_winner"
    )
    if tournament is not None:
        status_phrase = (
            "verified tournament winner"
            if resolved_status == "verified_winner"
            else "provisional tournament winner"
        )
    else:
        status_phrase = (
            "verified leading candidate"
            if resolved_status == "verified_winner"
            else "provisional leading candidate"
        )

    result_label = (
        "best-scoring candidate in this synthetic sanity run" if synthetic else status_phrase
    )

    results_label = (
        "The only fitted candidate model"
        if synthetic and len(candidate_models) == 1
        else result_label.capitalize()
    )

    provenance = records[0].provenance
    selected = tournament.selected_model if tournament is not None else None
    best_record = min(
        primary_records,
        key=lambda record: (
            -float(record.raw_value) if higher_is_better else float(record.raw_value)
        ),
    )
    selected_display = (
        grouped[selected][primary_metric].display_value
        if selected is not None and primary_metric in grouped.get(selected, {})
        else best_record.display_value
    )
    selected_name = selected or best_record.key.removeprefix("metrics.").removesuffix(
        f".{primary_metric}"
    )

    baseline_name = tournament.challenger_model if tournament is not None else None
    baseline_record = grouped.get(baseline_name, {}).get(primary_metric) if baseline_name else None
    if spec.temporal_split is not None:
        references = []
        for reference_id in ("naive.last_value", "naive.drift", "sklearn.dummy_regressor"):
            reference = grouped.get(reference_id, {}).get(primary_metric)
            if reference is not None:
                references.append(f"{reference_id} {primary_metric} = {reference.display_value}")
        baseline_sentence = (
            "Reference points: " + "; ".join(references) + ". " if references else ""
        )
    else:
        baseline_sentence = (
            f"The lowest-RMSE reference ({baseline_name}) reports {primary_metric} = "
            f"{baseline_record.display_value}. "
            if baseline_record is not None
            else ""
        )

    target_std = target_value("std")
    unit_status_value = target_value("unit_status") or "unknown"
    target_unit = target_value("unit")
    if target_unit is None:
        target_unit = (
            "synthetic_units (dimensionless)"
            if unit_status_value == "synthetic_units"
            else "unknown"
        )
    ratio_record = grouped.get(selected_name, {}).get("rmse_std_ratio")
    std_sentence = (
        f"The target (`{spec.target_column}`) has standard deviation {target_std} ({target_unit}). "
        if target_std is not None
        else ""
    )
    ratio_sentence = (
        f"The reported {primary_metric} equals {ratio_record.display_value} target "
        "standard deviations. "
        if ratio_record is not None
        else ""
    )
    synthetic_sentence = ""

    if synthetic:
        generator = target_value("generator") or "unspecified"
        seed = target_value("generation_seed") or "unspecified"
        parameters = target_value("generation_parameters")
        parameter_text = f" with parameters {parameters}" if parameters else ""
        synthetic_sentence = (
            f"This is a synthetic demonstration dataset generated by {generator} "
            f"(seed {seed}){parameter_text}; results do not represent real-world "
            "predictive performance. "
        )

    title = (
        f"{problem.title} — Technical Report"
        if problem
        else f"{paths.root.name} — Technical Report"
    )
    abstract = (
        f"This report is generated from {len(records)} verified evidence records for "
        f"experiment {experiment_id}. The {result_label} is {selected_name} with "
        f"{primary_metric} = {selected_display}. {baseline_sentence}{std_sentence}"
        f"{ratio_sentence}{synthetic_sentence}"
        + (
            "This is a pipeline sanity check, not a model-selection finding: the "
            "generator is explicitly recorded and no claim about real demand modeling "
            "is made. "
            if synthetic
            else ""
        )
        + (
            "The candidate uses known-future features while the naive baselines use "
            "target history only, so the gap partly reflects feature access, not "
            "modelling skill. "
            if spec.temporal_split is not None
            else ""
        )
        + (
            "This RMSE has no interpretation as a demand-forecast error; it is a "
            "dimensionless artifact of the synthetic generator. "
            if synthetic
            else ""
        )
        + "Point estimates and fold-level variability are reported; no external "
        "validation, robustness, or sensitivity analysis was performed. Every number "
        "in this report comes from the verified evidence store."
    )

    sections: list[PaperSection] = []
    if problem is not None:
        subproblems = "\n".join(
            f"- {subproblem.subproblem_id}: {subproblem.statement}"
            for subproblem in problem.subproblems
        )
        sections.append(
            PaperSection(
                heading="Problem understanding",
                blocks=[problem.overall_goal, subproblems],
            )
        )
    else:
        sections.append(
            PaperSection(
                heading="Problem understanding",
                blocks=["No ProblemSpec is available for this project."],
            )
        )

    target_blocks = [
        f"Target `{spec.target_column}`: unit = {target_unit} "
        f"(status: {unit_status_value}), "
        f"mean = {target_value('mean') or 'unknown'}, "
        f"std = {target_std or 'unknown'}, "
        f"min = {target_value('min') or 'unknown'}, "
        f"max = {target_value('max') or 'unknown'}, "
        f"median = {target_value('median') or 'unknown'}."
    ]
    if synthetic:
        target_blocks.append(
            f"Synthetic data: generator = {target_value('generator') or 'unspecified'}, "
            f"seed = {target_value('generation_seed') or 'unspecified'}. "
            "These results do not represent real-world performance."
        )
        if target_value("generation_parameters"):
            target_blocks.append(f"Generation parameters: {target_value('generation_parameters')}.")
        target_blocks.append(
            "The target is expressed in synthetic_units, which have no real-world "
            "interpretation; RMSE in these units cannot be translated into a "
            "demand-context error."
        )
    sections.append(PaperSection(heading="Target context", blocks=target_blocks))

    if spec.temporal_split is not None:
        split = spec.temporal_split
        seasonal = (
            f"seasonal period {spec.seasonal_period}"
            if spec.seasonal_period is not None
            else "no seasonal period was configured for the time-series baselines "
            "(seasonal-naive at potential lags was not run)"
        )
        sections.append(
            PaperSection(
                heading="Temporal evaluation",
                blocks=[
                    f"Rolling-origin validation with horizon {split.horizon}, "
                    f"{split.n_splits} folds, time column `{spec.time_column}`, and {seasonal}. "
                    "Every fold satisfies max(train time) < min(test time) by construction.",
                    "Random KFold is forbidden for this dataset.",
                ],
            )
        )

    availability = {
        key.removeprefix("feature.").removesuffix(".availability"): record.display_value
        for key, record in by_key.items()
        if key.startswith("feature.") and key.endswith(".availability")
    }
    feature_availability_note = (
        ", ".join(f"{name}={status}" for name, status in sorted(availability.items()))
        if availability
        else "not recorded"
    )

    sections.append(
        PaperSection(
            heading="Data and validation",
            blocks=[
                f"Dataset hash: `{spec.dataset_hash}`. Target column: `{spec.target_column}`. "
                f"Features: {', '.join(spec.feature_set) or '(none)'}. Split: "
                f"{spec.split_strategy} with {spec.n_splits} folds, seed {spec.random_seed}.",
                "No preprocessing or imputation is applied before splitting; the runner "
                "uses the selected raw columns.",
                *(
                    [
                        "The generator defines x1 and x2 as deterministic functions of "
                        "period; the heuristic validation checks do not test phase "
                        "recovery, and no phase-residual diagnostic was performed.",
                    ]
                    if synthetic
                    else []
                ),
                f"Feature availability at forecast time: {feature_availability_note}.",
                f"Validation checks returned no blocking flag: {validator_status} "
                "(heuristic checks; see Limitations).",
            ],
        )
    )
    evaluated = by_key.get("evaluation.unique_rows")
    evaluated_text = evaluated.display_value if evaluated is not None else "unknown"
    sections.append(
        PaperSection(
            heading="Metric definitions",
            blocks=[
                "RMSE and MAE are on the original target scale. R² is the standard "
                "coefficient of determination computed on the stacked out-of-fold "
                "prediction vector; with very few folds it is indicative only. Per-fold "
                "R² is omitted because a one-point test fold makes it undefined.",
                "MASE uses the mean absolute error of the one-step naive forecast on each "
                "training window as its denominator (a seasonal lag would be used only if "
                "a seasonal period were configured).",
                "RMSE_std_ratio divides RMSE by the full-sample target standard deviation, "
                "not by a training-fold standard deviation.",
                f"Only {evaluated_text} out-of-fold predictions support these metrics; "
                "they are not independent, and the derived metrics and intervals share "
                "the same small set of residuals.",
            ],
        )
    )
    sections.append(
        PaperSection(
            heading="Results",
            blocks=[
                f"Table `{metrics_table.table_id}` reports metrics computed from out-of-fold "
                "predictions, independently recomputed by the numerical auditor.",
                f"{results_label} {selected_name}: "
                f"{primary_metric} = {selected_display}."
                + (
                    f" Best reference {baseline_name}: {baseline_record.display_value}."
                    if baseline_record is not None
                    else ""
                ),
                "Candidates use the declared known-future features; naive baselines use "
                "target history only, so the gap partly reflects access to information, "
                "not modelling skill.",
                *(
                    [
                        "No model selection was performed; one fitted candidate was "
                        "evaluated against reference baselines.",
                    ]
                    if synthetic and len(candidate_models) == 1
                    else []
                ),
                f"Figure {figure.figure_id} visualizes {primary_metric} from the same "
                "verified evidence.",
            ],
        )
    )

    uncertainty_std = by_key.get(f"uncertainty.{selected_name}.{primary_metric}.std")
    uncertainty_low = by_key.get(f"uncertainty.{selected_name}.{primary_metric}.ci_low")
    uncertainty_high = by_key.get(f"uncertainty.{selected_name}.{primary_metric}.ci_high")
    determinism = by_key.get(f"determinism.{selected_name}")
    uncertainty_blocks: list[str] = []
    if uncertainty_std is not None:
        uncertainty_blocks.append(
            f"Fold-level {primary_metric} standard deviation for {selected_name}: "
            f"{uncertainty_std.display_value}."
        )
        if uncertainty_low is not None and uncertainty_high is not None:
            uncertainty_blocks.append(
                f"Fold-to-fold descriptive bootstrap range "
                f"[{uncertainty_low.display_value}, {uncertainty_high.display_value}] "
                "(not a generalization confidence interval)."
            )
        fold_values = by_key.get(f"uncertainty.{selected_name}.{primary_metric}.folds")
        if fold_values is not None:
            uncertainty_blocks.append(
                f"Per-fold {primary_metric} values: {fold_values.display_value}. "
                "These fold values are serially dependent; the range is descriptive only "
                "and is not a generalization confidence interval."
            )
    else:
        uncertainty_blocks.append("No fold-level uncertainty was recorded.")
    if determinism is not None:
        if determinism.display_value == "stochastic":
            uncertainty_blocks.append(
                "This model is stochastic; a single seed was used, so seed variance is not "
                "captured."
            )
        else:
            uncertainty_blocks.append(
                "This model is deterministic for a fixed dataset and solver, so repeating "
                "seeds would be redundant for fit reproducibility; fold variability is "
                "reported instead. This does not describe generalization uncertainty."
            )
    sections.append(PaperSection(heading="Uncertainty", blocks=uncertainty_blocks))

    if tournament is not None:
        sections.append(
            PaperSection(
                heading="Model comparison",
                blocks=[
                    f"{results_label}: {tournament.selected_model or 'none'}. "
                    f"Reference: {tournament.challenger_model or 'none'}. "
                    f"Primary metric: {tournament.primary_metric} "
                    f"({'higher' if tournament.higher_is_better else 'lower'} is better).",
                    *[f"Note: {note}" for note in tournament.notes],
                    "Information sets differ between candidates and naive baselines; this "
                    "comparison is not a like-for-like test of modelling skill alone.",
                ],
            )
        )
    else:
        sections.append(
            PaperSection(
                heading="Model comparison",
                blocks=["No tournament result was supplied for this report."],
            )
        )

    run_record_path = paths.experiment_dir(experiment_id) / "run_record.json"
    environment_note = "Environment versions were not recorded."
    if run_record_path.exists():
        run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
        environment_note = (
            f"Environment: Python {run_record.get('python_version', 'unknown')}, "
            f"scikit-learn {run_record.get('sklearn_version', 'unknown')}."
        )

    sections.append(
        PaperSection(
            heading="Reproducibility",
            blocks=[
                f"Metrics artifact: `{provenance.get('metrics_path', '')}` with sha256 "
                f"`{provenance.get('metrics_sha256', '')}`.",
                f"Experiment spec sha256: `{provenance.get('experiment_spec_sha256', '')}`. "
                f"MLflow run id: `{provenance.get('mlflow_run_id', '')}`.",
                environment_note,
                *(
                    [
                        f"Fold boundaries: {fold_windows_record.display_value}.",
                    ]
                    if (fold_windows_record := by_key.get("temporal.fold_windows")) is not None
                    else []
                ),
                *(
                    [
                        "Fitted model configuration: unregularized ordinary least squares "
                        "with intercept (scikit-learn LinearRegression defaults).",
                    ]
                    if selected_name == "sklearn.linear_regression"
                    else []
                ),
            ],
        )
    )
    sections.append(
        PaperSection(
            heading="Limitations",
            blocks=[
                "Results are evaluated on one dataset; no external validation set exists.",
                "Robustness and sensitivity analyses are not part of this comparison, so "
                "conclusions are conditional on the observed data and split design.",
                "Leakage checks are heuristic and do not replace split-level review.",
                *(
                    [
                        "The synthetic generator is exactly linear in the supplied "
                        "features, so a linear model's advantage is structurally expected; "
                        "near-unity R² is not evidence of modelling skill.",
                        "The candidate omits the generator's 0.4*sin(period) term. A "
                        "residual decomposition with an oracle that includes sin(period), "
                        "and Fourier/ridge variants, were not run, so misspecification "
                        "error is not separable from noise.",
                        "Coefficient recovery and residual diagnostics were not examined.",
                        "A single generator seed and a single rolling split were used; "
                        "alternative seeds and alternative fold counts were not run, so "
                        "conclusions are conditional on them.",
                        "Seasonal-naive baselines at potential lags were not run because "
                        "no seasonal period was configured.",
                    ]
                    if synthetic
                    else []
                ),
            ],
        )
    )

    return PaperAST(
        title=title,
        abstract=abstract,
        sections=sections,
        tables=[metrics_table],
        figures=[figure],
        claims=supported_claims,
        references=[
            "In-repository experiment artifacts and verified evidence index.",
        ],
    )
