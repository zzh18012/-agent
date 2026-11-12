import json

from mmos.core.project import (
    ProjectPaths,
    load_experiment_spec,
    load_problem_spec,
    load_tournament_result,
)
from mmos.evidence import load_claims, load_evidence_index
from mmos.paper.figures import create_metric_figure
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable


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

    tournament = load_tournament_result(paths, tournament_id) if tournament_id is not None else None
    spec = load_experiment_spec(paths, experiment_id)
    problem = None
    if paths.problem_spec_file.exists():
        problem = load_problem_spec(paths)

    figure = create_metric_figure(
        paths,
        experiment_id,
        primary_metric,
        higher_is_better=higher_is_better,
        records=primary_records,
    )

    model_order = sorted(grouped)
    if tournament is not None:
        rank = {entry.model_id: entry.rank for entry in tournament.entries}
        model_order = sorted(grouped, key=lambda model: (rank.get(model, 10_000), model))

    columns = ["Model", *metric_names]
    rows: list[list[str]] = []
    table_evidence_ids: list[str] = []
    for model_id in model_order:
        row = [model_id]
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
    validation_summary = json.loads((validation_dir / "summary.json").read_text(encoding="utf-8"))
    validator_status = ", ".join(
        f"{name}: {info['status']}" for name, info in sorted(validation_summary["reports"].items())
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
    baseline_sentence = (
        f"The best baseline ({baseline_name}) reports {primary_metric} = "
        f"{baseline_record.display_value}. "
        if baseline_record is not None
        else ""
    )

    title = (
        f"{problem.title} — Technical Report"
        if problem
        else f"{paths.root.name} — Technical Report"
    )
    abstract = (
        f"This report is generated from {len(records)} verified evidence records for "
        f"experiment {experiment_id}. Under the verified cross-validation protocol, "
        f"{selected_name} reports {primary_metric} = {selected_display}. "
        f"{baseline_sentence}"
        "This is a single point estimate. No external validation, robustness, or "
        "sensitivity analysis was performed. Every number in this report comes from "
        "the verified evidence store."
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

    sections.append(
        PaperSection(
            heading="Data and validation",
            blocks=[
                f"Dataset hash: `{spec.dataset_hash}`. Target column: `{spec.target_column}`. "
                f"Features: {', '.join(spec.feature_set)}. Split: {spec.split_strategy} with "
                f"{spec.n_splits} folds, seed {spec.random_seed}.",
                "No preprocessing or imputation is applied before splitting; the runner "
                "uses the selected raw columns.",
                f"Validation results: {validator_status}.",
            ],
        )
    )
    sections.append(
        PaperSection(
            heading="Results",
            blocks=[
                f"Table `{metrics_table.table_id}` reports metrics computed from out-of-fold "
                "predictions, independently recomputed by the numerical auditor.",
                f"Selected model {selected_name}: {primary_metric} = {selected_display}."
                + (
                    f" Best baseline {baseline_name}: {baseline_record.display_value}."
                    if baseline_record is not None
                    else ""
                ),
                f"Figure {figure.figure_id} visualizes {primary_metric} from the same "
                "verified evidence.",
            ],
        )
    )
    if tournament is not None:
        sections.append(
            PaperSection(
                heading="Model comparison",
                blocks=[
                    f"Selected model: {tournament.selected_model or 'none'}. "
                    f"Challenger: {tournament.challenger_model or 'none'}. "
                    f"Primary metric: {tournament.primary_metric} "
                    f"({'higher' if tournament.higher_is_better else 'lower'} is better).",
                    *[f"Note: {note}" for note in tournament.notes],
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
            ],
        )
    )
    sections.append(
        PaperSection(
            heading="Limitations",
            blocks=[
                "Results are cross-validated on one dataset; no external validation set exists.",
                "Robustness and sensitivity analyses have not been run, so conclusions are "
                "conditional on the observed data.",
                "Leakage checks are heuristic and do not replace split-level review.",
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
