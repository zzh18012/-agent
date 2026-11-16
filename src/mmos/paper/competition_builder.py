from dataclasses import dataclass
from pathlib import Path

from mmos.competition.context import CatalogEntry, EvidenceContext
from mmos.competition.number_audit import ExpandedText, expand_text
from mmos.core.project import ProjectPaths
from mmos.paper.competition_figures import build_figure
from mmos.schemas.competition import (
    CompetitionExperimentManifest,
    CompetitionPaperPlan,
    ExperimentAnalysisReport,
    FigurePlan,
    SectionKind,
    TablePlan,
    TableTemplate,
)
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.paper import PaperAST, PaperFigure, PaperSection, PaperTable


class CompetitionBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompetitionBuildResult:
    ast: PaperAST
    fragments: list[ExpandedText]


def _entry_map(context: EvidenceContext) -> dict[str, CatalogEntry]:
    return {entry.evidence_id: entry for entry in context.entries}


def _require_evidence(
    evidence_ids: list[str], known: set[str], *, location: str
) -> None:
    missing = sorted(set(evidence_ids) - known)
    if missing:
        raise CompetitionBuildError(
            f"{location} cites unknown evidence ids: {', '.join(missing)}"
        )


def _table_rows(
    table: TablePlan, context: EvidenceContext
) -> tuple[list[str], list[list[str]]]:
    if table.template is TableTemplate.METRIC_COMPARISON:
        records = [
            entry for entry in context.entries if entry.key.startswith("metrics.")
        ]
        if not records:
            raise CompetitionBuildError("metric_comparison has no metric evidence")
        labels = {
            experiment.experiment_id: experiment.label
            for experiment in context.experiments
        }
        rows = []
        for entry in records:
            parts = entry.key.split(".")
            model = parts[1] if len(parts) > 2 else "-"
            metric = ".".join(parts[2:]) if len(parts) > 2 else entry.key
            rows.append(
                [
                    labels.get(entry.experiment_id, entry.experiment_id),
                    model,
                    metric,
                    entry.display_value,
                    entry.unit or "-",
                ]
            )
        return ["Experiment", "Model", "Metric", "Value", "Unit"], rows

    if table.template is TableTemplate.EXPERIMENT_INVENTORY:
        if not context.experiments:
            raise CompetitionBuildError("experiment_inventory has no experiments")
        rows = [
            [
                experiment.experiment_id,
                experiment.label,
                experiment.task,
                str(experiment.evidence_count),
            ]
            for experiment in context.experiments
        ]
        return ["Experiment", "Role", "Task", "Evidence records"], rows

    if table.template is TableTemplate.EVIDENCE_LINEAGE:
        if not context.entries:
            raise CompetitionBuildError("evidence_lineage has no evidence")
        by_experiment: dict[str, list[CatalogEntry]] = {}
        for entry in context.entries:
            by_experiment.setdefault(entry.experiment_id, []).append(entry)
        rows = [
            [
                context.release_path.name,
                experiment_id,
                str(len(entries)),
                entries[0].artifact_hash[:16],
                ", ".join(entry.evidence_id for entry in entries[:3]),
            ]
            for experiment_id, entries in sorted(by_experiment.items())
        ]
        return [
            "Source release",
            "Experiment",
            "Evidence records",
            "Artifact hash (prefix)",
            "Example evidence ids",
        ], rows

    raise CompetitionBuildError(f"unsupported table template: {table.template}")


def _build_tables(
    context: EvidenceContext,
    plan: CompetitionPaperPlan,
    records: dict[str, EvidenceRecord],
    fragments: list[ExpandedText],
) -> list[PaperTable]:
    known = context.evidence_ids()
    tables: list[PaperTable] = []
    for table in plan.tables:
        _require_evidence(table.evidence_ids, known, location=table.table_id)
        available = context.available_tables.get(table.template.value, 0)
        if available <= 0:
            raise CompetitionBuildError(
                f"{table.table_id} requests unavailable template {table.template.value!r}"
            )
        caption = expand_text(
            table.caption,
            records,
            location=f"table:{table.table_id}",
            seeded_ids=tuple(table.evidence_ids),
        )
        fragments.append(caption)
        columns, rows = _table_rows(table, context)
        tables.append(
            PaperTable(
                table_id=table.table_id,
                caption=caption.text,
                columns=columns,
                rows=rows,
                evidence_ids=table.evidence_ids,
            )
        )
    tables.append(
        PaperTable(
            table_id="tab-notation",
            caption="Symbols, meanings, and units used in the model.",
            columns=["Symbol", "Meaning", "Unit"],
            rows=[
                [entry.symbol, entry.meaning, entry.unit or "-"]
                for entry in plan.notation
            ],
            evidence_ids=[],
        )
    )
    return tables


def _build_sections(
    context: EvidenceContext,
    plan: CompetitionPaperPlan,
    analysis: ExperimentAnalysisReport,
    manifest: CompetitionExperimentManifest,
    records: dict[str, EvidenceRecord],
    fragments: list[ExpandedText],
) -> list[PaperSection]:
    sections: list[PaperSection] = []
    known = context.evidence_ids()
    for section in plan.sections:
        _require_evidence(
            section.evidence_ids, known, location=f"section:{section.heading}"
        )
        blocks: list[str] = []
        structural = section.kind in {
            SectionKind.MODEL,
            SectionKind.SOLUTION,
            SectionKind.NOTATION,
        }
        for block in section.blocks:
            expanded = expand_text(
                block,
                records,
                location=f"section:{section.heading}",
                allow_structural_constants=structural,
                seeded_ids=tuple(section.evidence_ids),
            )
            fragments.append(expanded)
            blocks.append(expanded.text)
        for point in section.claim_points:
            _require_evidence(
                point.evidence_ids, known, location=f"claim point:{section.heading}"
            )
            expanded = expand_text(
                point.text,
                records,
                location=f"claim point:{section.heading}",
                allow_structural_constants=structural,
                seeded_ids=tuple(point.evidence_ids),
            )
            fragments.append(expanded)
            blocks.append(
                f"- {expanded.text} (evidence: {', '.join(point.evidence_ids)})"
            )
        if section.kind is SectionKind.ASSUMPTIONS:
            for assumption in plan.assumptions:
                expanded = expand_text(
                    assumption,
                    records,
                    location="Assumptions",
                    seeded_ids=tuple(section.evidence_ids),
                )
                fragments.append(expanded)
                blocks.append(f"- {expanded.text}")
        if section.kind is SectionKind.NOTATION:
            blocks.append("Notation is summarized in `tab-notation`.")
        if section.kind is SectionKind.LIMITATIONS:
            for limitation in [*plan.limitations, *analysis.limitations]:
                expanded = expand_text(
                    limitation,
                    records,
                    location="Limitations",
                    seeded_ids=tuple(section.evidence_ids),
                )
                fragments.append(expanded)
                blocks.append(f"- {expanded.text}")
            unsupported = [
                request
                for request in manifest.requests
                if not request.supported
            ]
            if unsupported:
                blocks.append(
                    "Some requested experiments are not supported by the current "
                    "engines; they are listed in the research-gap appendix and were "
                    "not executed."
                )
        sections.append(PaperSection(heading=section.heading, blocks=blocks))

    gap_blocks = [
        "The following experiments are proposed, not executed. No result from these "
        "requests appears in this paper."
    ]
    for request in manifest.requests:
        expanded = expand_text(
            (
                f"{request.question} Rationale: {request.rationale} "
                f"Proposed experiment: {request.proposed_experiment} "
                f"Expected evidence: {request.expected_evidence} "
                f"Capability: {request.capability.value}; "
                f"supported by current engines: {str(request.supported).lower()}."
            ),
            records,
            location="Research gaps",
            seeded_ids=tuple(request.supporting_evidence_ids),
            allow_structural_constants=True,
        )
        fragments.append(expanded)
        gap_blocks.append(f"- {expanded.text}")
    if not manifest.requests:
        gap_blocks.append(
            "No experiment requests were generated; no unexecuted experiment is claimed."
        )
    sections.append(
        PaperSection(heading="Research gaps and proposed experiments", blocks=gap_blocks)
    )
    return sections


def _build_figures(
    paths: ProjectPaths,
    context: EvidenceContext,
    figures: list[FigurePlan],
    records: dict[str, EvidenceRecord],
    out_dir: Path,
    fragments: list[ExpandedText],
) -> list[PaperFigure]:
    built: list[PaperFigure] = []
    known = context.evidence_ids()
    for figure in figures:
        _require_evidence(figure.evidence_ids, known, location=figure.figure_id)
        available = context.available_figures.get(figure.template.value, 0)
        if available <= 0:
            raise CompetitionBuildError(
                f"{figure.figure_id} requests unavailable template {figure.template.value!r}"
            )
        caption = expand_text(
            figure.caption,
            records,
            location=f"figure:{figure.figure_id}",
            seeded_ids=tuple(figure.evidence_ids),
        )
        fragments.append(caption)
        built_figure = build_figure(
            paths,
            context,
            figure.model_copy(update={"caption": caption.text}),
            out_dir,
        )
        built.append(built_figure)
    return built


def build_competition_ast(
    paths: ProjectPaths,
    context: EvidenceContext,
    plan: CompetitionPaperPlan,
    analysis: ExperimentAnalysisReport,
    figures: list[FigurePlan],
    manifest: CompetitionExperimentManifest,
    records: dict[str, EvidenceRecord],
) -> CompetitionBuildResult:
    out_dir = paths.root / "paper" / "competition" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fragments: list[ExpandedText] = []
    known = context.evidence_ids()
    title = expand_text(plan.title, records, location="Title")
    fragments.append(title)
    abstract_parts = []
    for point in plan.abstract_points:
        _require_evidence(point.evidence_ids, known, location="Abstract")
        expanded = expand_text(
            point.text,
            records,
            location="Abstract",
            seeded_ids=tuple(point.evidence_ids),
        )
        fragments.append(expanded)
        abstract_parts.append(expanded.text)
    abstract = "\n\n".join(abstract_parts)

    tables = _build_tables(context, plan, records, fragments)
    sections = _build_sections(
        context, plan, analysis, manifest, records, fragments
    )
    figures_built = _build_figures(
        paths, context, figures, records, out_dir, fragments
    )
    supported_claims = [
        claim
        for claim in context.claims
        if claim.status == "supported"
        and set(claim.evidence_ids) <= known
    ]
    references = [
        "Frozen problem specification: manifests/problem_spec.json.",
        "Verified evidence index: verified/numbers.json.",
        f"Source release: {context.release_path.relative_to(paths.root).as_posix()} "
        f"(sha256 {context.release_hash}).",
    ]
    bundle = paths.root / "paper" / "evidence_bundle"
    if bundle.exists():
        references.append("Evidence bundle: paper/evidence_bundle/.")
    references.append(
        "Experiment requests are proposals only; no requested experiment was executed "
        "in this stage."
    )
    for claim in supported_claims:
        fragments.append(
            ExpandedText(
                text=claim.statement,
                location=f"Supported claim: {claim.claim_id}",
                cited_ids=tuple(claim.evidence_ids),
            )
        )
    ast = PaperAST(
        title=title.text,
        abstract=abstract,
        sections=sections,
        tables=tables,
        figures=figures_built,
        claims=supported_claims,
        references=references,
    )
    return CompetitionBuildResult(ast=ast, fragments=fragments)


__all__ = [
    "CompetitionBuildError",
    "CompetitionBuildResult",
    "build_competition_ast",
]
