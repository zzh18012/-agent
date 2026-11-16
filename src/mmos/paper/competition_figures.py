import json
from pathlib import Path

from mmos.competition.context import CatalogEntry, EvidenceContext
from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.schemas.competition import FigurePlan, FigureTemplate
from mmos.schemas.paper import PaperFigure


class CompetitionFigureError(RuntimeError):
    pass


_TASK_ORDER = {
    "forecasting": 0,
    "optimization": 1,
    "simulation": 2,
    "mixed": 3,
    "unknown": 4,
}


def _write_figure_spec(
    target: Path,
    figure: FigurePlan,
    evidence: list[CatalogEntry],
) -> None:
    spec = {
        "figure_id": figure.figure_id,
        "template": figure.template.value,
        "caption": figure.caption,
        "evidence": [
            {
                "evidence_id": entry.evidence_id,
                "key": entry.key,
                "display_value": entry.display_value,
            }
            for entry in evidence
        ],
        "png_sha256": sha256_file(target),
    }
    atomic_write_text(
        target.with_suffix(".json"), json.dumps(spec, indent=2) + "\n"
    )


def _pipeline_schematic(
    paths: ProjectPaths, context: EvidenceContext, figure: FigurePlan, out_dir: Path
) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = sorted(
        context.experiments,
        key=lambda item: (_TASK_ORDER.get(item.task, 9), item.experiment_id),
    )
    if not experiments:
        raise CompetitionFigureError("no experiments available for pipeline schematic")
    figure_obj, axis = plt.subplots(figsize=(max(8, 3 * len(experiments)), 2.4))
    for index, experiment in enumerate(experiments):
        axis.text(
            index,
            0.5,
            f"{experiment.label}\n{experiment.task}\n{experiment.experiment_id}",
            ha="center",
            va="center",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#eef4fb",
                "edgecolor": "#1f77b4",
            },
        )
        if index:
            axis.annotate(
                "",
                xy=(index - 0.25, 0.5),
                xytext=(index - 0.75, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"},
            )
    axis.set_xlim(-0.5, len(experiments) - 0.5)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Frozen experiment lineage")
    target = out_dir / f"{figure.figure_id}.png"
    figure_obj.tight_layout()
    figure_obj.savefig(target, dpi=120)
    plt.close(figure_obj)
    _write_figure_spec(target, figure, [])
    return PaperFigure(
        figure_id=figure.figure_id,
        caption=figure.caption,
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=figure.evidence_ids,
    )


def _metric_bar(
    paths: ProjectPaths,
    context: EvidenceContext,
    figure: FigurePlan,
    out_dir: Path,
) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = figure.metric_prefix or ""
    entries = [
        entry
        for entry in context.entries
        if entry.numeric and entry.key.startswith(prefix)
    ]
    if not entries:
        raise CompetitionFigureError(
            f"no numeric evidence matches metric prefix {prefix!r}"
        )
    units = {entry.unit for entry in entries}
    if len(units) > 1:
        raise CompetitionFigureError("metric_bar evidence mixes units")
    labels = [entry.key.removeprefix(prefix).lstrip(".") or entry.key for entry in entries]
    values = [float(entry.raw_value) for entry in entries]
    figure_obj, axis = plt.subplots(figsize=(max(7, 0.6 * len(entries) + 3), 4))
    axis.bar(range(len(entries)), values, color="#4c72b0")
    axis.set_xticks(range(len(entries)), labels, rotation=45, ha="right", fontsize=8)
    unit = next(iter(units))
    axis.set_ylabel(unit or "value")
    axis.set_title(figure.caption)
    axis.grid(True, axis="y", alpha=0.3)
    target = out_dir / f"{figure.figure_id}.png"
    figure_obj.tight_layout()
    figure_obj.savefig(target, dpi=120)
    plt.close(figure_obj)
    _write_figure_spec(target, figure, entries)
    return PaperFigure(
        figure_id=figure.figure_id,
        caption=figure.caption,
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=figure.evidence_ids,
    )


def _uncertainty(
    paths: ProjectPaths,
    context: EvidenceContext,
    figure: FigurePlan,
    out_dir: Path,
) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lows = {
        entry.key.removesuffix(".ci_low"): entry
        for entry in context.entries
        if entry.numeric and entry.key.endswith(".ci_low")
    }
    highs = {
        entry.key.removesuffix(".ci_high"): entry
        for entry in context.entries
        if entry.numeric and entry.key.endswith(".ci_high")
    }
    prefixes = sorted(set(lows) & set(highs))
    if not prefixes:
        raise CompetitionFigureError("no paired confidence-interval evidence found")
    figure_obj, axis = plt.subplots(figsize=(8, max(3, 0.5 * len(prefixes) + 2)))
    for index, prefix in enumerate(prefixes):
        low = float(lows[prefix].raw_value)
        high = float(highs[prefix].raw_value)
        axis.plot([low, high], [index, index], color="#4c72b0", lw=2)
        axis.plot([low, high], [index, index], "|", color="#1f3b5c", ms=10)
    axis.set_yticks(range(len(prefixes)), prefixes, fontsize=8)
    axis.set_xlabel("confidence interval")
    axis.set_title(figure.caption)
    axis.grid(True, axis="x", alpha=0.3)
    target = out_dir / f"{figure.figure_id}.png"
    figure_obj.tight_layout()
    figure_obj.savefig(target, dpi=120)
    plt.close(figure_obj)
    _write_figure_spec(
        target,
        figure,
        [lows[prefix] for prefix in prefixes] + [highs[prefix] for prefix in prefixes],
    )
    return PaperFigure(
        figure_id=figure.figure_id,
        caption=figure.caption,
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=figure.evidence_ids,
    )


def build_figure(
    paths: ProjectPaths,
    context: EvidenceContext,
    figure: FigurePlan,
    out_dir: Path,
) -> PaperFigure:
    missing = sorted(set(figure.evidence_ids) - context.evidence_ids())
    if missing:
        raise CompetitionFigureError(
            "figure cites unknown evidence ids: " + ", ".join(missing)
        )
    if figure.template is FigureTemplate.PIPELINE_SCHEMATIC:
        return _pipeline_schematic(paths, context, figure, out_dir)
    if figure.template is FigureTemplate.METRIC_BAR:
        return _metric_bar(paths, context, figure, out_dir)
    if figure.template is FigureTemplate.UNCERTAINTY:
        return _uncertainty(paths, context, figure, out_dir)
    raise CompetitionFigureError(f"unsupported figure template: {figure.template}")


__all__ = ["CompetitionFigureError", "build_figure"]
