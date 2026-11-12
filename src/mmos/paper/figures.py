import json
import math
from datetime import UTC, datetime

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.paper import FigureSpec, PaperFigure

FIGURE_FORMATS = ("png", "pdf", "svg")


def create_metric_figure(
    paths: ProjectPaths,
    experiment_id: str,
    metric: str,
    *,
    higher_is_better: bool,
    records: list[EvidenceRecord],
    fold_values: dict[str, list[float]] | None = None,
) -> PaperFigure:
    """Render a bar chart from verified evidence only.

    No value is passed in except the evidence records, so a figure cannot show a
    number that was never verified.
    """
    if not records:
        raise ValueError(f"no evidence records for metric {metric!r}")

    ordered = sorted(records, key=lambda record: record.key)
    labels = [record.key.removeprefix("metrics.").removesuffix(f".{metric}") for record in ordered]
    values = [float(record.raw_value) for record in ordered]

    figure = Figure(figsize=(7, 4))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    direction = "higher is better" if higher_is_better else "lower is better"
    axis.set_title(f"{metric} by model ({direction})")
    axis.set_ylabel(metric)
    fold_note = ""
    if fold_values:
        positions = list(range(len(ordered)))
        for position, record, value in zip(positions, ordered, values, strict=True):
            model_id = record.key.removeprefix("metrics.").removesuffix(f".{metric}")
            folds = fold_values.get(model_id, [])
            deviation = (
                math.sqrt(sum((item - value) ** 2 for item in folds) / (len(folds) - 1))
                if len(folds) > 1
                else 0.0
            )
            axis.errorbar(
                position,
                value,
                yerr=deviation,
                fmt="o",
                color="#3b6ea5",
                capsize=4,
                zorder=3,
            )
            if folds:
                axis.scatter(
                    [position] * len(folds),
                    folds,
                    color="#d07b3b",
                    alpha=0.75,
                    s=18,
                    zorder=2,
                )
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        fold_sizes = {len(items) for items in fold_values.values() if items}
        fold_note = (
            f" Bars show fold standard deviation; dots show individual folds"
            f" (N={max(fold_sizes) if fold_sizes else 0})."
        )
    else:
        bars = axis.bar(labels, values, color="#3b6ea5")
        axis.bar_label(bars, labels=[record.display_value for record in ordered], padding=2)
        figure.autofmt_xdate(rotation=20, ha="right")
    figure.tight_layout()

    figures_dir = paths.verified_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_id = f"fig-{experiment_id}-{metric}"
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for file_format in FIGURE_FORMATS:
        target = figures_dir / f"{figure_id}.{file_format}"
        figure.savefig(target, dpi=200, format=file_format)
        files[file_format] = target.relative_to(paths.root).as_posix()
        hashes[file_format] = sha256_file(target)
    figure.clear()

    caption = f"Out-of-fold {metric} by model ({direction}).{fold_note}"
    spec = FigureSpec(
        figure_id=figure_id,
        caption=caption,
        experiment_id=experiment_id,
        metric=metric,
        evidence_ids=[record.evidence_id for record in ordered],
        files=files,
        hashes=hashes,
        created_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        figures_dir / f"{figure_id}.json", json.dumps(spec.model_dump(mode="json"), indent=2) + "\n"
    )

    return PaperFigure(
        figure_id=figure_id,
        caption=caption,
        path=files["png"],
        pdf_path=files["pdf"],
        svg_path=files["svg"],
        evidence_ids=spec.evidence_ids,
    )
