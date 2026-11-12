import json
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
    bars = axis.bar(labels, values, color="#3b6ea5")
    axis.set_ylabel(metric)
    direction = "higher is better" if higher_is_better else "lower is better"
    axis.set_title(f"{metric} by model ({direction})")
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

    caption = f"Cross-validated {metric} by model ({direction})."
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
