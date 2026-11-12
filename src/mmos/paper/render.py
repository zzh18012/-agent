import os
from collections.abc import Callable
from pathlib import Path

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.paper import PaperAST, PaperFigure, PaperTable


def _table_markdown(table: PaperTable) -> list[str]:
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table.rows]
    lines = [f"**Table {table.table_id}.** {table.caption}", "", header, divider, *rows, ""]
    return lines


def _render(ast: PaperAST, figure_href: Callable[[PaperFigure], str]) -> str:
    lines: list[str] = [f"# {ast.title}", "", "## Abstract", "", ast.abstract, ""]
    for section in ast.sections:
        lines.extend([f"## {section.heading}", ""])
        for block in section.blocks:
            lines.extend([block, ""])

    if ast.tables:
        lines.extend(["## Tables", ""])
        for table in ast.tables:
            lines.extend(_table_markdown(table))

    if ast.figures:
        lines.extend(["## Figures", ""])
        for figure in ast.figures:
            lines.extend(
                [
                    f"![{figure.caption}]({figure_href(figure)})",
                    "",
                    f"*Figure {figure.figure_id}.* {figure.caption}",
                    "",
                ]
            )

    lines.extend(["## Supported claims", ""])
    if ast.claims:
        for claim in ast.claims:
            evidence = ", ".join(claim.evidence_ids)
            lines.append(f"- **{claim.claim_id}**: {claim.statement} (evidence: {evidence})")
    else:
        lines.append("No supported claims were registered.")
    lines.append("")

    if ast.references:
        lines.extend(["## References", ""])
        lines.extend(f"- {reference}" for reference in ast.references)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_markdown(ast: PaperAST) -> str:
    return _render(ast, lambda figure: figure.path)


def render_paper(paths: ProjectPaths, ast: PaperAST) -> Path:
    paper_dir = paths.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    def figure_href(figure: PaperFigure) -> str:
        absolute = paths.root / figure.path
        return Path(os.path.relpath(absolute, paper_dir)).as_posix()

    target = paper_dir / "report.md"
    atomic_write_text(target, _render(ast, figure_href))
    return target
