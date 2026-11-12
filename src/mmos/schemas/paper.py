from datetime import datetime

from pydantic import Field

from mmos.schemas.common import MMOSModel
from mmos.schemas.evidence import ClaimRecord


class PaperTable(MMOSModel):
    table_id: str
    caption: str
    columns: list[str]
    rows: list[list[str]]
    evidence_ids: list[str] = Field(default_factory=list)


class PaperFigure(MMOSModel):
    figure_id: str
    caption: str
    path: str
    pdf_path: str | None = None
    svg_path: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class FigureSpec(MMOSModel):
    figure_id: str
    caption: str
    experiment_id: str
    metric: str
    evidence_ids: list[str]
    files: dict[str, str]
    hashes: dict[str, str]
    created_at: datetime


class PaperSection(MMOSModel):
    heading: str
    blocks: list[str] = Field(default_factory=list)


class PaperAST(MMOSModel):
    """Structured paper intermediate representation. Renderers must not add numbers."""

    title: str
    abstract: str
    sections: list[PaperSection] = Field(default_factory=list)
    tables: list[PaperTable] = Field(default_factory=list)
    figures: list[PaperFigure] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
