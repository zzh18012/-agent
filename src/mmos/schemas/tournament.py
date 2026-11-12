from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel
from mmos.schemas.equivalence import PairedComparison


class TournamentEntry(MMOSModel):
    model_id: str
    experiment_id: str
    metrics: dict[str, float]
    rank: int = Field(ge=1)
    status: Literal["selected", "finalist", "eliminated", "equivalent", "incomparable"]
    reason: str


class TournamentResult(MMOSModel):
    tournament_id: str
    primary_metric: str
    higher_is_better: bool
    dataset_hash: str
    entries: list[TournamentEntry]
    selected_model: str | None = None
    challenger_model: str | None = None
    blocked: bool = False
    selection_status: Literal["winner", "equivalent", "incomparable", "blocked", "no_candidate"] = (
        "winner"
    )
    equivalence: list[PairedComparison] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
