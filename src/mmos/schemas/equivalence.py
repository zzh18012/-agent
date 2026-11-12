from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel


class PairedComparison(MMOSModel):
    candidate_model: str
    baseline_model: str
    information_set_match: bool
    n_pairs: int
    mean_loss_difference: float | None = None
    median_loss_difference: float | None = None
    relative_improvement: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    threshold: float | None = None
    status: Literal["equivalent", "different", "inconclusive", "incomparable"]
    notes: list[str] = Field(default_factory=list)
