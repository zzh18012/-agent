from datetime import datetime
from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel


class PipelineState(MMOSModel):
    schema_version: Literal[1] = 1
    target_column: str
    feature_set: list[str] = Field(default_factory=list)
    primary_metric: str
    higher_is_better: bool
    metrics: list[str] | None = None
    n_splits: int
    random_seed: int
    time_column: str | None = None
    forecast_horizon: int | None = None
    seasonal_period: int | None = None
    improvement_threshold: float | None = None
    experiment_id: str | None = None
    tournament_id: str | None = None


class ReleaseSummary(MMOSModel):
    release_id: str
    experiment_id: str
    tournament_id: str
    selected_model: str
    challenger_model: str | None
    primary_metric: str
    metric_display_value: str
    validation: dict[str, str]
    red_team_overall: str
    red_team_blocking: bool
    selection_status: str = "winner"
    claim_ids: list[str]
    supported_claims: int
    paper_path: str
    created_at: datetime
    notes: list[str] = Field(default_factory=list)
