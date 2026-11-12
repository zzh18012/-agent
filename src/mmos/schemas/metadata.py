from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class UnitStatus(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    SYNTHETIC = "synthetic_units"


class FeatureAvailability(StrEnum):
    HISTORICAL_ONLY = "historical_only"
    KNOWN_FUTURE = "known_future"
    FORECASTED_FUTURE = "forecasted_future"
    STATIC = "static"
    UNKNOWN = "unknown"


class FeatureMetadata(MMOSModel):
    name: str
    availability: FeatureAvailability = FeatureAvailability.UNKNOWN


class TemporalMetadata(MMOSModel):
    is_temporal: bool = False
    time_column: str | None = None
    frequency: str | None = None
    forecast_horizon: int | None = Field(default=None, ge=1)
    seasonal_period: int | None = Field(default=None, ge=2)
    ordering_required: bool = False
    future_feature_policy: str | None = None

    @model_validator(mode="after")
    def _temporal_fields_are_complete(self) -> "TemporalMetadata":
        if self.is_temporal:
            if self.time_column is None:
                raise ValueError("temporal data requires a time_column")
            if self.forecast_horizon is None:
                raise ValueError("temporal data requires a forecast_horizon")
        return self


class DatasetProvenance(MMOSModel):
    is_synthetic: bool = False
    generator: str | None = None
    generation_parameters: dict[str, str | int | float] = Field(default_factory=dict)
    seed: int | None = None
    code_path: str | None = None
    description: str | None = None


class DatasetMetadata(MMOSModel):
    """Optional user-supplied sidecar: dataset_metadata.json."""

    schema_version: Literal[1] = 1
    target_name: str
    target_description: str | None = None
    target_unit: str | None = None
    unit_status: UnitStatus = UnitStatus.UNKNOWN
    temporal: TemporalMetadata = Field(default_factory=TemporalMetadata)
    features: list[FeatureMetadata] = Field(default_factory=list)
    provenance: DatasetProvenance = Field(default_factory=DatasetProvenance)

    @model_validator(mode="after")
    def _unit_status_is_consistent(self) -> "DatasetMetadata":
        if self.unit_status is UnitStatus.KNOWN and not self.target_unit:
            raise ValueError("unit_status 'known' requires target_unit")
        return self


class TargetSummary(MMOSModel):
    """Target context computed from the real data used by an experiment."""

    name: str
    description: str | None = None
    unit: str | None = None
    unit_status: UnitStatus = UnitStatus.UNKNOWN
    dtype: str
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    is_positive: bool | None = None
    is_synthetic: bool = False
    generator: str | None = None
    generation_seed: int | None = None
    generation_code_path: str | None = None
    generation_parameters: str | None = None
    rmse_std_ratio: float | None = None
