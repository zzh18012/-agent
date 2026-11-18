from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class Alternative(StrEnum):
    LESS = "less"
    GREATER = "greater"
    TWO_SIDED = "two_sided"


class BinomialInferenceSpec(MMOSModel):
    successes: int = Field(ge=0)
    trials: int = Field(gt=0)
    null_probability: float = Field(ge=0.0, le=1.0)
    alternative: Alternative
    confidence_level: float = Field(gt=0.0, lt=1.0)
    confidence_method: Literal["exact"] = "exact"

    @model_validator(mode="after")
    def _bounds(self) -> BinomialInferenceSpec:
        if self.successes > self.trials:
            raise ValueError("successes must be <= trials")
        return self


class BinomialInferenceResult(MMOSModel):
    experiment_id: str
    observed_proportion: float
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    confidence_method: Literal["exact"]
    alternative: Alternative
    input_hash: str
    spec_hash: str
    artifact_hash: str = ""


class AcceptanceSamplingDesignSpec(MMOSModel):
    good_quality_defect_rate: float = Field(ge=0.0, le=1.0)
    bad_quality_defect_rate: float = Field(ge=0.0, le=1.0)
    producer_risk_alpha: float = Field(gt=0.0, lt=1.0)
    consumer_risk_beta: float = Field(gt=0.0, lt=1.0)
    max_sample_size: int = Field(gt=0)
    lot_model: Literal["binomial_large_lot"] = "binomial_large_lot"

    @model_validator(mode="after")
    def _quality_order(self) -> AcceptanceSamplingDesignSpec:
        if not self.good_quality_defect_rate < self.bad_quality_defect_rate:
            raise ValueError("good_quality_defect_rate must be < bad_quality_defect_rate")
        return self


class OCPoint(MMOSModel):
    defect_rate: float
    acceptance_probability: float


class AcceptanceSamplingDesignResult(MMOSModel):
    experiment_id: str
    sample_size: int | None
    acceptance_number: int | None
    p_good: float
    p_bad: float
    probability_accept_good: float | None
    probability_accept_bad: float | None
    producer_risk_actual: float | None
    consumer_risk_actual: float | None
    producer_risk_limit: float
    consumer_risk_limit: float
    oc_curve_points: list[OCPoint]
    oc_grid_policy: str
    oc_grid_hash: str
    search_bound: int
    feasible: bool
    spec_hash: str
    artifact_hash: str = ""


class NLSStartResult(MMOSModel):
    start_index: int
    initial_parameters: list[float]
    final_parameters: list[float]
    cost: float
    sse: float
    success: bool
    status: int
    optimality: float
    nfev: int
    message: str


class ParameterInterval(MMOSModel):
    name: str
    estimate: float
    standard_error: float | None
    lower: float | None
    upper: float | None


class NonlinearLeastSquaresSpec(MMOSModel):
    forward_model_id: str = Field(min_length=1)
    x_values: list[float] = Field(min_length=2)
    y_values: list[float] = Field(min_length=2)
    parameter_names: list[str] = Field(min_length=1)
    lower_bounds: list[float]
    upper_bounds: list[float]
    initial_guesses: list[list[float]] = Field(min_length=1)
    loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "linear"
    ftol: float = Field(default=1e-10, gt=0.0)
    xtol: float = Field(default=1e-10, gt=0.0)
    gtol: float = Field(default=1e-10, gt=0.0)
    max_nfev: int = Field(default=2000, gt=0)

    @model_validator(mode="after")
    def _shape(self) -> NonlinearLeastSquaresSpec:
        if len(self.x_values) != len(self.y_values):
            raise ValueError("x_values and y_values must have the same length")
        width = len(self.parameter_names)
        if len(self.lower_bounds) != width or len(self.upper_bounds) != width:
            raise ValueError("bounds must match parameter_names")
        for guess in self.initial_guesses:
            if len(guess) != width:
                raise ValueError("every initial guess must match parameter_names")
        return self


class NonlinearLeastSquaresResult(MMOSModel):
    experiment_id: str
    forward_model_id: str
    forward_model_hash: str
    x_hash: str
    y_hash: str
    parameter_names: list[str]
    parameter_estimates: list[float]
    predictions: list[float]
    residuals: list[float]
    sse: float
    rmse: float
    jacobian: list[list[float]]
    jacobian_rank: int
    jacobian_condition_number: float | None
    optimizer_status: int
    optimizer_success: bool
    optimality: float
    all_start_results: list[NLSStartResult]
    selected_start: int
    spec_hash: str
    local_solution_validated: bool = False
    uncertainty_status: Literal[
        "available_local_linearization",
        "not_identifiable_or_unsupported",
    ]
    uncertainty_method: Literal["local_linearization"] = "local_linearization"
    approximate_uncertainty: bool = True
    parameter_intervals: list[ParameterInterval]
    artifact_hash: str = ""


__all__ = [
    "AcceptanceSamplingDesignResult",
    "AcceptanceSamplingDesignSpec",
    "Alternative",
    "BinomialInferenceResult",
    "BinomialInferenceSpec",
    "NLSStartResult",
    "NonlinearLeastSquaresResult",
    "NonlinearLeastSquaresSpec",
    "OCPoint",
    "ParameterInterval",
]
