from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from mmos.schemas.common import MMOSModel


class OptimalityClass(StrEnum):
    PROVEN_OPTIMAL = "proven_optimal"
    SOLVER_OPTIMAL = "solver_optimal"
    FEASIBLE_NOT_PROVEN_OPTIMAL = "feasible_not_proven_optimal"
    INFEASIBLE = "infeasible"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class EntryRole(StrEnum):
    CANDIDATE = "candidate"
    BASELINE = "baseline"


class BaselineRole(StrEnum):
    SANITY_BASELINE = "sanity_baseline"
    MATCHED_BASELINE = "matched_baseline"
    STRONG_BASELINE = "strong_baseline"
    HISTORICAL_BASELINE = "historical_baseline"


class MethodFamily(StrEnum):
    EXACT_SOLVER = "exact_solver"
    HEURISTIC = "heuristic"
    RULE_BASED = "rule_based"
    METAHEURISTIC = "metaheuristic"


class WinnerStatus(StrEnum):
    WINNER = "winner"
    EQUIVALENT = "equivalent"
    INCOMPARABLE = "incomparable"
    NO_WINNER = "no_winner"
    INVALID = "invalid"


class OptimizationTournamentEntry(MMOSModel):
    experiment_id: str
    model_id: str
    method_family: MethodFamily
    solver: str
    instance_hash: str
    objective_sense: Literal["minimize", "maximize"]
    feasible: bool
    valid: bool
    objective_value: float | None = None
    solver_status: str | None = None
    termination_condition: str | None = None
    best_bound: float | None = None
    absolute_gap: float | None = None
    relative_gap: float | None = None
    runtime_seconds: float | None = None
    time_limit_seconds: float | None = None
    validation_status: dict[str, str] = Field(default_factory=dict)
    independent_verification_method: str | None = None
    independent_verification_status: str | None = None
    optimality_class: OptimalityClass = OptimalityClass.UNKNOWN
    entry_role: EntryRole
    baseline_role: BaselineRole | None = None
    warnings: list[str] = Field(default_factory=list)


class ComparisonMatrixEntry(MMOSModel):
    experiment_a: str
    experiment_b: str
    comparable: bool
    reasons: list[str] = Field(default_factory=list)
    feasibility_relationship: str
    objective_relationship: str
    optimality_relationship: str
    warnings: list[str] = Field(default_factory=list)


class OptimizationTournamentResult(MMOSModel):
    tournament_id: str
    project_id: str
    instance_hash: str
    entries: list[OptimizationTournamentEntry] = Field(min_length=1)
    comparison_matrix: list[ComparisonMatrixEntry] = Field(default_factory=list)
    winner_experiment_id: str | None = None
    winner_status: WinnerStatus
    equivalent_experiment_ids: list[str] = Field(default_factory=list)
    baseline_experiment_ids: list[str] = Field(default_factory=list)
    incomparable_pairs: list[list[str]] = Field(default_factory=list)
    decision_trace: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    objective_tolerance: float = 1e-6
    created_at: datetime
