import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel
from mmos.schemas.optimization import ExperimentStatus


class ObjectiveSpec(MMOSModel):
    objective_id: str
    sense: Literal["maximize"] = "maximize"
    definition: str


class ProjectItem(MMOSModel):
    project_id: str
    cost: float = Field(ge=0, allow_inf_nan=False)
    benefits: dict[str, float]


class MultiObjectiveSpec(MMOSModel):
    schema_version: int = 1
    problem_id: str
    experiment_id: str
    items: list[ProjectItem] = Field(min_length=1)
    budget: float = Field(ge=0, allow_inf_nan=False)
    objectives: list[ObjectiveSpec] = Field(min_length=2, max_length=2)
    method: Literal["epsilon_constraint"] = "epsilon_constraint"
    solver: Literal["ortools.cbc"] = "ortools.cbc"
    solver_time_limit_seconds: float = Field(default=10.0, gt=0)
    epsilon_policy: Literal["full_integer_range"] = "full_integer_range"
    epsilon_step: int = Field(default=1, ge=1)
    reference_point: list[float] = Field(min_length=2, max_length=2)
    instance_hash: str = ""
    spec_hash: str = ""

    @model_validator(mode="after")
    def _valid_spec(self) -> "MultiObjectiveSpec":
        if len(self.objectives) != 2:
            raise ValueError("exactly two objectives are required")
        objective_ids = [objective.objective_id for objective in self.objectives]
        if len(set(objective_ids)) != 2:
            raise ValueError("objective ids must be unique")
        if any(objective.sense != "maximize" for objective in self.objectives):
            raise ValueError("this slice supports maximize objectives only")
        item_ids = [item.project_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("project ids must be unique")
        for item in self.items:
            if set(item.benefits) != set(objective_ids):
                raise ValueError(
                    f"project {item.project_id} must have exactly the declared benefits"
                )
            if abs(item.cost - round(item.cost)) > 1e-9:
                raise ValueError("costs must be integer-valued")
            for value in item.benefits.values():
                if abs(value - round(value)) > 1e-9:
                    raise ValueError("benefits must be integer-valued")
                if value < 0:
                    raise ValueError("benefits must be nonnegative")
        if abs(self.budget - round(self.budget)) > 1e-9:
            raise ValueError("budget must be integer-valued")
        if any(value > 0 for value in self.reference_point):
            raise ValueError(
                "reference point must be weakly dominated by nonnegative benefits"
            )
        return self

    @property
    def primary_objective_id(self) -> str:
        return self.objectives[0].objective_id

    @property
    def epsilon_objective_id(self) -> str:
        return self.objectives[1].objective_id

    def epsilon_levels(self) -> list[int]:
        maximum = int(
            sum(item.benefits[self.epsilon_objective_id] for item in self.items)
        )
        levels = list(range(0, maximum + 1, self.epsilon_step))
        if not levels or levels[-1] != maximum:
            levels.append(maximum)
        return levels

    def computed_instance_hash(self) -> str:
        payload = {
            "problem_id": self.problem_id,
            "items": [
                {
                    "project_id": item.project_id,
                    "cost": float(item.cost),
                    "benefits": {key: float(value) for key, value in sorted(item.benefits.items())},
                }
                for item in sorted(self.items, key=lambda item: item.project_id)
            ],
            "budget": float(self.budget),
            "objectives": [
                {
                    "objective_id": objective.objective_id,
                    "sense": objective.sense,
                    "definition": objective.definition,
                }
                for objective in self.objectives
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def computed_spec_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"spec_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def with_hash(self) -> "MultiObjectiveSpec":
        instance_hash = self.computed_instance_hash()
        with_instance = self.model_copy(update={"instance_hash": instance_hash})
        return with_instance.model_copy(
            update={"spec_hash": with_instance.computed_spec_hash()}
        )


class ParetoVector(MMOSModel):
    f1: float
    f2: float


class ParetoSolution(MMOSModel):
    f1: float
    f2: float
    decision_vector: dict[str, int]


class EpsilonSolve(MMOSModel):
    epsilon: float
    status: Literal["optimal", "infeasible", "error"]
    f1: float | None = None
    f2: float | None = None
    decision_vector: dict[str, int] = Field(default_factory=dict)
    feasible: bool = False
    stage1_status: str = ""
    stage2_status: str = ""
    runtime_seconds: float = 0.0


class EpsilonCoverage(MMOSModel):
    epsilon_min: float
    epsilon_max: float
    epsilon_count: int
    successful_solves: int
    infeasible_levels: int
    error_levels: int


class EpsilonSensitivity(MMOSModel):
    step: int
    levels: int
    feasible_levels: int
    candidate_front_size: int
    precision: float
    recall: float
    exact_match: bool
    missing_vectors: list[ParetoVector] = Field(default_factory=list)


class FrontComparison(MMOSModel):
    candidate_front_size: int
    oracle_front_size: int
    precision: float
    recall: float
    exact_match: bool
    missing_vectors: list[ParetoVector] = Field(default_factory=list)
    extra_vectors: list[ParetoVector] = Field(default_factory=list)
    dominated_candidates: list[ParetoVector] = Field(default_factory=list)


class ParetoOracle(MMOSModel):
    instance_hash: str
    search_space_size: int
    feasible_solution_count: int
    pareto_objective_vectors: list[ParetoVector]
    pareto_decision_vectors: list[ParetoSolution]
    front_size: int
    set_size: int
    hypervolume: float


class MultiObjectiveResult(MMOSModel):
    experiment_id: str
    problem_id: str
    instance_hash: str
    spec_hash: str
    status: Literal["succeeded", "failed"]
    method: str
    solver: str
    solver_version: str
    primary_objective_id: str
    epsilon_objective_id: str
    epsilon_coverage: EpsilonCoverage
    reference_point: list[float]
    raw_epsilon_solutions: list[EpsilonSolve]
    candidate_pareto_front: list[ParetoVector]
    candidate_pareto_set: list[ParetoSolution]
    candidate_front_size: int
    candidate_set_size: int
    comparison: FrontComparison | None = None
    epsilon_sensitivity: list[EpsilonSensitivity] = Field(default_factory=list)
    hypervolume: float
    hypervolume_gap: float | None = None
    runtime_total: float
    solver_calls: int
    determinism: dict[str, str | int | bool] = Field(default_factory=dict)
    git_commit: str = "unknown"
    started_at: datetime
    completed_at: datetime
    notes: list[str] = Field(default_factory=list)


class MultiObjectiveExperimentRecord(MMOSModel):
    schema_version: int = 1
    experiment_id: str
    project_id: str
    task_family: Literal["multi_objective"] = "multi_objective"
    instance_hash: str
    spec_hash: str
    method: str
    solver: str
    solver_version: str
    git_commit: str = "unknown"
    mlflow_run_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    validation_status: dict[str, str] = Field(default_factory=dict)
