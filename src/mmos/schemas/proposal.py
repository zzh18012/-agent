from enum import StrEnum

from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class ExpertRole(StrEnum):
    STATISTICIAN = "statistician"
    ML_SCIENTIST = "ml_scientist"
    OPTIMIZATION_SCIENTIST = "optimization_scientist"
    MECHANISTIC_MODELER = "mechanistic_modeler"
    SIMULATION_SCIENTIST = "simulation_scientist"
    CAUSAL_SCIENTIST = "causal_scientist"
    DECISION_SCIENTIST = "decision_scientist"


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelProposal(MMOSModel):
    proposal_id: str
    problem_id: str
    subproblem_id: str
    name: str
    model_family: str
    expert_role: ExpertRole
    description: str
    assumptions: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    objective: str | None = None
    constraints: list[str] = Field(default_factory=list)
    required_data: list[str] = Field(default_factory=list)
    feature_requirements: list[str] = Field(default_factory=list)
    expected_strengths: list[str] = Field(default_factory=list)
    expected_weaknesses: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    implementation_plan: list[str] = Field(default_factory=list)
    validation_plan: list[str] = Field(default_factory=list)
    baseline_requirements: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    estimated_complexity: Complexity = Complexity.MEDIUM
    registry_model_ids: list[str] = Field(default_factory=list)


class ProposalScore(MMOSModel):
    proposal_id: str
    problem_match: int = Field(ge=1, le=5)
    theoretical_soundness: int = Field(ge=1, le=5)
    data_fit: int = Field(ge=1, le=5)
    interpretability: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    paper_value: int = Field(ge=1, le=5)
    risk: int = Field(ge=1, le=5, description="1 = low risk, 5 = high risk")
    rationale: str


class CouncilDecision(MMOSModel):
    """Which proposals enter prototyping. The final winner is decided by experiments."""

    scores: list[ProposalScore] = Field(min_length=1)
    selected_for_prototype: list[str] = Field(min_length=1)
    notes: str = ""

    @model_validator(mode="after")
    def _selection_is_scored(self) -> "CouncilDecision":
        scored = {score.proposal_id for score in self.scores}
        unknown = [
            proposal_id for proposal_id in self.selected_for_prototype if proposal_id not in scored
        ]
        if unknown:
            raise ValueError(f"selected proposals have no score: {', '.join(unknown)}")
        return self

    @property
    def rejected(self) -> list[str]:
        selected = set(self.selected_for_prototype)
        return [score.proposal_id for score in self.scores if score.proposal_id not in selected]


class ExpertSelection(MMOSModel):
    roles: list[ExpertRole] = Field(min_length=2, max_length=4)
    rationale: str

    @model_validator(mode="after")
    def _roles_are_unique(self) -> "ExpertSelection":
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("expert roles must be unique")
        return self


class ProposalBatch(MMOSModel):
    proposals: list[ModelProposal] = Field(min_length=1)


class ProposalList(MMOSModel):
    proposals: list[ModelProposal] = Field(min_length=1)


class ProposalCritique(MMOSModel):
    proposal_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendation: str


class CritiqueReport(MMOSModel):
    critiques: list[ProposalCritique] = Field(min_length=1)
