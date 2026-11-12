from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class MethodCard(MMOSModel):
    """What a method is, when it applies, and how it fails. Not a recommendation."""

    method_id: str
    name: str
    problem_types: list[str] = Field(min_length=1)
    core_idea: str
    mathematical_assumptions: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    applicable_when: list[str] = Field(default_factory=list)
    not_applicable_when: list[str] = Field(default_factory=list)
    implementation_tools: list[str] = Field(default_factory=list)
    validation_approach: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class ResearchResult(MMOSModel):
    problem_id: str
    methods: list[MethodCard] = Field(min_length=1)

    @model_validator(mode="after")
    def _method_ids_are_unique(self) -> "ResearchResult":
        ids = [method.method_id for method in self.methods]
        if len(ids) != len(set(ids)):
            raise ValueError("method_id values must be unique")
        return self
