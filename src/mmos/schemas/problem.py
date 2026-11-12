from pydantic import Field, model_validator

from mmos.schemas.common import MMOSModel


class Variable(MMOSModel):
    name: str = Field(min_length=1)
    meaning: str = ""
    unit: str | None = None


class SubProblem(MMOSModel):
    subproblem_id: str = Field(min_length=1)
    statement: str
    goal: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    known_variables: list[Variable] = Field(default_factory=list)
    unknown_variables: list[Variable] = Field(default_factory=list)
    decision_variables: list[Variable] = Field(default_factory=list)
    target_variables: list[Variable] = Field(default_factory=list)
    objective: str | None = None
    hard_constraints: list[str] = Field(default_factory=list)
    soft_constraints: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    time_structure: str | None = None
    spatial_structure: str | None = None
    uncertainty_sources: list[str] = Field(default_factory=list)
    required_deliverables: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ProblemSpec(MMOSModel):
    """Structured understanding of the task. It describes the problem, not a solution."""

    problem_id: str
    title: str
    overall_goal: str
    subproblems: list[SubProblem] = Field(min_length=1)
    global_assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    required_deliverables: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _dependency_graph_is_valid(self) -> "ProblemSpec":
        ids = [subproblem.subproblem_id for subproblem in self.subproblems]
        if len(ids) != len(set(ids)):
            raise ValueError("subproblem_id values must be unique")

        graph = {
            subproblem.subproblem_id: list(subproblem.depends_on) for subproblem in self.subproblems
        }
        for node, dependencies in graph.items():
            for dependency in dependencies:
                if dependency not in graph:
                    raise ValueError(
                        f"subproblem {node!r} depends on unknown subproblem {dependency!r}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError(f"subproblem dependency cycle detected at {node!r}")
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)
        return self
