from pydantic_ai import Agent
from pydantic_ai.models import Model

from mmos.agents.catalog import registry_catalog
from mmos.registries import ModelRegistry
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.research import ResearchResult

SYSTEM_PROMPT = """\
You are the Research Agent of a mathematical modeling system.

Survey candidate methods for the given problem. You do not choose the final model;
a separate council and real experiments decide that.

For each method produce a MethodCard with:
- problem types it addresses, the core idea, and its mathematical assumptions;
- data requirements, strengths, weaknesses, and known failure modes;
- when it applies and when it does not;
- implementation tools and a validation approach.

Rules:
- Cover methods that are genuinely suitable for the stated subproblems, including
  simple baselines. Do not invent methods or citations.
- Prefer tools that appear in the provided model catalog when they fit.
- Record limitations honestly; the council needs them.
- Use unique short method ids (for example "m1", "m2").
"""


class ResearchAgent:
    def __init__(self, model: Model | str) -> None:
        self._agent: Agent[None, ResearchResult] = Agent(
            model,
            output_type=ResearchResult,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
        )

    def research(self, spec: ProblemSpec, *, registry: ModelRegistry) -> ResearchResult:
        prompt = (
            "Problem specification:\n"
            f"{spec.model_dump_json(indent=2)}\n\n"
            "Available executable models:\n"
            f"{registry_catalog(registry)}\n\n"
            "Produce 3-6 method cards relevant to these subproblems."
        )
        result = self._agent.run_sync(prompt)
        return result.output.model_copy(update={"problem_id": spec.problem_id})
