from pydantic_ai import Agent
from pydantic_ai.models import Model

from mmos.schemas.problem import ProblemSpec

SYSTEM_PROMPT = """\
You are the Problem Analyst of a mathematical modeling system.

Your job is to understand the task, not to solve it. Do not recommend models,
algorithms, or tools; a separate research stage does that.

Extract, per subproblem:
- what is given (inputs, known variables) and what must be produced (outputs);
- decision variables, target variables, and the objective;
- hard constraints, soft constraints, and evaluation criteria;
- time structure, spatial structure, and sources of uncertainty;
- required deliverables, assumptions that must be made, and genuine ambiguities;
- data or decision dependencies on other subproblems (use only existing ids).

Rules:
- Split the task into independent or clearly ordered subproblems when the statement
  contains distinct questions; keep a single subproblem otherwise.
- Use subproblem ids that are short and unique (for example "q1", "q2").
- Never invent numbers, data, or constraints that the statement does not support.
- If something is genuinely unclear, record it in ambiguities instead of guessing.
- Write all free-text fields in the language of the problem statement.
"""


class ProblemAnalyst:
    def __init__(self, model: Model | str) -> None:
        self._agent: Agent[None, ProblemSpec] = Agent(
            model,
            output_type=ProblemSpec,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
        )

    def analyze(self, problem_text: str, *, problem_id: str) -> ProblemSpec:
        prompt = (
            f"The project identifier is {problem_id!r}. Use exactly this value for problem_id.\n\n"
            f"Problem statement:\n\n{problem_text}"
        )
        result = self._agent.run_sync(prompt)
        # The identifier is assigned by the system, not trusted to the model.
        return result.output.model_copy(update={"problem_id": problem_id})
