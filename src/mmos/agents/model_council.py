from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model

from mmos.agents.catalog import registry_catalog
from mmos.registries import ModelRegistry
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.proposal import (
    CouncilDecision,
    CritiqueReport,
    ExpertRole,
    ExpertSelection,
    ModelProposal,
    ProposalBatch,
    ProposalCritique,
    ProposalScore,
)
from mmos.schemas.research import ResearchResult

SELECTOR_PROMPT = """\
You chair a model council. Choose 2-4 expert roles whose viewpoints are genuinely
useful for this problem. Do not include roles that would only repeat each other.
Return the selected roles and a short rationale.
"""

EXPERT_PROMPT = """\
You are one independent expert on a model council. Propose 1-2 concrete models.

Hard requirements:
- subproblem_id must be exactly one of the ids you are given.
- registry_model_ids must contain only ids from the provided catalog, and at least one.
- Fill every proposal field; no invented data, numbers, or citations.
- Include a validation plan and the baseline this model must be compared against.
- Do not assume what other experts will propose; argue from the problem and research.
"""

CRITIC_PROMPT = """\
You are a cross-reviewer on a model council. Critique every proposal you are given.
Use the exact proposal_id. Focus on assumptions, leakage risk, data fit, feasibility,
overfitting, baseline fairness, and whether the proposal answers the subproblem.
Be specific and fair; praise only what the proposal actually establishes.
"""

JUDGE_PROMPT = """\
You are the council judge. Score each proposal from 1 (poor) to 5 (excellent) on:
problem_match, theoretical_soundness, data_fit, interpretability, feasibility,
paper_value, and risk (1 = low risk, 5 = high risk). Then select 1-3 proposals for
prototyping. selected_for_prototype must contain only ids of scored proposals.

You are not choosing the final winner; real experiments decide that. Prefer proposals
that are executable, fairly comparable, and that include a baseline.
"""


class CouncilError(RuntimeError):
    pass


@dataclass(frozen=True)
class CouncilOutcome:
    roles: ExpertSelection
    proposals: list[ModelProposal]
    critiques: list[ProposalCritique]
    decision: CouncilDecision
    notes: list[str]


def normalize_proposals(
    batch: ProposalBatch,
    role: ExpertRole,
    spec: ProblemSpec,
    registry: ModelRegistry,
) -> tuple[list[ModelProposal], list[str]]:
    """Drop proposals that name unknown subproblems or unregistered models."""
    subproblem_ids = {subproblem.subproblem_id for subproblem in spec.subproblems}
    proposals: list[ModelProposal] = []
    notes: list[str] = []

    for index, proposal in enumerate(batch.proposals, start=1):
        if proposal.subproblem_id not in subproblem_ids:
            notes.append(
                f"dropped proposal {proposal.name!r}: unknown subproblem {proposal.subproblem_id!r}"
            )
            continue
        registry_ids = [
            model_id for model_id in proposal.registry_model_ids if registry.has(model_id)
        ]
        if not registry_ids:
            notes.append(f"dropped proposal {proposal.name!r}: no registered executable model")
            continue
        proposals.append(
            proposal.model_copy(
                update={
                    "proposal_id": f"prop-{role.value}-{index}",
                    "problem_id": spec.problem_id,
                    "expert_role": role,
                    "registry_model_ids": registry_ids,
                }
            )
        )
    return proposals, notes


def normalize_decision(
    decision: CouncilDecision, proposals: list[ModelProposal]
) -> tuple[CouncilDecision, list[str]]:
    valid_ids = {proposal.proposal_id for proposal in proposals}
    notes: list[str] = []

    scores: list[ProposalScore] = []
    seen: set[str] = set()
    for score in decision.scores:
        if score.proposal_id not in valid_ids:
            notes.append(f"judge scored unknown proposal {score.proposal_id!r}")
            continue
        if score.proposal_id in seen:
            continue
        seen.add(score.proposal_id)
        scores.append(score)

    selected = [
        proposal_id
        for proposal_id in decision.selected_for_prototype
        if proposal_id in valid_ids and proposal_id in seen
    ]
    if scores and not selected:
        best = max(
            scores,
            key=lambda score: (
                score.problem_match
                + score.theoretical_soundness
                + score.data_fit
                + score.interpretability
                + score.feasibility
                + score.paper_value
                - score.risk
            ),
        )
        selected = [best.proposal_id]
        notes.append(
            f"judge selected no valid proposal; fell back to top-scored {best.proposal_id!r}"
        )
    if not scores:
        raise CouncilError("judge returned no scores for any real proposal")

    notes.append("selection is for prototyping only; experiments decide the final model")
    return (
        CouncilDecision(
            scores=scores,
            selected_for_prototype=selected,
            notes=decision.notes,
        ),
        notes,
    )


class ModelCouncil:
    def __init__(self, model: Model | str) -> None:
        self._selector: Agent[None, ExpertSelection] = Agent(
            model, output_type=ExpertSelection, system_prompt=SELECTOR_PROMPT, retries=2
        )
        self._expert: Agent[None, ProposalBatch] = Agent(
            model, output_type=ProposalBatch, system_prompt=EXPERT_PROMPT, retries=2
        )
        self._critic: Agent[None, CritiqueReport] = Agent(
            model, output_type=CritiqueReport, system_prompt=CRITIC_PROMPT, retries=2
        )
        self._judge: Agent[None, CouncilDecision] = Agent(
            model, output_type=CouncilDecision, system_prompt=JUDGE_PROMPT, retries=2
        )

    def propose(
        self,
        spec: ProblemSpec,
        research: ResearchResult,
        *,
        registry: ModelRegistry,
        data_summary: str | None = None,
    ) -> CouncilOutcome:
        catalog = registry_catalog(registry)
        problem_json = spec.model_dump_json(indent=2)
        methods = "\n".join(
            f"- {method.name} ({method.method_id}): {method.core_idea}"
            for method in research.methods
        )
        subproblem_ids = ", ".join(subproblem.subproblem_id for subproblem in spec.subproblems)

        selection = self._selector.run_sync(
            f"Problem specification:\n{problem_json}\n\nChoose the council roles."
        ).output

        proposals: list[ModelProposal] = []
        notes: list[str] = []
        for role in selection.roles:
            batch = self._expert.run_sync(
                f"Expert role: {role.value}\n\n"
                f"Problem specification:\n{problem_json}\n\n"
                f"Research methods:\n{methods}\n\n"
                f"Allowed subproblem ids: {subproblem_ids}\n\n"
                f"Executable model catalog:\n{catalog}\n\n"
                + (f"Data summary:\n{data_summary}\n\n" if data_summary else "")
                + "Propose 1-2 models as this expert."
            ).output
            normalized, proposal_notes = normalize_proposals(batch, role, spec, registry)
            proposals.extend(normalized)
            notes.extend(proposal_notes)

        if not proposals:
            raise CouncilError("no proposal referenced a known subproblem and a registered model")

        proposal_ids = {proposal.proposal_id for proposal in proposals}
        critiques: list[ProposalCritique] = []
        for role in selection.roles:
            report = self._critic.run_sync(
                f"Reviewer role: {role.value}\n\n"
                f"Problem specification:\n{problem_json}\n\n"
                f"Proposals:\n"
                + "\n".join(proposal.model_dump_json(indent=2) for proposal in proposals)
                + "\n\nCritique each proposal using its exact proposal_id."
            ).output
            critiques.extend(
                critique for critique in report.critiques if critique.proposal_id in proposal_ids
            )

        critique_text = "\n".join(
            f"- {critique.proposal_id}: recommendation={critique.recommendation}; "
            f"weaknesses={'; '.join(critique.weaknesses) or 'none stated'}; "
            f"risks={'; '.join(critique.risks) or 'none stated'}"
            for critique in critiques
        )

        decision: CouncilDecision | None = None
        last_error: Exception | None = None
        for _attempt in range(2):
            candidate = self._judge.run_sync(
                f"Problem specification:\n{problem_json}\n\n"
                f"Proposals:\n"
                + "\n".join(proposal.model_dump_json(indent=2) for proposal in proposals)
                + f"\n\nCritiques:\n{critique_text or 'no valid critiques'}\n\n"
                "Score every proposal and select 1-3 for prototyping."
            ).output
            try:
                decision, decision_notes = normalize_decision(candidate, proposals)
            except CouncilError as exc:
                last_error = exc
                continue
            notes.extend(decision_notes)
            break

        if decision is None:
            raise CouncilError(f"judge produced no usable decision: {last_error}")

        return CouncilOutcome(
            roles=selection,
            proposals=proposals,
            critiques=critiques,
            decision=decision,
            notes=notes,
        )
