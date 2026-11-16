import json
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from mmos.competition.context import EvidenceContext
from mmos.competition.evidence_ids import clean_evidence_ids
from mmos.schemas.competition import (
    CompetitionPaperPlan,
    ExperimentAnalysisReport,
    FigurePlan,
    FigurePlanBatch,
    MathAuditReport,
    SectionKind,
)

ARCHITECT_PROMPT = """\
You are the Paper Architect of a mathematical-modeling competition paper.

Build a paper plan from the frozen project problem and the verified evidence catalog.
You do not solve the model and you do not invent results.

Rules:
- Use every required section kind exactly once, in the catalog's required order.
- Each section needs at least one prose block; write paper prose, not an outline.
- Every numeric value in prose must be written as an evidence placeholder
  `[[ev:EVIDENCE_ID]]`; never write a literal number, percentage, difference, or derived count.
- Cite only evidence ids present in the catalog. Every section needs at least one.
- Tables must use only templates listed with a positive available count. Prefer
  experiment_inventory and evidence_lineage when metric_comparison is unavailable.
- Leave figures empty; a separate FigureDesigner selects figures.
- Assumptions must be explicit and testable. Notation entries define symbols and units.
- Limitations must include missing analyses suggested by the catalog, with no invented runs.
- capability_needs must name a real gap, cite evidence, and propose a concrete experiment.
  Every numeric value in capability_needs must also be an evidence placeholder.
- Do not write PASS/FAIL/verdict language.
- Do not assert that residuals are iid, identically distributed, exchangeable, or
  representative; describe the frozen resampling protocol as an assumption only.
- Do not claim out-of-sample validity, robustness, service guarantees, capacity
  enforcement, reproducibility, or model superiority. Use a cautious title such as
  "Conditional Simulation of a Frozen Production Plan" when the evidence supports it.
- Use the provided `model_facts` equations as the authoritative formulations. Do not
  invent alternative recurrences, denominators, or slope rules. Separate stocks from
  per-period flows and give exact notation units. State that the demand floor is part of
  the protocol, not a tested safeguard. Do not state numeric capacities, costs, or initial
  inventory unless an evidence id exists; otherwise refer to the frozen plan artifact and
  mark the gap as unavailable.
- Put missing analyses (comparator, sensitivity, pool bootstrap, held-out fold) in
  limitations and capability_needs, not in unsupported claims.
- Keep output compact: at most 4 abstract points (each <= 60 words), exactly one
  block per section (<= 90 words), at most 2 claim_points per section, at most 5
  assumptions, at most 8 notation entries, at most 5 limitations, at most 3 tables,
  and at most 4 capability_needs.
"""

MATH_AUDIT_PROMPT = """\
You are the independent mathematical auditor. Audit the model formulation, assumptions, and
notation, not the writing style.

Rules:
- Report concrete issues: unsupported assumptions, dimensional/unit errors, ambiguous
  symbols, missing constraints, invalid objective/constraint relationships, unjustified
  approximations, and mathematical claims not supported by the evidence catalog.
- Do not assign an overall pass/fail; there is no verdict field.
- Do not invent numbers or experimental results.
- severity: minor (clarity), major (weakens a result), critical (invalidates a main claim).
- required_fix must be actionable.
- Return at most 8 concrete issues.
"""

EXPERIMENT_ANALYST_PROMPT = """\
You are the independent experiment analyst. Judge whether the verified experiments support
the paper's planned analyses and identify the most valuable missing experiments.

Rules:
- Classify findings as supported (evidence directly supports), tentative (limited/indirect),
  or unsupported (no supporting evidence).
- Cite evidence ids for every finding.
- Look for missing sensitivity analysis, ablations, comparator baselines, uncertainty
  quantification, out-of-sample/hold-out evaluation, and capacity/parameter sweeps.
- requested_checks is a structured experiment backlog; each check cites evidence and names
  a capability from prediction, forecasting, optimization, simulation, graph,
  multi_objective, mixed, or unsupported. Every numeric value in requested_checks must
  be an evidence placeholder.
- Do not claim that any requested experiment was run. Do not assign a pass/fail verdict.
- Every numeric value in findings and limitations must be written as an evidence
  placeholder `[[ev:EVIDENCE_ID]]`; never write literal numbers, percentages, or derived
  counts.
- Return at most 8 findings and at most 5 requested checks.
"""

FIGURE_PROMPT = """\
You are the independent figure designer. Select only figures that can be rendered from the
provided evidence and supported template counts.

Rules:
- Use only templates with a positive count.
- Cite evidence ids for every figure.
- Every numeric value in a caption must be an evidence placeholder `[[ev:EVIDENCE_ID]]`;
  never write literal numbers or percentages.
- metric_bar requires a metric_prefix matching available numeric keys.
- Do not invent data, annotations, numeric values, or trends. Do not include a figure whose
  template count is zero.
- Return zero figures if no template is supportable.
- Return at most 2 figures.
"""


class CompetitionAgentError(RuntimeError):
    pass


class ArchitectProtocol(Protocol):
    def plan(self, context: EvidenceContext) -> CompetitionPaperPlan: ...


class MathAuditorProtocol(Protocol):
    def audit(
        self, context: EvidenceContext, plan: CompetitionPaperPlan
    ) -> MathAuditReport: ...


class ExperimentAnalystProtocol(Protocol):
    def analyze(
        self, context: EvidenceContext, plan: CompetitionPaperPlan
    ) -> ExperimentAnalysisReport: ...


class FigureDesignerProtocol(Protocol):
    def design(
        self,
        context: EvidenceContext,
        plan: CompetitionPaperPlan,
        analysis: ExperimentAnalysisReport,
    ) -> list[FigurePlan]: ...


@dataclass(frozen=True)
class CompetitionAgents:
    architect: ArchitectProtocol
    math_auditor: MathAuditorProtocol
    experiment_analyst: ExperimentAnalystProtocol
    figure_designer: FigureDesignerProtocol


def _payload(context: EvidenceContext) -> str:
    return json.dumps(context.prompt_payload(), indent=2, sort_keys=True)


def _clean_plan(plan: CompetitionPaperPlan) -> CompetitionPaperPlan:
    abstract_points = [
        point.model_copy(
            update={"evidence_ids": clean_evidence_ids(point.evidence_ids)}
        )
        for point in plan.abstract_points
    ]
    sections = []
    for section in plan.sections:
        claim_points = [
            point.model_copy(
                update={"evidence_ids": clean_evidence_ids(point.evidence_ids)}
            )
            for point in section.claim_points
        ]
        sections.append(
            section.model_copy(
                update={
                    "evidence_ids": clean_evidence_ids(section.evidence_ids),
                    "claim_points": claim_points,
                }
            )
        )
    tables = [
        table.model_copy(
            update={"evidence_ids": clean_evidence_ids(table.evidence_ids)}
        )
        for table in plan.tables
    ]
    needs = [
        need.model_copy(
            update={
                "supporting_evidence_ids": clean_evidence_ids(
                    need.supporting_evidence_ids
                )
            }
        )
        for need in plan.capability_needs
    ]
    return plan.model_copy(
        update={
            "abstract_points": abstract_points,
            "sections": sections,
            "tables": tables,
            "capability_needs": needs,
        }
    )


def _model_section_payload(plan: CompetitionPaperPlan) -> dict:
    kinds = {
        SectionKind.ASSUMPTIONS,
        SectionKind.NOTATION,
        SectionKind.MODEL,
        SectionKind.SOLUTION,
    }
    return {
        "title": plan.title,
        "assumptions": plan.assumptions,
        "notation": [entry.model_dump(mode="json") for entry in plan.notation],
        "sections": [
            section.model_dump(mode="json")
            for section in plan.sections
            if section.kind in kinds
        ],
    }


class PaperArchitect:
    def __init__(self, model: Model | str) -> None:
        self._agent: Agent[None, CompetitionPaperPlan] = Agent(
            model,
            output_type=CompetitionPaperPlan,
            system_prompt=ARCHITECT_PROMPT,
            retries=2,
            model_settings=ModelSettings(temperature=0, max_tokens=8192),
        )

    def _finalize(
        self, plan: CompetitionPaperPlan, *, keep_tables: bool = True
    ) -> CompetitionPaperPlan:
        plan = _clean_plan(plan)
        plan = plan.model_copy(update={"figures": []})
        if not keep_tables:
            return plan
        tables = [
            table.model_copy(update={"table_id": f"tab-c{index}"})
            for index, table in enumerate(plan.tables, start=1)
        ]
        return plan.model_copy(update={"tables": tables})

    def plan(self, context: EvidenceContext) -> CompetitionPaperPlan:
        result = self._agent.run_sync(
            "Build the competition paper plan from this frozen context:\n\n"
            + _payload(context)
        )
        return self._finalize(result.output)

    def revise(
        self,
        context: EvidenceContext,
        plan: CompetitionPaperPlan,
        audit: MathAuditReport,
    ) -> CompetitionPaperPlan:
        prompt = (
            "Revise the competition paper plan to resolve these independent mathematical "
            "audit issues without inventing evidence or numbers:\n\n"
            + json.dumps(audit.model_dump(mode="json"), indent=2)
            + "\n\nCurrent plan:\n\n"
            + json.dumps(plan.model_dump(mode="json"), indent=2)
            + "\n\nFrozen context:\n\n"
            + _payload(context)
        )
        result = self._agent.run_sync(prompt)
        return self._finalize(result.output)


class MathAuditor:
    def __init__(self, model: Model | str) -> None:
        self._agent: Agent[None, MathAuditReport] = Agent(
            model,
            output_type=MathAuditReport,
            system_prompt=MATH_AUDIT_PROMPT,
            retries=2,
            model_settings=ModelSettings(temperature=0, max_tokens=8192),
        )

    def audit(
        self, context: EvidenceContext, plan: CompetitionPaperPlan
    ) -> MathAuditReport:
        prompt = (
            "Frozen problem and evidence catalog:\n\n"
            + json.dumps(context.problem.model_dump(mode="json"), indent=2)
            + "\n\nSystem-provided model facts:\n\n"
            + json.dumps(list(context.model_facts), indent=2)
            + "\n\nPlanned model materials:\n\n"
            + json.dumps(_model_section_payload(plan), indent=2)
            + "\n\nEvidence ids and values:\n\n"
            + json.dumps(
                [entry.prompt_dict() for entry in context.entries],
                indent=2,
                sort_keys=True,
            )
        )
        report = self._agent.run_sync(prompt).output
        issues = [
            issue.model_copy(update={"issue_id": f"ma-{index}"})
            for index, issue in enumerate(report.issues, start=1)
        ]
        return report.model_copy(update={"issues": issues})


class ExperimentAnalyst:
    def __init__(self, model: Model | str) -> None:
        self._agent: Agent[None, ExperimentAnalysisReport] = Agent(
            model,
            output_type=ExperimentAnalysisReport,
            system_prompt=EXPERIMENT_ANALYST_PROMPT,
            retries=2,
            model_settings=ModelSettings(temperature=0, max_tokens=8192),
        )

    def analyze(
        self, context: EvidenceContext, plan: CompetitionPaperPlan
    ) -> ExperimentAnalysisReport:
        prompt = (
            "Frozen problem:\n\n"
            + json.dumps(context.problem.model_dump(mode="json"), indent=2)
            + "\n\nPlanned tables and narrative section purposes:\n\n"
            + json.dumps(
                {
                    "tables": [table.model_dump(mode="json") for table in plan.tables],
                    "sections": [
                        {
                            "kind": section.kind.value,
                            "purpose": section.purpose,
                            "evidence_ids": section.evidence_ids,
                        }
                        for section in plan.sections
                    ],
                },
                indent=2,
            )
            + "\n\nVerified claims:\n\n"
            + json.dumps(
                [
                    {
                        "claim_id": claim.claim_id,
                        "statement": claim.statement,
                        "evidence_ids": claim.evidence_ids,
                        "status": claim.status,
                    }
                    for claim in context.claims
                ],
                indent=2,
            )
            + "\n\nEvidence catalog:\n\n"
            + json.dumps(
                [entry.prompt_dict() for entry in context.entries],
                indent=2,
                sort_keys=True,
            )
        )
        report = self._agent.run_sync(prompt).output
        findings = [
            finding.model_copy(
                update={
                    "finding_id": f"ef-{index}",
                    "evidence_ids": clean_evidence_ids(finding.evidence_ids),
                }
            )
            for index, finding in enumerate(report.findings, start=1)
        ]
        checks = [
            need.model_copy(
                update={
                    "supporting_evidence_ids": clean_evidence_ids(
                        need.supporting_evidence_ids
                    )
                }
            )
            for need in report.requested_checks
        ]
        return report.model_copy(
            update={"findings": findings, "requested_checks": checks}
        )


class FigureDesigner:
    def __init__(self, model: Model | str) -> None:
        self._agent: Agent[None, FigurePlanBatch] = Agent(
            model,
            output_type=FigurePlanBatch,
            system_prompt=FIGURE_PROMPT,
            retries=2,
            model_settings=ModelSettings(temperature=0, max_tokens=8192),
        )

    def design(
        self,
        context: EvidenceContext,
        plan: CompetitionPaperPlan,
        analysis: ExperimentAnalysisReport,
    ) -> list[FigurePlan]:
        numeric_keys: dict[str, int] = {}
        for entry in context.entries:
            if not entry.numeric:
                continue
            prefix = entry.key.rsplit(".", 1)[0] if "." in entry.key else entry.key
            numeric_keys[prefix] = numeric_keys.get(prefix, 0) + 1
        prompt = (
            "Available figure templates and counts:\n\n"
            + json.dumps(context.available_figures, indent=2)
            + "\n\nAvailable numeric metric prefixes:\n\n"
            + json.dumps(dict(sorted(numeric_keys.items())[:40]), indent=2)
            + "\n\nSection headings:\n\n"
            + json.dumps([section.heading for section in plan.sections], indent=2)
            + "\n\nExperiment findings (compact):\n\n"
            + json.dumps(
                [
                    {
                        "finding_id": finding.finding_id,
                        "kind": finding.kind.value,
                        "statement": finding.statement,
                    }
                    for finding in analysis.findings
                ],
                indent=2,
            )
            + "\n\nEvidence ids available for citation:\n\n"
            + json.dumps(
                [entry.evidence_id for entry in context.entries[:80]],
                indent=2,
            )
        )
        batch = self._agent.run_sync(prompt).output
        figures: list[FigurePlan] = []
        for index, figure in enumerate(batch.figures, start=1):
            if context.available_figures.get(figure.template.value, 0) <= 0:
                raise CompetitionAgentError(
                    f"figure template {figure.template.value!r} is unavailable"
                )
            if figure.template.value == "metric_bar":
                prefix = figure.metric_prefix or ""
                if not any(
                    entry.numeric and entry.key.startswith(prefix)
                    for entry in context.entries
                ):
                    raise CompetitionAgentError(
                        f"metric prefix {prefix!r} has no numeric evidence"
                    )
            figures.append(
                figure.model_copy(
                    update={
                        "figure_id": f"fig-c{index}",
                        "evidence_ids": clean_evidence_ids(figure.evidence_ids),
                    }
                )
            )
        return figures


__all__ = [
    "ArchitectProtocol",
    "CompetitionAgentError",
    "CompetitionAgents",
    "ExperimentAnalyst",
    "ExperimentAnalystProtocol",
    "FigureDesigner",
    "FigureDesignerProtocol",
    "MathAuditor",
    "MathAuditorProtocol",
    "PaperArchitect",
]
