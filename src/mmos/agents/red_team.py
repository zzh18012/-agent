from datetime import UTC, datetime

from pydantic_ai import Agent
from pydantic_ai.models import Model

from mmos.schemas.paper import PaperAST
from mmos.schemas.review import (
    Attack,
    AttackReport,
    AttackSeverity,
    AttackVerdict,
    AuthorAction,
    AuthorResponse,
    AuthorResponseReport,
    JudgeReport,
    RedTeamReview,
    Verdict,
)

ATTACK_PROMPT = """\
You are a red-team reviewer. Your job is to argue that the paper must not be trusted.

Attack what the paper actually says: data leakage, metric errors, unsupported
assumptions, overfitting, unfair baselines, unit errors, parameter identifiability,
causal overreach, numbers inconsistent with the verified evidence, misleading figures,
conclusions beyond evidence, irreproducibility, and missing sensitivity or robustness.

Rules:
- Use exact claim ids or section headings as targets.
- Quote only values present in the paper; never invent numbers.
- severity: critical means it invalidates a main conclusion; major weakens it; minor is
  a clarity or presentation issue.
- required_fix must be actionable.
- If the paper survives a line of attack, do not fabricate one; report fewer attacks.
"""

RESPONSE_PROMPT = """\
You are the author responding to a red-team review. For every attack id:
- choose an action: accept, rebut, supplement_experiment, revise_paper, or limit_claim;
- give a concrete response grounded only in existing evidence.
Do not claim that a new experiment was run unless the review materials say so.
"""

JUDGE_PROMPT = """\
You are the review judge. For every attack id, decide pass, revise, or reject:
- pass: the attack is answered or not supported;
- revise: the paper/claims must change before release;
- reject: the work must not be released.
An unresolved critical attack must not pass. Judge only from the attack and response.
"""


def normalize_attacks(report: AttackReport) -> AttackReport:
    normalized = [
        attack.model_copy(update={"attack_id": f"a{index}"})
        for index, attack in enumerate(report.attacks, start=1)
    ]
    return AttackReport(attacks=normalized, summary=report.summary)


def normalize_responses(
    report: AuthorResponseReport, attacks: list[Attack]
) -> AuthorResponseReport:
    by_attack = {response.attack_id: response for response in report.responses}
    responses: list[AuthorResponse] = []
    for attack in attacks:
        response = by_attack.get(attack.attack_id)
        if response is None:
            responses.append(
                AuthorResponse(
                    attack_id=attack.attack_id,
                    action=AuthorAction.REVISE_PAPER,
                    response="No response was produced; treat the attack as unresolved.",
                )
            )
        else:
            responses.append(response)
    return AuthorResponseReport(responses=responses)


def normalize_judge(report: JudgeReport, attacks: list[Attack]) -> JudgeReport:
    known = {attack.attack_id: attack for attack in attacks}
    verdicts: list[AttackVerdict] = []
    seen: set[str] = set()
    for verdict in report.verdicts:
        if verdict.attack_id not in known or verdict.attack_id in seen:
            continue
        seen.add(verdict.attack_id)
        verdicts.append(verdict)
    for attack in attacks:
        if attack.attack_id not in seen:
            verdicts.append(
                AttackVerdict(
                    attack_id=attack.attack_id,
                    verdict=Verdict.REVISE,
                    rationale="No verdict was produced; defaulting to revise.",
                )
            )

    severity = {attack.attack_id: attack.severity for attack in attacks}
    critical_issues = [
        verdict
        for verdict in verdicts
        if severity[verdict.attack_id] is AttackSeverity.CRITICAL
        and verdict.verdict is not Verdict.PASS
    ]
    blocking = bool(critical_issues)
    if any(verdict.verdict is Verdict.REJECT for verdict in critical_issues):
        overall = Verdict.REJECT
    elif blocking or any(verdict.verdict is Verdict.REVISE for verdict in verdicts):
        overall = Verdict.REVISE
    else:
        overall = Verdict.PASS

    return JudgeReport(
        verdicts=verdicts,
        overall=overall,
        blocking=blocking,
        summary=f"{report.summary} System aggregation: overall={overall.value}, "
        f"blocking={blocking}.",
    )


class RedTeam:
    def __init__(self, model: Model | str) -> None:
        self._attacker: Agent[None, AttackReport] = Agent(
            model, output_type=AttackReport, system_prompt=ATTACK_PROMPT, retries=2
        )
        self._author: Agent[None, AuthorResponseReport] = Agent(
            model, output_type=AuthorResponseReport, system_prompt=RESPONSE_PROMPT, retries=2
        )
        self._judge: Agent[None, JudgeReport] = Agent(
            model, output_type=JudgeReport, system_prompt=JUDGE_PROMPT, retries=2
        )

    def run(self, paper: PaperAST, *, experiment_id: str) -> RedTeamReview:
        paper_json = paper.model_dump_json(indent=2)
        attacks = normalize_attacks(
            self._attacker.run_sync(f"Paper under review:\n{paper_json}\n\nFind attacks.").output
        )
        responses = normalize_responses(
            self._author.run_sync(
                f"Paper under review:\n{paper_json}\n\n"
                f"Attacks:\n{attacks.model_dump_json(indent=2)}\n\nRespond to every attack id."
            ).output,
            attacks.attacks,
        )
        judge = normalize_judge(
            self._judge.run_sync(
                f"Attacks:\n{attacks.model_dump_json(indent=2)}\n\n"
                f"Author responses:\n{responses.model_dump_json(indent=2)}\n\n"
                "Judge every attack id."
            ).output,
            attacks.attacks,
        )
        return RedTeamReview(
            experiment_id=experiment_id,
            attacks=attacks,
            responses=responses,
            judge=judge,
            created_at=datetime.now(UTC).replace(microsecond=0),
        )
