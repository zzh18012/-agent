from datetime import datetime
from enum import StrEnum

from pydantic import Field

from mmos.schemas.common import MMOSModel


class AttackSeverity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class AttackCategory(StrEnum):
    DATA_LEAKAGE = "data_leakage"
    METRIC_ERROR = "metric_error"
    ASSUMPTION = "assumption"
    OVERFITTING = "overfitting"
    UNFAIR_BASELINE = "unfair_baseline"
    UNIT_ERROR = "unit_error"
    CAUSAL_OVERREACH = "causal_overreach"
    INCONSISTENCY = "inconsistency"
    MISLEADING_FIGURE = "misleading_figure"
    OVERREACH = "overreach"
    REPRODUCIBILITY = "reproducibility"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    OTHER = "other"


class Attack(MMOSModel):
    attack_id: str
    category: AttackCategory
    severity: AttackSeverity
    target: str
    argument: str
    required_fix: str


class AttackReport(MMOSModel):
    attacks: list[Attack] = Field(default_factory=list)
    summary: str = ""

    @property
    def blocking(self) -> bool:
        return any(attack.severity is AttackSeverity.CRITICAL for attack in self.attacks)


class AuthorAction(StrEnum):
    ACCEPT = "accept"
    REBUT = "rebut"
    SUPPLEMENT_EXPERIMENT = "supplement_experiment"
    REVISE_PAPER = "revise_paper"
    LIMIT_CLAIM = "limit_claim"


class AuthorResponse(MMOSModel):
    attack_id: str
    action: AuthorAction
    response: str


class AuthorResponseReport(MMOSModel):
    responses: list[AuthorResponse] = Field(default_factory=list)


class Verdict(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


class AttackVerdict(MMOSModel):
    attack_id: str
    verdict: Verdict
    rationale: str


class JudgeReport(MMOSModel):
    verdicts: list[AttackVerdict] = Field(default_factory=list)
    overall: Verdict = Verdict.PASS
    blocking: bool = False
    summary: str = ""


class RedTeamReview(MMOSModel):
    experiment_id: str
    attacks: AttackReport
    responses: AuthorResponseReport
    judge: JudgeReport
    created_at: datetime
