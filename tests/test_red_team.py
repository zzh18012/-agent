from pathlib import Path

from typer.testing import CliRunner

from mmos.agents import normalize_attacks, normalize_judge, normalize_responses
from mmos.cli.main import app
from mmos.core.project import create_project
from mmos.schemas.review import (
    Attack,
    AttackCategory,
    AttackReport,
    AttackSeverity,
    AttackVerdict,
    AuthorAction,
    AuthorResponse,
    AuthorResponseReport,
    JudgeReport,
    Verdict,
)

runner = CliRunner()


def make_attack(attack_id: str, severity: AttackSeverity) -> Attack:
    return Attack(
        attack_id=attack_id,
        category=AttackCategory.OVERREACH,
        severity=severity,
        target="claim-1",
        argument="argument",
        required_fix="fix",
    )


def make_verdict(attack_id: str, verdict: Verdict) -> AttackVerdict:
    return AttackVerdict(attack_id=attack_id, verdict=verdict, rationale="r")


def test_attack_ids_are_system_owned() -> None:
    report = AttackReport(
        attacks=[
            make_attack("llm-id", AttackSeverity.MAJOR),
            make_attack("another", AttackSeverity.MINOR),
        ]
    )
    normalized = normalize_attacks(report)
    assert [attack.attack_id for attack in normalized.attacks] == ["a1", "a2"]


def test_missing_and_unknown_responses_are_normalized() -> None:
    attacks = [make_attack("a1", AttackSeverity.MAJOR), make_attack("a2", AttackSeverity.MINOR)]
    report = AuthorResponseReport(
        responses=[
            AuthorResponse(attack_id="ghost", action=AuthorAction.ACCEPT, response="x"),
            AuthorResponse(attack_id="a1", action=AuthorAction.ACCEPT, response="ok"),
        ]
    )
    normalized = normalize_responses(report, attacks)
    assert [response.attack_id for response in normalized.responses] == ["a1", "a2"]
    assert normalized.responses[1].action is AuthorAction.REVISE_PAPER


def test_critical_revise_blocks() -> None:
    attacks = [make_attack("a1", AttackSeverity.CRITICAL), make_attack("a2", AttackSeverity.MAJOR)]
    report = JudgeReport(
        verdicts=[
            make_verdict("a1", Verdict.REVISE),
            make_verdict("a2", Verdict.PASS),
        ]
    )
    normalized = normalize_judge(report, attacks)
    assert normalized.blocking is True
    assert normalized.overall is Verdict.REVISE


def test_critical_reject_blocks_with_reject() -> None:
    attacks = [make_attack("a1", AttackSeverity.CRITICAL)]
    normalized = normalize_judge(
        JudgeReport(verdicts=[make_verdict("a1", Verdict.REJECT)]), attacks
    )
    assert normalized.blocking is True
    assert normalized.overall is Verdict.REJECT


def test_missing_verdict_on_critical_attack_is_conservative() -> None:
    attacks = [make_attack("a1", AttackSeverity.CRITICAL)]
    normalized = normalize_judge(JudgeReport(verdicts=[]), attacks)
    assert normalized.blocking is True
    assert normalized.verdicts[0].verdict is Verdict.REVISE


def test_all_pass_is_not_blocking() -> None:
    attacks = [make_attack("a1", AttackSeverity.MAJOR)]
    normalized = normalize_judge(JudgeReport(verdicts=[make_verdict("a1", Verdict.PASS)]), attacks)
    assert normalized.blocking is False
    assert normalized.overall is Verdict.PASS


def test_redteam_cli_requires_evidence(tmp_path: Path) -> None:
    paths, _ = create_project(tmp_path / "projects", "empty")
    result = runner.invoke(
        app,
        ["redteam", paths.root.name, "exp-missing", "--projects-dir", str(paths.root.parent)],
    )
    assert result.exit_code == 1
    assert "no verified evidence" in result.output
