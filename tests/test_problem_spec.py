from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mmos.schemas.problem import ProblemSpec, SubProblem, Variable

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def make_subproblem(subproblem_id: str, depends_on: list[str] | None = None) -> SubProblem:
    return SubProblem(
        subproblem_id=subproblem_id,
        statement="statement",
        goal="goal",
        depends_on=depends_on or [],
    )


def test_problem_spec_roundtrip() -> None:
    spec = ProblemSpec(
        problem_id="p-1",
        title="demo",
        overall_goal="goal",
        subproblems=[make_subproblem("q1"), make_subproblem("q2", ["q1"])],
    )
    restored = ProblemSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_duplicate_subproblem_ids_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProblemSpec(
            problem_id="p-1",
            title="demo",
            overall_goal="goal",
            subproblems=[make_subproblem("q1"), make_subproblem("q1")],
        )


def test_unknown_dependency_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProblemSpec(
            problem_id="p-1",
            title="demo",
            overall_goal="goal",
            subproblems=[make_subproblem("q1", ["missing"])],
        )


def test_dependency_cycle_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProblemSpec(
            problem_id="p-1",
            title="demo",
            overall_goal="goal",
            subproblems=[make_subproblem("q1", ["q2"]), make_subproblem("q2", ["q1"])],
        )


def test_variable_defaults() -> None:
    variable = Variable(name="x")
    assert variable.meaning == ""
    assert variable.unit is None
