from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mmos.core.project import ProjectPaths, create_project
from mmos.evidence.discrete import (
    validate_and_register_finite_policy,
    validate_and_register_linear_milp,
    verify_policy_provenance,
)
from mmos.evidence.inference import validate_and_register_inference
from mmos.evidence.store import EvidenceError
from mmos.experiments.discrete_runner import (
    _status_from_solver,
    run_finite_policy,
    run_linear_milp,
)
from mmos.experiments.inference_runner import run_binomial_inference
from mmos.experiments.policy_evaluators import (
    PolicyEvaluatorError,
    SyntheticProcessPolicy,
)
from mmos.historical_benchmark.capabilities import (
    assess_capability_fit,
    descriptor_hash,
    optimization_capability_descriptors,
)
from mmos.historical_benchmark.disposition import compute_disposition
from mmos.historical_benchmark.schemas import CapabilityVerdict, ExecutionDisposition
from mmos.schemas.discrete import (
    ConstraintSense,
    FinitePolicyExecutionStatus,
    FinitePolicySpec,
    FrozenParameterProvenance,
    LinearConstraintSpec,
    LinearExpression,
    LinearMILPSpec,
    LinearObjectiveSpec,
    LinearTerm,
    MILPSense,
    MILPSolverStatus,
    MILPVariableSpec,
    MILPVariableType,
    linear_milp_instance_hash,
)
from mmos.schemas.inference import Alternative, BinomialInferenceSpec
from mmos.schemas.problem import ProblemSpec, SubProblem
from mmos.schemas.routing import (
    CapabilityLevel,
    CapabilityReport,
    ExecutionStatus,
    ProblemSignalExtraction,
    ProblemSignals,
    RoutingDecision,
    RoutingStatus,
    SubproblemRoute,
    SubproblemSignals,
)
from mmos.schemas.states import TaskFamily
from mmos.validation.discrete import (
    exhaustive_milp_optimum,
    validate_finite_policy,
    validate_linear_milp,
)

NOW = datetime(2026, 11, 18, 10, tzinfo=UTC)


def project(tmp_path: Path, name: str = "c2") -> ProjectPaths:
    paths, _ = create_project(tmp_path / "projects", name, task=TaskFamily.OPTIMIZATION)
    return paths


def variable(
    variable_id: str, kind: MILPVariableType, lower: float, upper: float
) -> MILPVariableSpec:
    return MILPVariableSpec(
        variable_id=variable_id, type=kind, lower_bound=lower, upper_bound=upper
    )


def term(variable_id: str, coefficient: float) -> LinearTerm:
    return LinearTerm(variable_id=variable_id, coefficient=coefficient)


def general_milp_spec() -> LinearMILPSpec:
    return LinearMILPSpec(
        model_name="general-discrete-fixture",
        variables=[
            variable("x1", MILPVariableType.BINARY, 0, 1),
            variable("x2", MILPVariableType.BINARY, 0, 1),
            variable("y", MILPVariableType.INTEGER, 0, 4),
            variable("z", MILPVariableType.CONTINUOUS, 0, 10),
        ],
        objective=LinearObjectiveSpec(
            sense=MILPSense.MINIMIZE,
            expression=LinearExpression(
                constant=3.0,
                terms=[term("x1", 5.0), term("x2", 4.0), term("y", 2.0), term("z", 1.5)],
            ),
        ),
        constraints=[
            LinearConstraintSpec(
                constraint_id="capacity",
                expression=LinearExpression(
                    terms=[term("x1", 3.0), term("x2", 2.0), term("y", 1.0)]
                ),
                sense=ConstraintSense.LE,
                rhs=6.0,
            ),
            LinearConstraintSpec(
                constraint_id="demand",
                expression=LinearExpression(terms=[term("y", 1.0)]),
                sense=ConstraintSense.GE,
                rhs=2.0,
            ),
            LinearConstraintSpec(
                constraint_id="link",
                expression=LinearExpression(terms=[term("z", 1.0), term("y", -0.5)]),
                sense=ConstraintSense.EQ,
                rhs=0.0,
            ),
            LinearConstraintSpec(
                constraint_id="ship",
                expression=LinearExpression(terms=[term("z", 1.0), term("x2", 1.0)]),
                sense=ConstraintSense.LE,
                rhs=4.5,
            ),
            LinearConstraintSpec(
                constraint_id="choose_one",
                expression=LinearExpression(terms=[term("x1", 1.0), term("x2", 1.0)]),
                sense=ConstraintSense.GE,
                rhs=1.0,
            ),
        ],
    )


def test_general_milp_solve_and_validate(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = general_milp_spec()
    result = run_linear_milp(paths, spec, experiment_id="exp-c2-milp")
    assert result.solver_status is MILPSolverStatus.OPTIMAL
    assert result.objective_value == pytest.approx(12.5)
    assert result.variable_values["x1"] == pytest.approx(0.0)
    assert result.variable_values["x2"] == pytest.approx(1.0)
    assert result.variable_values["y"] == pytest.approx(2.0)
    assert result.variable_values["z"] == pytest.approx(1.0)
    assert result.gap == 0.0
    report = validate_linear_milp(paths, "exp-c2-milp")
    assert not report.blocking
    oracle = exhaustive_milp_optimum(spec)
    assert oracle.status == "complete"
    assert oracle.used_continuous_lp
    assert oracle.optimum == pytest.approx(12.5)


def test_maximize_and_equality_sense(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = LinearMILPSpec(
        model_name="maximize-fixture",
        variables=[
            variable("a", MILPVariableType.BINARY, 0, 1),
            variable("b", MILPVariableType.BINARY, 0, 1),
        ],
        objective=LinearObjectiveSpec(
            sense=MILPSense.MAXIMIZE,
            expression=LinearExpression(terms=[term("a", 3.0), term("b", 2.0)]),
        ),
        constraints=[
            LinearConstraintSpec(
                constraint_id="one",
                expression=LinearExpression(terms=[term("a", 1.0), term("b", 1.0)]),
                sense=ConstraintSense.LE,
                rhs=1.0,
            )
        ],
    )
    result = run_linear_milp(paths, spec, experiment_id="exp-c2-max")
    assert result.solver_status is MILPSolverStatus.OPTIMAL
    assert result.objective_value == pytest.approx(3.0)
    assert not validate_linear_milp(paths, "exp-c2-max").blocking


def test_schema_rejects_bad_models() -> None:
    variables = [
        variable("x", MILPVariableType.BINARY, 0, 1),
        variable("y", MILPVariableType.CONTINUOUS, 0, 2),
    ]
    with pytest.raises(ValueError, match="unknown variables"):
        LinearMILPSpec(
            model_name="unknown",
            variables=variables,
            objective=LinearObjectiveSpec(
                sense=MILPSense.MINIMIZE,
                expression=LinearExpression(terms=[term("ghost", 1.0)]),
            ),
        )
    with pytest.raises(ValueError, match="repeats a variable_id"):
        LinearExpression(terms=[term("x", 1.0), term("x", 2.0)])
    with pytest.raises(ValueError, match="unique"):
        LinearMILPSpec(
            model_name="duplicate",
            variables=[variable("x", MILPVariableType.BINARY, 0, 1)] * 2,
            objective=LinearObjectiveSpec(
                sense=MILPSense.MINIMIZE,
                expression=LinearExpression(terms=[term("x", 1.0)]),
            ),
        )
    with pytest.raises(ValueError):
        LinearTerm(variable_id="x", coefficient=float("nan"))
    with pytest.raises(ValueError):
        LinearTerm(variable_id="x", coefficient=float("inf"))
    with pytest.raises(ValueError, match="at least one binary or integer"):
        LinearMILPSpec(
            model_name="pure-continuous",
            variables=[variable("x", MILPVariableType.CONTINUOUS, 0, 2)],
            objective=LinearObjectiveSpec(
                sense=MILPSense.MINIMIZE,
                expression=LinearExpression(terms=[term("x", 1.0)]),
            ),
        )
    with pytest.raises(ValueError, match="non-0/1 bound"):
        variable("bad", MILPVariableType.BINARY, 0.2, 1.0)
    with pytest.raises(ValueError, match="unique"):
        LinearMILPSpec(
            model_name="duplicate-constraints",
            variables=[variable("x", MILPVariableType.BINARY, 0, 1)],
            objective=LinearObjectiveSpec(
                sense=MILPSense.MINIMIZE,
                expression=LinearExpression(terms=[term("x", 1.0)]),
            ),
            constraints=[
                LinearConstraintSpec(
                    constraint_id="c",
                    expression=LinearExpression(terms=[term("x", 1.0)]),
                    sense=ConstraintSense.LE,
                    rhs=1.0,
                )
            ]
            * 2,
        )


def test_objective_and_activity_tampering_is_blocking(tmp_path: Path) -> None:
    paths = project(tmp_path)
    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-tamper")
    artifact = paths.experiment_dir("exp-tamper") / "artifacts" / "linear_milp_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["objective_value"] = 0.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_linear_milp(paths, "exp-tamper").blocking

    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-tamper2")
    artifact = paths.experiment_dir("exp-tamper2") / "artifacts" / "linear_milp_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["constraint_activities"][0]["slack"] = 999.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_linear_milp(paths, "exp-tamper2").blocking


def test_constraint_mutation_changes_instance_hash() -> None:
    spec = general_milp_spec()
    mutated = spec.model_copy(deep=True)
    mutated.constraints[0].expression.terms[0].coefficient = 3.5
    assert linear_milp_instance_hash(spec) != linear_milp_instance_hash(mutated)


def test_spec_mutation_makes_existing_artifact_stale(tmp_path: Path) -> None:
    paths = project(tmp_path)
    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-spec-stale")
    spec_path = paths.experiment_dir("exp-spec-stale") / "linear_milp_spec.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["constraints"][0]["expression"]["terms"][0]["coefficient"] = 3.5
    spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert validate_linear_milp(paths, "exp-spec-stale").blocking


def test_integrality_and_bound_tampering_is_blocking(tmp_path: Path) -> None:
    paths = project(tmp_path)
    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-int-tamper")
    artifact = paths.experiment_dir("exp-int-tamper") / "artifacts" / "linear_milp_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["variable_values"]["x1"] = 0.5
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_linear_milp(paths, "exp-int-tamper").blocking

    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-bound-tamper")
    artifact = paths.experiment_dir("exp-bound-tamper") / "artifacts" / "linear_milp_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["variable_values"]["z"] = 999.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_linear_milp(paths, "exp-bound-tamper").blocking


def test_solver_status_mapping_does_not_upgrade_feasible() -> None:
    from ortools.linear_solver import pywraplp

    assert _status_from_solver(pywraplp.Solver.FEASIBLE) is MILPSolverStatus.FEASIBLE_NOT_PROVEN
    assert _status_from_solver(pywraplp.Solver.OPTIMAL) is MILPSolverStatus.OPTIMAL
    assert _status_from_solver(pywraplp.Solver.INFEASIBLE) is MILPSolverStatus.INFEASIBLE


def test_optimal_claim_with_nonzero_gap_is_blocking(tmp_path: Path) -> None:
    paths = project(tmp_path)
    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-gap")
    artifact = paths.experiment_dir("exp-gap") / "artifacts" / "linear_milp_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["gap"] = 0.25
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_linear_milp(paths, "exp-gap").blocking


def test_infeasible_model_has_no_fake_solution(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = LinearMILPSpec(
        model_name="infeasible-fixture",
        variables=[
            variable("x", MILPVariableType.BINARY, 0, 1),
            variable("y", MILPVariableType.INTEGER, 0, 2),
        ],
        objective=LinearObjectiveSpec(
            sense=MILPSense.MINIMIZE,
            expression=LinearExpression(terms=[term("x", 1.0), term("y", 1.0)]),
        ),
        constraints=[
            LinearConstraintSpec(
                constraint_id="impossible",
                expression=LinearExpression(terms=[term("x", 1.0), term("y", 1.0)]),
                sense=ConstraintSense.GE,
                rhs=5.0,
            )
        ],
    )
    result = run_linear_milp(paths, spec, experiment_id="exp-c2-infeasible")
    assert result.solver_status is MILPSolverStatus.INFEASIBLE
    assert result.variable_values == {}
    assert result.objective_value is None
    assert not validate_linear_milp(paths, "exp-c2-infeasible").blocking
    records = validate_and_register_linear_milp(paths, "exp-c2-infeasible")
    keys = {record.key for record in records}
    assert "linear_milp.solver.status" in keys
    assert not any("objective" in key or "variable." in key for key in keys)


def test_milp_evidence_registration(tmp_path: Path) -> None:
    paths = project(tmp_path)
    run_linear_milp(paths, general_milp_spec(), experiment_id="exp-evidence-milp")
    records = validate_and_register_linear_milp(paths, "exp-evidence-milp")
    keys = {record.key for record in records}
    assert "linear_milp.objective" in keys
    assert "linear_milp.variable.x2" in keys
    assert "linear_milp.solver.status" in keys


def policy_parameters(defect_rate: float = 0.18) -> dict[str, float]:
    return {
        "component_cost": 20.0,
        "assembly_cost": 10.0,
        "component_inspection_cost": 4.0,
        "final_inspection_cost": 6.0,
        "premium_route_extra": 5.0,
        "rework_cost": 8.0,
        "expedite_cost": 3.0,
        "defect_penalty": 100.0,
        "defect_rate": defect_rate,
        "residual_after_component_inspection": 0.1,
        "residual_after_final_inspection": 0.1,
        "rework_residual_factor": 0.5,
    }


def policy_spec(**overrides) -> FinitePolicySpec:
    values = {
        "model_name": "synthetic-process-fixture",
        "evaluator_id": "registered.synthetic_process_policy",
        "parameters": policy_parameters(),
        "max_enumerated_states": 100,
    }
    values.update(overrides)
    return FinitePolicySpec(**values)


def test_finite_policy_complete_enumeration(tmp_path: Path) -> None:
    paths = project(tmp_path)
    result = run_finite_policy(paths, policy_spec(), experiment_id="exp-c2-policy")
    assert result.execution_status is FinitePolicyExecutionStatus.COMPLETED
    assert result.state_space_size == 32
    assert result.evaluated_state_count == 32
    assert result.feasible_policy_count == 18
    assert result.optimal_policy_count == 1
    assert result.selected_policy == {
        "inspect_component": True,
        "premium_route": False,
        "inspect_final": False,
        "rework": False,
        "expedite": False,
    }
    assert result.objective_value == pytest.approx(35.8)
    assert result.objective_components is not None
    assert sum(result.objective_components.values()) == pytest.approx(35.8)
    assert not validate_finite_policy(paths, "exp-c2-policy").blocking
    records = validate_and_register_finite_policy(paths, "exp-c2-policy")
    assert any(record.key == "finite_policy.objective_value" for record in records)
    assert any(record.key == "finite_policy.optimal_policy_count" for record in records)


def test_policy_state_space_limit_fails_closed(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = policy_spec(max_enumerated_states=16)
    result = run_finite_policy(paths, spec, experiment_id="exp-policy-limit")
    assert result.execution_status is FinitePolicyExecutionStatus.NOT_EXECUTED_STATE_SPACE_LIMIT
    assert result.selected_policy is None
    assert result.objective_value is None
    assert result.evaluated_state_count == 0
    assert not validate_finite_policy(paths, "exp-policy-limit").blocking
    records = validate_and_register_finite_policy(paths, "exp-policy-limit")
    assert not any(
        record.key in {"finite_policy.objective_value"} or record.key.startswith(
            "finite_policy.selected_policy"
        )
        for record in records
    )


def test_policy_tie_handling_is_canonical(tmp_path: Path) -> None:
    paths = project(tmp_path)
    zero = policy_parameters(defect_rate=0.0)
    for name in (
        "component_inspection_cost",
        "final_inspection_cost",
        "premium_route_extra",
        "rework_cost",
        "expedite_cost",
        "defect_penalty",
    ):
        zero[name] = 0.0
    spec = policy_spec(parameters=zero)
    result = run_finite_policy(paths, spec, experiment_id="exp-policy-ties")
    assert result.feasible_policy_count == 18
    assert result.optimal_policy_count == 18
    assert result.selected_policy == {
        "inspect_component": False,
        "premium_route": False,
        "inspect_final": False,
        "rework": False,
        "expedite": False,
    }
    assert result.tie_break_rule == "canonical_lexicographic"
    assert not validate_finite_policy(paths, "exp-policy-ties").blocking


def test_policy_parameter_mismatch_rejected(tmp_path: Path) -> None:
    paths = project(tmp_path)
    params = policy_parameters()
    params["unexpected"] = 1.0
    with pytest.raises(PolicyEvaluatorError, match="parameter names"):
        run_finite_policy(paths, policy_spec(parameters=params), experiment_id="exp-bad-params")


def test_policy_evaluator_mutation_makes_artifact_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = project(tmp_path)
    run_finite_policy(paths, policy_spec(), experiment_id="exp-policy-stale")
    import mmos.experiments.policy_evaluators as module

    class SyntheticV2(SyntheticProcessPolicy):
        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "version", "2")

    monkeypatch.setitem(
        module._EVALUATORS, "registered.synthetic_process_policy", SyntheticV2()
    )
    report = validate_finite_policy(paths, "exp-policy-stale")
    assert report.blocking


def test_policy_artifact_tampering_is_blocking(tmp_path: Path) -> None:
    paths = project(tmp_path)
    run_finite_policy(paths, policy_spec(), experiment_id="exp-policy-tamper")
    artifact = paths.experiment_dir("exp-policy-tamper") / "artifacts" / "finite_policy_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["optimal_policy_count"] = 1
    payload["selected_policy"]["inspect_component"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_finite_policy(paths, "exp-policy-tamper").blocking


def test_point_estimate_provenance_and_staleness(tmp_path: Path) -> None:
    paths = project(tmp_path)
    inference = run_binomial_inference(
        paths,
        BinomialInferenceSpec(
            successes=18,
            trials=100,
            null_probability=0.10,
            alternative=Alternative.GREATER,
            confidence_level=0.95,
        ),
        experiment_id="exp-c2-binom",
    )
    validate_and_register_inference(paths, "exp-c2-binom", "binomial")
    index = json.loads(paths.verified_numbers_file.read_text(encoding="utf-8"))
    record = index["records"]["ev-exp-c2-binom-proportion"]
    assert record["raw_value"] == pytest.approx(inference.observed_proportion)
    provenance = FrozenParameterProvenance(
        name="defect_rate",
        value=inference.observed_proportion,
        source_evidence_id="ev-exp-c2-binom-proportion",
        source_artifact_hash=record["artifact_hash"],
        source_experiment_id="exp-c2-binom",
    )
    spec = policy_spec(
        parameters=policy_parameters(defect_rate=inference.observed_proportion),
        parameter_provenance=[provenance],
        point_estimate_conditioned=True,
    )
    result = run_finite_policy(paths, spec, experiment_id="exp-c2-inference-to-policy")
    assert result.point_estimate_conditioned
    assert result.uncertainty_disclosure == (
        "decision conditional on point estimates; uncertainty not propagated"
    )
    records = validate_and_register_finite_policy(
        paths, "exp-c2-inference-to-policy"
    )
    disclosure = [
        item for item in records if item.key == "finite_policy.uncertainty_disclosure"
    ]
    assert disclosure and disclosure[0].provenance["source_evidence"]

    payload = json.loads(paths.verified_numbers_file.read_text(encoding="utf-8"))
    payload["records"]["ev-exp-c2-binom-proportion"]["raw_value"] = 0.19
    paths.verified_numbers_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="stale"):
        verify_policy_provenance(paths, spec)


def _decision(family: TaskFamily, requirement_id: str) -> RoutingDecision:
    return RoutingDecision(
        routing_id="route-c2",
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        status=RoutingStatus.SINGLE_TASK,
        primary_task=family,
        required_task_families=[family],
        subproblem_routes=[
            SubproblemRoute(subproblem_id=requirement_id, task_families=[family])
        ],
        capabilities=[
            CapabilityReport(
                task_family=family,
                current_supported_scope="test",
                execution_readiness=CapabilityLevel.PARTIAL,
                limitations=[],
            )
        ],
        execution_status=ExecutionStatus.READY_FOR_SINGLE_TASK,
        provider="test",
        model="test",
        created_at=NOW,
    )


def _requirement(requirement_id: str, text: str):
    from mmos.historical_benchmark.schemas import ProblemRequirement

    return ProblemRequirement(
        requirement_id=requirement_id,
        source_location=requirement_id,
        question_text=text,
        question_hash=hashlib.sha256(text.encode()).hexdigest(),
        required_output="output",
    )


def test_optimization_capability_slices() -> None:
    descriptors = optimization_capability_descriptors()
    assert {item.capability_id for item in descriptors} == {
        "optimization.linear_milp",
        "optimization.finite_discrete_policy",
    }
    assert all(descriptor_hash(item) for item in descriptors)
    decision = _decision(TaskFamily.OPTIMIZATION, "q1")
    linear = assess_capability_fit(
        decision,
        [
            _requirement(
                "q1", "solve a mixed-integer linear assignment problem with binary decisions"
            )
        ],
    )
    assert linear.verdict is CapabilityVerdict.SUPPORTED
    assert linear.available_capabilities == ["optimization.linear_milp"]
    policy = assess_capability_fit(
        decision,
        [_requirement("q1", "choose an inspection and rework policy for each production stage")],
    )
    assert policy.verdict is CapabilityVerdict.PARTIALLY_SUPPORTED
    assert policy.available_capabilities == ["optimization.finite_discrete_policy"]
    nonlinear = assess_capability_fit(
        decision,
        [_requirement("q1", "optimize a nonlinear robotic trajectory with quadratic constraints")],
    )
    assert nonlinear.verdict is CapabilityVerdict.UNSUPPORTED


def test_linear_milp_capability_disposition_is_ready() -> None:
    decision = _decision(TaskFamily.OPTIMIZATION, "q1")
    requirement = _requirement(
        "q1", "create a linear mixed-integer assignment plan with binary decisions"
    )
    fit = assess_capability_fit(decision, [requirement])
    disposition = compute_disposition(
        router=decision, capability_fit=fit, requirements=[requirement], now=NOW
    )
    assert disposition.disposition is ExecutionDisposition.READY


def test_router_flags_discrete_optimization() -> None:
    from mmos.routing.policy import route_problem_spec

    problem = ProblemSpec(
        problem_id="c2-route",
        title="C2 routing",
        overall_goal="Decide a discrete plan",
        subproblems=[
            SubProblem(
                subproblem_id="q1",
                statement="assign items under linear constraints",
                goal="minimize cost",
                inputs=["items"],
                outputs=["assignment"],
            )
        ],
    )
    extraction = ProblemSignalExtraction(
        subproblems=[
            SubproblemSignals(
                subproblem_id="q1",
                signals=ProblemSignals(
                    has_decision_variables=True,
                    has_hard_constraints=True,
                    objective_count=1,
                ),
            )
        ]
    )
    decision = route_problem_spec(
        problem,
        extraction,
        problem_hash="a" * 64,
        problem_spec_hash="b" * 64,
        provider="test",
        model="test",
    )
    assert decision.status is RoutingStatus.SINGLE_TASK
    assert decision.primary_task is TaskFamily.OPTIMIZATION


def test_policy_artifact_hash_is_canonical(tmp_path: Path) -> None:
    paths = project(tmp_path)
    result = run_finite_policy(paths, policy_spec(), experiment_id="exp-hash")
    payload = json.loads(
        (paths.experiment_dir("exp-hash") / "artifacts" / "finite_policy_result.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = payload.pop("artifact_hash")
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded == recomputed == result.artifact_hash
