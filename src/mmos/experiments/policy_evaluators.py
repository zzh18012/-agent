from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field

from mmos.schemas.discrete import (
    DiscreteDecisionVariable,
    PolicyValue,
    policy_value_key,
)


class PolicyEvaluatorError(ValueError):
    pass


@dataclass(frozen=True)
class PolicyEvaluation:
    feasible: bool
    objective: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    objective_components: dict[str, float] | None = None
    infeasibility_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyEvaluator:
    evaluator_id: str
    version: str
    decision_variables: tuple[DiscreteDecisionVariable, ...]
    parameter_names: tuple[str, ...]
    description: str

    def evaluate(
        self, policy: dict[str, PolicyValue], parameters: dict[str, float]
    ) -> PolicyEvaluation:
        raise NotImplementedError

    @property
    def implementation_hash(self) -> str:
        try:
            source = inspect.getsource(type(self)).replace("\r\n", "\n")
        except (OSError, TypeError):
            source = ""
        payload = json.dumps(
            {
                "evaluator_id": self.evaluator_id,
                "version": self.version,
                "decision_variables": [
                    {
                        "variable_id": decision.variable_id,
                        "domain": [
                            policy_value_key(value) for value in decision.canonical_domain
                        ],
                    }
                    for decision in sorted(
                        self.decision_variables, key=lambda item: item.variable_id
                    )
                ],
                "parameter_names": list(self.parameter_names),
                "description": self.description,
                "source": source,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SyntheticProcessPolicy(PolicyEvaluator):
    """Synthetic five-decision production-process policy used only for engine fixtures.

    It is deliberately not a model of any real production problem.
    """

    def __init__(self) -> None:
        super().__init__(
            evaluator_id="registered.synthetic_process_policy",
            version="1",
            decision_variables=(
                DiscreteDecisionVariable(variable_id="inspect_component", domain=[False, True]),
                DiscreteDecisionVariable(variable_id="premium_route", domain=[False, True]),
                DiscreteDecisionVariable(variable_id="inspect_final", domain=[False, True]),
                DiscreteDecisionVariable(variable_id="rework", domain=[False, True]),
                DiscreteDecisionVariable(variable_id="expedite", domain=[False, True]),
            ),
            parameter_names=(
                "component_cost",
                "assembly_cost",
                "component_inspection_cost",
                "final_inspection_cost",
                "premium_route_extra",
                "rework_cost",
                "expedite_cost",
                "defect_penalty",
                "defect_rate",
                "residual_after_component_inspection",
                "residual_after_final_inspection",
                "rework_residual_factor",
            ),
            description=(
                "expected cost of a synthetic five-decision production policy; "
                "not a real production model"
            ),
        )

    def evaluate(
        self, policy: dict[str, PolicyValue], parameters: dict[str, float]
    ) -> PolicyEvaluation:
        inspect_component = bool(policy["inspect_component"])
        premium_route = bool(policy["premium_route"])
        inspect_final = bool(policy["inspect_final"])
        rework = bool(policy["rework"])
        expedite = bool(policy["expedite"])

        reasons: list[str] = []
        if premium_route and not inspect_component:
            reasons.append("premium_route_requires_inspect_component")
        if rework and not inspect_final:
            reasons.append("rework_requires_inspect_final")
        if reasons:
            return PolicyEvaluation(
                feasible=False,
                objective=None,
                infeasibility_reasons=tuple(reasons),
            )

        defect_rate = parameters["defect_rate"]
        purchase = parameters["component_cost"]
        assembly = parameters["assembly_cost"] + (
            parameters["premium_route_extra"] if premium_route else 0.0
        )
        inspection = 0.0
        if inspect_component:
            inspection += parameters["component_inspection_cost"]
        if inspect_final:
            inspection += parameters["final_inspection_cost"]
        escape = defect_rate
        if inspect_component:
            escape *= parameters["residual_after_component_inspection"]
        if inspect_final:
            escape *= parameters["residual_after_final_inspection"]
        rework_total = 0.0
        if rework:
            rework_total = parameters["rework_cost"] * defect_rate
            escape *= parameters["rework_residual_factor"]
        penalty = parameters["defect_penalty"] * escape
        expedite_total = parameters["expedite_cost"] if expedite else 0.0
        components = {
            "purchase": purchase,
            "assembly": assembly,
            "inspection": inspection,
            "rework": rework_total,
            "penalty": penalty,
            "expedite": expedite_total,
        }
        objective = sum(components.values())
        return PolicyEvaluation(
            feasible=True,
            objective=objective,
            metrics={
                "expected_total_cost": objective,
                "expected_failures": escape,
                "expected_quality_yield": 1.0 - escape,
            },
            objective_components=components,
        )


_EVALUATORS: dict[str, PolicyEvaluator] = {
    "registered.synthetic_process_policy": SyntheticProcessPolicy(),
}


def get_policy_evaluator(evaluator_id: str) -> PolicyEvaluator:
    try:
        return _EVALUATORS[evaluator_id]
    except KeyError as exc:
        raise PolicyEvaluatorError(
            f"unknown registered policy evaluator: {evaluator_id}"
        ) from exc


def policy_evaluator_registry() -> dict[str, PolicyEvaluator]:
    return dict(_EVALUATORS)


def validate_parameters(
    evaluator: PolicyEvaluator, parameters: dict[str, float]
) -> None:
    expected = set(evaluator.parameter_names)
    provided = set(parameters)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise PolicyEvaluatorError(
            f"parameter names do not match evaluator {evaluator.evaluator_id}: "
            f"missing={missing}, extra={extra}"
        )
    for name, value in parameters.items():
        if not math.isfinite(value):
            raise PolicyEvaluatorError(f"parameter {name!r} is not finite")


def validate_policy(
    evaluator: PolicyEvaluator, policy: dict[str, PolicyValue]
) -> None:
    expected = {decision.variable_id for decision in evaluator.decision_variables}
    provided = set(policy)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise PolicyEvaluatorError(
            f"policy keys do not match evaluator: missing={missing}, extra={extra}"
        )
    for decision in evaluator.decision_variables:
        allowed = {policy_value_key(value) for value in decision.domain}
        if policy_value_key(policy[decision.variable_id]) not in allowed:
            raise PolicyEvaluatorError(
                f"policy value for {decision.variable_id!r} is outside its declared domain"
            )


__all__ = [
    "PolicyEvaluation",
    "PolicyEvaluator",
    "PolicyEvaluatorError",
    "SyntheticProcessPolicy",
    "get_policy_evaluator",
    "policy_evaluator_registry",
    "validate_parameters",
    "validate_policy",
]
