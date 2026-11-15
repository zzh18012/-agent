
from mmos.experiments.pareto_utils import hypervolume_2d
from mmos.schemas.multi_objective import (
    MultiObjectiveSpec,
    ParetoOracle,
    ParetoSolution,
    ParetoVector,
)


def _dominates(left: tuple[float, float], right: tuple[float, float]) -> bool:
    # Independent maximize-dominance implementation for the oracle path.
    no_worse = left[0] >= right[0] and left[1] >= right[1]
    strictly_better = left[0] > right[0] or left[1] > right[1]
    return no_worse and strictly_better


def enumerate_pareto_oracle(spec: MultiObjectiveSpec) -> ParetoOracle:
    """Enumerate every binary subset from the frozen spec, never using the solver."""
    names = [item.project_id for item in spec.items]
    primary = spec.primary_objective_id
    secondary = spec.epsilon_objective_id
    feasible: list[tuple[float, float, dict[str, int]]] = []
    for mask in range(2 ** len(spec.items)):
        decision = {
            item.project_id: int((mask >> index) & 1)
            for index, item in enumerate(spec.items)
        }
        cost = sum(
            item.cost
            for index, item in enumerate(spec.items)
            if (mask >> index) & 1
        )
        if cost > spec.budget:
            continue
        f1 = sum(
            item.benefits[primary]
            for index, item in enumerate(spec.items)
            if (mask >> index) & 1
        )
        f2 = sum(
            item.benefits[secondary]
            for index, item in enumerate(spec.items)
            if (mask >> index) & 1
        )
        feasible.append((float(f1), float(f2), decision))

    vectors = [(f1, f2) for f1, f2, _ in feasible]
    pareto_vectors: list[tuple[float, float]] = []
    for candidate in vectors:
        if not any(
            _dominates(other, candidate)
            for other in vectors
            if other != candidate
        ):
            pareto_vectors.append(candidate)
    unique_vectors = sorted(set(pareto_vectors), key=lambda point: (point[0], point[1]))
    pareto_decisions = [
        ParetoSolution(f1=f1, f2=f2, decision_vector=decision)
        for f1, f2, decision in feasible
        if (f1, f2) in set(unique_vectors)
    ]
    pareto_decisions.sort(
        key=lambda solution: (
            solution.f1,
            solution.f2,
            tuple(solution.decision_vector[name] for name in names),
        )
    )
    reference = (float(spec.reference_point[0]), float(spec.reference_point[1]))
    return ParetoOracle(
        instance_hash=spec.instance_hash,
        search_space_size=2 ** len(spec.items),
        feasible_solution_count=len(feasible),
        pareto_objective_vectors=[
            ParetoVector(f1=f1, f2=f2) for f1, f2 in unique_vectors
        ],
        pareto_decision_vectors=pareto_decisions,
        front_size=len(unique_vectors),
        set_size=len(pareto_decisions),
        hypervolume=hypervolume_2d(unique_vectors, reference),
    )


__all__ = ["enumerate_pareto_oracle"]
