from mmos.schemas.graph import GraphProblemSpec, MinCutOracle

_TOLERANCE = 1e-12


def enumerate_min_cuts(spec: GraphProblemSpec) -> MinCutOracle:
    """Enumerate every legal s-t partition using only the frozen spec."""
    nodes = sorted(spec.nodes)
    others = [node for node in nodes if node not in (spec.source, spec.sink)]
    capacities = {(edge.u, edge.v): float(edge.capacity) for edge in spec.edges}
    minimum = float("inf")
    partitions: list[list[str]] = []
    enumerated = 0
    all_partitions: list[list[str]] = []
    all_capacities: list[float] = []
    for mask in range(2 ** len(others)):
        side = {spec.source}
        for index, node in enumerate(others):
            if mask & (1 << index):
                side.add(node)
        enumerated += 1
        capacity = sum(
            value
            for (u, v), value in capacities.items()
            if u in side and v not in side
        )
        all_partitions.append(sorted(side))
        all_capacities.append(capacity)
        if capacity < minimum - _TOLERANCE:
            minimum = capacity
            partitions = [sorted(side)]
        elif abs(capacity - minimum) <= _TOLERANCE:
            partitions.append(sorted(side))
    ordered = sorted(
        zip(all_partitions, all_capacities, strict=True), key=lambda item: item[0]
    )
    all_partitions = [item[0] for item in ordered]
    all_capacities = [item[1] for item in ordered]
    partitions.sort()
    partition_capacities = [
        sum(
            value
            for (u, v), value in capacities.items()
            if u in set(partition) and v not in set(partition)
        )
        for partition in partitions
    ]
    expected = 2 ** len(others)
    if enumerated != expected:
        raise ValueError(
            f"cut enumeration visited {enumerated} partitions, expected {expected}"
        )
    if any(abs(capacity - minimum) > _TOLERANCE for capacity in partition_capacities):
        raise ValueError("a listed minimum cut does not have the minimum capacity")
    return MinCutOracle(
        graph_hash=spec.graph_hash,
        node_count=len(spec.nodes),
        enumerated_cut_count=enumerated,
        expected_cut_count=expected,
        minimum_cut_capacity=minimum,
        minimum_cut_count=len(partitions),
        minimum_cut_partitions=partitions,
        minimum_cut_partition_capacities=partition_capacities,
        all_cut_partitions=all_partitions,
        all_cut_capacities=all_capacities,
        canonical_partition=partitions[0],
    )


__all__ = ["enumerate_min_cuts"]
