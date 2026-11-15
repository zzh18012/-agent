# Maximum-Flow / Minimum-Cut Network Benchmark

A directed capacitated network has a single source `s` and a single sink `t`. Every
edge `(u, v)` has a finite nonnegative capacity. The task is to compute a feasible
maximum s-t flow and independently validate it with the Max-Flow Min-Cut Theorem.

A feasible flow must satisfy:

- capacity constraints: 0 <= f(u,v) <= capacity(u,v) on every directed edge;
- flow conservation: inflow equals outflow at every node except source and sink;
- the objective is the value leaving the source minus flow entering it.

The network has multiple s-to-t paths and shared bottlenecks. It is small enough to
enumerate every legal s-t partition and compute the minimum directed cut independently
of the flow solver. Only edges from the source side S to the sink side T contribute to
cut capacity; reverse edges do not.

This is a graph/network problem, not a generic LP or prediction task. The independent
minimum-cut enumeration must not call the maximum-flow solver or read its residual
network. This is a conformance check for one frozen graph, not an algorithm performance
comparison.
