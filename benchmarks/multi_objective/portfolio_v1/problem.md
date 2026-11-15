# Bi-objective Binary Project Portfolio Selection

There are ten candidate projects. Each project has a cost, a benefit on objective A, and
a benefit on objective B. Select a subset of projects with binary decisions subject to a
total budget of 12.

Both objectives are maximized:

- f1 = total benefit A;
- f2 = total benefit B.

The project data are constructed so the budget binds and the objectives trade off. There
is no preference information, utility function, or weight supplied by the problem. The
task is therefore not to choose one "best" portfolio. The task is to construct and
validate the Pareto front of nondominated objective vectors, while distinguishing the
objective-space Pareto front from the decision-space Pareto set when multiple portfolios
produce the same objective vector.

Use a deterministic epsilon-constraint method with exact MILP solves and a two-stage
lexicographic tie-break: maximize objective A first, then fix that value and maximize
objective B. The epsilon schedule must come from the frozen instance and method
configuration, not from the oracle front. Independently enumerate all binary subsets to
construct an exhaustive oracle Pareto front. The candidate and oracle objective-vector
sets must match exactly. Hypervolume is secondary; no single winner or recommended
solution may be declared without preference information.
