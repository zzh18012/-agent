# M/M/1 Queue Simulation Benchmark

A single-server queue receives customers according to a Poisson arrival process with
rate lambda = 2 customers per abstract time unit. Service times are independent and
exponentially distributed with rate mu = 3 services per abstract time unit. The server
uses a first-in-first-out discipline and the system starts empty.

Because rho = lambda / mu = 2/3 < 1, the queue is stable and has finite steady-state
expectations. The task is to build a discrete-event stochastic simulation of this queue,
run multiple independent replications from a recorded seed schedule, apply a warm-up
period before measurement, and estimate:

- W: mean time in system,
- Wq: mean queue waiting time,
- L: mean number in system,
- Lq: mean number waiting in the queue,
- utilization of the server,
- throughput.

The simulation must use a measurement window with time-weighted L, Lq, and utilization,
must avoid right-censoring eligible customers at the end of the window, and must report
confidence intervals over replication means. The results should be validated against the
independent closed-form M/M/1 steady-state formulas (W, Wq, L, Lq, utilization) and
Little's Law. This is a simulation conformance problem, not a regression or forecasting
problem.
