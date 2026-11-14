from mmos.schemas.simulation import AnalyticOracle, SimulationSpec


class UnstableQueueError(ValueError):
    pass


def compute_analytic_oracle(spec: SimulationSpec) -> AnalyticOracle:
    """Closed-form M/M/1 steady-state values; never imports the simulator."""
    arrival_rate = spec.arrival_rate
    service_rate = spec.service_rate
    if arrival_rate <= 0 or service_rate <= 0:
        raise ValueError("arrival_rate and service_rate must be positive")
    if arrival_rate >= service_rate:
        raise UnstableQueueError(
            f"M/M/1 steady state is undefined for rho={arrival_rate / service_rate:.6g} >= 1"
        )
    rho = arrival_rate / service_rate
    W = 1.0 / (service_rate - arrival_rate)
    Wq = rho / (service_rate - arrival_rate)
    L = arrival_rate * W
    Lq = arrival_rate * Wq
    return AnalyticOracle(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        rho=rho,
        W=W,
        Wq=Wq,
        L=L,
        Lq=Lq,
        utilization=rho,
        throughput=arrival_rate,
    )
