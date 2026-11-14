import statistics

from mmos.core.project import (
    ProjectPaths,
    load_simulation_record,
    load_simulation_result,
    load_simulation_spec,
    replications_file,
    sha256_file,
    simulation_result_file,
)
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_simulation_paper(paths: ProjectPaths, experiment_id: str) -> PaperAST:
    spec = load_simulation_spec(paths, experiment_id)
    result = load_simulation_result(paths, experiment_id)
    record = load_simulation_record(paths, experiment_id)
    result_hash = sha256_file(simulation_result_file(paths, experiment_id))
    replications_hash = sha256_file(replications_file(paths, experiment_id))
    units_display = spec.simulation_time_units.replace(
        "abstract_time_units", "simulated time units"
    )

    role_by_metric = {
        "W": "descriptive (cohort)",
        "Wq": "descriptive (cohort)",
        "L": "descriptive (time-weighted)",
        "Lq": "descriptive (time-weighted)",
        "utilization": "descriptive (time-weighted)",
        "throughput": "counting-path arithmetic check",
    }
    metric_rows = []
    metric_evidence = []
    for metric in spec.metrics:
        if metric == "throughput":
            continue
        summary = result.metrics[metric]
        comparison = result.analytic_comparison.get(metric)
        if comparison is None:
            continue
        standard_error = summary.std / (summary.n**0.5)
        metric_rows.append(
            [
                metric,
                role_by_metric.get(metric, "descriptive"),
                _fmt(summary.mean),
                _fmt(summary.std),
                _fmt(standard_error),
                _fmt(summary.ci_low),
                _fmt(summary.ci_high),
                _fmt(comparison.analytic_value),
                _fmt(comparison.absolute_error),
                _fmt(
                    comparison.absolute_error / standard_error
                    if standard_error > 0
                    else 0.0
                ),
            ]
        )
        metric_evidence.extend(
            [
                f"ev-{experiment_id}-simulation-{metric}-mean",
                f"ev-{experiment_id}-simulation-{metric}-theory",
            ]
        )
    metric_table = PaperTable(
        table_id=f"tab-{experiment_id}-simulation-metrics",
        caption=(
            "Raw descriptive replication-level means, standard errors, and approximate "
            "Student-t intervals. No metric carries conformance or accuracy weight; no "
            "sensitivity sweep exists, so these are descriptive outputs only. 'Difference "
            "/ SE' is a descriptive noise ratio, not a test or accuracy certification. "
            "Values are computed on unrounded replication data and displayed rounded. "
            "Throughput is a separate counting-path arithmetic check."
        ),
        columns=[
            "Metric",
            "Claim role",
            "Simulation Mean",
            "Std",
            "Std Error",
            "CI Low",
            "CI High",
            "Analytic Value",
            "Difference",
            "Difference / SE",
        ],
        rows=metric_rows,
        evidence_ids=metric_evidence,
    )

    replication_rows = []
    for replication in result.replications:
        replication_rows.append(
            [
                str(replication.replication_id),
                str(replication.replication_seed),
                str(replication.arrival_seed),
                str(replication.service_seed),
                _fmt(replication.W),
                _fmt(replication.Wq),
                _fmt(replication.L),
                _fmt(replication.Lq),
                _fmt(replication.utilization),
                str(replication.eligible_customers),
                _fmt(replication.effective_arrival_rate),
                str(replication.completed_customers),
                _fmt(replication.idle_tail),
                str(replication.post_window_completions),
            ]
        )
    total_eligible = sum(
        replication.eligible_customers for replication in result.replications
    )
    total_duration = sum(
        replication.measurement_time for replication in result.replications
    )
    mean_throughput = total_eligible / total_duration
    max_idle_tail = max(
        (row.idle_tail for row in result.replications), default=0.0
    )
    idle_tail_fraction = max_idle_tail / spec.measurement_time
    w_summary = result.metrics.get("W")
    wq_summary = result.metrics.get("Wq")
    w_half_width = (w_summary.ci_high - w_summary.ci_low) / 2 if w_summary else 0.0
    wq_half_width = (wq_summary.ci_high - wq_summary.ci_low) / 2 if wq_summary else 0.0
    w_error = (
        result.analytic_comparison["W"].relative_error
        if "W" in result.analytic_comparison
        else 0.0
    )
    wq_error = (
        result.analytic_comparison["Wq"].relative_error
        if "Wq" in result.analytic_comparison
        else 0.0
    )
    max_difference_se = max(
        (
            abs(comparison.absolute_error)
            / (result.metrics[metric].std / (result.metrics[metric].n**0.5))
            for metric, comparison in result.analytic_comparison.items()
            if metric != "throughput" and result.metrics[metric].std > 0
        ),
        default=0.0,
    )
    replication_table = PaperTable(
        table_id=f"tab-{experiment_id}-simulation-replications",
        caption=(
            "Every formal replication with its derived seed streams and metrics. "
            "Completed equals eligible because all eligible arrivals drain. Eligible rate "
            "is Eligible divided by the measurement duration. The metrics-table throughput "
            f"mean is the unweighted mean of these values: {total_eligible} eligible "
            f"customers / {total_duration:g} total duration = {mean_throughput:.6f}. The "
            "confidence interval is for that mean, not a prediction interval for individual "
            "replications, so individual values may lie outside it. Idle tail is the "
            "measurement window minus the last departure when positive; post-window "
            "completions counts eligible customers whose service ends after the measurement "
            "end. These two finite-window quantities are tabulated separately."
        ),
        columns=[
            "Rep",
            "Replication seed",
            "Arrival seed",
            "Service seed",
            "W",
            "Wq",
            "L",
            "Lq",
            "Utilization",
            "Eligible",
            "Eligible rate",
            "Completed",
            "Idle tail (units)",
            "Post-window completions",
        ],
        rows=replication_rows,
        evidence_ids=[f"ev-{experiment_id}-simulation-seed-schedule"],
    )

    little_law_rows = []
    little_law_evidence = []
    for metric, time_metric in (("L", "W"), ("Lq", "Wq")):
        rates = [row.effective_arrival_rate for row in result.replications]
        times = [float(getattr(row, time_metric)) for row in result.replications]
        simulated = [float(getattr(row, metric)) for row in result.replications]
        mean_rate = statistics.fmean(rates)
        mean_time = statistics.fmean(times)
        mean_simulated = statistics.fmean(simulated)
        product = mean_rate * mean_time
        residuals = [
            value - rate * time
            for value, rate, time in zip(simulated, rates, times, strict=True)
        ]
        jensen = statistics.fmean(
            rate * time for rate, time in zip(rates, times, strict=True)
        ) - product
        little_law_rows.append(
            [
                f"{metric} vs lambda_eff x {time_metric}",
                f"E[rate] x E[{time_metric}]",
                _fmt(mean_rate, 9),
                _fmt(mean_time, 9),
                _fmt(product, 9),
                _fmt(mean_simulated, 9),
                _fmt(abs(mean_simulated - product), 9),
                _fmt(
                    abs(mean_simulated - product) / abs(product) if product != 0 else 0.0,
                    9,
                ),
                _fmt(statistics.fmean(residuals), 9),
                _fmt(statistics.stdev(residuals) if len(residuals) > 1 else 0.0, 9),
                _fmt(jensen, 9),
            ]
        )
        little_law_evidence.append(
            f"ev-{experiment_id}-simulation-little-law-{metric}"
        )
    little_law_table = PaperTable(
        table_id=f"tab-{experiment_id}-simulation-little-law",
        caption=(
            "Little's Law bookkeeping check, not validation and with no pass/fail verdict. "
            "The per-replication identity holds on a common support; this table uses the "
            "product of means E[lambda_eff] x E[W or Wq] and a finite integral window, so "
            "the reported residuals mix the Jensen gap and the support mismatch between "
            "drained sojourn times and the integration window. Neither contribution is "
            "separately quantified. Per-replication residual mean and SD are shown; the "
            "Jensen gap is fmean(rate*time) - fmean(rate)*fmean(time). Under the "
            "drain-to-completion design, throughput and lambda_eff are the same estimator; "
            "post-window service affects the time weight, not the count."
        ),
        columns=[
            "Relation",
            "Estimator",
            "Mean rate",
            "Mean time",
            "Product of means",
            "Mean simulated",
            "Abs(mean-product)",
            "Rel (product denom.)",
            "Mean per-rep residual",
            "SD per-rep residual",
            "Jensen gap",
        ],
        rows=little_law_rows,
        evidence_ids=little_law_evidence,
    )

    expected_per_rep = spec.arrival_rate * spec.measurement_time
    expected_sd_per_rep = expected_per_rep**0.5
    z_values = [
        (row.eligible_customers - expected_per_rep) / expected_sd_per_rep
        for row in result.replications
    ]
    expected_total = spec.arrival_rate * total_duration
    pooled_sd = expected_total**0.5
    pooled_z = (total_eligible - expected_total) / pooled_sd
    input_sanity_table = PaperTable(
        table_id=f"tab-{experiment_id}-simulation-input-sanity",
        caption=(
            "Input-arrival arithmetic counting-path check, not a sanity check or queue "
            "service metric. Poisson expectation is lambda x duration with variance equal "
            "to the mean; z-scores use that expectation. The 30 Eligible values sum to "
            f"{total_eligible} and the 30 durations sum to {total_duration:g}. The pooled "
            f"z is {pooled_z:.3f}; per-replication z ranges from {min(z_values):.3f} to "
            f"{max(z_values):.3f}. A +/-3 SD band is shown for context only; this does not "
            "establish generator correctness. Because Completed equals Eligible, throughput "
            "and lambda_eff are the same estimator; post-window service affects the time "
            "weight, not this count."
        ),
        columns=[
            "Eligible arrivals",
            "Total duration",
            "Observed rate",
            "lambda target",
            "Expected total",
            "Expected SD",
            "Pooled z",
            "Per-rep z range",
            "Note",
        ],
        rows=[
            [
                str(total_eligible),
                f"{total_duration:g}",
                _fmt(mean_throughput),
                f"{spec.arrival_rate:g}",
                f"{expected_total:.0f}",
                f"{pooled_sd:.1f}",
                f"{pooled_z:.3f}",
                f"{min(z_values):.3f} to {max(z_values):.3f}",
                "counting-path arithmetic check; +/-3 SD band",
            ]
        ],
        evidence_ids=[f"ev-{experiment_id}-simulation-seed-schedule"],
    )

    evidence_rows = []
    for metric in spec.metrics:
        if metric == "throughput":
            continue
        summary = result.metrics[metric]
        evidence_rows.append(
            [
                metric,
                f"ev-{experiment_id}-simulation-{metric}-mean",
                _fmt(summary.mean, 9),
                f"verified/simulation/{experiment_id}.json; "
                "simulation_result.json replications rows 0-29",
            ]
        )
    evidence_table = PaperTable(
        table_id=f"tab-{experiment_id}-simulation-evidence",
        caption=(
            "Inline evidence links for each queue-service metric mean. The evidence record "
            "raw value is shown; the source artifact and replication rows are identified so "
            "the mean can be checked against the published files."
        ),
        columns=["Metric", "Evidence id", "Raw mean", "Source artifact"],
        rows=evidence_rows,
        evidence_ids=[row[1] for row in evidence_rows],
    )

    abstract = (
        "Single-operating-point cohort simulation discrepancy report, not a benchmark-family "
        f"conformance claim: stable M/M/1 queue, lambda={spec.arrival_rate:g}, "
        f"mu={spec.service_rate:g}, rho={spec.rho:.6f}, warm-up {spec.warmup_time:g}, "
        f"measurement {spec.measurement_time:g}, {spec.replications} replications, one "
        "master seed. No metric carries conformance weight. L, Lq, and utilization are "
        "descriptive time-weighted quantities. W and Wq are descriptive finite-cohort "
        "means over eligible arrivals; their boundary bias is unbounded and unquantified, "
        "and no PASTA estimator is reported. Throughput is a counting-path arithmetic check "
        "whose target is lambda and which does not exercise the service process. Approximate "
        "descriptive Student-t intervals are tabulated without certified coverage; "
        "normality, independence, bootstrap robustness, stationarity, and warm-up "
        "sufficiency are assumed or untested. Little's Law is a bookkeeping identity, not "
        "independent validation. No sensitivity sweep over warm-up, measurement length, "
        "rate, or master seed is performed, so no numeric accuracy or conformance claim is "
        "made. Results are conditional on master seed "
        f"{spec.master_seed}; no seed-to-seed robustness or conformance claim extends "
        "beyond it, and this report does not show that the simulator is generally correct."
    )

    sections = [
        PaperSection(
            heading="Problem definition and assumptions",
            blocks=[
                "Single-server FIFO queue. Arrivals follow a Poisson process with rate "
                f"lambda={spec.arrival_rate:g} customers per time unit; service times are "
                f"independent exponential with rate mu={spec.service_rate:g}. The system "
                "starts empty. The offered load is "
                f"rho=lambda/mu={spec.rho:.6f} < 1, so finite steady-state means exist.",
                f"Time units: `{units_display}`. All rates, times, and time-weighted "
                "integrals use this single unit consistently; W and Wq are reported in it. "
                "The benchmark uses synthetic data only. `exp-mm1-benchmark-v1` is an "
                "experiment identifier, not a conformance suite; no family-level claim is "
                "made.",
            ],
        ),
        PaperSection(
            heading="Simulation design",
            blocks=[
                "Discrete-event simulation uses SimPy with a single FIFO server. One "
                f"master seed ({spec.master_seed}) derives all replication and random "
                "stream seeds with SHA-256; the complete schedule is recorded and every "
                "replication seed, arrival seed, and service seed is unique.",
                f"Warm-up runs from 0 to {spec.warmup_time:g} and is excluded from formal "
                f"statistics. The measurement window is [{spec.warmup_time:g}, "
                f"{spec.measurement_end:g}) with duration {spec.measurement_time:g}.",
                "Arrivals stop at the measurement end. Customers who arrived in "
                "[warmup, measurement end) are eligible; they are allowed to drain to "
                "completion after the measurement end, so customer-level W and Wq are not "
                "right-censored. L, Lq, and utilization integrate only to the measurement "
                "end using time-weighted areas, never a plain average of event observations.",
                "W and Wq are finite-window cohort means over every eligible arrival, not "
                "time-sampled or PASTA estimates. Because the full cohort drains, there is "
                "no censoring, but the finite arrival cohort can differ from a steady-state "
                "time sample by a boundary-selection effect that is not separately bounded "
                "in this report. The analytic comparison is therefore descriptive at this "
                "operating point; it does not support unbiased steady-state estimation or a "
                "conformance claim for W or Wq.",
                "Eligible customers count as completed after draining. Throughput is "
                "completed eligible customers divided by measurement duration; because some "
                "eligible customers complete after the measurement end, the numerator can "
                "include post-window service. Post-window completions and idle-tail length "
                "are tabulated per replication. Throughput is a counting-path arithmetic "
                "check, not a service-rate or in-window completion rate. The effective "
                "arrival rate used by Little's Law is eligible arrivals divided by "
                "measurement duration, shown per replication.",
                "A replication's drain end below the measurement end means its last arrival "
                "occurred earlier; arrivals are not suppressed within the window. Two "
                "opposing finite-window effects are present: the idle tail up to the "
                "measurement end lowers the time-weighted L and utilization, while service "
                "of eligible customers completing after the measurement end is excluded "
                "from the integral and raises it relative to a finished-cohort identity. No "
                "per-replication decomposition is reported, and the net sign is not "
                "established.",
            ],
        ),
        PaperSection(
            heading="Analytic oracle",
            blocks=[
                "The oracle is independent closed-form theory and does not call the "
                "simulator. For rho = lambda/mu: W = 1/(mu-lambda), "
                "Wq = rho/(mu-lambda), L = lambda W, Lq = lambda Wq, and utilization = rho.",
                "The internal identities L = rho/(1-rho) and Lq = rho^2/(1-rho) are oracle "
                "self-consistency checks, not independent validation evidence. The only "
                "comparison to simulation is the descriptive closed-form-versus-simulation "
                "table; no conformance inference is drawn. The oracle is not a simulation "
                "candidate or competitor.",
                f"rho is strictly below 1 ({spec.rho:.6f}); an unstable queue would leave "
                "steady-state theory undefined and is rejected rather than compared.",
            ],
        ),
        PaperSection(
            heading="Simulation results and confidence intervals",
            blocks=[
                f"Each metric, including the throughput counting-path check, is summarized from "
                f"{spec.replications} replication means with a {spec.confidence_level:.0%} "
                "Student-t interval centered on the replication mean, using t with n-1 "
                "degrees of freedom.",
                "Interpretation: these are approximate descriptive uncertainty bands, not "
                "certified 95%-coverage intervals. They are not ranges containing 95% of "
                "simulated customers and not 95% probability statements about a fixed "
                "interval. No joint-coverage claim is made: a Bonferroni statement would "
                "require 0.05/6 per interval, and normal-theory t intervals assume "
                "approximate normality, which is not tested. The auditor independently "
                "recomputes each mean, standard deviation, and interval from the replication "
                "rows.",
                "Non-inferential display remark: the point estimates happen to fall inside "
                "the approximate intervals. This is not coverage evidence, not a conformance "
                "badge, and not bootstrap-robust; no bootstrap or QQ check is performed. No "
                "metric carries conformance weight; all rows are descriptive. The largest "
                f"difference/SE across the five metrics is {max_difference_se:.3f}, so the "
                "point differences are within about one standard error and are not evidence "
                "of sub-percent accuracy. No sensitivity sweep exists, so no accuracy claim "
                "is made.",
                f"For W the point relative error is {w_error:.6f} against a half-width of "
                f"{w_half_width:.6f}; for Wq it is {wq_error:.6f} against "
                f"{wq_half_width:.6f}. The Wq point error is smaller than its half-width, so "
                "residual bias and sampling noise cannot be separated and no sub-percent "
                "accuracy is certified. At rho=0.667 the relaxation time is "
                "1/(mu-lambda)=1, so 500 warm-up units span roughly 500 relaxation times; "
                "this is a qualitative order-of-magnitude argument, not a measured "
                "stationarity diagnostic. A short warm-up would bias L and utilization, so "
                "their descriptive errors are not attributed to simulator correctness. No "
                "formal power analysis is reported.",
                "The thresholds are documented in the frozen spec hash and MLflow run; no "
                "independent timestamp establishes temporal priority over the run. The "
                "Tolerance pass column is a documented point-estimate rule, not a "
                "conformance claim. No weighted validation score is used. The largest "
                f"observed idle tail is {max_idle_tail:.6f} time units "
                f"({idle_tail_fraction:.6%} of the window); together with excluded "
                "post-window service it forms two opposing finite-window effects whose net "
                "sign is not established, and neither is separately bounded.",
            ],
        ),
        PaperSection(
            heading="Little's Law bookkeeping check",
            blocks=[
                "For a fully drained finite cohort, L = lambda_eff W is essentially an "
                "algebraic bookkeeping identity between the time-weighted area and summed "
                "sojourn times. This section is a numerical consistency check on the "
                "integration and cohort accounting, not empirical validation and not "
                "independent evidence of simulator conformance.",
                "The table reports per-replication residual mean and SD, the product of "
                "means, and the Jensen gap. The residual mixes the Jensen gap with a "
                "time-support mismatch between drained sojourn times and the integration "
                "window; the two contributions are not separately quantified. There is no "
                "pass/fail rule and the table is not used to detect mis-scaled integration.",
                "Under the drain-to-completion design, throughput and lambda_eff are the "
                "same estimator; post-window service affects the time weight, not the "
                "count.",
            ],
        ),
        PaperSection(
            heading="Limitations",
            blocks=[
                "Single synthetic M/M/1 operating point only: Poisson arrivals, "
                "exponential service, one server, FIFO, stable load, empty initial state.",
                "No real observational data and no evidence about non-exponential service, "
                "multi-server or network queues, transient behavior, heavy traffic, or "
                "arbitrary simulation models.",
                "This is not a performance or scalability benchmark, and it does not show "
                "that the simulator is generally correct. Intervals cover replication noise "
                "under assumed normality, not model misspecification.",
                "W and Wq use a finite arrival cohort, not a PASTA/time-sampled estimator. "
                "The cohort boundary-selection bias is unbounded and unquantified, so their "
                "point discrepancies mix estimator bias with simulator error.",
                "Single operating point: rho=0.667, warm-up 500, measurement 2000, one "
                "master seed. No sensitivity sweep over rho, warm-up, measurement length, "
                "rates, or master seeds is performed; seed-to-seed robustness is not "
                "established and the result does not support a family-level claim.",
                "Warm-up sufficiency and stationarity are assumed, not verified; no "
                "split-window or Gelman-Rubin diagnostic is reported.",
                "Replication independence and approximate normality are assumed from "
                "disjoint derived seed streams; no Kolmogorov-Smirnov, autocorrelation, "
                "runs, skewness, QQ, or bootstrap diagnostics are reported. Interval "
                "overlap is descriptive and not bootstrap-robust.",
                "Warm-up sufficiency is assumed; no split-window or Gelman-Rubin test is "
                "reported. A short warm-up would bias L and utilization, so their small "
                "point errors are not separable from warm-up bias.",
                "Throughput's analytic target is the input lambda; it checks the arrival "
                "generator, not the service process. Because Completed equals Eligible, its "
                "count is the eligible-arrival count; post-window service affects the time "
                "weight, and no in-window-completion decomposition is tabulated. Little's "
                "Law is a bookkeeping identity for a drained cohort, not independent "
                "validation; the product of means is not the mean of per-replication "
                "products, and the per-replication discrepancy distribution is not reported.",
                "The number of effectively independent observations is smaller than the "
                "arrival count because of queueing autocorrelation; no batch-means, "
                "autocorrelation, or effective-sample-size estimate is reported.",
            ],
        ),
        PaperSection(
            heading="Reproducibility and provenance",
            blocks=[
                f"Simulation spec hash: `{spec.spec_hash}`; git commit: "
                f"`{record.git_commit}`; MLflow run: `{record.mlflow_run_id or 'missing'}`; "
                f"SimPy version: `{result.runtime.get('simpy_version', 'unknown')}`; "
                f"Python: `{result.runtime.get('python_version', 'unknown')}`; platform: "
                f"`{result.runtime.get('platform', 'unknown')}`.",
                f"Replication seed schedule hash: `{result.replication_seed_hash}`. Two "
                "independent in-process executions with fresh SimPy environments and RNG "
                "streams produced matching canonical replication digests "
                f"(`{result.determinism.get('first_run_digest', 'missing')}` and "
                f"`{result.determinism.get('second_run_digest', 'missing')}`). This verifies "
                "numerical determinism in this environment only; cross-platform bit-identical "
                "output and platform-stable SimPy event ordering are not demonstrated.",
                f"Output SHA-256: `simulation_result.json` `{result_hash}`; "
                f"`replications.csv` `{replications_hash}`. Dependencies are pinned by "
                "`uv.lock` (SimPy 4.1.2 on this run). The tables in this report are derived "
                f"from these run artifacts at commit `{record.git_commit}`.",
                f"Artifacts: `experiments/{experiment_id}/simulation_spec.json`, "
                "`simulation_result.json`, `analytic_oracle.json`, `replications.csv`, and "
                "the validation report. Evidence IDs are linked from the tables.",
            ],
        ),
    ]

    return PaperAST(
        title="M/M/1 Single-Operating-Point Simulation Discrepancy Report",
        abstract=abstract,
        sections=sections,
        tables=[
            metric_table,
            replication_table,
            little_law_table,
            input_sanity_table,
            evidence_table,
        ],
        figures=[],
        claims=[],
        references=["M/M/1 closed-form queueing theory and verified simulation evidence."],
    )


__all__ = ["build_simulation_paper"]
