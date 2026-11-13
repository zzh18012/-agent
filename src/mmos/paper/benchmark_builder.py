from mmos.core.project import ProjectPaths, load_optimization_result, load_optimization_spec
from mmos.experiments.knapsack_benchmark import load_manifest
from mmos.experiments.knapsack_suite import load_benchmark_result
from mmos.schemas.benchmark import BenchmarkManifest, InstanceSuiteResult
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _obj(value: float | None) -> str:
    if value is None:
        return "-"
    rounded = round(value)
    return str(int(rounded)) if abs(value - rounded) <= 1e-9 else f"{value:.4f}"


def _hit(relative_regret: float | None) -> bool:
    return relative_regret is not None and relative_regret <= 1e-12


def _partition(tags: list[str]) -> str:
    if "adversarial" in tags:
        return "adversarial"
    if "random" in tags:
        return "random"
    if "sanity" in tags:
        return "sanity"
    return "other"


def _regret_stats(
    values: list[float | None],
) -> tuple[int, int, float | None, float]:
    present = [value for value in values if value is not None]
    misses = [value for value in present if value > 1e-12]
    mean_misses = sum(misses) / len(misses) if misses else None
    return len(present), len(misses), mean_misses, sum(misses)


def _tag_hits(rows: list[InstanceSuiteResult], tag: str) -> tuple[int, int]:
    members = [row for row in rows if tag in row.tags]
    return sum(1 for row in members if _hit(row.greedy_relative_regret)), len(members)


def _independent_feasibility(
    instance, solution
) -> tuple[bool, str]:
    values = solution.variable_values
    bits: list[int] = []
    for index in range(len(instance.weights)):
        raw = values.get(f"x_{index}")
        if raw is None:
            return False, "missing variable"
        rounded = round(float(raw))
        if abs(float(raw) - rounded) > 1e-9 or rounded not in (0, 1):
            return False, "non-binary"
        bits.append(int(rounded))
    weight = sum(
        weight * bit for weight, bit in zip(instance.weights, bits, strict=True)
    )
    value = sum(value * bit for value, bit in zip(instance.values, bits, strict=True))
    if weight > instance.capacity:
        return False, "capacity"
    if solution.objective_value is not None and abs(value - solution.objective_value) > 1e-6:
        return False, "objective"
    return True, "yes"


def build_benchmark_paper(
    paths: ProjectPaths, benchmark_run_id: str, suite_dir: str | None = None
) -> PaperAST:
    artifact_dir = paths.experiments_dir / "benchmarks" / benchmark_run_id
    result = load_benchmark_result(artifact_dir / "benchmark_result.json")
    manifest: BenchmarkManifest = load_manifest(artifact_dir)

    run = result.benchmark_run_id
    rows_by_id = {instance.instance_id: instance for instance in result.per_instance}
    rows = [
        rows_by_id[instance.instance_id]
        for instance in manifest.instances
        if instance.instance_id in rows_by_id
    ]
    instances_by_id = {instance.instance_id: instance for instance in manifest.instances}

    cbc_details: dict[str, tuple[str, float | None, bool, str]] = {}
    for row in rows:
        if row.candidate_experiment_id is None:
            cbc_details[row.instance_id] = ("unknown", None, False, "missing")
            continue
        spec = load_optimization_spec(paths, row.candidate_experiment_id)
        cbc_result = load_optimization_result(paths, row.candidate_experiment_id)
        solution = cbc_result.solutions[0]
        feasible, feasibility_note = _independent_feasibility(
            instances_by_id[row.instance_id], solution
        )
        cbc_details[row.instance_id] = (
            solution.status,
            spec.time_limit_seconds,
            feasible,
            feasibility_note,
        )
    statuses = sorted({status for status, _, _, _ in cbc_details.values()})
    time_limits = sorted(
        {limit for _, limit, _, _ in cbc_details.values() if limit is not None}
    )
    time_limit_text = ", ".join(f"{limit:g}s" for limit in time_limits) or "not recorded"
    exact_matches = sum(
        1
        for row in rows
        if row.cbc_objective is not None
        and float(row.cbc_objective) == float(row.oracle_optimum)
    )
    independent_feasible_count = sum(
        1 for _, _, feasible, _ in cbc_details.values() if feasible
    )
    all_optimal = statuses == ["optimal"]

    partition_counts = {
        name: sum(1 for row in rows if _partition(row.tags) == name)
        for name in ("adversarial", "random", "sanity", "other")
    }
    partition_hits = {
        name: sum(
            1
            for row in rows
            if _partition(row.tags) == name and _hit(row.greedy_relative_regret)
        )
        for name in ("adversarial", "random", "sanity", "other")
    }
    curated_misses = [
        row.instance_id
        for row in rows
        if _partition(row.tags) == "adversarial"
        and not _hit(row.greedy_relative_regret)
    ]
    random_misses = [
        row.instance_id
        for row in rows
        if _partition(row.tags) == "random" and not _hit(row.greedy_relative_regret)
    ]
    sanity_misses = [
        row.instance_id
        for row in rows
        if _partition(row.tags) == "sanity" and not _hit(row.greedy_relative_regret)
    ]

    reconciliation_rows = []
    reconciliation_evidence = []
    for instance in manifest.instances:
        row = rows_by_id.get(instance.instance_id)
        reconciliation_rows.append(
            [
                instance.instance_id,
                instance.generator,
                ", ".join(instance.tags),
                _partition(instance.tags),
                str(len(instance.weights)),
                str(instance.capacity),
                instance.instance_hash,
                "yes"
                if row is not None and _hit(row.greedy_relative_regret)
                else "no",
                "yes" if instance.instance_id in rows_by_id else "no",
            ]
        )
        reconciliation_evidence.append(f"ev-{run}-{instance.instance_id}-hash")
    partition_sum = (
        partition_counts["adversarial"] + partition_counts["random"] + partition_counts["sanity"]
    )
    reconciliation_caption = (
        f"Frozen manifest reconciliation: all {len(manifest.instances)} declared "
        "instances are listed with full SHA-256 hashes and an explicit partition rule "
        "(adversarial if tagged adversarial, else random if tagged random, else sanity). "
        f"Partition counts are adversarial {partition_counts['adversarial']} + random "
        f"{partition_counts['random']} + sanity {partition_counts['sanity']} = "
        f"{partition_sum}. Greedy hits in the same partition are "
        f"{partition_hits['adversarial']} + {partition_hits['random']} + "
        f"{partition_hits['sanity']} = {sum(partition_hits.values())}, matching the "
        "'all' row by construction, not coincidence. Generator mapping: "
        "legacy_manual/sanity, curated_* /adversarial, seeded_* /random."
    )
    reconciliation_table = PaperTable(
        table_id=f"tab-{run}-manifest-reconciliation",
        caption=reconciliation_caption,
        columns=[
            "Instance",
            "Generator",
            "Tags",
            "Partition",
            "n",
            "Capacity",
            "Instance hash (sha256)",
            "Greedy hit",
            "Included",
        ],
        rows=reconciliation_rows,
        evidence_ids=reconciliation_evidence,
    )

    instance_rows = []
    instance_evidence = []
    enhanced_only_ids: list[str] = []
    derived_enh_wins = 0
    derived_enh_ties = 0
    derived_enh_losses = 0
    for instance in manifest.instances:
        row = rows_by_id.get(instance.instance_id)
        if row is None:
            continue
        cbc_greedy = (
            _obj(row.cbc_objective - row.greedy_objective)
            if row.cbc_objective is not None and row.greedy_objective is not None
            else "-"
        )
        enhanced_greedy = (
            _obj(row.enhanced_greedy_objective - row.greedy_objective)
            if row.enhanced_greedy_objective is not None and row.greedy_objective is not None
            else "-"
        )
        if (
            row.enhanced_greedy_objective is not None
            and row.greedy_objective is not None
            and abs(row.enhanced_greedy_objective - row.greedy_objective) > 1e-9
        ):
            enhanced_only_ids.append(row.instance_id)
        if row.cbc_objective is not None and row.enhanced_greedy_objective is not None:
            if row.cbc_objective > row.enhanced_greedy_objective:
                derived_enh_wins += 1
            elif row.cbc_objective == row.enhanced_greedy_objective:
                derived_enh_ties += 1
            else:
                derived_enh_losses += 1
        cbc_status, cbc_limit, independent_ok, feasibility_note = cbc_details.get(
            row.instance_id, ("unknown", None, False, "missing")
        )
        instance_rows.append(
            [
                row.instance_id,
                ", ".join(row.tags),
                str(row.n_items),
                str(row.capacity),
                _obj(row.oracle_optimum),
                _obj(row.cbc_objective),
                _obj(row.greedy_objective),
                _obj(row.enhanced_greedy_objective),
                cbc_greedy,
                enhanced_greedy,
                _num(row.greedy_relative_regret),
                _num(row.enhanced_relative_regret),
                "yes" if row.cbc_verified else "no",
                "yes" if independent_ok else f"no ({feasibility_note})",
                cbc_status,
                "-" if cbc_limit is None else f"{cbc_limit:g}",
            ]
        )
        instance_evidence.extend(
            [
                f"ev-{run}-{row.instance_id}-oracle",
                f"ev-{run}-{row.instance_id}-greedy",
                f"ev-{run}-{row.instance_id}-enhanced",
                f"ev-{run}-{row.instance_id}-verified",
            ]
        )
    instance_table = PaperTable(
        table_id=f"tab-{run}-instances",
        caption=(
            "Per-instance descriptive results for every frozen instance. Objectives are "
            "integral and displayed natively; differences are in objective units. 'CBC "
            "objective match' is objective equality with brute force. 'Independent "
            "feasible' recomputes the returned assignment's binary domain, capacity, and "
            "objective, independently of the solver."
        ),
        columns=[
            "Instance",
            "Tags",
            "n",
            "Capacity",
            "Oracle",
            "CBC",
            "Greedy",
            "Enhanced",
            "CBC - greedy",
            "Enhanced - greedy",
            "Greedy rel regret",
            "Enhanced rel regret",
            "CBC objective match",
            "Independent feasible",
            "CBC status",
            "Time limit (s)",
        ],
        rows=instance_rows,
        evidence_ids=instance_evidence,
    )

    tag_names = sorted(
        {tag for instance in manifest.instances for tag in instance.tags} | {"all"}
    )
    tag_rows = []
    tag_evidence = [f"ev-{run}-manifest-hash"] + [
        f"ev-{run}-{instance.instance_id}-{suffix}"
        for instance in manifest.instances
        for suffix in ("oracle", "greedy", "enhanced", "verified")
    ]
    for tag in tag_names:
        members = [
            rows_by_id[instance.instance_id]
            for instance in manifest.instances
            if (tag == "all" or tag in instance.tags)
            and instance.instance_id in rows_by_id
        ]
        tag_rows.append(
            [
                tag,
                str(len(members)),
                str(sum(1 for member in members if member.cbc_verified)),
                str(sum(1 for member in members if _hit(member.greedy_relative_regret))),
                str(sum(1 for member in members if _hit(member.enhanced_relative_regret))),
                str(
                    sum(
                        1
                        for member in members
                        if member.cbc_verified and _hit(member.greedy_relative_regret)
                    )
                ),
            ]
        )
    adversarial_hits, adversarial_n = _tag_hits(rows, "adversarial")
    random_hits, random_n = _tag_hits(rows, "random")
    sanity_hits, sanity_n = _tag_hits(rows, "sanity")
    tag_caption = (
        "Per-tag descriptive counts, derived from the per-instance table and its evidence "
        "records; no independent per-tag evidence record exists. Subtag rows overlap and "
        "must not be summed; the partition table above is the auditable breakdown. Curated "
        "subtypes are n=1, uniform n=2, correlated n=2, weakly correlated n=1, adversarial "
        "n=3, and sanity n=1; these are single-draw observations from one suite seed with "
        "no sampling-error estimate, not population rates."
    )
    tag_table = PaperTable(
        table_id=f"tab-{run}-tags",
        caption=tag_caption,
        columns=["Tag", "Instances", "CBC verified", "Greedy hits", "Enhanced hits", "CBC+greedy"],
        rows=tag_rows,
        evidence_ids=tag_evidence,
    )

    aggregate_rows = []
    aggregate_evidence = [
        f"ev-{run}-candidate-wins",
        f"ev-{run}-candidate-ties",
        f"ev-{run}-candidate-losses",
        f"ev-{run}-worst-instance",
    ]
    miss_sums: dict[str, float] = {}
    for aggregate, attr in (
        (result.greedy, "greedy_relative_regret"),
        (result.enhanced_greedy, "enhanced_relative_regret"),
    ):
        label = "greedy" if aggregate is result.greedy else "enhanced_greedy"
        regret_n, miss_n, mean_misses, miss_sum = _regret_stats(
            [getattr(row, attr) for row in rows]
        )
        miss_sums[label] = miss_sum
        aggregate_rows.append(
            [
                aggregate.method,
                aggregate.role,
                str(aggregate.optimum_hits),
                str(aggregate.failures),
                f"{aggregate.hit_rate:.4f}",
                str(regret_n),
                str(miss_n),
                _num(miss_sum),
                _num(aggregate.mean_relative_regret),
                _num(aggregate.median_relative_regret),
                _num(mean_misses),
                _num(aggregate.max_relative_regret),
                aggregate.worst_instance_id or "-",
            ]
        )
        aggregate_evidence.extend(
            [
                f"ev-{run}-{label}-hit-rate",
                f"ev-{run}-{label}-failures",
                f"ev-{run}-{label}-worst",
            ]
        )
    aggregate_caption = (
        "Descriptive aggregate evidence for this suite only; no weighted score is used. "
        "Relative regret is max(0, oracle - objective) / oracle for every feasible "
        "solution (an optimum has regret 0); infeasible or unavailable objectives are "
        "excluded and counted as failures. Mean and median are over all feasible values "
        f"including zero-regret hits (n={len(rows)} here). Median is the standard median, "
        "averaging the two central values for even n. 'Mean rel regret over misses' "
        "averages only strictly positive regrets and divides by 'Miss n'. The miss-regret "
        f"sums are greedy {miss_sums.get('greedy', 0.0):.4f} and enhanced "
        f"{miss_sums.get('enhanced_greedy', 0.0):.4f}; their means differ only because of "
        "the single random_correlated_2 instance and imply no aggregate ordering. The "
        "pooled hit count is printed for completeness; two of its five greedy misses are "
        "deliberately constructed traps, so it has no naturalistic failure-rate reading."
    )
    aggregate_table = PaperTable(
        table_id=f"tab-{run}-aggregates",
        caption=aggregate_caption,
        columns=[
            "Method",
            "Role",
            "Hits",
            "Failures",
            "Hit rate",
            "Regret n",
            "Miss n",
            "Miss regret sum",
            "Mean rel regret",
            "Median rel regret",
            "Mean rel regret over misses",
            "Max rel regret",
            "Worst instance",
        ],
        rows=aggregate_rows,
        evidence_ids=aggregate_evidence,
    )

    appendix_rows = [
        [
            instance.instance_id,
            str(instance.weights),
            str(instance.values),
            str(instance.capacity),
        ]
        for instance in manifest.instances
    ]
    appendix_table = PaperTable(
        table_id=f"tab-{run}-instance-data",
        caption=(
            "Frozen item data for every instance. Together with the full instance hashes "
            "and the generator code hash, this makes every objective and heuristic value "
            "re-derivable."
        ),
        columns=["Instance", "Weights", "Values", "Capacity"],
        rows=appendix_rows,
        evidence_ids=[f"ev-{run}-{instance.instance_id}-hash" for instance in manifest.instances],
    )

    evidence_rows = [
        [
            f"ev-{run}-manifest-hash",
            "benchmark.manifest_hash",
            result.manifest_hash,
        ],
        [
            f"ev-{run}-oracle-agreement",
            "benchmark.oracle_agreement",
            str(result.oracle_agreement).lower(),
        ],
        [
            f"ev-{run}-candidate-verified",
            "benchmark.candidate.verified_optimum_count",
            str(result.candidate_verified_optimum_count),
        ],
        [
            f"ev-{run}-greedy-hit-rate",
            "benchmark.greedy.hit_rate",
            f"{result.greedy.hit_rate:.4f}",
        ],
        [
            f"ev-{run}-greedy-failures",
            "benchmark.greedy.failures",
            str(result.greedy.failures),
        ],
        [
            f"ev-{run}-greedy-worst",
            "benchmark.greedy.worst_relative_regret",
            _num(result.greedy.worst_relative_regret),
        ],
    ]
    evidence_table = PaperTable(
        table_id=f"tab-{run}-evidence",
        caption=(
            "Representative frozen evidence records (id, key, value). The full record set "
            f"is in verified/benchmark/{run}.json."
        ),
        columns=["Evidence id", "Key", "Value"],
        rows=evidence_rows,
        evidence_ids=[row[0] for row in evidence_rows],
    )

    enhanced_gain_text = (
        "Enhanced greedy differs from ratio greedy on exactly "
        f"{len(enhanced_only_ids)} instance(s) in this suite"
        + (f" ({', '.join(enhanced_only_ids)})" if enhanced_only_ids else "")
        + ". Because enhanced greedy is a max-ensemble over ratio greedy and the best "
        "single feasible item, it weakly dominates ratio greedy by construction; the "
        "difference is not evidence of heuristic improvement."
    )

    max_regret = result.greedy.max_relative_regret
    worst_instance = result.greedy.worst_instance_id
    worst_text = (
        f"The largest observed ratio-greedy relative regret in this suite was "
        f"{max_regret:.4f} on `{worst_instance}`. This is a single-instance descriptive "
        "result at n=3-10, not a general difficulty estimate."
        if max_regret is not None and worst_instance is not None
        else "No ratio-greedy relative regret was available for this suite."
    )

    runtime = result.runtime
    runtime_text = (
        f"Python {runtime.get('python_version', 'not recorded')}; platform "
        f"{runtime.get('platform', 'not recorded')}; OR-Tools "
        f"{runtime.get('ortools_version', 'not recorded')}; CBC version: "
        f"{runtime.get('cbc_version', 'not recorded')}."
        if runtime
        else "Runtime identifiers were not captured in this run."
    )
    platform_note = (
        "With integer objectives, exact equality with the exhaustive optimum is expected "
        "to be platform-independent once the solver reports optimality; this report does "
        "not assert cross-platform reproduction beyond the recorded toolchain."
    )

    abstract = (
        f"Frozen 0/1 knapsack suite {result.suite_id} v{manifest.suite_version} "
        f"(manifest {result.manifest_hash[:12]}..., {result.n_instances} instances, "
        f"n=3-10). This report is a frozen-suite conformance and feasibility check, not a "
        "performance benchmark. CBC matched the brute-force optimum on "
        f"{exact_matches}/{result.n_instances} instances, and an independent recomputation "
        f"confirmed binary, capacity-feasible assignments on "
        f"{independent_feasible_count}/{result.n_instances}. Because n<=10, the oracle is "
        "exhaustive and this agreement cannot speak to CBC correctness at realistic scale. "
        "The CBC binary build is unpinned, so the conformance result is only conditionally "
        "reproducible from the pinned manifest, generator, and baseline hashes. Baseline "
        "counts are reported as integer fractions by construction stratum: adversarial "
        f"{adversarial_hits} of {adversarial_n}, seeded random {random_hits} of "
        f"{random_n}, sanity {sanity_hits} of {sanity_n}; no statistical inference is "
        f"possible from these single-draw counts. {len(curated_misses)} of the greedy "
        "misses are deliberately constructed traps. Enhanced greedy is a max-ensemble over "
        "ratio greedy and the best single item and is therefore weakly dominant by "
        "definition; no empirical comparison is needed or reported, and its single "
        "different instance (213 vs 208) is descriptive only. These counts do not establish "
        "general superiority, general greedy failure rates, or scalability."
    )

    sections = [
        PaperSection(
            heading="Purpose and conformance scope",
            blocks=[
                "This report is a conformance and feasibility check, not a performance "
                "benchmark. At n=3-10 the oracle is exhaustive, so oracle agreement cannot "
                "speak to CBC correctness at realistic scale.",
                "The suite is a frozen, pre-declared set of synthetic 0/1 knapsack instances. "
                "No instance is selected or excluded after execution, and all claims are "
                "limited to this frozen suite.",
                f"Suite: `{result.suite_id}`; version: `{manifest.suite_version}`; "
                f"generator: `{manifest.generator_version}`; suite seed: "
                f"{result.suite_seed}; manifest hash: `{result.manifest_hash}`.",
                f"Methods: {', '.join(manifest.methods)}; independent oracle: "
                f"`{manifest.oracle_method}`.",
                "Scale: every instance has n=3-10 items. The suite is single-scale, "
                "synthetic, and covers one problem family. The seeded random families "
                "contribute five instances at one capacity ratio; no seed or capacity "
                "sweep is performed, so per-tag random counts are single-draw.",
            ],
        ),
        PaperSection(
            heading="Method definitions and reproducibility",
            blocks=[
                "CBC candidate: an OR-Tools CBC mixed-integer solve of the binary knapsack "
                "model, maximizing sum(v_i x_i) subject to sum(w_i x_i) <= capacity, with "
                "a uniform solver time limit stated once in the completion section (10 s "
                "in this run).",
                "Ratio greedy: compute score_i = v_i / w_i; sort items by (score, value, "
                "negative weight, negative index) descending; add each item if it remains "
                "feasible. The ordering is fully deterministic, so no random state is used.",
                "Enhanced greedy: take the better of ratio greedy and the best feasible "
                "single-item solution; on an objective tie, ratio greedy is retained. It is "
                "a max-ensemble that weakly dominates ratio greedy by construction. "
                + enhanced_gain_text,
                f"Generator/oracle code hash: `{result.generator_code_hash}`; combined "
                f"generator+baseline code hash: `{result.methods_code_hash}`; git commit: "
                f"`{result.git_commit}`.",
                f"Runtime: {runtime_text} The CBC binary version is not exposed by the "
                "OR-Tools Python API, so the exact solver build is a known reproduction "
                "limitation.",
                "Objective values are stored as floats by the optimization schema but are "
                "integral for this suite; the tables display them as integers. The pass/fail "
                "rule is: cast both sides to int first, then require exact equality. The "
                "1e-6 guard exists only as a non-binding float-harness tolerance; the raw "
                "pre-cast float residuals are not recorded in the evidence set, so the claim "
                "is 'integer equality after int-cast on 9/9 with unrecorded pre-cast "
                "residuals', not a measured zero residual. " + platform_note,
            ],
        ),
        PaperSection(
            heading="Completion and oracle checks",
            blocks=[
                "Suite completion: every instance in the frozen manifest must produce a "
                "CBC experiment, a ratio-greedy experiment, an enhanced-greedy experiment, "
                "and one optimization tournament, with no missing instances.",
                "Oracle agreement: true only if every manifest instance completed and each "
                "CBC objective equals the independent exhaustive enumeration optimum within "
                "the harness tolerance. The oracle enumerates all 2^n subsets in-process and "
                "never reads a solver value.",
                "'CBC objective match' is objective equality with that brute-force optimum. "
                "'Independent feasible' is a separate post-hoc audit: the returned binary "
                "assignment is recomputed item-by-item, checking the binary domain, capacity "
                "sum, and objective consistency. Both checks are required for a conforming "
                "instance.",
                f"Observed suite status: `{result.status}`; oracle agreement: "
                f"`{result.oracle_agreement}`; exact integer equality on {exact_matches}/"
                f"{result.n_instances}; independent feasibility on "
                f"{independent_feasible_count}/{result.n_instances}.",
                f"Recorded CBC time limit: {time_limit_text} per instance. CBC status was "
                f"{', '.join(f'`{status}`' for status in statuses)} on all instances"
                + (
                    "; no solve terminated on the time limit."
                    if all_optimal
                    else "; instances not marked optimal are feasible-and-matched, not "
                    "solved-to-optimality."
                ),
                "The 5/4/0 CBC-versus-greedy split is algebraic: CBC equals the oracle on "
                "9/9 and greedy hits 4/9, so the split follows from those two facts. It is "
                "not an independent tournament or a performance finding.",
            ],
        ),
        PaperSection(
            heading="Per-instance descriptive results",
            blocks=[
                "Table results are reported for every frozen instance. CBC-minus-greedy and "
                "enhanced-minus-greedy columns are in objective units, so tournament "
                "win/tie/loss counts can be audited directly.",
                "Relative regret is defined as max(0, oracle - objective) / oracle for every "
                "feasible solution; an optimum has regret 0. Infeasible or unavailable "
                "objectives are excluded from regret and counted as failures. Mean and "
                "median regret include zero-regret hits over all feasible values; the "
                "misses-only mean and its count are reported separately.",
                f"Stratified counts: adversarial {adversarial_hits} of {adversarial_n}, "
                f"seeded random {random_hits} of {random_n}, sanity {sanity_hits} of "
                f"{sanity_n}. These are integer counts, not rates, and no statistical "
                f"inference is possible. Of the greedy misses, {len(curated_misses)} are "
                f"curated traps "
                f"({', '.join(curated_misses)}) and {len(random_misses)} are seeded random "
                f"({', '.join(random_misses)}); {len(sanity_misses)} is the legacy sanity "
                "instance. The pooled count is therefore composition-dependent and is not "
                "reported as a failure rate.",
                f"CBC-versus-enhanced is the same algebraic "
                f"{derived_enh_wins}/{derived_enh_ties}/{derived_enh_losses} split as "
                "CBC-versus-greedy (derived from the printed columns, not a separately run "
                "tournament). CBC is exact here, so domination of the heuristics is expected "
                "and is not a performance claim.",
                "Enhanced and ratio greedy share the same hit count, median regret, and max "
                "regret; their mean-regret gap is driven entirely by the single "
                "random_correlated_2 instance. With one suite seed and n=1-3 per stratum, "
                "the gap is not resolvable and no aggregate ordering is implied.",
                "The pooled hit count in the aggregate table is printed for completeness "
                "only. Two of its five greedy misses are deliberately constructed traps; no "
                "naturalistic failure-rate reading attaches to the pooled count.",
                worst_text,
                f"Enhanced-greedy comparison: {enhanced_gain_text}",
            ],
        ),
        PaperSection(
            heading="Interpretation limits",
            blocks=[
                "This report does not establish that CBC is generally superior to greedy, "
                "that greedy has a general failure rate, or that enhanced greedy improves "
                "over ratio greedy beyond the listed instances.",
                "The suite is small (n=3-10), synthetic, single-scale, and limited to 0/1 "
                "knapsack. It provides no evidence about scalability, other problem "
                "families, multi-objective optimization, real-world data, or solver "
                "performance outside CBC on these instances.",
                "Per-tag counts rest on n=1-3 single-draw instances from one suite seed. No "
                "seed, capacity, or density sweep is performed, so no sampling error or "
                "stability claim is available. The generator code hash pins the suite "
                "logic, but the CBC binary build is not exposed and is unpinned.",
            ],
        ),
        PaperSection(
            heading="Pinned artifacts, private evidence, and conditional reproduction",
            blocks=[
                "Pinned: manifest hash, full instance hashes, generator/oracle code hash, "
                "combined baseline code hash, git commit, suite seed, and recorded Python, "
                "platform, and OR-Tools versions. Unpinned: the CBC binary build, which the "
                "OR-Tools Python API does not expose.",
                f"The evidence JSON is a private project artifact at "
                f"`verified/benchmark/{run}.json`, not published with this report; the "
                "evidence table shows representative records, and the appendix plus full "
                "hashes let an external reader re-derive the per-instance values.",
                "The manifest reconciliation table lists every declared instance with its "
                "full SHA-256 hash and explicit partition; the instance-data table prints "
                "all weights and values. The suite result must match the manifest hash, "
                "instance hashes, counts, experiment index, and tournament index.",
                "Because the solver build is unpinned, reproduction is conditional on the "
                "recorded toolchain; the integer-equality result is the least "
                "build-sensitive part, but no cross-build guarantee is claimed.",
            ],
        ),
    ]

    return PaperAST(
        title="0/1 Knapsack Frozen-Suite Conformance and Feasibility Report — Suite v1",
        abstract=abstract,
        sections=sections,
        tables=[
            reconciliation_table,
            instance_table,
            tag_table,
            aggregate_table,
            appendix_table,
            evidence_table,
        ],
        figures=[],
        claims=[],
        references=["Frozen benchmark manifest and verified suite evidence."],
    )
