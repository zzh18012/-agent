from mmos.core.project import ProjectPaths
from mmos.experiments.knapsack_benchmark import load_manifest
from mmos.experiments.knapsack_suite import load_benchmark_result
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable


def build_benchmark_paper(
    paths: ProjectPaths, benchmark_run_id: str, suite_dir: str | None = None
) -> PaperAST:
    artifact_dir = paths.experiments_dir / "benchmarks" / benchmark_run_id
    result = load_benchmark_result(artifact_dir / "benchmark_result.json")
    manifest = (
        load_manifest(artifact_dir) if (artifact_dir / "benchmark_manifest.json").exists() else None
    )
    if manifest is None:
        from mmos.schemas.benchmark import BenchmarkManifest

        manifest = BenchmarkManifest.model_validate_json(
            (artifact_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
        )

    instance_rows = []
    evidence_ids = []
    for instance in result.per_instance:
        instance_rows.append(
            [
                instance.instance_id,
                ", ".join(instance.tags),
                str(instance.n_items),
                str(instance.capacity),
                f"{instance.oracle_optimum:.6g}",
                f"{instance.cbc_objective:.6g}" if instance.cbc_objective is not None else "-",
                f"{instance.greedy_objective:.6g}"
                if instance.greedy_objective is not None
                else "-",
                f"{instance.enhanced_greedy_objective:.6g}"
                if instance.enhanced_greedy_objective is not None
                else "-",
                f"{instance.greedy_relative_regret:.6g}"
                if instance.greedy_relative_regret is not None
                else "-",
                f"{instance.enhanced_relative_regret:.6g}"
                if instance.enhanced_relative_regret is not None
                else "-",
                "yes" if instance.cbc_verified else "no",
            ]
        )
    instance_table = PaperTable(
        table_id=f"tab-{benchmark_run_id}-instances",
        caption="All frozen benchmark instances and method objectives; no instance is excluded.",
        columns=[
            "Instance",
            "Tags",
            "n",
            "Capacity",
            "Oracle",
            "CBC",
            "Greedy",
            "Enhanced",
            "Greedy rel regret",
            "Enhanced rel regret",
            "CBC verified",
        ],
        rows=instance_rows,
        evidence_ids=evidence_ids,
    )

    aggregate_rows = []
    for aggregate in (result.greedy, result.enhanced_greedy):
        aggregate_rows.append(
            [
                aggregate.method,
                aggregate.role,
                str(aggregate.optimum_hits),
                str(aggregate.failures),
                f"{aggregate.hit_rate:.4f}",
                f"{aggregate.mean_relative_regret:.6g}"
                if aggregate.mean_relative_regret is not None
                else "-",
                f"{aggregate.median_relative_regret:.6g}"
                if aggregate.median_relative_regret is not None
                else "-",
                f"{aggregate.max_relative_regret:.6g}"
                if aggregate.max_relative_regret is not None
                else "-",
                aggregate.worst_instance_id or "-",
            ]
        )
    aggregate_table = PaperTable(
        table_id=f"tab-{benchmark_run_id}-aggregates",
        caption="Descriptive aggregate evidence; no weighted benchmark score is used.",
        columns=[
            "Method",
            "Role",
            "Hits",
            "Failures",
            "Hit rate",
            "Mean rel regret",
            "Median rel regret",
            "Max rel regret",
            "Worst instance",
        ],
        rows=aggregate_rows,
    )

    adversarial = [instance for instance in result.per_instance if "adversarial" in instance.tags]
    adversarial_lines = []
    for instance in adversarial:
        ratio_regret = (
            f"{instance.greedy_relative_regret:.4g}"
            if instance.greedy_relative_regret is not None
            else "unavailable"
        )
        enhanced_regret = (
            f"{instance.enhanced_relative_regret:.4g}"
            if instance.enhanced_relative_regret is not None
            else "unavailable"
        )
        adversarial_lines.append(
            f"- {instance.instance_id}: oracle {instance.oracle_optimum:.6g}, "
            f"greedy regret {ratio_regret}, enhanced regret {enhanced_regret}."
        )

    abstract = (
        f"This report covers frozen benchmark suite {result.suite_id} "
        f"(manifest {result.manifest_hash[:12]}…), {result.n_instances} instances. "
        f"CBC independently matched the oracle on "
        f"{result.candidate_verified_optimum_count}/{result.n_instances} instances; "
        f"candidate wins/ties/losses = {result.candidate_win_count}/"
        f"{result.candidate_tie_count}/{result.candidate_loss_count}. "
        f"Greedy hit rate {result.greedy.hit_rate:.3f}; enhanced greedy hit rate "
        f"{result.enhanced_greedy.hit_rate:.3f}. "
        f"Worst greedy instance: {result.worst_case_instance or 'none'}."
    )

    sections = [
        PaperSection(
            heading="Benchmark purpose and frozen manifest",
            blocks=[
                "The suite is a frozen, pre-declared set of 0/1 knapsack instances. "
                "Charts and claims are limited to this suite.",
                f"Suite: {result.suite_id}; version: {manifest.suite_version}; "
                f"manifest hash: `{result.manifest_hash}`; suite seed: {result.suite_seed}; "
                f"generator version: {manifest.generator_version}.",
                "Methods: CBC MILP candidate; ratio-greedy sanity baseline; "
                "enhanced greedy heuristic baseline; independent exhaustive oracle.",
            ],
        ),
        PaperSection(
            heading="Aggregate results",
            blocks=[
                "All frozen instances are included in the tables. Aggregates report "
                "mean, median, maximum, hit rate, failures, and worst instance rather "
                "than a weighted score.",
                f"Oracle agreement: {result.oracle_agreement}. Suite status: {result.status}.",
            ],
        ),
        PaperSection(
            heading="Adversarial findings",
            blocks=adversarial_lines or ["No adversarial instances were present."],
        ),
        PaperSection(
            heading="Limitations",
            blocks=[
                "This is a small synthetic 0/1 knapsack suite, not evidence about MILP, "
                "heuristics, or real-world optimization in general.",
                "The greedy and enhanced greedy baselines are transparent but limited.",
                "Instance sizes are chosen so the exhaustive oracle is feasible.",
                "No scale sensitivity, multi-objective, or non-knapsack evidence is included.",
            ],
        ),
    ]

    return PaperAST(
        title="0/1 Knapsack Benchmark Suite v1 — Benchmark Report",
        abstract=abstract,
        sections=sections,
        tables=[instance_table, aggregate_table],
        figures=[],
        claims=[],
        references=["Frozen benchmark manifest and verified suite evidence."],
    )
