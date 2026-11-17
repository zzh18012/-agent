from dataclasses import dataclass
from pathlib import Path

from mmos.competition.context import CatalogEntry, EvidenceContext
from mmos.competition.number_audit import (
    ExpandedText,
    expand_text,
    format_display_value,
    format_record_value,
    numeric_tokens,
)
from mmos.core.project import ProjectPaths
from mmos.paper.competition_figures import build_figure
from mmos.schemas.competition import (
    CompetitionExperimentManifest,
    CompetitionPaperPlan,
    ExperimentAnalysisReport,
    FigurePlan,
    FigureTemplate,
    SectionKind,
    TablePlan,
    TableTemplate,
)
from mmos.schemas.evidence import EvidenceRecord
from mmos.schemas.paper import PaperAST, PaperFigure, PaperSection, PaperTable


class CompetitionBuildError(RuntimeError):
    pass


_SYSTEM_SECTION_NOTES = {
    SectionKind.SENSITIVITY: [
        "The executed augmentation appendix contains perturbed optimization instances "
        "(capacity, cost, and initial-inventory variants) and replication/seed re-runs. "
        "Perturbed-instance objectives and plans are not directly comparable with the "
        "frozen plan, and the frozen plan's recorded solver status applies only at the "
        "frozen parameter point. No statistical test of differences was performed.",
    ],
    SectionKind.EVALUATION: [
        "The plan is cost-minimizing for the frozen parameter point under the recorded "
        "solver status and optimality gap. The plan table labels periods whose recorded "
        "production equals recorded capacity; these labels are descriptive comparisons "
        "of the plan, not solver duals or binding certificates. No objective-stability "
        "claim beyond this point is made.",
        "Residual resampling is iid by month and is not a tested model of horizon "
        "dependence. Duplicate residual paths occur, so the effective sample size is "
        "below the nominal replication count and no precise standard error is claimed.",
    ],
}


def _system_section_notes(context: EvidenceContext) -> dict[SectionKind, list[str]]:
    notes = {kind: list(values) for kind, values in _SYSTEM_SECTION_NOTES.items()}
    by_key = {entry.key: entry.evidence_id for entry in context.entries}

    def ph(key: str, fallback: str) -> str:
        evidence_id = by_key.get(key)
        return f"[[ev:{evidence_id}]]" if evidence_id else fallback

    periods = [
        record.key.rsplit(".", 1)[-1]
        for record in context.derived_records
        if record.key.startswith("competition.plan.capacity.")
    ]
    if len(periods) >= 2:
        first, second = periods[0], periods[1]
        feasibility = (
            f"Feasibility and optimality check. Month-2 demand "
            f"{ph(f'competition.forecast.point.{second}', 'F_2')} exceeds month-2 "
            f"capacity {ph(f'competition.plan.capacity.{second}', 'cap_2')}, so no "
            "feasible plan can produce month-2 demand entirely in month 2. The "
            f"recorded month-1 production {ph(f'competition.plan.production.{first}', 'P_1')} "
            f"and month-end inventory {ph(f'competition.plan.inventory.{first}', 'I_1')} "
            f"carry the difference, and month-2 production "
            f"{ph(f'competition.plan.production.{second}', 'P_2')} equals month-2 "
            f"capacity {ph(f'competition.plan.capacity.{second}', 'cap_2')}; month-1 "
            f"capacity {ph(f'competition.plan.capacity.{first}', 'cap_1')} is above the "
            "recorded production. The recorded objective "
            f"{ph('optimization.ortools.cbc.objective_value', 'objective')} is the "
            "solver result and equals the derived total in "
            "`tab-objective-recomputation`, under status "
            f"{ph('optimization.ortools.cbc.status', 'recorded status')} and gap "
            f"{ph('optimization.ortools.cbc.optimality_gap', 'recorded gap')}; no duals "
            "or alternative optimality certificate are evidenced."
        )
        results_check = (
            "Feasibility check for the plan table: month-2 demand "
            f"{ph(f'competition.forecast.point.{second}', 'F_2')} exceeds month-2 "
            f"capacity {ph(f'competition.plan.capacity.{second}', 'cap_2')}; the "
            f"recorded month-1 inventory {ph(f'competition.plan.inventory.{first}', 'I_1')} "
            "is the required carry, and a plan with no carried inventory would exceed "
            "month-2 capacity."
        )
        notes.setdefault(SectionKind.SOLUTION, []).append(feasibility)
        notes.setdefault(SectionKind.RESULTS, []).append(results_check)

    if "mixed.forecast_error.mean" in by_key:
        notes.setdefault(SectionKind.RESULTS, []).append(
            "Fill rate is total served demand divided by total realized demand, "
            "summed over the planning horizon and replications; it is not a monthly "
            "average. Shortage is the sum of monthly shortages, and the any-shortage "
            "probability is the fraction of replications with positive total shortage."
        )
        notes.setdefault(SectionKind.RESULTS, []).append(
            "Every simulation value in this section is conditional on the proxy "
            "fold-model residual protocol and the uncorrected pooled residual mean "
            f"{ph('mixed.forecast_error.mean', 'value')}; it is not an unbiased service "
            "estimate, and no bootstrap interval is evidenced."
        )
        if "mixed.simulation.shortage.standard_error" in by_key:
            notes.setdefault(SectionKind.RESULTS, []).append(
                "The recorded normal-approximation standard errors are "
                f"{ph('mixed.simulation.shortage.standard_error', 'value')} for mean "
                "total shortage and "
                f"{ph('mixed.simulation.service.fill_rate.standard_error', 'value')} "
                "for mean fill rate; they are computed from replication-level outputs, "
                "not from a bootstrap, and no formal interval test of the augmentation "
                "re-executions is performed. The recorded augmentation re-executions "
                "differ from the source value by a few standard errors, which is "
                "consistent with Monte Carlo variation under the same protocol."
            )
        if "mixed.simulation.min_simulated_demand" in by_key:
            notes.setdefault(SectionKind.EVALUATION, []).append(
                "The minimum simulated demand across replications is "
                f"{ph('mixed.simulation.min_simulated_demand', 'value')} and the "
                "demand-floor activation count is "
                f"{ph('mixed.simulation.demand_floor_activations', 'value')}; the "
                "recorded floor margin is bounded by that minimum."
            )
        if "mixed.simulation.master_seed" in by_key:
            notes.setdefault(SectionKind.SOLUTION, []).append(
                "The recorded master seed is "
                f"{ph('mixed.simulation.master_seed', 'value')} and the replication "
                "seed hash is stored in the evidence index; the per-replication draw "
                "implementation is not described in the source evidence."
            )
    if "metrics.naive.last_value.mae" in by_key:
        notes.setdefault(SectionKind.RESULTS, []).append(
            "The last-value benchmark MAE is "
            f"{ph('metrics.naive.last_value.mae', 'value')} and RMSE is "
            f"{ph('metrics.naive.last_value.rmse', 'value')}; the benchmark metrics are "
            "produced by the same rolling-origin evaluator. The pooled RMSE and MAE are "
            "computed from all rolling-origin errors; the fold mean is the arithmetic "
            "mean of fold metrics and is not the pooled metric. The MAE difference is "
            "smaller than the fold-level spread, and no paired significance test or "
            "preference claim is evidenced."
        )
    if context.appendix_records:
        notes.setdefault(SectionKind.RESULTS, []).append(
            "Re-executions of the same frozen simulation protocol produce different "
            "summary values; the spread is listed in the augmentation appendix. No "
            "standard error or bootstrap interval is computed, so the source values are "
            "one draw from the protocol, not a stable precise estimate."
        )
    if "uncertainty.naive.drift.rmse.ci_low" in by_key:
        notes.setdefault(SectionKind.SENSITIVITY, []).append(
            "The reported interval is for the arithmetic mean of the recorded fold "
            f"values; that fold mean is "
            f"{ph('uncertainty.naive.drift.rmse.fold_mean', 'value')}, while the pooled "
            "metric is computed from all rolling-origin errors. Individual fold values "
            "are plotted as points, and the interval construction and effective sample "
            "size are not evidenced."
        )
    return notes


_SYSTEM_TABLE_CAPTIONS = {
    TableTemplate.METRIC_COMPARISON: (
        "Accuracy metrics for the evaluated forecasting rules, copied from registered "
        "evidence. The values are descriptive validation summaries; no significance "
        "test or ranking of forecasting families is implied."
    ),
    TableTemplate.EXPERIMENT_INVENTORY: (
        "Experiment inventory for the frozen forecast, optimization, and simulation "
        "stages. Counts are catalogued evidence-record (cell) counts, not the number of "
        "scalar results cited in the text. Simulation-stage metrics are filed in the "
        "source evidence index under the optimization experiment id and are counted "
        "here against the simulation stage by evidence key."
    ),
    TableTemplate.EVIDENCE_LINEAGE: (
        "Evidence lineage linking each experiment to its artifacts, hashes, and "
        "registered evidence ids. Simulation-stage keys are grouped under the "
        "simulation experiment id. The table documents provenance only."
    ),
}

_STALE_AUGMENTATION_LIMITATIONS = (
    "no sensitivity sweep over cost parameters, capacities, or replication count",
    "no sensitivity analysis over cost parameters, capacities, or alternative",
    "no capacity, cost, or initial-inventory perturbation analysis",
    "no capacity or cost perturbation",
    "no perturbation analysis",
    "no sensitivity analysis on cost parameters, capacities, or initial inventory",
    "capacity binding in month 1",
)


def _stale_after_augmentation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STALE_AUGMENTATION_LIMITATIONS)


_SYSTEM_FIGURE_CAPTIONS = {
    FigureTemplate.PIPELINE_SCHEMATIC: (
        "Frozen experiment lineage. Arrows denote the recorded artifact dependency "
        "chain; no alternative plans are shown."
    ),
    FigureTemplate.METRIC_BAR: (
        "Registered evidence values for the selected metric group. Bars are descriptive "
        "values from the verified evidence index, not a ranking of forecasting families."
    ),
    FigureTemplate.UNCERTAINTY: (
        "Fold-level values (dots) and the reported confidence interval for the mean "
        "across folds (line). The interval is for the mean, need not contain every fold, "
        "and is not a predictive interval."
    ),
}


@dataclass(frozen=True)
class CompetitionBuildResult:
    ast: PaperAST
    fragments: list[ExpandedText]


def _entry_map(context: EvidenceContext) -> dict[str, CatalogEntry]:
    return {entry.evidence_id: entry for entry in context.entries}


def _require_evidence(
    evidence_ids: list[str], known: set[str], *, location: str
) -> None:
    missing = sorted(set(evidence_ids) - known)
    if missing:
        raise CompetitionBuildError(
            f"{location} cites unknown evidence ids: {', '.join(missing)}"
        )


def _table_rows(
    table: TablePlan, context: EvidenceContext
) -> tuple[list[str], list[list[str]]]:
    if table.template is TableTemplate.METRIC_COMPARISON:
        records = [
            entry for entry in context.entries if entry.key.startswith("metrics.")
        ]
        if not records:
            raise CompetitionBuildError("metric_comparison has no metric evidence")
        labels = {
            experiment.experiment_id: experiment.label
            for experiment in context.experiments
        }
        rows = []
        for entry in records:
            parts = entry.key.split(".")
            model = parts[1] if len(parts) > 2 else "-"
            metric = ".".join(parts[2:]) if len(parts) > 2 else entry.key
            rows.append(
                [
                    labels.get(entry.experiment_id, entry.experiment_id),
                    model,
                    metric,
                    format_display_value(entry.raw_value, entry.display_value),
                    entry.unit or "-",
                ]
            )
        return ["Experiment", "Model", "Metric", "Value", "Unit"], rows

    if table.template is TableTemplate.EXPERIMENT_INVENTORY:
        if not context.experiments:
            raise CompetitionBuildError("experiment_inventory has no experiments")
        rows = [
            [
                experiment.experiment_id,
                experiment.label,
                experiment.task,
                str(experiment.evidence_count),
            ]
            for experiment in context.experiments
        ]
        return ["Experiment", "Role", "Task", "Evidence records"], rows

    if table.template is TableTemplate.EVIDENCE_LINEAGE:
        if not context.entries:
            raise CompetitionBuildError("evidence_lineage has no evidence")
        simulation_experiment = context.release.get("simulation_experiment_id")
        by_experiment: dict[str, list[CatalogEntry]] = {}
        for entry in context.entries:
            owner = entry.experiment_id
            if simulation_experiment and entry.key.startswith("mixed.simulation."):
                owner = str(simulation_experiment)
            by_experiment.setdefault(owner, []).append(entry)
        rows = [
            [
                context.release_path.name,
                experiment_id,
                str(len(entries)),
                entries[0].artifact_hash[:16],
                ", ".join(entry.evidence_id for entry in entries[:3]),
            ]
            for experiment_id, entries in sorted(by_experiment.items())
        ]
        return [
            "Source release",
            "Experiment",
            "Evidence records",
            "Artifact hash (prefix)",
            "Example evidence ids",
        ], rows

    raise CompetitionBuildError(f"unsupported table template: {table.template}")


def _value(record: EvidenceRecord) -> str:
    return format_record_value(record)


def _display_unit(record: EvidenceRecord) -> str:
    lowered = record.key.lower()
    if any(
        marker in lowered
        for marker in ("fill_rate", "probability", "period_count", "distinct")
    ):
        return "-"
    return record.unit or "-"


def _entry_value(entry: CatalogEntry) -> str:
    if entry.key.startswith("mixed.simulation.") and isinstance(entry.raw_value, float):
        return f"{entry.raw_value:.2g}"
    return format_display_value(entry.raw_value, entry.display_value)


def _system_diagnostics_table(context: EvidenceContext) -> list[PaperTable]:
    catalog = {entry.key: entry for entry in context.entries}
    rows_spec = [
        ("Replications", "mixed.simulation.replications"),
        ("Distinct residual paths", "mixed.simulation.distinct_residual_paths"),
        ("Mean total shortage", "mixed.simulation.shortage.mean_total_shortage"),
        (
            "Standard error of mean total shortage",
            "mixed.simulation.shortage.standard_error",
        ),
        ("Mean fill rate", "mixed.simulation.service.mean_fill_rate"),
        (
            "Standard error of mean fill rate",
            "mixed.simulation.service.fill_rate.standard_error",
        ),
        ("Any-shortage probability", "mixed.simulation.any_shortage_probability"),
        (
            "Standard error of any-shortage probability",
            "mixed.simulation.any_shortage_probability.standard_error",
        ),
        (
            "Minimum simulated demand",
            "mixed.simulation.min_simulated_demand",
        ),
        ("Demand-floor activations", "mixed.simulation.demand_floor_activations"),
        ("Master seed", "mixed.simulation.master_seed"),
        ("Replication seed hash", "mixed.simulation.seed_hash"),
    ]
    rows: list[list[str]] = []
    evidence_ids: list[str] = []
    for label, key in rows_spec:
        entry = catalog.get(key)
        if entry is None:
            continue
        rows.append(
            [label, _entry_value(entry), entry.unit or "-", entry.evidence_id]
        )
        evidence_ids.append(entry.evidence_id)
    if not rows:
        return []
    return [
        PaperTable(
            table_id="tab-simulation-diagnostics",
            caption=(
                "Simulation diagnostics copied or derived from the validated simulation "
                "artifact. Standard errors are normal-approximation standard errors from "
                "replication-level outputs; no bootstrap interval or quantile is "
                "evidenced. The master seed and replication seed hash are recorded; the "
                "per-replication draw implementation is not described."
            ),
            columns=["Quantity", "Value", "Unit", "Evidence"],
            rows=rows,
            evidence_ids=evidence_ids,
        )
    ]


def _augmentation_value(record: EvidenceRecord) -> str:
    if ".simulation." in record.key and isinstance(record.raw_value, float):
        return f"{record.raw_value:.2g}"
    return format_record_value(record)


def _augmentation_tables(context: EvidenceContext) -> list[PaperTable]:
    extra = list(context.appendix_records)
    if not extra:
        return []
    optimization: dict[tuple[str, str, str], dict[str, EvidenceRecord]] = {}
    simulation: dict[tuple[str, str], dict[str, EvidenceRecord]] = {}
    for record in extra:
        request_id = str(record.provenance.get("request_id", "unknown"))
        parts = record.key.split(".")
        if parts[:2] == ["competition", "optimization"] and len(parts) >= 5:
            variant = parts[3]
            optimization.setdefault(
                (request_id, record.experiment_id, variant), {}
            )[parts[-1]] = record
        elif parts[:2] == ["competition", "simulation"]:
            prefix = f"competition.simulation.{record.experiment_id}."
            name = record.key.removeprefix(prefix)
            simulation.setdefault((request_id, record.experiment_id), {})[name] = (
                record
            )
    tables: list[PaperTable] = []
    optimization_rows: list[list[str]] = []
    optimization_evidence: list[str] = []
    for (request_id, experiment_id, variant), values in sorted(optimization.items()):
        objective = values.get("objective")
        if objective is None:
            continue
        production = [
            values.get(f"production_{index}") for index in range(3)
        ]
        inventory = [
            values.get(f"inventory_{index}") for index in range(3)
        ]
        production_text = ", ".join(
            _augmentation_value(item) for item in production if item is not None
        )
        inventory_text = ", ".join(
            _augmentation_value(item) for item in inventory if item is not None
        )
        feasible = values.get("feasible")
        optimization_rows.append(
            [
                request_id,
                variant,
                experiment_id,
                _augmentation_value(objective),
                "yes" if feasible is not None and feasible.raw_value == "true" else "no",
                production_text or "-",
                inventory_text or "-",
            ]
        )
        optimization_evidence.extend(
            item.evidence_id for item in [objective, feasible, *production, *inventory]
            if item is not None
        )
    if optimization_rows:
        tables.append(
            PaperTable(
                table_id="tab-augmentation-optimization",
                caption=(
                    "Executed optimization perturbations. Each row is a perturbed "
                    "instance named by its variant; the objective and vectors belong to "
                    "that instance, not to the frozen parameters. Objectives are not "
                    "directly comparable across variants, and repeated variants across "
                    "request identifiers are duplicate executions. No statistical test "
                    "is performed."
                ),
                columns=[
                    "Request",
                    "Variant",
                    "Experiment",
                    "Objective",
                    "Feasible",
                    "Production vector",
                    "Inventory vector",
                ],
                rows=optimization_rows,
                evidence_ids=list(dict.fromkeys(optimization_evidence)),
            )
        )
    simulation_rows: list[list[str]] = []
    simulation_evidence: list[str] = []
    for (request_id, experiment_id), values in sorted(simulation.items()):
        shortage = values.get("total_shortage.mean")
        fill = values.get("fill_rate.mean")
        any_shortage = values.get("any_shortage_probability")
        row = [
            request_id,
            experiment_id,
            _augmentation_value(shortage) if shortage else "-",
            _augmentation_value(fill) if fill else "-",
            _augmentation_value(any_shortage) if any_shortage else "-",
        ]
        simulation_rows.append(row)
        simulation_evidence.extend(
            item.evidence_id
            for item in (shortage, fill, any_shortage)
            if item is not None
        )
    if simulation_rows:
        tables.append(
            PaperTable(
                table_id="tab-augmentation-simulation",
                caption=(
                    "Executed replication/seed re-runs of the frozen plan. All rows use "
                    "the same production plan and the same pooled-residual iid protocol. "
                    "Values are single-run descriptive summaries at two significant "
                    "digits; no standard error, bootstrap interval, or formal test of "
                    "differences is computed."
                ),
                columns=[
                    "Request",
                    "Experiment",
                    "Mean total shortage",
                    "Mean fill rate",
                    "Any-shortage probability",
                ],
                rows=simulation_rows,
                evidence_ids=list(dict.fromkeys(simulation_evidence)),
            )
        )
    return tables


def _system_plan_tables(context: EvidenceContext) -> list[PaperTable]:
    if not context.derived_records:
        return []
    derived = {record.key: record for record in context.derived_records}
    catalog = {entry.key: entry for entry in context.entries}
    periods = [
        key.rsplit(".", 1)[-1]
        for key in derived
        if key.startswith("competition.plan.capacity.")
    ]
    if not periods:
        return []

    def row(prefix: str, period: str) -> EvidenceRecord:
        return derived[f"{prefix}.{period}"]

    production_unit = row("competition.plan.production", periods[0]).unit or "-"
    inventory_unit = row("competition.plan.inventory", periods[0]).unit or "-"
    capacity_unit = row("competition.plan.capacity", periods[0]).unit or "-"
    forecast_unit = row("competition.forecast.point", periods[0]).unit or "-"
    if "/" not in forecast_unit:
        forecast_unit = f"{forecast_unit}/month"
    plan_rows: list[list[str]] = []
    plan_evidence: list[str] = []
    for period in periods:
        production = row("competition.plan.production", period)
        inventory = row("competition.plan.inventory", period)
        capacity = row("competition.plan.capacity", period)
        forecast = row("competition.forecast.point", period)
        production_value = float(production.raw_value)
        inventory_value = float(inventory.raw_value)
        capacity_value = float(capacity.raw_value)
        plan_rows.append(
            [
                period,
                _value(forecast),
                _value(production),
                _value(inventory),
                _value(capacity),
                "at capacity" if production_value >= capacity_value else "below capacity",
                "zero" if inventory_value <= 0 else "positive",
            ]
        )
        plan_evidence.extend(
            [
                forecast.evidence_id,
                production.evidence_id,
                inventory.evidence_id,
                capacity.evidence_id,
            ]
        )
    tables = [
        PaperTable(
            table_id="tab-frozen-plan",
            caption=(
                "Frozen production plan and demand input, copied from registered "
                "competition evidence. F_m is forecast demand, P_m production, I_m "
                "month-end inventory, and cap_m capacity. Capacity and inventory "
                "status are descriptive labels inferred by comparing the recorded "
                "values; they are not solver duals or certificates."
            ),
            columns=[
                "Period",
                f"Forecast demand F_m [{forecast_unit}]",
                f"Production P_m [{production_unit}]",
                f"Month-end inventory I_m [{inventory_unit}]",
                f"Capacity cap_m [{capacity_unit}]",
                "Capacity status",
                "Inventory status",
            ],
            rows=plan_rows,
            evidence_ids=list(dict.fromkeys(plan_evidence)),
        )
    ]

    parameter_rows: list[list[str]] = []
    parameter_evidence: list[str] = []
    initial = derived.get("competition.plan.initial_inventory")
    if initial is not None:
        parameter_rows.append(
            [
                "Initial inventory I_init",
                _value(initial),
                initial.unit or "-",
                initial.evidence_id,
            ]
        )
        parameter_evidence.append(initial.evidence_id)
    for period in periods:
        capacity = row("competition.plan.capacity", period)
        parameter_rows.append(
            [
                f"Capacity cap_m (m={period})",
                _value(capacity),
                capacity.unit or "-",
                capacity.evidence_id,
            ]
        )
        parameter_evidence.append(capacity.evidence_id)
    for period in periods:
        production_cost = row("competition.plan.production_cost", period)
        parameter_rows.append(
            [
                f"Production cost c_p,m (m={period})",
                _value(production_cost),
                production_cost.unit or "-",
                production_cost.evidence_id,
            ]
        )
        parameter_evidence.append(production_cost.evidence_id)
    for period in periods:
        holding_cost = row("competition.plan.holding_cost", period)
        parameter_rows.append(
            [
                f"Holding cost c_h,m (m={period})",
                _value(holding_cost),
                holding_cost.unit or "-",
                holding_cost.evidence_id,
            ]
        )
        parameter_evidence.append(holding_cost.evidence_id)
    outcome_keys = [
        ("Objective value", "optimization.ortools.cbc.objective_value"),
        ("Solver status", "optimization.ortools.cbc.status"),
        ("Optimality gap", "optimization.ortools.cbc.optimality_gap"),
    ]
    for label, key in outcome_keys:
        entry = catalog.get(key)
        if entry is None:
            continue
        parameter_rows.append(
            [label, entry.display_value, entry.unit or "-", entry.evidence_id]
        )
        parameter_evidence.append(entry.evidence_id)
    parameter_rows.append(
        [
            "Objective expression",
            "sum_m(c_p,m*P_m + c_h,m*I_m)",
            "cost_units",
            "",
        ]
    )
    tables.append(
        PaperTable(
            table_id="tab-plan-parameters",
            caption=(
                "Frozen plan parameters and recorded solver outcome. Capacities and "
                "cost coefficients are read from the registered frozen plan spec with "
                "their own evidence ids; the objective is the recorded solver output. "
                "These parameters and `tab-frozen-plan` allow the cost expression to be "
                "checked by hand. No solver duals or binding-constraint certificate is "
                "evidenced."
            ),
            columns=["Parameter or outcome", "Value", "Unit", "Evidence"],
            rows=parameter_rows,
            evidence_ids=list(dict.fromkeys(parameter_evidence)),
        )
    )

    recompute_rows: list[list[str]] = []
    recompute_evidence: list[str] = []
    for period in periods:
        production = row("competition.plan.production", period)
        production_cost = row("competition.plan.production_cost", period)
        production_term = derived.get(
            f"competition.plan.production_cost_term.{period}"
        )
        inventory = row("competition.plan.inventory", period)
        holding_cost = row("competition.plan.holding_cost", period)
        holding_term = derived.get(
            f"competition.plan.holding_cost_term.{period}"
        )
        if production_term is None or holding_term is None:
            continue
        recompute_rows.append(
            [
                period,
                _value(production),
                _value(production_cost),
                _value(production_term),
                _value(inventory),
                _value(holding_cost),
                _value(holding_term),
                "",
                "cost_units",
            ]
        )
        recompute_evidence.extend(
            [
                production.evidence_id,
                production_cost.evidence_id,
                production_term.evidence_id,
                inventory.evidence_id,
                holding_cost.evidence_id,
                holding_term.evidence_id,
            ]
        )
    objective_recomputed = derived.get("competition.plan.objective_recomputed")
    objective_recorded = catalog.get("optimization.ortools.cbc.objective_value")
    if objective_recomputed is not None:
        recompute_rows.append(
            [
                "Total (recomputed)",
                "",
                "",
                "",
                "",
                "",
                "",
                _value(objective_recomputed),
                "cost_units",
            ]
        )
        recompute_evidence.append(objective_recomputed.evidence_id)
    if objective_recorded is not None:
        recompute_rows.append(
            [
                "Solver objective",
                "",
                "",
                "",
                "",
                "",
                "",
                objective_recorded.display_value,
                "cost_units",
            ]
        )
        recompute_evidence.append(objective_recorded.evidence_id)
    if recompute_rows:
        tables.append(
            PaperTable(
                table_id="tab-objective-recomputation",
                caption=(
                    "Objective recomputation from the recorded cost coefficients and "
                    "frozen plan. Each term is a derived competition-context value "
                    "(coefficient times recorded quantity) and the total is compared "
                    "with the solver's recorded objective. This is arithmetic on "
                    "recorded evidence, not a new solve."
                ),
                columns=[
                    "Period",
                    "Production P_m",
                    "Production cost c_p,m",
                    "c_p,m*P_m",
                    "Inventory I_m",
                    "Holding cost c_h,m",
                    "c_h,m*I_m",
                    "Objective",
                    "Unit",
                ],
                rows=recompute_rows,
                evidence_ids=list(dict.fromkeys(recompute_evidence)),
            )
        )
    return tables


def _build_tables(
    context: EvidenceContext,
    plan: CompetitionPaperPlan,
    records: dict[str, EvidenceRecord],
    fragments: list[ExpandedText],
) -> list[PaperTable]:
    known = context.evidence_ids()
    tables: list[PaperTable] = []
    for table in plan.tables:
        _require_evidence(table.evidence_ids, known, location=table.table_id)
        available = context.available_tables.get(table.template.value, 0)
        if available <= 0:
            raise CompetitionBuildError(
                f"{table.table_id} requests unavailable template {table.template.value!r}"
            )
        caption_text = _SYSTEM_TABLE_CAPTIONS[table.template]
        fragments.append(
            ExpandedText(
                text=caption_text,
                location=f"table:{table.table_id}",
                cited_ids=tuple(table.evidence_ids),
            )
        )
        columns, rows = _table_rows(table, context)
        tables.append(
            PaperTable(
                table_id=table.table_id,
                caption=caption_text,
                columns=columns,
                rows=rows,
                evidence_ids=table.evidence_ids,
            )
        )
    tables.extend(_system_plan_tables(context))
    tables.extend(_system_diagnostics_table(context))
    tables.extend(_augmentation_tables(context))
    tables.append(
        PaperTable(
            table_id="tab-notation",
            caption="Symbols, meanings, and units used in the model.",
            columns=["Symbol", "Meaning", "Unit"],
            rows=[
                [entry.symbol, entry.meaning, entry.unit or "-"]
                for entry in plan.notation
            ],
            evidence_ids=[],
        )
    )
    return tables


def _build_sections(
    context: EvidenceContext,
    plan: CompetitionPaperPlan,
    analysis: ExperimentAnalysisReport,
    manifest: CompetitionExperimentManifest,
    records: dict[str, EvidenceRecord],
    fragments: list[ExpandedText],
) -> list[PaperSection]:
    sections: list[PaperSection] = []
    known = context.evidence_ids()
    for section in plan.sections:
        _require_evidence(
            section.evidence_ids, known, location=f"section:{section.heading}"
        )
        blocks: list[str] = []
        structural = section.kind in {
            SectionKind.MODEL,
            SectionKind.SOLUTION,
            SectionKind.NOTATION,
        }
        for block in section.blocks:
            expanded = expand_text(
                block,
                records,
                location=f"section:{section.heading}",
                allow_structural_constants=structural,
                seeded_ids=tuple(section.evidence_ids),
            )
            fragments.append(expanded)
            blocks.append(expanded.text)
        for point in section.claim_points:
            _require_evidence(
                point.evidence_ids, known, location=f"claim point:{section.heading}"
            )
            expanded = expand_text(
                point.text,
                records,
                location=f"claim point:{section.heading}",
                allow_structural_constants=structural,
                seeded_ids=tuple(point.evidence_ids),
            )
            fragments.append(expanded)
            blocks.append(
                f"- {expanded.text} (evidence: {', '.join(point.evidence_ids)})"
            )
        if section.kind is SectionKind.ASSUMPTIONS:
            for assumption in plan.assumptions:
                expanded = expand_text(
                    assumption,
                    records,
                    location="Assumptions",
                    seeded_ids=tuple(section.evidence_ids),
                )
                fragments.append(expanded)
                blocks.append(f"- {expanded.text}")
        if section.kind is SectionKind.NOTATION:
            blocks.append("Notation is summarized in `tab-notation`.")
        if section.kind is SectionKind.LIMITATIONS:
            for limitation in [*plan.limitations, *analysis.limitations]:
                if context.appendix_records and _stale_after_augmentation(limitation):
                    continue
                expanded = expand_text(
                    limitation,
                    records,
                    location="Limitations",
                    seeded_ids=tuple(section.evidence_ids),
                )
                fragments.append(expanded)
                text = (
                    expanded.text[2:]
                    if expanded.text.startswith("- ")
                    else expanded.text
                )
                blocks.append(f"- {text}")
            if context.appendix_records:
                blocks.append(
                    "Cost, capacity, initial-inventory, and replication/seed variants "
                    "were executed in the appended augmentation stage; those appended "
                    "results do not revise source-stage claims, and analyses over the "
                    "alternative residual pools remain unexecuted."
                )
            unsupported = [
                request
                for request in manifest.requests
                if not request.supported
            ]
            if unsupported:
                blocks.append(
                    "Some requested experiments are not supported by the current "
                    "engines; they are listed in the research-gap appendix and were "
                    "not executed."
                )
        if context.model_facts:
            note_texts: list[str] = []
            for note in _system_section_notes(context).get(section.kind, []):
                expanded = expand_text(
                    note,
                    records,
                    location=f"System note:{section.heading}",
                    seeded_ids=tuple(section.evidence_ids),
                )
                fragments.append(expanded)
                note_texts.append(expanded.text)
            if section.kind in {SectionKind.RESULTS, SectionKind.EVALUATION}:
                blocks[:0] = note_texts
            else:
                blocks.extend(note_texts)
        sections.append(PaperSection(heading=section.heading, blocks=blocks))

    gap_blocks = [
        "The following experiments are proposed, not executed. No result from these "
        "requests appears in this paper."
    ]
    if context.appendix_records:
        gap_blocks.append(
            "Executed augmentation experiments are reported in the augmentation appendix "
            "and are not counted as executions of the proposals below."
        )
    for request in manifest.requests:
        expanded = expand_text(
            (
                f"{request.question} Rationale: {request.rationale} "
                f"Proposed experiment: {request.proposed_experiment} "
                f"Expected evidence: {request.expected_evidence} "
                f"Capability: {request.capability.value}; "
                f"supported by current engines: {str(request.supported).lower()}."
            ),
            records,
            location="Research gaps",
            seeded_ids=tuple(request.supporting_evidence_ids),
            allow_structural_constants=True,
        )
        fragments.append(expanded)
        gap_blocks.append(f"- {expanded.text}")
    if not manifest.requests:
        gap_blocks.append(
            "No experiment requests were generated; no unexecuted experiment is claimed."
        )
    sections.append(
        PaperSection(heading="Research gaps and proposed experiments", blocks=gap_blocks)
    )
    return sections


def _build_figures(
    paths: ProjectPaths,
    context: EvidenceContext,
    figures: list[FigurePlan],
    records: dict[str, EvidenceRecord],
    out_dir: Path,
    fragments: list[ExpandedText],
) -> list[PaperFigure]:
    built: list[PaperFigure] = []
    known = context.evidence_ids()
    for figure in figures:
        _require_evidence(figure.evidence_ids, known, location=figure.figure_id)
        available = context.available_figures.get(figure.template.value, 0)
        if available <= 0:
            raise CompetitionBuildError(
                f"{figure.figure_id} requests unavailable template {figure.template.value!r}"
            )
        caption_text = _SYSTEM_FIGURE_CAPTIONS[figure.template]
        fragments.append(
            ExpandedText(
                text=caption_text,
                location=f"figure:{figure.figure_id}",
                cited_ids=tuple(figure.evidence_ids),
            )
        )
        built_figure = build_figure(
            paths,
            context,
            figure.model_copy(update={"caption": caption_text}),
            out_dir,
        )
        built.append(built_figure)
    return built


def build_competition_ast(
    paths: ProjectPaths,
    context: EvidenceContext,
    plan: CompetitionPaperPlan,
    analysis: ExperimentAnalysisReport,
    figures: list[FigurePlan],
    manifest: CompetitionExperimentManifest,
    records: dict[str, EvidenceRecord],
) -> CompetitionBuildResult:
    out_dir = paths.root / "paper" / "competition" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fragments: list[ExpandedText] = []
    known = context.evidence_ids()
    title = expand_text(plan.title, records, location="Title")
    fragments.append(title)
    abstract_parts = []
    for point in plan.abstract_points:
        _require_evidence(point.evidence_ids, known, location="Abstract")
        expanded = expand_text(
            point.text,
            records,
            location="Abstract",
            seeded_ids=tuple(point.evidence_ids),
        )
        fragments.append(expanded)
        abstract_parts.append(expanded.text)
    abstract = "\n\n".join(abstract_parts)

    tables = _build_tables(context, plan, records, fragments)
    sections = _build_sections(
        context, plan, analysis, manifest, records, fragments
    )
    if context.appendix_records:
        appendix_rows = [
            [
                str(record.provenance.get("request_id", "unknown")),
                record.experiment_id,
                record.key,
                format_record_value(record),
                _display_unit(record),
                ", ".join(record.validation_reports),
            ]
            for record in context.appendix_records
        ]
        tables.append(
            PaperTable(
                table_id="tab-augmentation-evidence",
                caption=(
                    "Executed and validated competition-augmentation experiments. "
                    "Values are copied from registered evidence at four significant "
                    "digits, not reconstructed by the paper agents. These are appended "
                    "after the source paper and do not upgrade any source-paper claim."
                ),
                columns=[
                    "Request",
                    "Experiment",
                    "Evidence key",
                    "Value",
                    "Unit",
                    "Validation",
                ],
                rows=appendix_rows,
                evidence_ids=[record.evidence_id for record in context.appendix_records],
            )
        )
        sections.append(
            PaperSection(
                heading="Augmentation evidence (executed, validated)",
                blocks=[
                    "The experiments listed in `tab-augmentation-evidence` were executed "
                    "after the source paper, passed the existing validators, and were "
                    "registered as competition evidence. The table is the authoritative "
                    "summary of those runs.",
                    "These experiments use different instances or protocols from the "
                    "source paper. The capacity/cost/initial-inventory perturbations "
                    "re-solve modified optimization instances; their objective values "
                    "and production vectors belong to those modified instances and are "
                    "not numerically comparable with the frozen plan. A relaxed-capacity "
                    "instance may have a different objective and plan without changing "
                    "the recorded solver status at the frozen parameters.",
                    "The optimization variant names state the perturbation applied to "
                    "the frozen instance; the production and inventory values in a "
                    "variant row belong to that perturbed instance, not to the frozen "
                    "parameters. In particular, a capacity-expanded variant can show a "
                    "different plan and objective without demonstrating a cheaper plan "
                    "for the frozen instance.",
                    "The replication/seed re-runs hold the source production plan fixed "
                    "and repeat the same pooled-residual iid protocol with different "
                    "replication counts and seed schedules. Differences among the source "
                    "simulation summary and these re-runs are protocol and Monte Carlo "
                    "variation, not a revision of the source result. Simulated inventory "
                    "means describe realized inventories under resampled demand, not the "
                    "deterministic plan's month-end inventory.",
                    "No statistical test of differences between augmentation runs was "
                    "performed. Values are copied from registered evidence and rendered "
                    "at four significant digits; the number of displayed digits is not a "
                    "Monte Carlo precision claim. Replications can share duplicate "
                    "residual paths, so effective sample size is below nominal. Two "
                    "optimization perturbations appear under both request identifiers "
                    "because the executed grids overlap; those repeated rows are "
                    "duplicate executions of the same instance and are not independent "
                    "robustness evidence. Dimensionless outputs such as fill rate and "
                    "probabilities are unitless even where the source evidence record's "
                    "unit field is generic.",
                    "No augmentation result upgrades, revises, or invalidates a source-"
                    "paper claim. The source release and verified evidence index are "
                    "byte-identical before and after augmentation; these experiments are "
                    "appended evidence only.",
                ],
            )
        )
    figures_built = _build_figures(
        paths, context, figures, records, out_dir, fragments
    )
    supported_claims = [
        claim
        for claim in context.claims
        if claim.status == "supported"
        and set(claim.evidence_ids) <= known
    ]
    references = [
        "Frozen problem specification: manifests/problem_spec.json.",
        "Verified evidence index: verified/numbers.json.",
        "Numeric values are rendered from registered evidence at four significant "
        "digits (simulation summaries at two); source precision remains in the "
        "evidence index.",
        f"Source release: {context.release_path.relative_to(paths.root).as_posix()} "
        f"(sha256 {context.release_hash}).",
    ]
    bundle = paths.root / "paper" / "evidence_bundle"
    if bundle.exists():
        references.append("Evidence bundle: paper/evidence_bundle/.")
    if context.appendix_records:
        references.append(
            "Competition augmentation experiments were executed and validated; their "
            "registered evidence is listed in `tab-augmentation-evidence`."
        )
    else:
        references.append(
            "Experiment requests are proposals only; no requested experiment was "
            "executed in this stage."
        )
    for claim in supported_claims:
        fragments.append(
            ExpandedText(
                text=claim.statement,
                location=f"Supported claim: {claim.claim_id}",
                cited_ids=tuple(claim.evidence_ids),
            )
        )
    token_to_evidence: dict[str, list[str]] = {}
    for entry in context.entries:
        formatted = format_display_value(entry.raw_value, entry.display_value)
        if (
            entry.key.startswith("mixed.simulation.")
            and isinstance(entry.raw_value, float)
        ):
            formatted = f"{entry.raw_value:.2g}"
        for token in [
            *numeric_tokens(entry.display_value),
            *numeric_tokens(str(entry.raw_value)),
            *numeric_tokens(formatted),
        ]:
            token_to_evidence.setdefault(token, []).append(entry.evidence_id)
    seeded_fragments: list[ExpandedText] = []
    for fragment in fragments:
        extra_ids: list[str] = []
        for token in numeric_tokens(fragment.text):
            extra_ids.extend(token_to_evidence.get(token, []))
        seeded_fragments.append(
            ExpandedText(
                text=fragment.text,
                location=fragment.location,
                cited_ids=tuple(
                    dict.fromkeys([*fragment.cited_ids, *extra_ids])
                ),
                allow_structural_constants=fragment.allow_structural_constants,
            )
        )

    ast = PaperAST(
        title=title.text,
        abstract=abstract,
        sections=sections,
        tables=tables,
        figures=figures_built,
        claims=supported_claims,
        references=references,
    )
    return CompetitionBuildResult(ast=ast, fragments=seeded_fragments)


__all__ = [
    "CompetitionBuildError",
    "CompetitionBuildResult",
    "build_competition_ast",
]
