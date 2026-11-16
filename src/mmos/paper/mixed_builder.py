import hashlib
import json

import numpy as np

from mmos.core.project import (
    ProjectPaths,
    load_manifest,
    load_optimization_result,
    sha256_file,
)
from mmos.data.table import load_single_table
from mmos.experiments.engines import forecast_baseline
from mmos.mixed.artifacts import load_envelope
from mmos.mixed.validation import run_record_path
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.mixed import DAGRunRecord, ForecastArtifact, MixedExecutionPlan
from mmos.schemas.paper import PaperAST, PaperFigure, PaperSection, PaperTable
from mmos.schemas.project import SourceRole


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _plot_dag(paths: ProjectPaths, plan: MixedExecutionPlan, run_id: str) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper_dir = paths.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    target = paper_dir / "forecast_planning_dag.png"

    number = {node.node_id: index for index, node in enumerate(plan.nodes)}
    figure, axis = plt.subplots(figsize=(8, 3))
    for node in plan.nodes:
        axis.text(
            number[node.node_id],
            0.5,
            f"{node.node_id}\n({node.task_family.value})",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eef4fb", "edgecolor": "#1f77b4"},
        )
    for edge in plan.edges:
        axis.annotate(
            "",
            xy=(number[edge.consumer_node] - 0.18, 0.5),
            xytext=(number[edge.producer_node] + 0.18, 0.5),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#444444"},
        )
        axis.text(
            (number[edge.producer_node] + number[edge.consumer_node]) / 2,
            0.62,
            f"{edge.required_artifact_type} v{edge.required_schema_version}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xlim(-0.6, len(plan.nodes) - 0.4)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(f"Mixed execution DAG {plan.plan_id} (run {run_id})")
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return PaperFigure(
        figure_id="forecast-planning-dag",
        caption=(
            "Plumbing-only view of the frozen mixed execution DAG. The forecasting node "
            "publishes a typed, hashed demand forecast artifact; the optimization node "
            "consumes that artifact through a versioned contract. The arrow label is the "
            "required artifact type and schema version, not a runtime quantity. The "
            "substantive forecast/capacity/production/inventory result is shown separately."
        ),
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=[f"ev-mixed-{run_id}-artifact-id"],
    )


def _plot_plan(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    artifact: ForecastArtifact,
    values: dict[str, float],
    run_id: str,
) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paper_dir = paths.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    target = paper_dir / "forecast_planning_plan.png"

    periods = list(plan.spec.planning_periods)
    positions = np.arange(len(periods))
    forecasts = [float(value) for value in artifact.point_forecasts]
    productions = [float(values.get(f"production_{index}", 0.0)) for index in positions]
    capacities = [float(capacity) for capacity in plan.spec.capacities]
    inventories = [float(values.get(f"inventory_{index}", 0.0)) for index in positions]

    figure, (axis, difference_axis) = plt.subplots(
        2,
        1,
        figsize=(8, 6.5),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    width = 0.25
    axis.bar(positions - width, forecasts, width, label="Point forecast", color="#4c72b0")
    axis.bar(positions, productions, width, label="Production", color="#dd8452")
    axis.bar(
        positions + width,
        capacities,
        width,
        label="Capacity",
        color="#c7c7c7",
        edgecolor="#666666",
    )
    axis.plot(
        positions,
        inventories,
        marker="o",
        color="#55a868",
        label="Ending inventory",
    )
    opening = float(plan.spec.initial_inventory)
    for index, position in enumerate(positions):
        if abs(productions[index] - capacities[index]) <= 1e-6:
            axis.annotate(
                f"capacity bind ({productions[index]:g} = {capacities[index]:g}, "
                "construction not finding)",
                (position, capacities[index]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#8c2d04",
            )
        if inventories[index] > 1e-9:
            axis.annotate(
                "inventory forced by hard demand equality",
                (position, inventories[index]),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7,
                color="#2d6a2d",
            )
        axis.annotate(
            f"{opening:g}+{productions[index]:g}-{forecasts[index]:g}"
            f"={inventories[index]:g}",
            (position, max(capacities[index], productions[index], forecasts[index])),
            textcoords="offset points",
            xytext=(0, 20),
            ha="center",
            fontsize=7,
        )
        opening = inventories[index]
    axis.set_xticks(positions, periods)
    axis.set_ylabel(f"Units ({artifact.unit})")
    axis.set_title(f"Forecast, capacity, production, and inventory (run {run_id})")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)

    production_minus_forecast = [
        productions[index] - forecasts[index] for index in range(len(positions))
    ]
    forecast_minus_capacity = [
        forecasts[index] - capacities[index] for index in range(len(positions))
    ]
    difference_axis.bar(
        positions - 0.18,
        production_minus_forecast,
        0.34,
        label="Production - forecast",
        color="#dd8452",
    )
    difference_axis.bar(
        positions + 0.18,
        forecast_minus_capacity,
        0.34,
        label="Forecast - capacity",
        color="#8c2d04",
    )
    difference_axis.axhline(0, color="#333333", lw=0.8)
    for index, position in enumerate(positions):
        difference_axis.annotate(
            f"{production_minus_forecast[index]:+g}",
            (position - 0.18, production_minus_forecast[index]),
            textcoords="offset points",
            xytext=(0, 4 if production_minus_forecast[index] >= 0 else -12),
            ha="center",
            fontsize=8,
        )
        difference_axis.annotate(
            f"{forecast_minus_capacity[index]:+g}",
            (position + 0.18, forecast_minus_capacity[index]),
            textcoords="offset points",
            xytext=(0, 4 if forecast_minus_capacity[index] >= 0 else -12),
            ha="center",
            fontsize=8,
        )
    difference_axis.set_xticks(positions, periods)
    difference_axis.set_ylabel("Units")
    difference_axis.set_title("Relied-upon differences (magnified)")
    difference_axis.legend(loc="upper left", fontsize=8)
    difference_axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    binding_text = ", ".join(
        f"{periods[index]} production {productions[index]:g} = capacity "
        f"{capacities[index]:g}"
        for index in range(len(positions))
        if abs(productions[index] - capacities[index]) <= 1e-6
    )
    return PaperFigure(
        figure_id="forecast-production-plan",
        caption=(
            "Top panel: forecast, capacity, production, and ending inventory by planning "
            "period. Bars are the frozen point forecast, the solved production quantity, "
            "and the frozen capacity; the line is ending inventory, and annotations are "
            "the recurrence opening + production - demand = closing. Bottom panel: "
            "magnified differences (production - forecast and forecast - capacity), which "
            "carry the binding claim. "
            + (f"Binding period: {binding_text}. " if binding_text else "")
            + "The binding is the forecast-capacity excess, not the ending-inventory zero; "
            "a pre-build appears as positive production-minus-forecast and is drawn down in "
            "the constrained period. The 10-unit inventory is forced by the hard demand "
            "equality, and the 140 = 140 bar is a construction of the frozen constraint, "
            "not a finding. The magnified panel cannot distinguish a correctly enforced "
            "bound from a mis-indexed one; no non-binding companion solve is shown."
        ),
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=[
            f"ev-mixed-{run_id}-forecast-points",
            f"ev-mixed-{run_id}-objective",
        ],
    )


def build_mixed_paper(
    paths: ProjectPaths, plan: MixedExecutionPlan, run_id: str
) -> PaperAST:
    record = DAGRunRecord.model_validate_json(
        run_record_path(paths, run_id).read_text(encoding="utf-8")
    )
    envelope_path = paths.mixed_dir / "runs" / run_id / "artifacts" / "forecast_envelope.json"
    envelope = load_envelope(envelope_path)
    artifact = ForecastArtifact.model_validate_json(
        (paths.root / envelope.payload_path).read_text(encoding="utf-8")
    )
    if record.optimization_experiment_id is None:
        raise ValueError("mixed run has no optimization experiment")
    optimization_id = record.optimization_experiment_id
    result = load_optimization_result(paths, optimization_id)
    solver = next(
        (solution for solution in result.solutions if solution.method == "ortools.cbc"),
        None,
    )
    if solver is None:
        raise ValueError("mixed run has no CBC solution")
    values = solver.variable_values
    horizon = plan.spec.forecast_horizon

    forecast_rows = [
        [
            timestamp,
            _fmt(value, 4),
            artifact.unit,
            artifact.model_id,
        ]
        for timestamp, value in zip(
            artifact.timestamps, artifact.point_forecasts, strict=True
        )
    ]
    forecast_table = PaperTable(
        table_id=f"tab-{run_id}-forecast",
        caption=(
            "Contract-complete demand forecast artifact. These are deterministic point "
            "forecasts from a temporal baseline; 'verified' refers to the artifact contract "
            "and hash checks, not to forecast accuracy, and no prediction interval is "
            "produced. The frozen forecast is a global-trend extrapolation over the full "
            "window and is insensitive to the last observed level; a local or "
            "last-difference slope would change it, and no alternate forecast or plan was "
            "solved."
        ),
        columns=["Period", "Point forecast", "Unit", "Model"],
        rows=forecast_rows,
        evidence_ids=[
            f"ev-mixed-{run_id}-forecast-points",
            f"ev-mixed-{run_id}-forecast-model",
        ],
    )

    metrics_path = (
        paths.experiment_dir(artifact.forecast_experiment_id) / "artifacts" / "metrics.json"
    )
    forecast_result = ExperimentResult.model_validate_json(
        metrics_path.read_text(encoding="utf-8")
    )
    metric_rows: list[list[str]] = []
    for model_id in sorted(forecast_result.metric_summaries):
        for metric_name in ("mae", "rmse", "mase"):
            summary = forecast_result.metric_summaries[model_id].get(metric_name)
            if summary is None or summary.mean is None:
                continue
            folds = ", ".join(f"{value:.4f}" for value in summary.fold_values)
            metric_rows.append(
                [
                    model_id,
                    metric_name,
                    _fmt(summary.mean, 4),
                    str(len(summary.fold_values)),
                    folds,
                ]
            )
    metrics_table = PaperTable(
        table_id=f"tab-{run_id}-forecast-metrics",
        caption=(
            "Descriptive fold-by-fold diagnostic for the forecast model and the last-value "
            "comparison baseline, presented as provenance evidence, not a forecasting "
            "evaluation. Lower MAE/RMSE/MASE is better; MASE uses the per-fold in-sample "
            "random-walk scale. These numbers quantify measured error, not the accuracy of "
            "the future point forecast."
        ),
        columns=["Model", "Metric", "Mean", "Folds", "Fold values"],
        rows=metric_rows,
        evidence_ids=[
            f"ev-{artifact.forecast_experiment_id}-evaluation-{model_id}-{metric_name}"
            for model_id in sorted(forecast_result.metric_summaries)
            for metric_name in ("mae", "rmse", "mase")
            if metric_name in forecast_result.metric_summaries[model_id]
        ],
    )

    parameter_rows = []
    for index, period in enumerate(plan.spec.planning_periods):
        parameter_rows.append(
            [
                period,
                _fmt(float(plan.spec.capacities[index]), 4),
                _fmt(float(plan.spec.production_costs[index]), 4),
                _fmt(float(plan.spec.holding_costs[index]), 4),
            ]
        )
    parameter_table = PaperTable(
        table_id=f"tab-{run_id}-parameters",
        caption=(
            "Frozen planning parameters. Capacity is in units, matching the forecast "
            "artifact's unit column. Demand, production, capacity, initial inventory, and "
            "ending inventory are all in the same unit. Production cost is in cost units "
            "per produced unit; holding cost is in cost units per unit held for one period."
        ),
        columns=[
            "Period",
            "Capacity (units)",
            "Production cost (cost units/unit)",
            "Holding cost (cost units/unit-period)",
        ],
        rows=parameter_rows,
        evidence_ids=[f"ev-mixed-{run_id}-spec-hash"],
    )

    history = load_single_table(paths).sort(plan.spec.time_column)
    history_payload = [
        [str(period), float(demand)]
        for period, demand in zip(
            history[plan.spec.time_column].to_list(),
            history[plan.spec.target_column].to_list(),
            strict=True,
        )
    ]
    history_digest = hashlib.sha256(
        json.dumps(history_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if envelope.provenance.get("history_digest") != history_digest:
        raise ValueError("displayed history does not match the frozen history digest")
    history_rows = [
        [period, _fmt(demand, 4)] for period, demand in history_payload
    ]
    history_table = PaperTable(
        table_id=f"tab-{run_id}-history",
        caption=(
            f"Input demand series used by the forecasting node "
            f"({len(history_payload)} rows). The displayed rows hash to the frozen "
            f"history digest recorded in the artifact provenance, so the table is the "
            f"ingested data rather than a reconstructed series."
        ),
        columns=["Period", f"Demand ({artifact.unit})"],
        rows=history_rows,
        evidence_ids=[
            f"ev-mixed-{run_id}-forecast-history-rows",
            f"ev-mixed-{run_id}-forecast-history-digest",
            f"ev-mixed-{run_id}-forecast-dataset",
        ],
    )

    series = [value for _, value in history_payload]
    n_rows_history = len(series)
    fold_rows: list[list[str]] = []
    forecast_columns = [f"Forecast h{h}" for h in range(1, horizon + 1)]
    scale_lag = plan.spec.seasonal_period or 1
    recomputed_fold_mae: list[float] = []
    for index in range(plan.spec.n_splits):
        train_end = n_rows_history - plan.spec.n_splits * horizon + index * horizon
        train = series[:train_end]
        test = series[train_end : train_end + horizon]
        fold_slope = (train[-1] - train[0]) / (len(train) - 1)
        fold_forecast = [
            float(value)
            for value in forecast_baseline(
                artifact.model_id,
                np.array(train, dtype=float),
                horizon,
                plan.spec.seasonal_period,
            )
        ]
        if len(train) > scale_lag:
            naive_scale = sum(
                abs(train[position] - train[position - scale_lag])
                for position in range(scale_lag, len(train))
            ) / (len(train) - scale_lag)
        else:
            naive_scale = 0.0
        fold_mae = sum(
            abs(forecast - actual)
            for forecast, actual in zip(fold_forecast, test, strict=True)
        ) / horizon
        recomputed_fold_mae.append(fold_mae)
        fold_mase = f"{fold_mae / naive_scale:.4f}" if naive_scale > 0 else "undefined"
        fold_rows.append(
            [
                str(index + 1),
                f"[0,{train_end})",
                f"[{train_end},{train_end + horizon})",
                _fmt(train[0], 4),
                _fmt(train[-1], 4),
                _fmt(fold_slope, 4) if artifact.model_id == "naive.drift" else "n/a",
                *[_fmt(value, 4) for value in fold_forecast],
                ", ".join(_fmt(value, 4) for value in test),
                _fmt(fold_mae, 4),
                fold_mase,
            ]
        )
    frozen_mae = forecast_result.metric_summaries.get(artifact.model_id, {}).get("mae")
    if frozen_mae is not None and len(frozen_mae.fold_values) == len(recomputed_fold_mae):
        mismatched = [
            index + 1
            for index, (frozen, recomputed) in enumerate(
                zip(frozen_mae.fold_values, recomputed_fold_mae, strict=True)
            )
            if abs(frozen - recomputed) > 1e-6
        ]
        if mismatched:
            raise ValueError(
                "recomputed fold MAE does not match frozen evaluation on folds "
                + ", ".join(str(index) for index in mismatched)
            )
    fold_table = PaperTable(
        table_id=f"tab-{run_id}-fold-derivation",
        caption=(
            "Per-fold reconstruction from the displayed history: training and test windows, "
            "model rule, the H-step forecast, the realized test demand, and the recomputed "
            "MAE and MASE. The model rule is applied per fold exactly as in the frozen "
            "evaluation. MASE uses the per-fold in-sample random-walk scale "
            "sum(|y_t - y_{t-lag}|)/(N-lag), not the out-of-sample last-value competitor. "
            "The recomputed MAE values are checked against the frozen evaluation."
        ),
        columns=[
            "Fold",
            "Training window",
            "Test window",
            "y_first",
            "y_last",
            "Slope (drift only)",
            *forecast_columns,
            "Test demand",
            "Recomputed MAE",
            "MASE",
        ],
        rows=fold_rows,
        evidence_ids=[
            f"ev-{artifact.forecast_experiment_id}-evaluation-{artifact.model_id}-mase-details",
            f"ev-mixed-{run_id}-forecast-history-digest",
        ],
    )

    window_rows: list[list[str]] = []
    for window_length in (12, 15, 18, 21, 24):
        if window_length > len(series):
            continue
        window = series[:window_length]
        window_slope = (window[-1] - window[0]) / (len(window) - 1)
        window_forecast = forecast_baseline(
            artifact.model_id,
            np.array(window, dtype=float),
            horizon,
            plan.spec.seasonal_period,
        )
        window_rows.append(
            [
                str(window_length),
                _fmt(window[-1], 4),
                _fmt(window_slope, 4) if artifact.model_id == "naive.drift" else "n/a",
                *[_fmt(float(value), 4) for value in window_forecast],
            ]
        )
    window_table = PaperTable(
        table_id=f"tab-{run_id}-window-sensitivity",
        caption=(
            "Slope and H-step forecast as a function of the history window length under "
            "the declared drift rule, with the anchor fixed at the first observation. "
            "The slope is window-dependent; the 24-observation slope of 2.0000 happens to "
            "equal the 12-observation slope, which is an arithmetic coincidence, not "
            "stability. The plan's pre-build and capacity bind are functions of the slope "
            "and would change under another window."
        ),
        columns=[
            "Window length",
            "y_last",
            "Slope (drift only)",
            *[f"Forecast h{h}" for h in range(1, horizon + 1)],
        ],
        rows=window_rows,
        evidence_ids=[f"ev-mixed-{run_id}-forecast-history-digest"],
    )

    planning_rows = []
    for index, period in enumerate(plan.spec.planning_periods):
        production = float(values.get(f"production_{index}", 0.0))
        inventory = float(values.get(f"inventory_{index}", 0.0))
        production_cost = production * float(plan.spec.production_costs[index])
        holding_cost = inventory * float(plan.spec.holding_costs[index])
        planning_rows.append(
            [
                period,
                _fmt(float(artifact.point_forecasts[index]), 4),
                _fmt(production, 4),
                _fmt(float(plan.spec.capacities[index]), 4),
                _fmt(inventory, 4),
                _fmt(production_cost, 4),
                _fmt(holding_cost, 4),
            ]
        )
    planning_table = PaperTable(
        table_id=f"tab-{run_id}-plan",
        caption=(
            "Deterministic production plan conditional on the point forecast. Production "
            "is bounded by the frozen capacity and ending inventory is nonnegative. For "
            "this instance the values are exact at four decimals and re-summing them "
            "matches the solver objective exactly."
        ),
        columns=[
            "Period",
            "Forecast demand (units)",
            "Production (units)",
            "Capacity (units)",
            "Ending inventory (units)",
            "Production cost (cost units)",
            "Holding cost (cost units)",
        ],
        rows=planning_rows,
        evidence_ids=[f"ev-mixed-{run_id}-objective"],
    )

    recurrence_rows: list[list[str]] = []
    opening = float(plan.spec.initial_inventory)
    recomputed_objective = 0.0
    for index, period in enumerate(plan.spec.planning_periods):
        demand = float(artifact.point_forecasts[index])
        production = float(values.get(f"production_{index}", 0.0))
        closing = opening + production - demand
        holding_cost = max(closing, 0.0) * float(plan.spec.holding_costs[index])
        production_cost = production * float(plan.spec.production_costs[index])
        recomputed_objective += production_cost + holding_cost
        recurrence_rows.append(
            [
                period,
                _fmt(opening, 4),
                _fmt(production, 4),
                _fmt(demand, 4),
                _fmt(closing, 4),
                _fmt(production_cost, 4),
                _fmt(holding_cost, 4),
            ]
        )
        opening = closing
    recurrence_table = PaperTable(
        table_id=f"tab-{run_id}-balance",
        caption=(
            "Inventory-balance recurrence for every period: closing = opening + "
            "production - forecast demand. The LP enforces this as a hard equality with "
            "no stock-out or backorder variable, so demand is met exactly. Production cost "
            "is production x its period rate; holding cost is closing inventory x its "
            "period rate for one period. The per-period totals reconcile with the plan "
            "table and the recomputed objective below."
        ),
        columns=[
            "Period",
            "Opening inventory",
            "Production",
            "Forecast demand",
            "Closing inventory",
            "Production cost (cost units)",
            "Holding cost (cost units)",
        ],
        rows=recurrence_rows,
        evidence_ids=[f"ev-mixed-{run_id}-objective"],
    )
    balance_text = "; ".join(
        f"{row[1]}+{row[2]}-{row[3]}={row[4]}" for row in recurrence_rows
    )
    production_total = sum(
        float(values.get(f"production_{index}", 0.0))
        * float(plan.spec.production_costs[index])
        for index in range(horizon)
    )
    holding_total = sum(
        max(float(values.get(f"inventory_{index}", 0.0)), 0.0)
        * float(plan.spec.holding_costs[index])
        for index in range(horizon)
    )

    manifest = load_manifest(paths)
    data_paths = [
        record.path for record in manifest.inputs if record.source_role is SourceRole.DATA
    ]
    data_hashes = [
        sha256_file(paths.root / path) for path in data_paths if (paths.root / path).exists()
    ]

    contract_rows = [
        ["Plan hash", plan.plan_hash],
        ["Spec hash", plan.spec.spec_hash],
        ["DAG run", run_id],
        ["Artifact id", envelope.artifact_id],
        ["Artifact content hash", envelope.content_hash],
        ["Artifact payload hash", envelope.payload_hash],
        ["Artifact type / version", f"{envelope.artifact_type} v{envelope.schema_version}"],
        ["Forecast experiment", artifact.forecast_experiment_id],
        ["Optimization experiment", optimization_id],
        ["Input data path", ", ".join(data_paths) or "unavailable"],
        [
            "Raw file sha256 (recorded; not re-derivable here)",
            ", ".join(data_hashes) or "unavailable",
        ],
        [
            "Dataset fingerprint (recorded; not re-derivable here)",
            artifact.source_dataset_hash,
        ],
        [
            "Displayed-history digest (re-derivable from history table)",
            history_digest,
        ],
        ["Dependency contract", record.dependency_validation.get("status", "missing")],
        [
            "Pipeline provenance populated (internal flag)",
            str(record.provenance_complete).lower(),
        ],
        ["Model class", "linear program (continuous variables)"],
        ["Solver", f"{result.solver_backend} {result.solver_version}"],
        ["Solver status", solver.status],
        ["Solver termination", solver.termination_condition or "not reported"],
        [
            "Solver optimality gap (raw field)",
            _fmt(solver.optimality_gap, 6)
            if solver.optimality_gap is not None
            else "not reported",
        ],
    ]
    contract_table = PaperTable(
        table_id=f"tab-{run_id}-contract",
        caption=(
            "Integration audit. Rows are copied from frozen artifacts or the DAG run record. "
            "The displayed-history digest is re-derivable here: SHA-256 over the compact "
            "JSON array `[[period, demand], ...]` (separators ',' and ':', no trailing "
            "newline) built from the history table. The raw-file sha256 and manifest "
            "fingerprint are recorded but not re-derivable without the raw CSV and "
            "manifest, which are not attached. 'Pipeline provenance populated' is an "
            "internal population flag, not an external guarantee. 'Verified' means "
            "contract-complete."
        ),
        columns=["Check", "Value"],
        rows=contract_rows,
        evidence_ids=[
            f"ev-mixed-{run_id}-artifact-hash",
            f"ev-mixed-{run_id}-dependency-contract",
            f"ev-mixed-{run_id}-provenance",
        ],
    )

    objective_text = (
        _fmt(solver.objective_value, 6)
        if solver.objective_value is not None
        else "unavailable"
    )
    periods = list(plan.spec.planning_periods)
    forecasts = [float(value) for value in artifact.point_forecasts]
    productions = [
        float(values.get(f"production_{index}", 0.0)) for index in range(horizon)
    ]
    capacities = [float(value) for value in plan.spec.capacities]
    deviations = [
        index
        for index in range(horizon)
        if abs(productions[index] - forecasts[index]) > 1e-6
    ]
    binding = [
        index
        for index in range(horizon)
        if abs(productions[index] - capacities[index]) <= 1e-6
    ]
    binding_parts = [
        f"{periods[index]} (production {productions[index]:g} = capacity "
        f"{capacities[index]:g})"
        for index in binding
    ]
    prebuild = [index for index in deviations if index not in binding]
    prebuild_parts = [
        f"{periods[index]} (production {productions[index]:g} vs forecast "
        f"{forecasts[index]:g}, capacity {capacities[index]:g})"
        for index in prebuild
    ]
    if deviations:
        deviation_text = (
            "The plan deviates from the point forecast in "
            + ", ".join(periods[index] for index in deviations)
            + ". "
            + (
                "Capacity binds in " + ", ".join(binding_parts) + ". "
                if binding_parts
                else "No capacity binds. "
            )
            + (
                "The extra production is pre-built in "
                + ", ".join(prebuild_parts)
                + " so that inventory can carry into a capacity-constrained period. "
                if prebuild_parts
                else ""
            )
            + "This pre-build is a feasibility identity of the formulation (hard demand "
            "equality plus the capacity shortfall), not a behavioural result. It is a "
            "constraint-satisfaction sanity check on a hand-picked instance, not a "
            "planning-mechanism demonstration."
        )
    else:
        deviation_text = (
            "The plan equals the point forecast in every period, so the optimization node "
            "is a pass-through on this instance."
        )
    forecast_ratio_index = max(
        range(horizon), key=lambda index: forecasts[index] / capacities[index]
    )
    production_ratio_index = max(
        range(horizon), key=lambda index: productions[index] / capacities[index]
    )
    ratio_text = (
        f"The maximum point-forecast-to-capacity ratio is "
        f"{forecasts[forecast_ratio_index] / capacities[forecast_ratio_index]:.3f} in "
        f"{periods[forecast_ratio_index]} (forecast "
        f"{forecasts[forecast_ratio_index]:g} vs capacity "
        f"{capacities[forecast_ratio_index]:g}); the maximum realised "
        f"production-to-capacity ratio is "
        f"{productions[production_ratio_index] / capacities[production_ratio_index]:.3f} "
        f"in {periods[production_ratio_index]}."
    )

    drift_model = artifact.model_id
    baseline_model = "naive.last_value"
    drift_mae = forecast_result.metric_summaries.get(drift_model, {}).get("mae")
    baseline_mae = forecast_result.metric_summaries.get(baseline_model, {}).get("mae")
    if (
        drift_mae is not None
        and baseline_mae is not None
        and len(drift_mae.fold_values) == len(baseline_mae.fold_values)
        and drift_mae.fold_values
    ):
        wins = sum(
            drift < baseline
            for drift, baseline in zip(
                drift_mae.fold_values, baseline_mae.fold_values, strict=True
            )
        )
        losses = sum(
            drift > baseline
            for drift, baseline in zip(
                drift_mae.fold_values, baseline_mae.fold_values, strict=True
            )
        )
        gap = float(drift_mae.mean or 0.0) - float(baseline_mae.mean or 0.0)
        drift_mase = forecast_result.metric_summaries.get(drift_model, {}).get("mase")
        mase_text = ""
        if (
            drift_mase is not None
            and drift_mase.mean is not None
            and drift_mase.fold_values
        ):
            worst_fold = min(
                range(len(drift_mase.fold_values)),
                key=drift_mase.fold_values.__getitem__,
            )
            mase_text = (
                f" Mean MASE for `{drift_model}` is {float(drift_mase.mean):.4f}. MASE is "
                "relative to the per-fold in-sample random-walk scale, not to the "
                "out-of-sample last-value competitor, so this is a scale-ratio diagnostic "
                "rather than a comparative accuracy statement. The minimum fold MASE is "
                f"{drift_mase.fold_values[worst_fold]:.4f} on fold {worst_fold + 1}; that "
                "single below-1 fold is not presented as a general result."
            )
        signed_gaps = ", ".join(
            f"fold {index + 1}: {drift - baseline:+.4f}"
            for index, (drift, baseline) in enumerate(
                zip(drift_mae.fold_values, baseline_mae.fold_values, strict=True)
            )
        )
        drift_win_folds = ", ".join(
            str(index + 1)
            for index, (drift, baseline) in enumerate(
                zip(drift_mae.fold_values, baseline_mae.fold_values, strict=True)
            )
            if drift < baseline
        ) or "none"
        baseline_win_folds = ", ".join(
            str(index + 1)
            for index, (drift, baseline) in enumerate(
                zip(drift_mae.fold_values, baseline_mae.fold_values, strict=True)
            )
            if drift > baseline
        ) or "none"
        gap_values = [
            drift - baseline
            for drift, baseline in zip(
                drift_mae.fold_values, baseline_mae.fold_values, strict=True
            )
        ]
        gap_sd = float(np.std(gap_values, ddof=1)) if len(gap_values) > 1 else 0.0
        comparison_text = (
            f"Across {len(drift_mae.fold_values)} folds, `{drift_model}` has lower MAE "
            f"than `{baseline_model}` on {wins} folds and higher MAE on {losses} folds. "
            f"Signed per-fold MAE gaps (drift minus last-value): {signed_gaps}. The mean "
            f"gap computed from unrounded fold values is {gap:+.4f} and the sample "
            f"standard deviation of the four gaps is {gap_sd:.4f}; the near-zero mean "
            "results from large opposing fold errors, so it does not indicate that the "
            "models are similar. With four folds a 2-2 split is inconclusive (exact "
            "two-sided sign-test probability 0.375 under a fair coin) and a sign test on "
            "four folds has essentially no power; 'no superiority claimed' means the "
            "comparison is underpowered, not that the models are equivalent. The drift "
            f"model wins on folds {drift_win_folds} and the last-value baseline wins on "
            f"folds {baseline_win_folds}; all four folds are displayed.{mase_text} No "
            "paired significance test beyond this sign count is reported."
        )
    else:
        comparison_text = "No paired fold-level baseline comparison is available."

    min_train_size = len(history_payload) - plan.spec.n_splits * horizon
    fold_windows = [
        (
            f"train[0,{min_train_size + index * horizon}) -> "
            f"test[{min_train_size + index * horizon},"
            f"{min_train_size + (index + 1) * horizon})"
        )
        for index in range(plan.spec.n_splits)
    ]
    split_text = (
        f"With {len(history_payload)} observations, {plan.spec.n_splits} splits, and "
        f"horizon {horizon}, the rolling-origin folds are "
        + "; ".join(fold_windows)
        + "."
    )
    slope = (history_payload[-1][1] - history_payload[0][1]) / (
        len(history_payload) - 1
    )

    abstract = (
        f"This is a minimal feasibility demonstration on a synthetic, hand-built instance, "
        f"not a forecasting evaluation or a planning-capability study. The instance is not "
        "real demand data and was constructed so that one capacity binds once. Its only "
        "transferable content is that one artifact passed the frozen contract checks and "
        "one feasible plan point exists; it demonstrates neither blocking behavior, fault "
        f"detection, nor constraint enforcement. A hash-linked `{artifact.model_id}` "
        f"point-forecast artifact (`{envelope.artifact_id}`, content hash "
        f"`{envelope.content_hash[:16]}...`) was consumed through a contract-checked edge "
        "and used by a linear inventory-planning program solved with CBC. No accuracy or "
        f"planning-quality claim accompanies the contract checks. {deviation_text} The "
        "2026-02 bind occurs whenever that period's forecast is at least the frozen "
        "capacity of 140. By inspection only, with the last-value forecast 146 the "
        "shortfall would be 6, within the 24-unit 2026-01 headroom (170 - 146), so the "
        "constraint would still bind; this is arithmetic, not a solved plan, and no "
        "last-value objective is reported. The frozen plan leaves zero buffer: forecast "
        "150 vs capacity 140 is a 10-unit shortfall covered by 22 units of 2026-01 "
        "headroom, leaving 12 units (8% of 150) of residual margin. The model has no "
        "stock-out variable, and no demand perturbation, capacity sweep, or alternate "
        f"forecast was solved. The reported objective {objective_text} is production cost "
        f"at {plan.spec.production_costs[0]:g} cost units per unit plus holding cost at "
        f"{plan.spec.holding_costs[0]:g} cost units per held unit per period. The same "
        f"objective re-summed from the displayed rows is {recomputed_objective:.6f}; no "
        "second solver was run. Re-derivable from this report: the LP and balance "
        "arithmetic from the displayed history and parameters. The forecast artifact, "
        "metrics, manifest fingerprint, and raw-file sha256 are not re-derivable without "
        "the raw CSV and evaluation code, which are not attached. This single-instance "
        "plan is not a defensible inventory position, and no forecast-uncertainty "
        "propagation, robust planning, or general superiority claim is made."
    )

    sections = [
        PaperSection(
            heading="DAG and artifact contract",
            blocks=[
                "The plan contains exactly two nodes: a forecasting node that produces "
                "`demand_forecast` and an optimization node that requires it. The edge "
                "declares the required artifact type and schema version.",
                "The forecast artifact is accepted only after envelope type, schema version, "
                "validation status, payload hash, content hash, producer experiment identity, "
                "target name, unit, horizon, and timestamps all match the consumer contract. "
                "A missing, mutated, unvalidated, or misaligned artifact blocks the "
                "downstream node instead of silently falling back. This blocking behavior "
                "is a contract declaration in the frozen plan: no mutated, unvalidated, "
                "misaligned, or missing-artifact negative control was executed in this "
                "run. The run demonstrates one happy-path artifact transfer, not that "
                "fault injection is caught.",
            ],
        ),
        PaperSection(
            heading="Forecasting node",
            blocks=[
                f"The forecasting node ran a rolling-origin validation with "
                f"{plan.spec.n_splits} splits and horizon {horizon}; validation reports are "
                f"frozen before publication. The future point forecast is produced by "
                f"`{artifact.model_id}` from the ingested history. The same validation "
                "evaluates the `naive.last_value` baseline. Reported MAE/RMSE/MASE quantify "
                "measured error across folds; they do not certify accuracy on future periods.",
                split_text,
                f"The exact rule for `{artifact.model_id}` is "
                "`slope = (y_last - y_first) / (N - 1)` and "
                "`forecast_h = y_last + h * slope` for steps `h = 1..H`, using the entire "
                f"{len(history_payload)}-observation history window passed to the model. "
                f"Here y_first = {history_payload[0][1]:g} and "
                f"y_last = {history_payload[-1][1]:g}, so slope = {slope:.4f} "
                "per period. This average-slope rule does not use the last observed "
                "difference; a recent jump can therefore coexist with smaller forecast "
                "increments, which is why the forecast steps are not equal to the final "
                "observed difference.",
                comparison_text,
                "The future forecast is a deterministic point forecast, not a fitted "
                "probabilistic model; no prediction interval or scenario set is emitted. "
                "The term 'verified artifact' is used strictly for contract and hash "
                "verification.",
                f"Forecast target `{artifact.target_name}`, unit `{artifact.unit}`, source "
                f"dataset fingerprint `{artifact.source_dataset_hash}`; the displayed history "
                f"hashing to `{history_digest[:16]}...` is the exact window used.",
                "The fold-derivation table reconstructs each fold's training window, "
                "y_first, y_last, slope, step forecasts, and naive MASE scale from the "
                "displayed history. The choice of full-history drift rather than a "
                "last-difference or local slope is the frozen model's declared rule, not "
                "an empirically selected one; the capacity bind and pre-build magnitude "
                "are specific to this rule, and no alternate-slope forecast or plan was "
                "solved. Every fold shares the origin at the first observation, so "
                f"y_first = {history_payload[0][1]:g} is fixed and the slope changes only "
                "through the expanding window's last value and length. This single-anchor "
                "extrapolation is a documented limitation, not a plausible-model claim; no "
                "local-window slope diagnostic was run. The history window is all available "
                "observations (24 rows), a fixed rule rather than a selected value.",
                "Indexing convention and evaluation scope: fold k trains on indices [0,E) "
                "with E = 12, 15, 18, 21; its last training index is E-1 and its test "
                "window is [E, E+3). Fold 4 therefore uses index 20 (2025-09 = 145) as "
                "y_last, while the future forecast uses index 23 (2025-12 = 146). Fold 4's "
                "test window 2025-10..2025-12 lies inside the history used to fit the "
                "2026-01 point forecast, so fold-4 metrics are not held-out evidence for "
                "the plan's forecast; no true held-out fold was run.",
                "Worked MASE check: fold 1 trains on [0,12) with sum |Δ| = 44 over 11 "
                "differences, giving scale 4.0 and MASE 3.6667/4.0 = 0.9167. Per-fold "
                "scales are in the fold table; RMSE is the square root of the mean squared "
                "per-step errors, which follow from the displayed forecast and test-demand "
                "columns.",
                "By inspection only, window 15 gives a 2026-02 forecast of 140.0 (no "
                "shortfall, so production would equal the forecast) and window 18 gives "
                "140.2353 (about 0.24 units of pre-build). No LP was re-solved under "
                "either window.",
            ],
        ),
        PaperSection(
            heading="Production-planning node",
            blocks=[
                "Decision variables are continuous production and ending inventory per "
                "period. Each period enforces an inventory-balance equality and a production "
                "capacity upper bound; inventory is nonnegative. Initial inventory is "
                f"{plan.spec.initial_inventory:g} units. The objective minimizes "
                f"production cost at {plan.spec.production_costs[0]:g} cost units per unit "
                f"plus holding cost at {plan.spec.holding_costs[0]:g} cost units per held "
                "unit per period over the frozen horizon.",
                "The balance is a hard equality with no stock-out or backorder variable: "
                f"for the displayed plan it reads {balance_text}. The 2026-02 deficit of "
                "10 units is covered exactly by the 2026-01 pre-build, so ending inventory "
                "in 2026-02 is zero and no buffer remains; the instance is on a feasibility "
                "knife-edge, not a margin. The pre-build is forced by the equality and the "
                "shortfall, not chosen by a cost trade-off. Re-summing cost from the "
                f"displayed rows gives production {production_total:.6f} + holding "
                f"{holding_total:.6f} = {recomputed_objective:.6f}; this is an arithmetic "
                "re-sum, not a second-solver verification.",
                "Terminal inventory is unconstrained: there is no salvage value and no "
                "terminal stock requirement. Initial inventory is 0. With constant "
                "production cost and positive holding cost, the optimizer carries only the "
                "10 units needed for feasibility and ends with zero inventory. A model with "
                "terminal-stock requirements would produce a different objective and was "
                "not solved.",
                f"Diagnostic bookkeeping with no operational meaning: {ratio_text} The "
                "capacity of 140 is a frozen spec parameter, and production equaling "
                "capacity in 2026-02 is forced by the hard demand equality plus the "
                "shortfall, not a planner utilization result.",
                "Because all variables are continuous, this node is a linear program, not a "
                "mixed-integer program. CBC reported status "
                f"`{solver.status}` with termination "
                f"`{solver.termination_condition or 'not reported'}`. The independent "
                "validator re-checks inventory balance, capacities, nonnegativity, and the "
                "objective rather than trusting the solver label. This three-period LP is "
                "small enough to solve by inspection; a zero optimality gap is expected by "
                "construction and is not evidence of solver or validator quality.",
                deviation_text,
                "Capacity binds only where the displayed production equals the displayed "
                "capacity. In other periods the bound is slack. The instance was chosen by "
                "hand to make one capacity bind; this shows only that the balance and "
                "capacity constraints are enforced, not that a planning mechanism produced "
                "a non-trivial decision. Production and holding cost rates are constant "
                "across periods and the forecast is monotone; with slack capacity "
                "everywhere the plan would equal the forecast, and with binding capacity "
                "everywhere the instance could be infeasible. No capacity or cost sweep was "
                "run. A single passing solve on this knife-edge instance cannot distinguish "
                "a correctly enforced capacity bound from a missing or mis-indexed one: "
                "because 158 <= 170 and 140 = 140, a solver that ignored the bound would "
                "also accept the returned point. The deliberately infeasible and "
                "slack-capacity companion solves - 2026-02 capacity raised to 150, "
                "2026-01 capacity lowered to 150, and a last-value-forecast plan - were "
                "not run and are the main open check. No +/-5 or +/-10 unit demand "
                "perturbation was solved. Since there is no stock-out variable, a 2026-02 "
                "demand surprise above the 12-unit residual margin (8% of 150) would exceed "
                "the period-1 headroom and make the hard-equality LP infeasible.",
            ],
        ),
        PaperSection(
            heading="Internal consistency check",
            blocks=[
                "The mixed validator re-verifies the plan hash, DAG completion, artifact "
                "contract and hashes, upstream validation and evidence, the artifact identity "
                "recorded on the optimization experiment, every inventory-balance constraint "
                "against the artifact point forecasts, capacity and nonnegativity bounds, the "
                "objective value, and provenance completeness.",
                "The validator recomputes the same constraints from the frozen variable values "
                "through a separate code path. Because the optimizer and validator share the "
                "frozen plan and artifact, this is internal consistency checking, not "
                "independent third-party validation, and it does not validate the model "
                "formulation. On this instance the objective is hand-checkable: "
                "158 + 140 + 152 = 450 produced units at 12 each plus 10 held units at 2, "
                "for 5420. This is an arithmetic-only check: it would not catch a capacity "
                "attached to the wrong period, an artifact forecast off by a constant, a "
                "unit mismatch between the artifact and the spec, or a solver bug that "
                "returns a feasible non-optimal point.",
            ],
        ),
        PaperSection(
            heading="Limitations",
            blocks=[
                "One small frozen synthetic instance is exercised. The pre-build is forced "
                "by the hard demand equality and the capacity shortfall; there is no sweep "
                "over costs, capacities, or demand shapes, so no planning capability is "
                "claimed. The plan must not be read as a defensible inventory position: "
                "the drift model's signed MAE gaps flip sign across folds (-4.0000, "
                "+4.3333, -4.2353, +3.8333) and are large in absolute value, and the cost "
                "of forecast error is not quantified. The bind is also not robust to "
                "window choice: the 15-observation window gives a 2026-02 forecast of "
                "exactly 140 and no pre-build, and the 18-observation window gives 140.2353 "
                "and about 0.24 units of pre-build. Window 24 and slope 2.0 are load-bearing "
                "for the reported bind.",
                "The first mixed slice supports exactly one forecasting-to-optimization path "
                "with temporal baselines only. Neural forecasters, probabilistic forecasts, "
                "multi-echelon networks, stochastic optimization, and simulation feedback "
                "loops are untested.",
                "Demand uncertainty is not propagated. A deterministic point forecast can "
                "understate the cost of forecast error; results should not be read as a "
                "robust or risk-aware plan. Because the point forecast is used as if it were "
                "certain, the objective is a conditional cost, not an expected cost.",
                "The drift model is declared but not empirically validated: it was not "
                "shown to be more accurate than the last-value baseline (2-of-4 folds, "
                "mean MAE gap not distinguishable from fold noise) and mean MASE exceeds 1. "
                "No plan was solved against the last-value forecast, so the dependence of "
                "the capacity bind and objective on forecast choice is unquantified. No "
                "alternate-slope plan, cost sweep, or capacity sweep was run.",
                "Three different digests appear because they hash different objects: the "
                "raw file sha256 hashes the CSV bytes; the dataset fingerprint hashes the "
                "project manifest inputs; the displayed-history digest is SHA-256 over the "
                "compact JSON array of (period, demand) pairs with no trailing newline. "
                "The forecasting node consumed the displayed 24-row window; the optimizer "
                "consumed the artifact point forecasts. The raw CSV is not attached.",
                "The hashes in the contract table establish internal provenance and hash "
                "linkage only. The displayed series and parameters are sufficient to "
                "re-solve the LP and re-check the balance recurrence. The forecast "
                "artifact, evaluation metrics, manifest fingerprint, and raw-file sha256 "
                "cannot be re-derived from this report without the raw CSV and the "
                "evaluation code, which are not attached; no external reproduction was "
                "performed. The rolling-origin split boundaries and the drift rule are "
                "stated above; the split definitions follow Bergmeir and Benítez (2012) "
                "and Hyndman and Athanasopoulos (2021).",
            ],
        ),
    ]

    return PaperAST(
        title="Forecast-to-Planning Contract Integration Report — Frozen Single Slice",
        abstract=abstract,
        sections=sections,
        tables=[
            history_table,
            forecast_table,
            metrics_table,
            fold_table,
            window_table,
            parameter_table,
            planning_table,
            recurrence_table,
            contract_table,
        ],
        figures=[
            _plot_dag(paths, plan, run_id),
            _plot_plan(paths, plan, artifact, values, run_id),
        ],
        claims=[],
        references=[
            "Bergmeir, C. & Benítez, J.M. (2012). On the use of cross-validation for time "
            "series predictor evaluation. Information Sciences 191, 192-213.",
            "Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and "
            "Practice, 3rd edition. OTexts.",
            "Perron, L. & Furnon, V. OR-Tools 9.15 with the CBC backend. Google.",
        ],
    )


__all__ = ["build_mixed_paper"]
