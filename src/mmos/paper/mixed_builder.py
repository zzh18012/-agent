import hashlib
import json
import shutil
import statistics
from pathlib import Path

import numpy as np
import polars as pl

from mmos.core.project import (
    ProjectPaths,
    load_manifest,
    load_optimization_result,
    sha256_file,
)
from mmos.data.table import load_single_table
from mmos.experiments.engines import forecast_baseline
from mmos.mixed.artifacts import load_envelope
from mmos.mixed.simulation import (
    load_mixed_simulation_result,
    load_mixed_simulation_spec,
)
from mmos.mixed.validation import run_record_path
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.mixed import (
    DAGRunRecord,
    ForecastArtifact,
    ForecastErrorArtifact,
    MixedExecutionPlan,
    ProductionPlanArtifact,
)
from mmos.schemas.mixed_simulation import (
    MixedSimulationResult,
    MixedSimulationSpec,
)
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
            "publishes typed, hashed demand-forecast and forecast-error artifacts; the "
            "optimization node consumes the forecast; the simulation node consumes the "
            "production plan and forecast-error artifacts. The arrow labels are required "
            "artifact types and schema versions, not runtime quantities. This figure "
            "records one happy-path transfer only; no mutated, missing, or misaligned "
            "artifact was injected, so it demonstrates no gate and no fault tolerance."
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
    simulation_result: MixedSimulationResult | None = None,
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

    if simulation_result is not None:
        figure, (axis, simulation_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        simulation_axis = None
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
    if simulation_result is not None and simulation_axis is not None:
        demand_means = [
            sum(
                replication.periods[index].simulated_demand
                for replication in simulation_result.replications
            )
            / simulation_result.n_replications
            for index in range(len(positions))
        ]
        demand_lows = [
            float(
                np.percentile(
                    [
                        replication.periods[index].simulated_demand
                        for replication in simulation_result.replications
                    ],
                    5,
                )
            )
            for index in range(len(positions))
        ]
        demand_highs = [
            float(
                np.percentile(
                    [
                        replication.periods[index].simulated_demand
                        for replication in simulation_result.replications
                    ],
                    95,
                )
            )
            for index in range(len(positions))
        ]
        inventory_means = [
            sum(
                replication.periods[index].ending_inventory
                for replication in simulation_result.replications
            )
            / simulation_result.n_replications
            for index in range(len(positions))
        ]
        simulation_axis.bar(
            positions,
            productions,
            0.4,
            label="Frozen production",
            color="#dd8452",
        )
        simulation_axis.errorbar(
            positions,
            demand_means,
            yerr=[
                [mean - low for mean, low in zip(demand_means, demand_lows, strict=True)],
                [
                    high - mean
                    for mean, high in zip(demand_means, demand_highs, strict=True)
                ],
            ],
            fmt="D",
            color="#4c72b0",
            capsize=3,
            label="Simulated demand mean (p05-p95)",
        )
        simulation_axis.plot(
            positions,
            inventory_means,
            marker="o",
            color="#55a868",
            label="Mean simulated ending inventory",
        )
        simulation_axis.set_xticks(positions, periods)
        simulation_axis.set_title("Frozen production vs simulated demand")
        simulation_axis.legend(loc="best", fontsize=8)
        simulation_axis.grid(True, axis="y", alpha=0.3)
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
            "Forecast, capacity, production, and ending inventory by planning period. "
            "Bars are the frozen point forecast, the solved production quantity, and the "
            "frozen capacity; the line is deterministic ending inventory, and annotations "
            "are the deterministic recurrence opening + production - demand = closing. "
            + (
                "Left panel: deterministic plan only. Right panel: frozen production with "
                "the simulated demand mean and p05-p95 band plus mean simulated ending "
                "inventory; this separates the deterministic point path, which has zero "
                "shortage, from the simulated demand distribution and does not depict the "
                "band as a deterministic plan failure. The band is a finite-resampling "
                "band whose center is offset below the point forecast by the pool mean "
                "(about -1.27 units per period, -3.80 over the horizon). The band is "
                "residuals from fold slopes that are not the deployed slope, not forecast "
                "uncertainty around the deployed point forecast; the bars are the frozen "
                "production plan. "
                if simulation_result is not None
                else ""
            )
            + (f"Production equals capacity in: {binding_text}. " if binding_text else "")
            + "This equality is arithmetic of the returned point, not an enforcement "
            "finding; no non-binding companion solve exists in this run. The 10-unit "
            "pre-build is the cost-minimizing member of the feasible [0, 22] pre-build "
            "range, not a unique requirement."
        ),
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=[
            f"ev-mixed-{run_id}-forecast-points",
            f"ev-mixed-{run_id}-objective",
        ],
    )


def _write_evidence_bundle(
    paths: ProjectPaths,
    plan: MixedExecutionPlan,
    run_id: str,
    record: DAGRunRecord,
    error_artifact: ForecastErrorArtifact,
) -> tuple[str, str, int]:
    bundle = paths.root / "paper" / "evidence_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    inputs = bundle / "inputs"
    inputs.mkdir(exist_ok=True)

    def copy_if_exists(source: Path, relative: str) -> None:
        if not source.exists():
            return
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest = load_manifest(paths)
    for entry in manifest.inputs:
        copy_if_exists(paths.root / entry.path, f"inputs/{entry.path}")
    copy_if_exists(paths.config_file, "inputs/project.yaml")
    copy_if_exists(paths.manifest_file, "inputs/project_manifest.json")
    copy_if_exists(paths.problem_spec_file, "inputs/problem_spec.json")
    copy_if_exists(paths.mixed_dir / "plan.json", "inputs/mixed_plan.json")
    (inputs / "mixed_spec.json").write_text(
        plan.spec.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    run_dir = paths.mixed_dir / "runs" / run_id
    copy_if_exists(run_dir / "dag_run.json", "run/dag_run.json")
    for name in (
        "forecast_artifact.json",
        "forecast_envelope.json",
        "forecast_error_artifact.json",
        "forecast_error_envelope.json",
        "production_plan_artifact.json",
        "production_plan_envelope.json",
    ):
        copy_if_exists(run_dir / "artifacts" / name, f"run/artifacts/{name}")
    residual_csv = bundle / "residuals.csv"
    residual_csv.write_text(
        "\n".join(
            ["fold_id,row_index,timestamp,actual,forecast,residual"]
            + [
                f"{item.fold_id},{item.row_index},{item.timestamp},"
                f"{item.actual},{item.forecast},{item.residual}"
                for item in error_artifact.residuals
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    forecast_node = next(
        (
            node
            for node in record.nodes
            if node.task_family.value == "forecasting"
        ),
        None,
    )
    forecast_id = forecast_node.experiment_id if forecast_node else None
    if forecast_id is not None:
        exp_dir = paths.experiment_dir(forecast_id)
        copy_if_exists(exp_dir / "experiment_spec.json", "experiments/forecast/spec.json")
        copy_if_exists(
            exp_dir / "artifacts" / "metrics.json", "experiments/forecast/metrics.json"
        )
        copy_if_exists(
            exp_dir / "artifacts" / "predictions.parquet",
            "experiments/forecast/predictions.parquet",
        )
        for report in (exp_dir / "validation").glob("*.json"):
            copy_if_exists(report, f"experiments/forecast/validation/{report.name}")
    if record.optimization_experiment_id is not None:
        exp_dir = paths.experiment_dir(record.optimization_experiment_id)
        copy_if_exists(
            exp_dir / "optimization_spec.json", "experiments/optimization/spec.json"
        )
        copy_if_exists(
            exp_dir / "artifacts" / "optimization_result.json",
            "experiments/optimization/result.json",
        )
        copy_if_exists(
            exp_dir / "experiment_record.json",
            "experiments/optimization/record.json",
        )
        for report in (exp_dir / "validation").glob("*.json"):
            copy_if_exists(report, f"experiments/optimization/validation/{report.name}")
    if record.simulation_experiment_id is not None:
        exp_dir = paths.experiment_dir(record.simulation_experiment_id)
        copy_if_exists(
            exp_dir / "mixed_simulation_spec.json", "experiments/simulation/spec.json"
        )
        copy_if_exists(
            exp_dir / "artifacts" / "mixed_simulation_result.json",
            "experiments/simulation/result.json",
        )
        copy_if_exists(
            exp_dir / "artifacts" / "mixed_simulation_replications.csv",
            "experiments/simulation/replications.csv",
        )
        copy_if_exists(
            exp_dir / "mixed_simulation_record.json",
            "experiments/simulation/record.json",
        )
        copy_if_exists(
            exp_dir / "mlflow_run.json", "experiments/simulation/mlflow_run.json"
        )
        for report in (exp_dir / "validation").glob("*.json"):
            copy_if_exists(report, f"experiments/simulation/validation/{report.name}")
    copy_if_exists(
        paths.verified_dir / "mixed" / f"{run_id}.json", "evidence/mixed_evidence.json"
    )
    copy_if_exists(paths.verified_dir / "provenance.json", "evidence/provenance.json")
    repository_root = Path(__file__).resolve().parents[3]
    if (repository_root / "src" / "mmos").exists():
        shutil.copytree(
            repository_root / "src" / "mmos",
            bundle / "code" / "mmos",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    if (repository_root / "workflow").exists():
        shutil.copytree(
            repository_root / "workflow",
            bundle / "code" / "workflow",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    copy_if_exists(repository_root / "pyproject.toml", "code/pyproject.toml")

    (bundle / "README.md").write_text(
        "# Evidence bundle\n\n"
        f"Frozen artifacts for mixed run `{run_id}`. Contains raw inputs, the frozen "
        "plan and spec, forecast/optimization/simulation experiment specs and outputs, "
        "the rolling-origin residual pool, the replication CSV, DAG evidence, and the "
        "evaluation/simulation code. `SHA256SUMS` lists every file hash; verify with "
        "`sha256sum -c SHA256SUMS` from this directory.\n",
        encoding="utf-8",
    )
    files = sorted(
        path
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(bundle).as_posix()}"
        for path in files
    ]
    sums = bundle / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (
        bundle.relative_to(paths.root).as_posix(),
        hashlib.sha256(sums.read_bytes()).hexdigest(),
        len(files) + 1,
    )


def _plot_simulation(
    paths: ProjectPaths, result: MixedSimulationResult, run_id: str
) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper_dir = paths.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    target = paper_dir / "simulation_outcomes.png"

    shortages = [replication.total_shortage for replication in result.replications]
    fills = [replication.fill_rate for replication in result.replications]
    figure, (shortage_axis, fill_axis) = plt.subplots(1, 2, figsize=(10, 4))
    shortage_axis.hist(shortages, bins=min(30, max(5, len(shortages) // 5)), color="#c44e52")
    shortage_axis.axvline(
        result.metrics["total_shortage"].mean, color="#333333", lw=1.2, linestyle="--"
    )
    shortage_axis.set_title("Total shortage per replication")
    shortage_axis.set_xlabel(f"Shortage ({result.unit})")
    shortage_axis.set_ylabel("Replications")
    fill_axis.hist(fills, bins=min(30, max(5, len(fills) // 5)), color="#4c72b0")
    fill_axis.axvline(
        result.metrics["fill_rate"].mean, color="#333333", lw=1.2, linestyle="--"
    )
    fill_axis.set_title("Fill rate per replication")
    fill_axis.set_xlabel("Fill rate (served / demand)")
    fill_axis.set_ylabel("Replications")
    figure.suptitle(
        f"Frozen-plan conditional simulation: {result.n_replications} residual resamples"
    )
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return PaperFigure(
        figure_id="simulation-outcomes",
        caption=(
            "Empirical Monte Carlo histograms of total shortage and fill rate across the "
            "configured replications. Dashed lines are the sample means. These are finite-"
            "sample summaries of the frozen empirical resampling protocol, not estimates of "
            "a true demand distribution or guarantees about future service."
        ),
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=[
            f"ev-mixed-{run_id}-simulation-shortage",
            f"ev-mixed-{run_id}-simulation-fill-rate",
            f"ev-mixed-{run_id}-simulation-replications",
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
    plan_artifact: ProductionPlanArtifact | None = None
    error_artifact: ForecastErrorArtifact | None = None
    simulation_spec: MixedSimulationSpec | None = None
    simulation_result: MixedSimulationResult | None = None
    bundle_path = ""
    bundle_hash = ""
    bundle_files = 0
    if plan.has_simulation_node():
        if record.simulation_experiment_id is None:
            raise ValueError("mixed run has no simulation experiment")
        artifacts_dir = paths.mixed_dir / "runs" / run_id / "artifacts"
        plan_envelope = load_envelope(artifacts_dir / "production_plan_envelope.json")
        plan_artifact = ProductionPlanArtifact.model_validate_json(
            (paths.root / plan_envelope.payload_path).read_text(encoding="utf-8")
        )
        error_envelope = load_envelope(artifacts_dir / "forecast_error_envelope.json")
        error_artifact = ForecastErrorArtifact.model_validate_json(
            (paths.root / error_envelope.payload_path).read_text(encoding="utf-8")
        )
        simulation_spec = load_mixed_simulation_spec(
            paths, record.simulation_experiment_id
        )
        simulation_result = load_mixed_simulation_result(
            paths, record.simulation_experiment_id
        )
        bundle_path, bundle_hash, bundle_files = _write_evidence_bundle(
            paths, plan, run_id, record, error_artifact
        )

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
            if model_id == "naive.last_value" and metric_name == "mase":
                continue
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
            "evaluation. The mean column is an unweighted mean over folds, not a pooled "
            "MAE. Lower MAE/RMSE is better. MASE uses the one-step in-sample "
            "random-walk scale; the `naive.last_value` MASE row is omitted because that "
            "denominator is its own one-step error, making the value tautological. Drift "
            "MASE is provenance only and no cross-model MASE comparison or model ranking "
            "is claimed. These numbers quantify measured error, not the accuracy of the "
            "future point forecast."
        ),
        columns=[
            "Model",
            "Metric",
            "Unweighted fold mean (4 folds; fold 4 in-sample)",
            "Folds",
            "Fold values",
        ],
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
            "ending inventory are all in the same unit. Costs are in dimensionless "
            "normalized cost units: production cost is 12 normalized cost units per "
            "produced unit and holding cost is 2 per unit held for one period; all "
            "objective values use the same normalized currency."
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
    history_digest_alt = hashlib.sha256(
        json.dumps(
            [[period, f"{demand:.4f}"] for period, demand in history_payload],
            separators=(",", ":"),
        ).encode("utf-8")
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
    fold_scales: list[float] = []
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
        fold_scales.append(naive_scale)
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
                _fmt(naive_scale, 6),
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
            "evaluation. MASE uses lag l=1: the per-fold in-sample one-step scale "
            "sum(|y_t - y_{t-1}|)/(N-1), not the out-of-sample last-value competitor. "
            "The scale column is printed so MASE is reproducible; fold 2's scale is "
            "4.071429, and 7.0000/4.071429 = 1.7193. The recomputed MAE values are checked "
            "against the frozen evaluation."
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
            "MASE scale (l=1)",
            "MASE",
        ],
        rows=fold_rows,
        evidence_ids=[
            f"ev-{artifact.forecast_experiment_id}-evaluation-{artifact.model_id}-mase-details",
            f"ev-mixed-{run_id}-forecast-history-digest",
        ],
    )

    window_rows: list[list[str]] = []
    plan_window_rows: list[list[str]] = []
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
        capacity_second = float(plan.spec.capacities[1])
        prebuild = max(0.0, float(window_forecast[1]) - capacity_second)
        projected_production = [
            float(window_forecast[0]) + prebuild,
            max(0.0, float(window_forecast[1]) - prebuild),
            float(window_forecast[2]),
        ]
        plan_window_rows.append(
            [
                str(window_length),
                *[_fmt(float(value), 4) for value in window_forecast],
                _fmt(max(0.0, float(window_forecast[1]) - capacity_second), 4),
                _fmt(prebuild, 4),
                *[_fmt(value, 4) for value in projected_production],
                "yes (projection only)"
                if abs(projected_production[1] - capacity_second) <= 1e-6
                else "no",
            ]
        )
    window_table = PaperTable(
        table_id=f"tab-{run_id}-window-sensitivity",
        caption=(
            "Slope and H-step forecast as a function of the history window length under "
            "the declared drift rule, with the anchor fixed at the first observation. "
            "The slope is window-dependent; the 24-observation slope of 2.0000 happens to "
            "equal the 12-observation slope, which is an arithmetic coincidence, not "
            "stability. Window 24 is not selected by an out-of-sample criterion; all five "
            "rows are equally defensible sensitivity results, and no production plan was "
            "solved for the other windows. The reported plan's pre-build is therefore a "
            "function of an arbitrary window choice."
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
    plan_window_table = PaperTable(
        table_id=f"tab-{run_id}-window-plan-sensitivity",
        caption=(
            "Arithmetic projection of the production plan under each equally defensible "
            "window, using the same frozen capacities and the balance identity. This is "
            "not a re-solve of the LP: it shows forecast, 2026-02 shortfall versus "
            "capacity 140, the implied pre-build, and the resulting production schedule. "
            "Only window 24 is the reported plan. Windows 15 and 18 have no pre-build, "
            "but their arithmetic projections still equal capacity because the forecast "
            "itself is at or near 140; that flag is not a bind indicator. Windows 21 and "
            "24 have positive pre-builds."
        ),
        columns=[
            "Window length",
            *[f"Forecast h{h}" for h in range(1, horizon + 1)],
            "2026-02 shortfall",
            "Projected pre-build",
            *[f"Projected production h{h}" for h in range(1, horizon + 1)],
            "Projection equals capacity (not a bind indicator)",
        ],
        rows=plan_window_rows,
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
        [
            "Solver",
            f"{result.solver_backend} backend via OR-Tools {result.solver_version}; "
            "underlying CBC solver version not recorded",
        ],
        [
            "Solver status (recorded metadata; optimal = feasibility + balance identity)",
            solver.status,
        ],
        ["Solver termination", solver.termination_condition or "not reported"],
        [
            "Solver optimality gap (raw field; not a cost-optimization result)",
            _fmt(solver.optimality_gap, 6)
            if solver.optimality_gap is not None
            else "not reported",
        ],
    ]
    contract_table = PaperTable(
        table_id=f"tab-{run_id}-contract",
        caption=(
            "Integration audit. Rows are copied from frozen artifacts or the DAG run record. "
            "Canonical displayed-history bytes: `json.dumps` of "
            "`[[period_string, float_demand], ...]` with separators ',' and ':' and no "
            "trailing newline, e.g. `[[\"2024-01\",100.0],...]`; this reproduces the listed "
            "digest. As a serialization check, rendering demand as a four-decimal string "
            f"instead would give `{history_digest_alt}` (different), confirming the digest is "
            "byte-level specified. The raw-file sha256 and manifest fingerprint are recorded "
            "but not re-derivable from the report text alone; for three-node runs they are "
            "included in `paper/evidence_bundle`. "
            "'Pipeline provenance populated' is an internal population flag, not an external "
            "guarantee. 'Verified' is a pipeline label only: payload/content hashes, "
            "dependency-contract 'passed', and the provenance flag are not recomputable "
            "from this text, and no hash or contract check has been independently verified."
        ),
        columns=["Check", "Value"],
        rows=contract_rows,
        evidence_ids=[
            f"ev-mixed-{run_id}-artifact-hash",
            f"ev-mixed-{run_id}-dependency-contract",
            f"ev-mixed-{run_id}-provenance",
        ],
    )

    simulation_tables: list[PaperTable] = []
    simulation_figures: list[PaperFigure] = []
    simulation_abstract = ""
    simulation_sections: list[PaperSection] = []
    if (
        simulation_result is not None
        and simulation_spec is not None
        and error_artifact is not None
        and plan_artifact is not None
    ):
        residual_rows = [
            [
                str(item.fold_id + 1),
                item.timestamp,
                _fmt(item.actual, 4),
                _fmt(item.forecast, 4),
                _fmt(item.residual, 4),
            ]
            for item in error_artifact.residuals
        ]
        residual_table = PaperTable(
            table_id=f"tab-{run_id}-residuals",
            caption=(
                f"Fixed-anchor historical residual pool used by the simulation. "
                f"Residual definition: {error_artifact.residual_definition.replace('_', ' ')}. "
                f"Source label: {error_artifact.residual_source}. Sample count: "
                f"{error_artifact.sample_count}; mean {error_artifact.mean:.6f}; sample std "
                f"{error_artifact.std:.6f}. Only rolling-origin test rows are included; "
                "training rows are not used. No row is held out with respect to the "
                "deployed forecast, which uses the full history."
            ),
            columns=["Fold", "Test period", "Actual", "Forecast", "Residual"],
            rows=residual_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-forecast-error-id",
                f"ev-mixed-{run_id}-forecast-error-hash",
                f"ev-mixed-{run_id}-forecast-error-source",
                f"ev-mixed-{run_id}-forecast-error-samples",
            ],
        )
        residual_by_fold: dict[int, list[float]] = {}
        for item in error_artifact.residuals:
            residual_by_fold.setdefault(item.fold_id, []).append(item.residual)
        residual_diagnostic_rows = [
            [
                str(fold_id + 1),
                str(len(values)),
                _fmt(statistics.fmean(values), 6),
                _fmt(statistics.stdev(values) if len(values) > 1 else 0.0, 6),
                _fmt(min(values), 6),
                _fmt(max(values), 6),
            ]
            for fold_id, values in sorted(residual_by_fold.items())
        ]
        residual_diagnostics_table = PaperTable(
            table_id=f"tab-{run_id}-residual-diagnostics",
            caption=(
                "Per-fold fixed-anchor historical residual diagnostics. Folds 1-3 "
                "contain rows that are out-of-sample for their own fold models; all 12 "
                "rows, including fold 4, lie inside the full history used for the final "
                "2026 point forecast, so none is held out with respect to the deployed "
                "forecast. A fold-1-3-only rerun was not performed. The max(0, demand) "
                "floor activation count in "
                f"the recorded simulation is {simulation_result.demand_floor_activations}."
            ),
            columns=["Fold", "Residual count", "Mean", "Std", "Min", "Max"],
            rows=residual_diagnostic_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-forecast-error-samples",
                f"ev-mixed-{run_id}-forecast-error-mean",
                f"ev-mixed-{run_id}-simulation-floor-activations",
            ],
        )
        metric_definitions = {
            "total_shortage": "sum over periods of max(0, demand - available)",
            "shortage_period_count": "periods with shortage > 0",
            "fill_rate": "total served / total demand",
            "ending_inventory": "inventory after the final period",
            "average_inventory": "mean ending inventory across periods",
            "max_shortage": "maximum single-period shortage",
        }
        metrics_rows = []
        for metric in simulation_spec.metrics:
            summary = simulation_result.metrics[metric]
            metrics_rows.append(
                [
                    metric,
                    metric_definitions.get(metric, "defined in the frozen spec"),
                    _fmt(summary.mean, 3),
                    _fmt(summary.std, 3),
                    _fmt(summary.median, 3),
                    _fmt(summary.p05, 3),
                    _fmt(summary.p25, 3),
                    _fmt(summary.p75, 3),
                    _fmt(summary.p95, 3),
                ]
            )
        simulation_metrics_table = PaperTable(
            table_id=f"tab-{run_id}-simulation-metrics",
            caption=(
                "Internal protocol check: simulation outcome summaries over "
                f"{simulation_result.n_replications} replications. No mean interval is "
                "reported: Student-t intervals over "
                "replications would be Monte Carlo integration intervals, not path-"
                "population intervals, and were removed. Only "
                f"{simulation_result.distinct_residual_paths} distinct residual paths were "
                "observed, out of at most "
                f"{error_artifact.sample_count ** simulation_spec.planning_horizon} "
                "with-replacement paths, so the effective sample is smaller than the "
                "replication count. Quantiles are sample quantiles. The any-shortage "
                f"probability is {simulation_result.any_shortage_probability:.6f}, an "
                "empirical finite-Monte-Carlo estimate on a discrete path grid, not a "
                "guarantee. No pool-level or mean interval is reported, and the p95 is a "
                "sample order statistic with no interval. The fill rate has no comparator "
                "(no buffered, safety-stock, or alternative plan was simulated), so it has "
                "no evaluative scale. "
                "Fill-rate and shortage intervals are not "
                "derivable from one another: per-replication demand covaries with "
                "shortage, and no delta-method or bootstrap ratio interval was computed. "
                "The fill-rate distribution has a mass point at 1.0 and the ending-"
                "inventory distribution is zero-inflated; the printed intervals are mean "
                "intervals and do not describe their central or modal behavior. "
                "Within a path, period shortages are positively coupled through shared "
                "opening inventory. Total shortage is extremely right-skewed because of "
                "the 12-point residual grid: its mean is tail-dominated, while the median "
                "and lower quantiles can be zero. No quantile-uncertainty interval was "
                "computed for the p95."
            ),
            columns=[
                "Metric",
                "Definition",
                "Mean",
                "Std",
                "Median",
                "p05",
                "p25",
                "p75",
                "p95",
            ],
            rows=metrics_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-simulation-shortage",
                f"ev-mixed-{run_id}-simulation-fill-rate",
                f"ev-mixed-{run_id}-simulation-any-shortage",
            ],
        )
        path_space = error_artifact.sample_count ** simulation_spec.planning_horizon
        expected_distinct = (
            path_space * (1 - (1 - 1 / path_space) ** simulation_result.n_replications)
            if path_space > 1
            else 1.0
        )
        protocol_rows = [
            ["Point forecast artifact", simulation_spec.forecast_artifact_id],
            ["Forecast-error artifact", simulation_spec.forecast_error_artifact_id],
            ["Production plan artifact", simulation_spec.production_plan_artifact_id],
            ["Planning horizon", str(simulation_spec.planning_horizon)],
            ["Timestamps", ", ".join(simulation_spec.timestamps)],
            ["Unit", simulation_spec.unit],
            ["Replications", str(simulation_spec.replications)],
            ["Master seed", str(simulation_spec.master_seed)],
            ["Replication seed hash", simulation_result.replication_seed_hash],
            ["Residual sampling", simulation_spec.residual_sampling_method],
            ["Demand floor policy", simulation_spec.demand_floor_policy],
            [
                "Inventory semantics",
                "lost sales: served=min(available,demand); shortage=max(0,demand-available)",
            ],
            ["Production plan", "frozen; never re-optimized during simulation"],
        ]
        protocol_table = PaperTable(
            table_id=f"tab-{run_id}-simulation-protocol",
            caption=(
                "Simulation protocol. Each replication samples H residuals with "
                "replacement from the frozen out-of-sample pool, forms simulated demand as "
                "max(0, point forecast + residual), and evaluates the frozen production "
                "plan with lost-sales inventory dynamics. The seed schedule reuses the "
                "formal sha256 derivation."
            ),
            columns=["Parameter", "Value"],
            rows=protocol_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-simulation-experiment",
                f"ev-mixed-{run_id}-simulation-seed-hash",
                f"ev-mixed-{run_id}-simulation-replications",
            ],
        )
        period_rows = []
        sample_count = simulation_result.n_replications
        for index, timestamp in enumerate(simulation_spec.timestamps):
            mean_demand = sum(
                replication.periods[index].simulated_demand
                for replication in simulation_result.replications
            ) / simulation_result.n_replications
            mean_shortage = sum(
                replication.periods[index].shortage
                for replication in simulation_result.replications
            ) / simulation_result.n_replications
            opening_values = [
                (
                    0.0
                    if index == 0
                    else replication.periods[index - 1].ending_inventory
                )
                for replication in simulation_result.replications
            ]
            mean_opening = sum(opening_values) / simulation_result.n_replications
            mean_served = sum(
                replication.periods[index].served
                for replication in simulation_result.replications
            ) / simulation_result.n_replications
            inventory_values = [
                replication.periods[index].ending_inventory
                for replication in simulation_result.replications
            ]
            mean_inventory = sum(inventory_values) / sample_count
            period_rows.append(
                [
                    timestamp,
                    _fmt(plan_artifact.production[index], 4),
                    _fmt(mean_opening, 4),
                    _fmt(mean_demand, 4),
                    _fmt(mean_served, 4),
                    _fmt(mean_shortage, 4),
                    _fmt(mean_inventory, 4),
                    _fmt(float(np.percentile(inventory_values, 5)), 4),
                    _fmt(float(np.percentile(inventory_values, 50)), 4),
                    _fmt(float(np.percentile(inventory_values, 95)), 4),
                    _fmt(
                        sum(1 for value in inventory_values if value <= 1e-9)
                        / sample_count,
                        6,
                    ),
                    _fmt(
                        sum(
                            1
                            for replication in simulation_result.replications
                            if replication.periods[index].shortage > 1e-9
                        )
                        / sample_count,
                        6,
                    ),
                    _fmt(
                        statistics.stdev(
                            [
                                replication.periods[index].simulated_demand
                                for replication in simulation_result.replications
                            ]
                        ),
                        4,
                    ),
                    _fmt(
                        statistics.stdev(
                            [
                                replication.periods[index].shortage
                                for replication in simulation_result.replications
                            ]
                        ),
                        4,
                    ),
                ]
            )
        period_table = PaperTable(
            table_id=f"tab-{run_id}-simulation-periods",
            caption=(
                "Per-period means, standard deviations, and ending-inventory "
                "distribution across replications for the frozen plan. Under lost sales, "
                "closing = opening + production - served, and total shortage reconciles "
                "row by row. Ending inventory is not opening + production - demand when "
                "shortage is nonzero; positive ending inventory occurs on low-demand paths, "
                "while high-demand paths deplete the deterministic pre-build and can end at "
                "zero. The final-period inventory is the terminal inventory of each path. "
                "Column means satisfy mean closing = mean opening + production - mean "
                "served at unrounded precision; printed four-decimal values can differ in "
                "the last digit. The deterministic LP balance table uses a different "
                "recurrence and is not expected to match the simulation means. The per-period "
                "shortage frequencies let the mean shortage-period count be checked as "
                "their sum; the joint frequency of shortages in two periods was not "
                "separately tabulated."
            ),
            columns=[
                "Period",
                "Production (frozen)",
                "Mean opening inventory",
                "Mean simulated demand",
                "Mean served",
                "Mean shortage",
                "Mean ending inventory",
                "Ending inv p05",
                "Ending inv median",
                "Ending inv p95",
                "Fraction ending inv = 0",
                "Fraction periods with shortage",
                "Demand std",
                "Shortage std",
            ],
            rows=period_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-production-plan-id",
                f"ev-mixed-{run_id}-simulation-shortage",
            ],
        )
        pool_mean = error_artifact.mean
        centered_shortages: list[float] = []
        centered_fills: list[float] = []
        centered_any = 0
        centered_opening_02: list[float] = []
        for replication in simulation_result.replications:
            inventory = 0.0
            total_demand = 0.0
            total_served = 0.0
            total_shortage = 0.0
            for index, period in enumerate(replication.periods):
                centered_demand = max(
                    0.0, period.point_forecast + period.sampled_residual - pool_mean
                )
                available = inventory + period.production
                served = min(available, centered_demand)
                shortage = max(0.0, centered_demand - available)
                inventory = max(0.0, available - centered_demand)
                if index == 0:
                    centered_opening_02.append(inventory)
                total_demand += centered_demand
                total_served += served
                total_shortage += shortage
            centered_shortages.append(total_shortage)
            centered_fills.append(
                total_served / total_demand if total_demand > 0 else 1.0
            )
            if total_shortage > 1e-9:
                centered_any += 1
        centered_any_shortage = centered_any / len(simulation_result.replications)
        centered_mean_shortage = sum(centered_shortages) / len(centered_shortages)
        centered_mean_fill = sum(centered_fills) / len(centered_fills)
        centered_mean_opening_02 = sum(centered_opening_02) / len(centered_opening_02)
        shortage_mean = simulation_result.metrics["total_shortage"].mean
        fill_mean = simulation_result.metrics["fill_rate"].mean
        uncentered_opening_02 = sum(
            replication.periods[0].ending_inventory
            for replication in simulation_result.replications
        ) / simulation_result.n_replications
        centered_rows = [
            [
                "Mean total shortage",
                f"{shortage_mean:.3f}",
                f"{centered_mean_shortage:.3f}",
            ],
            ["Mean fill rate", f"{fill_mean:.3f}", f"{centered_mean_fill:.3f}"],
            [
                "Any-shortage fraction",
                f"{simulation_result.any_shortage_probability:.3f}",
                f"{centered_any_shortage:.3f}",
            ],
            [
                "Mean 2026-02 opening inventory",
                f"{uncentered_opening_02:.3f}",
                f"{centered_mean_opening_02:.3f}",
            ],
        ]
        predictions_path = (
            paths.experiment_dir(artifact.forecast_experiment_id)
            / "artifacts"
            / "predictions.parquet"
        )
        history_frame = load_single_table(paths).sort(plan.spec.time_column)
        timestamps = [
            str(value) for value in history_frame[plan.spec.time_column].to_list()
        ]
        per_period_error_table = None
        if predictions_path.exists():
            predictions = pl.read_parquet(predictions_path)
            drift_rows = {
                int(row["row_index"]): row
                for row in predictions.filter(
                    pl.col("model") == artifact.model_id
                ).iter_rows(named=True)
            }
            baseline_rows = {
                int(row["row_index"]): row
                for row in predictions.filter(
                    pl.col("model") == "naive.last_value"
                ).iter_rows(named=True)
            }
            error_rows = []
            for row_index in sorted(set(drift_rows) & set(baseline_rows)):
                drift_row = drift_rows[row_index]
                baseline_row = baseline_rows[row_index]
                error_rows.append(
                    [
                        str(row_index),
                        str(timestamps[row_index]),
                        _fmt(float(drift_row["actual"]), 4),
                        _fmt(float(drift_row["prediction"]), 4),
                        _fmt(
                            float(drift_row["actual"])
                            - float(drift_row["prediction"]),
                            4,
                        ),
                        _fmt(float(baseline_row["prediction"]), 4),
                        _fmt(
                            float(baseline_row["actual"])
                            - float(baseline_row["prediction"]),
                            4,
                        ),
                    ]
                )
            per_period_error_table = PaperTable(
                table_id=f"tab-{run_id}-per-period-errors",
                caption=(
                    "Per-period errors for drift and last-value over the 12 rolling test "
                    "rows. These are regime-confounded and non-exchangeable; the four-fold "
                    "mean is provenance only and is not a headline accuracy comparison."
                ),
                columns=[
                    "Row index",
                    "Period",
                    "Actual",
                    "Drift forecast",
                    "Drift error (actual - forecast)",
                    "Last-value forecast",
                    "Last-value error",
                ],
                rows=error_rows,
                evidence_ids=[
                    f"ev-{artifact.forecast_experiment_id}-evaluation-{artifact.model_id}-mae",
                    f"ev-{artifact.forecast_experiment_id}-evaluation-naive.last_value-mae",
                ],
            )
        centered_table = PaperTable(
            table_id=f"tab-{run_id}-centered-pool",
            caption=(
                "Centred-pool diagnostic using the same sampled residual draws with the "
                "pool mean subtracted from each residual; no new RNG draws. This checks how "
                "much of the uncentred result is driven by the nonzero pool mean. It is a "
                "diagnostic, not a comparator claim."
            ),
            columns=["Diagnostic", "Uncentred pool", "Centred pool"],
            rows=centered_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-forecast-error-mean",
                f"ev-mixed-{run_id}-simulation-shortage",
                f"ev-mixed-{run_id}-simulation-fill-rate",
            ],
        )
        total_production = sum(plan_artifact.production)
        mean_total_demand = sum(
            sum(
                period.simulated_demand
                for period in replication.periods
            )
            for replication in simulation_result.replications
        ) / simulation_result.n_replications
        mechanical_fill_bound = (
            total_production / mean_total_demand if mean_total_demand > 0 else 1.0
        )
        sample_rows = [
            [
                str(replication.replication_id),
                str(replication.replication_seed),
                _fmt(replication.total_demand, 3),
                _fmt(replication.total_served, 3),
                _fmt(replication.total_shortage, 3),
                _fmt(replication.fill_rate, 6),
                _fmt(replication.ending_inventory, 3),
                _fmt(replication.average_inventory, 3),
                _fmt(replication.max_shortage, 3),
                ", ".join(
                    _fmt(period.simulated_demand, 3)
                    for period in replication.periods
                ),
            ]
            for replication in simulation_result.replications[:5]
        ]
        sample_table = PaperTable(
            table_id=f"tab-{run_id}-simulation-sample",
            caption=(
                "First five replications as display samples. Demand, served, and shortage "
                "are totals over the horizon; ending inventory is the final period; average "
                "inventory is the unweighted mean of per-period ending inventories; max "
                "shortage is the largest single-period shortage. The complete replication "
                "table is `experiments/simulation/replications.csv` in the evidence bundle."
            ),
            columns=[
                "Replication",
                "Seed",
                "Total demand (units)",
                "Total served (units)",
                "Total shortage (units)",
                "Fill rate (served/demand)",
                "Ending inventory (units)",
                "Average inventory (units)",
                "Max period shortage (units)",
                "Period demand vector (units, in order)",
            ],
            rows=sample_rows,
            evidence_ids=[
                f"ev-mixed-{run_id}-simulation-replications",
                f"ev-mixed-{run_id}-simulation-seed-hash",
            ],
        )
        simulation_tables = [
            residual_table,
            residual_diagnostics_table,
            protocol_table,
            simulation_metrics_table,
            period_table,
            centered_table,
            sample_table,
        ]
        if per_period_error_table is not None:
            simulation_tables.insert(2, per_period_error_table)
        simulation_figures = [_plot_simulation(paths, simulation_result, run_id)]
        shortage_mean = simulation_result.metrics["total_shortage"].mean
        fill_mean = simulation_result.metrics["fill_rate"].mean
        simulation_abstract = (
            f" The frozen production plan was then evaluated without re-optimization "
            f"by {simulation_result.n_replications} empirical replications that sample the "
            f"{error_artifact.sample_count} out-of-sample residuals with replacement. "
            f"Across replications, mean total shortage was {shortage_mean:.6f} "
            f"{simulation_result.unit} and mean fill rate was {fill_mean:.6f}; the "
            f"empirical any-shortage fraction was "
            f"{simulation_result.any_shortage_probability:.6f}. These are finite-sample "
            "Monte Carlo summaries under an iid residual-resampling assumption, not a "
            "robustness guarantee or a model of the true demand process."
        )
        simulation_sections = [
            PaperSection(
                heading="Fixed-anchor historical residual pool",
                blocks=[
                    f"The forecast-error artifact contains {error_artifact.sample_count} "
                    "fixed-anchor rolling residuals with frozen definition "
                    f"`{error_artifact.residual_definition}` "
                    f"(mean {error_artifact.mean:.6f}, sample std "
                    f"{error_artifact.std:.6f}). Each residual row is tied to a validation "
                    "fold, a test timestamp, and its source prediction. Training residuals "
                    "are not used.",
                    "Each residual row carries its fold id, row index, and timestamp; the "
                    "validator recomputes every value against the frozen "
                    "`predictions.parquet` and checks that each row lies in a rolling-"
                    "origin test window. That links rows to the experiment artifact "
                    "internally; the predictions file is not part of the report text but is "
                    "included in `paper/evidence_bundle/experiments/forecast/predictions."
                    "parquet`, so a reader with the project can re-derive the pool.",
                    "The pool mixes residuals from four fold slopes (2.0000, 2.5000, "
                    "2.1176, 2.2500); none is the deployed 24-observation slope, and the "
                    "fold-1 numerical equality at 2.0000 uses a different window. Per-fold "
                    "residual means differ, so resampling the pooled values iid is not the "
                    "error distribution of the deployed forecast. The p95 shortage is a "
                    "sample order statistic of this pool, not a deployed-forecast risk "
                    "estimate.",
                    "The residual pool is small relative to the demand scale. Empirical iid "
                    "resampling ignores residual autocorrelation and any time-varying "
                    "variance; it does not represent a complete demand process. Nine rows "
                    "come from folds 1-3 (out-of-sample for their own fold models) and "
                    "three from fold 4; all 12 test windows lie inside the full history "
                    "used for the final 2026 point forecast, so none is held out from the "
                    "plan forecast. A folds-1-3-only rerun was not performed. The "
                    f"max(0, demand) floor activated in "
                    f"{simulation_result.demand_floor_activations} period(s); with pool "
                    f"minimum {min(item.residual for item in error_artifact.residuals):g} "
                    f"and point-forecast minimum {min(artifact.point_forecasts):g}, the "
                    "floor is provably unreachable on this pool and these forecasts. No "
                    "wider synthetic pool was run to test the floor.",
                ],
            ),
            PaperSection(
                heading="Conditional simulation protocol: frozen plan under residual resampling",
                blocks=[
                    "The production plan artifact is fixed before simulation and is never "
                    "re-optimized. Each replication draws one residual per planning period "
                    "from the out-of-sample pool with replacement, forms simulated demand as "
                    "max(0, point forecast + residual), and applies lost-sales dynamics: "
                    "available = opening inventory + production; served = min(available, "
                    "demand); shortage = max(0, demand - available); ending inventory = "
                    "available - served.",
                    f"Replication seeds use the formal sha256 master-seed schedule: "
                    f"{simulation_result.n_replications} unique seeds with hash "
                    f"`{simulation_result.replication_seed_hash}`. In-process determinism "
                    "was observed; the digest strings and seed schedule are in the "
                    "evidence bundle. Cross-process reproducibility was not tested, and no "
                    "external reproduction was performed.",
                    "Service level in this report means fill rate = total served / total "
                    "demand. Shortage-period count and maximum shortage are reported "
                    "separately; no second metric is labeled service level.",
                    "The frozen residual pool has mean "
                    f"{error_artifact.mean:.6f}, so sampled demand is biased low relative "
                    "to the point forecast; fill is inflated and shortage deflated versus "
                    "mean-zero residuals. No centered-pool rerun was performed. "
                    "The optimizer and simulator use different demand semantics. The LP "
                    "has a hard demand-satisfaction equality and no stock-out or backorder "
                    "variable, so its optimum is conditional on realized demand never "
                    "exceeding availability. The simulator uses lost sales and reports "
                    "shortage when demand exceeds availability. The two result sets "
                    "therefore answer different questions and are not mutually "
                    "reconciled.",
                    f"The {simulation_result.n_replications} replications produced only "
                    f"{simulation_result.distinct_residual_paths} distinct residual paths "
                    f"out of {path_space} possible with-replacement paths; the expected "
                    f"distinct count under uniform sampling is {expected_distinct:.1f}, "
                    "consistent with the observed count. The effective path sample is much "
                    "smaller than the replication count. No chi-square usage test, "
                    "running-mean convergence diagnostic, or fold-preserving block "
                    "bootstrap was run, so the iid per-period draw "
                    "departure from fold structure is unquantified. No interval is "
                    "reported: a naive replication-level interval would be a Monte Carlo "
                    "integration interval, not a path-population interval, and no "
                    "pool-level bootstrap was computed.",
                ],
            ),
            PaperSection(
                heading="Simulation outcomes and risk language",
                blocks=[
                    "These are conditional diagnostics of one plan under in-sample "
                    "residual resampling, not deployed-forecast risk estimates. The "
                    "residual pool contains no row held out with respect to the deployed "
                    "forecast, so the reported tail behavior is not an out-of-sample error "
                    "estimate against that forecast. No comparator plan was simulated, so "
                    "the fill behavior has no evaluative scale; no uncertainty interval is "
                    "reported.",
                    f"Across {simulation_result.n_replications} replications, the "
                    f"diagnostic mean total shortage is {shortage_mean:.6f} "
                    f"{simulation_result.unit}, the mean fill rate is {fill_mean:.6f}, and "
                    f"the empirical any-shortage fraction is "
                    f"{simulation_result.any_shortage_probability:.6f}. These numbers "
                    "describe one deterministic plan under one frozen empirical "
                    "resampling protocol. They do not establish robustness, a service "
                    "guarantee, or the true future demand distribution. No robust "
                    "optimization, stochastic programming, or scenario-tree method was "
                    "run. The any-shortage frequency is grid-valued on the observed "
                    "residual paths and should not be read as stable to a third decimal.",
                    f"Mechanical fill bound: total production {total_production:.3f} divided "
                    f"by mean simulated demand {mean_total_demand:.3f} is "
                    f"{mechanical_fill_bound:.3f}; realized fill is set mainly by this "
                    "total-production level, not by the pre-build shape. A same-total-"
                    "production shape comparison was not run. The centred-pool diagnostic "
                    f"gives mean shortage {centered_mean_shortage:.3f}, mean fill "
                    f"{centered_mean_fill:.3f}, and any-shortage "
                    f"{centered_any_shortage:.3f}; this checks the nonzero-pool-mean "
                    "contribution only.",
                ],
            ),
            PaperSection(
                heading="Cross-task state propagation",
                blocks=[
                    "A changed point forecast invalidates both the optimization and "
                    "simulation nodes. A changed forecast-error artifact or production plan "
                    "invalidates only the simulation node; the optimization node remains "
                    "valid. Review resume checks the frozen artifact and evidence hashes and "
                    "never re-executes a node.",
                    "The DAG evidence records the plan, forecast-error, simulation, and seed "
                    "lineage without duplicating every node metric. The final simulation "
                    "result traces upward through the plan artifact, optimization experiment, "
                    "forecast artifact, forecast experiment, and historical dataset, and "
                    "through the residual artifact to the rolling-origin test windows.",
                ],
            ),
            PaperSection(
                heading="Simulation limitations",
                blocks=[
                    "The residual pool is small and drawn iid with replacement. This ignores "
                    "temporal dependence, changing variance, and the empirical residual "
                    "distribution's own sampling error. It is not a scenario tree, "
                    "stochastic program, or robust optimization.",
                    "Finite Monte Carlo replications give finite-precision risk estimates, "
                    "and the small residual pool means the distinct-path support is far "
                    "smaller than the replication count. No uncertainty interval is "
                    "reported; a naive replication-level interval would not reflect pool "
                    "dependence. The any-shortage fraction is an empirical frequency under this "
                    "protocol, not a guarantee of future service or a calibrated "
                    "probability. No comparator plan was simulated, so the result is "
                    "descriptive rather than evidence that this plan is better than an "
                    "alternative; the fill rate has no evaluative scale here.",
                    "The plan was optimized against the point forecast and then evaluated "
                    "under a different empirical distribution; this is a conditional "
                    "simulation of a deterministic plan, not a stochastic optimal plan.",
                ],
            ),
            PaperSection(
                heading="Attached evidence bundle",
                blocks=[
                    f"The paper build writes `{bundle_path}` containing {bundle_files} "
                    "files: raw inputs, project manifest, frozen mixed plan and spec, "
                    "forecast predictions and validation outputs, optimization and "
                    "simulation specs/results/records, the residual pool CSV, the replication "
                    "CSV, DAG evidence, and the evaluation/simulation code. `SHA256SUMS` "
                    f"has hash `{bundle_hash}`. Hash map: raw-file sha256 -> `inputs/` "
                    "CSV; dataset fingerprint and manifest -> `inputs/project_manifest.json`; "
                    "displayed-history digest -> `inputs/` history; forecast predictions -> "
                    "`experiments/forecast/predictions.parquet`; residual rows -> "
                    "`residuals.csv`; replication rows -> `experiments/simulation/"
                    "replications.csv`; code -> `code/`. This is internal provenance "
                    "bookkeeping: no hash in the contract table has been verified by any "
                    "party other than the producing pipeline, the bundle was not externally "
                    "audited, and re-running the code is left to the reader. No external "
                    "reproduction was performed.",
                ],
            ),
        ]

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
            borrowed = [
                baseline / scale if scale > 0 else float("nan")
                for baseline, scale in zip(
                    baseline_mae.fold_values, fold_scales, strict=True
                )
            ]
            borrowed_text = ", ".join(f"{value:.3f}" for value in borrowed)
            borrowed_mean = sum(borrowed) / len(borrowed) if borrowed else float("nan")
            mase_text = (
                " The last-value MASE row is omitted because under its own random-walk "
                "scale the ratio is identically 1 and not informative. Borrowing the "
                f"drift fold scales gives last-value MASE values {borrowed_text} "
                f"(mean {borrowed_mean:.3f}); that is not a standard same-model MASE. "
                f"Drift MASE values are provenance only; the minimum is "
                f"{drift_mase.fold_values[worst_fold]:.4f} on fold {worst_fold + 1}. No "
                "cross-model MASE comparison is implied."
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
        mean_gap = sum(gap_values) / len(gap_values)
        gap_sd = float(np.std(gap_values, ddof=1)) if len(gap_values) > 1 else 0.0
        first_three = min(3, len(drift_mae.fold_values))
        drift_first_three = (
            sum(drift_mae.fold_values[:first_three]) / first_three
            if first_three
            else 0.0
        )
        baseline_first_three = (
            sum(baseline_mae.fold_values[:first_three]) / first_three
            if first_three
            else 0.0
        )
        comparison_text = (
            f"Across the four folds, `{drift_model}` has lower MAE than `{baseline_model}` "
            f"on folds {drift_win_folds} and higher MAE on folds {baseline_win_folds}. "
            f"Unweighted fold means are {float(drift_mae.mean or 0.0):.4f} and "
            f"{float(baseline_mae.mean or 0.0):.4f}. Signed per-fold MAE gaps (drift minus "
            f"last-value): {signed_gaps}; the mean signed gap is {mean_gap:+.6f} and its "
            f"sample standard deviation is {gap_sd:.4f}. The folds are nested (all share "
            "the origin and y_first), so these gaps are not independent observations; no "
            "nesting-aware or sign test is valid at n=4 and none is reported. At n=4 with "
            f"gap standard deviation {gap_sd:.4f}, the design has negligible power and "
            "cannot detect a moderate effect; any meaningful comparison would require "
            "substantially more independent folds. No minimum-detectable-effect or power "
            "calculation is presented. The four-fold mean includes fold 4, whose test "
            "window lies inside the deployed forecast history; the folds-1-3 only MAE "
            f"means are {drift_first_three:.4f} and {baseline_first_three:.4f}. No valid "
            "accuracy comparison is established, goodness of fit provides no basis to "
            "prefer drift over last-value on this instance, and drift is used only as the "
            f"declared frozen rule.{mase_text}"
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
        "plan is not a defensible inventory position. "
    )
    if simulation_result is None:
        abstract += (
            "Demand uncertainty is not propagated, and no robust planning or general "
            "superiority claim is made."
        )
    else:
        abstract += (
            "The simulation propagates demand uncertainty into plan evaluation only; it "
            "does not change the deterministic plan and is not robust optimization."
        )
    abstract += simulation_abstract
    if (
        simulation_result is not None
        and simulation_spec is not None
        and error_artifact is not None
        and plan_artifact is not None
    ):
        plan_text = "/".join(_fmt(value, 3) for value in plan_artifact.production)
        capacity_text = "/".join(_fmt(value, 3) for value in plan_artifact.capacity)
        binding_equalities = [
            f"{plan_artifact.production[index]:g} = {plan_artifact.capacity[index]:g}"
            for index in range(plan_artifact.planning_horizon)
            if abs(
                plan_artifact.production[index] - plan_artifact.capacity[index]
            )
            <= 1e-6
        ]
        min_forecast = min(artifact.point_forecasts)
        min_residual = min(item.residual for item in error_artifact.residuals)
        floor_reachable = min_forecast + min_residual < 0
        floor_text = (
            f"The max(0, demand) floor activated {simulation_result.demand_floor_activations} "
            "time(s); given pool minimum "
            f"{min_residual:g} and point-forecast minimum {min_forecast:g}, the floor is "
            + (
                "reachable in principle."
                if floor_reachable
                else "provably unreachable, so it is not a tested safeguard."
            )
        )
        abstract = (
            "This is a single-instance contract-integration and conditional-simulation "
            "report, not a stress test or a robustness study. The hash-linked point-"
            f"forecast artifact `{artifact.model_id}` was consumed via a declared contract "
            f"edge with no negative control, producing plan production {plan_text} under "
            f"frozen capacities "
            f"{capacity_text}. The equality "
            + (", ".join(binding_equalities) if binding_equalities else "none")
            + " is arithmetic of the returned point; no non-binding companion solve or "
            "fault-injection run exists, so no gate or capacity-enforcement claim is made. "
            "The instance is hand-built so production equals capacity in one period; "
            "10 = 150 - 140 is a mechanical balance identity, and the results are not "
            "extrapolable beyond this construction. The production-at-capacity equality "
            "is an artifact of the fixed-anchor slope and window choice, not a demand "
            "finding; no local-slope or last-difference plan was solved. The plan is "
            "consistent with, but not evidence for, correct capacity enforcement. "
            "Window length 24 was not selected by an out-of-sample criterion; the five "
            "windows are equally defensible, and no plan or simulation was run for the "
            "other windows, so the reported shortage and fill numbers are conditional on "
            "window 24. "
            f"The fixed-anchor historical residual pool has {error_artifact.sample_count} "
            "rows (nine from folds 1-3, three from fold 4); all lie inside the history "
            "used for the deployed forecast, so this is in-sample diagnostic "
            "resampling, not forecast-error stress or a deployed-forecast risk estimate. "
            "A genuine hold-out is impossible on this 24-row history, and no "
            "folds-1-3-only run was performed. No fold-based evidence distinguishes "
            "drift from last-value; drift is frozen for contract reasons only. The pool "
            f"mean is {error_artifact.mean:.6f}, so resampled demand is biased low: fill "
            "is inflated and shortage deflated relative to mean-zero residuals, and no "
            "centered-pool rerun was performed. "
            f"{floor_text} The simulation samples residuals iid with replacement and "
            f"observed {simulation_result.distinct_residual_paths} distinct paths over "
            f"{simulation_result.n_replications} replications (single-process determinism "
            "only). The outcome table reports conditional diagnostics under this in-sample "
            "resampling; the shortage and fill values are not deployed-forecast risk "
            "estimates, service guarantees, or evaluative scores. No comparator plan was "
            "simulated, so the fill behavior has no evaluative scale. No uncertainty "
            "interval is reported; a naive replication-level interval would ignore shared-"
            "pool dependence and no pool bootstrap was computed. "
            f"An evidence bundle is written to `{bundle_path}` ({bundle_files} files) with "
            f"SHA256SUMS `{bundle_hash}`; it contains the raw inputs, manifest, frozen "
            "spec, forecast predictions, residual pool, replication CSV, DAG evidence, and "
            "evaluation/simulation code. A reader with that directory can recompute the "
            "listed hashes only with the same code and declared environment (the underlying "
            "CBC solver version is not recorded); the bundle hash is not externally "
            "verified and no third-party reproduction was performed. No LP formulation "
            "validation was performed; the plan and "
            "recurrence check is arithmetic-only. No robust planning, stochastic "
            "optimization, uncertainty-aware optimization, or accuracy/superiority claim "
            "is made."
        )

    sim_reconcile_text = ""
    if simulation_result is not None:
        opening_02 = sum(
            replication.periods[0].ending_inventory
            for replication in simulation_result.replications
        ) / simulation_result.n_replications
        zero_02 = sum(
            1
            for replication in simulation_result.replications
            if replication.periods[1].ending_inventory <= 1e-9
        ) / simulation_result.n_replications
        sim_reconcile_text = (
            f" In the simulation, mean 2026-02 opening inventory is {opening_02:.4f} "
            f"and {zero_02:.3f} of paths end 2026-02 at zero; the positive mean ending "
            "inventory comes from low-demand paths that build surplus, while high-demand "
            "paths exhaust the carry and can short."
        )
    dag_shape_text = (
        "The plan contains exactly three nodes and three edges: forecasting produces "
        "`demand_forecast` and `forecast_error`; optimization consumes the forecast and "
        "produces `production_plan`; simulation consumes both the production plan and the "
        "forecast-error artifact."
        if simulation_result is not None
        else "The plan contains exactly two nodes: a forecasting node that produces "
        "`demand_forecast` and an optimization node that requires it. The edge declares "
        "the required artifact type and schema version."
    )
    sections = [
        PaperSection(
            heading="DAG and artifact contract",
            blocks=[
                dag_shape_text,
                "The forecast artifact is accepted only after envelope type, schema version, "
                "validation status, payload hash, content hash, producer experiment identity, "
                "target name, unit, horizon, and timestamps all match the consumer contract. "
                "The declared design intent is that a missing, mutated, unvalidated, or "
                "misaligned artifact would block the downstream node instead of silently "
                "falling back. This is a subjunctive contract declaration in the frozen "
                "plan, not an executed result: no mutated, unvalidated, misaligned, or "
                "missing-artifact negative control was run. Only the accept-valid path was "
                "exercised; no blocking behaviour is demonstrated.",
            ],
        ),
        PaperSection(
            heading="Forecasting node",
            blocks=[
                f"The forecasting node ran a fixed-anchor rolling validation with "
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
                "observed difference. The final observed difference is +5 (146 - 141), the "
                "last-four-period average increment is 2.75 ((146 - 135)/4), and the global "
                "slope is 2.0. The 2026-02 forecast is 150 versus 146 under a last-value "
                "rule, so the anchored rule contributes about +4 relative to last-value. "
                "The forecast is not a defensible demand estimate. No last-difference, "
                "local-mean-difference, OLS, or damped-trend forecast was computed.",
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
                "the plan's forecast; no true held-out fold was run. Every fold keeps "
                "y_first = 100 as its slope anchor, so these are not standard "
                "rolling-origin folds that re-anchor at each training-window start. The "
                "fixed-anchor assumption propagates to every MAE/MASE fold value and to "
                "the residual pool; a re-anchored drift diagnostic was not run.",
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
            heading="Arithmetic-only feasibility check (no binding diagnosis)",
            blocks=[
                "All production-planning results are conditional on a 24-observation "
                "forecast window chosen by no out-of-sample criterion. The LP was not "
                "re-solved under windows 15, 18, or 21; those alternatives are reported "
                "only as forecast sensitivity, so the reported plan is one conditional "
                "outcome of an arbitrary window choice.",
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
                "in 2026-02 is zero and no deterministic buffer remains; the deterministic "
                "plan is on a feasibility knife-edge, not a margin. Simulation later shows "
                "positive ending inventory on low-demand paths and buffer exhaustion on "
                "high-demand paths. The 10-unit pre-build is the cost-minimizing member "
                "of the feasible [0, 22] pre-build range under the 2026-01 capacity; it is "
                "not uniquely required by feasibility and is attributable to the fixed-"
                "anchor rule and frozen capacities, not to demand uncertainty. The plan is "
                "consistent with, but not evidence for, correct capacity enforcement. "
                "Re-summing cost from the "
                f"displayed rows gives production {production_total:.6f} + holding "
                f"{holding_total:.6f} = {recomputed_objective:.6f}; this is an arithmetic "
                "re-sum, not a second-solver verification.",
                "Terminal inventory is unconstrained: there is no salvage value, no "
                "terminal stock requirement, and no terminal holding penalty. The "
                "objective charges holding cost on ending inventory in every period, "
                "including the final period; here the final inventory is zero, so the "
                "terminal convention is numerically moot but is stated for reconstruction. "
                "Initial inventory is 0. With constant production cost and positive "
                "holding cost, the optimizer carries only the 10 units needed for "
                "feasibility and ends with zero inventory. A model with terminal-stock "
                "requirements would produce a different objective and was not solved.",
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
                "construction and is not evidence of solver or validator quality. With "
                "constant production cost and no stock-out variable, the optimum is "
                "essentially the balance-satisfying minimum schedule; the objective is "
                "near-forced by the instance, so this is a solver-arithmetic sanity check "
                "rather than a cost trade-off result.",
                deviation_text,
                "Production equals the displayed capacity in one period. In other periods "
                "the bound is slack. The instance was chosen by hand to create that "
                "equality; it shows only that one feasible point exists under the stated "
                "constraints, not that capacity enforcement was demonstrated or that a "
                "planning mechanism produced a non-trivial decision. Production and "
                "holding cost rates are constant "
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
                "perturbation was solved. The only buffer in the deterministic plan is "
                "the 10-unit 2026-01 pre-build above 2026-01 demand; its ending inventory "
                "after 2026-02 and 2026-03 is zero at the point forecast. The simulator, "
                "however, opens each period with that path's simulated carry, so "
                "low-demand paths have positive slack and high-demand paths can short."
                + sim_reconcile_text
                + " The 'knife-edge' characterization applies to the deterministic point "
                "forecast only, not to the simulated distribution. There is no stock-out "
                "or backorder variable, and a demand surprise large enough to exceed the "
                "period-1 headroom would make the LP infeasible.",
            ],
        ),
        PaperSection(
            heading="Arithmetic-only plan and recurrence check",
            blocks=[
                "The mixed validator re-verifies the plan hash, DAG completion, artifact "
                "contract and hashes, upstream validation and evidence, the artifact identity "
                "recorded on the optimization experiment, every inventory-balance constraint "
                "against the artifact point forecasts, capacity and nonnegativity bounds, the "
                "objective value, and provenance completeness.",
                "The validator recomputes the same constraints from the frozen variable values "
                "through a separate code path. Because the optimizer and validator share the "
                "frozen plan and artifact, this is arithmetic-only checking, not "
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
                "of forecast error is not quantified. The reported plan is also not robust "
                "to window choice: window 15 gives a 2026-02 forecast of exactly 140 and no "
                "pre-build, window 18 gives 140.2353 and about 0.24 units of pre-build, and "
                "windows 21 and 24 give 9.5 and 10 units. All five arithmetic projections "
                "equal 2026-02 capacity because the forecast is at or near 140; only the "
                "pre-build magnitude varies. Window 12 gives 2026-02/03 forecasts of 126/128, "
                "both below capacity, so it is the one window with no 2026-02 shortfall; its "
                "fold-1 realized MAE is 3.6667. The reported 10-unit pre-build is therefore "
                "conditional on window 24, which has no held-out evaluation (only in-sample "
                "fold-4 rows); windows 21 and 24 likewise have no held-out support.",
                "The mixed slice supports one forecast-to-plan-to-simulation path with "
                "temporal baselines only. Neural forecasters, probabilistic forecasts, "
                "multi-echelon networks, stochastic optimization, and simulation feedback "
                "loops are untested.",
                "Demand uncertainty is propagated only into the conditional simulation, not "
                "into optimization. A deterministic point forecast can understate the cost "
                "of forecast error; results should not be read as a "
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
                "consumed the artifact point forecasts. The raw CSV is not in the report "
                "text but is included in `paper/evidence_bundle/inputs/`.",
                "The hashes in the contract table establish internal provenance and hash "
                "linkage only. The displayed series and parameters are sufficient to "
                "re-solve the LP and re-check the balance recurrence. The forecast "
                "artifact, evaluation metrics, manifest fingerprint, and raw-file sha256 "
                "cannot be re-derived from this report without the raw CSV and the "
                "evaluation code; for three-node runs these are in `paper/evidence_bundle`, "
                "but no external reproduction was "
                "performed. The rolling-origin split boundaries and the drift rule are "
                "stated above. The cited rolling-origin references (Bergmeir and Benítez "
                "2012; Hyndman and Athanasopoulos 2021) describe re-anchoring at each "
                "training-window start; this fixed-anchor fold construction departs from "
                "that convention.",
            ],
        ),
    ]

    title = (
        "Single-Instance Contract-Integration and Conditional-Simulation Report — "
        "Frozen Slice"
        if simulation_result is not None
        else "Forecast-to-Planning Contract Integration Report — Frozen Single Slice"
    )
    return PaperAST(
        title=title,
        abstract=abstract,
        sections=[*sections, *simulation_sections],
        tables=[
            history_table,
            forecast_table,
            metrics_table,
            fold_table,
            window_table,
            plan_window_table,
            parameter_table,
            planning_table,
            recurrence_table,
            contract_table,
            *simulation_tables,
        ],
        figures=[
            _plot_dag(paths, plan, run_id),
            _plot_plan(
                paths,
                plan,
                artifact,
                values,
                run_id,
                simulation_result=simulation_result,
            ),
            *simulation_figures,
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
