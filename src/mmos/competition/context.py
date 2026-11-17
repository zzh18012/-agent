import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import (
    ProjectPaths,
    load_manifest,
    load_problem_spec,
    load_project_config,
    sha256_file,
)
from mmos.evidence.store import load_claims, load_evidence_index
from mmos.schemas.competition import (
    FigureTemplate,
    TableTemplate,
)
from mmos.schemas.evidence import ClaimRecord, EvidenceRecord
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.states import ProjectState

MAX_CATALOG_RECORDS = 300
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_/:-])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z0-9_/:-])"
)


class CompetitionContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    evidence_id: str
    key: str
    display_value: str
    raw_value: float | int | str
    unit: str | None
    experiment_id: str
    artifact_hash: str
    numeric: bool

    def prompt_dict(self) -> dict[str, str | None]:
        return {
            "evidence_id": self.evidence_id,
            "key": self.key,
            "display_value": self.display_value,
            "unit": self.unit,
            "experiment_id": self.experiment_id,
        }


@dataclass(frozen=True)
class ExperimentInfo:
    experiment_id: str
    label: str
    task: str
    evidence_count: int

    def prompt_dict(self) -> dict[str, str | int]:
        return {
            "experiment_id": self.experiment_id,
            "label": self.label,
            "task": self.task,
            "evidence_count": self.evidence_count,
        }


@dataclass(frozen=True)
class EvidenceContext:
    paths: ProjectPaths
    problem: ProblemSpec
    release: dict
    release_path: Path
    release_hash: str
    entries: tuple[CatalogEntry, ...]
    claims: tuple[ClaimRecord, ...]
    experiments: tuple[ExperimentInfo, ...]
    allowed_metadata_numbers: tuple[str, ...]
    context_hash: str
    available_tables: dict[str, int]
    available_figures: dict[str, int]
    model_facts: tuple[str, ...]
    derived_records: tuple[EvidenceRecord, ...]
    appendix_records: tuple[EvidenceRecord, ...]
    extra_evidence_ids: tuple[str, ...]
    augmentation_id: str | None
    total_evidence_count: int
    catalog_truncated: bool

    def evidence_ids(self) -> set[str]:
        return {entry.evidence_id for entry in self.entries}

    def prompt_payload(self) -> dict:
        return {
            "problem": self.problem.model_dump(mode="json"),
            "release_summary": self.release,
            "experiments": [item.prompt_dict() for item in self.experiments],
            "evidence_catalog": [entry.prompt_dict() for entry in self.entries],
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "statement": claim.statement,
                    "evidence_ids": claim.evidence_ids,
                    "status": claim.status,
                }
                for claim in self.claims
            ],
            "available_tables": self.available_tables,
            "available_figures": self.available_figures,
            "model_facts": list(self.model_facts),
            "augmentation_id": self.augmentation_id,
            "augmentation_appendix_record_count": len(self.appendix_records),
            "catalog_truncated": self.catalog_truncated,
        }


def _latest_release(paths: ProjectPaths) -> Path:
    releases = [
        path
        for path in paths.releases_dir.glob("*.json")
        if not path.name.startswith("competition_paper_")
    ]
    if not releases:
        raise CompetitionContextError(
            "no frozen release found; run the modeling pipeline to RELEASED first"
        )
    return max(releases, key=lambda path: path.stat().st_mtime)


def _collect_experiment_ids(payload: object, prefix: str = "") -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, str) and (key.endswith("experiment_id") or key == "experiment_id"):
                label = key.removesuffix("_experiment_id") or "experiment"
                task = (
                    "forecasting"
                    if "forecast" in label
                    else "optimization"
                    if "optim" in label
                    else "simulation"
                    if "simulation" in label
                    else "unknown"
                )
                found.append((value, label, task))
            else:
                found.extend(_collect_experiment_ids(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_collect_experiment_ids(value, f"{prefix}[{index}]"))
    return found


def _problem_numbers(problem: ProblemSpec) -> set[str]:
    text = " ".join(
        [
            problem.title,
            problem.overall_goal,
            *problem.global_assumptions,
            *problem.ambiguities,
            *problem.required_deliverables,
            *[
                value
                for subproblem in problem.subproblems
                for value in (
                    subproblem.statement,
                    subproblem.goal,
                    *subproblem.assumptions,
                    *subproblem.hard_constraints,
                    *subproblem.soft_constraints,
                    *subproblem.evaluation_criteria,
                )
                if value
            ],
        ]
    )
    return set(_NUMBER_RE.findall(text)) | set(re.findall(r"\b(?:19|20)\d{2}\b", text))


def _release_numbers(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            found.update(_release_numbers(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_release_numbers(value))
    elif isinstance(payload, str):
        found.update(_NUMBER_RE.findall(payload))
        found.update(re.findall(r"\b(?:19|20)\d{2}\b", payload))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        found.add(f"{payload}")
    return found


def _model_facts(release: dict) -> tuple[str, ...]:
    keys = set(release)
    if ({"forecast_experiment_id", "optimization_experiment_id"} <= keys) and (
        "simulation_experiment_id" in keys
    ):
        return (
            "Period convention: planning periods m=1,2,3. Use `I_init` for the frozen "
            "initial inventory and `I_m` for month-end stock. Deterministic balance: "
            "I_m = I_(m-1) + P_m - F_m for m=1..3 with I_0 = I_init, 0 <= P_m <= cap_m, "
            "I_m >= 0, minimize sum_m(c_p*P_m + c_h*I_m). Artifact index 0 corresponds "
            "to m=1: artifact `inventory_0` is I_1. Never define I_0 as both the initial "
            "inventory and the first reported inventory; use I_init explicitly.",
            "Forecast convention: T is the count of historical observations (T=24); y_1 "
            "is the first and y_T the last observation; slope=(y_T-y_1)/(T-1); "
            "F_(T+h)=y_T+h*slope for h=1,2,3. Fold windows are 0-based half-open "
            "intervals, but the slope counts observations; the deployed future forecast "
            "uses the full 24-observation window. Do not write '/1' or another rule.",
            "Residual provenance: each residual is actual minus forecast on a "
            "rolling-origin test row. Those rows are out-of-sample for their own fold "
            "models but all lie inside the 24-observation history used by the deployed "
            "forecast. Say 'rolling-origin historical residuals'; never claim "
            "out-of-sample validity for the deployed forecast. The 12 residuals are "
            "pooled across 4 folds x 3 horizons and drawn with replacement; no fold or "
            "horizon stratification is performed.",
            "Simulation: for replication s and m=1..3, e_m is sampled with replacement "
            "from the pooled residuals; D_m=max(0,F_m+e_m); A_m=I_(m-1)+P_m with P "
            "fixed; served_m=min(A_m,D_m); shortage_m=max(0,D_m-A_m); "
            "I_m=A_m-served_m. Total shortage is sum_m shortage_m; fill rate is "
            "sum_m served_m / sum_m D_m; any-shortage probability is the fraction of "
            "replications with total shortage > 0; means and quantiles are taken over "
            "replications.",
            "Units and capacity: quantities use the artifact unit with a one-month "
            "period; P_m and F_m are per-month flows, I_m and I_0 are month-end stocks; "
            "cap_m and I_0 are frozen exogenous parameters, not decisions; costs are "
            "normalized cost units per unit and per held unit-month. State the "
            "one-month convention. Do not claim capacity enforcement; the plan only "
            "exhibits one feasible point.",
            "Do not claim exchangeability, iid validity, unbiasedness, robustness, "
            "service guarantees, reproducibility, or model superiority. Do not claim "
            "the objective is degenerate or that the reported plan is one of a family "
            "of optimal solutions; no uniqueness or degeneracy analysis is evidenced. "
            "The demand floor is protocol bookkeeping; with zero activations it is "
            "untested.",
            "Uncertainty convention: `uncertainty.*.folds` lists the four fold-level "
            "values; `uncertainty.*.ci_low/ci_high` is a confidence interval for the "
            "fold mean, not the min/max fold range and not a predictive interval. Never "
            "describe the interval as containing all folds; report fold values "
            "separately and say the interval is for the mean.",
            "Residual caveat: the pooled residual mean is a registered evidence value "
            "(key mixed.forecast_error.mean) and is not assumed zero. Residuals are not "
            "recentered before resampling, so simulated demand inherits the pool's mean "
            "shift and the simulation does not represent the deployed forecast's own "
            "error distribution. State both points explicitly; do not claim unbiased or "
            "calibrated service.",
            "Precision caveat: evidence values are rendered at four significant digits "
            "(simulation summaries at two); the digit count is not a Monte Carlo "
            "precision claim. Duplicate residual paths exist (compare "
            "`mixed.simulation.distinct_residual_paths` with "
            "`mixed.simulation.replications`), so effective sample size is below nominal. "
            "Report the recorded normal-approximation standard errors "
            "`mixed.simulation.shortage.standard_error`, "
            "`mixed.simulation.service.fill_rate.standard_error`, and "
            "`mixed.simulation.any_shortage_probability.standard_error` alongside the "
            "means; no bootstrap interval was executed.",
            "Fold mean: `uncertainty.*.fold_mean` is the arithmetic mean of the recorded "
            "fold values, while the pooled metric is computed from all rolling-origin "
            "errors. Do not describe the fold-mean interval as an interval for the pooled "
            "metric; state that the interval construction is not evidenced.",
            "Simulation seed: `mixed.simulation.master_seed` and the replication seed "
            "hash are recorded. The per-replication draw implementation is not described "
            "beyond the protocol text, so state the reproducibility limits explicitly.",
            "Optimality convention: report cost minimization only for the frozen "
            "parameter point under the recorded solver status and optimality gap. Do not "
            "claim duals, binding-constraint certificates, or objective stability; a "
            "capacity-perturbed instance is a different instance and its objective is "
            "not comparable.",
            "Augmentation convention: executed augmentation experiments are appended "
            "after the source paper. They use perturbed instances or different "
            "replication/seed protocols and are not numerically comparable to source "
            "headline values. No augmentation result revises, upgrades, or invalidates a "
            "source claim, and the source release is unchanged. When "
            "`augmentation_appendix_record_count` is positive, do not claim that no "
            "cost, capacity, initial-inventory, or replication/seed sensitivity analysis "
            "was run; state that source-stage analyses are limited and that executed "
            "augmentation results are listed separately.",
        )
    return ()


def _derived_mixed_records(
    paths: ProjectPaths, release: dict
) -> list[EvidenceRecord]:
    required = {
        "forecast_experiment_id",
        "optimization_experiment_id",
        "simulation_experiment_id",
        "run_id",
    }
    if not required <= set(release):
        return []
    plan_path = paths.mixed_dir / "plan.json"
    run_dir = paths.mixed_dir / "runs" / str(release["run_id"]) / "artifacts"
    forecast_envelope_path = run_dir / "forecast_envelope.json"
    plan_envelope_path = run_dir / "production_plan_envelope.json"
    if not (plan_path.exists() and forecast_envelope_path.exists() and plan_envelope_path.exists()):
        return []
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    spec = plan_payload.get("spec", {})
    periods = list(spec.get("planning_periods", []))
    capacities = list(spec.get("capacities", []))
    production_costs = list(spec.get("production_costs", []))
    holding_costs = list(spec.get("holding_costs", []))
    unit = str(spec.get("unit", "units"))
    if not periods or not (
        len(periods)
        == len(capacities)
        == len(production_costs)
        == len(holding_costs)
    ):
        return []
    forecast_envelope = json.loads(
        forecast_envelope_path.read_text(encoding="utf-8")
    )
    plan_envelope = json.loads(plan_envelope_path.read_text(encoding="utf-8"))
    forecast_path = paths.root / str(forecast_envelope["payload_path"])
    plan_payload_path = paths.root / str(plan_envelope["payload_path"])
    if not forecast_path.exists() or not plan_payload_path.exists():
        return []
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    plan_artifact = json.loads(plan_payload_path.read_text(encoding="utf-8"))
    plan_hash = sha256_file(plan_path)
    forecast_hash = sha256_file(forecast_path)
    plan_artifact_hash = sha256_file(plan_payload_path)
    now = datetime.now(UTC).replace(microsecond=0)
    records: list[EvidenceRecord] = []

    def add(
        evidence_id: str,
        key: str,
        value: float | int | str,
        value_unit: str | None,
        experiment_id: str,
        artifact_path: Path,
        artifact_hash: str,
        source_field: str,
        source_file: str,
        derivation: str = "frozen mixed plan/forecast/plan artifact",
    ) -> None:
        display = f"{value:.10g}" if isinstance(value, float) else str(value)
        record = EvidenceRecord(
            evidence_id=evidence_id,
            key=key,
            raw_value=value,
            display_value=display,
            unit=value_unit,
            experiment_id=experiment_id,
            artifact_path=artifact_path.relative_to(paths.root).as_posix(),
            artifact_hash=artifact_hash,
            source_field=source_field,
            validation_reports=["competition_context"],
            verified_at=now,
            provenance={
                "kind": "competition_derived",
                "source_file": source_file,
                "source_sha256": artifact_hash,
                "derivation": derivation,
            },
        )
        records.append(record)

    forecast_experiment = str(release["forecast_experiment_id"])
    optimization_experiment = str(release["optimization_experiment_id"])
    production_terms_total = 0.0
    holding_terms_total = 0.0
    for index, period in enumerate(periods):
        add(
            f"ev-comp-capacity-{period}",
            f"competition.plan.capacity.{period}",
            float(capacities[index]),
            f"{unit}/month",
            optimization_experiment,
            plan_path,
            plan_hash,
            f"spec.capacities[{index}]",
            "mixed/plan.json",
        )
        add(
            f"ev-comp-production-cost-{period}",
            f"competition.plan.production_cost.{period}",
            float(production_costs[index]),
            "cost_units/unit",
            optimization_experiment,
            plan_path,
            plan_hash,
            f"spec.production_costs[{index}]",
            "mixed/plan.json",
        )
        add(
            f"ev-comp-holding-cost-{period}",
            f"competition.plan.holding_cost.{period}",
            float(holding_costs[index]),
            "cost_units/unit-month",
            optimization_experiment,
            plan_path,
            plan_hash,
            f"spec.holding_costs[{index}]",
            "mixed/plan.json",
        )
        add(
            f"ev-comp-forecast-{period}",
            f"competition.forecast.point.{period}",
            float(forecast["point_forecasts"][index]),
            unit,
            forecast_experiment,
            forecast_path,
            forecast_hash,
            f"point_forecasts[{index}]",
            str(forecast_envelope["payload_path"]),
        )
        add(
            f"ev-comp-production-{period}",
            f"competition.plan.production.{period}",
            float(plan_artifact["production"][index]),
            f"{unit}/month",
            optimization_experiment,
            plan_payload_path,
            plan_artifact_hash,
            f"production[{index}]",
            str(plan_envelope["payload_path"]),
        )
        add(
            f"ev-comp-inventory-{period}",
            f"competition.plan.inventory.{period}",
            float(plan_artifact["planned_inventory"][index]),
            unit,
            optimization_experiment,
            plan_payload_path,
            plan_artifact_hash,
            f"planned_inventory[{index}]",
            str(plan_envelope["payload_path"]),
        )
        production_term = float(production_costs[index]) * float(
            plan_artifact["production"][index]
        )
        holding_term = float(holding_costs[index]) * float(
            plan_artifact["planned_inventory"][index]
        )
        production_terms_total += production_term
        holding_terms_total += holding_term
        add(
            f"ev-comp-production-cost-term-{period}",
            f"competition.plan.production_cost_term.{period}",
            production_term,
            "cost_units",
            optimization_experiment,
            plan_payload_path,
            plan_artifact_hash,
            f"production_costs[{index}] * production[{index}]",
            str(plan_envelope["payload_path"]),
            "production cost coefficient times recorded production",
        )
        add(
            f"ev-comp-holding-cost-term-{period}",
            f"competition.plan.holding_cost_term.{period}",
            holding_term,
            "cost_units",
            optimization_experiment,
            plan_payload_path,
            plan_artifact_hash,
            f"holding_costs[{index}] * planned_inventory[{index}]",
            str(plan_envelope["payload_path"]),
            "holding cost coefficient times recorded month-end inventory",
        )
    add(
        "ev-comp-objective-recomputed",
        "competition.plan.objective_recomputed",
        production_terms_total + holding_terms_total,
        "cost_units",
        optimization_experiment,
        plan_payload_path,
        plan_artifact_hash,
        "sum of production and holding cost terms",
        str(plan_envelope["payload_path"]),
        "sum over periods of production and holding cost terms",
    )
    add(
        "ev-comp-initial-inventory",
        "competition.plan.initial_inventory",
        float(spec.get("initial_inventory", 0.0)),
        unit,
        optimization_experiment,
        plan_path,
        plan_hash,
        "spec.initial_inventory",
        "mixed/plan.json",
    )

    simulation_experiment = str(release["simulation_experiment_id"])
    simulation_result = (
        paths.experiment_dir(simulation_experiment)
        / "artifacts"
        / "mixed_simulation_result.json"
    )
    if simulation_result.exists():
        payload = json.loads(simulation_result.read_text(encoding="utf-8"))
        replications = payload.get("replications")
        if isinstance(replications, list) and len(replications) >= 2:
            simulation_hash = sha256_file(simulation_result)
            source_file = simulation_result.relative_to(paths.root).as_posix()
            shortages = [
                float(item["total_shortage"])
                for item in replications
                if "total_shortage" in item
            ]
            fill_rates = [
                float(item["fill_rate"])
                for item in replications
                if "fill_rate" in item
            ]
            n = len(replications)
            any_shortage = float(
                payload.get(
                    "any_shortage_probability",
                    sum(1 for value in shortages if value > 0) / n,
                )
            )
            demands = [
                float(period["simulated_demand"])
                for item in replications
                for period in item.get("periods", [])
                if "simulated_demand" in period
            ]

            def add_simulation(
                evidence_id: str,
                key: str,
                value: float | int,
                value_unit: str | None,
                source_field: str,
                derivation: str,
            ) -> None:
                add(
                    evidence_id,
                    key,
                    value,
                    value_unit,
                    simulation_experiment,
                    simulation_result,
                    simulation_hash,
                    source_field,
                    source_file,
                    derivation,
                )

            if shortages:
                add_simulation(
                    "ev-comp-simulation-shortage-se",
                    "mixed.simulation.shortage.standard_error",
                    statistics.stdev(shortages) / math.sqrt(len(shortages)),
                    payload.get("unit"),
                    "derived from replications[].total_shortage",
                    "sample standard deviation of replication-level total shortage "
                    "divided by sqrt(n)",
                )
            if fill_rates:
                add_simulation(
                    "ev-comp-simulation-fill-rate-se",
                    "mixed.simulation.service.fill_rate.standard_error",
                    statistics.stdev(fill_rates) / math.sqrt(len(fill_rates)),
                    None,
                    "derived from replications[].fill_rate",
                    "sample standard deviation of replication-level fill rate divided "
                    "by sqrt(n)",
                )
            add_simulation(
                "ev-comp-simulation-any-shortage-se",
                "mixed.simulation.any_shortage_probability.standard_error",
                math.sqrt(any_shortage * (1.0 - any_shortage) / n),
                None,
                "derived from any_shortage_probability",
                "normal-approximation standard error sqrt(p*(1-p)/n)",
            )
            if demands:
                add_simulation(
                    "ev-comp-simulation-min-demand",
                    "mixed.simulation.min_simulated_demand",
                    min(demands),
                    payload.get("unit"),
                    "derived from replications[].periods[].simulated_demand",
                    "minimum realized simulated demand across all replications",
                )
            master_seed = payload.get("master_seed")
            if master_seed is not None:
                add_simulation(
                    "ev-comp-simulation-master-seed",
                    "mixed.simulation.master_seed",
                    master_seed,
                    None,
                    "master_seed",
                    "recorded master seed from the validated simulation artifact",
                )
    return records


def _derived_fold_mean_records(
    records: dict[str, EvidenceRecord],
) -> list[EvidenceRecord]:
    derived: list[EvidenceRecord] = []
    now = datetime.now(UTC).replace(microsecond=0)
    for record in records.values():
        if not record.key.endswith(".folds"):
            continue
        values = [float(token) for token in _NUMBER_RE.findall(record.display_value)]
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        derived.append(
            EvidenceRecord(
                evidence_id=f"{record.evidence_id}-fold-mean",
                key=record.key.removesuffix(".folds") + ".fold_mean",
                raw_value=mean,
                display_value=f"{mean:.10g}",
                unit=None,
                experiment_id=record.experiment_id,
                artifact_path=record.artifact_path,
                artifact_hash=record.artifact_hash,
                source_field=f"derived mean of {record.source_field}",
                validation_reports=["competition_context"],
                verified_at=now,
                provenance={
                    "kind": "competition_derived",
                    "source_file": record.artifact_path,
                    "source_sha256": record.artifact_hash,
                    "derivation": "arithmetic mean of recorded fold values",
                },
            )
        )
    return derived


def _entry(record: EvidenceRecord) -> CatalogEntry:
    numeric = isinstance(record.raw_value, (int, float)) and not isinstance(
        record.raw_value, bool
    )
    return CatalogEntry(
        evidence_id=record.evidence_id,
        key=record.key,
        display_value=record.display_value,
        raw_value=record.raw_value,
        unit=record.unit,
        experiment_id=record.experiment_id,
        artifact_hash=record.artifact_hash,
        numeric=numeric,
    )


def _uncertainty_prefixes(entries: list[CatalogEntry]) -> set[str]:
    lows = {
        entry.key.removesuffix(".ci_low")
        for entry in entries
        if entry.key.endswith(".ci_low")
    }
    highs = {
        entry.key.removesuffix(".ci_high")
        for entry in entries
        if entry.key.endswith(".ci_high")
    }
    return lows & highs


def load_context(
    paths: ProjectPaths,
    *,
    extra_records: list[EvidenceRecord] | None = None,
    augmentation_id: str | None = None,
) -> EvidenceContext:
    config = load_project_config(paths)
    if config.state is not ProjectState.RELEASED:
        raise CompetitionContextError(
            "competition paper requires a RELEASED project; "
            f"current state is {config.state.value}"
        )
    load_manifest(paths)
    problem = load_problem_spec(paths)
    index = load_evidence_index(paths)
    if not index.records:
        raise CompetitionContextError("verified evidence index is empty")
    claims = load_claims(paths)
    release_path = _latest_release(paths)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release_hash = hashlib.sha256(release_path.read_bytes()).hexdigest()

    experiment_rows = _collect_experiment_ids(release)
    release_experiment_ids = {item[0] for item in experiment_rows}
    release_records = [
        record
        for record in index.records.values()
        if record.experiment_id in release_experiment_ids
    ]
    other_records = [
        record
        for record in index.records.values()
        if record.experiment_id not in release_experiment_ids
    ]
    derived_records = [
        *_derived_mixed_records(paths, release),
        *_derived_fold_mean_records(index.records),
    ]
    extra = list(extra_records or [])
    selected_records = [
        *derived_records,
        *release_records,
        *other_records,
    ][:MAX_CATALOG_RECORDS]
    entries = [_entry(record) for record in selected_records]
    total_evidence_count = len(index.records) + len(derived_records) + len(extra)
    catalog_truncated = total_evidence_count > len(entries) + len(extra)

    simulation_experiment = release.get("simulation_experiment_id")
    counts: dict[str, int] = {}
    for entry in entries:
        owner = entry.experiment_id
        if simulation_experiment and entry.key.startswith("mixed.simulation."):
            owner = str(simulation_experiment)
        counts[owner] = counts.get(owner, 0) + 1
    labels = {experiment_id: (label, task) for experiment_id, label, task in experiment_rows}
    experiment_ids = sorted(set(counts) | release_experiment_ids)
    experiments = tuple(
        ExperimentInfo(
            experiment_id=experiment_id,
            label=labels.get(experiment_id, ("experiment", "unknown"))[0],
            task=labels.get(experiment_id, ("experiment", "unknown"))[1],
            evidence_count=counts.get(experiment_id, 0),
        )
        for experiment_id in experiment_ids
    )

    metric_records = sum(1 for entry in entries if entry.key.startswith("metrics."))
    prefix_units: dict[str, set[str | None]] = {}
    for entry in entries:
        if not entry.numeric:
            continue
        prefix = entry.key.rsplit(".", 1)[0] if "." in entry.key else entry.key
        prefix_units.setdefault(prefix, set()).add(entry.unit)
    metric_bar_prefixes = sum(
        1 for units in prefix_units.values() if len(units) == 1
    )
    uncertainty = len(_uncertainty_prefixes(entries))
    available_tables = {
        TableTemplate.METRIC_COMPARISON.value: metric_records,
        TableTemplate.EXPERIMENT_INVENTORY.value: len(experiments),
        TableTemplate.EVIDENCE_LINEAGE.value: len(entries),
    }
    available_figures = {
        FigureTemplate.PIPELINE_SCHEMATIC.value: len(experiments),
        FigureTemplate.METRIC_BAR.value: metric_bar_prefixes,
        FigureTemplate.UNCERTAINTY.value: uncertainty,
    }

    allowed_metadata_numbers = sorted(
        _problem_numbers(problem) | _release_numbers(release)
    )
    hash_payload = {
        "problem": problem.model_dump(mode="json"),
        "release": release,
        "evidence": [
            [entry.evidence_id, entry.key, entry.display_value, entry.unit]
            for entry in entries
        ],
        "claims": [
            [claim.claim_id, claim.status, claim.evidence_ids] for claim in claims
        ],
        "model_facts": list(_model_facts(release)),
        "extra_evidence": [
            [record.evidence_id, record.key, record.display_value, record.unit]
            for record in extra
        ],
        "augmentation_id": augmentation_id,
    }
    context_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return EvidenceContext(
        paths=paths,
        problem=problem,
        release=release,
        release_path=release_path,
        release_hash=release_hash,
        entries=tuple(entries),
        claims=tuple(claims),
        experiments=experiments,
        allowed_metadata_numbers=tuple(allowed_metadata_numbers),
        context_hash=context_hash,
        available_tables=available_tables,
        available_figures=available_figures,
        model_facts=_model_facts(release),
        derived_records=tuple(derived_records),
        appendix_records=tuple(extra),
        extra_evidence_ids=tuple(record.evidence_id for record in extra),
        augmentation_id=augmentation_id,
        total_evidence_count=total_evidence_count,
        catalog_truncated=catalog_truncated,
    )


def resolve_evidence_ids(
    context: EvidenceContext, values: list[str]
) -> list[str]:
    known = context.evidence_ids()
    by_experiment: dict[str, list[str]] = {}
    for entry in context.entries:
        by_experiment.setdefault(entry.experiment_id, []).append(entry.evidence_id)
    resolved: list[str] = []
    for value in values:
        if value in known:
            resolved.append(value)
        elif value in by_experiment:
            resolved.extend(by_experiment[value][:10])
        else:
            resolved.append(value)
    return list(dict.fromkeys(resolved))


__all__ = [
    "CatalogEntry",
    "CompetitionContextError",
    "EvidenceContext",
    "ExperimentInfo",
    "load_context",
    "resolve_evidence_ids",
]
