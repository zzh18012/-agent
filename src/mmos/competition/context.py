import hashlib
import json
import re
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
                    if "optim"
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
            "Period convention: planning periods m=1,2,3. I_0 is the frozen initial "
            "inventory. Deterministic balance: I_m = I_(m-1) + P_m - F_m for m=1..3, "
            "0 <= P_m <= cap_m, I_m >= 0, minimize sum_m(c_p*P_m + c_h*I_m). Do not "
            "index production from zero; the reported inventory_0 is I_1.",
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
            "service guarantees, reproducibility, or model superiority. The demand "
            "floor is protocol bookkeeping; with zero activations it is untested.",
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
        value: float | int,
        value_unit: str,
        experiment_id: str,
        artifact_path: Path,
        artifact_hash: str,
        source_field: str,
        source_file: str,
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
                "derivation": "frozen mixed plan/forecast/plan artifact",
            },
        )
        records.append(record)

    forecast_experiment = str(release["forecast_experiment_id"])
    optimization_experiment = str(release["optimization_experiment_id"])
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
    return records


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


def load_context(paths: ProjectPaths) -> EvidenceContext:
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
    derived_records = _derived_mixed_records(paths, release)
    selected_records = [
        *derived_records,
        *release_records,
        *other_records,
    ][:MAX_CATALOG_RECORDS]
    entries = [_entry(record) for record in selected_records]
    total_evidence_count = len(index.records) + len(derived_records)
    catalog_truncated = total_evidence_count > len(entries)

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.experiment_id] = counts.get(entry.experiment_id, 0) + 1
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
    unavailable = sum(1 for entry in entries if entry.numeric)
    uncertainty = len(_uncertainty_prefixes(entries))
    available_tables = {
        TableTemplate.METRIC_COMPARISON.value: metric_records,
        TableTemplate.EXPERIMENT_INVENTORY.value: len(experiments),
        TableTemplate.EVIDENCE_LINEAGE.value: len(entries),
    }
    available_figures = {
        FigureTemplate.PIPELINE_SCHEMATIC.value: len(experiments),
        FigureTemplate.METRIC_BAR.value: unavailable,
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
