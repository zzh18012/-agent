from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import atomic_write_text
from mmos.historical_benchmark.schemas import BurnRegistry, HistoricalBenchmarkRun


class FreshnessError(RuntimeError):
    pass


def _hash_payload(payload: object, *, exclude: tuple[str, ...] = ()) -> str:
    data = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for key in exclude:
            data.pop(key, None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def freshness_preflight(
    repo_root: Path,
    *,
    competition: str,
    year: int,
    labels: list[str],
    now: datetime | None = None,
) -> BurnRegistry:
    """Detect labels already revealed by prior benchmark runs without reading problem bodies."""
    burned: set[str] = set()
    sources: list[str] = []
    historical = repo_root / "benchmarks" / "historical"
    if historical.exists():
        for run_file in sorted(historical.glob("*/benchmark_run.json")):
            try:
                run = HistoricalBenchmarkRun.model_validate_json(
                    run_file.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            if run.competition != competition or run.year != year:
                continue
            if run.problem_label:
                burned.add(run.problem_label)
                sources.append(run_file.relative_to(repo_root).as_posix())
    # A prior runtime project containing this competition/year must be resolved
    # manually rather than silently reused or silently ignored.
    projects_root = repo_root / "projects"
    if projects_root.exists():
        for spec_file in sorted(projects_root.glob("*/manifests/problem_spec.json")):
            try:
                text = spec_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if str(year) in text and (
                "高教社杯" in text or "CUMCM" in text or "数学建模竞赛" in text
            ):
                raise FreshnessError(
                    "possible prior runtime reveal in "
                    f"{spec_file.relative_to(repo_root).as_posix()}; resolve burn registry manually"
                )
    registry_file = repo_root / "benchmarks" / "burn_registry.json"
    if registry_file.exists():
        existing = BurnRegistry.model_validate_json(registry_file.read_text(encoding="utf-8"))
        if existing.competition == competition and existing.year == year:
            burned.update(existing.burned_labels)
            sources.extend(existing.sources)
    remaining = sorted(set(labels) - burned)
    if not remaining:
        raise FreshnessError(f"all {competition} {year} candidate labels are already burned")
    registry = BurnRegistry(
        competition=competition,
        year=year,
        burned_labels=sorted(burned),
        sources=sorted(set(sources)),
        created_at=(now or datetime.now(UTC)),
    )
    registry = registry.model_copy(
        update={
            "registry_hash": _hash_payload(
                registry.model_dump(mode="json"), exclude=("registry_hash",)
            )
        }
    )
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(registry_file, registry.model_dump_json(indent=2) + "\n")
    return registry


__all__ = ["FreshnessError", "freshness_preflight"]
