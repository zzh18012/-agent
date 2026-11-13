import json
from pathlib import Path

import pytest

from mmos.core.project import create_project
from mmos.evidence import freeze_benchmark_suite
from mmos.experiments.knapsack_benchmark import (
    ManifestError,
    build_instances,
    build_manifest,
    enumerate_knapsack,
    instance_model,
    load_manifest,
)
from mmos.experiments.knapsack_suite import _aggregate, _relative_regret, run_knapsack_suite
from mmos.experiments.optimization_runner import solve_enhanced_greedy, solve_greedy
from mmos.paper.benchmark_builder import build_benchmark_paper
from mmos.schemas.benchmark import knapsack_instance_hash
from mmos.schemas.states import TaskFamily


def test_manifest_is_deterministic_and_seed_sensitive() -> None:
    first = build_manifest(20261113)
    second = build_manifest(20261113)
    assert first.manifest_hash == second.manifest_hash

    changed = build_manifest(20261114)
    curated = {
        instance.instance_id: instance.instance_hash
        for instance in first.instances
        if instance.generator.startswith("curated") or instance.generator == "legacy_manual"
    }
    changed_curated = {
        instance.instance_id: instance.instance_hash
        for instance in changed.instances
        if instance.generator.startswith("curated") or instance.generator == "legacy_manual"
    }
    assert curated == changed_curated
    assert first.manifest_hash != changed.manifest_hash


def test_instance_hash_tracks_data_changes() -> None:
    base = knapsack_instance_hash([1, 2], [3, 4], 5)
    assert base == knapsack_instance_hash([1, 2], [3, 4], 5)
    assert base != knapsack_instance_hash([1, 3], [3, 4], 5)
    assert base != knapsack_instance_hash([1, 2], [3, 5], 5)
    assert base != knapsack_instance_hash([1, 2], [3, 4], 6)


def test_frozen_manifest_rejects_tampering(tmp_path: Path) -> None:
    manifest = build_manifest()
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "instances").mkdir()
    for instance in manifest.instances:
        (suite_dir / "instances" / f"{instance.instance_id}.json").write_text(
            instance.model_dump_json(indent=2), encoding="utf-8"
        )
    manifest_path = suite_dir / "benchmark_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    load_manifest(suite_dir)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["instances"][0]["weights"][0] += 1
    payload["instances"][0]["instance_hash"] = knapsack_instance_hash(
        payload["instances"][0]["weights"],
        payload["instances"][0]["values"],
        payload["instances"][0]["capacity"],
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="hash mismatch"):
        load_manifest(suite_dir)


def test_greedy_trap_beats_ratio_greedy() -> None:
    instance = next(
        instance for instance in build_instances() if instance.instance_id == "greedy_ratio_trap"
    )
    oracle = enumerate_knapsack(instance)
    greedy = solve_greedy(instance_model(instance))

    assert oracle.verified_optimum == 18
    assert greedy.objective_value == 12
    assert greedy.objective_value < oracle.verified_optimum


def test_enhanced_greedy_is_at_least_ratio_greedy() -> None:
    for instance in build_instances():
        model = instance_model(instance)
        greedy = solve_greedy(model)
        enhanced = solve_enhanced_greedy(model)
        if greedy.feasible and greedy.objective_value is not None:
            assert enhanced.feasible
            assert enhanced.objective_value is not None
            assert enhanced.objective_value >= greedy.objective_value


def test_regret_formulas() -> None:
    assert _relative_regret(100, 80, True) == pytest.approx(0.2)
    assert _relative_regret(100, 100, True) == 0.0
    assert _relative_regret(0, 0, True) is None
    assert _relative_regret(100, None, False) is None


def test_aggregate_known_values() -> None:
    rows = [
        ("a", 100.0, True, 100.0),
        ("b", 80.0, True, 100.0),
        ("c", None, False, 100.0),
    ]
    aggregate = _aggregate("test", "sanity_baseline", rows)

    assert aggregate.optimum_hits == 1
    assert aggregate.failures == 2
    assert aggregate.hit_rate == pytest.approx(1 / 3)
    assert aggregate.mean_relative_regret == pytest.approx(0.1)
    assert aggregate.median_relative_regret == pytest.approx(0.1)
    assert aggregate.max_relative_regret == pytest.approx(0.2)
    assert aggregate.worst_instance_id == "b"


def test_full_suite_vertical_slice(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    paths, _ = create_project(projects_dir, "suite", task=TaskFamily.OPTIMIZATION)
    manifest = build_manifest()
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "instances").mkdir()
    for instance in manifest.instances:
        (suite_dir / "instances" / f"{instance.instance_id}.json").write_text(
            instance.model_dump_json(indent=2), encoding="utf-8"
        )
    (suite_dir / "benchmark_manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    result = run_knapsack_suite(paths, suite_dir, time_limit_seconds=5)

    assert result.status == "complete"
    assert result.n_instances == len(manifest.instances)
    assert result.oracle_agreement is True
    assert result.candidate_verified_optimum_count == result.n_instances
    assert result.candidate_feasible_count == result.n_instances
    assert len(result.per_instance) == result.n_instances
    assert all(instance.tournament_id for instance in result.per_instance)
    greedy_trap = next(
        instance for instance in result.per_instance if instance.instance_id == "greedy_ratio_trap"
    )
    assert greedy_trap.cbc_verified is True
    assert greedy_trap.greedy_relative_regret and greedy_trap.greedy_relative_regret > 0

    artifact = (
        paths.experiments_dir / "benchmarks" / result.benchmark_run_id / "benchmark_result.json"
    )
    records = freeze_benchmark_suite(paths, result, artifact)
    keys = {record.key for record in records}
    assert "benchmark.manifest_hash" in keys
    assert "benchmark.greedy.max_relative_regret" in keys
    assert "benchmark.instance.greedy_ratio_trap.oracle_optimum" in keys

    paper = build_benchmark_paper(paths, result.benchmark_run_id)
    assert len(paper.tables) == 2
    assert len(paper.tables[0].rows) == result.n_instances
    titles = " ".join(section.heading for section in paper.sections)
    assert "Benchmark purpose" in titles
    assert "Adversarial findings" in titles
