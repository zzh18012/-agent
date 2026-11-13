import itertools
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from mmos.core.project import atomic_write_text
from mmos.schemas.benchmark import (
    BenchmarkManifest,
    KnapsackInstance,
    OracleResult,
    knapsack_instance_hash,
)
from mmos.schemas.optimization import (
    DecisionVariable,
    LinearConstraint,
    OptimizationModel,
    VariableType,
)

SUITE_ID = "knapsack-suite-v1"
SUITE_VERSION = "1.0.0"
GENERATOR_VERSION = "knapsack-generator-v1"
DEFAULT_SUITE_SEED = 20261113
DEFAULT_SUITE_DIR = Path("benchmarks/knapsack/suite_v1")

METHODS = ["ortools.cbc", "greedy_feasible", "enhanced_greedy_feasible"]
ORACLE_METHOD = "exhaustive_enumeration"


class ManifestError(RuntimeError):
    pass


def _instance(
    instance_id: str,
    weights: list[int],
    values: list[int],
    capacity: int,
    *,
    generator: str,
    seed: int | None = None,
    tags: list[str],
) -> KnapsackInstance:
    return KnapsackInstance(
        instance_id=instance_id,
        weights=weights,
        values=values,
        capacity=capacity,
        generator=generator,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        tags=tags,
        instance_hash=knapsack_instance_hash(weights, values, capacity),
    )


def build_instances(suite_seed: int = DEFAULT_SUITE_SEED) -> list[KnapsackInstance]:
    instances: list[KnapsackInstance] = [
        _instance(
            "sanity_legacy_10",
            [10, 20, 30, 15, 12, 8, 6, 5, 4, 3],
            [60, 100, 120, 90, 80, 70, 50, 40, 30, 20],
            50,
            generator="legacy_manual",
            tags=["sanity", "legacy"],
        ),
        _instance(
            "greedy_ratio_trap",
            [6, 5, 5],
            [12, 9, 9],
            10,
            generator="curated_trap",
            tags=["adversarial", "greedy_ratio_trap"],
        ),
        _instance(
            "capacity_fragmentation",
            [5, 11, 10, 4, 7, 11, 9, 12],
            [9, 5, 35, 21, 40, 19, 17, 35],
            32,
            generator="curated_adversarial",
            tags=["adversarial", "capacity_fragmentation"],
        ),
        _instance(
            "tie_stress",
            [2, 4, 6],
            [4, 8, 12],
            12,
            generator="curated_tie",
            tags=["adversarial", "tie_stress"],
        ),
    ]

    def add_random(instance_id: str, seed: int, family: str) -> None:
        rng = random.Random(seed)
        weights = [rng.randint(1, 20) for _ in range(10)]
        if family == "uniform":
            values = [rng.randint(1, 20) for _ in range(10)]
        elif family == "correlated":
            weights = [rng.randint(5, 20) for _ in range(10)]
            values = [max(1, 3 * weight + rng.randint(-3, 3)) for weight in weights]
        else:
            values = [max(1, weight + rng.randint(-5, 5)) for weight in weights]
        capacity = max(1, sum(weights) // 2)
        instances.append(
            _instance(
                instance_id,
                weights,
                values,
                capacity,
                generator=f"seeded_{family}",
                seed=seed,
                tags=["random", family],
            )
        )

    random_specs = [
        ("random_uniform_1", suite_seed + 101, "uniform"),
        ("random_uniform_2", suite_seed + 202, "uniform"),
        ("random_correlated_1", suite_seed + 303, "correlated"),
        ("random_correlated_2", suite_seed + 404, "correlated"),
        ("random_weak_1", suite_seed + 505, "weakly_correlated"),
    ]
    for instance_id, seed, family in random_specs:
        add_random(instance_id, seed, family)
    return instances


def build_manifest(suite_seed: int = DEFAULT_SUITE_SEED) -> BenchmarkManifest:
    instances = build_instances(suite_seed)
    manifest = BenchmarkManifest(
        suite_id=SUITE_ID,
        suite_version=SUITE_VERSION,
        created_at=datetime.now(UTC).replace(microsecond=0),
        generator_version=GENERATOR_VERSION,
        suite_seed=suite_seed,
        instances=instances,
        methods=METHODS,
        oracle_method=ORACLE_METHOD,
        manifest_hash="0" * 64,
    )
    return manifest.model_copy(update={"manifest_hash": manifest.computed_hash()})


def freeze_manifest(
    suite_dir: Path = DEFAULT_SUITE_DIR, *, suite_seed: int = DEFAULT_SUITE_SEED
) -> tuple[BenchmarkManifest, Path]:
    manifest_path = suite_dir / "benchmark_manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(suite_dir)
        return manifest, manifest_path

    manifest = build_manifest(suite_seed)
    instances_dir = suite_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    for instance in manifest.instances:
        atomic_write_text(
            instances_dir / f"{instance.instance_id}.json",
            instance.model_dump_json(indent=2) + "\n",
        )
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return manifest, manifest_path


def load_manifest(suite_dir: Path = DEFAULT_SUITE_DIR) -> BenchmarkManifest:
    manifest_path = suite_dir / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise ManifestError(f"benchmark manifest not found: {manifest_path}")
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if not manifest.frozen:
        raise ManifestError("benchmark manifest is not frozen")
    expected = manifest.computed_hash()
    if manifest.manifest_hash != expected:
        raise ManifestError(
            "benchmark manifest hash mismatch; create a new suite version instead of "
            "editing a frozen manifest"
        )
    return manifest


def instance_model(instance: KnapsackInstance) -> OptimizationModel:
    names = [f"x_{index}" for index in range(len(instance.weights))]
    return OptimizationModel(
        name=f"knapsack_{instance.instance_id}",
        sense="maximize",
        variables=[
            DecisionVariable(name=name, type=VariableType.BINARY, lower=0.0, upper=1.0)
            for name in names
        ],
        objective={name: float(value) for name, value in zip(names, instance.values, strict=True)},
        constraints=[
            LinearConstraint(
                name="capacity",
                coefficients={
                    name: float(weight)
                    for name, weight in zip(names, instance.weights, strict=True)
                },
                upper=float(instance.capacity),
            )
        ],
    )


def enumerate_knapsack(instance: KnapsackInstance) -> OracleResult:
    """Independent exhaustive oracle; it never calls a solver or reads solver values."""
    n_items = len(instance.weights)
    if n_items > 20:
        raise ValueError("exhaustive oracle is limited to 20 items")
    started = time.perf_counter()
    best_value = float("-inf")
    best_subset: list[int] = []
    feasible_count = 0
    for bits in itertools.product([0, 1], repeat=n_items):
        weight = sum(instance.weights[index] * bit for index, bit in enumerate(bits))
        if weight > instance.capacity:
            continue
        feasible_count += 1
        value = sum(instance.values[index] * bit for index, bit in enumerate(bits))
        if value > best_value:
            best_value = float(value)
            best_subset = [index for index, bit in enumerate(bits) if bit]
    return OracleResult(
        instance_id=instance.instance_id,
        verified_optimum=best_value,
        optimal_subset=best_subset,
        search_space_size=2**n_items,
        feasible_subset_count=feasible_count,
        runtime_seconds=time.perf_counter() - started,
    )


def write_manifest_copy(manifest: BenchmarkManifest, path: Path) -> None:
    atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")


def load_suite_result(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data
