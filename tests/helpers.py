import math
from pathlib import Path

import polars as pl

from mmos.core.dataset import dataset_fingerprint
from mmos.core.intake import ingest_files
from mmos.core.project import (
    ProjectPaths,
    create_project,
    load_manifest,
    save_experiment_spec,
)
from mmos.schemas.experiment import ExperimentSpec
from mmos.schemas.project import SourceRole


def regression_frame(rows: int = 40) -> pl.DataFrame:
    x1 = [float(index % 10) for index in range(rows)]
    x2 = [float(index % 7) for index in range(rows)]
    y = [
        3.0 * a + 2.0 * b + 0.01 * math.sin(index)
        for index, (a, b) in enumerate(zip(x1, x2, strict=True))
    ]
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def classification_frame(rows: int = 60) -> pl.DataFrame:
    x1 = [float(index % 12) - 6.0 for index in range(rows)]
    x2 = [float((index * 5) % 9) - 4.0 for index in range(rows)]
    y = [1 if a + 0.5 * b > 0 else 0 for a, b in zip(x1, x2, strict=True)]
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def make_experiment_project(
    tmp_path: Path,
    df: pl.DataFrame,
    *,
    models: list[str],
    metrics: list[str],
    feature_set: list[str] | None = None,
    target_column: str = "y",
    experiment_id: str = "exp-test",
) -> tuple[ProjectPaths, ExperimentSpec]:
    paths, _ = create_project(tmp_path / "projects", "demo")
    source = tmp_path / "data.csv"
    df.write_csv(source)
    ingest_files(paths, [source], SourceRole.DATA)

    spec = ExperimentSpec(
        experiment_id=experiment_id,
        problem_id="demo",
        proposal_id="p1",
        dataset_hash=dataset_fingerprint(load_manifest(paths)),
        target_column=target_column,
        feature_set=feature_set or ["x1", "x2"],
        split_strategy="kfold",
        validation_strategy="cross_validation",
        n_splits=3,
        models=models,
        metrics=metrics,
        random_seed=0,
        required_artifacts=["metrics.json", "predictions.parquet"],
        acceptance_criteria=["beat the baseline"],
        failure_criteria=["worse than baseline"],
        compute_requirements="test",
        expected_runtime_class="seconds",
    )
    save_experiment_spec(paths, spec)
    return paths, spec
