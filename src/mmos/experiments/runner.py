import argparse
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold

from mmos.core.dataset import dataset_fingerprint
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    load_experiment_spec,
    load_manifest,
)
from mmos.experiments.engines import build_estimator, task_type_for
from mmos.experiments.metrics import compute_metrics
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.project import SourceRole
from mmos.schemas.registry import TaskType
from mmos.tracking import log_experiment_run


def _data_files(paths: ProjectPaths) -> list[Path]:
    manifest = load_manifest(paths)
    return [
        paths.root / record.path
        for record in manifest.inputs
        if record.source_role is SourceRole.DATA
    ]


def _load_single_table(paths: ProjectPaths) -> pl.DataFrame:
    data_files = _data_files(paths)
    if len(data_files) != 1:
        raise ValueError(
            f"the executable runner expects exactly one data table, found {len(data_files)}; "
            "multi-table experiments are not implemented yet"
        )
    return pl.read_csv(data_files[0], try_parse_dates=True)


def run_experiment(paths: ProjectPaths, experiment_id: str) -> ExperimentResult:
    spec = load_experiment_spec(paths, experiment_id)
    manifest = load_manifest(paths)
    fingerprint = dataset_fingerprint(manifest)
    if fingerprint != spec.dataset_hash:
        raise ValueError(
            "dataset hash mismatch: the project data changed after this experiment was planned "
            f"(spec {spec.dataset_hash[:12]}, current {fingerprint[:12]})"
        )

    df = _load_single_table(paths)
    required = [*spec.feature_set, spec.target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns in data: {', '.join(missing)}")

    features = df.select(spec.feature_set)
    if features.null_count().to_numpy().sum() > 0 or df[spec.target_column].null_count() > 0:
        raise ValueError(
            "null values in features or target; cleaning must be an explicit, recorded step"
        )

    x = features.to_numpy()
    y = df[spec.target_column].to_numpy()
    task_type = task_type_for(spec.models[0])
    for model_id in spec.models[1:]:
        if task_type_for(model_id) is not task_type:
            raise ValueError(f"experiment mixes task types: {spec.models[0]!r} and {model_id!r}")

    if task_type is TaskType.CLASSIFICATION:
        splitter = StratifiedKFold(
            n_splits=spec.n_splits, shuffle=True, random_state=spec.random_seed
        )
        split_iter = splitter.split(x, y)
        classes = np.unique(y)
        class_positions = {value: index for index, value in enumerate(classes)}
    elif task_type is TaskType.REGRESSION:
        splitter = KFold(n_splits=spec.n_splits, shuffle=True, random_state=spec.random_seed)
        split_iter = splitter.split(x, y)
        classes = None
        class_positions = {}
    else:
        raise NotImplementedError(
            f"cross-validated execution is not implemented for {task_type.value}"
        )

    folds = list(split_iter)
    metrics_by_model: dict[str, dict[str, float]] = {}
    prediction_rows: list[dict[str, object]] = []

    for model_id in spec.models:
        oof_pred = np.full(len(y), np.nan)
        oof_proba = np.full((len(y), len(classes)), np.nan) if classes is not None else None

        for train_index, test_index in folds:
            estimator = build_estimator(model_id, random_state=spec.random_seed)
            estimator.fit(x[train_index], y[train_index])
            oof_pred[test_index] = estimator.predict(x[test_index])

            if oof_proba is not None:
                fold_proba = estimator.predict_proba(x[test_index])
                for column_index, class_value in enumerate(estimator.classes_):
                    oof_proba[test_index, class_positions[class_value]] = fold_proba[
                        :, column_index
                    ]

        if np.isnan(oof_pred).any():
            raise RuntimeError(f"incomplete out-of-fold predictions for {model_id}")
        if oof_proba is not None and np.isnan(oof_proba).any():
            raise RuntimeError(f"incomplete out-of-fold probabilities for {model_id}")

        metrics_by_model[model_id] = compute_metrics(
            task_type, y, oof_pred, oof_proba, spec.metrics
        )
        prediction_rows.extend(
            {"row_index": index, "model": model_id, "prediction": float(oof_pred[index])}
            for index in range(len(y))
        )

    result = ExperimentResult(
        experiment_id=spec.experiment_id,
        dataset_hash=spec.dataset_hash,
        task_type=task_type,
        random_seed=spec.random_seed,
        n_splits=spec.n_splits,
        models=metrics_by_model,
        generated_at=datetime.now(UTC).replace(microsecond=0),
    )

    experiment_dir = paths.experiments_dir / experiment_id
    artifacts = experiment_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_write_text(artifacts / "metrics.json", result.model_dump_json(indent=2) + "\n")
    pl.DataFrame(prediction_rows).write_parquet(artifacts / "predictions.parquet")
    run_record = {
        "experiment_id": experiment_id,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "created_at": result.generated_at.isoformat(),
    }
    atomic_write_text(experiment_dir / "run_record.json", json.dumps(run_record, indent=2) + "\n")
    log_experiment_run(paths, spec, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one planned experiment.")
    parser.add_argument("--project", required=True, help="Project directory.")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)

    paths = ProjectPaths(Path(args.project).resolve())
    result = run_experiment(paths, args.experiment_id)
    print(json.dumps(result.models, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
