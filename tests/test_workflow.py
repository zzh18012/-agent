import shutil
from pathlib import Path

import pytest

from mmos.core.dataset import dataset_fingerprint
from mmos.core.intake import ingest_files
from mmos.core.project import (
    create_project,
    load_manifest,
    save_experiment_spec,
)
from mmos.execution import LocalExecutor
from mmos.schemas.experiment import ExperimentSpec
from mmos.schemas.project import SourceRole

WORKFLOW_FILE = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"


def test_snakemake_workflow_runs_gate_then_experiment(tmp_path: Path) -> None:
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("snakemake is not installed")

    paths, _ = create_project(tmp_path / "projects", "workflow-demo")
    source = tmp_path / "data.csv"
    source.write_text(
        "\n".join(
            ["x1,x2,y"]
            + [
                f"{index % 10},{index % 7},{3 * (index % 10) + 2 * (index % 7)}"
                for index in range(30)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ingest_files(paths, [source], SourceRole.DATA)

    spec = ExperimentSpec(
        experiment_id="exp-workflow",
        problem_id="demo",
        proposal_id="p1",
        dataset_hash=dataset_fingerprint(load_manifest(paths)),
        target_column="y",
        feature_set=["x1", "x2"],
        split_strategy="kfold",
        validation_strategy="cross_validation",
        n_splits=3,
        models=["sklearn.dummy_regressor", "sklearn.linear_regression"],
        metrics=["rmse", "mae"],
        random_seed=0,
        required_artifacts=["metrics.json", "predictions.parquet"],
        acceptance_criteria=["beat baseline"],
        failure_criteria=["worse than baseline"],
        compute_requirements="test",
        expected_runtime_class="seconds",
    )
    save_experiment_spec(paths, spec)

    result = LocalExecutor(paths).run(
        [
            snakemake,
            "-s",
            str(WORKFLOW_FILE),
            "--cores",
            "1",
            "--config",
            f"experiment_id={spec.experiment_id}",
        ],
        cwd=paths.root,
        timeout_seconds=300,
    )

    stderr = (paths.root / result.stderr_path).read_text(encoding="utf-8", errors="replace")
    stdout = (paths.root / result.stdout_path).read_text(encoding="utf-8", errors="replace")
    assert result.succeeded(), f"snakemake failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    artifacts = paths.experiments_dir / spec.experiment_id / "artifacts"
    assert (artifacts / "data_gate.json").exists()
    assert (artifacts / "metrics.json").exists()
    assert (artifacts / "predictions.parquet").exists()
    assert (paths.experiments_dir / spec.experiment_id / "run_record.json").exists()
