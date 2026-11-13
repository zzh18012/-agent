from pathlib import Path

import pytest

from mmos.core.project import create_project
from mmos.experiments.knapsack_benchmark import DEFAULT_SUITE_DIR
from mmos.experiments.knapsack_suite import run_knapsack_suite
from mmos.schemas.states import TaskFamily


@pytest.fixture(scope="session")
def benchmark_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    base = tmp_path_factory.mktemp("benchmark-template")
    paths, _ = create_project(
        base / "projects", "benchmark-template", task=TaskFamily.OPTIMIZATION
    )
    result = run_knapsack_suite(paths, DEFAULT_SUITE_DIR, time_limit_seconds=5)
    assert result.status == "complete"
    return paths.root, result.benchmark_run_id
