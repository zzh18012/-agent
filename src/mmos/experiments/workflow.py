import shutil
from pathlib import Path

from mmos.core.project import ProjectPaths
from mmos.execution import LocalExecutor
from mmos.schemas.execution import ExecutionResult


def default_workflow_file() -> Path:
    candidates = (
        Path("workflow/Snakefile"),
        Path(__file__).resolve().parents[3] / "workflow" / "Snakefile",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("workflow/Snakefile not found")


def run_experiment_workflow(
    paths: ProjectPaths,
    experiment_id: str,
    *,
    timeout_seconds: float = 600.0,
    dry_run: bool = False,
) -> ExecutionResult:
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        raise RuntimeError("snakemake is not installed in this environment")
    workflow_file = default_workflow_file()
    return LocalExecutor(paths).run(
        [
            snakemake,
            "-s",
            str(workflow_file),
            "--cores",
            "1",
            "--config",
            f"experiment_id={experiment_id}",
        ],
        cwd=paths.root,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
    )
