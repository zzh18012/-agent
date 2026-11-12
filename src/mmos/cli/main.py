import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mmos import __version__
from mmos.agents import MissingAPIKeyError, ProblemAnalyst, UnsupportedProviderError, build_model
from mmos.config import get_settings
from mmos.core.project import (
    ProjectPaths,
    create_project,
    experiment_spec_file,
    load_manifest,
    load_project_config,
    resolve_project,
    save_problem_spec,
    save_tournament_result,
)
from mmos.core.state_machine import allowed_targets
from mmos.data import DataAgent
from mmos.execution import LocalExecutor
from mmos.experiments.tournament import Tournament
from mmos.orchestration import Orchestrator
from mmos.registries import ModelRegistry
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.project import SourceRole
from mmos.schemas.states import ESCAPE_STATES, ProjectState
from mmos.utils.logging import configure_logging

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mmos {__version__}")
        raise typer.Exit()


def _git_version() -> str:
    executable = shutil.which("git")
    if executable is None:
        return "not found"
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {exc}"
    return result.stdout.strip() or "unknown"


def _projects_base(projects_dir: Path | None) -> Path:
    return projects_dir if projects_dir is not None else get_settings().projects_dir


_TEXT_PROBLEM_SUFFIXES = {".md", ".markdown", ".txt"}


def _collect_problem_text(paths: ProjectPaths) -> str:
    manifest = load_manifest(paths)
    problem_files = [
        paths.root / record.path
        for record in manifest.inputs
        if record.source_role is SourceRole.PROBLEM
    ]
    if not problem_files:
        raise ValueError(
            "no problem files ingested; run: mmos ingest <project> <file> --role problem"
        )

    sections: list[str] = []
    for problem_file in problem_files:
        if problem_file.suffix.lower() not in _TEXT_PROBLEM_SUFFIXES:
            raise ValueError(
                f"unsupported problem file type: {problem_file.name} "
                "(Phase 3 accepts .md, .markdown, .txt; PDF/Word parsing comes later)"
            )
        sections.append(f"## {problem_file.name}\n\n{problem_file.read_text(encoding='utf-8')}")
    return "\n\n".join(sections)


app = typer.Typer(
    name="mmos",
    help="MathModelingOS — evidence-driven mathematical modeling automation.",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """MathModelingOS command line interface."""
    configure_logging(get_settings().log_level)


@app.command()
def doctor() -> None:
    """Check the local environment MathModelingOS depends on."""
    settings = get_settings()

    table = Table(title="MathModelingOS environment")
    table.add_column("Check")
    table.add_column("Value")
    table.add_column("Status")

    python_ok = sys.version_info >= (3, 12)
    table.add_row("Python", sys.version.split()[0], "ok" if python_ok else "too old (need 3.12+)")

    uv_path = shutil.which("uv")
    table.add_row("uv", uv_path or "not found", "ok" if uv_path else "missing")

    git_version = _git_version()
    table.add_row("git", git_version, "ok" if git_version != "not found" else "missing")

    table.add_row(
        "Virtualenv", sys.prefix, "active" if sys.prefix != sys.base_prefix else "base Python"
    )

    env_file = Path(".env")
    table.add_row(".env", str(env_file.resolve()), "present" if env_file.exists() else "absent")

    api_key_status = "configured" if settings.llm_api_key is not None else "not configured"
    table.add_row("LLM API key", api_key_status, "ok" if settings.llm_api_key else "optional")
    table.add_row("LLM", f"{settings.llm_provider} / {settings.llm_model}", "ok")

    console.print(table)


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Human-readable project name.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Create a new project with its full directory layout."""
    paths, config = create_project(_projects_base(projects_dir), name)
    console.print(f"Created project [bold]{config.project_id}[/bold]")
    console.print(f"Location: {paths.root}")
    console.print(f"State: {config.state.value}")


@app.command()
def ingest(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    files: Annotated[list[Path], typer.Argument(help="Files to ingest.")],
    role: Annotated[
        SourceRole, typer.Option("--role", help="problem | rules | data | attachment")
    ] = SourceRole.DATA,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Copy input files into a project and register their hashes."""
    paths = resolve_project(project, _projects_base(projects_dir))
    orchestrator = Orchestrator(paths)
    records, state = orchestrator.ingest(files, role)

    if not records:
        console.print("No new files ingested.")
    for record in records:
        console.print(f"+ {record.path}  sha256={record.sha256[:12]}...  {record.size_bytes} bytes")
    console.print(f"State: {state.value}")


@app.command()
def status(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Show a project's state, inputs, and legal next states."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    manifest = load_manifest(paths)
    role_counts = Counter(record.source_role.value for record in manifest.inputs)

    table = Table(title=f"Project {config.project_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Name", config.name)
    table.add_row("State", config.state.value)
    table.add_row("Resume state", config.resume_state.value if config.resume_state else "-")
    table.add_row("Inputs", str(len(manifest.inputs)))
    for role in SourceRole:
        table.add_row(f"  {role.value}", str(role_counts.get(role.value, 0)))
    next_states = ", ".join(
        sorted(state.value for state in allowed_targets(config.state, config.resume_state))
    )
    table.add_row("Allowed next", next_states or "none")
    console.print(table)


@app.command(name="analyze-problem")
def analyze_problem(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Extract a structured ProblemSpec from the ingested problem statement."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state not in (ProjectState.PROJECT_INGESTED, ProjectState.PROBLEM_UNDERSTOOD):
        console.print(
            f"Project state {config.state.value} does not allow problem analysis; "
            "expected PROJECT_INGESTED or PROBLEM_UNDERSTOOD."
        )
        raise typer.Exit(code=1)

    try:
        problem_text = _collect_problem_text(paths)
        model = build_model(get_settings())
    except (ValueError, MissingAPIKeyError, UnsupportedProviderError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    spec = ProblemAnalyst(model).analyze(problem_text, problem_id=config.project_id)
    save_problem_spec(paths, spec)
    console.print(f"Saved ProblemSpec: {paths.problem_spec_file}")
    console.print(f"Subproblems: {len(spec.subproblems)}")

    if config.state is ProjectState.PROJECT_INGESTED:
        config = Orchestrator(paths).transition(
            ProjectState.PROBLEM_UNDERSTOOD,
            reason=f"problem analyzed ({len(spec.subproblems)} subproblems)",
        )
    console.print(f"State: {config.state.value}")


@app.command(name="analyze-data")
def analyze_data(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    target: Annotated[
        str | None, typer.Option("--target", help="Target column to validate against.")
    ] = None,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Profile ingested data and run the data-quality and leakage gates."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state not in (ProjectState.PROBLEM_UNDERSTOOD, ProjectState.DATA_VALIDATED):
        console.print(
            f"Project state {config.state.value} does not allow data analysis; "
            "expected PROBLEM_UNDERSTOOD or DATA_VALIDATED."
        )
        raise typer.Exit(code=1)

    try:
        result = DataAgent(paths).analyze(target=target)
    except (ValueError, OSError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    summary = Table(title="Data profile")
    summary.add_column("Table")
    summary.add_column("Rows", justify="right")
    summary.add_column("Columns", justify="right")
    summary.add_column("Duplicate rows", justify="right")
    for table_profile in result.profile.tables:
        summary.add_row(
            table_profile.name,
            str(table_profile.row_count),
            str(table_profile.column_count),
            str(table_profile.duplicate_row_count),
        )
    console.print(summary)

    if result.quality.issues:
        issues = Table(title="Data quality issues")
        issues.add_column("Severity")
        issues.add_column("Code")
        issues.add_column("Table")
        issues.add_column("Column")
        issues.add_column("Message")
        for issue in result.quality.issues:
            issues.add_row(
                issue.severity.value,
                issue.code,
                issue.table or "-",
                issue.column or "-",
                issue.message,
            )
        console.print(issues)

    if result.leakage.findings:
        findings = Table(title="Leakage findings")
        findings.add_column("Severity")
        findings.add_column("Code")
        findings.add_column("Columns")
        findings.add_column("Message")
        for finding in result.leakage.findings:
            findings.add_row(
                finding.severity.value,
                finding.code,
                ", ".join(finding.columns) or "-",
                finding.message,
            )
        console.print(findings)

    console.print(f"Quality: {result.quality.status.value}")
    console.print(f"Leakage: {result.leakage.status}")
    console.print(f"Artifacts: {paths.manifest_dir}")

    if result.quality.blocking or result.leakage.blocking:
        console.print("Data quality gate FAILED; project was not advanced.", style="red")
        raise typer.Exit(code=1)

    if config.state is ProjectState.PROBLEM_UNDERSTOOD:
        config = Orchestrator(paths).transition(
            ProjectState.DATA_VALIDATED,
            reason=f"data validated ({len(result.profile.tables)} tables)",
        )
    console.print(f"State: {config.state.value}")


_EARLY_STATES = frozenset(
    {ProjectState.INIT, ProjectState.PROJECT_INGESTED, ProjectState.PROBLEM_UNDERSTOOD}
)


@app.command(name="run-experiment")
def run_experiment_command(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    experiment_id: Annotated[str, typer.Argument(help="Planned experiment id.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Record the workflow command without running it.")
    ] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Workflow timeout in seconds.")
    ] = 600.0,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Execute a planned experiment through the Snakemake workflow."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state in _EARLY_STATES or config.state in ESCAPE_STATES:
        console.print(
            f"Project state {config.state.value} does not allow experiments; "
            "validate the data first and resolve any blocking state."
        )
        raise typer.Exit(code=1)

    spec_file = experiment_spec_file(paths, experiment_id)
    if not spec_file.exists():
        console.print(f"Error: experiment spec not found: {spec_file}", style="red")
        raise typer.Exit(code=1)

    snakemake = shutil.which("snakemake")
    if snakemake is None:
        console.print("Error: snakemake is not installed in this environment.", style="red")
        raise typer.Exit(code=1)

    workflow_file = Path("workflow/Snakefile")
    if not workflow_file.exists():
        console.print(f"Error: workflow file not found: {workflow_file}", style="red")
        raise typer.Exit(code=1)

    result = LocalExecutor(paths).run(
        [
            snakemake,
            "-s",
            str(workflow_file.resolve()),
            "--cores",
            "1",
            "--config",
            f"experiment_id={experiment_id}",
        ],
        cwd=paths.root,
        timeout_seconds=timeout,
        dry_run=dry_run,
    )

    console.print(f"Execution: {result.execution_id}")
    console.print(f"Duration: {result.duration_seconds:.2f}s")
    console.print(f"stdout: {result.stdout_path}")
    console.print(f"stderr: {result.stderr_path}")

    if dry_run:
        console.print("Dry run: workflow was not executed.")
        return
    if not result.succeeded():
        console.print(f"Workflow failed: {result.error or 'non-zero exit'}", style="red")
        raise typer.Exit(code=result.exit_code or 1)

    artifacts = paths.experiments_dir / experiment_id / "artifacts"
    console.print(f"Metrics: {artifacts / 'metrics.json'}")
    console.print(f"Predictions: {artifacts / 'predictions.parquet'}")


@app.command()
def tournament(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    experiments: Annotated[list[str], typer.Argument(help="Completed experiment ids.")],
    primary_metric: Annotated[
        str, typer.Option("--metric", help="Primary metric used for ranking.")
    ] = "rmse",
    higher_is_better: Annotated[
        bool, typer.Option("--higher-is-better", help="Rank larger metric values first.")
    ] = False,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Rank models from comparable experiments against a baseline."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state in _EARLY_STATES or config.state in ESCAPE_STATES:
        console.print(
            f"Project state {config.state.value} does not allow a tournament; "
            "run experiments first and resolve any blocking state."
        )
        raise typer.Exit(code=1)

    results: list[ExperimentResult] = []
    for experiment_id in experiments:
        metrics_file = paths.experiments_dir / experiment_id / "artifacts" / "metrics.json"
        if not metrics_file.exists():
            console.print(
                f"Error: no metrics for experiment {experiment_id!r}: {metrics_file}",
                style="red",
            )
            raise typer.Exit(code=1)
        results.append(
            ExperimentResult.model_validate_json(metrics_file.read_text(encoding="utf-8"))
        )

    registry_file = get_settings().model_registry_file
    if not registry_file.exists():
        console.print(f"Error: model registry not found: {registry_file}", style="red")
        raise typer.Exit(code=1)

    result = Tournament(ModelRegistry.load(registry_file)).run(
        results,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
    )
    saved_to = save_tournament_result(paths, result)

    table = Table(title=f"Tournament {result.tournament_id}")
    table.add_column("Rank", justify="right")
    table.add_column("Model")
    table.add_column(primary_metric, justify="right")
    table.add_column("Status")
    table.add_column("Reason")
    for entry in result.entries:
        table.add_row(
            str(entry.rank),
            entry.model_id,
            f"{entry.metrics[primary_metric]:.6g}",
            entry.status,
            entry.reason,
        )
    console.print(table)

    console.print(f"Selected: {result.selected_model or '-'}")
    console.print(f"Challenger: {result.challenger_model or '-'}")
    if result.blocked:
        console.print("Tournament is blocked; no model was selected.", style="red")
        for note in result.notes:
            console.print(f"  note: {note}")
        raise typer.Exit(code=1)
    console.print(f"Saved: {saved_to}")
