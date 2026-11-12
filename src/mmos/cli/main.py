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
    load_manifest,
    load_project_config,
    resolve_project,
    save_problem_spec,
)
from mmos.core.state_machine import allowed_targets
from mmos.orchestration import Orchestrator
from mmos.schemas.project import SourceRole
from mmos.schemas.states import ProjectState
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
