import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from mmos import __version__
from mmos.agents import (
    CouncilError,
    MissingAPIKeyError,
    ModelCouncil,
    ProblemAnalyst,
    RedTeam,
    ResearchAgent,
    UnsupportedProviderError,
    build_model,
    release_allowed,
)
from mmos.agents.catalog import EXECUTABLE_MODEL_IDS
from mmos.config import get_settings
from mmos.core.dataset import dataset_fingerprint
from mmos.core.problem_text import collect_problem_text
from mmos.core.project import (
    atomic_write_text,
    create_project,
    experiment_spec_file,
    load_manifest,
    load_problem_spec,
    load_project_config,
    load_proposals,
    load_research_result,
    optimization_spec_file,
    resolve_project,
    save_council_decision,
    save_critiques,
    save_experiment_record,
    save_experiment_spec,
    save_optimization_spec,
    save_problem_spec,
    save_proposals,
    save_red_team_review,
    save_research_result,
    save_tournament_result,
)
from mmos.core.state_machine import allowed_targets
from mmos.data import DataAgent
from mmos.evidence import (
    EvidenceError,
    add_claim,
    freeze_experiment,
    freeze_optimization,
)
from mmos.experiments.optimization_runner import (
    build_experiment_record,
    create_baseline_experiment,
    run_optimization,
)
from mmos.experiments.optimization_tournament import (
    run_optimization_tournament,
)
from mmos.experiments.planner import ExperimentPlanner
from mmos.experiments.tournament import Tournament
from mmos.experiments.workflow import run_experiment_workflow
from mmos.orchestration import Orchestrator
from mmos.paper import PaperError, build_paper, render_paper
from mmos.paper.optimization_builder import build_optimization_paper
from mmos.pipeline import PipelineBlocked, PipelineOptions, run_pipeline
from mmos.registries import ModelRegistry
from mmos.schemas.data import DataSchema
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.optimization import (
    ExperimentStatus,
    OptimizationSpec,
    optimization_instance_hash,
)
from mmos.schemas.project import SourceRole
from mmos.schemas.proposal import CritiqueReport
from mmos.schemas.states import ESCAPE_STATES, SUPPORTED_TASKS, ProjectState, TaskFamily
from mmos.tracking import log_optimization_run, log_optimization_tournament
from mmos.utils.logging import configure_logging
from mmos.validation import validate_experiment
from mmos.validation.optimization import validate_optimization_experiment

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
    task: Annotated[
        TaskFamily,
        typer.Option("--task", help="Task family: prediction | forecasting | optimization."),
    ] = TaskFamily.PREDICTION,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Create a new project with its full directory layout."""
    if task not in SUPPORTED_TASKS:
        console.print(f"Error: task family {task.value!r} is not implemented.", style="red")
        raise typer.Exit(code=1)
    paths, config = create_project(_projects_base(projects_dir), name, task=task)
    console.print(f"Created project [bold]{config.project_id}[/bold]")
    console.print(f"Location: {paths.root}")
    console.print(f"Task: {config.task.value}")
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
    table.add_row("Task", config.task.value)
    table.add_row("State", config.state.value)
    table.add_row("Resume state", config.resume_state.value if config.resume_state else "-")
    table.add_row("Inputs", str(len(manifest.inputs)))
    for role in SourceRole:
        table.add_row(f"  {role.value}", str(role_counts.get(role.value, 0)))
    next_states = ", ".join(
        sorted(
            state.value for state in allowed_targets(config.state, config.resume_state, config.task)
        )
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
        problem_text = collect_problem_text(paths)
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

    try:
        result = run_experiment_workflow(
            paths,
            experiment_id,
            timeout_seconds=timeout,
            dry_run=dry_run,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

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
    improvement_threshold: Annotated[
        float | None,
        typer.Option(
            "--improvement-threshold",
            help="Relative improvement required for practical difference.",
        ),
    ] = None,
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

    predictions = None
    if len(experiments) == 1:
        prediction_file = (
            paths.experiments_dir / experiments[0] / "artifacts" / "predictions.parquet"
        )
        if prediction_file.exists():
            predictions = pl.read_parquet(prediction_file)
    result = Tournament(ModelRegistry.load(registry_file)).run(
        results,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        predictions=predictions,
        improvement_threshold=improvement_threshold,
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

    console.print(f"Selection status: {result.selection_status}")
    console.print(f"Selected: {result.selected_model or '-'}")
    console.print(f"Reference: {result.challenger_model or '-'}")
    for comparison in result.equivalence:
        console.print(
            f"  {comparison.candidate_model} vs {comparison.baseline_model}: "
            f"status={comparison.status}, relative improvement={comparison.relative_improvement}, "
            f"CI=[{comparison.ci_low}, {comparison.ci_high}]"
        )
    if result.blocked:
        console.print("Tournament is blocked; no model was selected.", style="red")
        for note in result.notes:
            console.print(f"  note: {note}")
        raise typer.Exit(code=1)
    console.print(f"Saved: {saved_to}")


@app.command()
def validate(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    experiment_id: Annotated[str, typer.Argument(help="Completed experiment id.")],
    validators: Annotated[
        list[str] | None,
        typer.Option("--validator", help="Validator name; repeat to select a subset."),
    ] = None,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Run independent validators against a completed experiment."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state in _EARLY_STATES or config.state in ESCAPE_STATES:
        console.print(
            f"Project state {config.state.value} does not allow validation; "
            "run an experiment first and resolve any blocking state."
        )
        raise typer.Exit(code=1)

    metrics_file = paths.experiments_dir / experiment_id / "artifacts" / "metrics.json"
    if not metrics_file.exists():
        console.print(f"Error: no metrics for experiment {experiment_id!r}", style="red")
        raise typer.Exit(code=1)

    try:
        reports = validate_experiment(paths, experiment_id, validators)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Validation {experiment_id}")
    table.add_column("Validator")
    table.add_column("Status")
    table.add_column("Severity")
    table.add_column("Blocking")
    table.add_column("Issues", justify="right")
    for report in reports:
        table.add_row(
            report.validator,
            report.status.value,
            report.severity.value,
            "yes" if report.blocking else "no",
            str(len(report.issues)),
        )
    console.print(table)

    for report in reports:
        for issue in report.issues:
            console.print(
                f"  [{issue.severity.value}] {report.validator}/{issue.code}: {issue.message}"
            )

    if any(report.blocking for report in reports):
        console.print("Validation FAILED with blocking issues.", style="red")
        raise typer.Exit(code=1)
    console.print("Validation passed.")


@app.command(name="freeze-evidence")
def freeze_evidence_command(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    experiment_id: Annotated[str, typer.Argument(help="Validated experiment id.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Freeze validated experiment results into the verified evidence store."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state in _EARLY_STATES or config.state in ESCAPE_STATES:
        console.print(f"Project state {config.state.value} does not allow evidence freezing.")
        raise typer.Exit(code=1)

    try:
        records = freeze_experiment(paths, experiment_id)
    except (EvidenceError, FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Verified evidence ({len(records)} records)")
    table.add_column("Evidence id")
    table.add_column("Key")
    table.add_column("Value", justify="right")
    for record in records:
        table.add_row(record.evidence_id, record.key, record.display_value)
    console.print(table)
    console.print(f"Store: {paths.verified_numbers_file}")


@app.command(name="add-claim")
def add_claim_command(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    statement: Annotated[str, typer.Option("--statement", help="Claim statement.")],
    evidence: Annotated[
        list[str] | None,
        typer.Option("--evidence", help="Evidence id; repeat for multiple ids."),
    ] = None,
    claim_id: Annotated[str | None, typer.Option("--claim-id", help="Explicit claim id.")] = None,
    accept: Annotated[
        bool, typer.Option("--accept", help="Mark the claim as supported by a human.")
    ] = False,
    notes: Annotated[str, typer.Option("--notes", help="Reviewer notes.")] = "",
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Register a paper claim backed by verified evidence."""
    if not evidence:
        console.print("Error: at least one --evidence id is required.", style="red")
        raise typer.Exit(code=1)

    paths = resolve_project(project, _projects_base(projects_dir))
    try:
        claim = add_claim(
            paths,
            statement=statement,
            evidence_ids=evidence,
            claim_id=claim_id,
            accept=accept,
            reviewer="cli" if accept else None,
            notes=notes,
        )
    except EvidenceError as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    console.print(f"Claim {claim.claim_id}: {claim.status}")
    console.print(f"Statement: {claim.statement}")
    console.print(f"Saved: {paths.verified_claims_file}")


@app.command()
def paper(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    experiment_id: Annotated[str, typer.Argument(help="Frozen experiment id.")],
    tournament_id: Annotated[
        str | None, typer.Option("--tournament", help="Tournament id to include.")
    ] = None,
    primary_metric: Annotated[
        str, typer.Option("--metric", help="Primary metric for the figure.")
    ] = "rmse",
    higher_is_better: Annotated[
        bool, typer.Option("--higher-is-better", help="Metric direction for the figure.")
    ] = False,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Build and render a Markdown technical report from verified evidence."""
    paths = resolve_project(project, _projects_base(projects_dir))
    try:
        ast = build_paper(
            paths,
            experiment_id,
            tournament_id=tournament_id,
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
        )
        target = render_paper(paths, ast)
    except (PaperError, FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    console.print(f"Title: {ast.title}")
    console.print(
        f"Tables: {len(ast.tables)}  Figures: {len(ast.figures)}  Claims: {len(ast.claims)}"
    )
    console.print(f"Report: {target}")
    for figure in ast.figures:
        console.print(f"Figure: {figure.path}")


def _registry_or_exit() -> ModelRegistry:
    registry_file = get_settings().model_registry_file
    if not registry_file.exists():
        console.print(f"Error: model registry not found: {registry_file}", style="red")
        raise typer.Exit(code=1)
    return ModelRegistry.load(registry_file)


def _model_or_exit():
    try:
        return build_model(get_settings())
    except (MissingAPIKeyError, UnsupportedProviderError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc


@app.command()
def research(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Survey candidate methods for the understood problem."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state not in (ProjectState.DATA_VALIDATED, ProjectState.RESEARCH_COMPLETED):
        console.print(
            f"Project state {config.state.value} does not allow research; "
            "expected DATA_VALIDATED or RESEARCH_COMPLETED."
        )
        raise typer.Exit(code=1)
    if not paths.problem_spec_file.exists():
        console.print("Error: no ProblemSpec found; run analyze-problem first.", style="red")
        raise typer.Exit(code=1)

    spec = load_problem_spec(paths)
    result = ResearchAgent(_model_or_exit()).research(spec, registry=_registry_or_exit())
    save_research_result(paths, result)

    table = Table(title=f"Research methods ({len(result.methods)})")
    table.add_column("Id")
    table.add_column("Method")
    table.add_column("Tools")
    for method in result.methods:
        table.add_row(method.method_id, method.name, ", ".join(method.implementation_tools))
    console.print(table)

    if config.state is ProjectState.DATA_VALIDATED:
        config = Orchestrator(paths).transition(
            ProjectState.RESEARCH_COMPLETED,
            reason=f"research completed ({len(result.methods)} methods)",
        )
    console.print(f"State: {config.state.value}")


@app.command(name="propose-models")
def propose_models(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Run the expert council and record prototypable model proposals."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state not in (ProjectState.RESEARCH_COMPLETED, ProjectState.CANDIDATES_PROPOSED):
        console.print(
            f"Project state {config.state.value} does not allow council proposals; "
            "expected RESEARCH_COMPLETED or CANDIDATES_PROPOSED."
        )
        raise typer.Exit(code=1)
    for required in (paths.problem_spec_file, paths.research_result_file):
        if not required.exists():
            console.print(f"Error: missing artifact: {required.name}", style="red")
            raise typer.Exit(code=1)

    data_summary = None
    if paths.data_schema_file.exists():
        schema = DataSchema.model_validate_json(paths.data_schema_file.read_text(encoding="utf-8"))
        data_summary = "\n".join(
            f"- {table.name}: "
            + ", ".join(f"{column.name} ({column.dtype})" for column in table.columns)
            for table in schema.tables
        )

    try:
        outcome = ModelCouncil(_model_or_exit()).propose(
            load_problem_spec(paths),
            load_research_result(paths),
            registry=_registry_or_exit(),
            data_summary=data_summary,
        )
    except CouncilError as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    save_proposals(paths, outcome.proposals)
    save_council_decision(paths, outcome.decision)
    if outcome.critiques:
        save_critiques(paths, CritiqueReport(critiques=outcome.critiques))

    table = Table(title=f"Council proposals ({len(outcome.proposals)})")
    table.add_column("Proposal id")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Models")
    table.add_column("Selected")
    selected = set(outcome.decision.selected_for_prototype)
    for proposal in outcome.proposals:
        table.add_row(
            proposal.proposal_id,
            proposal.name,
            proposal.expert_role.value,
            ", ".join(proposal.registry_model_ids),
            "yes" if proposal.proposal_id in selected else "no",
        )
    console.print(table)
    for note in outcome.notes:
        console.print(f"  note: {note}")

    if config.state is ProjectState.RESEARCH_COMPLETED:
        config = Orchestrator(paths).transition(
            ProjectState.CANDIDATES_PROPOSED,
            reason=f"council selected {len(outcome.decision.selected_for_prototype)} proposal(s)",
        )
    console.print(f"State: {config.state.value}")


@app.command(name="plan-experiment")
def plan_experiment(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    proposal_id: Annotated[str, typer.Option("--proposal-id", help="Proposal to plan.")],
    target: Annotated[str, typer.Option("--target", help="Target column.")],
    features: Annotated[
        list[str] | None,
        typer.Option("--feature", help="Feature column; repeat for multiple."),
    ] = None,
    metrics: Annotated[
        list[str] | None,
        typer.Option("--metric", help="Metric; repeat to override defaults."),
    ] = None,
    n_splits: Annotated[int, typer.Option("--splits", help="Cross-validation folds.")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Turn one council proposal into a reproducible experiment spec."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.state not in (
        ProjectState.CANDIDATES_PROPOSED,
        ProjectState.PROTOTYPING,
    ):
        console.print(
            f"Project state {config.state.value} does not allow experiment planning; "
            "expected CANDIDATES_PROPOSED or PROTOTYPING."
        )
        raise typer.Exit(code=1)
    if not features:
        console.print("Error: at least one --feature is required.", style="red")
        raise typer.Exit(code=1)
    if not paths.proposals_file.exists():
        console.print("Error: no proposals found; run propose-models first.", style="red")
        raise typer.Exit(code=1)

    proposals = {proposal.proposal_id: proposal for proposal in load_proposals(paths)}
    if proposal_id not in proposals:
        console.print(f"Error: unknown proposal id: {proposal_id}", style="red")
        raise typer.Exit(code=1)

    try:
        spec = ExperimentPlanner(_registry_or_exit()).plan(
            proposals[proposal_id],
            dataset_hash=dataset_fingerprint(load_manifest(paths)),
            target_column=target,
            feature_set=features,
            metrics=metrics,
            n_splits=n_splits,
            random_seed=seed,
            include_baseline=True,
        )
    except (ValueError, KeyError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    non_executable = [model_id for model_id in spec.models if model_id not in EXECUTABLE_MODEL_IDS]
    if non_executable:
        console.print(
            "Error: registered models without an executable engine: " + ", ".join(non_executable),
            style="red",
        )
        raise typer.Exit(code=1)

    target_path = save_experiment_spec(paths, spec)
    console.print(f"Experiment: {spec.experiment_id}")
    console.print(f"Models: {', '.join(spec.models)}")
    console.print(f"Metrics: {', '.join(spec.metrics)}")
    console.print(f"Spec: {target_path}")

    if config.state is ProjectState.CANDIDATES_PROPOSED:
        config = Orchestrator(paths).transition(
            ProjectState.PROTOTYPING,
            reason=f"experiment planned for {proposal_id}",
        )
    console.print(f"State: {config.state.value}")


@app.command()
def redteam(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    experiment_id: Annotated[str, typer.Argument(help="Frozen experiment id.")],
    tournament_id: Annotated[
        str | None, typer.Option("--tournament", help="Tournament id to include.")
    ] = None,
    primary_metric: Annotated[
        str, typer.Option("--metric", help="Primary metric used in the report.")
    ] = "rmse",
    higher_is_better: Annotated[
        bool, typer.Option("--higher-is-better", help="Metric direction.")
    ] = False,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Adversarial review, author response, and judge verdict."""
    paths = resolve_project(project, _projects_base(projects_dir))
    try:
        ast = build_paper(
            paths,
            experiment_id,
            tournament_id=tournament_id,
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
        )
        review = RedTeam(_model_or_exit()).run(ast, experiment_id=experiment_id)
    except (PaperError, FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    target_path = save_red_team_review(paths, review)

    attacks_table = Table(title=f"Red team attacks ({len(review.attacks.attacks)})")
    attacks_table.add_column("Id")
    attacks_table.add_column("Severity")
    attacks_table.add_column("Category")
    attacks_table.add_column("Target")
    attacks_table.add_column("Fix")
    for attack in review.attacks.attacks:
        attacks_table.add_row(
            attack.attack_id,
            attack.severity.value,
            attack.category.value,
            attack.target,
            attack.required_fix,
        )
    console.print(attacks_table)

    verdict_table = Table(title="Judge verdicts")
    verdict_table.add_column("Attack")
    verdict_table.add_column("Verdict")
    verdict_table.add_column("Rationale")
    for verdict in review.judge.verdicts:
        verdict_table.add_row(verdict.attack_id, verdict.verdict.value, verdict.rationale)
    console.print(verdict_table)

    console.print(f"Overall: {review.judge.overall.value}")
    console.print(f"Blocking: {'yes' if review.judge.blocking else 'no'}")
    console.print(f"Saved: {target_path}")
    if not release_allowed(review):
        console.print(
            f"Release blocked: judge overall is {review.judge.overall.value}; PASS required.",
            style="red",
        )
        raise typer.Exit(code=1)


@app.command()
def run(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    target: Annotated[str, typer.Option("--target", help="Target column.")],
    features: Annotated[
        list[str] | None,
        typer.Option("--feature", help="Feature column; repeat for multiple."),
    ] = None,
    primary_metric: Annotated[
        str, typer.Option("--primary-metric", help="Metric used for selection and figures.")
    ] = "rmse",
    higher_is_better: Annotated[
        bool, typer.Option("--higher-is-better", help="Metric direction.")
    ] = False,
    metrics: Annotated[
        list[str] | None,
        typer.Option("--metric", help="Metric to compute; repeat to override defaults."),
    ] = None,
    n_splits: Annotated[int, typer.Option("--splits", help="Cross-validation folds.")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    time_column: Annotated[
        str | None, typer.Option("--time-column", help="Time column for temporal data.")
    ] = None,
    forecast_horizon: Annotated[
        int | None,
        typer.Option("--horizon", help="Forecast horizon; required for temporal data."),
    ] = None,
    seasonal_period: Annotated[
        int | None,
        typer.Option("--seasonal-period", help="Confirmed seasonal period, if any."),
    ] = None,
    improvement_threshold: Annotated[
        float | None,
        typer.Option(
            "--improvement-threshold",
            help="Relative improvement required for practical difference.",
        ),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Workflow timeout in seconds.")
    ] = 900.0,
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Run the full pipeline from problem analysis to a release summary."""
    paths = resolve_project(project, _projects_base(projects_dir))
    options = PipelineOptions(
        target=target,
        features=features or [],
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        metrics=metrics,
        n_splits=n_splits,
        seed=seed,
        timeout_seconds=timeout,
        time_column=time_column,
        forecast_horizon=forecast_horizon,
        seasonal_period=seasonal_period,
        improvement_threshold=improvement_threshold,
    )

    try:
        summary = run_pipeline(paths, options)
    except (
        PipelineBlocked,
        ValueError,
        FileNotFoundError,
        RuntimeError,
        EvidenceError,
        PaperError,
        CouncilError,
    ) as exc:
        console.print(f"Pipeline stopped: {exc}", style="red")
        raise typer.Exit(code=1) from exc

    console.print(f"Selected model: {summary.selected_model}")
    console.print(
        f"{summary.primary_metric}: {summary.metric_display_value} "
        f"(challenger: {summary.challenger_model or '-'})"
    )
    console.print(f"Validation: {', '.join(sorted(summary.validation))}")
    console.print(f"Red team: {summary.red_team_overall}")
    console.print(
        f"Claims: {len(summary.claim_ids)} registered, {summary.supported_claims} supported"
    )
    console.print(f"Paper: {paths.root / summary.paper_path}")
    console.print(f"Release: {paths.releases_dir / (summary.release_id + '.md')}")


@app.command()
def optimize(
    project: Annotated[str, typer.Argument(help="Project id or directory.")],
    spec_file: Annotated[Path, typer.Argument(help="OptimizationSpec JSON file.")],
    projects_dir: Annotated[
        Path | None, typer.Option("--projects-dir", help="Base directory for projects.")
    ] = None,
) -> None:
    """Run a MILP experiment through the shared evidence, paper, and review pipeline."""
    paths = resolve_project(project, _projects_base(projects_dir))
    config = load_project_config(paths)
    if config.task is not TaskFamily.OPTIMIZATION:
        console.print(
            f"Error: project task is {config.task.value}; create it with --task optimization.",
            style="red",
        )
        raise typer.Exit(code=1)
    if config.state is not ProjectState.PROBLEM_UNDERSTOOD:
        console.print(
            f"Error: optimization requires PROBLEM_UNDERSTOOD (currently "
            f"{config.state.value}); run analyze-problem first.",
            style="red",
        )
        raise typer.Exit(code=1)
    if not spec_file.exists():
        console.print(f"Error: optimization spec not found: {spec_file}", style="red")
        raise typer.Exit(code=1)
    spec = OptimizationSpec.model_validate_json(spec_file.read_text(encoding="utf-8"))
    target_spec = optimization_spec_file(paths, spec.experiment_id)
    if target_spec.exists():
        console.print(f"Error: experiment already exists: {spec.experiment_id}", style="red")
        raise typer.Exit(code=1)

    spec = spec.model_copy(update={"instance_hash": optimization_instance_hash(spec.model)})
    orchestrator = Orchestrator(paths)
    save_optimization_spec(paths, spec)
    orchestrator.transition(ProjectState.PROTOTYPING, reason="optimization spec planned")
    result = run_optimization(paths, spec.experiment_id)

    baseline_spec, baseline_result = create_baseline_experiment(paths, spec, result)
    baseline_id = baseline_spec.experiment_id
    candidate_record = build_experiment_record(
        paths,
        spec,
        result,
        status=ExperimentStatus.SUCCEEDED,
        linked_experiment_ids=[baseline_id],
    )
    candidate_mlflow = log_optimization_run(paths, spec, result, candidate_record)
    candidate_record = candidate_record.model_copy(
        update={"mlflow_run_id": candidate_mlflow.run_id}
    )
    save_experiment_record(paths, candidate_record)

    baseline_record = build_experiment_record(
        paths, baseline_spec, baseline_result, status=ExperimentStatus.SUCCEEDED
    )
    baseline_mlflow = log_optimization_run(paths, baseline_spec, baseline_result, baseline_record)
    baseline_record = baseline_record.model_copy(update={"mlflow_run_id": baseline_mlflow.run_id})
    save_experiment_record(paths, baseline_record)

    orchestrator.advance_to(
        ProjectState.EXPERIMENT_COMPLETED, reason="optimization experiment completed"
    )
    orchestrator.advance_to(ProjectState.VALIDATING, reason="optimization validation started")
    reports = validate_optimization_experiment(paths, spec.experiment_id)

    table = Table(title=f"Optimization {spec.experiment_id}")
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Objective", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Feasible")
    for solution in result.solutions:
        table.add_row(
            solution.method,
            solution.status,
            f"{solution.objective_value:.6g}" if solution.objective_value is not None else "-",
            f"{solution.optimality_gap:.6g}" if solution.optimality_gap is not None else "-",
            "yes" if solution.feasible else "no",
        )
    console.print(table)
    for report in reports:
        for issue in report.issues:
            console.print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")
    if any(report.blocking for report in reports):
        save_experiment_record(
            paths,
            candidate_record.model_copy(update={"status": ExperimentStatus.INVALID}),
        )
        orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="optimization validation failed",
        )
        console.print("Optimization validation FAILED.", style="red")
        raise typer.Exit(code=1)
    candidate_record = candidate_record.model_copy(
        update={
            "status": ExperimentStatus.VALIDATED,
            "validation_status": {report.validator: report.status.value for report in reports},
        }
    )
    save_experiment_record(paths, candidate_record)
    orchestrator.advance_to(ProjectState.VERIFIED, reason="optimization validators passed")

    tournament = run_optimization_tournament(paths, spec.experiment_id, baseline_id)
    tournament_mlflow = log_optimization_tournament(paths, tournament)

    records = freeze_optimization(paths, spec.experiment_id, tournament_id=tournament.tournament_id)
    orchestrator.advance_to(ProjectState.EVIDENCE_FROZEN, reason="optimization evidence frozen")
    console.print(f"Verified evidence records: {len(records)}")
    solver_solution = next(
        solution for solution in result.solutions if solution.method == "ortools.cbc"
    )
    objective = solver_solution.objective_value
    claim_id = f"claim-auto-{spec.experiment_id}"
    claim_note = "Auto-generated from the optimization tournament; human acceptance is required."
    tournament_evidence = {
        "winner": f"ev-{spec.experiment_id}-tournament-winner",
        "candidate_objective": f"ev-{spec.experiment_id}-tournament-candidate-objective",
        "baseline_objective": f"ev-{spec.experiment_id}-tournament-baseline-objective",
        "status": f"ev-{spec.experiment_id}-tournament-status",
    }
    try:
        if (
            tournament.winner_status.value == "winner"
            and tournament.winner_experiment_id == spec.experiment_id
        ):
            add_claim(
                paths,
                statement=(
                    f"{spec.experiment_id} achieved a higher objective than the "
                    "greedy sanity baseline on the evaluated benchmark instance."
                ),
                evidence_ids=[
                    tournament_evidence["winner"],
                    tournament_evidence["candidate_objective"],
                    tournament_evidence["baseline_objective"],
                ],
                claim_id=claim_id,
                accept=False,
                notes=claim_note,
            )
        elif tournament.winner_status.value == "equivalent":
            add_claim(
                paths,
                statement=(
                    f"{spec.experiment_id} and {baseline_id} are objective-equivalent "
                    "on the evaluated benchmark instance."
                ),
                evidence_ids=[
                    tournament_evidence["status"],
                    tournament_evidence["candidate_objective"],
                    tournament_evidence["baseline_objective"],
                ],
                claim_id=claim_id,
                accept=False,
                notes=claim_note,
            )
        else:
            console.print(
                "No superiority claim registered: tournament did not select a winner.",
                style="yellow",
            )
    except EvidenceError as exc:
        console.print(f"Warning: claim not registered: {exc}", style="yellow")

    ast = build_optimization_paper(paths, spec.experiment_id)
    report_path = render_paper(paths, ast)
    orchestrator.advance_to(ProjectState.PAPER_DRAFTED, reason="optimization paper rendered")
    orchestrator.advance_to(ProjectState.RED_TEAM_REVIEW, reason="optimization red team started")
    review = RedTeam(_model_or_exit()).run(ast, experiment_id=spec.experiment_id)
    review_path = save_red_team_review(paths, review)

    console.print(f"Paper: {report_path}")
    console.print(f"Red team: {review.judge.overall.value} (blocking: {review.judge.blocking})")
    if not release_allowed(review):
        blocked_path = paths.reviews_dir / f"blocked_optimization_{spec.experiment_id}.md"
        lines = [f"# Optimization release blocked: {spec.experiment_id}", ""]
        for attack in review.attacks.attacks:
            verdict = next(
                verdict
                for verdict in review.judge.verdicts
                if verdict.attack_id == attack.attack_id
            )
            lines.extend(
                [
                    f"## {attack.attack_id}: {attack.severity.value} / "
                    f"{attack.category.value} -> {verdict.verdict.value}",
                    attack.argument,
                    f"Required fix: {attack.required_fix}",
                    f"Judge: {verdict.rationale}",
                    "",
                ]
            )
        atomic_write_text(blocked_path, "\n".join(lines))
        orchestrator.transition(
            ProjectState.REQUIRES_HUMAN_REVIEW,
            reason="optimization red team did not pass",
        )
        console.print(
            f"Release blocked; judge is {review.judge.overall.value}. See {blocked_path}",
            style="red",
        )
        raise typer.Exit(code=1)

    release = {
        "experiment_id": spec.experiment_id,
        "baseline_experiment_id": baseline_id,
        "mlflow_run_id": candidate_mlflow.run_id,
        "baseline_mlflow_run_id": baseline_mlflow.run_id,
        "instance_hash": spec.instance_hash,
        "tournament_id": tournament.tournament_id,
        "tournament_status": tournament.winner_status.value,
        "winner_experiment_id": tournament.winner_experiment_id,
        "tournament_mlflow_run_id": tournament_mlflow.run_id,
        "model": spec.model.name,
        "sense": spec.model.sense,
        "selected_method": result.selected_method,
        "solver_status": solver_solution.status,
        "objective": objective,
        "optimality_gap": solver_solution.optimality_gap,
        "feasible": solver_solution.feasible,
        "validation": {report.validator: report.status.value for report in reports},
        "red_team": review.judge.overall.value,
        "paper": report_path.relative_to(paths.root).as_posix(),
        "review": review_path.relative_to(paths.root).as_posix(),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    orchestrator.advance_to(ProjectState.READY_FOR_RELEASE, reason="optimization red team passed")
    paths.releases_dir.mkdir(parents=True, exist_ok=True)
    release_json = paths.releases_dir / f"optimization_{spec.experiment_id}.json"
    release_md = paths.releases_dir / f"optimization_{spec.experiment_id}.md"
    atomic_write_text(release_json, json.dumps(release, indent=2) + "\n")
    atomic_write_text(
        release_md,
        f"# Optimization release {spec.experiment_id}\n\n"
        f"- Model: `{spec.model.name}` ({spec.model.sense})\n"
        f"- Selected method: `{result.selected_method}`\n"
        f"- Solver status: `{solver_solution.status}`\n"
        f"- Objective: `{objective}`\n"
        f"- Optimality gap: `{solver_solution.optimality_gap}`\n"
        f"- Feasible: `{solver_solution.feasible}`\n"
        f"- Red team: `{review.judge.overall.value}`\n"
        f"- Paper: `{report_path.relative_to(paths.root).as_posix()}`\n",
    )
    orchestrator.advance_to(ProjectState.RELEASED, reason="optimization release written")
    console.print(f"Project: {paths.root.name}")
    console.print(f"Candidate experiment: {spec.experiment_id}")
    console.print(f"Baseline experiment: {baseline_id}")
    console.print(f"MLflow run: {candidate_mlflow.run_id}")
    console.print(f"Baseline MLflow run: {baseline_mlflow.run_id}")
    console.print(f"Instance hash: {spec.instance_hash}")
    console.print(f"Tournament: {tournament.tournament_id}")
    console.print(f"Tournament status: {tournament.winner_status.value}")
    console.print(f"Tournament winner: {tournament.winner_experiment_id or 'none'}")
    console.print(f"Tournament MLflow run: {tournament_mlflow.run_id}")
    console.print(f"Solver status: {solver_solution.status}")
    console.print(f"Objective: {objective}")
    console.print(f"Feasible: {solver_solution.feasible}")
    console.print(f"Validation: {candidate_record.validation_status}")
    console.print(f"State: {load_project_config(paths).state.value}")
    console.print(f"Release: {release_md}")
