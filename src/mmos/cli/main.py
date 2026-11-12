import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mmos import __version__
from mmos.config import get_settings

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
