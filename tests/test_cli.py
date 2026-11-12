from typer.testing import CliRunner

from mmos import __version__
from mmos.cli.main import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_reports_python() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.stdout


def test_init_ingest_and_status_flow(tmp_path) -> None:
    init_result = runner.invoke(app, ["init", "My Demo", "--projects-dir", str(tmp_path)])
    assert init_result.exit_code == 0, init_result.stdout
    assert "INIT" in init_result.stdout

    project_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    project_id = project_dir.name

    source = tmp_path / "train.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")

    ingest_result = runner.invoke(
        app,
        ["ingest", project_id, str(source), "--role", "data", "--projects-dir", str(tmp_path)],
    )
    assert ingest_result.exit_code == 0, ingest_result.stdout
    assert "PROJECT_INGESTED" in ingest_result.stdout

    status_result = runner.invoke(app, ["status", project_id, "--projects-dir", str(tmp_path)])
    assert status_result.exit_code == 0, status_result.stdout
    assert "PROJECT_INGESTED" in status_result.stdout
