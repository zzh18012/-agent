from dataclasses import dataclass
from pathlib import Path

import pandera.errors as pa_errors
import pandera.polars as pa
import polars as pl
import structlog

from mmos.core.project import ProjectPaths, atomic_write_text, load_manifest
from mmos.data.leakage import LEAKAGE_LIMITATIONS, check_leakage
from mmos.data.profiling import infer_table_schema, profile_table, read_csv
from mmos.data.quality import build_quality_report, check_table
from mmos.schemas.common import MMOSModel
from mmos.schemas.data import (
    DataQualityReport,
    DataSchema,
    DatasetProfile,
    Issue,
    LeakageReport,
    Severity,
    TableProfile,
)
from mmos.schemas.project import SourceRole

SUPPORTED_DATA_SUFFIXES = {".csv"}


@dataclass(frozen=True)
class DataAnalysisResult:
    profile: DatasetProfile
    schema: DataSchema
    quality: DataQualityReport
    leakage: LeakageReport


def _write_model(path: Path, model: MMOSModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def _schema_validation_issues(df: pl.DataFrame, table: TableProfile) -> list[Issue]:
    """Validate the inferred nullability contract with an independent engine."""
    schema = pa.DataFrameSchema(
        {
            column.name: pa.Column(df.schema[column.name], nullable=column.null_count > 0)
            for column in table.columns
        }
    )
    try:
        schema.validate(df)
    except pa_errors.SchemaError as exc:
        return [
            Issue(
                code="schema_validation_failed",
                severity=Severity.ERROR,
                message=str(exc).splitlines()[0],
                table=table.name,
            )
        ]
    return []


class DataAgent:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self._log = structlog.get_logger("mmos.data")

    def analyze(self, *, target: str | None = None) -> DataAnalysisResult:
        manifest = load_manifest(self.paths)
        data_files = [
            self.paths.root / record.path
            for record in manifest.inputs
            if record.source_role is SourceRole.DATA
        ]
        if not data_files:
            raise ValueError(
                "no data files ingested; run: mmos ingest <project> <file> --role data"
            )
        for data_file in data_files:
            if data_file.suffix.lower() not in SUPPORTED_DATA_SUFFIXES:
                raise ValueError(
                    f"unsupported data file type in Phase 4: {data_file.name} (CSV only)"
                )

        tables: list[TableProfile] = []
        schemas = []
        issues: list[Issue] = []
        findings = []

        for data_file in data_files:
            df = read_csv(data_file)
            table = profile_table(
                data_file.stem,
                data_file.relative_to(self.paths.root).as_posix(),
                df,
            )
            tables.append(table)
            schemas.append(infer_table_schema(table))
            issues.extend(check_table(table, df, target=target))
            issues.extend(_schema_validation_issues(df, table))
            findings.extend(check_leakage(table, df, target=target))

        result = DataAnalysisResult(
            profile=DatasetProfile(tables=tables),
            schema=DataSchema(tables=schemas),
            quality=build_quality_report([table.name for table in tables], issues),
            leakage=LeakageReport(
                status="failed"
                if any(f.severity is Severity.ERROR for f in findings)
                else "passed",
                blocking=any(f.severity is Severity.ERROR for f in findings),
                findings=findings,
                limitations=LEAKAGE_LIMITATIONS,
            ),
        )
        self._save(result)
        self._log.info(
            "data_analysis_completed",
            project_id=self.paths.root.name,
            tables=len(tables),
            blocking=result.quality.blocking or result.leakage.blocking,
        )
        return result

    def _save(self, result: DataAnalysisResult) -> None:
        _write_model(self.paths.data_profile_file, result.profile)
        _write_model(self.paths.data_schema_file, result.schema)
        _write_model(self.paths.data_quality_file, result.quality)
        _write_model(self.paths.leakage_report_file, result.leakage)
