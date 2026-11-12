import polars as pl

from mmos.schemas.data import (
    DataQualityReport,
    Issue,
    QualityStatus,
    Severity,
    TableProfile,
)

# Thresholds are deliberately explicit so a reader can argue with them instead of
# discovering hidden heuristics. ERROR findings block the data-quality gate.
HIGH_NULL_FRACTION = 0.5
CRITICAL_NULL_FRACTION = 0.9
NUMERIC_TEXT_RATIO = 0.9


def check_table(table: TableProfile, df: pl.DataFrame, *, target: str | None = None) -> list[Issue]:
    issues: list[Issue] = []

    def add(
        code: str,
        severity: Severity,
        message: str,
        column: str | None = None,
    ) -> None:
        issues.append(
            Issue(
                code=code,
                severity=severity,
                message=message,
                table=table.name,
                column=column,
            )
        )

    if table.row_count == 0:
        add("empty_table", Severity.ERROR, "table has no rows")

    if table.duplicate_row_count > 0:
        add(
            "duplicate_rows",
            Severity.WARNING,
            f"{table.duplicate_row_count} duplicate row(s) found",
        )

    if target is not None and target not in df.columns:
        add("target_missing", Severity.ERROR, f"target column {target!r} is not present")

    for column in table.columns:
        series = df[column.name]

        if column.null_count > 0 and column.null_fraction >= CRITICAL_NULL_FRACTION:
            add(
                "critical_null_fraction",
                Severity.ERROR,
                f"{column.null_fraction:.0%} of values are null",
                column.name,
            )
        elif column.null_count > 0 and column.null_fraction >= HIGH_NULL_FRACTION:
            add(
                "high_null_fraction",
                Severity.WARNING,
                f"{column.null_fraction:.0%} of values are null",
                column.name,
            )

        if table.row_count > 0 and column.unique_count == 1:
            add("constant_column", Severity.INFO, "column has a single distinct value", column.name)

        if series.dtype.is_float():
            nan_count = int(series.is_nan().sum())
            infinite_count = int(series.is_infinite().sum())
            if nan_count:
                add("nan_values", Severity.WARNING, f"{nan_count} NaN value(s)", column.name)
            if infinite_count:
                add(
                    "infinite_values",
                    Severity.ERROR,
                    f"{infinite_count} infinite value(s)",
                    column.name,
                )

        non_null = table.row_count - column.null_count
        if series.dtype == pl.String and non_null > 0:
            numeric_ratio = (
                int(series.cast(pl.Float64, strict=False).is_not_null().sum()) / non_null
            )
            if numeric_ratio >= NUMERIC_TEXT_RATIO and column.unique_count > 1:
                add(
                    "numeric_text_column",
                    Severity.WARNING,
                    f"{numeric_ratio:.0%} of values parse as numbers but the column is text",
                    column.name,
                )

    if target is not None and target in df.columns:
        target_profile = next((column for column in table.columns if column.name == target), None)
        if target_profile is not None and target_profile.null_count > 0:
            severity = Severity.ERROR if target_profile.null_fraction >= 1.0 else Severity.WARNING
            add("target_has_nulls", severity, "target column contains nulls", target)

    return issues


def build_quality_report(tables_checked: list[str], issues: list[Issue]) -> DataQualityReport:
    blocking = any(issue.severity is Severity.ERROR for issue in issues)
    return DataQualityReport(
        status=QualityStatus.FAILED if blocking else QualityStatus.PASSED,
        blocking=blocking,
        tables_checked=tables_checked,
        issues=issues,
    )
