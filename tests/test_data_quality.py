import polars as pl

from mmos.data import build_quality_report, check_table, profile_table
from mmos.schemas.data import Severity


def issues_for(df: pl.DataFrame, target: str | None = None):
    table = profile_table("demo", "path", df)
    return table, check_table(table, df, target=target)


def test_missing_target_is_blocking() -> None:
    table, issues = issues_for(pl.DataFrame({"x": [1, 2]}), target="y")
    missing = [issue for issue in issues if issue.code == "target_missing"]
    assert missing and missing[0].severity is Severity.ERROR
    assert build_quality_report([table.name], issues).blocking is True


def test_high_null_fraction_warns_and_critical_null_fraction_blocks() -> None:
    _, warning_issues = issues_for(pl.DataFrame({"a": [1, None, None, None]}))
    assert any(
        issue.code == "high_null_fraction" and issue.severity is Severity.WARNING
        for issue in warning_issues
    )

    _, error_issues = issues_for(pl.DataFrame({"a": [1] + [None] * 9}))
    assert any(
        issue.code == "critical_null_fraction" and issue.severity is Severity.ERROR
        for issue in error_issues
    )


def test_constant_column_is_reported() -> None:
    _, issues = issues_for(pl.DataFrame({"a": [1, 1, 1]}))
    assert any(issue.code == "constant_column" for issue in issues)


def test_text_column_of_numbers_warns() -> None:
    _, issues = issues_for(pl.DataFrame({"s": ["1", "2", "3"]}))
    assert any(
        issue.code == "numeric_text_column" and issue.severity is Severity.WARNING
        for issue in issues
    )


def test_nan_and_infinite_values_are_flagged() -> None:
    _, issues = issues_for(pl.DataFrame({"a": [1.0, float("nan"), float("inf")]}))
    assert any(issue.code == "nan_values" for issue in issues)
    assert any(
        issue.code == "infinite_values" and issue.severity is Severity.ERROR for issue in issues
    )


def test_duplicate_rows_warn() -> None:
    _, issues = issues_for(pl.DataFrame({"a": [1, 1, 2]}))
    assert any(
        issue.code == "duplicate_rows" and issue.severity is Severity.WARNING for issue in issues
    )


def test_target_nulls_warn() -> None:
    _, issues = issues_for(pl.DataFrame({"y": [1, None, 2]}), target="y")
    assert any(
        issue.code == "target_has_nulls" and issue.severity is Severity.WARNING for issue in issues
    )


def test_clean_table_passes() -> None:
    table, issues = issues_for(pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}), target="y")
    report = build_quality_report([table.name], issues)
    assert report.status.value == "passed"
    assert report.blocking is False
    assert report.issues == []
