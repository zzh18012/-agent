from datetime import date

import polars as pl

from mmos.data import check_leakage, profile_table
from mmos.schemas.data import Severity


def findings_for(df: pl.DataFrame, target: str | None = None):
    table = profile_table("demo", "path", df)
    return check_leakage(table, df, target=target)


def test_id_like_unique_column_is_flagged() -> None:
    findings = findings_for(pl.DataFrame({"row_id": [1, 2, 3], "x": [4, 5, 6]}))
    assert any(finding.code == "id_like_column" for finding in findings)


def test_duplicate_columns_are_flagged() -> None:
    findings = findings_for(pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}))
    assert any(finding.code == "duplicate_columns" for finding in findings)


def test_exact_target_copy_is_blocking() -> None:
    findings = findings_for(pl.DataFrame({"y": [0, 1, 0], "leak": [0, 1, 0]}), target="y")
    blocking = [finding for finding in findings if finding.code == "target_duplicate_column"]
    assert blocking and blocking[0].severity is Severity.ERROR


def test_temporal_column_is_reported_as_info() -> None:
    df = pl.DataFrame({"day": [date(2024, 1, 1), date(2024, 1, 2)], "x": [1, 2]})
    findings = findings_for(df)
    temporal = [finding for finding in findings if finding.code == "temporal_column"]
    assert temporal and temporal[0].severity is Severity.INFO
