import polars as pl
import pytest

from mmos.data import infer_table_schema, profile_table


def test_profile_reports_shape_nulls_and_statistics() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": ["x", "y", "x"],
            "c": [1.0, 1.0, 1.0],
        }
    )

    table = profile_table("demo", "data/raw/demo.csv", df)

    assert table.row_count == 3
    assert table.column_count == 3
    assert table.duplicate_row_count == 0

    column_a = next(column for column in table.columns if column.name == "a")
    assert column_a.null_count == 1
    assert column_a.null_fraction == pytest.approx(1 / 3)
    assert column_a.min == "1"
    assert column_a.max == "2"

    column_c = next(column for column in table.columns if column.name == "c")
    assert column_c.mean == pytest.approx(1.0)
    assert column_c.std == pytest.approx(0.0)
    assert column_c.p50 == pytest.approx(1.0)


def test_duplicate_row_count_counts_repeated_rows() -> None:
    df = pl.DataFrame({"a": [1, 2, 2]})
    table = profile_table("demo", "path", df)
    assert table.duplicate_row_count == 1


def test_inferred_schema_finds_unique_non_null_candidate_keys() -> None:
    df = pl.DataFrame({"row_id": [1, 2, 3], "x": [5, 5, 6]})
    table = profile_table("demo", "path", df)

    schema = infer_table_schema(table)

    assert schema.candidate_keys == ["row_id"]
    assert all(not column.nullable for column in schema.columns)


def test_inferred_schema_marks_nullable_columns() -> None:
    df = pl.DataFrame({"a": [1, None]})
    schema = infer_table_schema(profile_table("demo", "path", df))
    assert schema.columns[0].nullable is True
