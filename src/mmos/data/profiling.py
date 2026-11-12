import math
from pathlib import Path

import polars as pl

from mmos.schemas.data import (
    ColumnProfile,
    ColumnSpec,
    TableProfile,
    TableSchema,
)


def read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def profile_table(name: str, path_label: str, df: pl.DataFrame) -> TableProfile:
    row_count = df.height
    columns: list[ColumnProfile] = []

    for column_name in df.columns:
        series = df[column_name]
        null_count = series.null_count()
        is_numeric = series.dtype.is_numeric()
        columns.append(
            ColumnProfile(
                name=column_name,
                dtype=str(series.dtype),
                null_count=null_count,
                null_fraction=null_count / row_count if row_count else 0.0,
                unique_count=series.n_unique(),
                min=_text(series.min()),
                max=_text(series.max()),
                mean=_finite_float(series.mean()) if is_numeric else None,
                std=_finite_float(series.std()) if is_numeric else None,
                p50=_finite_float(series.quantile(0.5)) if is_numeric else None,
            )
        )

    return TableProfile(
        name=name,
        path=path_label,
        row_count=row_count,
        column_count=df.width,
        duplicate_row_count=df.height - df.unique().height,
        columns=columns,
    )


def infer_table_schema(table: TableProfile) -> TableSchema:
    candidate_keys = [
        column.name
        for column in table.columns
        if table.row_count > 0 and column.null_count == 0 and column.unique_count == table.row_count
    ]
    return TableSchema(
        name=table.name,
        path=table.path,
        columns=[
            ColumnSpec(name=column.name, dtype=column.dtype, nullable=column.null_count > 0)
            for column in table.columns
        ],
        candidate_keys=candidate_keys,
    )
