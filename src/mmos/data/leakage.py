import polars as pl

from mmos.schemas.data import LeakageFinding, Severity, TableProfile

ID_NAME_TOKENS = ("id", "uuid", "guid", "key", "编号", "序号")
TEMPORAL_NAME_TOKENS = ("date", "time", "timestamp", "日期", "时间")

LEAKAGE_LIMITATIONS = [
    "heuristic checks only; no causal or target-relationship analysis",
    "train/test overlap and temporal leakage require the split design, which does not exist yet",
    "target leakage is only detected as an exact duplicate of the target column",
]


def check_leakage(
    table: TableProfile, df: pl.DataFrame, *, target: str | None = None
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []

    for column in table.columns:
        name = column.name.lower()
        is_unique = (
            table.row_count > 0
            and column.null_count == 0
            and column.unique_count == table.row_count
        )
        if is_unique and any(token in name for token in ID_NAME_TOKENS):
            findings.append(
                LeakageFinding(
                    code="id_like_column",
                    severity=Severity.WARNING,
                    message="column looks like a row identifier; exclude it from features "
                    "unless there is a justified reason",
                    table=table.name,
                    columns=[column.name],
                )
            )

        dtype_is_temporal = "Date" in column.dtype or "Time" in column.dtype
        if dtype_is_temporal or any(token in name for token in TEMPORAL_NAME_TOKENS):
            findings.append(
                LeakageFinding(
                    code="temporal_column",
                    severity=Severity.INFO,
                    message="temporal column; verify that splits do not use future information",
                    table=table.name,
                    columns=[column.name],
                )
            )

    columns = df.columns
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            if df[first].equals(df[second]):
                findings.append(
                    LeakageFinding(
                        code="duplicate_columns",
                        severity=Severity.WARNING,
                        message="columns contain identical values; remove one to avoid "
                        "redundancy and hidden leakage",
                        table=table.name,
                        columns=[first, second],
                    )
                )

    if target is not None and target in df.columns:
        for column in columns:
            if column != target and df[column].equals(df[target]):
                findings.append(
                    LeakageFinding(
                        code="target_duplicate_column",
                        severity=Severity.ERROR,
                        message="column is an exact copy of the target and leaks the label",
                        table=table.name,
                        columns=[column, target],
                    )
                )

    return findings
