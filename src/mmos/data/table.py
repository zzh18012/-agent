from pathlib import Path

import polars as pl

from mmos.core.project import ProjectPaths, load_manifest
from mmos.schemas.project import SourceRole


def data_files(paths: ProjectPaths) -> list[Path]:
    manifest = load_manifest(paths)
    return [
        paths.root / record.path
        for record in manifest.inputs
        if record.source_role is SourceRole.DATA
    ]


def load_single_table(paths: ProjectPaths) -> pl.DataFrame:
    files = data_files(paths)
    if len(files) != 1:
        raise ValueError(
            f"expected exactly one data table, found {len(files)}; "
            "multi-table experiments are not implemented yet"
        )
    return pl.read_csv(files[0], try_parse_dates=True)
