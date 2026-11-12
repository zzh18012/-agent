import json
import math
from pathlib import Path

import polars as pl

from mmos.core.project import ProjectPaths, load_manifest
from mmos.schemas.metadata import DatasetMetadata, TargetSummary, UnitStatus

METADATA_FILENAME = "dataset_metadata.json"


def load_dataset_metadata(paths: ProjectPaths) -> DatasetMetadata | None:
    manifest = load_manifest(paths)
    for record in manifest.inputs:
        if Path(record.path).name == METADATA_FILENAME:
            path = paths.root / record.path
            return DatasetMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    return None


def _finite(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def dataset_metadata_record(paths: ProjectPaths):
    manifest = load_manifest(paths)
    for record in manifest.inputs:
        if Path(record.path).name == METADATA_FILENAME:
            return record
    return None


def build_target_summary(
    df: pl.DataFrame, target: str, metadata: DatasetMetadata | None
) -> TargetSummary:
    series = df[target]
    dtype_name = str(series.dtype)
    numeric = series.dtype.is_numeric()
    non_null = series.drop_nulls()

    mean = _finite(series.mean()) if numeric else None
    std = _finite(series.std()) if numeric else None
    median = _finite(series.median()) if numeric else None
    minimum = _finite(series.min()) if numeric else None
    maximum = _finite(series.max()) if numeric else None

    is_positive: bool | None = None
    if numeric and non_null.len() > 0:
        is_positive = bool((non_null > 0).all())

    return TargetSummary(
        name=target,
        description=metadata.target_description if metadata else None,
        unit=metadata.target_unit if metadata else None,
        unit_status=metadata.unit_status if metadata else UnitStatus.UNKNOWN,
        dtype=dtype_name,
        mean=mean,
        std=std,
        min=minimum,
        max=maximum,
        median=median,
        is_positive=is_positive,
        is_synthetic=metadata.provenance.is_synthetic if metadata else False,
        generator=metadata.provenance.generator if metadata else None,
        generation_seed=metadata.provenance.seed if metadata else None,
        generation_code_path=metadata.provenance.code_path if metadata else None,
        generation_parameters=(
            json.dumps(
                metadata.provenance.generation_parameters, sort_keys=True, ensure_ascii=False
            )
            if metadata and metadata.provenance.generation_parameters
            else None
        ),
    )
