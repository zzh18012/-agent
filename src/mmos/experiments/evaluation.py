import math

import numpy as np

from mmos.schemas.experiment import EvaluationRecord, MaseFoldDetail


def regression_evaluation(
    model_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mase_details: list[MaseFoldDetail] | None = None,
) -> EvaluationRecord:
    """All formal regression metrics come from this one pooled evaluation set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        raise ValueError("evaluation set is empty")
    errors = y_pred - y_true
    sse = float(np.sum(errors**2))
    target_mean = float(np.mean(y_true))
    sst = float(np.sum((y_true - target_mean) ** 2))
    rmse = math.sqrt(sse / n)
    mae = float(np.mean(np.abs(errors)))

    r2: float | None = None
    r2_reason: str | None = None
    if sst > 0:
        r2 = 1.0 - sse / sst
    else:
        r2_reason = "constant evaluation target (SST = 0)"

    std_population = math.sqrt(sst / n)
    ratio: float | None = None
    if std_population > 0:
        ratio = rmse / std_population

    mase: float | None = None
    if mase_details:
        scaled = [detail.scaled_error for detail in mase_details if detail.scaled_error is not None]
        if scaled:
            mase = float(np.mean(scaled))

    return EvaluationRecord(
        model_id=model_id,
        n_observations=n,
        sse=sse,
        sst=sst,
        r2=r2,
        r2_reason=r2_reason,
        target_mean=target_mean,
        target_std_population=std_population,
        rmse=rmse,
        mae=mae,
        rmse_eval_std_ratio=ratio,
        mase=mase,
        mase_details=mase_details or [],
    )


def metrics_from_evaluation(record: EvaluationRecord) -> dict[str, float]:
    metrics: dict[str, float] = {"rmse": record.rmse, "mae": record.mae}  # type: ignore[dict-item]
    if record.r2 is not None:
        metrics["r2"] = record.r2
    if record.mase is not None:
        metrics["mase"] = record.mase
    if record.rmse_eval_std_ratio is not None:
        metrics["rmse_eval_std_ratio"] = record.rmse_eval_std_ratio
    return metrics
