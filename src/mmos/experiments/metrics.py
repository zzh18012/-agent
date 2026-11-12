import math
from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from mmos.schemas.registry import TaskType


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric {name!r} is not finite: {number}")
    return number


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, requested: list[str]
) -> dict[str, float]:
    definitions: dict[str, Callable[[], float]] = {
        "rmse": lambda: math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": lambda: float(mean_absolute_error(y_true, y_pred)),
        "r2": lambda: float(r2_score(y_true, y_pred)),
    }
    return _select(definitions, requested)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    requested: list[str],
) -> dict[str, float]:
    def roc_auc() -> float:
        if y_proba is None:
            raise ValueError("roc_auc requires probabilities")
        if y_proba.shape[1] > 2:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        return float(roc_auc_score(y_true, y_proba[:, 1]))

    definitions: dict[str, Callable[[], float]] = {
        "accuracy": lambda: float(accuracy_score(y_true, y_pred)),
        "f1_macro": lambda: float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)  # type: ignore[arg-type]
        ),
        "roc_auc": roc_auc,
    }
    return _select(definitions, requested)


def _select(definitions: dict[str, Callable[[], float]], requested: list[str]) -> dict[str, float]:
    unknown = sorted(set(requested) - set(definitions))
    if unknown:
        raise ValueError(
            f"unsupported metric(s): {', '.join(unknown)}; "
            f"supported: {', '.join(sorted(definitions))}"
        )
    return {name: _finite(definitions[name](), name) for name in requested}


def compute_metrics(
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    requested: list[str],
) -> dict[str, float]:
    if task_type is TaskType.REGRESSION:
        return regression_metrics(y_true, y_pred, requested)
    if task_type is TaskType.CLASSIFICATION:
        return classification_metrics(y_true, y_pred, y_proba, requested)
    raise NotImplementedError(f"cross-validated execution is not implemented for {task_type.value}")
