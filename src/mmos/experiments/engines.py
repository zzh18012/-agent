from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from mmos.schemas.registry import TaskType

TEMPORAL_BASELINES = frozenset({"naive.last_value", "naive.drift", "naive.seasonal"})
STOCHASTIC_MODEL_IDS = frozenset(
    {
        "sklearn.random_forest_regressor",
        "sklearn.random_forest_classifier",
        "sklearn.hist_gradient_boosting_regressor",
    }
)

EstimatorFactory = Callable[[int], Any]


def _dummy_regressor(_: int) -> Any:
    return DummyRegressor(strategy="mean")


def _linear_regression(_: int) -> Any:
    return LinearRegression()


def _ridge(_: int) -> Any:
    return Ridge(alpha=1.0)


def _random_forest_regressor(random_state: int) -> Any:
    return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)


def _hist_gradient_boosting_regressor(random_state: int) -> Any:
    return HistGradientBoostingRegressor(random_state=random_state)


def _logistic_regression(random_state: int) -> Any:
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _random_forest_classifier(random_state: int) -> Any:
    return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)


ENGINES: dict[str, tuple[TaskType, EstimatorFactory]] = {
    "sklearn.dummy_regressor": (TaskType.REGRESSION, _dummy_regressor),
    "sklearn.linear_regression": (TaskType.REGRESSION, _linear_regression),
    "sklearn.ridge": (TaskType.REGRESSION, _ridge),
    "sklearn.random_forest_regressor": (TaskType.REGRESSION, _random_forest_regressor),
    "sklearn.hist_gradient_boosting_regressor": (
        TaskType.REGRESSION,
        _hist_gradient_boosting_regressor,
    ),
    "sklearn.logistic_regression": (TaskType.CLASSIFICATION, _logistic_regression),
    "sklearn.random_forest_classifier": (TaskType.CLASSIFICATION, _random_forest_classifier),
}


def task_type_for(model_id: str) -> TaskType:
    try:
        return ENGINES[model_id][0]
    except KeyError:
        raise NotImplementedError(
            f"model {model_id!r} is registered but has no executable engine yet; "
            f"available: {', '.join(sorted(ENGINES))}"
        ) from None


def build_estimator(model_id: str, *, random_state: int) -> Any:
    try:
        _, factory = ENGINES[model_id]
    except KeyError:
        raise NotImplementedError(
            f"model {model_id!r} has no executable engine yet; "
            f"available: {', '.join(sorted(ENGINES))}"
        ) from None
    return factory(random_state)


def is_stochastic(model_id: str) -> bool:
    return model_id in STOCHASTIC_MODEL_IDS


def forecast_baseline(
    model_id: str,
    y_train: np.ndarray,
    horizon: int,
    seasonal_period: int | None,
) -> np.ndarray:
    if len(y_train) == 0:
        raise ValueError("temporal baseline needs at least one training observation")
    steps = np.arange(1, horizon + 1, dtype=float)
    if model_id == "naive.last_value":
        return np.full(horizon, float(y_train[-1]))
    if model_id == "naive.drift":
        if len(y_train) < 2:
            return np.full(horizon, float(y_train[-1]))
        slope = (float(y_train[-1]) - float(y_train[0])) / (len(y_train) - 1)
        return float(y_train[-1]) + steps * slope
    if model_id == "naive.seasonal":
        if seasonal_period is None:
            raise ValueError("seasonal naive requires a confirmed seasonal_period")
        if len(y_train) < seasonal_period:
            raise ValueError("seasonal naive needs at least one full season in training")
        indices = [
            len(y_train) - seasonal_period + ((step - 1) % seasonal_period)
            for step in range(1, horizon + 1)
        ]
        return np.array([float(y_train[index]) for index in indices])
    raise ValueError(f"{model_id!r} is not a temporal baseline")
