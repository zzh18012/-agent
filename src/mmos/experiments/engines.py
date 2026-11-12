from collections.abc import Callable
from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from mmos.schemas.registry import TaskType

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
