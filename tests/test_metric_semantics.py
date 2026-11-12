import math

import numpy as np
import pytest

from mmos.experiments.evaluation import metrics_from_evaluation, regression_evaluation
from mmos.experiments.runner import _mase_fold_detail
from mmos.schemas.experiment import FoldWindow


def test_known_r2_components_and_identity() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    record = regression_evaluation("model", y_true, y_pred)

    assert record.n_observations == 3
    assert record.sse == 1.0
    assert record.sst == 2.0
    assert record.r2 == 0.5
    assert record.target_mean == 2.0
    assert record.target_std_population == math.sqrt(2.0 / 3.0)
    assert record.rmse == math.sqrt(1.0 / 3.0)
    assert record.rmse_eval_std_ratio == record.rmse / record.target_std_population
    assert (record.rmse_eval_std_ratio**2) == pytest.approx(1.0 - record.r2)

    metrics = metrics_from_evaluation(record)
    assert set(metrics) == {"rmse", "mae", "r2", "rmse_eval_std_ratio"}


def test_constant_target_r2_is_undefined_with_reason() -> None:
    record = regression_evaluation("model", np.array([5.0, 5.0, 5.0]), np.array([5.0, 4.0, 6.0]))

    assert record.r2 is None
    assert record.r2_reason is not None
    assert "constant" in record.r2_reason
    assert record.rmse_eval_std_ratio is None


def test_mase_denominator_uses_training_history_only() -> None:
    window = FoldWindow(train_start=0, train_end=5, test_start=5, test_end=6)
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    detail = _mase_fold_detail(0, window, y_train, np.array([6.0]), np.array([6.0]), None)

    assert detail.scale_lag == 1
    assert detail.scale_denominator == 1.0
    assert detail.absolute_forecast_error == 0.0
    assert detail.scaled_error == 0.0
    assert detail.train_end == 5

    changed_future = _mase_fold_detail(
        0, window, y_train, np.array([100.0]), np.array([120.0]), None
    )
    assert changed_future.scale_denominator == detail.scale_denominator


def test_mase_zero_denominator_is_undefined() -> None:
    window = FoldWindow(train_start=0, train_end=4, test_start=4, test_end=5)
    detail = _mase_fold_detail(
        0,
        window,
        np.array([3.0, 3.0, 3.0, 3.0]),
        np.array([4.0]),
        np.array([4.0]),
        None,
    )

    assert detail.scale_denominator == 0.0
    assert detail.scaled_error is None
    assert detail.status.startswith("undefined")
