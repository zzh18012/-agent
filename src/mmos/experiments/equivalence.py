import math

import numpy as np
import polars as pl

from mmos.schemas.equivalence import PairedComparison
from mmos.schemas.metadata import InformationSet

_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 0


def paired_comparison(
    predictions: pl.DataFrame,
    candidate_model: str,
    baseline_model: str,
    *,
    candidate_information: InformationSet | None,
    baseline_information: InformationSet | None,
    threshold: float | None,
) -> PairedComparison:
    candidate = predictions.filter(pl.col("model") == candidate_model).select(
        "row_index", "prediction", "actual"
    )
    baseline = predictions.filter(pl.col("model") == baseline_model).select(
        "row_index", pl.col("prediction").alias("baseline_prediction")
    )
    joined = candidate.join(baseline, on="row_index", how="inner").sort("row_index")
    if joined.is_empty():
        return PairedComparison(
            candidate_model=candidate_model,
            baseline_model=baseline_model,
            information_set_match=False,
            n_pairs=0,
            status="inconclusive",
            threshold=threshold,
            notes=["no paired predictions"],
        )

    actual = joined["actual"].to_numpy()
    candidate_pred = joined["prediction"].to_numpy()
    baseline_pred = joined["baseline_prediction"].to_numpy()
    candidate_loss = (candidate_pred - actual) ** 2
    baseline_loss = (baseline_pred - actual) ** 2
    differences = candidate_loss - baseline_loss
    n = len(differences)

    rmse_candidate = math.sqrt(float(np.mean(candidate_loss)))
    rmse_baseline = math.sqrt(float(np.mean(baseline_loss)))
    relative: float | None
    if rmse_baseline > 0:
        relative = (rmse_baseline - rmse_candidate) / rmse_baseline
    else:
        relative = 0.0 if rmse_candidate == 0 else None

    ci_low: float | None = None
    ci_high: float | None = None
    notes: list[str] = []
    if rmse_baseline == 0 and rmse_candidate == 0:
        ci_low = 0.0
        ci_high = 0.0
        notes.append("both models have zero evaluation error")
    elif n >= 2 and rmse_baseline > 0:
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        indices = rng.integers(0, n, size=(_BOOTSTRAP_RESAMPLES, n))
        resampled_candidate = candidate_loss[indices].mean(axis=1)
        resampled_baseline = baseline_loss[indices].mean(axis=1)
        relative_samples = (np.sqrt(resampled_baseline) - np.sqrt(resampled_candidate)) / np.sqrt(
            resampled_baseline
        )
        ci_low = float(np.percentile(relative_samples, 2.5))
        ci_high = float(np.percentile(relative_samples, 97.5))
        notes.append(
            f"percentile bootstrap over {n} paired losses "
            f"({_BOOTSTRAP_RESAMPLES} resamples, seed {_BOOTSTRAP_SEED})"
        )

    information_match = bool(
        candidate_information is not None
        and baseline_information is not None
        and candidate_information.information_set_id == baseline_information.information_set_id
    )

    if not information_match:
        status = "incomparable"
        notes.append("models use different information sets")
    elif relative is None:
        status = "different"
        notes.append("baseline has zero error while the candidate does not")
    elif n < 2 or ci_low is None or ci_high is None:
        status = "inconclusive"
        notes.append("too few paired observations for an interval")
    elif threshold is None:
        if ci_low <= 0.0 <= ci_high:
            status = "equivalent"
            notes.append("relative-improvement interval includes zero")
        else:
            status = "different"
            notes.append("no practical threshold specified; interval excludes zero")
    elif ci_low >= -threshold and ci_high <= threshold:
        status = "equivalent"
        notes.append(f"interval lies within +/-{threshold:.3g} practical threshold")
    elif ci_low > threshold:
        status = "different"
        notes.append(f"candidate improvement exceeds +{threshold:.3g} threshold")
    elif ci_high < -threshold:
        status = "different"
        notes.append(f"candidate is worse than -{threshold:.3g} threshold")
    else:
        status = "inconclusive"
        notes.append("interval overlaps the practical threshold boundary")

    return PairedComparison(
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        information_set_match=information_match,
        n_pairs=n,
        mean_loss_difference=float(np.mean(differences)),
        median_loss_difference=float(np.median(differences)),
        relative_improvement=relative,
        ci_low=ci_low,
        ci_high=ci_high,
        threshold=threshold,
        status=status,
        notes=notes,
    )
