from __future__ import annotations

import math

import numpy as np
from scipy import stats

from mmos.core.project import ProjectPaths, sha256_file
from mmos.experiments.inference_models import get_forward_model, predict_array
from mmos.schemas.data import Severity
from mmos.schemas.inference import (
    AcceptanceSamplingDesignResult,
    AcceptanceSamplingDesignSpec,
    BinomialInferenceResult,
    BinomialInferenceSpec,
    NonlinearLeastSquaresResult,
    NonlinearLeastSquaresSpec,
)
from mmos.schemas.validation import ValidationIssue, ValidationReport
from mmos.validation.base import build_report

_TOL = 1e-8


def _load(paths: ProjectPaths, experiment_id: str, name: str):
    return paths.experiment_dir(experiment_id) / name


def _issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.ERROR, message=message)


def _binomial_p_value(k: int, n: int, p0: float, alternative: str) -> float:
    if alternative == "less":
        return float(stats.binom.cdf(k, n, p0))
    if alternative == "greater":
        return float(stats.binom.sf(k - 1, n, p0))
    observed = float(stats.binom.pmf(k, n, p0))
    return float(
        sum(
            value
            for value in (stats.binom.pmf(index, n, p0) for index in range(n + 1))
            if value <= observed + 1e-15
        )
    )


def _clopper_pearson(k: int, n: int, level: float) -> tuple[float, float]:
    alpha = 1.0 - level
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def validate_binomial_inference(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = BinomialInferenceSpec.model_validate_json(
        _load(paths, experiment_id, "inference_spec.json").read_text(encoding="utf-8")
    )
    result = BinomialInferenceResult.model_validate_json(
        _load(paths, experiment_id, "artifacts/inference_result.json").read_text(encoding="utf-8")
    )
    issues: list[ValidationIssue] = []
    if result.experiment_id != experiment_id:
        issues.append(_issue("experiment_mismatch", "result experiment id differs"))
    if result.spec_hash != sha256_file(_load(paths, experiment_id, "inference_spec.json")):
        issues.append(_issue("spec_hash_mismatch", "spec hash differs from artifact"))
    proportion = spec.successes / spec.trials
    if abs(result.observed_proportion - proportion) > _TOL:
        issues.append(_issue("proportion_mismatch", "observed proportion mismatch"))
    p_value = _binomial_p_value(
        spec.successes, spec.trials, spec.null_probability, spec.alternative.value
    )
    if abs(result.p_value - p_value) > 1e-7:
        issues.append(_issue("p_value_mismatch", "exact p-value mismatch"))
    lower, upper = _clopper_pearson(spec.successes, spec.trials, spec.confidence_level)
    if abs(result.confidence_interval_lower - lower) > 1e-7:
        issues.append(_issue("ci_lower_mismatch", "Clopper-Pearson lower mismatch"))
    if abs(result.confidence_interval_upper - upper) > 1e-7:
        issues.append(_issue("ci_upper_mismatch", "Clopper-Pearson upper mismatch"))
    if not (0.0 <= result.confidence_interval_lower <= result.observed_proportion):
        issues.append(_issue("ci_bounds", "CI lower bound is invalid"))
    if not (result.observed_proportion <= result.confidence_interval_upper <= 1.0):
        issues.append(_issue("ci_bounds", "CI upper bound is invalid"))
    return build_report("inference", experiment_id, issues)


def _accept_probability(n: int, c: int, p: float) -> float:
    return float(sum(math.comb(n, k) * p**k * (1.0 - p) ** (n - k) for k in range(c + 1)))


def validate_acceptance_sampling(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = AcceptanceSamplingDesignSpec.model_validate_json(
        _load(paths, experiment_id, "inference_spec.json").read_text(encoding="utf-8")
    )
    result = AcceptanceSamplingDesignResult.model_validate_json(
        _load(paths, experiment_id, "artifacts/inference_result.json").read_text(encoding="utf-8")
    )
    issues: list[ValidationIssue] = []
    if result.spec_hash != sha256_file(_load(paths, experiment_id, "inference_spec.json")):
        issues.append(_issue("spec_hash_mismatch", "spec hash differs from artifact"))
    if not result.feasible:
        # Recompute that no plan exists within the declared search bound.
        for n in range(1, spec.max_sample_size + 1):
            for c in range(n + 1):
                good = _accept_probability(n, c, spec.good_quality_defect_rate)
                bad = _accept_probability(n, c, spec.bad_quality_defect_rate)
                if good >= 1.0 - spec.producer_risk_alpha and bad <= spec.consumer_risk_beta:
                    issues.append(_issue("false_infeasible", f"plan ({n}, {c}) is feasible"))
                    break
            if issues:
                break
        return build_report("inference", experiment_id, issues)
    n = result.sample_size or 0
    c = result.acceptance_number or 0
    if not (1 <= n <= spec.max_sample_size and 0 <= c <= n):
        issues.append(_issue("plan_bounds", "selected plan is outside the search bound"))
    good = _accept_probability(n, c, spec.good_quality_defect_rate)
    bad = _accept_probability(n, c, spec.bad_quality_defect_rate)
    if abs((result.probability_accept_good or -1) - good) > 1e-9:
        issues.append(_issue("p_accept_good", "acceptance probability at good quality"))
    if abs((result.probability_accept_bad or -1) - bad) > 1e-9:
        issues.append(_issue("p_accept_bad", "acceptance probability at bad quality"))
    if good < 1.0 - spec.producer_risk_alpha:
        issues.append(_issue("producer_risk", "producer risk constraint violated"))
    if bad > spec.consumer_risk_beta:
        issues.append(_issue("consumer_risk", "consumer risk constraint violated"))
    # Independent minimality: no smaller n may have any feasible c.
    for smaller in range(1, n):
        for candidate_c in range(smaller + 1):
            candidate_good = _accept_probability(
                smaller, candidate_c, spec.good_quality_defect_rate
            )
            candidate_bad = _accept_probability(smaller, candidate_c, spec.bad_quality_defect_rate)
            if (
                candidate_good >= 1.0 - spec.producer_risk_alpha
                and candidate_bad <= spec.consumer_risk_beta
            ):
                issues.append(
                    _issue(
                        "not_minimum_n",
                        f"smaller feasible plan exists: ({smaller}, {candidate_c})",
                    )
                )
                break
        if issues:
            break
    # For equal n, no smaller c may be feasible.
    for candidate_c in range(c):
        candidate_good = _accept_probability(n, candidate_c, spec.good_quality_defect_rate)
        candidate_bad = _accept_probability(n, candidate_c, spec.bad_quality_defect_rate)
        if (
            candidate_good >= 1.0 - spec.producer_risk_alpha
            and candidate_bad <= spec.consumer_risk_beta
        ):
            issues.append(_issue("not_minimum_c", f"smaller feasible c exists at n={n}"))
            break
    if len(result.oc_curve_points) != 21:
        issues.append(_issue("oc_grid", "OC curve grid is not the declared 21 points"))
    for point in result.oc_curve_points:
        expected = _accept_probability(n, c, point.defect_rate)
        if abs(point.acceptance_probability - expected) > 1e-9:
            issues.append(_issue("oc_point", "OC curve point mismatch"))
            break
    return build_report("inference", experiment_id, issues)


def validate_nls(paths: ProjectPaths, experiment_id: str) -> ValidationReport:
    spec = NonlinearLeastSquaresSpec.model_validate_json(
        _load(paths, experiment_id, "inference_spec.json").read_text(encoding="utf-8")
    )
    result = NonlinearLeastSquaresResult.model_validate_json(
        _load(paths, experiment_id, "artifacts/inference_result.json").read_text(encoding="utf-8")
    )
    issues: list[ValidationIssue] = []
    if result.spec_hash != sha256_file(_load(paths, experiment_id, "inference_spec.json")):
        issues.append(_issue("spec_hash_mismatch", "spec hash differs from artifact"))
    model = get_forward_model(spec.forward_model_id)
    if result.forward_model_hash != model.implementation_hash:
        issues.append(_issue("model_hash_mismatch", "forward model hash changed"))
    estimates = np.asarray(result.parameter_estimates, dtype=float)
    lower = np.asarray(spec.lower_bounds, dtype=float)
    upper = np.asarray(spec.upper_bounds, dtype=float)
    if np.any(estimates < lower - 1e-12) or np.any(estimates > upper + 1e-12):
        issues.append(_issue("bounds", "parameter estimates violate bounds"))
    x = np.asarray(spec.x_values, dtype=float)
    y = np.asarray(spec.y_values, dtype=float)
    predictions = predict_array(model, x, estimates)
    residuals = predictions - y
    sse = float(np.dot(residuals, residuals))
    rmse = math.sqrt(sse / len(y))
    if abs(result.sse - sse) > 1e-8 * max(1.0, abs(sse)):
        issues.append(_issue("sse", "SSE recomputation mismatch"))
    if abs(result.rmse - rmse) > 1e-8 * max(1.0, abs(rmse)):
        issues.append(_issue("rmse", "RMSE recomputation mismatch"))
    converged = [start for start in result.all_start_results if start.success]
    if not converged:
        issues.append(_issue("no_converged_start", "no converged start retained"))
    else:
        best = min(converged, key=lambda start: (start.cost, start.start_index))
        if best.start_index != result.selected_start:
            issues.append(_issue("selected_start", "selected start is not minimum cost"))
        selected = next(
            (
                start
                for start in result.all_start_results
                if start.start_index == result.selected_start
            ),
            None,
        )
        if selected is None or not selected.success:
            issues.append(_issue("selected_start_missing", "selected start missing"))
        elif abs(selected.sse - sse) > 1e-8 * max(1.0, abs(sse)):
            issues.append(_issue("selected_sse", "selected start SSE mismatch"))
    jacobian = np.asarray(result.jacobian, dtype=float)
    if jacobian.shape != (len(y), len(spec.parameter_names)):
        issues.append(_issue("jacobian_shape", "Jacobian dimensions are wrong"))
    rank = int(np.linalg.matrix_rank(jacobian))
    if rank != result.jacobian_rank:
        issues.append(_issue("jacobian_rank", "Jacobian rank mismatch"))
    if result.uncertainty_status == "available_local_linearization":
        if spec.loss != "linear":
            issues.append(_issue("robust_uncertainty", "robust loss cannot use LS covariance"))
        if len(y) <= len(spec.parameter_names) or rank < len(spec.parameter_names):
            issues.append(_issue("rank_deficient_ci", "CI created for rank-deficient fit"))
        if not result.approximate_uncertainty:
            issues.append(_issue("approximate_flag", "local uncertainty must be approximate"))
    return build_report("inference", experiment_id, issues)


__all__ = [
    "validate_acceptance_sampling",
    "validate_binomial_inference",
    "validate_nls",
]
