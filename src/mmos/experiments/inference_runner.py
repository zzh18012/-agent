from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy import optimize, stats

from mmos.core.project import ProjectPaths, atomic_write_text, sha256_file
from mmos.experiments.inference_models import get_forward_model, predict_array
from mmos.schemas.inference import (
    AcceptanceSamplingDesignResult,
    AcceptanceSamplingDesignSpec,
    BinomialInferenceResult,
    BinomialInferenceSpec,
    NLSStartResult,
    NonlinearLeastSquaresResult,
    NonlinearLeastSquaresSpec,
    OCPoint,
    ParameterInterval,
)


class InferenceRunError(RuntimeError):
    pass


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_spec(paths: ProjectPaths, experiment_id: str, spec: object) -> Path:
    directory = paths.experiment_dir(experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "inference_spec.json"
    atomic_write_text(target, spec.model_dump_json(indent=2) + "\n")  # type: ignore[attr-defined]
    return target


def _write_result[T: BaseModel](
    paths: ProjectPaths, experiment_id: str, result: T
) -> tuple[Path, T]:
    directory = paths.experiment_dir(experiment_id)
    target = directory / "artifacts" / "inference_result.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    payload.pop("artifact_hash", None)
    artifact_hash = _canonical_hash(payload)
    result = result.model_copy(update={"artifact_hash": artifact_hash})
    atomic_write_text(target, result.model_dump_json(indent=2) + "\n")  # type: ignore[attr-defined]
    return target, result


def run_binomial_inference(
    paths: ProjectPaths, spec: BinomialInferenceSpec, *, experiment_id: str
) -> BinomialInferenceResult:
    spec_path = _write_spec(paths, experiment_id, spec)
    alternative = "two-sided" if spec.alternative.value == "two_sided" else spec.alternative.value
    test = stats.binomtest(
        spec.successes,
        spec.trials,
        spec.null_probability,
        alternative=alternative,
    )
    alpha = 1.0 - spec.confidence_level
    ci_low = (
        0.0
        if spec.successes == 0
        else float(stats.beta.ppf(alpha / 2, spec.successes, spec.trials - spec.successes + 1))
    )
    ci_high = (
        1.0
        if spec.successes == spec.trials
        else float(
            stats.beta.ppf(
                1 - alpha / 2, spec.successes + 1, spec.trials - spec.successes
            )
        )
    )
    result = BinomialInferenceResult(
        experiment_id=experiment_id,
        observed_proportion=spec.successes / spec.trials,
        p_value=float(test.pvalue),
        confidence_interval_lower=ci_low,
        confidence_interval_upper=ci_high,
        confidence_level=spec.confidence_level,
        confidence_method="exact",
        alternative=spec.alternative,
        input_hash=_canonical_hash({"successes": spec.successes, "trials": spec.trials}),
        spec_hash=sha256_file(spec_path),
    )
    _target, result = _write_result(paths, experiment_id, result)
    return result


def _accept_probability(n: int, c: int, p: float) -> float:
    return float(stats.binom.cdf(c, n, p))


def run_acceptance_sampling(
    paths: ProjectPaths,
    spec: AcceptanceSamplingDesignSpec,
    *,
    experiment_id: str,
) -> AcceptanceSamplingDesignResult:
    spec_path = _write_spec(paths, experiment_id, spec)
    selected: tuple[int, int] | None = None
    for n in range(1, spec.max_sample_size + 1):
        for c in range(0, n + 1):
            good = _accept_probability(n, c, spec.good_quality_defect_rate)
            bad = _accept_probability(n, c, spec.bad_quality_defect_rate)
            if good >= 1.0 - spec.producer_risk_alpha and bad <= spec.consumer_risk_beta:
                selected = (n, c)
                break
        if selected is not None:
            break
    grid = [index / 20 for index in range(21)]
    grid_policy = "defect_rate=0.00..1.00 step=0.05 inclusive"
    points: list[OCPoint] = []
    good = bad = None
    if selected is not None:
        n, c = selected
        good = _accept_probability(n, c, spec.good_quality_defect_rate)
        bad = _accept_probability(n, c, spec.bad_quality_defect_rate)
        points = [
            OCPoint(defect_rate=p, acceptance_probability=_accept_probability(n, c, p))
            for p in grid
        ]
    result = AcceptanceSamplingDesignResult(
        experiment_id=experiment_id,
        sample_size=selected[0] if selected else None,
        acceptance_number=selected[1] if selected else None,
        p_good=spec.good_quality_defect_rate,
        p_bad=spec.bad_quality_defect_rate,
        probability_accept_good=good,
        probability_accept_bad=bad,
        producer_risk_actual=(1.0 - good) if good is not None else None,
        consumer_risk_actual=bad,
        producer_risk_limit=spec.producer_risk_alpha,
        consumer_risk_limit=spec.consumer_risk_beta,
        oc_curve_points=points,
        oc_grid_policy=grid_policy,
        oc_grid_hash=_canonical_hash(grid),
        search_bound=spec.max_sample_size,
        feasible=selected is not None,
        spec_hash=sha256_file(spec_path),
    )
    _target, result = _write_result(paths, experiment_id, result)
    return result


def run_nls(
    paths: ProjectPaths, spec: NonlinearLeastSquaresSpec, *, experiment_id: str
) -> NonlinearLeastSquaresResult:
    model = get_forward_model(spec.forward_model_id)
    if tuple(spec.parameter_names) != model.parameter_names:
        raise InferenceRunError(f"parameter names do not match forward model {model.model_id}")
    spec_path = _write_spec(paths, experiment_id, spec)
    x = np.asarray(spec.x_values, dtype=float)
    y = np.asarray(spec.y_values, dtype=float)
    starts: list[NLSStartResult] = []
    converged: list[tuple[int, NLSStartResult, optimize.OptimizeResult]] = []
    for index, guess in enumerate(spec.initial_guesses):
        result = optimize.least_squares(
            lambda parameters: predict_array(model, x, parameters) - y,
            x0=np.asarray(guess, dtype=float),
            bounds=(np.asarray(spec.lower_bounds), np.asarray(spec.upper_bounds)),
            loss=spec.loss,
            ftol=spec.ftol,
            xtol=spec.xtol,
            gtol=spec.gtol,
            max_nfev=spec.max_nfev,
        )
        predictions = predict_array(model, x, result.x)
        residuals = predictions - y
        start = NLSStartResult(
            start_index=index,
            initial_parameters=list(map(float, guess)),
            final_parameters=list(map(float, result.x)),
            cost=float(result.cost),
            sse=float(np.dot(residuals, residuals)),
            success=bool(result.success),
            status=int(result.status),
            optimality=float(result.optimality),
            nfev=int(result.nfev),
            message=str(result.message),
        )
        starts.append(start)
        if start.success:
            converged.append((index, start, result))
    if not converged:
        raise InferenceRunError("no least-squares start converged")
    index, selected, opt = min(converged, key=lambda item: (item[1].cost, item[0]))
    predictions = predict_array(model, x, selected.final_parameters)
    residuals = predictions - y
    jacobian = np.asarray(opt.jac, dtype=float)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(np.linalg.cond(jacobian)) if rank == len(spec.parameter_names) else None
    sse = float(np.dot(residuals, residuals))
    rmse = math.sqrt(sse / len(y))
    intervals: list[ParameterInterval] = []
    uncertainty_status = "not_identifiable_or_unsupported"
    m, p = len(y), len(spec.parameter_names)
    if spec.loss == "linear" and m > p and rank == p:
        jtj = jacobian.T @ jacobian
        covariance = (sse / (m - p)) * np.linalg.inv(jtj)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        for name, estimate, se in zip(
            spec.parameter_names, selected.final_parameters, standard_errors, strict=False
        ):
            intervals.append(
                ParameterInterval(
                    name=name,
                    estimate=float(estimate),
                    standard_error=float(se),
                    lower=float(estimate - 1.96 * se),
                    upper=float(estimate + 1.96 * se),
                )
            )
        uncertainty_status = "available_local_linearization"
    else:
        intervals = [
            ParameterInterval(
                name=name,
                estimate=float(estimate),
                standard_error=None,
                lower=None,
                upper=None,
            )
            for name, estimate in zip(
                spec.parameter_names, selected.final_parameters, strict=False
            )
        ]
    result = NonlinearLeastSquaresResult(
        experiment_id=experiment_id,
        forward_model_id=model.model_id,
        forward_model_hash=model.implementation_hash,
        x_hash=_canonical_hash(spec.x_values),
        y_hash=_canonical_hash(spec.y_values),
        parameter_names=spec.parameter_names,
        parameter_estimates=selected.final_parameters,
        predictions=list(map(float, predictions)),
        residuals=list(map(float, residuals)),
        sse=sse,
        rmse=rmse,
        jacobian=jacobian.tolist(),
        jacobian_rank=rank,
        jacobian_condition_number=condition,
        optimizer_status=selected.status,
        optimizer_success=selected.success,
        optimality=selected.optimality,
        all_start_results=starts,
        selected_start=index,
        spec_hash=sha256_file(spec_path),
        uncertainty_status=uncertainty_status,
        parameter_intervals=intervals,
    )
    _target, result = _write_result(paths, experiment_id, result)
    return result


__all__ = [
    "InferenceRunError",
    "run_acceptance_sampling",
    "run_binomial_inference",
    "run_nls",
]
