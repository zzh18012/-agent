from __future__ import annotations

import json
from pathlib import Path

from mmos.core.project import ProjectPaths, create_project
from mmos.evidence.inference import validate_and_register_inference
from mmos.experiments.inference_runner import (
    run_acceptance_sampling,
    run_binomial_inference,
    run_nls,
)
from mmos.schemas.inference import (
    AcceptanceSamplingDesignSpec,
    Alternative,
    BinomialInferenceSpec,
    NonlinearLeastSquaresSpec,
)
from mmos.schemas.states import TaskFamily
from mmos.validation.inference import (
    validate_acceptance_sampling,
    validate_binomial_inference,
    validate_nls,
)


def project(tmp_path: Path) -> ProjectPaths:
    paths, _ = create_project(tmp_path / "projects", "inference", task=TaskFamily.PREDICTION)
    return paths


def test_binomial_exact_and_validator(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = BinomialInferenceSpec(
        successes=3,
        trials=20,
        null_probability=0.1,
        alternative=Alternative.GREATER,
        confidence_level=0.95,
    )
    result = run_binomial_inference(paths, spec, experiment_id="exp-bin-1")
    report = validate_binomial_inference(paths, "exp-bin-1")
    assert not report.blocking
    assert 0.0 <= result.p_value <= 1.0
    assert 0.0 <= result.confidence_interval_lower <= result.observed_proportion
    assert result.observed_proportion <= result.confidence_interval_upper <= 1.0
    # Tampering is caught.
    artifact = paths.experiment_dir("exp-bin-1") / "artifacts" / "inference_result.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["p_value"] = 0.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_binomial_inference(paths, "exp-bin-1").blocking


def test_binomial_edge_cases(tmp_path: Path) -> None:
    paths = project(tmp_path)
    for index, (k, n, p0) in enumerate([(0, 1, 0.0), (1, 1, 1.0)]):
        spec = BinomialInferenceSpec(
            successes=k,
            trials=n,
            null_probability=p0,
            alternative=Alternative.TWO_SIDED,
            confidence_level=0.9,
        )
        run_binomial_inference(paths, spec, experiment_id=f"exp-bin-edge-{index}")
        assert not validate_binomial_inference(paths, f"exp-bin-edge-{index}").blocking


def _accept_probability(n: int, c: int, p: float) -> float:
    import math

    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(c + 1))


def test_acceptance_plan_deterministic_minimum(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = AcceptanceSamplingDesignSpec(
        good_quality_defect_rate=0.02,
        bad_quality_defect_rate=0.15,
        producer_risk_alpha=0.05,
        consumer_risk_beta=0.10,
        max_sample_size=80,
    )
    result = run_acceptance_sampling(paths, spec, experiment_id="exp-acc-1")
    assert result.feasible
    assert result.sample_size is not None
    assert result.acceptance_number is not None
    # Independent exhaustive reference.
    expected = None
    for n in range(1, spec.max_sample_size + 1):
        for c in range(n + 1):
            if _accept_probability(n, c, 0.02) >= 0.95 and _accept_probability(n, c, 0.15) <= 0.10:
                expected = (n, c)
                break
        if expected:
            break
    assert (result.sample_size, result.acceptance_number) == expected
    assert not validate_acceptance_sampling(paths, "exp-acc-1").blocking
    assert len(result.oc_curve_points) == 21


def test_acceptance_no_feasible_does_not_expand_bound(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = AcceptanceSamplingDesignSpec(
        good_quality_defect_rate=0.01,
        bad_quality_defect_rate=0.02,
        producer_risk_alpha=0.001,
        consumer_risk_beta=0.001,
        max_sample_size=5,
    )
    result = run_acceptance_sampling(paths, spec, experiment_id="exp-acc-inf")
    assert result.feasible is False
    assert result.sample_size is None
    assert not validate_acceptance_sampling(paths, "exp-acc-inf").blocking


def _nls_spec(*, starts: list[list[float]] | None = None, loss: str = "linear", n: int = 12):
    import math

    x = [index * 0.5 for index in range(n)]
    y = [2.0 * math.exp(-0.5 * value) + 1.0 for value in x]
    return NonlinearLeastSquaresSpec(
        forward_model_id="registered.exponential_decay",
        x_values=x,
        y_values=y,
        parameter_names=["amplitude", "decay", "offset"],
        lower_bounds=[0.0, 0.0, -10.0],
        upper_bounds=[10.0, 5.0, 10.0],
        initial_guesses=starts or [[1.0, 1.0, 0.0], [5.0, 0.1, 3.0]],
        loss=loss,
    )


def test_nls_multistart_and_local_uncertainty(tmp_path: Path) -> None:
    paths = project(tmp_path)
    result = run_nls(paths, _nls_spec(), experiment_id="exp-nls-1")
    assert len(result.all_start_results) == 2
    assert (
        result.selected_start
        == min(
            (start for start in result.all_start_results if start.success),
            key=lambda start: (start.cost, start.start_index),
        ).start_index
    )
    assert result.jacobian_rank == 3
    assert result.uncertainty_status == "available_local_linearization"
    assert result.approximate_uncertainty
    assert all(interval.standard_error is not None for interval in result.parameter_intervals)
    assert abs(result.parameter_estimates[0] - 2.0) < 1e-6
    assert abs(result.parameter_estimates[1] - 0.5) < 1e-6
    assert not validate_nls(paths, "exp-nls-1").blocking


def test_nls_m_le_p_has_no_ci(tmp_path: Path) -> None:
    paths = project(tmp_path)
    result = run_nls(paths, _nls_spec(n=2), experiment_id="exp-nls-few")
    assert result.uncertainty_status == "not_identifiable_or_unsupported"
    assert all(interval.standard_error is None for interval in result.parameter_intervals)
    assert not validate_nls(paths, "exp-nls-few").blocking


def test_robust_loss_has_no_local_covariance(tmp_path: Path) -> None:
    paths = project(tmp_path)
    result = run_nls(paths, _nls_spec(loss="soft_l1"), experiment_id="exp-nls-robust")
    assert result.uncertainty_status == "not_identifiable_or_unsupported"
    assert not validate_nls(paths, "exp-nls-robust").blocking


def test_inference_evidence_registration(tmp_path: Path) -> None:
    paths = project(tmp_path)
    spec = BinomialInferenceSpec(
        successes=3,
        trials=20,
        null_probability=0.1,
        alternative=Alternative.GREATER,
        confidence_level=0.95,
    )
    run_binomial_inference(paths, spec, experiment_id="exp-evidence-bin")
    records = validate_and_register_inference(paths, "exp-evidence-bin", "binomial")
    assert records
    index = json.loads(paths.verified_numbers_file.read_text(encoding="utf-8"))
    for record in records:
        assert record.evidence_id in index["records"]
        assert record.key.startswith("inference.binomial.")
        assert record.validation_reports == ["inference"]
