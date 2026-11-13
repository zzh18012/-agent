import json
import shutil
from pathlib import Path

from mmos.core.project import ProjectPaths
from mmos.experiments.knapsack_benchmark import DEFAULT_SUITE_DIR
from mmos.validation.benchmark import validate_benchmark_suite


def _clone(template_root: Path, tmp_path: Path, name: str) -> ProjectPaths:
    target = tmp_path / name
    shutil.copytree(template_root, target)
    return ProjectPaths(target.resolve())


def _report_path(paths: ProjectPaths, run_id: str) -> Path:
    return paths.experiments_dir / "benchmarks" / run_id / "benchmark_result.json"


def _tap(paths: ProjectPaths, run_id: str, mutate) -> None:
    target = _report_path(paths, run_id)
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _validate(paths: ProjectPaths, run_id: str):
    return validate_benchmark_suite(paths, run_id, DEFAULT_SUITE_DIR)


def test_complete_suite_passes_all_validators(
    benchmark_template: tuple[Path, str], tmp_path: Path
) -> None:
    template_root, run_id = benchmark_template
    paths = _clone(template_root, tmp_path, "validator-pass")

    reports = _validate(paths, run_id)

    assert [report.validator for report in reports] == [
        "benchmark_integrity",
        "benchmark_oracle",
        "benchmark_aggregate",
    ]
    assert all(report.status.value == "passed" for report in reports)
    assert not any(report.blocking for report in reports)
    summary = paths.experiments_dir / "benchmarks" / run_id / "validation" / "summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["blocking"] is False
    assert set(payload["reports"]) == {
        "benchmark_integrity",
        "benchmark_oracle",
        "benchmark_aggregate",
    }


def test_oracle_mismatch_is_blocking(
    benchmark_template: tuple[Path, str], tmp_path: Path
) -> None:
    template_root, run_id = benchmark_template
    paths = _clone(template_root, tmp_path, "validator-oracle")

    def mutate(payload: dict) -> None:
        payload["per_instance"][0]["oracle_optimum"] += 1.0

    _tap(paths, run_id, mutate)
    reports = _validate(paths, run_id)

    oracle = next(report for report in reports if report.validator == "benchmark_oracle")
    assert oracle.blocking is True
    assert any(issue.code == "oracle_recompute_mismatch" for issue in oracle.issues)


def test_aggregate_mismatch_is_blocking(
    benchmark_template: tuple[Path, str], tmp_path: Path
) -> None:
    template_root, run_id = benchmark_template
    paths = _clone(template_root, tmp_path, "validator-aggregate")

    def mutate(payload: dict) -> None:
        payload["greedy"]["mean_relative_regret"] += 0.1

    _tap(paths, run_id, mutate)
    reports = _validate(paths, run_id)

    aggregate = next(
        report for report in reports if report.validator == "benchmark_aggregate"
    )
    assert aggregate.blocking is True
    assert any(issue.code == "aggregate_mismatch" for issue in aggregate.issues)


def test_missing_instance_result_is_blocking(
    benchmark_template: tuple[Path, str], tmp_path: Path
) -> None:
    template_root, run_id = benchmark_template
    paths = _clone(template_root, tmp_path, "validator-missing")

    def mutate(payload: dict) -> None:
        payload["per_instance"] = payload["per_instance"][1:]

    _tap(paths, run_id, mutate)
    reports = _validate(paths, run_id)

    integrity = next(
        report for report in reports if report.validator == "benchmark_integrity"
    )
    assert integrity.blocking is True
    codes = {issue.code for issue in integrity.issues}
    assert "missing_instance_result" in codes
    assert "instance_count_mismatch" in codes


def test_manifest_hash_mismatch_is_blocking(
    benchmark_template: tuple[Path, str], tmp_path: Path
) -> None:
    template_root, run_id = benchmark_template
    paths = _clone(template_root, tmp_path, "validator-manifest")

    def mutate(payload: dict) -> None:
        payload["manifest_hash"] = "0" * 64

    _tap(paths, run_id, mutate)
    reports = _validate(paths, run_id)

    integrity = next(
        report for report in reports if report.validator == "benchmark_integrity"
    )
    assert integrity.blocking is True
    assert any(issue.code == "manifest_hash_mismatch" for issue in integrity.issues)
