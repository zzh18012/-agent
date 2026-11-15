import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.models import Model

from mmos.agents import ProblemAnalyst
from mmos.config import get_settings
from mmos.core.project import atomic_write_text, current_git_commit
from mmos.routing.audit import audit_routing
from mmos.routing.policy import route_problem_spec
from mmos.routing.signals import SignalExtractor
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.routing import (
    RoutingBenchmarkCaseResult,
    RoutingBenchmarkManifest,
    RoutingBenchmarkReport,
    RoutingBenchmarkRun,
    RoutingStatus,
)


class RoutingBenchmarkError(RuntimeError):
    pass


def load_routing_manifest(path: Path) -> RoutingBenchmarkManifest:
    manifest = RoutingBenchmarkManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.manifest_hash != expected:
        raise RoutingBenchmarkError("routing manifest hash mismatch")
    return manifest


def _case_result(
    case,
    run: int,
    decision,
    audit,
) -> RoutingBenchmarkCaseResult:
    predicted_tasks = [task.value for task in decision.required_task_families]
    expected_tasks = [task.value for task in case.expected_tasks]
    hints = [f"{hint.from_task.value}->{hint.to_task.value}" for hint in decision.dependency_hints]
    status_correct = decision.status is case.expected_status
    single_exact = (
        case.expected_status is RoutingStatus.SINGLE_TASK
        and decision.status is RoutingStatus.SINGLE_TASK
        and predicted_tasks == expected_tasks
    )
    critical_unsafe = (
        case.expected_status
        in (
            RoutingStatus.MULTI_TASK,
            RoutingStatus.AMBIGUOUS,
            RoutingStatus.UNSUPPORTED,
        )
        and decision.status is RoutingStatus.SINGLE_TASK
    ) or audit.critical_unsafe
    return RoutingBenchmarkCaseResult(
        case_id=case.case_id,
        run=run,
        expected_status=case.expected_status,
        expected_tasks=case.expected_tasks,
        expected_dependency_hints=case.expected_dependency_hints,
        predicted_status=decision.status,
        predicted_primary=decision.primary_task,
        predicted_tasks=decision.required_task_families,
        predicted_dependency_hints=hints,
        status_correct=status_correct,
        single_task_exact=single_exact,
        critical_unsafe=critical_unsafe,
        audit_passed=audit.passed,
        predicted_route=f"{decision.status.value}|{','.join(predicted_tasks) or '-'}",
    )


def _metrics(cases: list[RoutingBenchmarkCaseResult], run: int) -> RoutingBenchmarkRun:
    total = len(cases)
    status_correct = sum(1 for case in cases if case.status_correct)
    single_expected = [
        case for case in cases if case.expected_status is RoutingStatus.SINGLE_TASK
    ]
    single_correct = sum(1 for case in single_expected if case.single_task_exact)
    multi_expected = [
        case for case in cases if case.expected_status is RoutingStatus.MULTI_TASK
    ]
    multi_correct = sum(
        1
        for case in multi_expected
        if case.predicted_status is RoutingStatus.MULTI_TASK
        and {task.value for task in case.expected_tasks}
        == {task.value for task in case.predicted_tasks}
        and set(case.predicted_dependency_hints)
        == set(case.expected_dependency_hints)
    )
    ambiguous_expected = [
        case for case in cases if case.expected_status is RoutingStatus.AMBIGUOUS
    ]
    ambiguous_correct = sum(
        1
        for case in ambiguous_expected
        if case.predicted_status is RoutingStatus.AMBIGUOUS
    )
    unsupported_expected = [
        case for case in cases if case.expected_status is RoutingStatus.UNSUPPORTED
    ]
    unsupported_correct = sum(
        1
        for case in unsupported_expected
        if case.predicted_status is RoutingStatus.UNSUPPORTED
    )
    return RoutingBenchmarkRun(
        run=run,
        cases=cases,
        status_accuracy=status_correct / total if total else 0.0,
        single_task_accuracy=(
            single_correct / len(single_expected) if single_expected else 1.0
        ),
        multi_task_recall=(
            multi_correct / len(multi_expected) if multi_expected else 1.0
        ),
        ambiguous_detection=(
            ambiguous_correct / len(ambiguous_expected)
            if ambiguous_expected
            else 1.0
        ),
        unsupported_detection=(
            unsupported_correct / len(unsupported_expected)
            if unsupported_expected
            else 1.0
        ),
        critical_unsafe_count=sum(1 for case in cases if case.critical_unsafe),
    )


def run_routing_benchmark(
    manifest_path: Path,
    model: Model | str,
    output_dir: Path,
    *,
    reuse_specs: bool = False,
) -> RoutingBenchmarkReport:
    manifest = load_routing_manifest(manifest_path)
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = output_dir / "problem_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, ProblemSpec] = {}
    for case in manifest.cases:
        spec_path = specs_dir / f"{case.case_id}.json"
        if reuse_specs and spec_path.exists():
            specs[case.case_id] = ProblemSpec.model_validate_json(
                spec_path.read_text(encoding="utf-8")
            )
            continue
        opaque_id = f"route-case-{case.problem_hash[:8]}"
        try:
            spec = ProblemAnalyst(model).analyze(case.problem, problem_id=opaque_id)
        except Exception as exc:
            raise RoutingBenchmarkError(
                f"Problem Analyst failed for {case.case_id}: {exc}"
            ) from exc
        specs[case.case_id] = spec
        atomic_write_text(
            spec_path,
            spec.model_dump_json(indent=2) + "\n",
        )

    runs: list[RoutingBenchmarkRun] = []
    for run_number in (1, 2):
        case_results: list[RoutingBenchmarkCaseResult] = []
        for case in manifest.cases:
            spec = specs[case.case_id]
            extraction = None
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    extraction = SignalExtractor(model).extract(case.problem, spec)
                    break
                except Exception as exc:
                    last_error = exc
            if extraction is None:
                raise RoutingBenchmarkError(
                    f"signal extraction failed for {case.case_id} run {run_number}: "
                    f"{last_error}"
                )
            decision = route_problem_spec(
                spec,
                extraction,
                problem_hash=case.problem_hash,
                problem_spec_hash=hashlib.sha256(
                    json.dumps(
                        spec.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                provider=settings.llm_provider,
                model=settings.llm_model,
                git_commit=current_git_commit(),
            )
            audit = audit_routing(decision, spec, extraction)
            case_results.append(_case_result(case, run_number, decision, audit))
            case_dir = output_dir / f"run_{run_number}" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                case_dir / "problem_signals.json",
                extraction.model_dump_json(indent=2) + "\n",
            )
            atomic_write_text(
                case_dir / "routing_decision.json",
                decision.model_dump_json(indent=2) + "\n",
            )
            atomic_write_text(
                case_dir / "routing_audit.json", audit.model_dump_json(indent=2) + "\n"
            )
        runs.append(_metrics(case_results, run_number))

    by_id_1 = {case.case_id: case for case in runs[0].cases}
    by_id_2 = {case.case_id: case for case in runs[1].cases}
    stable = sum(
        1
        for case_id in by_id_1
        if (
            by_id_1[case_id].predicted_status,
            sorted(by_id_1[case_id].predicted_tasks, key=lambda task: task.value),
            sorted(by_id_1[case_id].predicted_dependency_hints),
        )
        == (
            by_id_2[case_id].predicted_status,
            sorted(by_id_2[case_id].predicted_tasks, key=lambda task: task.value),
            sorted(by_id_2[case_id].predicted_dependency_hints),
        )
    )
    stability = stable / len(by_id_1) if by_id_1 else 0.0
    confusion: dict[str, dict[str, int]] = {}
    for case in runs[0].cases:
        expected = case.expected_status.value
        predicted = case.predicted_status.value
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
    misroutes = []
    for case in runs[0].cases:
        expected_tasks = {task.value for task in case.expected_tasks}
        predicted_tasks = {task.value for task in case.predicted_tasks}
        if (
            case.predicted_status is not case.expected_status
            or (
                case.expected_status is RoutingStatus.SINGLE_TASK
                and expected_tasks != predicted_tasks
            )
            or (
                case.expected_status is RoutingStatus.MULTI_TASK
                and expected_tasks != predicted_tasks
            )
            or (
                case.expected_status is RoutingStatus.MULTI_TASK
                and set(case.expected_dependency_hints)
                != set(case.predicted_dependency_hints)
            )
            or case.critical_unsafe
        ):
            misroutes.append(
                f"{case.case_id}: expected {case.expected_status.value} "
                f"{sorted(expected_tasks)}; predicted {case.predicted_status.value} "
                f"{sorted(predicted_tasks)}; critical={case.critical_unsafe}"
            )
    report = RoutingBenchmarkReport(
        suite_id=manifest.suite_id,
        manifest_hash=manifest.manifest_hash,
        run_1=runs[0],
        run_2=runs[1],
        stability=stability,
        confusion_matrix=confusion,
        misroutes=misroutes,
        critical_unsafe_count=runs[0].critical_unsafe_count
        + runs[1].critical_unsafe_count,
        limitations=[
            "ProblemSpec is analyzed once per case and reused across the two signal/routing "
            "runs; the runs are independent at the signal and routing layers.",
            "Gold routes are frozen; no gold label or case id is shown to the LLM.",
            "The benchmark contains 24 synthetic routing cases, not a general taxonomy.",
        ],
        created_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        output_dir / "routing_benchmark_report.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


__all__ = ["RoutingBenchmarkError", "load_routing_manifest", "run_routing_benchmark"]
