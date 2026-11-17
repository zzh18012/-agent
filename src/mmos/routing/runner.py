import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.models import Model

from mmos.agents import ProblemAnalyst
from mmos.config import get_settings
from mmos.core.problem_text import collect_problem_text
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    current_git_commit,
    load_problem_spec,
    save_problem_spec,
)
from mmos.routing.audit import audit_routing
from mmos.routing.policy import route_problem_spec
from mmos.routing.signals import SignalExtractor
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.routing import RoutingAuditResult, RoutingDecision


class RoutingError(RuntimeError):
    pass


class RoutingInfrastructureError(RoutingError):
    pass


class RoutingOutputError(RoutingError):
    pass


@dataclass(frozen=True)
class RoutingOutcome:
    problem_hash: str
    problem_spec_hash: str
    decision: RoutingDecision
    audit: RoutingAuditResult
    output_dir: Path


def _wrap_output_error(exc: Exception) -> RoutingError:
    if type(exc).__name__ in {"UnexpectedModelBehavior", "ValidationError"}:
        return RoutingOutputError(f"structured routing output failed: {exc}")
    return RoutingInfrastructureError(f"routing infrastructure failed: {exc}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_spec(spec: ProblemSpec) -> str:
    payload = spec.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def route_text(
    problem_text: str,
    problem_id: str,
    model: Model | str,
    output_dir: Path,
    *,
    problem_spec: ProblemSpec | None = None,
) -> RoutingOutcome:
    settings = get_settings()
    problem_hash = _hash_text(problem_text)
    if problem_spec is None:
        try:
            problem_spec = ProblemAnalyst(model).analyze(
                problem_text,
                problem_id=problem_id,
                diagnostic_dir=output_dir,
            )
        except Exception as exc:
            raise _wrap_output_error(exc) from exc
    problem_spec_hash = _hash_spec(problem_spec)
    try:
        extraction = SignalExtractor(model).extract(
            problem_text,
            problem_spec,
            diagnostic_dir=output_dir,
        )
    except Exception as exc:
        raise _wrap_output_error(exc) from exc
    decision = route_problem_spec(
        problem_spec,
        extraction,
        problem_hash=problem_hash,
        problem_spec_hash=problem_spec_hash,
        provider=settings.llm_provider,
        model=settings.llm_model,
        git_commit=current_git_commit(),
    )
    audit = audit_routing(decision, problem_spec, extraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "problem_signals.json", extraction.model_dump_json(indent=2) + "\n"
    )
    atomic_write_text(
        output_dir / "routing_decision.json", decision.model_dump_json(indent=2) + "\n"
    )
    atomic_write_text(
        output_dir / "routing_audit.json", audit.model_dump_json(indent=2) + "\n"
    )
    atomic_write_text(
        output_dir / "problem_spec.json", problem_spec.model_dump_json(indent=2) + "\n"
    )
    return RoutingOutcome(
        problem_hash=problem_hash,
        problem_spec_hash=problem_spec_hash,
        decision=decision,
        audit=audit,
        output_dir=output_dir,
    )


def route_project(paths: ProjectPaths, model: Model | str) -> RoutingOutcome:
    problem_text = collect_problem_text(paths)
    problem_spec = (
        load_problem_spec(paths) if paths.problem_spec_file.exists() else None
    )
    outcome = route_text(
        problem_text,
        paths.root.name,
        model,
        paths.routing_dir,
        problem_spec=problem_spec,
    )
    if not paths.problem_spec_file.exists():
        save_problem_spec(paths, ProblemSpec.model_validate_json(
            (paths.routing_dir / "problem_spec.json").read_text(encoding="utf-8")
        ))
    return outcome


__all__ = [
    "RoutingError",
    "RoutingInfrastructureError",
    "RoutingOutcome",
    "RoutingOutputError",
    "route_project",
    "route_text",
]
