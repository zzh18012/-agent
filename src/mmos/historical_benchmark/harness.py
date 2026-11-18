from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmos.core.intake import ingest_files
from mmos.core.project import (
    ProjectPaths,
    atomic_write_text,
    create_project,
    current_git_commit,
    sha256_file,
)
from mmos.historical_benchmark.schemas import (
    BenchmarkBlindnessAudit,
    BenchmarkCodeFreeze,
    BenchmarkFile,
    BenchmarkResult,
    BenchmarkSourceManifest,
    BlindnessPolicy,
    CapabilityFitAssessment,
    CapabilityGap,
    CapabilityVerdict,
    HistoricalBenchmarkRun,
    HistoricalBenchmarkState,
    LeakageEvent,
    LeakageLogEntry,
    ProblemRequirement,
    ProblemSelectionManifest,
    ProtocolValidity,
    RequirementStatus,
    RevealRecord,
    SelectionCandidate,
)
from mmos.routing.runner import RoutingOutcome, route_project
from mmos.schemas.problem import ProblemSpec
from mmos.schemas.project import SourceRole
from mmos.schemas.routing import RoutingDecision
from mmos.schemas.states import TaskFamily


class HistoricalBenchmarkError(RuntimeError):
    pass


class BenchmarkInfrastructureError(HistoricalBenchmarkError):
    pass


class BlindnessViolation(HistoricalBenchmarkError):
    pass


def _hash_payload(payload: object, *, exclude: tuple[str, ...] = ()) -> str:
    data = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        for key in exclude:
            data.pop(key, None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _hash_model(model: Any, *, exclude: tuple[str, ...] = ()) -> str:
    return _hash_payload(model.model_dump(mode="json"), exclude=exclude)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HistoricalBenchmarkError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _write_json(path: Path, model: Any) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HistoricalBenchmarkError(f"missing benchmark artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([ABC])(?:题|卷|[_\-.]|$|[/\\])")


def _labels_from_zip(archive: Path) -> list[str]:
    labels: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            for candidate in (Path(name).name, name):
                for match in _LABEL_RE.finditer(candidate):
                    labels.add(match.group(1))
    return sorted(labels)


def _safe_extract(archive: Path, target: Path) -> list[BenchmarkFile]:
    target.mkdir(parents=True, exist_ok=True)
    entries: list[BenchmarkFile] = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise HistoricalBenchmarkError(f"unsafe archive entry: {name}")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            entries.append(
                BenchmarkFile(
                    path=destination.relative_to(target).as_posix(),
                    sha256=sha256_file(destination),
                    size_bytes=destination.stat().st_size,
                )
            )
    return entries


def _code_tree_hash(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for base in ("src/mmos", "tests", "pyproject.toml"):
        path = repo_root / base
        if not path.exists():
            continue
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for file in files:
            if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
                continue
            digest.update(file.relative_to(repo_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def create_code_freeze(
    repo_root: Path,
    benchmark_run_id: str,
    *,
    test_status: str,
    ruff_status: str,
    pyright_status: str,
    now: datetime | None = None,
    git_commit: str | None = None,
) -> BenchmarkCodeFreeze:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=20,
    )
    clean = status.returncode == 0 and not status.stdout.strip()
    freeze = BenchmarkCodeFreeze(
        benchmark_run_id=benchmark_run_id,
        git_commit=git_commit or current_git_commit(),
        worktree_clean=clean,
        test_status=test_status,
        ruff_status=ruff_status,
        pyright_status=pyright_status,
        code_tree_hash=_code_tree_hash(repo_root),
        created_at=_utc(now or datetime.now(UTC)),
        freeze_hash="",
    )
    freeze = freeze.model_copy(
        update={"freeze_hash": _hash_model(freeze, exclude=("freeze_hash",))}
    )
    return freeze


class HistoricalBenchmarkHarness:
    def __init__(self, repo_root: Path, benchmark_run_id: str, *, competition: str, year: int):
        self.repo_root = repo_root
        self.benchmark_run_id = benchmark_run_id
        self.competition = competition
        self.year = year
        self.root = repo_root / "benchmarks" / "historical" / benchmark_run_id
        self.source_root = repo_root / "benchmarks" / "sources"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def run_file(self) -> Path:
        return self.root / "benchmark_run.json"

    @property
    def source_manifest_file(self) -> Path:
        return self.root / "source_manifest.json"

    @property
    def selection_file(self) -> Path:
        return self.root / "selection_manifest.json"

    @property
    def reveal_file(self) -> Path:
        return self.root / "reveal.json"

    @property
    def code_freeze_file(self) -> Path:
        return self.root / "code_freeze.json"

    @property
    def blindness_policy_file(self) -> Path:
        return self.root / "blindness_policy.json"

    @property
    def reserve_authorization_file(self) -> Path:
        return self.root / "reserve_authorization.json"

    @property
    def leakage_file(self) -> Path:
        return self.root / "leakage_log.jsonl"

    @property
    def problem_package_dir(self) -> Path:
        return self.root / "problem_package"

    @property
    def project_dir(self) -> Path:
        return self.root / "project"

    def _write_run(self, run: HistoricalBenchmarkRun) -> HistoricalBenchmarkRun:
        run = run.model_copy(update={"run_hash": _hash_model(run, exclude=("run_hash",))})
        _write_json(self.run_file, run)
        return run

    def start(self, now: datetime | None = None) -> HistoricalBenchmarkRun:
        now = _utc(now or datetime.now(UTC))
        run = HistoricalBenchmarkRun(
            benchmark_run_id=self.benchmark_run_id,
            competition=self.competition,
            year=self.year,
            created_at=now,
            updated_at=now,
        )
        atomic_write_text(
            self.blindness_policy_file, BlindnessPolicy().model_dump_json(indent=2) + "\n"
        )
        return self._write_run(run)

    def load_run(self) -> HistoricalBenchmarkRun:
        return HistoricalBenchmarkRun.model_validate_json(self.run_file.read_text(encoding="utf-8"))

    def record_freshness_registry(self, registry_hash: str) -> HistoricalBenchmarkRun:
        run = self.load_run()
        return self._write_run(run.model_copy(update={"freshness_registry_hash": registry_hash}))

    def log_leakage(
        self,
        event: LeakageEvent,
        detail: str,
        *,
        prohibited: bool = True,
        now: datetime | None = None,
    ) -> LeakageLogEntry:
        entry = LeakageLogEntry(
            event_type=event,
            occurred_at=_utc(now or datetime.now(UTC)),
            detail=detail,
            prohibited=prohibited,
        )
        with self.leakage_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(entry.model_dump_json() + "\n")
        return entry

    def leakage_entries(self) -> list[LeakageLogEntry]:
        if not self.leakage_file.exists():
            return []
        return [
            LeakageLogEntry.model_validate_json(line)
            for line in self.leakage_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def seal_archive(
        self,
        archive: Path,
        *,
        official_source: str,
        downloaded_at: datetime,
        benchmark_source_id: str,
    ) -> BenchmarkSourceManifest:
        archive = archive.resolve()
        if not archive.exists():
            raise FileNotFoundError(f"official archive not found: {archive}")
        digest = sha256_file(archive)
        source_dir = self.source_root / benchmark_source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        stored = source_dir / archive.name
        if stored.exists():
            if sha256_file(stored) != digest:
                raise HistoricalBenchmarkError(
                    "an immutable source archive with different bytes already exists"
                )
        else:
            shutil.copy2(archive, stored)
        manifest = BenchmarkSourceManifest(
            benchmark_source_id=benchmark_source_id,
            competition=self.competition,
            year=self.year,
            official_source=official_source,
            downloaded_at=_utc(downloaded_at),
            archive_path=stored.relative_to(self.repo_root).as_posix(),
            archive_sha256=digest,
            archive_size=stored.stat().st_size,
            extracted_files=[],
        )
        manifest = manifest.model_copy(
            update={"manifest_hash": _hash_model(manifest, exclude=("manifest_hash",))}
        )
        _write_json(self.source_manifest_file, manifest)
        run = self.load_run()
        run = self._write_run(
            run.model_copy(update={"source_manifest_hash": manifest.manifest_hash})
        )
        self.log_leakage(
            LeakageEvent.SOURCE_ACCESS,
            f"official archive sealed: {manifest.archive_path}",
            prohibited=False,
            now=downloaded_at,
        )
        return manifest

    def load_source_manifest(self) -> BenchmarkSourceManifest:
        return BenchmarkSourceManifest.model_validate_json(
            self.source_manifest_file.read_text(encoding="utf-8")
        )

    def archive_path(self) -> Path:
        return self.repo_root / self.load_source_manifest().archive_path

    def discover_labels(self) -> list[str]:
        return _labels_from_zip(self.archive_path())

    def commit_selection(
        self,
        now: datetime | None = None,
        *,
        eligible_labels: list[str] | None = None,
    ) -> ProblemSelectionManifest:
        if self.selection_file.exists():
            return ProblemSelectionManifest.model_validate_json(
                self.selection_file.read_text(encoding="utf-8")
            )
        source = self.load_source_manifest()
        labels = self.discover_labels()
        if eligible_labels is not None:
            labels = [label for label in labels if label in set(eligible_labels)]
        if not labels:
            raise HistoricalBenchmarkError("no unburned A/B/C labels discovered")
        candidates = [
            SelectionCandidate(
                label=label,
                digest=hashlib.sha256(f"{source.archive_sha256}:{label}".encode()).hexdigest(),
                rank=1,
            )
            for label in labels
        ]
        candidates.sort(key=lambda item: (item.digest, item.label))
        ordered = [
            item.model_copy(update={"rank": index})
            for index, item in enumerate(candidates, start=1)
        ]
        selection = ProblemSelectionManifest(
            archive_sha256=source.archive_sha256,
            labels=labels,
            ordered=ordered,
            primary_label=ordered[0].label,
            reserve_labels=[item.label for item in ordered[1:]],
            created_at=_utc(now or datetime.now(UTC)),
        )
        selection = selection.model_copy(
            update={"selection_hash": _hash_model(selection, exclude=("selection_hash",))}
        )
        _write_json(self.selection_file, selection)
        run = self.load_run()
        self._write_run(
            run.model_copy(update={"selection_manifest_hash": selection.selection_hash})
        )
        return selection

    def extract_package(self) -> BenchmarkSourceManifest:
        selection = self.load_selection()
        source = self.load_source_manifest()
        if sha256_file(self.archive_path()) != source.archive_sha256:
            raise HistoricalBenchmarkError("official archive changed after sealing")
        entries = _safe_extract(self.archive_path(), self.problem_package_dir)
        source = source.model_copy(update={"extracted_files": entries})
        source = source.model_copy(
            update={"manifest_hash": _hash_model(source, exclude=("manifest_hash",))}
        )
        _write_json(self.source_manifest_file, source)
        run = self.load_run()
        self._write_run(run.model_copy(update={"source_manifest_hash": source.manifest_hash}))
        del selection
        return source

    def load_selection(self) -> ProblemSelectionManifest:
        return ProblemSelectionManifest.model_validate_json(
            self.selection_file.read_text(encoding="utf-8")
        )

    def authorize_reserve(self, prior_run_id: str, reason: str) -> None:
        prior_dir = self.repo_root / "benchmarks" / "historical" / prior_run_id
        prior_run_file = prior_dir / "benchmark_run.json"
        if not prior_run_file.exists():
            raise HistoricalBenchmarkError(f"prior benchmark run not found: {prior_run_id}")
        prior = HistoricalBenchmarkRun.model_validate_json(
            prior_run_file.read_text(encoding="utf-8")
        )
        if prior.protocol_validity is not ProtocolValidity.INVALIDATED_INFRASTRUCTURE:
            raise BlindnessViolation(
                "reserve authorization requires an invalidated infrastructure run"
            )
        atomic_write_text(
            self.reserve_authorization_file,
            json.dumps(
                {
                    "prior_run_id": prior_run_id,
                    "prior_run_hash": prior.run_hash,
                    "reason": reason,
                },
                indent=2,
            )
            + "\n",
        )

    def _reserve_allowed(self, label: str) -> bool:
        selection = self.load_selection()
        if label not in selection.reserve_labels:
            return False
        run = self.load_run()
        if run.protocol_validity is ProtocolValidity.INVALIDATED_INFRASTRUCTURE:
            return True
        if not self.reserve_authorization_file.exists():
            return False
        authorization = json.loads(self.reserve_authorization_file.read_text(encoding="utf-8"))
        prior_id = str(authorization.get("prior_run_id", ""))
        prior_file = self.repo_root / "benchmarks" / "historical" / prior_id / "benchmark_run.json"
        if not prior_file.exists():
            return False
        prior = HistoricalBenchmarkRun.model_validate_json(prior_file.read_text(encoding="utf-8"))
        return prior.protocol_validity is ProtocolValidity.INVALIDATED_INFRASTRUCTURE

    def reveal(
        self,
        *,
        label: str | None = None,
        attempt_id: str,
        now: datetime | None = None,
    ) -> RevealRecord:
        selection = self.load_selection()
        if not self.code_freeze_file.exists():
            raise BlindnessViolation("code freeze must exist before problem reveal")
        if self.reveal_file.exists():
            raise BlindnessViolation("problem has already been revealed for this run")
        chosen = label or selection.primary_label
        if chosen != selection.primary_label and not self._reserve_allowed(chosen):
            raise BlindnessViolation(
                "reserve problems are only allowed after infrastructure invalidation"
            )
        record = RevealRecord(
            selection_hash=selection.selection_hash,
            problem_label=chosen,
            revealed_at=_utc(now or datetime.now(UTC)),
            attempt_id=attempt_id,
        )
        record = record.model_copy(
            update={"reveal_hash": _hash_model(record, exclude=("reveal_hash",))}
        )
        _write_json(self.reveal_file, record)
        self.log_leakage(
            LeakageEvent.SOURCE_ACCESS,
            f"problem revealed: {chosen}",
            prohibited=False,
            now=record.revealed_at,
        )
        run = self.load_run()
        self._write_run(
            run.model_copy(
                update={
                    "problem_label": chosen,
                    "state": HistoricalBenchmarkState.PROBLEM_REVEALED,
                    "problem_revealed_at": record.revealed_at,
                    "updated_at": record.revealed_at,
                }
            )
        )
        return record

    def load_reveal(self) -> RevealRecord:
        return RevealRecord.model_validate_json(self.reveal_file.read_text(encoding="utf-8"))

    def selected_files(self) -> list[Path]:
        reveal = self.load_reveal()
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(reveal.problem_label)}(?:题|卷|[_\-.]|$|[/\\])"
        )
        files = [
            self.problem_package_dir / entry.path
            for entry in self.load_source_manifest().extracted_files
            if pattern.search(entry.path)
        ]
        if not files:
            raise HistoricalBenchmarkError(
                f"no files matched selected label {reveal.problem_label}"
            )
        return sorted(files)

    def _derive_text(self, source: Path) -> Path:
        suffix = source.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return source
        target = self.root / "derived" / (source.name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".pdf":
            try:
                import pymupdf
            except ImportError as exc:
                raise HistoricalBenchmarkError(
                    "PyMuPDF is required for Unicode PDF extraction"
                ) from exc
            document = pymupdf.open(source)
            try:
                page_texts: list[str] = []
                for page in document:
                    extracted = page.get_text()
                    page_texts.append(extracted if isinstance(extracted, str) else "")
                text = "\n".join(page_texts)
            finally:
                document.close()
            target.write_text(text, encoding="utf-8")
        elif suffix in {".docx", ".doc"}:
            result = subprocess.run(
                ["pandoc", str(source), "-t", "plain", "-o", str(target)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise HistoricalBenchmarkError(f"pandoc failed for {source.name}: {result.stderr}")
        else:
            raise HistoricalBenchmarkError(f"unsupported problem file: {source.name}")
        return target

    @staticmethod
    def _is_problem_document(path: Path) -> bool:
        return path.suffix.lower() in {".txt", ".md", ".markdown", ".pdf", ".doc", ".docx"}

    def problem_text(self) -> str:
        self.load_reveal()
        parts: list[str] = []
        for source in self.selected_files():
            if not self._is_problem_document(source):
                continue
            derived = self._derive_text(source)
            parts.append(f"## {source.name}\n\n{derived.read_text(encoding='utf-8')}")
        self.log_leakage(
            LeakageEvent.SOURCE_ACCESS,
            "read selected official problem files",
            prohibited=False,
        )
        return "\n\n".join(parts)

    def create_project(self) -> ProjectPaths:
        if self.project_dir.exists():
            raise HistoricalBenchmarkError("benchmark project already exists")
        paths, _ = create_project(
            self.project_dir,
            f"bench-{self.benchmark_run_id}",
            task=TaskFamily.PREDICTION,
        )
        reveal = self.load_reveal()
        problem_files: list[Path] = []
        for source in self.selected_files():
            if not self._is_problem_document(source):
                continue
            derived = self._derive_text(source)
            if derived.suffix.lower() not in {".txt", ".md", ".markdown"}:
                continue
            problem_files.append(derived)
        if not problem_files:
            raise HistoricalBenchmarkError("no text problem file for selected label")
        ingest_files(paths, problem_files, SourceRole.PROBLEM)
        for source in self.selected_files():
            if source.suffix.lower() in {".csv", ".xlsx", ".json", ".zip"}:
                ingest_files(paths, [source], SourceRole.ATTACHMENT)
        self.log_leakage(
            LeakageEvent.SOURCE_ACCESS,
            f"ingested selected label {reveal.problem_label} into project",
            prohibited=False,
        )
        return paths

    def run_router(self, model: Any) -> RoutingOutcome:
        if not self.project_dir.exists():
            raise HistoricalBenchmarkError("benchmark project has not been created")
        paths = ProjectPaths(root=self.project_dir)
        outcome = route_project(paths, model)
        atomic_write_text(
            self.root / "router_result.json",
            outcome.decision.model_dump_json(indent=2) + "\n",
        )
        run = self.load_run()
        self._write_run(
            run.model_copy(
                update={
                    "state": HistoricalBenchmarkState.PROBLEM_ANALYZED,
                    "updated_at": datetime.now(UTC),
                }
            )
        )
        return outcome

    def requirements_from_spec(self, spec: ProblemSpec) -> list[ProblemRequirement]:
        requirements: list[ProblemRequirement] = []
        for subproblem in spec.subproblems:
            text = " ".join([subproblem.statement, subproblem.goal, *subproblem.outputs]).strip()
            requirements.append(
                ProblemRequirement(
                    requirement_id=subproblem.subproblem_id,
                    source_location=f"problem_spec.subproblems[{subproblem.subproblem_id}]",
                    question_text=text,
                    question_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    required_output="; ".join(subproblem.outputs) or subproblem.goal,
                )
            )
        return requirements

    def capability_gaps(
        self,
        decision: RoutingDecision,
        requirements: list[ProblemRequirement],
    ) -> list[CapabilityGap]:
        gaps: list[CapabilityGap] = []
        unsupported = list(decision.unsupported_requirements)
        if unsupported:
            for index, item in enumerate(unsupported, start=1):
                gaps.append(
                    CapabilityGap(
                        gap_id=f"gap-{index}",
                        problem_requirement=item,
                        failure_mode="router_unsupported_requirement",
                        existing_capability_attempted="problem router and capability audit",
                        why_insufficient="no existing TaskFamily fully supports this requirement",
                        scientific_consequence="the requirement remains unresolved",
                        severity="critical",
                        possible_future_capability=(
                            "add the missing modeling capability after Phase 24.1"
                        ),
                    )
                )
        capabilities = {item.task_family for item in decision.capabilities}
        for requirement in requirements:
            if requirement.status is not RequirementStatus.UNRESOLVED:
                continue
            if not gaps:
                gaps.append(
                    CapabilityGap(
                        gap_id="gap-1",
                        problem_requirement=requirement.requirement_id,
                        failure_mode="routing_or_execution_blocked",
                        existing_capability_attempted="problem router",
                        why_insufficient=(
                            "the routed capability did not produce evidence for this requirement"
                        ),
                        scientific_consequence="the requirement remains unresolved",
                        severity="critical",
                        possible_future_capability="requires an explicit benchmark follow-up",
                    )
                )
            break
        del capabilities
        return gaps

    def blindness_audit(self, now: datetime | None = None) -> BenchmarkBlindnessAudit:
        entries = self.leakage_entries()
        prohibited = [entry for entry in entries if entry.prohibited]
        human = [entry for entry in entries if entry.event_type is LeakageEvent.HUMAN_INTERVENTION]
        checks: dict[str, bool] = {}
        source = self.load_source_manifest()
        archive = self.repo_root / source.archive_path
        checks["official_archive_present"] = archive.exists()
        checks["official_archive_hash_matches"] = (
            archive.exists() and sha256_file(archive) == source.archive_sha256
        )
        selection = self.load_selection()
        reveal = self.load_reveal()
        checks["selection_before_reveal"] = selection.created_at <= reveal.revealed_at
        checks["selection_committed"] = bool(
            selection.selection_hash
            and selection.selection_hash == self.load_run().selection_manifest_hash
        )
        freeze = BenchmarkCodeFreeze.model_validate_json(
            self.code_freeze_file.read_text(encoding="utf-8")
        )
        checks["code_freeze_present"] = bool(freeze.freeze_hash)
        checks["code_freeze_commit_matches_run"] = (
            freeze.git_commit == self.load_run().code_freeze_commit
        )
        checks["no_prohibited_access"] = not prohibited
        checks["no_human_scientific_intervention"] = not human
        checks["one_reveal"] = self.reveal_file.exists()
        if reveal.problem_label != selection.primary_label:
            checks["reserve_authorized"] = self._reserve_allowed(reveal.problem_label)
        audit = BenchmarkBlindnessAudit(
            checks=checks,
            prohibited_access_count=len(prohibited),
            human_scientific_intervention_count=len(human),
            passed=all(checks.values()),
            notes=[
                "Operational blindness verifies runtime information isolation, not "
                "absence of relevant material from pretrained model parameters."
            ],
        )
        _write_json(self.root / "blindness_audit.json", audit)
        return audit

    def invalidate(self, reason: str, now: datetime | None = None) -> HistoricalBenchmarkRun:
        run = self.load_run()
        run = self._write_run(
            run.model_copy(
                update={
                    "state": HistoricalBenchmarkState.INVALIDATED_INFRASTRUCTURE,
                    "protocol_validity": ProtocolValidity.INVALIDATED_INFRASTRUCTURE,
                    "invalidation_reason": reason,
                    "updated_at": _utc(now or datetime.now(UTC)),
                }
            )
        )
        atomic_write_text(
            self.root / "invalidation.json",
            json.dumps({"reason": reason, "created_at": run.updated_at.isoformat()}, indent=2)
            + "\n",
        )
        return run

    def finalize_result(
        self,
        *,
        scientific_verdict: str,
        experiment_ids: list[str] | None = None,
        failed_experiment_ids: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        model_freeze_hash: str | None = None,
        evidence_freeze_hash: str | None = None,
        paper_freeze_hash: str | None = None,
        paper_hash: str | None = None,
        number_audit: str | None = None,
        math_audit: str | None = None,
        red_team: str | None = None,
        judge: str | None = None,
        unresolved_issues: list[str] | None = None,
        capability_gaps: list[CapabilityGap] | None = None,
        capability_fit: CapabilityFitAssessment | None = None,
        freshness_registry_hash: str = "",
    ) -> BenchmarkResult:
        run = self.load_run()
        if run.problem_label is None:
            raise HistoricalBenchmarkError("cannot finalize before problem reveal")
        audit = self.blindness_audit()
        router = _read_json(self.root / "router_result.json")
        requirements = self.load_requirements()
        coverage = {status.value: 0 for status in RequirementStatus}
        for requirement in requirements:
            coverage[requirement.status.value] += 1
        gaps = capability_gaps or (
            capability_fit.blocking_gaps if capability_fit is not None else []
        )
        result = BenchmarkResult(
            benchmark_run_id=self.benchmark_run_id,
            competition=self.competition,
            year=self.year,
            problem_label=run.problem_label,
            source_archive_hash=self.load_source_manifest().archive_sha256,
            selection_manifest_hash=self.load_selection().selection_hash,
            freshness_registry_hash=freshness_registry_hash,
            code_freeze_commit=run.code_freeze_commit,
            protocol_validity=run.protocol_validity,
            capability_fit=capability_fit,
            capability_fit_verdict=(
                capability_fit.verdict
                if capability_fit is not None
                else CapabilityVerdict.NOT_REACHED
            ),
            blindness_audit=audit,
            router_result=router,
            requirement_coverage=coverage,
            experiment_ids=experiment_ids or [],
            failed_experiment_ids=failed_experiment_ids or [],
            evidence_ids=evidence_ids or [],
            model_freeze_hash=model_freeze_hash,
            evidence_freeze_hash=evidence_freeze_hash,
            paper_freeze_hash=paper_freeze_hash,
            paper_hash=paper_hash,
            number_audit=number_audit,
            math_audit=math_audit,
            red_team=red_team,
            judge=judge,
            scientific_verdict=scientific_verdict,
            unresolved_issues=unresolved_issues or [],
            new_capability_gaps=[gap.gap_id for gap in gaps],
            human_intervention_count=audit.human_scientific_intervention_count,
        )
        result = result.model_copy(
            update={"result_hash": _hash_model(result, exclude=("result_hash",))}
        )
        _write_json(self.root / "benchmark_result.json", result)
        if capability_fit is not None:
            _write_json(self.root / "capability_fit.json", capability_fit)
        atomic_write_text(
            self.root / "capability_gaps.json",
            json.dumps({"gaps": [gap.model_dump(mode="json") for gap in gaps]}, indent=2) + "\n",
        )
        run = self._write_run(
            run.model_copy(
                update={
                    "state": (
                        HistoricalBenchmarkState.PASS
                        if scientific_verdict == "pass"
                        and audit.passed
                        and run.protocol_validity is ProtocolValidity.VALID
                        else HistoricalBenchmarkState.BLOCKED
                    ),
                    "updated_at": datetime.now(UTC),
                }
            )
        )
        return result

    def requirements_file(self) -> Path:
        return self.root / "requirements.json"

    def write_requirements(self, requirements: list[ProblemRequirement]) -> None:
        atomic_write_text(
            self.requirements_file(),
            json.dumps(
                [requirement.model_dump(mode="json") for requirement in requirements],
                indent=2,
            )
            + "\n",
        )

    def load_requirements(self) -> list[ProblemRequirement]:
        if not self.requirements_file().exists():
            return []
        return [
            ProblemRequirement.model_validate(item)
            for item in json.loads(self.requirements_file().read_text(encoding="utf-8"))
        ]

    def write_code_freeze(self, freeze: BenchmarkCodeFreeze) -> None:
        _write_json(self.code_freeze_file, freeze)
        run = self.load_run()
        self._write_run(run.model_copy(update={"code_freeze_commit": freeze.git_commit}))

    def verify_result(self) -> dict[str, object]:
        path = self.root / "benchmark_result.json"
        if not path.exists():
            return {"ok": False, "reason": "benchmark_result_missing"}
        result = BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
        expected = _hash_model(result, exclude=("result_hash",))
        return {
            "ok": expected == result.result_hash,
            "expected": expected,
            "recorded": result.result_hash,
        }

    def verify_code_freeze(self) -> dict[str, object]:
        if not self.code_freeze_file.exists():
            return {"ok": False, "reason": "code_freeze_missing"}
        freeze = BenchmarkCodeFreeze.model_validate_json(
            self.code_freeze_file.read_text(encoding="utf-8")
        )
        expected = _hash_model(freeze, exclude=("freeze_hash",))
        current_tree = _code_tree_hash(self.repo_root)
        return {
            "ok": expected == freeze.freeze_hash and current_tree == freeze.code_tree_hash,
            "freeze_hash_ok": expected == freeze.freeze_hash,
            "code_tree_unchanged": current_tree == freeze.code_tree_hash,
        }

    def write_sha256sums(self) -> Path:
        lines: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                lines.append(f"{sha256_file(path)}  {path.relative_to(self.root).as_posix()}")
        target = self.root / "SHA256SUMS"
        atomic_write_text(target, "\n".join(lines) + "\n")
        return target


__all__ = [
    "BenchmarkInfrastructureError",
    "BlindnessViolation",
    "HistoricalBenchmarkError",
    "HistoricalBenchmarkHarness",
    "create_code_freeze",
]
