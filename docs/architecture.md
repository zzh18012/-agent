# MathModelingOS Architecture

## The one invariant

> Agent output is a proposal. Execution output is evidence.

Everything else follows from this. The system is split into four layers so that the
deterministic parts of a research project are never controlled by the probabilistically
generated parts.

## Four layers

### Control plane

Decides *what to try next*.

- Master Orchestrator and its explicit state machine
- Problem Analyst, Research Agent, Data Agent, Model Council
- Experiment Planner, Tournament, human-review gates

The control plane emits proposals (`ProblemSpec`, `ModelProposal`, `ExperimentSpec`) and
queues experiments. It cannot manufacture results.

### Compute plane

Does *the actual work*.

- `LocalExecutor` (process execution with cwd, timeout, captured stdout/stderr/exit code)
- Snakemake workflows
- Model engines (scikit-learn, statsmodels, Pyomo, ...)
- MLflow experiment ledger

Every formal artifact is produced here, from a recorded code version and configuration.

### Evidence plane

Decides *which facts may be trusted*.

- Independent validators (leakage, numerical, statistical, reproducibility, ...)
- Blind reviewer with a second implementation where practical
- `verified/` store with hashes and provenance
- `ClaimRecord`s linking paper statements to evidence IDs

A result enters `verified/` only with an experiment ID, source artifact, hash, and
passing validation reports. Anything that fails validation stays failed; it is never
papered over.

### Paper layer

Expresses *only facts from the evidence plane*.

- Paper AST (structured document intermediate representation)
- Renderer (Markdown/LaTeX/Quarto) separated from the AST
- Red team review and author response

The writer has no computation privileges and no access to unverified numbers.

## Data flow

```text
problem + rules + raw data
        │
        ▼
Project intake ──► manifest with sha256 per input ──► data/raw (read-only)
        │
        ▼
ProblemSpec + DataQualityReport ──(quality gate)──► ModelProposal(s)
        │
        ▼
ExperimentSpec ──► Experiment Planner ──► Snakemake ──► MLflow
        │
        ▼
ValidationReport(s) ──(blocking issues?)──► verified/
        │
        ▼
Paper AST ──► renderer ──► Red Team ──► release
```

## State machine

Orchestrator transitions are rule-based, not LLM-chosen:

`INIT → PROJECT_INGESTED → PROBLEM_UNDERSTOOD → DATA_VALIDATED →
RESEARCH_COMPLETED → CANDIDATES_PROPOSED → PROTOTYPING → PROTOTYPED →
FULL_EXPERIMENT → EXPERIMENT_COMPLETED → VALIDATING → VERIFIED →
EVIDENCE_FROZEN → PAPER_DRAFTED → RED_TEAM_REVIEW → READY_FOR_RELEASE → RELEASED`

With `BLOCKED`, `FAILED`, and `REQUIRES_HUMAN_REVIEW` as terminal-ish escape states.

## Safety boundary (no Docker)

- All writes stay inside the repository or the active project directory.
- `data/raw/` is read-only after intake.
- No `sudo`, no system Python changes, no OS configuration changes.
- Destructive commands are rejected by the executor; commands are recorded.
- Secrets live in `.env`, which is git-ignored.

## Deliberate non-goals (for now)

- Containers, VMs, or distributed execution.
- Prebuilt adapters for every model vendor.
- A web UI before the CLI, evidence chain, and workflow are real.
