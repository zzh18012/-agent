# MathModelingOS

MathModelingOS (`mmos`) is a local-first system for running mathematical modeling
competitions and research projects end to end: understanding the problem, checking
data, proposing and competing models, executing real experiments, validating results,
freezing verified evidence, and only then writing the paper.

The core rule of the system is simple:

> **An LLM may propose anything. Only a real program execution may produce a result.**

Every number that ends up in a report must trace back to source data, an experiment
configuration, a code version, an actual run, and a validation report. Natural-language
output from an agent is never treated as evidence.

## Status

This repository is under phased construction. See `docs/progress/` for what is real today.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Environment and repository inspection | done |
| 1 | Project skeleton, configuration, schemas, logging, CLI, tests | done |
| 2 | Project intake, manifest hashing, state machine, orchestrator | done |
| 3 | Agent runtime (PydanticAI), model factory, Problem Analyst, ProblemSpec | done (live-verified) |
| 4 | Data profiling (Polars), Pandera schema validation, quality/leakage gates | done (CSV only) |
| 5 | Research Agent, Model Council, registry, experiment planner | done (live-verified) |
| 6 | LocalExecutor with safety boundary, timeout, audited execution records | done |
| 7 | Snakemake workflow: data gate → baseline/candidates → evaluation → artifacts | done (tabular regression/classification) |
| 8 | MLflow ledger: per-project SQLite store, params/tags/metrics/artifacts | done |
| 9 | Tournament: comparability checks, baseline requirement, selection + challenger | done |
| 10 | Independent validators: leakage, numerical, statistical, reproducibility | done (optimization/robustness/sensitivity auditors pending) |
| 11 | Verified evidence store, artifact hashes, provenance, claims, freeze gate | done |
| 12 | Paper AST, evidence-grounded tables/figures, Markdown renderer | done |
| 13 | Red Team attack, author response, judge verdict with conservative aggregation | done (live-verified) |
| 14 | Automated `mmos run` pipeline, release gate, blocked-release handoff | implemented |
| 15 | Scientific hardening: target/temporal metadata, rolling-origin validation, naive baselines, MASE, uncertainty | implemented; demo required revision before release |
| 16 | EvaluationSet components, MASE fold semantics, information sets, practical equivalence | done; demo released with judge PASS |

Nothing in the README should be read as a claim that later phases are implemented.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git

Docker is intentionally not used. Everything runs directly in a project-local virtual
environment on the host machine.

## Install

```bash
uv sync
```

This creates `.venv/`, installs the package in editable mode, and installs the dev tools.

## Configure

```bash
cp .env.example .env
```

Then set `MMOS_LLM_API_KEY` (and optionally `MMOS_LLM_PROVIDER`, `MMOS_LLM_MODEL`,
`MMOS_LLM_BASE_URL`). Never commit `.env`.

Check the environment:

```bash
uv run mmos doctor
```

## CLI

```bash
uv run mmos --version
uv run mmos doctor

# create a project under projects/
uv run mmos init "My First Project"

# ingest inputs; originals are copied into the project, never modified
uv run mmos ingest <project_id> ./problem.pdf --role problem
uv run mmos ingest <project_id> ./train.csv --role data

# inspect state, inputs, and legal next states
uv run mmos status <project_id>

# extract a structured ProblemSpec (requires MMOS_LLM_API_KEY)
uv run mmos analyze-problem <project_id>

# profile data and run quality/leakage gates (CSV)
uv run mmos analyze-data <project_id> --target <target_column>

# survey methods, run the expert council, plan a concrete experiment
uv run mmos research <project_id>
uv run mmos propose-models <project_id>
uv run mmos plan-experiment <project_id> --proposal-id <id> --target <column> \
  --feature <column> --feature <column>

# execute a planned experiment through Snakemake (spec must exist first)
uv run mmos run-experiment <project_id> <experiment_id> [--dry-run]

# compare completed experiments against a registered baseline
uv run mmos tournament <project_id> <experiment_id...> --metric rmse

# run independent validators against one experiment
uv run mmos validate <project_id> <experiment_id>

# freeze validated results into verified evidence, then register claims
uv run mmos freeze-evidence <project_id> <experiment_id>
uv run mmos add-claim <project_id> --statement "..." --evidence <evidence_id> --accept

# build and render a Markdown technical report from verified evidence
uv run mmos paper <project_id> <experiment_id> --tournament <tournament_id>

# adversarial review of the generated paper
uv run mmos redteam <project_id> <experiment_id> [--tournament <tournament_id>]

# full pipeline: analyze -> research -> council -> experiment -> evidence -> paper -> red team
uv run mmos run <project_id> --target <column> --feature <column> --feature <column>

# temporal data: rolling-origin evaluation with naive baselines (horizon required)
uv run mmos run <project_id> --target y --feature x1 --feature x2 \n  --time-column period --horizon 1 --primary-metric rmse
```


## Dataset metadata (optional sidecar)

Ingest a `dataset_metadata.json` as an attachment to declare provenance and time
structure. Statistics are always computed from the real data, never guessed by an LLM.

```json
{
  "target_name": "y",
  "target_description": "demand per period",
  "target_unit": null,
  "unit_status": "synthetic_units",
  "temporal": {
    "is_temporal": true,
    "time_column": "period",
    "forecast_horizon": 1,
    "seasonal_period": null,
    "ordering_required": true
  },
  "features": [
    {"name": "x1", "availability": "known_future"},
    {"name": "x2", "availability": "known_future"}
  ],
  "provenance": {
    "is_synthetic": true,
    "generator": "linear_demand_generator_v1",
    "generation_parameters": {"periods": 60},
    "seed": 20261112,
    "code_path": "examples/linear_demand/generate.py"
  }
}
```

Temporal datasets use rolling-origin validation; random KFold is rejected by the
temporal validator, and a naive/drift baseline is required. If the horizon is unknown,
the pipeline stops instead of guessing. The paper reports target unit status, descriptive
statistics, `RMSE / target_std`, MASE, fold variability, and bootstrap intervals.

## Concepts

- **Agent** — an LLM-backed role with a typed input and a typed output (for example a
  problem analyst that produces a `ProblemSpec`). Agents propose; they do not certify.
- **Skill** — reusable domain knowledge or procedure loaded into an agent's context.
- **Workflow** — the Snakemake DAG that performs the real computation. Formal results
  must be produced by the workflow, not by an agent running a notebook by hand.
- **Experiment** — one fully specified, reproducible run: data hash, split, model,
  parameters, seed, metrics, artifacts. Tracked in MLflow.
- **Evidence** — a verified artifact or number with a provenance record. The paper layer
  may only read from verified evidence.

## Repository layout

```text
src/mmos/          package code (cli, config, core, schemas, ...)
tests/             pytest suite
docs/              architecture and phase reports
projects/          runtime project data (git-ignored)
configs/           reusable configuration
workflow/          Snakemake workflows (Phase 7)
model_registry/    model capability registry
skills/            agent skills
templates/         report/paper templates
```

A runtime project looks like:

```text
projects/<project_id>/
├── problem/       题目原文（只读）
├── rules/         比赛规则（只读）
├── data/raw/      原始数据（只读，永不自动覆盖）
├── data/interim/  中间数据
├── data/processed/
├── data/features/
├── experiments/   实验配置与输出
├── artifacts/     产物
├── verified/      经过验证的证据
├── reviews/       审稿与复核
├── paper/         论文草稿
├── logs/
└── manifests/
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Common problems

- **`uv sync` fails** — confirm `uv --version` works and the network allows PyPI access.
- **`mmos` command not found** — use `uv run mmos ...` or activate `.venv`.
- **Python version mismatch** — `.python-version` pins 3.12 for the scientific stack;
  let uv manage the interpreter rather than changing the system Python.

## Design documents

- `docs/architecture.md` — control plane, compute plane, evidence plane, and paper layer.
