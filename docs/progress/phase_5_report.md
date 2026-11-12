# Phase 5 Report (deterministic core)

Date: 2026-11-12

## Scope

Typed contracts for research and modeling, a capability registry, dataset
fingerprinting, and the experiment planner. The LLM-backed Research Agent and Model
Council are intentionally deferred until an API key exists for live verification.

## Created / modified

- `src/mmos/schemas/research.py` — `MethodCard`, `ResearchResult`
- `src/mmos/schemas/proposal.py` — `ExpertRole`, `Complexity`, `ModelProposal`,
  `ProposalScore`, `CouncilDecision` (selected/rejected from scores)
- `src/mmos/schemas/experiment.py` — `ExperimentSpec`
- `src/mmos/schemas/registry.py` — `ModelCapability`, `TaskType`, `Interpretability`,
  `ComputeClass`
- `src/mmos/registries/model_registry.py` — YAML-backed registry with duplicate-id
  rejection and task filtering
- `model_registry/registry.yaml` — 16 entries across regression, classification,
  forecasting, optimization, sensitivity, including explicit baselines
- `src/mmos/core/dataset.py` — order-independent dataset fingerprint over registered
  data hashes
- `src/mmos/experiments/planner.py` — proposal → `ExperimentSpec` with fair defaults
  (same split, metrics, seed) and no winner selection
- `src/mmos/core/project.py` — experiment spec save/load under
  `experiments/<id>/experiment_spec.json`
- Tests: registry, planner, dataset fingerprint

## Key decisions

1. **The planner cannot choose a winner.** It only creates a reproducible contract;
   the winner must come from experiments.
2. **Proposals must name registry models.** A proposal with no registry model raises
   instead of being silently converted into some default model. This prevents the LLM
   from naming a model that cannot actually run.
3. **Baselines are first-class registry entries** (`is_baseline: true`), so the
   tournament can require them explicitly.
4. **Dataset identity comes from the manifest.** The fingerprint is a hash over ordered
   (path, sha256) pairs of DATA inputs, independent of ingestion order.
5. **CouncilDecision is a selection, not a verdict.** It selects proposals for
   prototyping and computes rejected ids from scores; it cannot claim a final model.
6. **Council/research agents are deferred.** Without a key there is no way to verify
   DeepSeek handles these nested tool schemas. Writing untested orchestration would
   violate the "no fabricated results" rule. Schemas and prompts can be added when the
   key is available.

## Verification (actually executed)

```text
uv run pytest        -> 74 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

## Next phase (Phase 6)

`LocalExecutor`: project-scoped command execution with argv-only invocation, denylist
for dangerous executables, deny-by-default shell usage, timeout, captured stdout/stderr,
and an auditable `ExecutionResult`.

## Update — 2026-11-12 (after API key was configured)

The deferred research and council agents are now implemented and live-verified:

- `ResearchAgent` produces `ResearchResult` from the ProblemSpec and registry catalog.
- `ModelCouncil` selects 2-4 expert roles, runs independent proposal rounds, a
  cross-critique round, and a judge; proposals are normalized to system-owned ids and
  dropped if they name unknown subproblems or unregistered models; judge output is
  filtered and falls back deterministically only when it selects nothing valid.
- CLI: `research`, `propose-models`, `plan-experiment`. Experiment planning injects the
  registered baseline by default.
- Live integration test (`tests/test_live_agents.py`) passed in 220s with real DeepSeek
  calls, including nested structured output (9 model calls).

Known limitation: the council can only select models that exist in the executable
registry; mechanistic proposals without a registered engine are dropped.
