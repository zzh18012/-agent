# Phase 12 Report

Date: 2026-11-12

## Scope

The paper layer: a structured paper AST, evidence-grounded tables and figures, and a
Markdown renderer.

## Created / modified

- `src/mmos/schemas/paper.py` — `PaperAST`, `PaperSection`, `PaperTable`, `PaperFigure`,
  `FigureSpec`
- `src/mmos/paper/figures.py` — bar chart generated only from verified evidence records,
  saved as PNG/PDF/SVG with hashes and a figure spec
- `src/mmos/paper/builder.py` — assembles the AST from the verified evidence index,
  supported claims, experiment spec, validation summary, and optional tournament
- `src/mmos/paper/render.py` — Markdown renderer separated from the AST
- `src/mmos/core/project.py` — `load_tournament_result`
- `src/mmos/cli/main.py` — `paper` command
- Tests: full evidence → claim → tournament → paper chain, missing-evidence refusal,
  CLI rendering

## Key decisions

1. **The builder cannot read unverified numbers.** Its only numeric source is
   `verified/numbers.json`; displayed values come from `EvidenceRecord.display_value`.
   Configuration values (dataset hash, seed, splits) are identifiers, not findings.
2. **Figures are generated from evidence ids.** Each figure spec records the evidence
   ids and sha256 for PNG/PDF/SVG, so a chart can be traced back to the numbers.
3. **Only supported claims appear.** Proposed claims are excluded, and supported claims
   whose evidence ids are missing from the store are excluded.
4. **AST and renderer are separate.** The renderer does no computation and adds no
   numbers; it only formats the AST. LaTeX/Quarto renderers can be added later.
5. **Limitations are explicit** in every generated report: one dataset, no robustness or
   sensitivity analysis yet, heuristic leakage checks.
6. **No LLM is involved in this phase.** The report is a deterministic technical report
   from verified evidence. An LLM paper writer must later operate on the same AST and
   evidence, and cannot bypass them.

## Verification (actually executed)

```text
uv run pytest        -> 118 passed, 1 skipped
uv run ruff check .  -> All checks passed
uv run pyright       -> 0 errors, 0 warnings
```

The paper test runs a real experiment, validates it, freezes evidence, runs a real
tournament, registers one supported and one proposed claim, builds the AST, checks that
table values equal evidence display values, checks figure files and hashes, renders the
Markdown, and exercises the CLI.

## Deferred

- LLM-assisted paper writing (needs an API key; must consume only the AST/evidence).
- LaTeX and Quarto renderers, competition templates.
- Red Team review and judge loop.

## Next phase (Phase 13)

Red Team review over the generated paper/claims, plus author response and judge.
This phase is LLM-backed and cannot be honestly verified until a project API key exists.
