import json

from mmos.core.project import ProjectPaths, load_optimization_spec, load_problem_spec
from mmos.evidence import load_claims, load_evidence_index
from mmos.schemas.paper import PaperAST, PaperSection, PaperTable


def build_optimization_paper(paths: ProjectPaths, experiment_id: str) -> PaperAST:
    index = load_evidence_index(paths)
    records = [record for record in index.records.values() if record.experiment_id == experiment_id]
    if not records:
        raise ValueError(f"no verified optimization evidence for {experiment_id!r}; freeze first")
    by_key = {record.key: record for record in records}
    spec = load_optimization_spec(paths, experiment_id)
    problem = load_problem_spec(paths) if paths.problem_spec_file.exists() else None
    model = spec.model

    methods = sorted(
        {
            key[len("optimization.") : -len(".status")]
            for key in by_key
            if key.startswith("optimization.") and key.endswith(".status")
        }
    )
    rows: list[list[str]] = []
    evidence_ids: list[str] = []
    for method in methods:
        status = by_key.get(f"optimization.{method}.status")
        objective = by_key.get(f"optimization.{method}.objective_value")
        gap = by_key.get(f"optimization.{method}.optimality_gap")
        feasible = by_key.get(f"optimization.{method}.feasible")
        rows.append(
            [
                method,
                status.display_value if status else "-",
                objective.display_value if objective else "-",
                gap.display_value if gap else "-",
                feasible.display_value if feasible else "-",
            ]
        )
        for record in (status, objective, gap, feasible):
            if record is not None:
                evidence_ids.append(record.evidence_id)
    table = PaperTable(
        table_id=f"tab-{experiment_id}-optimization",
        caption="Solver and baseline results recomputed from the frozen solution.",
        columns=["Method", "Status", "Objective", "Gap", "Feasible"],
        rows=rows,
        evidence_ids=evidence_ids,
    )

    selected = by_key.get("optimization.selected_method")
    solver_objective = by_key.get("optimization.ortools.cbc.objective_value")
    solver_gap = by_key.get("optimization.ortools.cbc.optimality_gap")
    solver_feasible = by_key.get("optimization.ortools.cbc.feasible")
    baseline_objective = by_key.get("optimization.greedy_feasible.objective_value")
    improvement = by_key.get("optimization.relative_improvement")
    abstract = (
        f"This report is generated from {len(records)} verified optimization records for "
        f"experiment {experiment_id}. Model `{model.name}` has sense {model.sense}. "
        f"Selected method: {selected.display_value if selected else 'none'}. "
        f"Solver objective: "
        f"{solver_objective.display_value if solver_objective else 'unknown'}, "
        f"gap: {solver_gap.display_value if solver_gap else 'unknown'}, "
        f"feasible: {solver_feasible.display_value if solver_feasible else 'unknown'}. "
    )
    if baseline_objective is not None:
        abstract += f"Greedy baseline objective: {baseline_objective.display_value}. "
    if improvement is not None:
        abstract += f"Relative improvement over the baseline: {improvement.display_value}. "
    abstract += "Every number here comes from the verified evidence store."

    sections: list[PaperSection] = []
    if problem is not None:
        sections.append(
            PaperSection(
                heading="Problem understanding",
                blocks=[
                    problem.overall_goal,
                    "\n".join(
                        f"- {subproblem.subproblem_id}: {subproblem.statement}"
                        for subproblem in problem.subproblems
                    ),
                ],
            )
        )
    else:
        sections.append(
            PaperSection(
                heading="Problem understanding",
                blocks=["No ProblemSpec is available for this project."],
            )
        )

    variable_lines = "\n".join(
        f"- {variable.name}: {variable.type.value} in [{variable.lower}, {variable.upper}]"
        for variable in model.variables
    )
    objective_terms = " + ".join(
        f"{coefficient}*{name}" for name, coefficient in model.objective.items()
    )
    constraint_lines = "\n".join(
        f"- {constraint.name}: "
        + (f"lower {constraint.lower} " if constraint.lower is not None else "")
        + (f"upper {constraint.upper}" if constraint.upper is not None else "")
        + " on "
        + ", ".join(
            f"{coefficient}*{name}" for name, coefficient in constraint.coefficients.items()
        )
        for constraint in model.constraints
    )
    sections.append(
        PaperSection(
            heading="Optimization model",
            blocks=[
                f"Variables and bounds:\n{variable_lines}",
                f"Objective ({model.sense}): {objective_terms}.",
                f"Linear constraints:\n{constraint_lines or 'none'}.",
            ],
        )
    )
    enumeration_optimum = by_key.get("optimization.enumeration.optimum_value")
    enumeration_count = by_key.get("optimization.enumeration.evaluated_subsets")
    enumeration_subsets = by_key.get("optimization.enumeration.optimal_subsets")
    enumeration_text = ""
    if enumeration_optimum is not None and enumeration_count is not None:
        enumeration_text = (
            f"Independent enumeration of {enumeration_count.display_value} binary subsets "
            f"found optimum {enumeration_optimum.display_value}"
            + (
                f" ({enumeration_subsets.display_value} optimal subsets)."
                if enumeration_subsets is not None
                else "."
            )
        )
    sections.append(
        PaperSection(
            heading="Results",
            blocks=[
                f"Table `{table.table_id}` reports solver and baseline results.",
                "Optimality gap is |best_bound - objective| / max(|objective|, 1e-9); "
                "an optimal status has gap 0. Feasibility and the objective were "
                "independently recomputed from the variable values by the optimization "
                "auditor.",
                *([enumeration_text] if enumeration_text else []),
            ],
        )
    )

    tournament_status = by_key.get("tournament.status")
    if tournament_status is not None:

        def tournament_value(key: str) -> str:
            record = by_key.get(key)
            return record.display_value if record is not None else "unknown"

        winner = tournament_value("tournament.winner_experiment_id")
        candidate_id = tournament_value("tournament.candidate_experiment_id")
        baseline_id = tournament_value("tournament.baseline_experiment_id")
        sense = tournament_value("tournament.objective_sense")
        candidate_objective = tournament_value("tournament.candidate_objective")
        baseline_objective = tournament_value("tournament.baseline_objective")
        relative = tournament_value("tournament.relative_improvement")
        candidate_feasible = tournament_value("tournament.candidate_feasible")
        baseline_feasible = tournament_value("tournament.baseline_feasible")
        candidate_optimality = tournament_value("tournament.candidate_optimality")
        baseline_role = tournament_value("tournament.baseline_role")
        verification = tournament_value("tournament.independent_verification")
        warnings = tournament_value("tournament.warnings")
        status = tournament_status.display_value
        if status == "winner" and winner == candidate_id:
            conclusion = (
                f"On this benchmark instance, the candidate experiment achieved a "
                f"higher {sense} objective ({candidate_objective}) than the baseline "
                f"({baseline_objective})."
            )
        elif status == "equivalent":
            conclusion = (
                "The candidate and baseline are objective-equivalent on this benchmark "
                "instance within the configured tolerance."
            )
        elif status == "winner":
            conclusion = (
                f"On this benchmark instance, the baseline experiment achieved a "
                f"better {sense} objective than the candidate."
            )
        else:
            conclusion = (
                "No winner was declared: the tournament result is "
                f"{status}; no superiority claim is made."
            )
        sections.append(
            PaperSection(
                heading="Optimization comparison",
                blocks=[
                    conclusion,
                    f"Candidate `{candidate_id}` (feasible={candidate_feasible}, "
                    f"optimality={candidate_optimality}); baseline `{baseline_id}` "
                    f"(feasible={baseline_feasible}, role={baseline_role}).",
                    f"Objective sense: {sense}. Candidate objective: "
                    f"{candidate_objective}; baseline objective: {baseline_objective}; "
                    f"relative improvement: {relative}.",
                    f"Independent verification: {verification}. Warnings: {warnings}.",
                    "These results refer to the evaluated instance only; they do not "
                    "establish general superiority of one method class over another.",
                ],
            )
        )

    validation_path = paths.experiment_dir(experiment_id) / "validation" / "summary.json"
    validation_text = "not available"
    if validation_path.exists():
        summary = json.loads(validation_path.read_text(encoding="utf-8"))
        validation_text = ", ".join(
            f"{name}: {info['status']}" for name, info in sorted(summary["reports"].items())
        )
    sections.append(
        PaperSection(heading="Validation", blocks=[f"Validation results: {validation_text}."])
    )

    provenance = records[0].provenance
    run_record_path = paths.experiment_dir(experiment_id) / "run_record.json"
    environment = "not recorded"
    if run_record_path.exists():
        run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
        environment = (
            f"OR-Tools {run_record.get('ortools_version', 'unknown')}, "
            f"Python {run_record.get('python_version', 'unknown')}."
        )
    sections.append(
        PaperSection(
            heading="Reproducibility",
            blocks=[
                f"Result artifact: `{provenance.get('result_path', '')}` with sha256 "
                f"`{provenance.get('result_sha256', '')}`.",
                f"Spec sha256: `{provenance.get('spec_sha256', '')}`. {environment}",
            ],
        )
    )
    sections.append(
        PaperSection(
            heading="Limitations",
            blocks=[
                "This slice supports linear objectives and linear constraints only; "
                "nonlinear and multi-objective formulations are not implemented.",
                "The greedy baseline is a simple feasible heuristic and may be weak; "
                "it is not proof that the solver solution is practically superior.",
                "Results are conditional on the solver time limit and version.",
            ],
        )
    )

    return PaperAST(
        title=(
            f"{problem.title} — Optimization Report"
            if problem is not None
            else f"{model.name} — Optimization Report"
        ),
        abstract=abstract,
        sections=sections,
        tables=[table],
        figures=[],
        claims=[
            claim
            for claim in load_claims(paths)
            if claim.status == "supported"
            and all(evidence_id in index.records for evidence_id in claim.evidence_ids)
        ],
        references=["Verified optimization artifacts in this repository."],
    )
