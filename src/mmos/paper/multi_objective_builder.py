
from mmos.core.project import (
    ProjectPaths,
    load_multi_objective_result,
    load_multi_objective_spec,
    load_pareto_oracle,
)
from mmos.schemas.paper import PaperAST, PaperFigure, PaperSection, PaperTable


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _decision_label(decision: dict[str, int]) -> str:
    return ", ".join(
        f"{key}=1" for key, value in sorted(decision.items()) if value == 1
    ) or "none"


def _decision_metrics(
    spec, decision: dict[str, int]
) -> tuple[float, float, float]:
    cost = sum(
        item.cost * decision[item.project_id] for item in spec.items
    )
    f1 = sum(
        item.benefits[spec.primary_objective_id] * decision[item.project_id]
        for item in spec.items
    )
    f2 = sum(
        item.benefits[spec.epsilon_objective_id] * decision[item.project_id]
        for item in spec.items
    )
    return float(cost), float(f1), float(f2)


def _plot_front(
    paths: ProjectPaths,
    result,
    oracle,
    primary: str,
    secondary: str,
) -> PaperFigure:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper_dir = paths.root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    target = paper_dir / "pareto_front.png"

    oracle_points = sorted(
        [(vector.f1, vector.f2) for vector in oracle.pareto_objective_vectors]
    )
    candidate_points = sorted(
        [(vector.f1, vector.f2) for vector in result.candidate_pareto_front]
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    if oracle_points:
        axis.scatter(
            [point[0] for point in oracle_points],
            [point[1] for point in oracle_points],
            marker="o",
            s=60,
            label="Oracle Pareto points",
            facecolors="none",
            edgecolors="#1f77b4",
        )
    if candidate_points:
        axis.scatter(
            [point[0] for point in candidate_points],
            [point[1] for point in candidate_points],
            marker="x",
            s=50,
            label="Candidate Pareto points",
            color="#d62728",
        )
    if oracle_points:
        axis.annotate(
            f"{primary}-max endpoint",
            oracle_points[-1],
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
        axis.annotate(
            f"{secondary}-max endpoint",
            oracle_points[0],
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=8,
        )
    for vector in oracle.pareto_objective_vectors:
        decisions = [
            solution
            for solution in oracle.pareto_decision_vectors
            if abs(solution.f1 - vector.f1) <= 1e-9
            and abs(solution.f2 - vector.f2) <= 1e-9
        ]
        if len(decisions) > 1:
            axis.annotate(
                f"tie: {len(decisions)} oracle decisions, 1 returned",
                (vector.f1, vector.f2),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#555555",
            )
            break
    axis.set_xlabel(primary)
    axis.set_ylabel(secondary)
    axis.set_title("Bi-objective Pareto alternatives (no unique winner)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return PaperFigure(
        figure_id="pareto-front",
        caption=(
            "Oracle and candidate Pareto objective vectors. Points are trade-off "
            "alternatives; no point is marked as a winner. Only the f1-max and f2-max "
            "endpoints and an objective-vector tie are annotated. Coincident candidate and "
            "oracle vectors are rendered as overlapping markers with a combined legend; "
            "the tie annotation reports oracle decision multiplicity, not candidate "
            "multiplicity. Do not infer frontier geometry from marker spacing."
        ),
        path=target.relative_to(paths.root).as_posix(),
        evidence_ids=[
            f"ev-{result.experiment_id}-multiobjective-candidate-front",
            f"ev-{result.experiment_id}-multiobjective-oracle-front",
        ],
    )


def build_multi_objective_paper(
    paths: ProjectPaths, experiment_id: str
) -> PaperAST:
    spec = load_multi_objective_spec(paths, experiment_id)
    result = load_multi_objective_result(paths, experiment_id)
    oracle = load_pareto_oracle(paths, experiment_id)
    comparison = result.comparison
    primary = spec.primary_objective_id
    secondary = spec.epsilon_objective_id

    project_rows = []
    for item in spec.items:
        project_rows.append(
            [
                item.project_id,
                _fmt(item.cost),
                _fmt(item.benefits[primary]),
                _fmt(item.benefits[secondary]),
            ]
        )
    project_table = PaperTable(
        table_id=f"tab-{experiment_id}-projects",
        caption=(
            f"Candidate projects with cost and both maximization benefits. Budget = "
            f"{spec.budget:g}."
        ),
        columns=["Project", "Cost", f"Benefit {primary}", f"Benefit {secondary}"],
        rows=project_rows,
        evidence_ids=[f"ev-{experiment_id}-multiobjective-instance"],
    )

    candidate_decisions = result.candidate_pareto_set
    oracle_decisions = oracle.pareto_decision_vectors
    candidate_rows = []
    for vector in result.candidate_pareto_front:
        decisions = [
            solution
            for solution in candidate_decisions
            if abs(solution.f1 - vector.f1) <= 1e-9
            and abs(solution.f2 - vector.f2) <= 1e-9
        ]
        candidate_rows.append(
            [
                _fmt(vector.f1),
                _fmt(vector.f2),
                str(len(decisions)),
                _decision_label(decisions[0].decision_vector) if decisions else "-",
            ]
        )
    candidate_table = PaperTable(
        table_id=f"tab-{experiment_id}-candidate-front",
        caption=(
            "Candidate Pareto objective vectors from the epsilon-constraint sweep. "
            "The candidate decision count is structurally one per vector because the "
            "lexicographic stage-2 solve returns a single tie-broken decision; it is not "
            "evidence about decision-space multiplicity."
        ),
        columns=[
            f"Objective {primary}",
            f"Objective {secondary}",
            "Decisions returned (single tiebreak)",
            "Example decision",
        ],
        rows=candidate_rows,
        evidence_ids=[f"ev-{experiment_id}-multiobjective-candidate-front"],
    )

    oracle_rows = []
    for vector in oracle.pareto_objective_vectors:
        decisions = [
            solution
            for solution in oracle_decisions
            if abs(solution.f1 - vector.f1) <= 1e-9
            and abs(solution.f2 - vector.f2) <= 1e-9
        ]
        oracle_rows.append(
            [
                _fmt(vector.f1),
                _fmt(vector.f2),
                str(len(decisions)),
                _decision_label(decisions[0].decision_vector) if decisions else "-",
            ]
        )
    oracle_table = PaperTable(
        table_id=f"tab-{experiment_id}-oracle-front",
        caption=(
            "Oracle Pareto objective vectors from exhaustive enumeration. Oracle decision "
            "count shows that one objective vector may correspond to multiple decisions."
        ),
        columns=[
            f"Objective {primary}",
            f"Objective {secondary}",
            "Oracle decision count",
            "Example decision",
        ],
        rows=oracle_rows,
        evidence_ids=[f"ev-{experiment_id}-multiobjective-oracle-front"],
    )

    candidate_decision_keys = {
        tuple(sorted(solution.decision_vector.items()))
        for solution in candidate_decisions
    }
    oracle_decision_keys = {
        tuple(sorted(solution.decision_vector.items()))
        for solution in oracle_decisions
    }
    decision_intersection = candidate_decision_keys & oracle_decision_keys
    decision_recall = (
        len(decision_intersection) / len(oracle_decision_keys)
        if oracle_decision_keys
        else 1.0
    )
    comparison_rows = [
        ["Candidate front size", str(result.candidate_front_size)],
        ["Oracle front size", str(oracle.front_size)],
        ["Front precision", _fmt(comparison.precision if comparison else 0.0)],
        ["Front recall", _fmt(comparison.recall if comparison else 0.0)],
        ["Exact objective-vector match", str(bool(comparison and comparison.exact_match))],
        ["Candidate Pareto set size", str(result.candidate_set_size)],
        ["Oracle Pareto set size", str(oracle.set_size)],
        [
            "Decision-space recall (candidate/oracle)",
            f"{len(decision_intersection)}/{len(oracle_decision_keys)} = "
            f"{decision_recall:.3f}",
        ],
        ["Search space", str(oracle.search_space_size)],
        ["Feasible solutions", str(oracle.feasible_solution_count)],
        ["Reference point", str(spec.reference_point)],
        ["Candidate hypervolume", _fmt(result.hypervolume)],
        ["Oracle hypervolume", _fmt(oracle.hypervolume)],
        ["Hypervolume gap", _fmt(result.hypervolume_gap or 0.0)],
        ["Solver calls", str(result.solver_calls)],
    ]
    comparison_table = PaperTable(
        table_id=f"tab-{experiment_id}-front-comparison",
        caption=(
            "Candidate/oracle comparison. Exact objective-vector set equality is the "
            "primary correctness criterion; hypervolume is secondary and not used to select "
            "a solution."
        ),
        columns=["Check", "Value"],
        rows=comparison_rows,
        evidence_ids=[
            f"ev-{experiment_id}-multiobjective-candidate-front",
            f"ev-{experiment_id}-multiobjective-oracle-front",
            f"ev-{experiment_id}-multiobjective-precision",
            f"ev-{experiment_id}-multiobjective-recall",
        ],
    )

    sensitivity_rows = [
        [
            "1 (primary)",
            str(result.epsilon_coverage.epsilon_count),
            str(result.epsilon_coverage.successful_solves),
            str(result.candidate_front_size),
            _fmt(comparison.precision if comparison else 0.0),
            _fmt(comparison.recall if comparison else 0.0),
            str(bool(comparison and comparison.exact_match)),
        ]
    ]
    for item in result.epsilon_sensitivity:
        sensitivity_rows.append(
            [
                str(item.step),
                str(item.levels),
                str(item.feasible_levels),
                str(item.candidate_front_size),
                _fmt(item.precision),
                _fmt(item.recall),
                str(item.exact_match),
            ]
        )
    sensitivity_table = PaperTable(
        table_id=f"tab-{experiment_id}-epsilon-sensitivity",
        caption=(
            "Epsilon-grid sensitivity. Step 1 is the primary pre-registered grid; steps "
            "2 and 5 are secondary coarser grids. Recall is conditional on grid resolution "
            "and is not evidence of general method superiority."
        ),
        columns=[
            "Epsilon step",
            "Levels",
            "Feasible levels",
            "Candidate front size",
            "Precision",
            "Recall",
            "Exact match",
        ],
        rows=sensitivity_rows,
        evidence_ids=[f"ev-{experiment_id}-multiobjective-epsilon-policy"],
    )

    oracle_ties = []
    for vector in oracle.pareto_objective_vectors:
        decisions = [
            solution
            for solution in oracle_decisions
            if abs(solution.f1 - vector.f1) <= 1e-9
            and abs(solution.f2 - vector.f2) <= 1e-9
        ]
        if len(decisions) > 1:
            details = []
            for decision in decisions:
                cost, f1, f2 = _decision_metrics(spec, decision.decision_vector)
                details.append(
                    f"[{_decision_label(decision.decision_vector)}] cost {cost:g}, "
                    f"f1 {f1:g}, f2 {f2:g}"
                )
            oracle_ties.append(
                (vector.f1, vector.f2, len(decisions), "; ".join(details))
            )
    tie_text = (
        "Objective-vector ties: "
        + "; ".join(
            f"({_fmt(f1)}, {_fmt(f2)}) has {count} oracle decisions: {details}; the "
            "candidate returns one tie-broken decision for that vector."
            for f1, f2, count, details in oracle_ties
        )
        if oracle_ties
        else "No objective-vector ties were found in the oracle Pareto set."
    )

    sensitivity_miss_lines = []
    for item in result.epsilon_sensitivity:
        for vector in item.missing_vectors:
            on_grid = int(vector.f2) % item.step == 0
            cause = (
                "stage-2/lexicographic cause"
                if on_grid
                else "f2 is not on the coarse grid"
            )
            sensitivity_miss_lines.append(
                f"step {item.step}: missed ({_fmt(vector.f1)}, {_fmt(vector.f2)}) - {cause}"
            )
    sensitivity_miss_text = (
        "Coarse-grid misses: " + "; ".join(sensitivity_miss_lines) + "."
        if sensitivity_miss_lines
        else "No coarse-grid misses were observed."
    )

    figure = _plot_front(
        paths,
        result,
        oracle,
        primary,
        secondary,
    )

    secondary_sum = sum(item.benefits[secondary] for item in spec.items)
    min_front_primary = min(
        (vector.f1 for vector in oracle.pareto_objective_vectors), default=0.0
    )
    min_front_secondary = min(
        (vector.f2 for vector in oracle.pareto_objective_vectors), default=0.0
    )
    abstract = (
        f"Bi-objective binary project-selection instance `{spec.problem_id}` with "
        f"{len(spec.items)} projects and budget {spec.budget:g}. Both objectives are "
        "maximized; no preference or weighted score is supplied. The epsilon-constraint "
        f"method used step-1 levels from 0 to {result.epsilon_coverage.epsilon_max:g} "
        f"({result.epsilon_coverage.epsilon_count} levels). Feasible levels use two CBC "
        f"solves and infeasible levels one: {result.epsilon_coverage.successful_solves}*2 + "
        f"{result.epsilon_coverage.infeasible_levels}*1 = {result.solver_calls} calls. "
        f"The candidate Pareto front has {result.candidate_front_size} objective vectors and "
        f"{result.candidate_set_size} decisions; the exhaustive oracle has {oracle.front_size} "
        f"vectors and {oracle.set_size} decisions. At step 1, front precision "
        f"{_fmt(comparison.precision if comparison else 0.0)}, recall "
        f"{_fmt(comparison.recall if comparison else 0.0)}, exact objective-vector match "
        f"{bool(comparison and comparison.exact_match)}. Decision-space recall is "
        f"{len(decision_intersection)}/{len(oracle_decision_keys)} = {decision_recall:.3f}; "
        "decision-space recall 1.0 is not claimed. The schedule is a loose a-priori bound: "
        f"the secondary benefit sum over all projects is {secondary_sum:g}, "
        "so levels 0 through that sum are run without reading the oracle front; no "
        "pre-registration timestamp is available. With a grid covering every distinct "
        "secondary value, exact objective-vector recovery is expected by construction; it "
        "is a tautological coverage/implementation check, not evidence that "
        "epsilon-constraint is generally superior. Coarser steps 2 and 5 are reported "
        "separately and miss vectors. Hypervolume is secondary. The result is a set of "
        "Pareto-optimal trade-off alternatives; there is no unique recommended solution."
    )

    sections = [
        PaperSection(
            heading="Decision model and conflicting objectives",
            blocks=[
                f"Binary decisions x_i in {{0,1}} select projects. Cost constraint: total "
                f"cost <= {spec.budget:g}. Objective {primary}: maximized total benefit "
                f"{primary}; objective {secondary}: maximized total benefit {secondary}.",
                "The instance is constructed so budget binds and the two benefits trade "
                "off: no single portfolio simultaneously maximizes both objectives.",
            ],
        ),
        PaperSection(
            heading="Dominance and Pareto semantics",
            blocks=[
                "For two maximization objectives, A dominates B iff A.f1 >= B.f1 and "
                "A.f2 >= B.f2 with at least one strict inequality. Equal objective vectors "
                "do not dominate each other. A feasible vector is Pareto-optimal iff no "
                "other feasible vector dominates it.",
                "The objective-space Pareto front is a set of distinct (f1, f2) vectors. "
                "The decision-space Pareto set may contain multiple distinct decisions with "
                "the same objective vector; front size and set size are reported separately.",
            ],
        ),
        PaperSection(
            heading="Epsilon-constraint method",
            blocks=[
                f"The primary objective {primary} is maximized subject to a secondary "
                f"threshold {secondary} >= epsilon. The epsilon schedule is generated only "
                "from the frozen instance's integer benefit range and the declared policy; "
                "it does not read the oracle front.",
                "Each epsilon solve is lexicographic: stage 1 maximizes the primary "
                "objective; stage 2 fixes that value exactly and maximizes the secondary "
                "objective. No weighted sum or Big-M surrogate is used.",
                f"Schedule: step {spec.epsilon_step}, epsilon "
                f"{result.epsilon_coverage.epsilon_min:g} to "
                f"{result.epsilon_coverage.epsilon_max:g}, "
                f"{result.epsilon_coverage.epsilon_count} levels. Coverage: "
                f"{result.epsilon_coverage.successful_solves} feasible, "
                f"{result.epsilon_coverage.infeasible_levels} infeasible, "
                f"{result.epsilon_coverage.error_levels} errors. The infeasible levels are "
                "high secondary thresholds that no budget-feasible portfolio can meet; the "
                "empty portfolio has secondary value 0 and only satisfies epsilon = 0.",
                f"Schedule derivation: the upper bound is the sum of all project "
                f"{secondary} benefits = {secondary_sum:g}; it is a loose "
                "a-priori bound, not an oracle-derived front value. No independent "
                "pre-registration timestamp exists; the frozen spec hash is the only "
                "provenance.",
                f"Call accounting: feasible levels use two stage solves, infeasible levels "
                f"one: {result.epsilon_coverage.successful_solves}*2 + "
                f"{result.epsilon_coverage.infeasible_levels}*1 = {result.solver_calls} CBC "
                "calls. This is arithmetic from the level split, not an independent "
                "measurement. An error is a stage status other than OPTIMAL or INFEASIBLE; "
                f"this run has {result.epsilon_coverage.error_levels} errors. Total "
                f"wall-clock runtime is {result.runtime_total:.3f} s. Per-level statuses "
                "are in epsilon_solutions.csv.",
            ],
        ),
        PaperSection(
            heading="Exhaustive oracle and shared-spec validation",
            blocks=[
                f"The oracle enumerates all {oracle.search_space_size} binary subsets "
                f"from the frozen spec, keeps {oracle.feasible_solution_count} budget-feasible "
                "solutions, and applies an independently written dominance rule. It never "
                "calls CBC, the epsilon runner, or the candidate front.",
                "The oracle and candidate share the frozen spec, so this is independent "
                "implementation-path validation, not independent input validation. A shared "
                "spec error would affect both; no perturbed-input negative control was run.",
                "Validation recomputes binary feasibility, budget, and both objectives for "
                "every candidate decision; it compares objective-vector sets exactly. A "
                "missing oracle point or a candidate point dominated by an oracle feasible "
                "point is blocking.",
            ],
        ),
        PaperSection(
            heading="Results: Pareto alternatives, no winner",
            blocks=[
                f"Candidate front has {result.candidate_front_size} vectors; oracle front "
                f"has {oracle.front_size}. Step-1 precision and recall are "
                f"{_fmt(comparison.precision if comparison else 0.0)} and "
                f"{_fmt(comparison.recall if comparison else 0.0)}; exact objective-vector "
                f"match is {bool(comparison and comparison.exact_match)}. Because the step-1 "
                "grid covers every distinct secondary value on this integer instance, "
                "exact recovery is expected by construction; this is an implementation and "
                "grid-coverage check, not a demonstration that epsilon-constraint is "
                "generally superior.",
                tie_text,
                "Set-level equality is on objective vectors only. Candidate decision counts "
                "are structurally one per vector because of the lexicographic tie-break; "
                "decision-space recall is reported separately and is not claimed to be 1.0.",
                sensitivity_miss_text,
                "This integer, antagonistic, binding-budget instance is the most favourable "
                "regime for epsilon-constraint; non-binding budgets, non-integer objectives, "
                "larger n, and three or more objectives are untested.",
                "All reported points are alternatives. No point is labeled a winner, a best "
                "compromise, or a recommended portfolio. The f1-max and f2-max endpoints are "
                "annotated only to orient the trade-off.",
            ],
        ),
        PaperSection(
            heading="Hypervolume and reference point",
            blocks=[
                f"Reference point {spec.reference_point} was fixed in the frozen spec before "
                "the experiment; both objectives are nonnegative, so the origin lies below "
                "the Pareto front. Two-dimensional hypervolume is reported as a secondary "
                "descriptive metric; it is not used to select a solution. The minimum Pareto "
                f"{primary} and {secondary} values are {min_front_primary:g} and "
                f"{min_front_secondary:g}.",
                "Equal hypervolume would not by itself prove equal fronts; correctness rests "
                "on exact objective-vector set comparison. No alternative-reference "
                "sensitivity is reported.",
            ],
        ),
        PaperSection(
            heading="Limitations and reproducibility",
            blocks=[
                "One small frozen synthetic integer instance in the most favourable "
                "antagonistic/binding-budget regime; two maximize objectives; binary "
                "selection with one budget constraint. No 3+ objectives, stochastic "
                "metaheuristics, preference elicitation, knee/compromise recommendation, or "
                "algorithm performance comparison. Non-binding budgets, non-integer ranges, "
                "and larger n are untested.",
                "The result is an internal implementation-path and grid-coverage check. "
                "Oracle/runner code, epsilon_solutions.csv, and front lists are not published "
                "with this report; a third party cannot re-execute or verify the exact-match "
                "claim. The claim is therefore scoped to internal verification.",
                "The oracle and CBC share the same frozen spec, so independence is "
                "implementation-path independence, not external third-party validation.",
                f"Instance hash `{spec.instance_hash}`; spec hash `{spec.spec_hash}`; "
                f"method `{result.method}`; solver `{result.solver}` "
                f"{result.solver_version}.",
                "Evidence is stored in the project's private verified store.",
            ],
        ),
    ]

    return PaperAST(
        title="Bi-objective Pareto-front Conformance Report — Frozen Project Portfolio",
        abstract=abstract,
        sections=sections,
        tables=[
            project_table,
            candidate_table,
            oracle_table,
            comparison_table,
            sensitivity_table,
        ],
        figures=[figure],
        claims=[],
        references=[
            "Multi-objective optimization: Pareto dominance and epsilon-constraint method.",
            "Frozen bi-objective project-selection instance and verified evidence.",
        ],
    )


__all__ = ["build_multi_objective_paper"]
