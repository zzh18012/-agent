import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.benchmark import BenchmarkSuiteResult
from mmos.schemas.experiment import ExperimentResult, ExperimentSpec
from mmos.schemas.graph import GraphProblemSpec, GraphResult, MinCutOracle
from mmos.schemas.optimization import (
    OptimizationExperimentRecord,
    OptimizationResult,
    OptimizationSpec,
    relative_improvement,
)
from mmos.schemas.optimization_tournament import OptimizationTournamentResult
from mmos.schemas.simulation import SimulationResult, SimulationSpec
from mmos.schemas.tracking import MLflowRunRecord


def tracking_uri(paths: ProjectPaths) -> str:
    """Each project owns its ledger; no global or remote store is assumed."""
    database = paths.experiments_dir / "mlflow.db"
    return f"sqlite:///{database.as_posix()}"


def _experiment_id(client: MlflowClient, paths: ProjectPaths) -> str:
    name = paths.root.name
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    artifact_root: Path = paths.experiments_dir / "mlflow_artifacts"
    return client.create_experiment(name, artifact_location=artifact_root.as_uri())


def log_experiment_run(
    paths: ProjectPaths, spec: ExperimentSpec, result: ExperimentResult
) -> MLflowRunRecord:
    """Record one formal run. Failures propagate; tracking is not optional."""
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)

    with mlflow.start_run(experiment_id=experiment_id, run_name=spec.experiment_id) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "problem_id": spec.problem_id,
                "proposal_id": spec.proposal_id,
                "dataset_hash": spec.dataset_hash,
            }
        )
        mlflow.log_params(
            {
                "dataset_hash": spec.dataset_hash,
                "random_seed": spec.random_seed,
                "n_splits": spec.n_splits,
                "models": ",".join(spec.models),
                "metrics": ",".join(spec.metrics),
                "target_column": spec.target_column,
                "feature_count": len(spec.feature_set),
            }
        )
        for model_id, metrics in result.models.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_id}.{metric_name}", value)

        experiment_dir = paths.experiments_dir / spec.experiment_id
        mlflow.log_artifacts(str(experiment_dir), artifact_path="experiment")
        run_id = run.info.run_id

    record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        paths.experiments_dir / spec.experiment_id / "mlflow_run.json",
        record.model_dump_json(indent=2) + "\n",
    )
    return record


def log_optimization_run(
    paths: ProjectPaths,
    spec: OptimizationSpec,
    result: OptimizationResult,
    record: OptimizationExperimentRecord,
) -> MLflowRunRecord:
    """Record a formal optimization experiment in the project ledger."""
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)
    baseline = spec.solver == "greedy"

    with mlflow.start_run(experiment_id=experiment_id, run_name=spec.experiment_id) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "experiment_id": spec.experiment_id,
                "task_family": "optimization",
                "model_id": spec.model.name,
                "solver": spec.solver,
                "objective_sense": spec.model.sense,
                "instance_hash": spec.instance_hash or result.instance_hash or "",
                "git_commit": record.git_commit,
                "baseline": str(baseline).lower(),
                "solver_feasible": str(
                    any(solution.feasible for solution in result.solutions)
                ).lower(),
            }
        )
        mlflow.log_params(
            {
                "time_limit_seconds": spec.time_limit_seconds,
                "number_of_variables": len(spec.model.variables),
                "number_of_constraints": len(spec.model.constraints),
                "baseline": spec.baseline,
                "solver_parameters": json.dumps(spec.solver_parameters, sort_keys=True),
                "random_seed": "" if spec.random_seed is None else spec.random_seed,
            }
        )
        solver_solution = next(
            (solution for solution in result.solutions if solution.method != "greedy_feasible"),
            None,
        )
        baseline_solution = next(
            (solution for solution in result.solutions if solution.method == "greedy_feasible"),
            None,
        )
        for label, solution in (("solver", solver_solution), ("baseline", baseline_solution)):
            if solution is None:
                continue
            if solution.objective_value is not None:
                mlflow.log_metric(f"{label}.objective_value", solution.objective_value)
            mlflow.log_metric(f"{label}.runtime_seconds", solution.runtime_seconds)
            if solution.optimality_gap is not None:
                mlflow.log_metric(f"{label}.relative_gap", solution.optimality_gap)
            if solution.absolute_gap is not None:
                mlflow.log_metric(f"{label}.absolute_gap", solution.absolute_gap)
        if (
            solver_solution is not None
            and baseline_solution is not None
            and solver_solution.objective_value is not None
            and baseline_solution.objective_value is not None
        ):
            improvement = relative_improvement(
                spec.model.sense,
                solver_solution.objective_value,
                baseline_solution.objective_value,
            )
            if improvement is not None:
                mlflow.log_metric("baseline_relative_improvement", improvement)

        experiment_dir = paths.experiments_dir / spec.experiment_id
        mlflow.log_artifacts(str(experiment_dir), artifact_path="experiment")
        run_id = run.info.run_id

    mlflow_record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        paths.experiments_dir / spec.experiment_id / "mlflow_run.json",
        mlflow_record.model_dump_json(indent=2) + "\n",
    )
    return mlflow_record


def log_graph_run(
    paths: ProjectPaths,
    spec: GraphProblemSpec,
    result: GraphResult,
    oracle: MinCutOracle,
) -> MLflowRunRecord:
    """One formal MLflow run per graph experiment."""
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)
    gap = abs(result.max_flow_value - oracle.minimum_cut_capacity)

    with mlflow.start_run(experiment_id=experiment_id, run_name=spec.experiment_id) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "experiment_id": spec.experiment_id,
                "graph_id": spec.graph_id,
                "task_family": "graph",
                "problem_type": spec.problem_type,
                "directed": str(spec.directed).lower(),
                "algorithm": result.algorithm,
                "library": result.library,
                "library_version": result.library_version,
                "graph_hash": spec.graph_hash,
                "spec_hash": spec.spec_hash,
                "source": spec.source,
                "sink": spec.sink,
            }
        )
        mlflow.log_params(
            {
                "node_count": len(spec.nodes),
                "edge_count": len(spec.edges),
                "source": spec.source,
                "sink": spec.sink,
                "algorithm": spec.algorithm,
                "capacity_attribute": spec.capacity_attribute,
                "graph_hash": spec.graph_hash,
            }
        )
        mlflow.log_metrics(
            {
                "max_flow_value": result.max_flow_value,
                "min_cut_capacity": oracle.minimum_cut_capacity,
                "flow_cut_gap": gap,
                "runtime_seconds": result.runtime_seconds,
                "min_cut_count": float(oracle.minimum_cut_count),
            }
        )
        experiment_dir = paths.experiments_dir / spec.experiment_id
        mlflow.log_artifacts(str(experiment_dir), artifact_path="experiment")
        run_id = run.info.run_id

    record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        paths.experiments_dir / spec.experiment_id / "mlflow_run.json",
        record.model_dump_json(indent=2) + "\n",
    )
    return record


def log_simulation_run(
    paths: ProjectPaths,
    spec: SimulationSpec,
    result: SimulationResult,
) -> MLflowRunRecord:
    """One formal MLflow run per simulation experiment, not per replication."""
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)

    with mlflow.start_run(experiment_id=experiment_id, run_name=spec.experiment_id) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "experiment_id": spec.experiment_id,
                "simulation_id": spec.simulation_id,
                "task_family": "simulation",
                "model_family": spec.model_family,
                "queue_discipline": spec.queue_discipline,
                "spec_hash": spec.spec_hash,
                "git_commit": result.git_commit,
                "simpy_version": result.runtime.get("simpy_version", ""),
            }
        )
        mlflow.log_params(
            {
                "arrival_rate": spec.arrival_rate,
                "service_rate": spec.service_rate,
                "rho": spec.rho,
                "warmup_time": spec.warmup_time,
                "measurement_time": spec.measurement_time,
                "replications": spec.replications,
                "master_seed": spec.master_seed,
                "seed_derivation": spec.seed_derivation,
                "confidence_level": spec.confidence_level,
                "analytic_relative_tolerance": spec.analytic_relative_tolerance,
                "little_law_relative_tolerance": spec.little_law_relative_tolerance,
                "simulation_time_units": spec.simulation_time_units,
            }
        )
        for metric, summary in result.metrics.items():
            mlflow.log_metrics(
                {
                    f"mean_{metric}": summary.mean,
                    f"std_{metric}": summary.std,
                    f"ci_low_{metric}": summary.ci_low,
                    f"ci_high_{metric}": summary.ci_high,
                    f"ci_width_{metric}": summary.ci_width,
                }
            )
        for metric, comparison in result.analytic_comparison.items():
            mlflow.log_metric(f"relative_error_{metric}", comparison.relative_error)
        for metric, check in result.little_law.items():
            mlflow.log_metric(
                f"little_law_{metric}_relative", check.relative_discrepancy
            )

        experiment_dir = paths.experiments_dir / spec.experiment_id
        mlflow.log_artifacts(str(experiment_dir), artifact_path="experiment")
        run_id = run.info.run_id

    record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        paths.experiments_dir / spec.experiment_id / "mlflow_run.json",
        record.model_dump_json(indent=2) + "\n",
    )
    return record


def log_optimization_tournament(
    paths: ProjectPaths, result: OptimizationTournamentResult
) -> MLflowRunRecord:
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)
    candidate = next(entry for entry in result.entries if entry.entry_role.value == "candidate")
    baseline = next(entry for entry in result.entries if entry.entry_role.value == "baseline")

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"tournament:{result.tournament_id}"
    ) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "task_family": "optimization",
                "tournament_id": result.tournament_id,
                "instance_hash": result.instance_hash,
                "candidate_experiment_id": candidate.experiment_id,
                "baseline_experiment_id": baseline.experiment_id,
                "winner_experiment_id": result.winner_experiment_id or "",
                "comparison_status": result.winner_status.value,
            }
        )
        mlflow.log_param("objective_tolerance", result.objective_tolerance)
        if candidate.objective_value is not None:
            mlflow.log_metric("candidate.objective_value", candidate.objective_value)
        if baseline.objective_value is not None:
            mlflow.log_metric("baseline.objective_value", baseline.objective_value)
        if candidate.objective_value is not None and baseline.objective_value is not None:
            improvement = relative_improvement(
                candidate.objective_sense,
                candidate.objective_value,
                baseline.objective_value,
            )
            if improvement is not None:
                mlflow.log_metric("candidate.relative_improvement", improvement)

        from mmos.experiments.optimization_tournament import optimization_tournament_dir

        mlflow.log_artifacts(
            str(optimization_tournament_dir(paths, result.tournament_id)),
            artifact_path="tournament",
        )
        run_id = run.info.run_id

    mlflow_record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(
        optimization_tournament_dir(paths, result.tournament_id) / "mlflow_run.json",
        mlflow_record.model_dump_json(indent=2) + "\n",
    )
    return mlflow_record


def log_benchmark_suite(
    paths: ProjectPaths, result: BenchmarkSuiteResult, artifact_dir: Path
) -> MLflowRunRecord:
    uri = tracking_uri(paths)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = _experiment_id(client, paths)
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"benchmark:{result.benchmark_run_id}"
    ) as run:
        mlflow.set_tags(
            {
                "project_id": paths.root.name,
                "task_family": "optimization_benchmark",
                "suite_id": result.suite_id,
                "manifest_hash": result.manifest_hash,
                "status": result.status,
                "oracle_agreement": str(result.oracle_agreement).lower(),
                "git_commit": result.git_commit,
            }
        )
        mlflow.log_param("n_instances", result.n_instances)
        mlflow.log_metrics(
            {
                "candidate.feasible_count": result.candidate_feasible_count,
                "candidate.verified_optimum_count": result.candidate_verified_optimum_count,
                "candidate.win_count": result.candidate_win_count,
                "candidate.tie_count": result.candidate_tie_count,
                "candidate.loss_count": result.candidate_loss_count,
                "greedy.hit_rate": result.greedy.hit_rate,
                "enhanced_greedy.hit_rate": result.enhanced_greedy.hit_rate,
            }
        )
        for label, aggregate in (
            ("greedy", result.greedy),
            ("enhanced_greedy", result.enhanced_greedy),
        ):
            for field in (
                "mean_relative_regret",
                "median_relative_regret",
                "max_relative_regret",
                "mean_absolute_regret",
                "max_absolute_regret",
            ):
                value = getattr(aggregate, field)
                if value is not None:
                    mlflow.log_metric(f"{label}.{field}", value)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="benchmark")
        run_id = run.info.run_id

    record = MLflowRunRecord(
        run_id=run_id,
        tracking_uri=uri,
        experiment_name=paths.root.name,
        logged_at=datetime.now(UTC).replace(microsecond=0),
    )
    atomic_write_text(artifact_dir / "mlflow_run.json", record.model_dump_json(indent=2) + chr(10))
    return record
