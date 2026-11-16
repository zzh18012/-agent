from mmos.core.project import ProjectPaths, atomic_write_text
from mmos.schemas.mixed import DAGRunRecord, DAGStatus, NodeStatus
from mmos.schemas.states import TaskFamily


def apply_stale_propagation(
    paths: ProjectPaths,
    plan_id: str,
    *,
    forecast_hash: str | None = None,
    forecast_error_hash: str | None = None,
    production_plan_hash: str | None = None,
) -> list[str]:
    """Mark prior runs stale according to artifact lineage.

    A changed point forecast invalidates optimization and simulation. A changed
    forecast-error artifact or production plan invalidates only simulation; the
    optimization node remains succeeded.
    """
    runs_dir = paths.mixed_dir / "runs"
    if not runs_dir.exists():
        return []
    updated: list[str] = []
    for path in sorted(runs_dir.glob("*/dag_run.json")):
        record = DAGRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.plan_id != plan_id:
            continue
        if record.status not in {DAGStatus.SUCCEEDED, DAGStatus.STALE}:
            continue

        full_stale = bool(
            forecast_hash
            and record.forecast_artifact_hash
            and record.forecast_artifact_hash != forecast_hash
        )
        simulation_only = False
        reasons: list[str] = []
        if full_stale:
            reasons.append(
                "upstream forecast artifact hash changed; optimization and simulation "
                "are stale"
            )
        else:
            if (
                forecast_error_hash
                and record.forecast_error_artifact_hash
                and record.forecast_error_artifact_hash != forecast_error_hash
            ):
                simulation_only = True
                reasons.append(
                    "forecast-error artifact hash changed; simulation is stale; "
                    "optimization remains valid"
                )
            if (
                production_plan_hash
                and record.production_plan_artifact_hash
                and record.production_plan_artifact_hash != production_plan_hash
            ):
                simulation_only = True
                reasons.append(
                    "production plan artifact hash changed; simulation is stale; "
                    "optimization remains valid"
                )
        if not full_stale and not simulation_only:
            continue
        simulation_nodes = [
            node for node in record.nodes if node.task_family is TaskFamily.SIMULATION
        ]
        if simulation_only and not simulation_nodes:
            continue

        stale_families = (
            {TaskFamily.OPTIMIZATION, TaskFamily.SIMULATION}
            if full_stale
            else {TaskFamily.SIMULATION}
        )
        nodes = [
            node.model_copy(
                update={
                    "execution_status": NodeStatus.STALE,
                    "status_reason": reasons[0],
                }
            )
            if node.task_family in stale_families
            else node
            for node in record.nodes
        ]
        notes = list(record.notes)
        for reason in reasons:
            if reason not in notes:
                notes.append(reason)
        stale = record.model_copy(
            update={"status": DAGStatus.STALE, "nodes": nodes, "notes": notes}
        )
        atomic_write_text(path, stale.model_dump_json(indent=2) + "\n")
        updated.append(record.run_id)
    return updated


__all__ = ["apply_stale_propagation"]
