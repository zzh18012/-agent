import math
import secrets

from mmos.registries import ModelRegistry
from mmos.schemas.experiment import ExperimentResult
from mmos.schemas.tournament import TournamentEntry, TournamentResult

_COMPARABILITY_FIELDS = (
    "dataset_hash",
    "task_type",
    "feature_set",
    "split_strategy",
    "validation_strategy",
    "random_seed",
    "n_splits",
)


class Tournament:
    """Ranks models from comparable, completed experiments.

    It never scores models on language; every number here was produced by a run. A
    candidate must beat the best registered baseline on the primary metric.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def run(
        self,
        results: list[ExperimentResult],
        *,
        primary_metric: str,
        higher_is_better: bool,
    ) -> TournamentResult:
        if not results:
            raise ValueError("a tournament needs at least one experiment result")
        self._check_comparable(results, primary_metric)

        entries: list[TournamentEntry] = []
        seen_models: set[str] = set()
        for result in results:
            for model_id, metrics in result.models.items():
                if model_id in seen_models:
                    raise ValueError(f"model {model_id!r} appears in more than one experiment")
                seen_models.add(model_id)
                for value in metrics.values():
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite metric for model {model_id!r}")
                entries.append(
                    TournamentEntry(
                        model_id=model_id,
                        experiment_id=result.experiment_id,
                        metrics=metrics,
                        rank=1,
                        status="finalist",
                        reason="",
                    )
                )

        entries.sort(
            key=lambda entry: (
                -entry.metrics[primary_metric]
                if higher_is_better
                else entry.metrics[primary_metric],
                entry.model_id,
            )
        )
        ranked_entries = [
            entry.model_copy(update={"rank": index}) for index, entry in enumerate(entries, start=1)
        ]

        baseline_ids = {
            entry.model_id
            for entry in ranked_entries
            if self.registry.get(entry.model_id).is_baseline
        }
        notes = [
            "selection is based on executed metrics; robustness and sensitivity "
            "analyses are not part of this comparison"
        ]

        if not baseline_ids:
            blocked_entries = [
                entry.model_copy(update={"reason": "blocked: no registered baseline in comparison"})
                for entry in ranked_entries
            ]
            return TournamentResult(
                tournament_id=_new_tournament_id(),
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
                dataset_hash=results[0].dataset_hash,
                entries=blocked_entries,
                blocked=True,
                notes=[*notes, "no registered baseline present"],
            )

        best_baseline = next(entry for entry in ranked_entries if entry.model_id in baseline_ids)

        def beats_baseline(entry: TournamentEntry) -> bool:
            candidate = entry.metrics[primary_metric]
            baseline = best_baseline.metrics[primary_metric]
            return candidate > baseline if higher_is_better else candidate < baseline

        candidates = [entry for entry in ranked_entries if entry.model_id not in baseline_ids]
        winners = [entry for entry in candidates if beats_baseline(entry)]

        if not winners:
            final_entries = [
                entry.model_copy(
                    update={
                        "status": "finalist"
                        if entry.model_id == best_baseline.model_id
                        else "eliminated",
                        "reason": (
                            "best baseline; no candidate beat it"
                            if entry.model_id == best_baseline.model_id
                            else "did not beat the best baseline"
                        ),
                    }
                )
                for entry in ranked_entries
            ]
            return TournamentResult(
                tournament_id=_new_tournament_id(),
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
                dataset_hash=results[0].dataset_hash,
                entries=final_entries,
                selected_model=None,
                challenger_model=best_baseline.model_id,
                blocked=True,
                notes=[*notes, "baseline wins: no candidate beat it"],
            )

        selected = winners[0]
        winner_ids = {winner.model_id for winner in winners}
        final_entries = []
        for entry in ranked_entries:
            if entry.model_id == selected.model_id:
                final_entries.append(
                    entry.model_copy(update={"status": "selected", "reason": "best candidate"})
                )
            elif entry.model_id == best_baseline.model_id:
                final_entries.append(
                    entry.model_copy(update={"status": "finalist", "reason": "baseline challenger"})
                )
            elif entry.model_id in winner_ids:
                final_entries.append(
                    entry.model_copy(
                        update={"status": "finalist", "reason": "beat baseline, ranked lower"}
                    )
                )
            else:
                final_entries.append(
                    entry.model_copy(
                        update={"status": "eliminated", "reason": "did not beat the best baseline"}
                    )
                )

        return TournamentResult(
            tournament_id=_new_tournament_id(),
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            dataset_hash=results[0].dataset_hash,
            entries=final_entries,
            selected_model=selected.model_id,
            challenger_model=best_baseline.model_id,
            blocked=False,
            notes=notes,
        )

    def _check_comparable(self, results: list[ExperimentResult], primary_metric: str) -> None:
        reference = results[0]
        metric_names = self._metric_names(reference)
        if primary_metric not in metric_names:
            raise ValueError(
                f"primary metric {primary_metric!r} is not present; available: "
                f"{', '.join(sorted(metric_names))}"
            )
        for result in results[1:]:
            for field in _COMPARABILITY_FIELDS:
                if getattr(result, field) != getattr(reference, field):
                    raise ValueError(
                        f"experiments are not comparable: {field} differs between "
                        f"{reference.experiment_id!r} and {result.experiment_id!r}"
                    )
            if self._metric_names(result) != metric_names:
                raise ValueError(
                    f"experiments are not comparable: metric sets differ between "
                    f"{reference.experiment_id!r} and {result.experiment_id!r}"
                )

    @staticmethod
    def _metric_names(result: ExperimentResult) -> set[str]:
        names = {frozenset(metrics) for metrics in result.models.values()}
        if len(names) != 1:
            raise ValueError(
                f"models in experiment {result.experiment_id!r} report different metric sets"
            )
        return set(next(iter(names)))


def _new_tournament_id() -> str:
    return f"tournament-{secrets.token_hex(4)}"
