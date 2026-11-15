"""Red-Team Orchestrator: coordinates experiments over P3.1/P3.2 models.

In-memory coordination only: lifecycle transitions delegate to
`ExperimentState`/`AttackState`, selection delegates to `AttackDataset`
(deterministic: cases sorted by attack_id, filtered by the experiment's
`attack_types`). No execution, no model calls — the future Attack
Executor consumes `pending_attacks` and reports back through
`begin_attack` / `complete_attack` / `fail_attack` / `cancel_attack`.
"""

from __future__ import annotations

from contextlib import suppress

from .dataset import AttackCase, AttackDataset
from .state import (
    AttackState,
    ExperimentConfig,
    ExperimentState,
    InvalidTransitionError,
    LifecycleStatus,
)


class RedTeamOrchestrator:
    def __init__(self, dataset: AttackDataset, /) -> None:
        self._dataset = dataset
        self._experiments: dict[str, ExperimentState] = {}
        self._attacks: dict[str, dict[str, AttackState]] = {}

    # --- Experiment lifecycle ---------------------------------------------

    def create_experiment(self, config: ExperimentConfig, /) -> ExperimentState:
        if not isinstance(config, ExperimentConfig):
            raise ValueError("create_experiment requires an ExperimentConfig")
        if config.experiment_id in self._experiments:
            raise ValueError(f"experiment already registered: {config.experiment_id}")
        state = ExperimentState(config=config)
        self._experiments[config.experiment_id] = state
        return state

    def start_experiment(self, experiment_id: str, /) -> ExperimentState:
        selected = self.select_attacks(experiment_id)
        if not selected:
            raise ValueError(
                f"experiment {experiment_id}: no attacks selected "
                "(empty dataset or attack_types matched nothing)"
            )
        experiment = self._experiment(experiment_id)
        started = experiment.transition(LifecycleStatus.RUNNING)
        self._experiments[experiment_id] = started
        self._attacks[experiment_id] = {
            case.attack_id: AttackState(
                experiment_id=experiment_id,
                attack_type=case.category,
                prompt=case.prompt,
                attack_id=case.attack_id,
                metadata=dict(case.metadata),
            )
            for case in selected
        }
        return started

    def complete_experiment(self, experiment_id: str, /) -> ExperimentState:
        experiment = self._experiment(experiment_id)
        pending = [
            attack
            for attack in self._attacks.get(experiment_id, {}).values()
            if not attack.status.terminal
        ]
        if pending:
            raise ValueError(
                f"experiment {experiment_id}: "
                f"{len(pending)} attack(s) still non-terminal"
            )
        return self._update(experiment, LifecycleStatus.COMPLETED)

    def fail_experiment(self, experiment_id: str, error: str, /) -> ExperimentState:
        return self._update(
            self._experiment(experiment_id), LifecycleStatus.FAILED, error=error
        )

    def cancel_experiment(self, experiment_id: str, /) -> ExperimentState:
        experiment = self._update(
            self._experiment(experiment_id), LifecycleStatus.CANCELLED
        )
        for attack_id in list(self._attacks.get(experiment_id, {})):
            # Already-terminal attacks keep their state; context preserved.
            with suppress(InvalidTransitionError):
                self.cancel_attack(experiment_id, attack_id)
        return experiment

    # --- Attack selection / executor interface ------------------------------

    def select_attacks(self, experiment_id: str, /) -> tuple[AttackCase, ...]:
        """Deterministic dataset-side selection (sorted by attack_id)."""
        config = self._experiment(experiment_id).config
        if not config.attack_types:
            return self._dataset.list_cases()
        selected: list[AttackCase] = []
        seen: set[str] = set()
        for attack_type in config.attack_types:
            for case in self._dataset.filter(category=attack_type):
                if case.attack_id not in seen:
                    seen.add(case.attack_id)
                    selected.append(case)
        return tuple(selected)

    def attacks(self, experiment_id: str, /) -> tuple[AttackState, ...]:
        tracked = self._attacks.get(experiment_id, {})
        return tuple(
            tracked[case.attack_id] for case in self.select_attacks(experiment_id)
        )

    def pending_attacks(self, experiment_id: str, /) -> tuple[AttackState, ...]:
        return tuple(
            attack
            for attack in self.attacks(experiment_id)
            if not attack.status.terminal
        )

    def begin_attack(self, experiment_id: str, attack_id: str, /) -> AttackState:
        return self._transition_attack(
            experiment_id, attack_id, LifecycleStatus.RUNNING
        )

    def complete_attack(self, experiment_id: str, attack_id: str, /) -> AttackState:
        return self._transition_attack(
            experiment_id, attack_id, LifecycleStatus.COMPLETED
        )

    def fail_attack(
        self, experiment_id: str, attack_id: str, error: str, /
    ) -> AttackState:
        return self._transition_attack(
            experiment_id, attack_id, LifecycleStatus.FAILED, error=error
        )

    def cancel_attack(self, experiment_id: str, attack_id: str, /) -> AttackState:
        return self._transition_attack(
            experiment_id, attack_id, LifecycleStatus.CANCELLED
        )

    # --- Internals -----------------------------------------------------------

    def _experiment(self, experiment_id: str, /) -> ExperimentState:
        try:
            return self._experiments[experiment_id]
        except KeyError:
            raise KeyError(f"unknown experiment: {experiment_id}") from None

    def _update(
        self,
        experiment: ExperimentState,
        target: LifecycleStatus,
        *,
        error: str | None = None,
    ) -> ExperimentState:
        updated = experiment.transition(target, error=error)
        self._experiments[updated.config.experiment_id] = updated
        return updated

    def _transition_attack(
        self,
        experiment_id: str,
        attack_id: str,
        target: LifecycleStatus,
        *,
        error: str | None = None,
    ) -> AttackState:
        tracked = self._attacks.get(experiment_id, {})
        if attack_id not in tracked:
            raise KeyError(
                f"untracked attack {attack_id} in experiment {experiment_id}"
            )
        updated = tracked[attack_id].transition(target, error=error)
        tracked[attack_id] = updated
        return updated


__all__ = ["RedTeamOrchestrator"]
