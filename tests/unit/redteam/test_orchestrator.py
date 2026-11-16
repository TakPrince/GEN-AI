"""P3.3: RedTeamOrchestrator coordination tests."""

from __future__ import annotations

import pytest

from safety_platform.redteam import (
    AttackCase,
    AttackDataset,
    DuplicateAttackError,
    ExperimentConfig,
    ExperimentState,
    InvalidTransitionError,
    LifecycleStatus,
    RedTeamOrchestrator,
)


def _case(attack_id: str, category: str = "prompt_injection") -> AttackCase:
    return AttackCase(
        name=f"case-{attack_id}",
        category=category,
        prompt=f"attack prompt {attack_id}",
        attack_id=attack_id,
    )


def _dataset() -> AttackDataset:
    return AttackDataset(
        (_case("b-2"), _case("a-1"), _case("a-2", category="direct_jailbreak"))
    )


def _orchestrator(**config_overrides) -> tuple[RedTeamOrchestrator, str]:
    kwargs = {"name": "probe"}
    kwargs.update(config_overrides)
    orchestrator = RedTeamOrchestrator(_dataset())
    experiment_id = orchestrator.create_experiment(
        ExperimentConfig(**kwargs)
    ).config.experiment_id
    return orchestrator, experiment_id


# --- Experiment initialization -------------------------------------------


def test_create_experiment_starts_created() -> None:
    orchestrator, experiment_id = _orchestrator()
    assert orchestrator._experiment(experiment_id).status is LifecycleStatus.CREATED
    config = ExperimentConfig(name="dup2")
    orchestrator.create_experiment(config)
    with pytest.raises(ValueError, match="already registered"):
        orchestrator.create_experiment(config)
    with pytest.raises(ValueError):
        orchestrator.create_experiment("nope")


def test_unknown_experiment_rejected() -> None:
    orchestrator = RedTeamOrchestrator(_dataset())
    with pytest.raises(KeyError):
        orchestrator.select_attacks("missing")
    with pytest.raises(KeyError):
        orchestrator.start_experiment("missing")


# --- Attack selection ------------------------------------------------------


def test_selection_deterministic_and_filtered() -> None:
    orchestrator, experiment_id = _orchestrator()
    assert [c.attack_id for c in orchestrator.select_attacks(experiment_id)] == [
        "a-1",
        "a-2",
        "b-2",
    ]
    filtered_id = orchestrator.create_experiment(
        ExperimentConfig(name="filtered", attack_types=("prompt_injection",))
    ).config.experiment_id
    assert [c.attack_id for c in orchestrator.select_attacks(filtered_id)] == [
        "a-1",
        "b-2",
    ]


def test_empty_dataset_rejected_at_start() -> None:
    orchestrator = RedTeamOrchestrator(AttackDataset())
    experiment_id = orchestrator.create_experiment(
        ExperimentConfig(name="empty")
    ).config.experiment_id
    with pytest.raises(ValueError, match="no attacks selected"):
        orchestrator.start_experiment(experiment_id)
    assert orchestrator._experiment(experiment_id).status is LifecycleStatus.CREATED

    no_match = RedTeamOrchestrator(_dataset())
    no_match_id = no_match.create_experiment(
        ExperimentConfig(name="nomatch", attack_types=("policy_evasion",))
    ).config.experiment_id
    with pytest.raises(ValueError, match="no attacks selected"):
        no_match.start_experiment(no_match_id)
    assert no_match._experiment(no_match_id).status is LifecycleStatus.CREATED


def test_start_materializes_tracking() -> None:
    orchestrator, experiment_id = _orchestrator()
    orchestrator.start_experiment(experiment_id)
    attacks = orchestrator.attacks(experiment_id)
    assert [a.attack_id for a in attacks] == ["a-1", "a-2", "b-2"]
    assert all(a.status is LifecycleStatus.CREATED for a in attacks)
    assert len(orchestrator.pending_attacks(experiment_id)) == 3


# --- Attack execution interface -------------------------------------------


def test_begin_complete_and_duplicate_prevention() -> None:
    orchestrator, experiment_id = _orchestrator()
    orchestrator.start_experiment(experiment_id)
    running = orchestrator.begin_attack(experiment_id, "a-1")
    assert running.status is LifecycleStatus.RUNNING
    # Duplicate begin is an idempotent no-op: same object, no restart.
    assert orchestrator.begin_attack(experiment_id, "a-1") is running
    completed = orchestrator.complete_attack(experiment_id, "a-1")
    with pytest.raises(InvalidTransitionError):
        orchestrator.begin_attack(experiment_id, "a-1")
    # Terminal duplicate update: idempotent no-op.
    assert orchestrator.complete_attack(experiment_id, "a-1") is completed
    assert orchestrator.pending_attacks(experiment_id)[0].attack_id == "a-2"


def test_fail_attack_records_error() -> None:
    orchestrator, experiment_id = _orchestrator()
    orchestrator.start_experiment(experiment_id)
    running = orchestrator.begin_attack(experiment_id, "b-2")
    failed = orchestrator.fail_attack(
        experiment_id, running.attack_id, "executor crashed"
    )
    assert (
        failed.status is LifecycleStatus.FAILED and failed.error == "executor crashed"
    )
    with pytest.raises(KeyError, match="untracked attack"):
        orchestrator.complete_attack(experiment_id, "ghost")


def test_complete_experiment_requires_terminal_attacks() -> None:
    orchestrator, experiment_id = _orchestrator()
    orchestrator.start_experiment(experiment_id)
    with pytest.raises(ValueError, match="non-terminal"):
        orchestrator.complete_experiment(experiment_id)
    for attack_id in ("a-1", "a-2", "b-2"):
        orchestrator.begin_attack(experiment_id, attack_id)
        orchestrator.complete_attack(experiment_id, attack_id)
    completed = orchestrator.complete_experiment(experiment_id)
    assert completed.status is LifecycleStatus.COMPLETED
    assert orchestrator.complete_experiment(experiment_id) is completed
    with pytest.raises(InvalidTransitionError):
        orchestrator.fail_experiment(experiment_id, "late failure")


def test_failure_and_cancellation_preserve_context() -> None:
    orchestrator, experiment_id = _orchestrator()
    orchestrator.start_experiment(experiment_id)
    orchestrator.begin_attack(experiment_id, "a-1")
    orchestrator.complete_attack(experiment_id, "a-1")
    failed = orchestrator.fail_experiment(experiment_id, "orchestrator died")
    assert (
        failed.status is LifecycleStatus.FAILED and failed.error == "orchestrator died"
    )
    assert orchestrator.attacks(experiment_id)[0].status is LifecycleStatus.COMPLETED
    assert orchestrator.pending_attacks(experiment_id)[0].attack_id == "a-2"
    # FAILED is terminal: cancelling afterwards is rejected.
    with pytest.raises(InvalidTransitionError):
        orchestrator.cancel_experiment(experiment_id)

    fresh = RedTeamOrchestrator(_dataset())
    fresh_id = fresh.create_experiment(
        ExperimentConfig(name="cancel-flow")
    ).config.experiment_id
    fresh.start_experiment(fresh_id)
    fresh.begin_attack(fresh_id, "a-1")
    fresh.complete_attack(fresh_id, "a-1")
    state = fresh.cancel_experiment(fresh_id)
    assert state.status is LifecycleStatus.CANCELLED
    statuses = {a.attack_id: a.status for a in fresh.attacks(fresh_id)}
    assert statuses["a-1"] is LifecycleStatus.COMPLETED
    # Never-started attacks stay CREATED: CREATED->CANCELLED is not a valid
    # transition; the terminal experiment status is the authoritative signal.
    assert statuses["a-2"] is LifecycleStatus.CREATED
    assert statuses["b-2"] is LifecycleStatus.CREATED


def test_p31_p32_compatibility_via_public_api() -> None:
    dataset = AttackDataset((_case("a-1"),))
    orchestrator = RedTeamOrchestrator(dataset)
    state = ExperimentState(config=ExperimentConfig(name="compat"))
    assert state.status is LifecycleStatus.CREATED
    experiment_id = orchestrator.create_experiment(state.config).config.experiment_id
    orchestrator.start_experiment(experiment_id)
    orchestrator.begin_attack(experiment_id, "a-1")
    attack = orchestrator.begin_attack(experiment_id, "a-1")
    assert dataset.get("a-1").prompt == attack.prompt
    orchestrator.complete_attack(experiment_id, "a-1")
    assert orchestrator.complete_experiment(experiment_id).status is (
        LifecycleStatus.COMPLETED
    )
    with pytest.raises(DuplicateAttackError):
        dataset.add(_case("a-1"))
