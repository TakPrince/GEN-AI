"""Phase 3.1: experiment lifecycle + AttackState contract tests."""

from __future__ import annotations

import pytest

from safety_platform.redteam import (
    AttackState,
    ExperimentConfig,
    ExperimentState,
    InvalidTransitionError,
    LifecycleStatus,
)


def _config(**overrides) -> ExperimentConfig:
    kwargs = {"name": "probe-experiment", "seed": 42}
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _experiment(**overrides) -> ExperimentState:
    kwargs = {"config": _config()}
    kwargs.update(overrides)
    return ExperimentState(**kwargs)


def _attack(**overrides) -> AttackState:
    kwargs = {
        "experiment_id": "exp-1",
        "attack_type": "prompt_injection",
        "prompt": "ignore previous instructions",
    }
    kwargs.update(overrides)
    return AttackState(**kwargs)


# --- ExperimentConfig ---------------------------------------------------


def test_config_generates_unique_experiment_ids() -> None:
    assert _config().experiment_id != _config().experiment_id


def test_config_rejects_missing_name_and_bad_version() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="")
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", config_version=0)
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", seed="not-an-int")


def test_config_round_trip() -> None:
    config = _config(attack_types=("jailbreak", "injection"))
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_config_round_trip_missing_required_field() -> None:
    data = _config().to_dict()
    del data["name"]
    with pytest.raises(ValueError, match="missing config field"):
        ExperimentConfig.from_dict(data)


# --- Experiment lifecycle -----------------------------------------------


def test_valid_lifecycle_transitions() -> None:
    exp = _experiment().start().complete()
    assert exp.status is LifecycleStatus.COMPLETED
    cancelled = _experiment().start().cancel()
    assert cancelled.status is LifecycleStatus.CANCELLED
    failed = _experiment().start().fail("boom")
    assert failed.status is LifecycleStatus.FAILED and failed.error == "boom"


def test_invalid_transition_rejected() -> None:
    with pytest.raises(InvalidTransitionError):
        _experiment().complete()  # CREATED -> COMPLETED
    with pytest.raises(InvalidTransitionError):
        _experiment().cancel()  # CREATED -> CANCELLED


def test_terminal_states_are_enforced() -> None:
    done = _experiment().start().complete()
    for target in LifecycleStatus:
        if target is not done.status:
            with pytest.raises(InvalidTransitionError):
                done.transition(target)
    failed = _experiment().start().fail("x")
    with pytest.raises(InvalidTransitionError):
        failed.start()


def test_duplicate_state_update_is_idempotent_noop() -> None:
    exp = _experiment()
    again = exp.transition(LifecycleStatus.CREATED)
    assert again is exp
    running = exp.start()
    assert running.transition(LifecycleStatus.RUNNING) is running
    completed = running.complete()
    assert completed.complete() is completed


def test_timestamps_track_transitions() -> None:
    exp = _experiment()
    started = exp.start()
    assert started.created_at == exp.created_at
    assert started.updated_at >= exp.updated_at
    completed = started.complete()
    assert completed.updated_at >= started.updated_at
    assert completed.created_at == exp.created_at


def test_fail_requires_error_context() -> None:
    with pytest.raises(ValueError, match="error description"):
        _experiment().start().transition(LifecycleStatus.FAILED)
    with pytest.raises(ValueError, match="only recorded on FAILED"):
        _experiment().start().transition(LifecycleStatus.RUNNING, error="x")
    assert _experiment().start().fail("boom").error == "boom"
    assert _experiment().start().cancel().error is None


def test_experiment_round_trip_preserves_lifecycle() -> None:
    exp = _experiment().start().fail("model crashed")
    assert ExperimentState.from_dict(exp.to_dict()) == exp
    assert ExperimentState.from_dict(exp.to_dict()).status is LifecycleStatus.FAILED


def test_experiment_round_trip_rejects_malformed() -> None:
    data = _experiment().to_dict()
    missing = dict(data)
    del missing["updated_at"]
    with pytest.raises(ValueError, match="missing experiment field"):
        ExperimentState.from_dict(missing)
    with pytest.raises(ValueError, match="unknown status"):
        ExperimentState.from_dict({**data, "status": "paused"})
    with pytest.raises(ValueError, match="missing config field"):
        ExperimentState.from_dict({**data, "config": {"name": "x"}})


# --- AttackState ---------------------------------------------------------


def test_attack_creation_and_validation() -> None:
    attack = _attack(metadata={"source": "dataset-a"})
    assert attack.status is LifecycleStatus.CREATED
    assert attack.attack_id and attack.experiment_id == "exp-1"
    with pytest.raises(ValueError):
        _attack(prompt="")
    with pytest.raises(ValueError):
        _attack(attack_type="")
    with pytest.raises(ValueError, match="LifecycleStatus"):
        _attack(status="running")


def test_attack_lifecycle_follows_same_rules() -> None:
    attack = _attack().start().complete()
    assert attack.status is LifecycleStatus.COMPLETED
    with pytest.raises(InvalidTransitionError):
        attack.start()
    with pytest.raises(InvalidTransitionError):
        _attack().complete()
    assert _attack().start().fail("executor error").error == "executor error"
    duplicate = _attack().start()
    assert duplicate.start() is duplicate
    assert duplicate.updated_at == duplicate.start().updated_at


def test_attack_round_trip() -> None:
    attack = _attack(attack_id="a-1", metadata={"turn": "3"}).start().fail("x")
    assert AttackState.from_dict(attack.to_dict()) == attack
