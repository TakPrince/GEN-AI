from __future__ import annotations

from typing import Any

import pytest

from safety_platform.orchestrator.attack_state import AttackState, TurnRecord


def _turn(n: int = 1, **overrides):
    fields: dict[str, Any] = {
        "turn_number": n,
        "prompt": f"prompt-{n}",
        "strategy": "crescendo",
        "outcome": "safe",
        "blocked": False,
        "final_route": "allowed",
        "target_status": "executed",
        "final_response": f"reply-{n}",
        "raw_response": None,
        "judge_complied": False,
        "judge_error": None,
        "latency_ms": 5.0,
        "execution_error": None,
        "failure_stage": None,
        "timestamp": "t",
    }
    fields.update(overrides)
    return TurnRecord(**fields)


def _state(**overrides) -> AttackState:
    fields: dict[str, Any] = {
        "run_id": "run-1",
        "conversation_id": "run-1-attack-001",
        "case_id": "attack-001",
        "objective": "objective text",
        "attack_category": "malware",
        "max_turns": 3,
        "escalation": "crescendo",
    }
    fields.update(overrides)
    return AttackState(**fields)


def test_initial_state_is_running_and_empty() -> None:
    state = _state()
    assert state.turns == ()
    assert state.status == "running"
    assert state.termination_reason is None
    assert state.turn_count == 0


def test_add_turn_appends_and_preserves_order() -> None:
    state = _state().add_turn(_turn(1)).add_turn(_turn(2))
    assert [t.turn_number for t in state.turns] == [1, 2]
    assert state.history() == (("prompt-1", "reply-1"), ("prompt-2", "reply-2"))


def test_turn_numbers_must_be_sequential() -> None:
    state = _state().add_turn(_turn(1))
    with pytest.raises(ValueError):
        state.add_turn(_turn(3))


def test_max_turns_enforced() -> None:
    state = _state(max_turns=2).add_turn(_turn(1)).add_turn(_turn(2))
    with pytest.raises(ValueError):
        state.add_turn(_turn(3))


def test_terminal_state_ignores_new_turns() -> None:
    state = _state().add_turn(_turn(1)).complete("complied")
    with pytest.raises(ValueError):
        state.add_turn(_turn(2))
    assert state.status == "complete"
    assert state.termination_reason == "complied"


def test_failed_state_preserves_existing_history() -> None:
    state = _state().add_turn(_turn(1)).fail("RuntimeError")
    assert len(state.turns) == 1
    assert state.status == "failed"
    with pytest.raises(ValueError):
        state.add_turn(_turn(2))


def test_states_are_independent_between_scenarios() -> None:
    one = _state(case_id="attack-001").add_turn(_turn(1))
    other = _state(case_id="attack-002")
    one = one.add_turn(_turn(2))
    assert one.turn_count == 2
    assert other.turn_count == 0
    assert one.case_id != other.case_id


def test_failed_turn_is_not_appended() -> None:
    state = _state().add_turn(_turn(1))
    # A turn whose execute_defense raised is never recorded as a normal turn;
    # the conversation state stays at turn 1 and is marked failed by the loop.
    assert state.turn_count == 1
    assert state.status == "running"
    state = state.fail("ValueError")
    assert state.status == "failed"
    assert state.turn_count == 1
