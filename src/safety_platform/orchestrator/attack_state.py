"""Minimal per-scenario attack state for multi-turn red-team conversations.

One `AttackState` per attack scenario; frozen dataclass transitions are the
explicit state machine (running -> complete|failed) — no framework. Failed
defense turns are appended only when `execute_defense` returned a result;
exception-raising turns leave the history untouched and the loop marks the
scenario failed. History is derived from this state only, so nothing leaks
between independent scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_number: int
    prompt: str
    strategy: str
    outcome: str
    blocked: bool
    final_route: str
    target_status: str
    final_response: str | None
    raw_response: str | None
    judge_complied: bool | None
    judge_error: str | None
    latency_ms: float
    execution_error: str | None
    failure_stage: str | None
    timestamp: str


@dataclass(frozen=True, slots=True)
class AttackState:
    run_id: str
    conversation_id: str
    case_id: str
    objective: str
    attack_category: str | None
    max_turns: int
    escalation: str
    turns: tuple[TurnRecord, ...] = ()
    status: str = "running"  # running | complete | failed
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def history(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (turn.prompt, turn.final_response or "")
            for turn in self.turns
            if turn.execution_error is None
        )

    def add_turn(self, record: TurnRecord, /) -> AttackState:
        if self.status != "running":
            raise ValueError(f"scenario {self.status}, cannot add turns")
        if record.turn_number != self.turn_count + 1:
            raise ValueError(
                f"expected turn {self.turn_count + 1}, got {record.turn_number}"
            )
        if self.turn_count >= self.max_turns:
            raise ValueError(f"max_turns ({self.max_turns}) reached")
        return replace(self, turns=(*self.turns, record))

    def complete(self, reason: str, /) -> AttackState:
        return replace(self, status="complete", termination_reason=reason)

    def fail(self, reason: str, /) -> AttackState:
        return replace(self, status="failed", termination_reason=reason)

    def next_turn_number(self) -> int:
        return self.turn_count + 1


__all__ = ["AttackState", "TurnRecord"]
