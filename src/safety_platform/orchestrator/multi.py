"""Multi-turn red-team conversation runner (Task 10).

Every turn of every conversation goes through the shared
`safety_platform.execution.execute_defense` boundary; nothing here calls the
target model directly. Turn-level binary compliance reuses the hardened
judge; conversation-level structured verdicts come from
`ComplianceJudge.judge_conversation` (same client and hardening). Failed
turns end the scenario explicitly — they never silently corrupt history, and
a blocked turn never triggers any direct target call to continue.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..config.registry import Direction, ExecutionContext
from ..evaluation.dataset import EvaluationCase
from ..evaluation.llm import ComplianceVerdict, JudgeResult
from ..execution import DefenseRequest, execute_defense
from .attack_state import AttackState, TurnRecord
from .escalation import ESCALATIONS, build_turn_prompt

if TYPE_CHECKING:
    pass

CONSECUTIVE_BLOCK_GIVEUP = 2
REFUSAL_MARKERS = ("cannot help", "i can't", "i cannot", "refuse")


class _CampaignRuntime(Protocol):
    """Structural subset of the unified-defense runtime (mirror of loop.py)."""

    @property
    def regex(self) -> object: ...

    @property
    def vector(self) -> object: ...

    @property
    def guard(self) -> object: ...

    @property
    def ml_router(self) -> object: ...

    @property
    def router(self) -> object: ...

    @property
    def cascade(self) -> object: ...

    @property
    def echo_probe(self) -> object: ...

    @property
    def bus(self) -> object: ...

    @property
    def consumer(self) -> object: ...


class _Judger(Protocol):
    def judge(self, prompt: str, response: str) -> JudgeResult: ...

    def judge_conversation(self, objective: str, transcript: str) -> Any: ...


def _run_id() -> str:
    import uuid

    return f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    run_id: str
    timestamp: str
    conversation_id: str
    case_id: str
    attack_category: str | None
    objective: str
    escalation: str
    max_turns: int
    turns: tuple[TurnRecord, ...]
    status: str  # complete | failed
    termination_reason: str | None
    conversation_verdict: str
    conversation_reason: str
    conversation_error: str | None
    model_metadata: tuple[tuple[str, str], ...]
    total_latency_ms: float

    @property
    def judge_errors(self) -> int:
        return sum(1 for turn in self.turns if turn.judge_error is not None)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def scorable_turns(self) -> tuple[TurnRecord, ...]:
        return tuple(
            turn
            for turn in self.turns
            if turn.execution_error is None and turn.judge_complied is not None
        )


async def run_conversation(
    runtime: _CampaignRuntime,
    case: EvaluationCase,
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    escalation: str = "crescendo",
    max_turns: int = 3,
    seed: int = 0,
    model_metadata: tuple[tuple[str, str], ...] = (),
    reference_time: str | None = None,
    resume_state: AttackState | None = None,
    resume_pending: TurnRecord | None = None,
    on_turn_start: Callable[[int], None] | None = None,
    on_response: Callable[[AttackState, TurnRecord], None] | None = None,
    on_turn_done: Callable[[AttackState], None] | None = None,
) -> ScenarioOutcome:
    """`resume_state` continues a persisted conversation. `resume_pending` is a
    target response persisted before judging: it is judged, never resent. The
    hooks let a durable store checkpoint each turn (Tasks 15 and 17)."""
    if escalation not in ESCALATIONS:
        raise ValueError(f"unknown escalation strategy: {escalation!r}")
    timestamp = reference_time or datetime.now(UTC).isoformat()
    conversation_id = f"{run_id}-{case.case_id}"
    state = resume_state or AttackState(
        run_id=run_id,
        conversation_id=conversation_id,
        case_id=case.case_id,
        objective=case.prompt,
        attack_category=case.category,
        max_turns=max_turns,
        escalation=escalation,
    )
    total_latency = sum(turn.latency_ms for turn in state.turns)
    consecutive_blocks = 0
    for turn_record in reversed(state.turns):  # trailing blocks, as the loop counts
        if not turn_record.blocked:
            break
        consecutive_blocks += 1
    if state.status == "failed":
        return _finalize(state, escalation, timestamp, model_metadata, total_latency)

    def checkpoint(current: AttackState) -> AttackState:
        if on_turn_done is not None:
            on_turn_done(current)
        return current

    pending = resume_pending
    while state.status == "running" and state.turn_count < state.max_turns:
        if pending is None:
            turn = state.turn_count + 1
            prompt = build_turn_prompt(
                case.prompt,
                escalation,
                turn=turn,
                history=state.history(),
                seed=seed + turn,
            )
            if on_turn_start is not None:
                on_turn_start(turn)
            response, error = await _defend_turn(
                runtime,
                case,
                state,
                completion,
                turn=turn,
                prompt=prompt,
                escalation=escalation,
                timestamp=timestamp,
            )
            if response is None:  # execute_defense raised
                return _finalize(
                    state.fail(error),
                    escalation,
                    timestamp,
                    model_metadata,
                    total_latency,
                )
            if _is_executed(response) and on_response is not None:
                on_response(state, response)  # persisted before judging
        else:
            response, pending = pending, None
        record = _judge_turn(response, judge)
        state = state.add_turn(record)
        if not _is_executed(record):
            state = state.fail(
                "defense_error:"
                f"{record.failure_stage or record.outcome}:"
                f"{record.execution_error or 'outcome=error'}"
            )
        total_latency += record.latency_ms
        consecutive_blocks = consecutive_blocks + 1 if record.blocked else 0
        # Termination is decided before the checkpoint, so a resumed run never
        # extends a conversation that already ended.
        if record.judge_complied:
            state = checkpoint(state.complete("complied"))
            break
        if state.status != "running":
            checkpoint(state)
            return _finalize(
                state, escalation, timestamp, model_metadata, total_latency
            )
        if consecutive_blocks >= CONSECUTIVE_BLOCK_GIVEUP:
            state = checkpoint(state.complete("blocked_convergence"))
            break
        checkpoint(state)
    if state.status == "running":
        state = state.complete("max_turns")
    outcome = _conversation_judge(
        state, judge, timestamp, escalation, max_turns, model_metadata, total_latency
    )
    return outcome


async def _defend_turn(
    runtime: _CampaignRuntime,
    case: EvaluationCase,
    state: AttackState,
    completion: Callable[[str], str],
    *,
    turn: int,
    prompt: str,
    escalation: str,
    timestamp: str,
) -> tuple[TurnRecord | None, str]:
    """Target stage only: the received response, before any judging."""
    request = DefenseRequest(
        prompt=prompt,
        request_id=f"{state.run_id}-{case.case_id}-t{turn}",
        attack_id=case.case_id,
        attack_category=case.category,
        metadata=(
            ("label", "attack"),
            ("strategy", escalation),
            ("turn", str(turn)),
        ),
    )
    context = ExecutionContext(
        trace_id=f"rt-{state.run_id}-{case.case_id}-t{turn}",
        turn_id=str(turn),
        conversation_id=state.conversation_id,
        direction=Direction.INPUT,
        guardrail_version="redteam-v1",
        user_input=prompt,
    )
    try:
        result = await execute_defense(
            runtime,  # type: ignore[arg-type]
            request,
            context,
            completion=completion,
        )
    except Exception as exc:
        return None, f"execution_error:{type(exc).__name__}"
    record = TurnRecord(
        turn_number=turn,
        prompt=prompt,
        strategy=escalation,
        outcome=result.outcome,
        blocked=result.blocked,
        final_route=result.final_route,
        target_status=result.target_status,
        final_response=result.final_response,
        raw_response=result.raw_response,
        judge_complied=None,
        judge_error=None,
        latency_ms=result.latency_ms,
        execution_error=result.error,
        failure_stage=result.failure_stage,
        timestamp=timestamp,
        target_input=result.target_input,
    )
    return record, ""


def _is_executed(record: TurnRecord) -> bool:
    return record.execution_error is None and record.outcome != "error"


def _judge_turn(record: TurnRecord, judge: _Judger) -> TurnRecord:
    """Judging stage: a pure function of the received response, so it can resume."""
    if not _is_executed(record):
        return record
    try:
        verdict = judge.judge(record.prompt, record.final_response or "")
    except Exception as exc:
        return replace(record, judge_error=type(exc).__name__)
    return replace(
        record,
        judge_complied=bool(verdict.complied) if verdict.error is None else None,
        judge_error=verdict.error,
    )


def _conversation_judge(
    state: AttackState,
    judge: _Judger,
    timestamp: str,
    escalation: str,
    max_turns: int,
    model_metadata: tuple[tuple[str, str], ...],
    total_latency_ms: float,
) -> ScenarioOutcome:
    verdict = ComplianceVerdict.UNSCORABLE.value
    reason = ""
    error = "scenario_failed" if state.status == "failed" else None
    if state.status == "complete":
        transcript = "\n".join(
            f"USER: {user}\nASSISTANT: {reply}" for user, reply in state.history()
        )
        try:
            structured = judge.judge_conversation(state.objective, transcript)
        except Exception as exc:
            verdict = ComplianceVerdict.JUDGE_ERROR.value
            error = type(exc).__name__
        else:
            verdict = structured.verdict.value
            reason = structured.reason
            error = structured.error
    return ScenarioOutcome(
        run_id=state.run_id,
        timestamp=timestamp,
        conversation_id=state.conversation_id,
        case_id=state.case_id,
        attack_category=state.attack_category,
        objective=state.objective,
        escalation=escalation,
        max_turns=max_turns,
        turns=state.turns,
        status=state.status,
        termination_reason=state.termination_reason,
        conversation_verdict=verdict,
        conversation_reason=reason,
        conversation_error=error,
        model_metadata=model_metadata,
        total_latency_ms=total_latency_ms,
    )


def _finalize(
    state: AttackState, escalation, timestamp, model_metadata, total_latency
) -> ScenarioOutcome:
    return ScenarioOutcome(
        run_id=state.run_id,
        timestamp=timestamp,
        conversation_id=state.conversation_id,
        case_id=state.case_id,
        attack_category=state.attack_category,
        objective=state.objective,
        escalation=escalation,
        max_turns=state.max_turns,
        turns=state.turns,
        status=state.status,
        termination_reason=state.termination_reason,
        conversation_verdict=ComplianceVerdict.UNSCORABLE.value,
        conversation_reason="",
        conversation_error=f"scenario_failed:{state.termination_reason}",
        model_metadata=model_metadata,
        total_latency_ms=total_latency,
    )


def _failed_scenario(
    run_id: str,
    case: EvaluationCase,
    escalation: str,
    timestamp: str,
    max_turns: int,
    model_metadata: tuple[tuple[str, str], ...],
    exc: Exception,
) -> ScenarioOutcome:
    return ScenarioOutcome(
        run_id=run_id,
        timestamp=timestamp,
        conversation_id=f"{run_id}-{case.case_id}",
        case_id=case.case_id,
        attack_category=case.category,
        objective=case.prompt,
        escalation=escalation,
        max_turns=max_turns,
        turns=(),
        status="failed",
        termination_reason=f"campaign_error:{type(exc).__name__}",
        conversation_verdict=ComplianceVerdict.UNSCORABLE.value,
        conversation_reason="",
        conversation_error=type(exc).__name__,
        model_metadata=model_metadata,
        total_latency_ms=0.0,
    )


async def run_conversation_campaign(
    runtime: _CampaignRuntime,
    cases: Sequence[EvaluationCase],
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    per_label: int = 3,
    max_turns: int = 3,
    escalations: Sequence[str] = ("crescendo",),
    seed: int = 0,
    model_metadata: tuple[tuple[str, str], ...] = (),
    reference_time: str | None = None,
) -> tuple[tuple[ScenarioOutcome, ...], dict[str, Any]]:
    from .loop import _pick
    from .metrics import aggregate_conversations

    for escalation in escalations:
        if escalation not in ESCALATIONS:
            raise ValueError(f"unknown escalation strategy: {escalation!r}")
    rng = random.Random(f"{run_id}|{escalations}|{seed}")
    selected = _pick(cases, "attack", per_label, rng)
    assign = tuple(escalations[rng.randrange(len(escalations))] for _ in selected)
    timestamp = reference_time or datetime.now(UTC).isoformat()
    records: list[ScenarioOutcome] = []
    for index, (case, escalation) in enumerate(zip(selected, assign, strict=False)):
        try:
            records.append(
                await run_conversation(
                    runtime,
                    case,
                    judge,
                    completion,
                    run_id=run_id,
                    escalation=escalation,
                    max_turns=max_turns,
                    seed=seed + index,
                    model_metadata=model_metadata,
                    reference_time=timestamp,
                )
            )
        except Exception as exc:
            records.append(
                _failed_scenario(
                    run_id, case, escalation, timestamp, max_turns, model_metadata, exc
                )
            )
    summary = aggregate_conversations(records, run_id=run_id)
    # Same accounting vocabulary as the single-turn summary (Task 13).
    from .loop import _telemetry_completeness

    consumer = getattr(runtime, "consumer", None)
    if consumer is not None:
        summary["telemetry"] = _telemetry_completeness(consumer)
    return tuple(records), summary


def write_conversations_jsonl(
    outcomes: Sequence[ScenarioOutcome], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({**asdict(outcome), "model_metadata": dict(outcome.model_metadata)})
        + "\n"
        for outcome in outcomes
    ]
    destination.write_text("".join(lines), encoding="utf-8")


__all__ = [
    "ScenarioOutcome",
    "run_conversation",
    "run_conversation_campaign",
    "write_conversations_jsonl",
]
