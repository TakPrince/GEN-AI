"""Single-turn red-team orchestrator.

Every defended attack runs through the shared defense boundary
(`safety_platform.execution.execute_defense`); this module never calls the
target model directly. The independent `ComplianceJudge` evaluates the
user-visible response afterwards; execution failures and judge failures are
recorded separately and excluded from ASR denominators.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..config.registry import Direction, ExecutionContext
from ..evaluation.dataset import EvaluationCase
from ..evaluation.llm import JudgeResult
from ..execution import DefenseRequest, execute_defense
from .metrics import aggregate
from .mutators import STRATEGIES, Mutation, mutate, selection_for

if TYPE_CHECKING:
    pass

REFUSAL_TEXT = "I cannot help with that request."


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    run_id: str
    timestamp: str
    case_id: str
    label: str
    attack_category: str | None
    original_prompt: str
    executed_prompt: str
    mutation_strategy: str
    outcome: str
    blocked: bool
    final_route: str
    target_status: str
    raw_response: str | None
    final_response: str | None
    judge_complied: bool | None
    judge_reason: str | None
    judge_error: str | None
    raw_judge_complied: bool | None
    raw_judge_error: str | None
    latency_ms: float
    execution_error: str | None
    failure_stage: str | None
    trace_id: str
    model_metadata: tuple[tuple[str, str], ...]
    partial_compliance: None  # binary judge: never fabricated
    # Exact text received by the completion callback (DefenseResult.target_input);
    # None when the target never ran.
    target_input: str | None = None


class _CampaignRuntime(Protocol):
    """Structural match on the unified-defense runtime fields."""

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


def _run_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _pick(cases: Sequence[EvaluationCase], label: str, count: int, rng: random.Random):
    selected = sorted((c for c in cases if c.label == label), key=lambda c: c.case_id)
    rng.shuffle(selected)
    return selected[: max(0, count)]


async def run_campaign(
    runtime: _CampaignRuntime,
    cases: Sequence[EvaluationCase],
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    per_label: int = 5,
    strategies: Sequence[str] = ("direct",),
    seed: int = 0,
    model_metadata: tuple[tuple[str, str], ...] = (),
    reference_time: str | None = None,
) -> tuple[tuple[AttemptOutcome, ...], dict[str, Any]]:
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown mutation strategy: {strategy!r}")
    rng = random.Random(f"{run_id}|{strategies}|{seed}")
    selected = _pick(cases, "attack", per_label, rng) + _pick(
        cases, "benign", per_label, rng
    )
    picks = selection_for(tuple(selected), tuple(strategies), seed)
    timestamp = reference_time or datetime.now(UTC).isoformat()

    records: list[AttemptOutcome] = []
    for index, (case, strategy) in enumerate(zip(selected, picks, strict=False)):
        mutation = mutate(case.prompt, strategy, seed=seed + index)
        records.append(
            await _execute_single(
                runtime,
                case,
                mutation,
                judge,
                completion,
                run_id=run_id,
                timestamp=timestamp,
                seed=seed + index,
                model_metadata=model_metadata,
            )
        )
    summary = aggregate(records, run_id=run_id)
    # Actual run-level stats from the live consumer; absent if the runtime
    # does not carry telemetry (historical summaries stay untouched).
    consumer = getattr(runtime, "consumer", None)
    if consumer is not None:
        summary["telemetry"] = _telemetry_completeness(consumer)
    return tuple(records), summary


async def _defend_single(
    runtime: _CampaignRuntime,
    case: EvaluationCase,
    mutation: Mutation,
    completion: Callable[[str], str],
    *,
    run_id: str,
    timestamp: str,
    seed: int,
    model_metadata: tuple[tuple[str, str], ...],
) -> AttemptOutcome:
    """Target stage only: the defended request/response, before any judging."""
    is_attack = case.label == "attack"
    request = DefenseRequest(
        prompt=mutation.executed_prompt,
        request_id=f"{run_id}-{case.case_id}",
        attack_id=case.case_id if is_attack else None,
        attack_category=case.category if is_attack else None,
        metadata=(
            ("label", case.label),
            ("strategy", mutation.strategy),
            ("seed", str(seed)),
        ),
    )
    context = ExecutionContext(
        trace_id=f"rt-{run_id}-{case.case_id}",
        turn_id="1",
        conversation_id=f"{run_id}-{case.label}-{case.case_id}",
        direction=Direction.INPUT,
        guardrail_version="redteam-v1",
        user_input=mutation.executed_prompt,
    )
    base = AttemptOutcome(
        run_id=run_id,
        timestamp=timestamp,
        case_id=case.case_id,
        label=case.label,
        attack_category=case.category if is_attack else None,
        original_prompt=mutation.original_prompt,
        executed_prompt=mutation.executed_prompt,
        mutation_strategy=mutation.strategy,
        outcome="error",
        blocked=False,
        final_route="error",
        target_status="skipped",
        raw_response=None,
        final_response=None,
        judge_complied=None,
        judge_reason=None,
        judge_error=None,
        raw_judge_complied=None,
        raw_judge_error=None,
        latency_ms=0.0,
        execution_error=None,
        failure_stage=None,
        trace_id=context.trace_id,
        model_metadata=model_metadata,
        partial_compliance=None,
    )
    try:
        result = await execute_defense(
            runtime,  # type: ignore[arg-type]
            request,
            context,
            completion=completion,
        )
    except Exception as exc:
        return replace(base, execution_error=type(exc).__name__)
    record = replace(
        base,
        outcome=result.outcome,
        blocked=result.blocked,
        final_route=result.final_route,
        target_status=result.target_status,
        raw_response=result.raw_response,
        final_response=result.final_response,
        latency_ms=result.latency_ms,
        failure_stage=result.failure_stage,
        execution_error=result.error,
        target_input=result.target_input,
    )
    return record


def _needs_judging(record: AttemptOutcome) -> bool:
    return record.execution_error is None and record.outcome != "error"


async def _judge_single(
    record: AttemptOutcome, mutation: Mutation, judge: _Judger
) -> AttemptOutcome:
    """Judging stage: a pure function of the received record, so it can resume."""
    if _needs_judging(record):
        record = await _judge_record(record, mutation, judge)
    if record.blocked and record.raw_response is not None:
        return await _raw_judge_record(record, mutation, judge)
    return record


async def _execute_single(
    runtime: _CampaignRuntime,
    case: EvaluationCase,
    mutation: Mutation,
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    timestamp: str,
    seed: int,
    model_metadata: tuple[tuple[str, str], ...],
) -> AttemptOutcome:
    record = await _defend_single(
        runtime,
        case,
        mutation,
        completion,
        run_id=run_id,
        timestamp=timestamp,
        seed=seed,
        model_metadata=model_metadata,
    )
    return await _judge_single(record, mutation, judge)


async def _judge_record(
    record: AttemptOutcome,
    mutation: Mutation,
    judge: _Judger,
) -> AttemptOutcome:
    """Independent compliance evaluation of the user-visible response only.

    The judge is a synchronous Ollama client (network call + admission wait);
    it must run in a worker thread so the event loop stays responsive.
    """
    try:
        verdict = await asyncio.to_thread(
            judge.judge, mutation.executed_prompt, record.final_response or ""
        )
    except Exception as exc:  # judge infrastructure failure is a measurement gap
        return replace(record, judge_error=type(exc).__name__)
    complied = bool(verdict.complied) if verdict.error is None else None
    return replace(
        record,
        judge_complied=complied,
        judge_reason=verdict.reason or None,
        judge_error=verdict.error,
    )


async def _raw_judge_record(
    record: AttemptOutcome,
    mutation: Mutation,
    judge: _Judger,
) -> AttemptOutcome:
    """Diagnostic-only evaluation of the raw model output (never the ASR)."""
    assert record.raw_response is not None
    try:
        verdict = await asyncio.to_thread(
            judge.judge, mutation.executed_prompt, record.raw_response
        )
    except Exception as exc:
        return replace(record, raw_judge_error=type(exc).__name__)
    return replace(
        record,
        raw_judge_complied=bool(verdict.complied) if verdict.error is None else None,
        raw_judge_error=verdict.error,
    )


def write_jsonl(records: Iterable[AttemptOutcome], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({**asdict(record), "model_metadata": dict(record.model_metadata)})
        + "\n"
        for record in records
    ]
    destination.write_text("".join(lines), encoding="utf-8")


def write_summary(summary: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _telemetry_completeness(consumer: object) -> dict[str, Any]:
    """Run-level accounting from the actual ConsumerStats (Task 13).

    Never estimates from SQLite row counts and never reports `complete`
    while events are unaccounted, lost, dropped or only dead-lettered.
    """
    stats = getattr(getattr(consumer, "stats", None), "events_received", None)
    if stats is None or not hasattr(consumer, "dropped_events"):
        return {"status": "unavailable"}
    s = consumer.stats  # type: ignore[attr-defined]
    received = s.events_received
    if received == 0 and consumer.dropped_events == 0:  # type: ignore[attr-defined]
        return {"status": "no_events"}
    unaccounted = received - (
        s.events_written + s.events_duplicated + s.events_dead_lettered + s.events_lost
    )
    losses = s.events_lost + consumer.dropped_events + unaccounted  # type: ignore[attr-defined]
    if losses > 0:
        status = "incomplete"
    elif s.events_dead_lettered > 0:
        status = "recoverable"
    else:
        status = "complete"
    return {
        "status": status,
        "events_received": received,
        "events_written": s.events_written,
        "events_duplicated": s.events_duplicated,
        "events_dropped": consumer.dropped_events,  # type: ignore[attr-defined]
        "batches_retried": s.batches_retried,
        "events_dead_lettered": s.events_dead_lettered,
        "events_lost": s.events_lost,
        "unaccounted_events": unaccounted,
    }


def attempt_fields() -> tuple[str, ...]:
    return tuple(field.name for field in fields(AttemptOutcome))


__all__ = [
    "AttemptOutcome",
    "_telemetry_completeness",
    "run_campaign",
    "write_jsonl",
    "write_summary",
]
