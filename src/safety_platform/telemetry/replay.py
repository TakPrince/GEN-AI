"""Dead-letter replay (Task 13): rebuild events from JSONL rows and re-insert.

Uses the existing SQLite sink's INSERT OR IGNORE idempotency, so repeated
replay is safe. Never modifies the original dead-letter file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .events import StageEvent

_ROW_LEN = 15  # sink schema columns per _row()


class _ReplaySink(Protocol):
    def append_many(self, events: Sequence[StageEvent]) -> int: ...


# Task 13 §4.6/4.7: a failed record is described, not discarded.
@dataclass(frozen=True, slots=True)
class ReplayFailure:
    line_no: int
    error: str
    detail: str


@dataclass(slots=True)
class ReplayReport:
    records_read: int = 0
    events_recovered: int = 0  # newly inserted by this replay
    events_duplicated: int = 0  # already present (idempotent replay)
    events_failed: int = 0
    malformed_records: int = 0
    invalid_records: int = 0
    failures: tuple[ReplayFailure, ...] = field(default_factory=tuple)


def _len_or_q(row) -> str:
    return str(len(row)) if isinstance(row, list | tuple) else "?"


def _event_from_row(row: list) -> StageEvent:
    from safety_platform.config.registry import (
        Direction,
        ExecutionContext,
        SafetyVerdict,
        Stage,
        TelemetryLane,
    )

    if not isinstance(row, list | tuple) or len(row) != _ROW_LEN:
        raise ValueError(f"row arity {_len_or_q(row)} != {_ROW_LEN}")
    (
        event_id,
        trace_id,
        turn_id,
        conversation_id,
        direction,
        guardrail_version,
        stage,
        verdict,
        lane,
        latency_ms,
        prompt_hash,
        timestamp_ns,
        rule_ids,
        redaction_spans,
        error,
    ) = row
    return StageEvent(
        event_id=event_id,
        context=ExecutionContext(
            trace_id=trace_id,
            turn_id=turn_id,
            conversation_id=conversation_id,
            direction=Direction(direction),
            guardrail_version=guardrail_version,
        ),
        stage=Stage(stage),
        verdict=SafetyVerdict(verdict),
        lane=TelemetryLane(lane),
        latency_ms=float(latency_ms),
        prompt_hash=prompt_hash,
        timestamp_ns=int(timestamp_ns),
        rule_ids=tuple(json.loads(rule_ids or "[]")),
        redaction_spans=tuple(
            tuple(span) for span in json.loads(redaction_spans or "[]")
        ),
        error=error,
    )


def _parse_record(record) -> list[StageEvent]:
    if not isinstance(record, dict) or not isinstance(record.get("rows"), list):
        raise ValueError("record is not an object with a 'rows' list")
    if record.get("error") is None or record.get("failed_at") is None:
        raise ValueError("record is missing 'error' or 'failed_at'")
    events = [_event_from_row(row) for row in record["rows"]]
    if record.get("event_ids") != [event.event_id for event in events]:
        raise ValueError("event_ids do not match rows")
    return events


def _replay_record(
    report: ReplayReport, sink: _ReplaySink, line_no: int, record
) -> None:
    events = _parse_record(record)
    try:
        inserted = sink.append_many(events)
    except Exception as exc:
        report.events_failed += len(events)
        report.failures += (
            ReplayFailure(
                line_no,
                type(exc).__name__,
                f"event_ids={tuple(e.event_id for e in events)}: {exc}",
            ),
        )
        return
    report.events_recovered += inserted
    report.events_duplicated += len(events) - inserted


def replay_dead_letter(
    sink: _ReplaySink,
    path: str | Path,
) -> ReplayReport:
    """Replay one dead-letter JSONL file into the sink.

    Never clears the source file. Malformed or schema-invalid records are
    counted and reported; valid neighbouring records are still replayed.
    """

    report = ReplayReport()
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            report.records_read += 1
            try:
                record = json.loads(line)
                _replay_record(report, sink, line_no, record)
            except Exception as exc:
                if isinstance(exc, json.JSONDecodeError):
                    report.malformed_records += 1
                else:
                    report.invalid_records += 1
                report.failures += (
                    ReplayFailure(
                        line_no,
                        type(exc).__name__,
                        # Keep the raw line so recovery is possible later.
                        f"{exc}: {line.strip()[:200]}",
                    ),
                )
    return report


__all__ = ["ReplayFailure", "ReplayReport", "replay_dead_letter"]
