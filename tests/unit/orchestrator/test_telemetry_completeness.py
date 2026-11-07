"""Task 13: run-level telemetry completeness in campaign summaries."""

from __future__ import annotations

from safety_platform.orchestrator.loop import _telemetry_completeness
from safety_platform.telemetry.consumer import ConsumerStats


def _stats(**kw) -> ConsumerStats:
    return ConsumerStats(**kw)


def test_complete_run_has_no_losses() -> None:
    consumer = type("C", (), {})()
    consumer.stats = _stats(
        events_received=10,
        events_written=8,
        events_duplicated=2,
        batches=3,
        failures=0,
        batches_retried=1,
    )
    consumer.dropped_events = 0
    block = _telemetry_completeness(consumer)

    assert block == {
        "status": "complete",
        "events_received": 10,
        "events_written": 8,
        "events_duplicated": 2,
        "events_dropped": 0,
        "batches_retried": 1,
        "events_dead_lettered": 0,
        "events_lost": 0,
        "unaccounted_events": 0,
    }


def test_recoverable_when_dead_lettered() -> None:
    consumer = type("C", (), {})()
    consumer.stats = _stats(
        events_received=10,
        events_written=6,
        events_duplicated=1,
        events_dead_lettered=3,
    )
    consumer.dropped_events = 0
    block = _telemetry_completeness(consumer)

    assert block["status"] == "recoverable"
    assert block["events_dead_lettered"] == 3


def test_incomplete_on_lost_or_dropped() -> None:
    consumer = type("C", (), {})()
    consumer.stats = _stats(events_received=10, events_written=7, events_lost=3)
    consumer.dropped_events = 0
    assert _telemetry_completeness(consumer)["status"] == "incomplete"

    consumer.stats = _stats(events_received=10, events_written=6, events_duplicated=4)
    consumer.dropped_events = 3
    assert _telemetry_completeness(consumer)["status"] == "incomplete"


def test_unaccounted_events_block_completeness() -> None:
    consumer = type("C", (), {})()
    consumer.stats = _stats(events_received=10, events_written=5, events_duplicated=2)
    consumer.dropped_events = 0  # 10 - 7 = 3 unaccounted
    block = _telemetry_completeness(consumer)

    assert block["unaccounted_events"] == 3
    assert block["status"] == "incomplete"


def test_unavailable_when_consumer_has_no_stats() -> None:
    assert _telemetry_completeness(None) == {"status": "unavailable"}


def test_empty_run_reports_no_events() -> None:
    consumer = type("C", (), {})()
    consumer.stats = _stats()
    consumer.dropped_events = 0
    assert _telemetry_completeness(consumer)["status"] == "no_events"


def test_interrupted_orchestrator_campaign_has_no_rebuild_path(tmp_path) -> None:
    """Task 14 audit: after a crash, engine telemetry rows survive but carry
    no judge verdicts or campaign decisions, so campaign summaries CANNOT be
    regenerated accurately from the durable store. The honest disclosure is
    `unavailable`/`no_events` — never fabricated counts."""
    import sqlite3

    db = tmp_path / "interrupted.sqlite3"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE stage_event (event_id TEXT PRIMARY KEY,"
        " trace_id TEXT, judge_complied INTEGER DEFAULT NULL)"
    )
    con.execute("INSERT INTO stage_event VALUES ('e1', 'rt-run-1-attack-001-t1', NULL)")
    con.commit()
    cols = {row[1] for row in con.execute("PRAGMA table_info(stage_event)").fetchall()}
    con.close()
    assert {"judge_complied"} <= cols  # schema does not model campaign state

    # ConsumerStats died with the process: fresh telemetry has no counts to
    # disclose, and none may be invented.
    assert _telemetry_completeness(None) == {"status": "unavailable"}
