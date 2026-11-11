"""GEN-08: dashboard sessions must not share mutable telemetry state."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from safety_platform.config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)


def _event(tag: str, index: int) -> TelemetryEvent:
    return TelemetryEvent(
        event_id=f"ev-{tag}-{index}",
        context=ExecutionContext(f"trace-{tag}", f"turn-{tag}", f"conv-{tag}"),
        stage=Stage.TIER1,
        verdict=SafetyVerdict.PASS,
        latency_ms=1.0,
        prompt_hash="h" * 64,
        timestamp_ns=time.time_ns(),
    )


def _drain(pipe, tag: str, count: int) -> int:
    async def go() -> int:
        for index in range(count):
            pipe.bus.emit_nowait(_event(tag, index))
        stop = asyncio.Event()
        stop.set()
        return await pipe.consumer.run(stop=stop)

    return asyncio.run(go())


def test_each_session_gets_private_bus_and_consumer(tmp_path: Path) -> None:
    from safety_platform import app

    database = str(tmp_path / "dashboard_demo.sqlite3")
    first = app._session_pipeline(database, "session-a")
    second = app._session_pipeline(database, "session-b")

    assert first is not second
    assert first.bus is not second.bus
    assert first.consumer is not second.consumer
    assert app._session_pipeline(database, "session-a") is first


def test_concurrent_session_drains_do_not_cross_write(tmp_path: Path) -> None:
    from safety_platform import app

    database = tmp_path / "dashboard_demo.sqlite3"
    first = app._session_pipeline(str(database), "session-a")
    second = app._session_pipeline(str(database), "session-b")
    count = 200
    written: dict[str, int] = {}
    barrier = threading.Barrier(2)

    def run(pipe, tag: str) -> None:
        async def go() -> int:
            for index in range(count):
                pipe.bus.emit_nowait(_event(tag, index))
            barrier.wait()
            stop = asyncio.Event()
            stop.set()
            return await pipe.consumer.run(stop=stop)

        written[tag] = asyncio.run(go())

    threads = [
        threading.Thread(target=run, args=(first, "A")),
        threading.Thread(target=run, args=(second, "B")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Each call reports exactly its own events, never the other session's.
    assert written == {"A": count, "B": count}
    connection = sqlite3.connect(database)
    try:
        rows = dict(
            connection.execute(
                "SELECT conversation_id, COUNT(*) FROM stage_event GROUP BY 1"
            ).fetchall()
        )
    finally:
        connection.close()
    assert rows == {"conv-A": count, "conv-B": count}


def test_two_app_sessions_keep_separate_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from safety_platform.evaluation.llm import OllamaClient

    monkeypatch.setenv("SAFETY_TELEMETRY_DB", str(tmp_path / "metrics.sqlite3"))
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    monkeypatch.setattr(OllamaClient, "generate", lambda self, prompt, **kw: "Paris.")
    app_path = str(Path(__file__).parents[2] / "src" / "safety_platform" / "app.py")

    first = AppTest.from_file(app_path, default_timeout=30).run()
    second = AppTest.from_file(app_path, default_timeout=30).run()
    for dashboard, prompt in ((first, "hello from A"), (second, "hello from B")):
        dashboard.session_state["page"] = "Live Test"
        dashboard.run()
        dashboard.text_area[0].set_value(prompt).run()
        dashboard.button[0].click().run()

    assert first.session_state["session_id"] != second.session_state["session_id"]
    assert [t["prompt"] for t in first.session_state["turns"]] == ["hello from A"]
    assert [t["prompt"] for t in second.session_state["turns"]] == ["hello from B"]
    assert first.session_state["turns"][0]["persisted"]
    assert second.session_state["turns"][0]["persisted"]
