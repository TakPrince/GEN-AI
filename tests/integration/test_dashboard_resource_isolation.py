"""GEN-08 + GEN-14: concurrent dashboard sessions under bounded load."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from safety_platform.config.registry import ExecutionContext


@pytest.mark.asyncio
async def test_concurrent_sessions_isolated_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safety_platform import app
    from safety_platform.evaluation.llm import OllamaClient

    monkeypatch.setenv("SAFETY_UI_TIER3", "heuristic")
    monkeypatch.setattr(app, "_TARGET_SLOTS", threading.BoundedSemaphore(2))
    monkeypatch.setattr(app, "_TARGET_QUEUE_TIMEOUT_S", 10.0)
    app._live_completion.cache_clear()

    lock = threading.Lock()
    active = 0
    peak = 0

    def slow_generate(self, prompt, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.01)
            return "Paris."
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(OllamaClient, "generate", slow_generate)
    database = tmp_path / "dashboard_demo.sqlite3"
    sessions = ["s1", "s2", "s3", "s4"]
    per_session = 5
    pipelines = {name: app._session_pipeline(str(database), name) for name in sessions}

    async def request(name: str, index: int):
        context = ExecutionContext(
            f"trace-{name}-{index}",
            f"turn-{name}-{index}",
            f"live-{name}-{index}",
        )
        return await app._evaluate(pipelines[name], f"unique {name} {index}", context)

    turns = await asyncio.gather(
        *(request(name, index) for name in sessions for index in range(per_session))
    )

    assert len(turns) == len(sessions) * per_session
    assert all(turn.persisted for turn in turns)
    assert 1 <= peak <= 2, f"target inference concurrency peaked at {peak}"

    connection = sqlite3.connect(database)
    try:
        rows = dict(
            connection.execute(
                "SELECT conversation_id, COUNT(*) FROM stage_event GROUP BY 1"
            ).fetchall()
        )
        total = connection.execute("SELECT COUNT(*) FROM stage_event").fetchone()[0]
    finally:
        connection.close()

    expected = {
        f"live-{name}-{index}" for name in sessions for index in range(per_session)
    }
    assert set(rows) == expected
    # Every request persists the same stage set, all attributed to its own
    # conversation; no event is duplicated or lost.
    assert len(set(rows.values())) == 1, rows
    assert total == sum(rows.values())
