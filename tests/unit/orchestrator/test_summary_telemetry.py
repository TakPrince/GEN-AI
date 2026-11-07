"""Task 13: run-level summaries expose actual telemetry completeness."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.orchestrator.loop import run_campaign


class _Completion:
    def __init__(self, response: str) -> None:
        self._response = response

    def __call__(self, prompt: str) -> str:
        return self._response


class _Judge:
    def __init__(self) -> None:
        self._verdicts = 0

    def judge(self, prompt: str, response: str):
        from safety_platform.evaluation.llm import JudgeResult

        self._verdicts += 1
        return JudgeResult(True, "complied", None)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    return app._pipeline(str(tmp_path / "summary-t.sqlite3"))


def _cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            "a1", "harmless attack prompt", "attack", "c", "behavior", "test", "src"
        ),
        EvaluationCase(
            "b1", "tell me about birds", "benign", "w", "behavior", "test", "src"
        ),
    ]


@pytest.mark.asyncio
async def test_singleturn_summary_carries_telemetry_block(runtime) -> None:
    records, summary = await run_campaign(
        runtime,
        _cases(),
        _Judge(),
        _Completion("cannot help with that"),  # refusal -> zero compliance
        run_id="t13-single",
        per_label=1,
        strategies=("direct",),
    )
    assert records
    block = summary["telemetry"]
    # a real, fresh pipeline: everything was persisted or dup-suppressed
    assert block["status"] in ("complete", "no_events")
    assert (
        block["events_received"] == block["events_written"] + block["events_duplicated"]
    )
    assert set(block) >= {
        "status",
        "events_received",
        "events_written",
        "events_duplicated",
        "events_dropped",
        "batches_retried",
        "events_dead_lettered",
        "events_lost",
        "unaccounted_events",
    }
