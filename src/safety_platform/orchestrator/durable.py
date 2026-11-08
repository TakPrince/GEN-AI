"""Durable red-team campaign ledger (Task 15).

Campaign plans and per-unit outcomes are committed to the existing SQLite
file as they happen, so a crash keeps every confirmed result. Additive tables
only; telemetry tables are untouched.

Unit states: pending -> in_flight -> completed | uncertain. A unit is marked
in_flight before its external request and only leaves it with a persisted
result. Recovery never re-sends an in_flight request.
Campaign states: pending -> running -> completed | incomplete. "interrupted"
is derived: running with no live lease.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..evaluation.dataset import EvaluationCase
from ..evaluation.llm import ComplianceVerdict, ConversationVerdict, JudgeResult
from .attack_state import AttackState, TurnRecord
from .escalation import ESCALATIONS
from .loop import (
    AttemptOutcome,
    _CampaignRuntime,
    _defend_single,
    _judge_single,
    _Judger,
    _needs_judging,
    _pick,
    _telemetry_completeness,
    write_jsonl,
    write_summary,
)
from .metrics import aggregate, aggregate_conversations
from .multi import (
    ScenarioOutcome,
    _failed_scenario,
    run_conversation,
    write_conversations_jsonl,
)
from .multi import _Judger as _ConversationJudger
from .mutators import Mutation, mutate, selection_for

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('single', 'multi')),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'incomplete')
    ),
    config_json TEXT NOT NULL,
    artifact_dir TEXT NOT NULL,
    owner TEXT,
    lease_until_ns INTEGER NOT NULL DEFAULT 0,
    segments INTEGER NOT NULL DEFAULT 0,
    telemetry_json TEXT,
    created_ns INTEGER NOT NULL,
    updated_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_unit (
    run_id TEXT NOT NULL REFERENCES campaign(run_id),
    ordinal INTEGER NOT NULL,
    unit_id TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'in_flight', 'completed', 'uncertain')
    ),
    progress_json TEXT,
    result_json TEXT,
    updated_ns INTEGER NOT NULL,
    PRIMARY KEY (run_id, unit_id),
    UNIQUE (run_id, ordinal)
);
CREATE TABLE IF NOT EXISTS judge_attempt (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES campaign(run_id),
    unit_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    judge_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'failed', 'interrupted')
    ),
    result_json TEXT,
    error TEXT,
    started_ns INTEGER NOT NULL,
    finished_ns INTEGER,
    UNIQUE (run_id, unit_id, stage, attempt)
);
"""

_LEASE_S = 900.0
_LOG = logging.getLogger(__name__)


class CampaignError(Exception):
    """Invalid campaign state transition or unknown campaign."""


class LeaseHeldError(CampaignError):
    """Another live process owns the campaign."""


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _rollback_quietly(con: sqlite3.Connection) -> None:
    with contextlib.suppress(sqlite3.Error):  # no transaction was open
        con.execute("ROLLBACK")


class _Heartbeat:
    """Renews the lease from its own thread and connection (SQLite handles are
    thread-bound). Renewal stops when ownership is gone or the thread is stopped."""

    def __init__(
        self,
        path: str,
        run_id: str,
        owner: str,
        lease_ns: int,
        lost: threading.Event,
    ) -> None:
        self._path = path
        self._run_id = run_id
        self._owner = owner
        self._lease_ns = lease_ns
        self._interval = lease_ns / 3 / 1_000_000_000  # renew well inside the TTL
        self._lost = lost  # set when the execution loop must stop
        self._stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"safety-lease-{run_id}", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=self._interval * 3 + 1)

    def _run(self) -> None:
        con = sqlite3.connect(self._path, timeout=self._interval, isolation_level=None)
        deadline = time.time_ns() + self._lease_ns  # acquire() just set the lease
        try:
            while not self._stop.wait(self._interval):
                started = time.time_ns()
                try:
                    if not self._renew(con):
                        _LOG.error("lease for %s lost; renewal stopped", self._run_id)
                        self._lost.set()
                        return
                    deadline = started + self._lease_ns
                except sqlite3.Error as exc:
                    # Transient contention: retry next tick. Once the lease must
                    # have lapsed unrenewed, stop the loop as well.
                    _LOG.warning(
                        "lease renewal for %s failed (%s); will retry",
                        self._run_id,
                        type(exc).__name__,
                    )
                    _rollback_quietly(con)
                    if time.time_ns() >= deadline:
                        _LOG.error("lease for %s expired unrenewed", self._run_id)
                        self._lost.set()
                        return
        finally:
            con.close()

    def _renew(self, con: sqlite3.Connection) -> bool:
        now = time.time_ns()
        con.execute("BEGIN IMMEDIATE")
        cursor = con.execute(
            "UPDATE campaign SET lease_until_ns = ?, updated_ns = ?"
            " WHERE run_id = ? AND owner = ? AND lease_until_ns >= ?",
            (now + self._lease_ns, now, self._run_id, self._owner, now),
        )
        if cursor.rowcount != 1:
            con.execute("ROLLBACK")
            return False
        con.execute("COMMIT")
        return True


class CampaignStore:
    def __init__(self, path: str, *, lease_s: float = _LEASE_S) -> None:
        # Autocommit connection; every multi-statement write uses _tx().
        self._path = path
        self._con = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self._con.executescript(_SCHEMA)
        self._lease_ns = int(lease_s * 1_000_000_000)
        self.lost = threading.Event()  # set by the heartbeat on lease loss

    def heartbeat(self, run_id: str, *, owner: str) -> _Heartbeat:
        return _Heartbeat(self._path, run_id, owner, self._lease_ns, self.lost)

    def raise_if_lost(self, run_id: str) -> None:
        """Once the heartbeat signals loss, no write from this worker may proceed."""
        if self.lost.is_set():
            raise CampaignError(f"lost lease on campaign {run_id}")

    def close(self) -> None:
        self._con.close()

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._con.execute("BEGIN IMMEDIATE")
        try:
            yield self._con
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        self._con.execute("COMMIT")

    def create(
        self,
        run_id: str,
        *,
        kind: str,
        config: dict[str, Any],
        artifact_dir: str,
        plan: Sequence[tuple[str, dict[str, Any]]],
    ) -> None:
        now = time.time_ns()
        if self._con.execute(
            "SELECT 1 FROM campaign WHERE run_id = ?", (run_id,)
        ).fetchone():
            raise CampaignError(f"campaign {run_id} already exists; use --resume")
        with self._tx() as con:
            con.execute(
                "INSERT INTO campaign (run_id, kind, status, config_json,"
                " artifact_dir, created_ns, updated_ns)"
                " VALUES (?, ?, 'pending', ?, ?, ?, ?)",
                (run_id, kind, _dump(config), artifact_dir, now, now),
            )
            con.executemany(
                "INSERT INTO campaign_unit (run_id, ordinal, unit_id, plan_json,"
                " status, updated_ns) VALUES (?, ?, ?, ?, 'pending', ?)",
                [
                    (run_id, index, unit_id, _dump(item), now)
                    for index, (unit_id, item) in enumerate(plan)
                ],
            )

    def acquire(self, run_id: str, *, owner: str, takeover: bool = False) -> None:
        now = time.time_ns()
        with self._tx() as con:
            cursor = con.execute(
                "UPDATE campaign SET status = 'running', owner = ?,"
                " lease_until_ns = ?, segments = segments + 1, updated_ns = ?"
                " WHERE run_id = ? AND status IN ('pending', 'running')"
                " AND (owner IS NULL OR lease_until_ns < ? OR ?)",
                (owner, now + self._lease_ns, now, run_id, now, int(takeover)),
            )
            if cursor.rowcount == 1:
                return
            row = con.execute(
                "SELECT status, owner FROM campaign WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise CampaignError(f"unknown campaign {run_id}")
        if row[0] not in ("pending", "running"):
            raise CampaignError(f"campaign {run_id} is already {row[0]}")
        raise LeaseHeldError(f"campaign {run_id} is held by {row[1]}")

    def release(self, run_id: str, *, owner: str) -> None:
        with self._tx() as con:
            con.execute(
                "UPDATE campaign SET owner = NULL WHERE run_id = ? AND owner = ?",
                (run_id, owner),
            )

    def finish(
        self,
        run_id: str,
        *,
        owner: str,
        status: str,
        telemetry: dict[str, Any] | None,
    ) -> None:
        self.raise_if_lost(run_id)  # a stale worker must never mark completion
        now = time.time_ns()
        with self._tx() as con:
            cursor = con.execute(
                "UPDATE campaign SET status = ?, telemetry_json = ?, owner = NULL,"
                " updated_ns = ? WHERE run_id = ? AND owner = ?"
                " AND lease_until_ns >= ?",
                (status, _dump(telemetry), now, run_id, owner, now),
            )
            if cursor.rowcount != 1:
                raise CampaignError(f"lost lease on campaign {run_id}")

    def _move(
        self,
        run_id: str,
        unit_id: str,
        *,
        owner: str,
        allowed_from: tuple[str, ...],
        to: str,
        progress: Any = None,
        result: Any = None,
    ) -> None:
        self.raise_if_lost(run_id)
        now = time.time_ns()
        marks = ",".join("?" * len(allowed_from))
        with self._tx() as con:
            cursor = con.execute(
                "UPDATE campaign_unit SET status = ?,"
                " progress_json = COALESCE(?, progress_json),"
                " result_json = COALESCE(?, result_json), updated_ns = ?"
                f" WHERE run_id = ? AND unit_id = ? AND status IN ({marks})",
                (
                    to,
                    None if progress is None else _dump(progress),
                    None if result is None else _dump(result),
                    now,
                    run_id,
                    unit_id,
                    *allowed_from,
                ),
            )
            if cursor.rowcount != 1:
                raise CampaignError(
                    f"unit {unit_id} of {run_id} is not in {allowed_from}"
                )
            # Fence, last: a write is valid only while this owner's lease is live.
            # Failing here rolls back the unit change too.
            beat = con.execute(
                "UPDATE campaign SET lease_until_ns = ?, updated_ns = ?"
                " WHERE run_id = ? AND owner = ? AND lease_until_ns >= ?",
                (now + self._lease_ns, now, run_id, owner, now),
            )
            if beat.rowcount != 1:
                raise CampaignError(f"lost lease on campaign {run_id}")

    def mark_in_flight(self, run_id: str, unit_id: str, *, owner: str) -> None:
        self._move(
            run_id, unit_id, owner=owner, allowed_from=("pending",), to="in_flight"
        )

    def save_progress(
        self, run_id: str, unit_id: str, progress: Any, *, owner: str
    ) -> None:
        # pending: a response was persisted before judging (Task 17 checkpoint).
        self._move(
            run_id,
            unit_id,
            owner=owner,
            allowed_from=("in_flight", "pending"),
            to="pending",
            progress=progress,
        )

    def complete_unit(
        self, run_id: str, unit_id: str, result: Any, *, owner: str
    ) -> None:
        self._move(
            run_id,
            unit_id,
            owner=owner,
            allowed_from=("in_flight", "pending"),
            to="completed",
            result=result,
        )

    def mark_uncertain(
        self, run_id: str, unit_id: str, result: Any, *, owner: str
    ) -> None:
        self._move(
            run_id,
            unit_id,
            owner=owner,
            allowed_from=("in_flight",),
            to="uncertain",
            result=result,
        )

    def _fence(
        self, con: sqlite3.Connection, run_id: str, owner: str, now: int
    ) -> None:
        self.raise_if_lost(run_id)
        beat = con.execute(
            "UPDATE campaign SET lease_until_ns = ?, updated_ns = ?"
            " WHERE run_id = ? AND owner = ? AND lease_until_ns >= ?",
            (now + self._lease_ns, now, run_id, owner, now),
        )
        if beat.rowcount != 1:
            raise CampaignError(f"lost lease on campaign {run_id}")

    def judge_begin(
        self, run_id: str, unit_id: str, stage: str, *, owner: str, judge_id: str
    ) -> str:
        now = time.time_ns()
        with self._tx() as con:
            (attempt,) = con.execute(
                "SELECT COUNT(*) + 1 FROM judge_attempt"
                " WHERE run_id = ? AND unit_id = ? AND stage = ?",
                (run_id, unit_id, stage),
            ).fetchone()
            attempt_id = f"{run_id}:{unit_id}:{stage}:{attempt}"
            con.execute(
                "INSERT INTO judge_attempt (attempt_id, run_id, unit_id, stage,"
                " attempt, judge_id, status, started_ns)"
                " VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
                (attempt_id, run_id, unit_id, stage, attempt, judge_id, now),
            )
            self._fence(con, run_id, owner, now)
        return attempt_id

    def judge_end(
        self,
        run_id: str,
        attempt_id: str,
        *,
        owner: str,
        status: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        now = time.time_ns()
        with self._tx() as con:
            cursor = con.execute(
                "UPDATE judge_attempt SET status = ?, result_json = ?, error = ?,"
                " finished_ns = ? WHERE attempt_id = ? AND status = 'running'",
                (
                    status,
                    None if result is None else _dump(result),
                    error,
                    now,
                    attempt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise CampaignError(f"judge attempt {attempt_id} is not running")
            self._fence(con, run_id, owner, now)

    def interrupt_judges(self, run_id: str, *, owner: str) -> None:
        """Attempts still running belong to a dead process: keep them, mark them."""
        now = time.time_ns()
        with self._tx() as con:
            con.execute(
                "UPDATE judge_attempt SET status = 'interrupted', finished_ns = ?"
                " WHERE run_id = ? AND status = 'running'",
                (now, run_id),
            )
            self._fence(con, run_id, owner, now)

    def completed_judge(
        self, run_id: str, unit_id: str, stage: str, judge_id: str
    ) -> dict[str, Any] | None:
        """Latest completed verdict of this exact judge identity, if any."""
        row = self._con.execute(
            "SELECT attempt_id, result_json FROM judge_attempt WHERE run_id = ?"
            " AND unit_id = ? AND stage = ? AND judge_id = ? AND status = 'completed'"
            " ORDER BY attempt DESC LIMIT 1",
            (run_id, unit_id, stage, judge_id),
        ).fetchone()
        if row is None:
            return None
        return {"attempt_id": row[0], "result": _load(row[1])}

    def judge_attempts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT attempt_id, unit_id, stage, attempt, judge_id, status,"
            " result_json, error FROM judge_attempt WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        attempts = []
        for (
            attempt_id,
            unit_id,
            stage,
            attempt,
            judge_id,
            status,
            result,
            error,
        ) in rows:
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "unit_id": unit_id,
                    "stage": stage,
                    "attempt": attempt,
                    "judge_id": judge_id,
                    "status": status,
                    "result": _load(result),
                    "error": error,
                }
            )
        return attempts

    def judge_summary(self, run_id: str, selected: list[str]) -> dict[str, Any]:
        """Status counts, plus the attempts that unit results reference."""
        attempts = self.judge_attempts(run_id)
        counts = {
            name: sum(1 for a in attempts if a["status"] == name)
            for name in ("running", "completed", "failed", "interrupted")
        }
        return {"counts": counts, "selected": sorted(selected)}

    def list_runs(self) -> list[dict[str, Any]]:
        """Read-only campaign listing for dashboards; nothing else reads this."""
        rows = self._con.execute(
            "SELECT run_id, kind, status, config_json, artifact_dir, created_ns,"
            " updated_ns FROM campaign ORDER BY created_ns"
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "kind": row[1],
                "status": row[2],
                "config": _load(row[3]),
                "artifact_dir": row[4],
                "created_ns": row[5],
                "updated_ns": row[6],
            }
            for row in rows
        ]

    def campaign(self, run_id: str) -> dict[str, Any]:
        row = self._con.execute(
            "SELECT kind, status, config_json, artifact_dir, owner,"
            " lease_until_ns, segments, telemetry_json FROM campaign WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise CampaignError(f"unknown campaign {run_id}")
        return {
            "run_id": run_id,
            "kind": row[0],
            "status": row[1],
            "config": json.loads(row[2]),
            "artifact_dir": row[3],
            "owner": row[4],
            "lease_until_ns": row[5],
            "segments": row[6],
            "telemetry": _load(row[7]),
        }

    def units(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT unit_id, status, plan_json, progress_json, result_json"
            " FROM campaign_unit WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        return [
            {
                "unit_id": unit_id,
                "status": status,
                "plan": json.loads(plan),
                "progress": _load(progress),
                "result": _load(result),
            }
            for unit_id, status, plan, progress, result in rows
        ]


@dataclass(frozen=True, slots=True)
class CampaignResult:
    run_id: str
    status: str
    records: tuple[Any, ...]
    summary: dict[str, Any]
    paths: tuple[str, str]  # (records JSONL, summary JSON)


def describe_campaign(store: CampaignStore, run_id: str) -> dict[str, Any]:
    """Read-only progress. A running campaign whose lease lapsed is interrupted."""
    campaign = store.campaign(run_id)
    units = store.units(run_id)
    counts = {
        name: sum(1 for u in units if u["status"] == name)
        for name in ("pending", "in_flight", "completed", "uncertain")
    }
    counts["total"] = len(units)
    live = campaign["owner"] is not None and campaign["lease_until_ns"] > time.time_ns()
    if campaign["status"] == "running":
        label = "running" if live else "interrupted"
    else:
        label = campaign["status"]
    return {
        "run_id": run_id,
        "kind": campaign["kind"],
        "label": label,
        "segments": campaign["segments"],
        "units": counts,
    }


def _owner() -> str:
    return f"{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _model_metadata(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((key, value) for key, value in config["model_metadata"])


def _single_plan(
    cases: Sequence[EvaluationCase],
    *,
    run_id: str,
    per_label: int,
    strategies: Sequence[str],
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Same selection and mutation sequence as loop.run_campaign, persisted up front."""
    rng = random.Random(f"{run_id}|{tuple(strategies)}|{seed}")
    selected = _pick(cases, "attack", per_label, rng) + _pick(
        cases, "benign", per_label, rng
    )
    picks = selection_for(tuple(selected), tuple(strategies), seed)
    plan = []
    for index, (case, strategy) in enumerate(zip(selected, picks, strict=False)):
        mutation = mutate(case.prompt, strategy, seed=seed + index)
        plan.append(
            (
                case.case_id,
                {
                    "case": asdict(case),
                    "strategy": mutation.strategy,
                    "original_prompt": mutation.original_prompt,
                    "executed_prompt": mutation.executed_prompt,
                    "seed": seed + index,
                },
            )
        )
    return plan


def _uncertain_attempt(
    run_id: str, config: dict[str, Any], unit: dict[str, Any]
) -> AttemptOutcome:
    """Unit whose request was in flight at a crash: no verdict, not a success."""
    plan = unit["plan"]
    case = EvaluationCase(**plan["case"])
    is_attack = case.label == "attack"
    return AttemptOutcome(
        run_id=run_id,
        timestamp=config["timestamp"],
        case_id=case.case_id,
        label=case.label,
        attack_category=case.category if is_attack else None,
        original_prompt=plan["original_prompt"],
        executed_prompt=plan["executed_prompt"],
        mutation_strategy=plan["strategy"],
        outcome="uncertain",
        blocked=False,
        final_route="uncertain",
        target_status="unknown",
        raw_response=None,
        final_response=None,
        judge_complied=None,
        judge_reason=None,
        judge_error=None,
        raw_judge_complied=None,
        raw_judge_error=None,
        latency_ms=0.0,
        execution_error="uncertain_in_flight",
        failure_stage=None,
        trace_id=f"rt-{run_id}-{case.case_id}",
        model_metadata=_model_metadata(config),
        partial_compliance=None,
    )


def _decode_attempt(data: dict[str, Any]) -> AttemptOutcome:
    fields = {k: v for k, v in data.items() if k != "judge_refs"}
    return AttemptOutcome(
        **{
            **fields,
            "model_metadata": tuple(tuple(p) for p in fields["model_metadata"]),
        }
    )


def _judge_id(config: dict[str, Any]) -> str:
    return dict(_model_metadata(config)).get("judge", "unknown")


def _judge_identity(judge: Any, config: dict[str, Any]) -> str:
    """The injected judge's own identity; the configured name only as a fallback."""
    identity = getattr(judge, "identity", None)
    return identity if identity else _judge_id(config)


def _judge_refs(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    return [ref for ref in result.get("judge_refs", {}).values() if ref]


class _JudgeRecorder:
    """Judge proxy: every real invocation gets its own recorded attempt. A stage that
    already has a completed attempt reuses it, so a retry never replaces a verdict."""

    def __init__(
        self,
        store: CampaignStore,
        run_id: str,
        unit_id: str,
        owner: str,
        judge: Any,
        judge_id: str,
        *,
        stage: str,
        next_stage: str | None = None,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._unit_id = unit_id
        self._owner = owner
        self._judge = judge
        self._judge_id = judge_id
        self.stage = stage
        self._next_stage = next_stage
        self.selected: dict[str, str | None] = {}  # stage -> attempt used, or None

    def judge(self, prompt: str, response: str) -> JudgeResult:
        stage = self._take_stage()
        done = self._reusable(stage)
        if done is not None:
            return JudgeResult(**done["result"])
        attempt_id = self._begin(stage)
        try:
            verdict = self._judge.judge(prompt, response)
        except Exception as exc:
            self._end(attempt_id, stage, "failed", error=type(exc).__name__)
            raise
        self._end(
            attempt_id,
            stage,
            "failed" if verdict.error else "completed",
            result=asdict(verdict),
            error=verdict.error,
        )
        return verdict

    def judge_conversation(
        self, objective: str, transcript: str
    ) -> ConversationVerdict:
        done = self._reusable("conversation")
        if done is not None:
            found = done["result"]
            return ConversationVerdict(
                ComplianceVerdict(found["verdict"]), found["reason"], found["error"]
            )
        attempt_id = self._begin("conversation")
        try:
            verdict = self._judge.judge_conversation(objective, transcript)
        except Exception as exc:
            self._end(attempt_id, "conversation", "failed", error=type(exc).__name__)
            raise
        result = {
            "verdict": verdict.verdict.value,
            "reason": verdict.reason,
            "error": verdict.error,
        }
        self._end(
            attempt_id,
            "conversation",
            "failed" if verdict.error else "completed",
            result=result,
            error=verdict.error,
        )
        return verdict

    def _reusable(self, stage: str) -> dict[str, Any] | None:
        done = self._store.completed_judge(
            self._run_id, self._unit_id, stage, self._judge_id
        )
        if done is not None:
            self.selected[stage] = done["attempt_id"]
        return done

    def _take_stage(self) -> str:
        stage = self.stage
        if self._next_stage is not None:
            self.stage, self._next_stage = self._next_stage, None
        return stage

    def _begin(self, stage: str) -> str:
        return self._store.judge_begin(
            self._run_id,
            self._unit_id,
            stage,
            owner=self._owner,
            judge_id=self._judge_id,
        )

    def _end(self, attempt_id: str, stage: str, status: str, **fields: Any) -> None:
        self._store.judge_end(
            self._run_id, attempt_id, owner=self._owner, status=status, **fields
        )
        self.selected[stage] = attempt_id if status == "completed" else None


async def _execute_single_units(
    store: CampaignStore,
    run_id: str,
    config: dict[str, Any],
    runtime: _CampaignRuntime,
    judge: _Judger,
    completion: Callable[[str], str],
    owner: str,
) -> None:
    for unit in store.units(run_id):
        store.raise_if_lost(run_id)  # no new unit once ownership is gone
        if unit["status"] != "pending":
            continue  # confirmed or uncertain work is never executed again
        plan = unit["plan"]
        mutation = Mutation(
            strategy=plan["strategy"],
            original_prompt=plan["original_prompt"],
            executed_prompt=plan["executed_prompt"],
        )
        if unit["progress"] is None:
            store.mark_in_flight(run_id, unit["unit_id"], owner=owner)
            record = await _defend_single(
                runtime,
                EvaluationCase(**plan["case"]),
                mutation,
                completion,
                run_id=run_id,
                timestamp=config["timestamp"],
                seed=plan["seed"],
                model_metadata=_model_metadata(config),
            )
            if _needs_judging(record):
                # Persist the received response before judging: a crash from here
                # resumes judging without resending the target request.
                store.save_progress(
                    run_id,
                    unit["unit_id"],
                    {"stage": "responded", "record": asdict(record)},
                    owner=owner,
                )
        else:  # a response survived an earlier crash: judge it, never resend
            record = _decode_attempt(unit["progress"]["record"])
        recorder = _JudgeRecorder(
            store,
            run_id,
            unit["unit_id"],
            owner,
            judge,
            _judge_identity(judge, config),
            stage="final",
            next_stage="raw",  # _judge_single judges the final, then the raw response
        )
        record = _judge_single(record, mutation, recorder)
        store.complete_unit(
            run_id,
            unit["unit_id"],
            {**asdict(record), "judge_refs": recorder.selected},
            owner=owner,
        )


def _multi_config(
    timestamp: str,
    *,
    max_turns: int,
    escalations: Sequence[str],
    seed: int,
    per_label: int,
    model_metadata: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "max_turns": max_turns,
        "escalations": list(escalations),
        "seed": seed,
        "per_label": per_label,
        "model_metadata": [list(pair) for pair in model_metadata],
    }


def _multi_plan(
    cases: Sequence[EvaluationCase],
    *,
    run_id: str,
    per_label: int,
    escalations: Sequence[str],
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Same selection and escalation draw as run_conversation_campaign."""
    rng = random.Random(f"{run_id}|{tuple(escalations)}|{seed}")
    selected = _pick(cases, "attack", per_label, rng)
    assign = tuple(escalations[rng.randrange(len(escalations))] for _ in selected)
    return [
        (
            case.case_id,
            {"case": asdict(case), "escalation": escalation, "seed": seed + index},
        )
        for index, (case, escalation) in enumerate(zip(selected, assign, strict=False))
    ]


def _progress_of(state: AttackState) -> dict[str, Any]:
    """Checkpoint: everything needed to continue, including termination."""
    return {
        "status": state.status,
        "termination_reason": state.termination_reason,
        "turns": [asdict(turn) for turn in state.turns],
    }


def _state_from_progress(
    run_id: str,
    plan: dict[str, Any],
    config: dict[str, Any],
    progress: dict[str, Any] | None,
) -> AttackState | None:
    if progress is None:
        return None
    case = EvaluationCase(**plan["case"])
    return AttackState(
        run_id=run_id,
        conversation_id=f"{run_id}-{case.case_id}",
        case_id=case.case_id,
        objective=case.prompt,
        attack_category=case.category,
        max_turns=config["max_turns"],
        escalation=plan["escalation"],
        turns=tuple(TurnRecord(**turn) for turn in progress["turns"]),
        status=progress["status"],
        termination_reason=progress["termination_reason"],
    )


def _decode_scenario(data: dict[str, Any]) -> ScenarioOutcome:
    fields = {k: v for k, v in data.items() if k != "judge_refs"}
    return ScenarioOutcome(
        **{
            **fields,
            "turns": tuple(TurnRecord(**turn) for turn in fields["turns"]),
            "model_metadata": tuple(tuple(p) for p in fields["model_metadata"]),
        }
    )


def _uncertain_scenario(
    run_id: str, config: dict[str, Any], unit: dict[str, Any]
) -> ScenarioOutcome:
    """Turn in flight at a crash: confirmed turns kept, no verdict."""
    plan = unit["plan"]
    case = EvaluationCase(**plan["case"])
    progress = unit["progress"] or {"turns": []}
    turns = tuple(TurnRecord(**turn) for turn in progress["turns"])
    return ScenarioOutcome(
        run_id=run_id,
        timestamp=config["timestamp"],
        conversation_id=f"{run_id}-{case.case_id}",
        case_id=case.case_id,
        attack_category=case.category,
        objective=case.prompt,
        escalation=plan["escalation"],
        max_turns=config["max_turns"],
        turns=turns,
        status="uncertain",
        termination_reason="uncertain_in_flight",
        conversation_verdict=ComplianceVerdict.UNSCORABLE.value,
        conversation_reason="",
        conversation_error="uncertain_in_flight",
        model_metadata=_model_metadata(config),
        total_latency_ms=sum(turn.latency_ms for turn in turns),
    )


def _checkpoint_hooks(
    store: CampaignStore,
    run_id: str,
    unit_id: str,
    owner: str,
    recorder: _JudgeRecorder,
):
    def on_turn_start(turn: int) -> None:
        recorder.stage = f"turn:{turn}"
        store.mark_in_flight(run_id, unit_id, owner=owner)

    def on_response(state: AttackState, record: TurnRecord) -> None:
        # Response persisted before judging; the unit stays pending and resumable.
        progress = {**_progress_of(state), "pending": asdict(record)}
        store.save_progress(run_id, unit_id, progress, owner=owner)

    def on_turn_done(state: AttackState) -> None:
        store.save_progress(run_id, unit_id, _progress_of(state), owner=owner)

    return on_turn_start, on_response, on_turn_done


async def _execute_multi_units(
    store: CampaignStore,
    run_id: str,
    config: dict[str, Any],
    runtime: _CampaignRuntime,
    judge: _ConversationJudger,
    completion: Callable[[str], str],
    owner: str,
) -> None:
    for unit in store.units(run_id):
        store.raise_if_lost(run_id)  # no new unit once ownership is gone
        if unit["status"] != "pending":
            continue  # finished or uncertain conversations are never resumed
        unit_id, plan = unit["unit_id"], unit["plan"]
        case = EvaluationCase(**plan["case"])
        pending = (unit["progress"] or {}).get("pending")
        recorder = _JudgeRecorder(
            store,
            run_id,
            unit_id,
            owner,
            judge,
            _judge_identity(judge, config),
            stage=f"turn:{pending['turn_number']}" if pending else "",
        )
        on_start, on_response, on_done = _checkpoint_hooks(
            store, run_id, unit_id, owner, recorder
        )
        try:
            outcome = await run_conversation(
                runtime,
                case,
                recorder,
                completion,
                run_id=run_id,
                escalation=plan["escalation"],
                max_turns=config["max_turns"],
                seed=plan["seed"],
                model_metadata=_model_metadata(config),
                reference_time=config["timestamp"],
                resume_state=_state_from_progress(
                    run_id, plan, config, unit["progress"]
                ),
                resume_pending=None if pending is None else TurnRecord(**pending),
                on_turn_start=on_start,
                on_response=on_response,
                on_turn_done=on_done,
            )
        except (
            Exception
        ) as exc:  # same confirmed-failure semantics as the campaign loop
            outcome = _failed_scenario(
                run_id,
                case,
                plan["escalation"],
                config["timestamp"],
                config["max_turns"],
                _model_metadata(config),
                exc,
            )
        store.complete_unit(
            run_id,
            unit_id,
            {**asdict(outcome), "judge_refs": recorder.selected},
            owner=owner,
        )


class _Kind:
    def __init__(self, prefix, writer, aggregate_fn, decode, uncertain, execute):
        self.prefix = prefix
        self.writer = writer
        self.aggregate_fn = aggregate_fn
        self.decode = decode
        self.uncertain = uncertain
        self.execute = execute


_KINDS = {
    "single": _Kind(
        "redteam-singleturn",
        write_jsonl,
        aggregate,
        _decode_attempt,
        _uncertain_attempt,
        _execute_single_units,
    ),
    "multi": _Kind(
        "redteam-multiturn",
        write_conversations_jsonl,
        aggregate_conversations,
        _decode_scenario,
        _uncertain_scenario,
        _execute_multi_units,
    ),
}


async def _drive(
    store: CampaignStore,
    run_id: str,
    runtime: _CampaignRuntime,
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    takeover: bool,
) -> CampaignResult:
    campaign = store.campaign(run_id)
    if campaign["status"] in ("completed", "incomplete"):
        return _export(store, run_id, campaign["artifact_dir"])
    owner = _owner()
    store.acquire(run_id, owner=owner, takeover=takeover)
    heartbeat = store.heartbeat(run_id, owner=owner)
    heartbeat.start()
    try:
        campaign = store.campaign(run_id)
        config = campaign["config"]
        kind = _KINDS[campaign["kind"]]
        for unit in store.units(run_id):
            if unit["status"] == "in_flight":  # left by a dead process
                store.mark_uncertain(
                    run_id,
                    unit["unit_id"],
                    asdict(kind.uncertain(run_id, config, unit)),
                    owner=owner,
                )
        store.interrupt_judges(run_id, owner=owner)  # attempts a dead process began
        await kind.execute(store, run_id, config, runtime, judge, completion, owner)
        units = store.units(run_id)
        status = (
            "completed"
            if all(u["status"] == "completed" for u in units)
            else "incomplete"
        )
        consumer = getattr(runtime, "consumer", None)
        if campaign["segments"] == 1 and consumer is not None:
            telemetry = _telemetry_completeness(consumer)
        else:
            # Counters from an earlier process died with it; never reconstruct them.
            telemetry = {"status": "unavailable", "reason": "resumed_after_restart"}
        store.finish(run_id, owner=owner, status=status, telemetry=telemetry)
    except BaseException:
        store.release(run_id, owner=owner)
        raise
    finally:
        heartbeat.stop()
    return _export(store, run_id, campaign["artifact_dir"])


def _export(store: CampaignStore, run_id: str, artifact_dir: str) -> CampaignResult:
    campaign = store.campaign(run_id)
    units = store.units(run_id)
    kind = _KINDS[campaign["kind"]]
    records = tuple(kind.decode(u["result"]) for u in units if u["result"] is not None)
    counts = {
        name: sum(1 for u in units if u["status"] == name)
        for name in ("completed", "uncertain", "pending")
    }
    counts["total"] = len(units)
    summary = kind.aggregate_fn(list(records), run_id=run_id)
    summary["telemetry"] = campaign["telemetry"]
    summary["campaign"] = {
        "run_id": run_id,
        "kind": campaign["kind"],
        "status": campaign["status"],
        "segments": campaign["segments"],
        "units": counts,
        "judge_attempts": store.judge_summary(
            run_id, [ref for unit in units for ref in _judge_refs(unit["result"])]
        ),
    }
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{kind.prefix}-{run_id}.jsonl"
    summary_path = out / f"{kind.prefix}-{run_id}-summary.json"
    kind.writer(list(records), jsonl)
    write_summary(summary, summary_path)
    return CampaignResult(
        run_id,
        campaign["status"],
        records,
        summary,
        (str(jsonl), str(summary_path)),
    )


async def start_single(
    db: str,
    runtime: _CampaignRuntime,
    cases: Sequence[EvaluationCase],
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    artifact_dir: str,
    per_label: int,
    strategies: Sequence[str],
    seed: int,
    model_metadata: tuple[tuple[str, str], ...],
    reference_time: str | None = None,
    lease_s: float = _LEASE_S,
) -> CampaignResult:
    plan = _single_plan(
        cases, run_id=run_id, per_label=per_label, strategies=strategies, seed=seed
    )
    config = {
        "timestamp": reference_time or datetime.now(UTC).isoformat(),
        "seed": seed,
        "per_label": per_label,
        "strategies": list(strategies),
        "model_metadata": [list(pair) for pair in model_metadata],
    }
    store = CampaignStore(db, lease_s=lease_s)
    try:
        store.create(
            run_id, kind="single", config=config, artifact_dir=artifact_dir, plan=plan
        )
        return await _drive(store, run_id, runtime, judge, completion, takeover=False)
    finally:
        store.close()


async def start_multi(
    db: str,
    runtime: _CampaignRuntime,
    cases: Sequence[EvaluationCase],
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    artifact_dir: str,
    per_label: int,
    max_turns: int,
    escalations: Sequence[str],
    seed: int,
    model_metadata: tuple[tuple[str, str], ...],
    reference_time: str | None = None,
    lease_s: float = _LEASE_S,
) -> CampaignResult:
    escalations = tuple(escalations)
    for escalation in escalations:
        if escalation not in ESCALATIONS:  # fail before any plan is persisted
            raise ValueError(f"unknown escalation strategy: {escalation!r}")
    plan = _multi_plan(
        cases, run_id=run_id, per_label=per_label, escalations=escalations, seed=seed
    )
    config = _multi_config(
        reference_time or datetime.now(UTC).isoformat(),
        max_turns=max_turns,
        escalations=escalations,
        seed=seed,
        per_label=per_label,
        model_metadata=model_metadata,
    )
    store = CampaignStore(db, lease_s=lease_s)
    try:
        store.create(
            run_id, kind="multi", config=config, artifact_dir=artifact_dir, plan=plan
        )
        return await _drive(store, run_id, runtime, judge, completion, takeover=False)
    finally:
        store.close()


async def resume_campaign(
    db: str,
    runtime: _CampaignRuntime,
    judge: _Judger,
    completion: Callable[[str], str],
    *,
    run_id: str,
    takeover: bool = False,
    lease_s: float = _LEASE_S,
) -> CampaignResult:
    store = CampaignStore(db, lease_s=lease_s)
    try:
        return await _drive(
            store, run_id, runtime, judge, completion, takeover=takeover
        )
    finally:
        store.close()


__all__ = [
    "CampaignError",
    "CampaignResult",
    "CampaignStore",
    "LeaseHeldError",
    "describe_campaign",
    "resume_campaign",
    "start_multi",
    "start_single",
]
