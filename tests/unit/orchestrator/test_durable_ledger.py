"""Task 15 slice 1: durable campaign ledger on real SQLite (red-first)."""

from __future__ import annotations

import sqlite3

import pytest

from safety_platform.orchestrator.durable import (
    CampaignError,
    CampaignStore,
    LeaseHeldError,
)


def _store(path, **kw) -> CampaignStore:
    return CampaignStore(str(path), **kw)


def _create(store: CampaignStore, run_id: str = "run-1") -> None:
    store.create(
        run_id,
        kind="single",
        config={"seed": 0},
        artifact_dir="out",
        plan=(("case-a", {"strategy": "direct"}), ("case-b", {"strategy": "direct"})),
    )


def test_create_persists_plan_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "db.sqlite3"
    _create(_store(path))

    reopened = _store(path)  # fresh store == fresh process
    campaign = reopened.campaign("run-1")
    units = reopened.units("run-1")

    assert campaign["status"] == "pending"
    assert campaign["config"] == {"seed": 0}
    assert [u["unit_id"] for u in units] == ["case-a", "case-b"]  # planned order
    assert all(u["status"] == "pending" for u in units)


def test_duplicate_campaign_is_rejected_and_points_to_resume(tmp_path) -> None:
    store = _store(tmp_path / "db.sqlite3")
    _create(store)
    with pytest.raises(CampaignError, match="--resume"):
        _create(store)  # same run id must never re-plan an existing campaign


def test_unit_lifecycle_is_validated(tmp_path) -> None:
    store = _store(tmp_path / "db.sqlite3")
    _create(store)
    store.acquire("run-1", owner="p1")

    store.mark_in_flight("run-1", "case-a", owner="p1")
    with pytest.raises(CampaignError):  # cannot start an in-flight unit twice
        store.mark_in_flight("run-1", "case-a", owner="p1")

    store.complete_unit("run-1", "case-a", {"outcome": "block"}, owner="p1")
    unit = {u["unit_id"]: u for u in store.units("run-1")}["case-a"]
    assert unit["status"] == "completed"
    assert unit["result"] == {"outcome": "block"}


def test_second_resume_is_refused_while_lease_is_fresh(tmp_path) -> None:
    path = tmp_path / "db.sqlite3"
    _create(_store(path))
    _store(path).acquire("run-1", owner="p1")

    with pytest.raises(LeaseHeldError):
        _store(path).acquire("run-1", owner="p2")


def test_expired_lease_allows_resume_and_takeover_is_explicit(tmp_path) -> None:
    path = tmp_path / "db.sqlite3"
    _create(_store(path))
    first = _store(path)
    first.acquire("run-1", owner="p1")
    con = sqlite3.connect(path)
    con.execute("UPDATE campaign SET lease_until_ns = 0")  # crash: lease lapses
    con.commit()
    con.close()

    _store(path).acquire("run-1", owner="p2")  # expired lease -> claimable
    assert _store(path).campaign("run-1")["owner"] == "p2"
    assert _store(path).campaign("run-1")["segments"] == 2


def test_failed_unit_write_rolls_back_unit_and_heartbeat(tmp_path) -> None:
    path = tmp_path / "db.sqlite3"
    store = _store(path)
    _create(store)
    store.acquire("run-1", owner="p1")
    store.mark_in_flight("run-1", "case-a", owner="p1")

    con = sqlite3.connect(path)  # real DB-level failure on the heartbeat write
    con.execute(
        "CREATE TRIGGER fail_hb BEFORE UPDATE ON campaign "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    con.commit()
    con.close()

    with pytest.raises(sqlite3.IntegrityError):
        store.complete_unit("run-1", "case-a", {"outcome": "block"}, owner="p1")

    # Neither the unit result nor any partial state was committed.
    fresh = _store(path)
    unit = {u["unit_id"]: u for u in fresh.units("run-1")}["case-a"]
    assert unit["status"] == "in_flight"
    assert unit["result"] is None
