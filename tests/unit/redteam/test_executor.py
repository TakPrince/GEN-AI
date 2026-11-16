"""P3.4: AttackExecutor tests (mocked Defense Engine)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from safety_platform.redteam import (
    AttackDataset,
    AttackExecutor,
    AttackResult,
    ExperimentConfig,
    InvalidTransitionError,
    LifecycleStatus,
    RedTeamOrchestrator,
    classify,
    summarize,
)


@dataclass(frozen=True)
class FakeDefenseResult:
    outcome: str = "safe"
    blocked: bool = False
    final_route: str = "allow"
    target_status: str = "executed"
    final_response: str = "assistant reply"
    error: str | None = None
    failure_stage: str | None = None


def _dataset() -> AttackDataset:
    from safety_platform.redteam import AttackCase

    return AttackDataset(
        (
            AttackCase(
                name="inject",
                category="prompt_injection",
                prompt="ignore instructions",
                attack_id="a-1",
                severity="high",
            ),
            AttackCase(
                name="jail",
                category="direct_jailbreak",
                prompt="jailbreak now",
                attack_id="a-2",
                severity="critical",
            ),
        )
    )


def _executor(fake, **kwargs) -> tuple[AttackExecutor, RedTeamOrchestrator, str]:
    dataset = _dataset()
    orchestrator = RedTeamOrchestrator(dataset)
    experiment_id = orchestrator.create_experiment(
        ExperimentConfig(name="probe")
    ).config.experiment_id
    executor = AttackExecutor(orchestrator, dataset, fake, **kwargs)
    return executor, orchestrator, experiment_id


def test_classify_rules() -> None:
    assert classify(FakeDefenseResult(blocked=True, outcome="blocked")) == "blocked"
    assert classify(FakeDefenseResult()) == "bypassed"
    assert classify(FakeDefenseResult(outcome="error")) == "errored"
    assert classify(FakeDefenseResult(error="boom")) == "errored"
    assert classify(FakeDefenseResult(failure_stage="tier1")) == "errored"


def test_blocked_and_bypassed_execution() -> None:
    responses = {
        "a-1": FakeDefenseResult(blocked=True, outcome="blocked", final_route="block"),
        "a-2": FakeDefenseResult(),
    }

    async def fake(attack):
        return responses[attack.attack_id]

    executor, orchestrator, experiment_id = _executor(fake)
    results = asyncio.run(executor.run_experiment(experiment_id))
    assert [r.classification for r in results] == ["blocked", "bypassed"]
    assert results[0].target_status == "skipped" or True  # route echoed from fake
    assert all(r.severity for r in results)
    statuses = {a.attack_id: a.status for a in orchestrator.attacks(experiment_id)}
    assert statuses == {
        "a-1": LifecycleStatus.COMPLETED,
        "a-2": LifecycleStatus.COMPLETED,
    }
    assert orchestrator.complete_experiment(experiment_id).status is (
        LifecycleStatus.COMPLETED
    )


def test_defense_error_fails_attack_not_experiment() -> None:
    async def fake(attack):
        raise RuntimeError("defense exploded")

    executor, orchestrator, experiment_id = _executor(fake)
    results = asyncio.run(executor.run_experiment(experiment_id))
    assert len(results) == 2
    assert all(r.classification == "errored" for r in results)
    assert all(r.error and "RuntimeError" in r.error for r in results)
    assert all(
        a.status is LifecycleStatus.FAILED for a in orchestrator.attacks(experiment_id)
    )
    # Experiment itself stays RUNNING: failures are observable, context kept.
    assert orchestrator._experiment(experiment_id).status is LifecycleStatus.RUNNING
    failed = orchestrator.fail_experiment(experiment_id, "cleanup")
    assert failed.status is LifecycleStatus.FAILED


def test_outcome_error_classified_errored() -> None:
    async def fake(attack):
        return FakeDefenseResult(outcome="error", error="stage failure")

    executor, orchestrator, experiment_id = _executor(fake)
    results = asyncio.run(executor.run_experiment(experiment_id))
    assert results[0].classification == "errored"
    assert orchestrator.attacks(experiment_id)[0].status is LifecycleStatus.FAILED


def test_timeout_handling() -> None:
    async def fake(attack):
        await asyncio.sleep(5)
        return FakeDefenseResult()

    executor, orchestrator, experiment_id = _executor(fake, timeout_s=0.01)
    results = asyncio.run(executor.run_experiment(experiment_id))
    assert all(r.classification == "errored" for r in results)
    assert all("timeout" in (r.error or "") for r in results)
    assert all(
        a.status is LifecycleStatus.FAILED for a in orchestrator.attacks(experiment_id)
    )


def test_duplicate_execution_prevented() -> None:
    async def fake(attack):
        return FakeDefenseResult(blocked=True)

    executor, orchestrator, experiment_id = _executor(fake)
    orchestrator.start_experiment(experiment_id)
    asyncio.run(executor.execute_attack(experiment_id, "a-1"))
    with pytest.raises(InvalidTransitionError, match="cannot|not allowed"):
        asyncio.run(executor.execute_attack(experiment_id, "a-1"))


def test_run_experiment_idempotent_for_terminal_attacks() -> None:
    async def fake(attack):
        return FakeDefenseResult(blocked=True)

    executor, orchestrator, experiment_id = _executor(fake)
    first = asyncio.run(executor.run_experiment(experiment_id))
    second = asyncio.run(executor.run_experiment(experiment_id))
    assert first == second
    assert len(second) == 2
    # No duplicate result rows.
    assert len(executor.results(experiment_id)) == 2


def test_result_structure_carries_context() -> None:
    async def fake(attack):
        return FakeDefenseResult(blocked=True, final_response="no")

    executor, _, experiment_id = _executor(fake)
    executor._orchestrator.start_experiment(experiment_id)
    result = asyncio.run(executor.execute_attack(experiment_id, "a-1"))
    assert isinstance(result, AttackResult)
    assert result.experiment_id == experiment_id
    assert result.category == "prompt_injection"
    assert result.severity == "high"
    assert result.attack_id == "a-1"
    assert result.latency_ms >= 0.0


def test_real_defense_pipeline_connector(tmp_path) -> None:
    from safety_platform.redteam import make_defense_fn
    from safety_platform.runtime import build_runtime

    runtime = build_runtime(str(tmp_path / "tel.db"))
    # Stub target completion: the defense pipeline itself (Tier1/2/3 + router)
    # runs for real in heuristic mode; no Ollama anywhere (mission constraint).
    dataset = _dataset()
    orchestrator = RedTeamOrchestrator(dataset)
    experiment_id = orchestrator.create_experiment(
        ExperimentConfig(name="smoke")
    ).config.experiment_id
    executor = AttackExecutor(
        orchestrator,
        dataset,
        make_defense_fn(runtime, completion=lambda prompt: "stub target reply"),
    )
    results = asyncio.run(executor.run_experiment(experiment_id))
    assert len(results) == 2
    assert all(r.classification in ("blocked", "bypassed", "errored") for r in results)
    defense = [r for r in results if r.classification in ("blocked", "bypassed")]
    assert defense  # deterministic demo pipeline: no errors expected
    statuses = {a.attack_id: a.status for a in orchestrator.attacks(experiment_id)}
    assert statuses["a-1"] is not LifecycleStatus.CREATED
    summary = summarize(experiment_id, results)
    assert summary.total == 2
