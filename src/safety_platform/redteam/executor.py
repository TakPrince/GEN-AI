"""Attack Executor: runs orchestrator-selected attacks through the Defense Engine.

The executor owns no defense logic: prompts go through `execute_defense`
via `make_defense_fn` (or any injected async/async defense callable for
mocked tests). Per-attack failures never crash the experiment: they are
recorded as classification "errored", the attack is failed with the
error, and the run continues.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .dataset import AttackDataset
from .orchestrator import RedTeamOrchestrator
from .state import AttackState, LifecycleStatus

Classification = str  # "bypassed" | "blocked" | "errored"
DefenseFn = Callable[[AttackState], Any]


class _DefenseRuntime(Protocol):
    """Structural subset of the Defense Engine runtime used per turn."""


@dataclass(frozen=True, slots=True)
class AttackResult:
    """Outcome of one attack execution; consumed by the Result Analyzer."""

    experiment_id: str
    attack_id: str
    category: str
    severity: str | None
    classification: Classification
    blocked: bool
    outcome: str
    final_route: str
    target_status: str
    final_response: str
    latency_ms: float
    error: str | None = None
    failure_stage: str | None = None


def make_defense_fn(
    runtime: Any,
    *,
    completion: Callable[[str], str],
) -> DefenseFn:
    """Real-engine connector: one attack prompt through `execute_defense`."""
    from ..config.registry import ExecutionContext
    from ..execution import DefenseRequest, execute_defense

    async def defense_fn(attack: AttackState) -> Any:
        request = DefenseRequest(
            prompt=attack.prompt,
            request_id=attack.attack_id,
            attack_id=attack.attack_id,
            attack_category=attack.attack_type,
            metadata=(("experiment_id", attack.experiment_id),),
        )
        context = ExecutionContext(
            trace_id=uuid.uuid4().hex,
            turn_id=uuid.uuid4().hex,
            conversation_id=f"redteam-{attack.experiment_id}",
            user_input=attack.prompt,
        )
        return await execute_defense(runtime, request, context, completion=completion)

    return defense_fn


def classify(result: Any) -> Classification:
    """Explicit rules: errors first, then blocked, otherwise bypassed."""
    if result.error is not None or getattr(result, "failure_stage", None) is not None:
        return "errored"
    if getattr(result, "outcome", "") == "error":
        return "errored"
    if result.blocked:
        return "blocked"
    return "bypassed"


class AttackExecutor:
    def __init__(
        self,
        orchestrator: RedTeamOrchestrator,
        dataset: AttackDataset,
        defense_fn: DefenseFn,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._dataset = dataset
        self._defense_fn = defense_fn
        self._timeout_s = timeout_s
        self._results: dict[str, list[AttackResult]] = {}

    def results(self, experiment_id: str, /) -> tuple[AttackResult, ...]:
        return tuple(self._results.get(experiment_id, ()))

    async def execute_attack(
        self, experiment_id: str, attack_id: str, /
    ) -> AttackResult:
        attack = self._orchestrator.begin_attack(experiment_id, attack_id)
        if attack.status is not LifecycleStatus.RUNNING:
            raise RuntimeError(
                f"attack {attack_id} is {attack.status.value}; cannot execute"
            )
        started = time.perf_counter()
        try:
            if self._timeout_s is not None:
                outcome = await asyncio.wait_for(
                    self._defense_fn(attack), timeout=self._timeout_s
                )
            else:
                outcome = self._defense_fn(attack)
                if asyncio.iscoroutine(outcome):
                    outcome = await outcome
        except TimeoutError:
            outcome = None
            error = f"timeout after {self._timeout_s}s"
            latency = time.perf_counter() - started
            result = self._record(
                experiment_id, attack, "errored", error, latency * 1000.0
            )
            self._orchestrator.fail_attack(experiment_id, attack_id, error)
            return result
        except Exception as exc:  # per-attack errors never kill the run
            error = f"{type(exc).__name__}: {exc}"
            result = self._record(
                experiment_id,
                attack,
                "errored",
                error,
                (time.perf_counter() - started) * 1000.0,
            )
            self._orchestrator.fail_attack(experiment_id, attack_id, error)
            return result
        result = self._record(
            experiment_id,
            attack,
            classify(outcome),
            None,
            (time.perf_counter() - started) * 1000.0,
            defense=outcome,
        )
        if result.classification == "errored":
            self._orchestrator.fail_attack(
                experiment_id, attack_id, result.error or "defense error"
            )
        else:
            self._orchestrator.complete_attack(experiment_id, attack_id)
        return result

    async def run_experiment(self, experiment_id: str, /) -> tuple[AttackResult, ...]:
        """Execute every non-terminal attack in deterministic order."""
        experiment = self._orchestrator._experiment(experiment_id)
        if experiment.status is LifecycleStatus.CREATED:
            self._orchestrator.start_experiment(experiment_id)
        for attack in self._orchestrator.pending_attacks(experiment_id):
            if attack.status is LifecycleStatus.CREATED:
                await self.execute_attack(experiment_id, attack.attack_id)
        return self.results(experiment_id)

    def _record(
        self,
        experiment_id: str,
        attack: AttackState,
        classification: Classification,
        error: str | None,
        latency_ms: float,
        *,
        defense: Any = None,
    ) -> AttackResult:
        case = self._dataset.get(attack.attack_id)
        result = AttackResult(
            experiment_id=experiment_id,
            attack_id=attack.attack_id,
            category=case.category,
            severity=case.severity,
            classification=classification,
            blocked=bool(getattr(defense, "blocked", False)),
            outcome=getattr(defense, "outcome", "error") if defense else "error",
            final_route=getattr(defense, "final_route", "") if defense else "",
            target_status=getattr(defense, "target_status", "") if defense else "",
            final_response=getattr(defense, "final_response", "") if defense else "",
            latency_ms=latency_ms,
            error=error,
            failure_stage=getattr(defense, "failure_stage", None) if defense else None,
        )
        self._results.setdefault(experiment_id, []).append(result)
        return result


__all__ = ["AttackExecutor", "AttackResult", "DefenseFn", "classify", "make_defense_fn"]
