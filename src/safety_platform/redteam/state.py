"""Experiment lifecycle and attack state for the Red-Team Engine.

Foundation only: reproducible experiment identity/configuration and the
explicit state machine shared by experiments and their attacks. No storage,
no execution, no coupling to the Defense Engine — later milestones
(Dataset Manager, Orchestrator, Executor) consume these models as-is.
Timestamps are ISO-8601 strings; the module clock is monkeypatchable for
deterministic tests.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class LifecycleStatus(str, Enum):
    """Lifecycle shared by experiments and individual attacks.

    Transitions: CREATED -> RUNNING -> COMPLETED | FAILED | CANCELLED.
    COMPLETED/FAILED/CANCELLED are terminal; re-applying the current
    status is an idempotent no-op, every other invalid move raises.
    """

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL

    def can_transition(self, target: LifecycleStatus) -> bool:
        if target is self:
            return True  # duplicate update: idempotent
        if self.terminal:
            return False
        return target in _ALLOWED[self]


_ALLOWED: dict[LifecycleStatus, frozenset[LifecycleStatus]] = {
    LifecycleStatus.CREATED: frozenset({LifecycleStatus.RUNNING}),
    LifecycleStatus.RUNNING: frozenset(
        {
            LifecycleStatus.COMPLETED,
            LifecycleStatus.FAILED,
            LifecycleStatus.CANCELLED,
        }
    ),
}
_TERMINAL = frozenset(
    {LifecycleStatus.COMPLETED, LifecycleStatus.FAILED, LifecycleStatus.CANCELLED}
)


class InvalidTransitionError(ValueError):
    """Raised when a lifecycle transition is not allowed."""


def _require_non_empty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Minimum identity + reproduction inputs for one experiment."""

    name: str
    experiment_id: str = field(default_factory=_new_id)
    config_version: int = 1
    description: str = ""
    # Attack selection: empty tuple means "no restriction yet" (later
    # milestones refine selection); `seed` makes dataset/order reproducible.
    attack_types: tuple[str, ...] = ()
    seed: int | None = None
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_empty("experiment_id", self.experiment_id)
        if not isinstance(self.config_version, int) or self.config_version < 1:
            raise ValueError("config_version must be a positive int")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an int or None")
        for attack_type in self.attack_types:
            _require_non_empty("attack_type", attack_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "experiment_id": self.experiment_id,
            "config_version": self.config_version,
            "description": self.description,
            "attack_types": list(self.attack_types),
            "seed": self.seed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        try:
            return cls(
                name=data["name"],
                experiment_id=data["experiment_id"],
                config_version=data["config_version"],
                description=data.get("description", ""),
                attack_types=tuple(data.get("attack_types", ())),
                seed=data.get("seed"),
                created_at=data["created_at"],
            )
        except KeyError as exc:
            raise ValueError(f"missing config field: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExperimentState:
    """Lifecycle holder for one experiment; frozen transitions, no storage."""

    config: ExperimentConfig
    status: LifecycleStatus = LifecycleStatus.CREATED
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def transition(
        self, target: LifecycleStatus, *, error: str | None = None
    ) -> ExperimentState:
        if not isinstance(target, LifecycleStatus):
            raise ValueError("target must be a LifecycleStatus")
        if error is not None and target is not LifecycleStatus.FAILED:
            raise ValueError("error is only recorded on FAILED")
        if not self.status.can_transition(target):
            raise InvalidTransitionError(
                f"experiment {self.config.experiment_id}: "
                f"{self.status.value} -> {target.value} not allowed"
            )
        if target is self.status:
            return self  # duplicate update: predictable no-op
        if target is LifecycleStatus.FAILED and not (error and error.strip()):
            raise ValueError("FAILED requires an error description")
        return replace(self, status=target, error=error, updated_at=_now())

    # Convenience wrappers (same rules as transition()).
    def start(self) -> ExperimentState:
        return self.transition(LifecycleStatus.RUNNING)

    def complete(self) -> ExperimentState:
        return self.transition(LifecycleStatus.COMPLETED)

    def fail(self, error: str, /) -> ExperimentState:
        return self.transition(LifecycleStatus.FAILED, error=error)

    def cancel(self) -> ExperimentState:
        return self.transition(LifecycleStatus.CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentState:
        try:
            config = ExperimentConfig.from_dict(data["config"])
            return cls(
                config=config,
                status=LifecycleStatus(data["status"]),
                error=data.get("error"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
            )
        except KeyError as exc:
            raise ValueError(f"missing experiment field: {exc}") from exc
        except ValueError as exc:
            if "is not a valid" in str(exc):
                raise ValueError(f"unknown status: {data.get('status')}") from exc
            raise


@dataclass(frozen=True, slots=True)
class AttackState:
    """One attack within an experiment; follows the same lifecycle."""

    experiment_id: str
    attack_type: str
    prompt: str
    attack_id: str = field(default_factory=_new_id)
    status: LifecycleStatus = LifecycleStatus.CREATED
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        _require_non_empty("attack_id", self.attack_id)
        _require_non_empty("experiment_id", self.experiment_id)
        _require_non_empty("attack_type", self.attack_type)
        _require_non_empty("prompt", self.prompt)
        if not isinstance(self.status, LifecycleStatus):
            raise ValueError("status must be a LifecycleStatus")

    def transition(
        self, target: LifecycleStatus, *, error: str | None = None
    ) -> AttackState:
        if not isinstance(target, LifecycleStatus):
            raise ValueError("target must be a LifecycleStatus")
        if error is not None and target is not LifecycleStatus.FAILED:
            raise ValueError("error is only recorded on FAILED")
        if not self.status.can_transition(target):
            raise InvalidTransitionError(
                f"attack {self.attack_id}: "
                f"{self.status.value} -> {target.value} not allowed"
            )
        if target is self.status:
            return self
        if target is LifecycleStatus.FAILED and not (error and error.strip()):
            raise ValueError("FAILED requires an error description")
        return replace(self, status=target, error=error, updated_at=_now())

    def start(self) -> AttackState:
        return self.transition(LifecycleStatus.RUNNING)

    def complete(self) -> AttackState:
        return self.transition(LifecycleStatus.COMPLETED)

    def fail(self, error: str, /) -> AttackState:
        return self.transition(LifecycleStatus.FAILED, error=error)

    def cancel(self) -> AttackState:
        return self.transition(LifecycleStatus.CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "attack_type": self.attack_type,
            "prompt": self.prompt,
            "error": self.error,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttackState:
        try:
            return cls(
                attack_id=data["attack_id"],
                experiment_id=data["experiment_id"],
                attack_type=data["attack_type"],
                prompt=data["prompt"],
                status=LifecycleStatus(data["status"]),
                error=data.get("error"),
                metadata=dict(data.get("metadata", {})),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
            )
        except KeyError as exc:
            raise ValueError(f"missing attack field: {exc}") from exc
        except ValueError as exc:
            if "is not a valid" in str(exc):
                raise ValueError(f"unknown status: {data.get('status')}") from exc
            raise


__all__ = [
    "AttackState",
    "ExperimentConfig",
    "ExperimentState",
    "InvalidTransitionError",
    "LifecycleStatus",
]
