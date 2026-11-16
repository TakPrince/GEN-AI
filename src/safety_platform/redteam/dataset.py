"""Attack Dataset Manager: validated single-turn red-team attack cases.

In-memory dataset with deterministic JSON export/import and a sha256
content version (same convention as the evaluation manifests). No storage,
no execution, no Defense Engine coupling — the future Orchestrator
consumes `get`/`list_cases`/`filter` through this public interface.
Categories are a controlled registry, extensible at runtime via
`register_category` without touching unrelated components.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .state import _new_id, _now, _require_non_empty

# --- Controlled vocabularies --------------------------------------------

_CATEGORIES: dict[str, str] = {
    "direct_jailbreak": "Direct jailbreak attempt",
    "prompt_injection": "Prompt injection",
    "instruction_override": "Instruction override",
    "roleplay_manipulation": "Role-play manipulation",
    "system_prompt_extraction": "System prompt extraction",
    "policy_evasion": "Policy evasion",
    "sensitive_info_extraction": "Sensitive information extraction",
}
_SEVERITY_LEVELS = frozenset({"low", "medium", "high", "critical"})


def register_category(value: str, label: str) -> None:
    """Extend the category registry at runtime (idempotent for same label)."""
    _require_non_empty("category", value)
    _require_non_empty("label", label)
    if value in _CATEGORIES and _CATEGORIES[value] != label:
        raise ValueError(f"category {value!r} already registered")
    _CATEGORIES[value] = label


def known_categories() -> tuple[str, ...]:
    return tuple(sorted(_CATEGORIES))


def _validate_category(value: str) -> None:
    if not isinstance(value, str) or value not in _CATEGORIES:
        raise ValueError(
            f"unknown category {value!r}; known: {', '.join(known_categories())}"
        )


@dataclass(frozen=True, slots=True)
class AttackCase:
    """One single-turn adversarial case; identity is its `attack_id`."""

    name: str
    category: str
    prompt: str
    attack_id: str = field(default_factory=_new_id)
    description: str = ""
    severity: str | None = None
    tags: tuple[str, ...] = ()
    expected_behavior: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    version: int = 1

    def __post_init__(self) -> None:
        _require_non_empty("attack_id", self.attack_id)
        _require_non_empty("name", self.name)
        _require_non_empty("prompt", self.prompt)
        _validate_category(self.category)
        if self.severity is not None and self.severity not in _SEVERITY_LEVELS:
            raise ValueError(
                f"unknown severity {self.severity!r}; "
                f"known: {', '.join(sorted(_SEVERITY_LEVELS))}"
            )
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive int")
        seen: set[str] = set()
        for tag in self.tags:
            _require_non_empty("tag", tag)
            if tag in seen:
                raise ValueError(f"duplicate tag {tag!r}")
            seen.add(tag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "name": self.name,
            "category": self.category,
            "prompt": self.prompt,
            "description": self.description,
            "severity": self.severity,
            "tags": list(self.tags),
            "expected_behavior": self.expected_behavior,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttackCase:
        try:
            return cls(
                attack_id=data["attack_id"],
                name=data["name"],
                category=data["category"],
                prompt=data["prompt"],
                description=data.get("description", ""),
                severity=data.get("severity"),
                tags=tuple(data.get("tags", ())),
                expected_behavior=data.get("expected_behavior", ""),
                metadata=dict(data.get("metadata", {})),
                created_at=data["created_at"],
                version=data["version"],
            )
        except KeyError as exc:
            raise ValueError(f"missing attack case field: {exc}") from exc


def case_fingerprint(case: AttackCase) -> str:
    """Content digest of a case: changed content is distinguishable.

    Excludes `created_at` (non-semantic timing) so two datasets built with
    identical logical content reproduce the same dataset version.
    """
    canonical = json.dumps(
        {k: v for k, v in case.to_dict().items() if k != "created_at"},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Errors --------------------------------------------------------------


class DuplicateAttackError(ValueError):
    """An attack_id already present in the dataset."""


class DatasetConflictError(ValueError):
    """Re-import of an attack_id whose content differs from the stored case."""


# --- Dataset manager ------------------------------------------------------


class AttackDataset:
    """In-memory, storage-independent collection of attack cases."""

    def __init__(self, cases: tuple[AttackCase, ...] = ()) -> None:
        self._cases: dict[str, AttackCase] = {}
        for case in cases:
            self.add(case)

    def add(self, case: AttackCase, /) -> None:
        if not isinstance(case, AttackCase):
            raise ValueError("add() requires an AttackCase")
        if case.attack_id in self._cases:
            raise DuplicateAttackError(f"duplicate attack_id: {case.attack_id}")
        self._cases[case.attack_id] = case

    def get(self, attack_id: str, /) -> AttackCase:
        try:
            return self._cases[attack_id]
        except KeyError:
            raise KeyError(f"unknown attack_id: {attack_id}") from None

    def list_cases(self) -> tuple[AttackCase, ...]:
        return tuple(sorted(self._cases.values(), key=lambda c: c.attack_id))

    def filter(
        self,
        *,
        category: str | None = None,
        tags: tuple[str, ...] | None = None,
        severity: str | None = None,
    ) -> tuple[AttackCase, ...]:
        if category is not None:
            _validate_category(category)
        selected = self.list_cases()
        if category is not None:
            selected = tuple(c for c in selected if c.category == category)
        if severity is not None:
            if severity not in _SEVERITY_LEVELS:
                raise ValueError(f"unknown severity {severity!r}")
            selected = tuple(c for c in selected if c.severity == severity)
        if tags:
            selected = tuple(c for c in selected if set(tags).issubset(set(c.tags)))
        return selected

    def __len__(self) -> int:
        return len(self._cases)

    def dataset_version(self) -> str:
        digest = hashlib.sha256()
        for fingerprint in sorted(
            case_fingerprint(case) for case in self._cases.values()
        ):
            digest.update(fingerprint.encode("ascii"))
        return digest.hexdigest()

    # --- Deterministic JSON export / atomic import ------------------------

    def to_json(self) -> str:
        payload = {
            "dataset_version": self.dataset_version(),
            "cases": [case.to_dict() for case in self.list_cases()],
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> AttackDataset:
        return cls(tuple(_parse_payload(text)[1].values()))

    def import_json(self, text: str, /) -> None:
        """Atomically import: parse + validate everything, then apply.

        Same-ID identical content is skipped; same-ID different content
        raises DatasetConflictError; duplicate IDs inside one payload raise.
        """
        _existing, incoming = _parse_payload(text)
        for attack_id, case in incoming.items():
            current = self._cases.get(attack_id)
            if current is not None and case_fingerprint(current) != case_fingerprint(
                case
            ):
                raise DatasetConflictError(
                    f"attack_id {attack_id} differs from stored case"
                )
        for case in incoming.values():
            self._cases.setdefault(case.attack_id, case)

    def replace_all(self, text: str, /) -> None:
        """Swap the entire dataset from a payload (used by re-imports)."""
        _version, cases = _parse_payload(text)
        self._cases = dict(cases)


def _parse_payload(text: str) -> tuple[str, dict[str, AttackCase]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("invalid dataset payload: expected object with 'cases'")
    cases: dict[str, AttackCase] = {}
    for entry in payload["cases"]:
        case = AttackCase.from_dict(entry)
        if case.attack_id in cases:
            raise DuplicateAttackError(
                f"duplicate attack_id in payload: {case.attack_id}"
            )
        cases[case.attack_id] = case
    return payload.get("dataset_version", ""), cases


__all__ = [
    "AttackCase",
    "AttackDataset",
    "DatasetConflictError",
    "DuplicateAttackError",
    "case_fingerprint",
    "known_categories",
    "register_category",
]
