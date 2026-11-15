"""Attack Selection: deterministic strategy-based case prioritization.

Registry of selection strategies over `AttackDataset` cases. Identical
inputs and configuration always produce identical selections; ties break
by attack_id everywhere. Result-informed selection prioritizes cases that
previously bypassed defenses using caller-supplied `AttackResult`
records — no history is stored here. Source cases are never mutated and
no attack is executed by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .dataset import AttackCase, AttackDataset
from .executor import AttackResult

StrategyFn = Callable[..., list[tuple[AttackCase, str]]]

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_UNRANKED = 4


def _dataset_order(
    cases: Sequence[AttackCase], limit: int, history: Sequence[AttackResult]
) -> list[tuple[AttackCase, str]]:
    return [(case, "dataset_order") for case in cases[:limit]]


def _severity_priority(
    cases: Sequence[AttackCase], limit: int, history: Sequence[AttackResult]
) -> list[tuple[AttackCase, str]]:
    ordered = sorted(
        cases,
        key=lambda c: (
            _SEVERITY_RANK.get(c.severity or "", _UNRANKED),
            c.attack_id,
        ),
    )[:limit]
    return [(case, f"severity:{case.severity or 'none'}") for case in ordered]


def _category_balanced(
    cases: Sequence[AttackCase], limit: int, history: Sequence[AttackResult]
) -> list[tuple[AttackCase, str]]:
    buckets: dict[str, list[AttackCase]] = {}
    for case in sorted(cases, key=lambda c: c.attack_id):
        buckets.setdefault(case.category, []).append(case)
    selected: list[tuple[AttackCase, str]] = []
    category_names = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for category in category_names:
            if buckets[category] and len(selected) < limit:
                case = buckets[category].pop(0)
                selected.append((case, f"category:{category}"))
                progressed = True
        if not progressed:
            break
    return selected


def _result_informed(
    cases: Sequence[AttackCase], limit: int, history: Sequence[AttackResult]
) -> list[tuple[AttackCase, str]]:
    bypassed = {r.attack_id for r in history if r.classification == "bypassed"}
    prioritized = sorted(
        cases, key=lambda c: (c.attack_id not in bypassed, c.attack_id)
    )[:limit]
    return [
        (case, "prior_bypass" if case.attack_id in bypassed else "no_history")
        for case in prioritized
    ]


_REGISTRY: dict[str, StrategyFn] = {
    "dataset_order": _dataset_order,
    "severity_priority": _severity_priority,
    "category_balanced": _category_balanced,
    "result_informed": _result_informed,
}


def register_selection_strategy(name: str, fn: StrategyFn) -> None:
    """Extend the registry at runtime (idempotent for the same function)."""
    if not name or not callable(fn):
        raise ValueError("strategy needs a name and a callable")
    if _REGISTRY.get(name) is not None and _REGISTRY[name] is not fn:
        raise ValueError(f"strategy {name!r} already registered")
    _REGISTRY[name] = fn


def known_selection_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """One selected case with strategy provenance."""

    case: AttackCase
    rank: int
    strategy: str
    reason: str

    @property
    def attack_id(self) -> str:
        return self.case.attack_id


def select_attacks(
    dataset: AttackDataset,
    *,
    strategy: str = "dataset_order",
    max_attacks: int = 10,
    category: str | None = None,
    severity: str | None = None,
    history: Sequence[AttackResult] = (),
) -> tuple[SelectionResult, ...]:
    """Deterministic selection; empty datasets yield an empty selection."""
    if strategy not in _REGISTRY:
        raise ValueError(
            f"unknown strategy {strategy!r}; "
            f"known: {', '.join(known_selection_strategies())}"
        )
    if not isinstance(max_attacks, int) or max_attacks < 1:
        raise ValueError("max_attacks must be a positive int")
    cases = dataset.filter(category=category, severity=severity)
    pairs = _REGISTRY[strategy](cases, max_attacks, history)
    return tuple(
        SelectionResult(case=case, rank=rank, strategy=strategy, reason=reason)
        for rank, (case, reason) in enumerate(pairs, start=1)
    )


__all__ = [
    "SelectionResult",
    "known_selection_strategies",
    "register_selection_strategy",
    "select_attacks",
]
