"""Phase 5: attack-selection strategy tests."""

from __future__ import annotations

import pytest

from safety_platform.redteam import (
    AttackCase,
    AttackDataset,
    AttackResult,
    ExperimentConfig,
    RedTeamOrchestrator,
    known_selection_strategies,
    register_selection_strategy,
    select_attacks,
)


def _case(attack_id: str, category: str, severity: str | None = None) -> AttackCase:
    return AttackCase(
        name=f"case-{attack_id}",
        category=category,
        prompt=f"attack {attack_id}",
        attack_id=attack_id,
        severity=severity,
    )


def _dataset() -> AttackDataset:
    return AttackDataset(
        (
            _case("a-1", "prompt_injection", "low"),
            _case("b-2", "prompt_injection", "critical"),
            _case("c-3", "direct_jailbreak"),
            _case("d-4", "prompt_injection", "high"),
        )
    )


def _result(attack_id: str, classification: str) -> AttackResult:
    return AttackResult(
        experiment_id="exp-1",
        attack_id=attack_id,
        category="prompt_injection",
        severity=None,
        classification=classification,
        blocked=classification == "blocked",
        outcome="safe",
        final_route="",
        target_status="",
        final_response="",
        latency_ms=1.0,
    )


def test_dataset_order_baseline() -> None:
    selection = select_attacks(_dataset())
    assert [s.attack_id for s in selection] == ["a-1", "b-2", "c-3", "d-4"]
    assert all(s.strategy == "dataset_order" for s in selection)
    assert all(s.reason == "dataset_order" for s in selection)
    assert [s.rank for s in selection] == [1, 2, 3, 4]


def test_severity_priority_with_tie_break() -> None:
    selection = select_attacks(_dataset(), strategy="severity_priority")
    assert [s.attack_id for s in selection] == ["b-2", "d-4", "a-1", "c-3"]
    assert selection[0].reason == "severity:critical"
    # c-3 and nothing else: unranked (no severity) sorts last; tie-break by id.
    assert selection[3].reason == "severity:none"
    same_severity = AttackDataset(
        (
            _case("z-9", "prompt_injection", "high"),
            _case("y-1", "policy_evasion", "high"),
        )
    )
    tie = select_attacks(same_severity, strategy="severity_priority")
    assert [s.attack_id for s in tie] == ["y-1", "z-9"]


def test_category_balanced_round_robin() -> None:
    selection = select_attacks(_dataset(), strategy="category_balanced")
    # Alphabetical categories first, within category by attack_id.
    assert [s.attack_id for s in selection] == ["c-3", "a-1", "b-2", "d-4"]
    assert selection[0].reason == "category:direct_jailbreak"
    assert selection[1].reason == "category:prompt_injection"


def test_result_informed_prioritizes_bypasses() -> None:
    history = (
        _result("d-4", "bypassed"),
        _result("a-1", "blocked"),
        _result("ghost", "bypassed"),  # not in dataset: ignored
    )
    selection = select_attacks(_dataset(), strategy="result_informed", history=history)
    assert [s.attack_id for s in selection] == ["d-4", "a-1", "b-2", "c-3"]
    assert selection[0].reason == "prior_bypass"
    assert selection[1].reason == "no_history"
    assert selection[3].reason == "no_history"


def test_result_informed_without_history() -> None:
    selection = select_attacks(_dataset(), strategy="result_informed")
    assert [s.attack_id for s in selection] == ["a-1", "b-2", "c-3", "d-4"]
    assert all(s.reason == "no_history" for s in selection)


def test_filters_and_limit() -> None:
    by_category = select_attacks(_dataset(), category="prompt_injection", max_attacks=2)
    assert [s.attack_id for s in by_category] == ["a-1", "b-2"]
    by_severity = select_attacks(_dataset(), severity="high")
    assert [s.attack_id for s in by_severity] == ["d-4"]
    capped = select_attacks(_dataset(), max_attacks=2, strategy="severity_priority")
    assert [s.attack_id for s in capped] == ["b-2", "d-4"]
    with pytest.raises(ValueError):
        select_attacks(_dataset(), max_attacks=0)


def test_empty_dataset_and_no_filter_matches() -> None:
    assert select_attacks(AttackDataset()) == ()
    assert select_attacks(_dataset(), category="direct_jailbreak", severity="low") == ()


def test_invalid_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        select_attacks(_dataset(), strategy="adaptive")


def test_registry_extension_and_listing() -> None:
    assert "result_informed" in known_selection_strategies()

    def reversed_order(cases, limit, history):
        return [(case, "reversed") for case in list(cases)[::-1][:limit]]

    register_selection_strategy("reversed_order", reversed_order)
    selection = select_attacks(_dataset(), strategy="reversed_order", max_attacks=2)
    assert [s.attack_id for s in selection] == ["d-4", "c-3"]
    with pytest.raises(ValueError):
        register_selection_strategy("dataset_order", reversed_order)


def test_deterministic_repeat_and_provenance() -> None:
    dataset = _dataset()
    kwargs = {"strategy": "severity_priority", "max_attacks": 3}
    assert select_attacks(dataset, **kwargs) == select_attacks(dataset, **kwargs)
    first = select_attacks(dataset, **kwargs)[0]
    assert first.case.prompt == "attack b-2"
    assert first.strategy == "severity_priority"
    assert first.reason == "severity:critical"
    assert first.rank == 1


def test_orchestrator_and_source_immutability_compat() -> None:
    dataset = _dataset()
    selection = select_attacks(dataset, strategy="severity_priority", max_attacks=2)
    orchestrator = RedTeamOrchestrator(dataset)
    experiment_id = orchestrator.create_experiment(
        ExperimentConfig(name="sel", attack_types=("prompt_injection",))
    ).config.experiment_id
    assert selection[0].attack_id == "b-2"  # critical first, in selected set
    orchestrator.start_experiment(experiment_id)
    assert "b-2" in {a.attack_id for a in orchestrator.attacks(experiment_id)}
    assert dataset.get("b-2").prompt == "attack b-2"
