"""End-to-End Evaluation: dataset → mutation → selection → executor → analysis.

`run_evaluation` drives the complete pipeline for one model target and
returns an `EvaluationReport` (analysis summary + model identity + target
usage + provenance). `compare_models` runs equivalent experiments for
several completion callables — identical dataset, selection, mutation
and defense configuration — and returns per-model reports. Ground truth
for FP/FN is caller-supplied; execution errors stay in every denominator
(analyzer policy). No persistence; reports are plain dataclasses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .analyzer import AnalysisSummary, summarize
from .dataset import AttackDataset
from .executor import AttackExecutor, AttackResult
from .mutation import generate_variants
from .orchestrator import RedTeamOrchestrator
from .selection import SelectionResult, select_attacks


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    model_name: str
    model_config: dict[str, Any]
    summary: AnalysisSummary
    selection: tuple[SelectionResult, ...]
    results: tuple[AttackResult, ...]
    target_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "summary": self.summary.to_dict(),
            "selection": [
                {
                    "attack_id": s.attack_id,
                    "rank": s.rank,
                    "strategy": s.strategy,
                    "reason": s.reason,
                }
                for s in self.selection
            ],
            "results": [
                {
                    "attack_id": r.attack_id,
                    "classification": r.classification,
                    "latency_ms": round(r.latency_ms, 3),
                    "error": r.error,
                }
                for r in self.results
            ],
            "target_usage": dict(self.target_usage),
        }


def run_evaluation(
    dataset: AttackDataset,
    runtime: Any,
    completion: Callable[[str], str],
    *,
    model_name: str,
    model_config: dict[str, Any] | None = None,
    strategy: str = "dataset_order",
    max_attacks: int = 10,
    category: str | None = None,
    severity: str | None = None,
    history: Sequence[AttackResult] = (),
    mutate: bool = False,
    ground_truth: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    target_usage: Callable[[], dict[str, Any]] | None = None,
) -> EvaluationReport:
    """One complete pass: selection → (mutation) → orchestrator → executor.

    Mutated cases are added to a fresh derived dataset; the source dataset
    is never modified.
    """
    working = dataset
    if mutate:
        variant_cases = [
            variant.as_attack_case()
            for case in dataset.list_cases()
            for variant in generate_variants(case, max_variants=2)
        ]
        working = AttackDataset(dataset.list_cases() + tuple(variant_cases))
    selection = select_attacks(
        working,
        strategy=strategy,
        max_attacks=max_attacks,
        category=category,
        severity=severity,
        history=history,
    )
    # Restrict the run dataset to the selected cases so the orchestrator
    # materializes exactly the chosen attacks (source dataset untouched).
    run_dataset = AttackDataset(tuple(s.case for s in selection))
    orchestrator = RedTeamOrchestrator(run_dataset)
    experiment_id = orchestrator.create_experiment(
        _config_for(run_dataset, {s.attack_id for s in selection})
    ).config.experiment_id
    executor = AttackExecutor(
        orchestrator,
        run_dataset,
        _defense_fn(runtime, completion),
        timeout_s=timeout_s,
    )
    if not selection:
        results: tuple[AttackResult, ...] = ()
    else:
        results = asyncio.run(executor.run_experiment(experiment_id))
    summary = summarize(
        experiment_id,
        results,
        dataset_version=run_dataset.dataset_version(),
        ground_truth=ground_truth,
    )
    return EvaluationReport(
        model_name=model_name,
        model_config=dict(model_config or {}),
        summary=summary,
        selection=selection,
        results=results,
        target_usage=target_usage() if target_usage else {},
    )


def _config_for(working: AttackDataset, selected_ids: set[str]) -> Any:
    from .state import ExperimentConfig

    return ExperimentConfig(
        name=f"evaluation-{len(selected_ids)}",
        attack_types=tuple(
            sorted({working.get(aid).category for aid in selected_ids}),
        )
        or (),
    )


def _defense_fn(runtime: Any, completion: Callable[[str], str]) -> Any:
    from .executor import make_defense_fn

    return make_defense_fn(runtime, completion=completion)


def compare_models(
    dataset: AttackDataset,
    runtime: Any,
    targets: Mapping[str, tuple[Callable[[str], str], dict[str, Any]]],
    *,
    strategy: str = "dataset_order",
    max_attacks: int = 10,
    mutate: bool = False,
    ground_truth: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    usage_probes: Mapping[str, Callable[[], dict[str, Any]]] | None = None,
) -> dict[str, EvaluationReport]:
    """Equivalent experiments per model; identical pipeline configuration."""
    reports: dict[str, EvaluationReport] = {}
    for name, (completion, config) in targets.items():
        reports[name] = run_evaluation(
            dataset,
            runtime,
            completion,
            model_name=name,
            model_config=config,
            strategy=strategy,
            max_attacks=max_attacks,
            mutate=mutate,
            ground_truth=ground_truth,
            timeout_s=timeout_s,
            target_usage=(usage_probes or {}).get(name),
        )
    return reports


__all__ = ["EvaluationReport", "compare_models", "run_evaluation"]
