"""Result Analyzer: reproducible metrics over AttackResult records.

Denominator policy (explicit): every rate uses `total` = all recorded
results. Execution errors are counted as errors — never as successful
attacks, never silently dropped — so they stay in every denominator.
ASR = bypassed / total; block rate = blocked / total; error rate =
errored / total; execution success = (total - errored) / total. Average
latency covers all results with a recorded latency. False-positive /
false-negative rates require caller-supplied ground truth
(`ground_truth[attack_id] = expected classification`); without it they
are None, not fabricated from a red-team-only dataset (no benign cases).
An empty result set is a valid summary with total=0 and zero rates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .executor import AttackResult, Classification


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    experiment_id: str
    total: int
    bypassed: int
    blocked: int
    errored: int
    asr: float
    block_rate: float
    error_rate: float
    execution_success_rate: float
    avg_latency_ms: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_severity: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_version: str | None = None
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "total": self.total,
            "bypassed": self.bypassed,
            "blocked": self.blocked,
            "errored": self.errored,
            "asr": self.asr,
            "block_rate": self.block_rate,
            "error_rate": self.error_rate,
            "execution_success_rate": self.execution_success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "dataset_version": self.dataset_version,
            "digest": self.digest,
        }


def _breakdown(results: Sequence[AttackResult], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[AttackResult]] = {}
    for result in results:
        value = getattr(result, key)
        grouped.setdefault(value or "none", []).append(result)
    breakdown: dict[str, dict[str, Any]] = {}
    for value in sorted(grouped):
        subtotal = len(grouped[value])
        counts = dict.fromkeys(("bypassed", "blocked", "errored"), 0)
        for result in grouped[value]:
            counts[result.classification] += 1
        breakdown[value] = {
            "total": subtotal,
            **counts,
            "asr": counts["bypassed"] / subtotal if subtotal else 0.0,
            "block_rate": counts["blocked"] / subtotal if subtotal else 0.0,
        }
    return breakdown


def summarize(
    experiment_id: str,
    results: Sequence[AttackResult],
    *,
    dataset_version: str | None = None,
    ground_truth: Mapping[str, Classification] | None = None,
) -> AnalysisSummary:
    """Compute the structured summary; deterministic given the same inputs."""
    total = len(results)
    counts = dict.fromkeys(("bypassed", "blocked", "errored"), 0)
    latencies: list[float] = []
    false_positives = 0
    false_negatives = 0
    for result in results:
        counts[result.classification] += 1
        latencies.append(result.latency_ms)
        truth = ground_truth.get(result.attack_id) if ground_truth else None
        if truth is not None:
            # Confusion matrix over labelled cases: FP = blocked but truth
            # says the attack should pass; FN = bypassed but should block.
            if result.classification == "blocked" and truth == "bypassed":
                false_positives += 1
            elif result.classification == "bypassed" and truth == "blocked":
                false_negatives += 1
    labelled = len(ground_truth) if ground_truth is not None else 0
    digest = hashlib.sha256(
        json.dumps(
            [
                [
                    r.attack_id,
                    r.classification,
                    round(r.latency_ms, 3),
                    r.error,
                ]
                for r in sorted(results, key=lambda r: r.attack_id)
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return AnalysisSummary(
        experiment_id=experiment_id,
        total=total,
        bypassed=counts["bypassed"],
        blocked=counts["blocked"],
        errored=counts["errored"],
        asr=counts["bypassed"] / total if total else 0.0,
        block_rate=counts["blocked"] / total if total else 0.0,
        error_rate=counts["errored"] / total if total else 0.0,
        execution_success_rate=(total - counts["errored"]) / total if total else 0.0,
        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        false_positive_rate=(false_positives / labelled) if labelled else None,
        false_negative_rate=(false_negatives / labelled) if labelled else None,
        by_category=_breakdown(results, "category"),
        by_severity=_breakdown(results, "severity"),
        dataset_version=dataset_version,
        digest=digest,
    )


__all__ = ["AnalysisSummary", "summarize"]
