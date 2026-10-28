"""Aggregate evaluation outcomes into ASR, FPR, and latency reports."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    system: str
    attack_cases: int
    attack_success: int
    asr_pct: float
    asr_low_pct: float
    asr_high_pct: float
    benign_cases: int
    false_positives: int
    fpr_pct: float
    fpr_low_pct: float
    fpr_high_pct: float
    p50_guard_ms: float
    p95_guard_ms: float
    p99_guard_ms: float
    tier3_rate_pct: float
    judge_errors: int


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    system: str
    category: str
    attack_cases: int
    attack_success: int
    asr_pct: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    run_id: str
    target_model: str
    dataset_digest: str
    systems: tuple[SystemMetrics, ...]
    categories: tuple[CategoryMetrics, ...]


def load_report(
    database: str,
    *,
    run_id: str | None = None,
) -> EvaluationReport:
    connection = sqlite3.connect(database, timeout=10.0)
    try:
        resolved = _resolve_run(connection, run_id)
        meta = connection.execute(
            "SELECT target_model, dataset_digest FROM run WHERE run_id = ?",
            (resolved,),
        ).fetchone()
        if meta is None:
            raise ValueError(f"unknown run_id: {resolved}")
        systems = tuple(
            _system_metrics(connection, resolved, system)
            for system in _systems(connection, resolved)
        )
        categories = tuple(_category_metrics(connection, resolved))
    finally:
        connection.close()
    return EvaluationReport(
        run_id=resolved,
        target_model=str(meta[0]),
        dataset_digest=str(meta[1]),
        systems=systems,
        categories=categories,
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation `{report.run_id}`",
        "",
        f"- Target model: `{report.target_model}`",
        f"- Dataset SHA-256: `{report.dataset_digest}`",
        "",
        (
            "| System | Attacks | ASR | 95% CI | Benign | FPR | 95% CI | "
            "p50 ms | p95 ms | p99 ms | Tier 3 | Judge err |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in report.systems:
        lines.append(
            "| "
            f"{metrics.system} | "
            f"{metrics.attack_cases} | "
            f"{metrics.asr_pct:.1f}% | "
            f"{metrics.asr_low_pct:.1f}-{metrics.asr_high_pct:.1f}% | "
            f"{metrics.benign_cases} | "
            f"{metrics.fpr_pct:.1f}% | "
            f"{metrics.fpr_low_pct:.1f}-{metrics.fpr_high_pct:.1f}% | "
            f"{metrics.p50_guard_ms:.2f} | "
            f"{metrics.p95_guard_ms:.2f} | "
            f"{metrics.p99_guard_ms:.2f} | "
            f"{metrics.tier3_rate_pct:.1f}% | "
            f"{metrics.judge_errors} |"
        )
    if report.categories:
        lines.extend(
            [
                "",
                "| System | Category | Attacks | ASR |",
                "|---|---|---:|---:|",
            ]
        )
        for category in report.categories:
            lines.append(
                "| "
                f"{category.system} | "
                f"{category.category} | "
                f"{category.attack_cases} | "
                f"{category.asr_pct:.1f}% |"
            )
    return "\n".join(lines) + "\n"


def _resolve_run(connection: sqlite3.Connection, run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    row = connection.execute(
        "SELECT run_id FROM run ORDER BY completed_ns DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("evaluation database contains no runs")
    return str(row[0])


def _systems(connection: sqlite3.Connection, run_id: str) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT DISTINCT system FROM run_result WHERE run_id = ? ORDER BY system",
        (run_id,),
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _system_metrics(
    connection: sqlite3.Connection,
    run_id: str,
    system: str,
) -> SystemMetrics:
    attack_rows = connection.execute(
        """
        SELECT judge_complied, judge_error
        FROM run_result
        WHERE run_id = ? AND system = ? AND label = 'attack'
        """,
        (run_id, system),
    ).fetchall()
    benign_rows = connection.execute(
        """
        SELECT blocked
        FROM run_result
        WHERE run_id = ? AND system = ? AND label = 'benign'
        """,
        (run_id, system),
    ).fetchall()
    latency_rows = connection.execute(
        """
        SELECT guard_ms, tier3_invoked
        FROM run_result
        WHERE run_id = ? AND system = ?
        """,
        (run_id, system),
    ).fetchall()

    attack_cases = len(attack_rows)
    attack_success = sum(1 for row in attack_rows if row[0] == 1)
    judge_errors = sum(1 for row in attack_rows if row[1] is not None)
    benign_cases = len(benign_rows)
    false_positives = sum(1 for row in benign_rows if row[0] == 1)
    latencies = [float(row[0]) for row in latency_rows]
    tier3_invocations = sum(1 for row in latency_rows if row[1] == 1)
    asr = _ratio(attack_success, attack_cases)
    fpr = _ratio(false_positives, benign_cases)
    asr_low, asr_high = _wilson(attack_success, attack_cases)
    fpr_low, fpr_high = _wilson(false_positives, benign_cases)

    return SystemMetrics(
        system=system,
        attack_cases=attack_cases,
        attack_success=attack_success,
        asr_pct=asr * 100.0,
        asr_low_pct=asr_low * 100.0,
        asr_high_pct=asr_high * 100.0,
        benign_cases=benign_cases,
        false_positives=false_positives,
        fpr_pct=fpr * 100.0,
        fpr_low_pct=fpr_low * 100.0,
        fpr_high_pct=fpr_high * 100.0,
        p50_guard_ms=_percentile(latencies, 0.50),
        p95_guard_ms=_percentile(latencies, 0.95),
        p99_guard_ms=_percentile(latencies, 0.99),
        tier3_rate_pct=_ratio(tier3_invocations, len(latency_rows)) * 100.0,
        judge_errors=judge_errors,
    )


def _category_metrics(
    connection: sqlite3.Connection,
    run_id: str,
) -> tuple[CategoryMetrics, ...]:
    rows = connection.execute(
        """
        SELECT system, category, COUNT(*), COALESCE(SUM(judge_complied), 0)
        FROM run_result
        WHERE run_id = ? AND label = 'attack'
        GROUP BY system, category
        ORDER BY system, category
        """,
        (run_id,),
    ).fetchall()
    return tuple(
        CategoryMetrics(
            system=str(row[0]),
            category=str(row[1]),
            attack_cases=int(row[2]),
            attack_success=int(row[3]),
            asr_pct=_ratio(int(row[3]), int(row[2])) * 100.0,
        )
        for row in rows
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


__all__ = [
    "CategoryMetrics",
    "EvaluationReport",
    "SystemMetrics",
    "load_report",
    "render_markdown",
]
