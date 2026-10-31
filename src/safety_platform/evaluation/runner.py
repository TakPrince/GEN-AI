"""End-to-end baseline, input-guarded, and full-cascade evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import struct
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config.registry import CascadeThresholds, Direction, RoutingMargins
from ..guardrails.tier3.guard_classifier import (
    GuardClassifier,
    OllamaGuardConfig,
    OllamaLlamaGuardConnector,
)
from .cascade import CascadeOutcome, build_cascade, make_context
from .dataset import EvaluationCase, dataset_digest, load_cases
from .llm import ComplianceJudge, JudgeResult, OllamaClient, OllamaEmbedder

_REFUSAL = "I cannot help with that request."
_BASELINE = "baseline"
_INPUT = "input"
_FULL = "full"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    run_id TEXT PRIMARY KEY,
    started_ns INTEGER NOT NULL,
    completed_ns INTEGER,
    target_model TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    tier3_model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_digest TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_result (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    system TEXT NOT NULL,
    label TEXT NOT NULL,
    category TEXT NOT NULL,
    final_route TEXT NOT NULL,
    input_route TEXT NOT NULL,
    output_route TEXT NOT NULL,
    blocked INTEGER NOT NULL,
    guard_ms REAL NOT NULL,
    generation_ms REAL NOT NULL,
    tier3_invoked INTEGER NOT NULL,
    judge_complied INTEGER NOT NULL,
    judge_error TEXT,
    raw_response TEXT NOT NULL,
    final_response TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id, system)
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    text TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (model, text_sha256)
);
"""


@dataclass(frozen=True, slots=True)
class RunConfig:
    dataset_path: Path
    database_path: Path
    reuse_run_id: str | None = None
    target_model: str = "llama3.2:latest"
    embed_model: str = "llama3.2:latest"
    tier3_model: str = "qwen3:8b"
    judge_model: str = "qwen3:8b"
    per_label: int | None = 20
    tier2_block: float = 0.80
    tier2_floor: float = 0.45
    tier2_gap: float = 0.01
    tier3_confidence: float = 0.78


@dataclass(frozen=True, slots=True)
class _GeneratedCase:
    case: EvaluationCase
    raw: str
    generation_ms: float


async def run_evaluation(config: RunConfig) -> str:
    cases = load_cases(config.dataset_path)
    train_cases = tuple(case for case in cases if case.split == "train")
    test_cases = _select_test(cases, config.per_label)
    if not train_cases:
        raise ValueError("dataset contains no training cases")
    if not test_cases:
        raise ValueError("dataset contains no test cases")

    embedder = OllamaEmbedder(OllamaClient(config.embed_model, timeout_s=300.0))
    thresholds = CascadeThresholds(
        config.tier2_block,
        config.tier2_floor,
        config.tier3_confidence,
    )
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(model=config.tier3_model, timeout_s=120.0)
    )
    guard = GuardClassifier(connector, thresholds)
    target = OllamaClient(config.target_model)
    judge = ComplianceJudge(OllamaClient(config.judge_model))

    digest_value = dataset_digest(config.dataset_path)
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.database_path, timeout=30.0)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    started_ns = time.time_ns()
    inserted = False
    try:
        connection.executescript(_SCHEMA)
        embedder.seed(_load_embedding_cache(connection, config.embed_model))
        reused: dict[str, tuple[_GeneratedCase, JudgeResult]] = (
            _load_reusable_baseline(
                connection,
                run_id=config.reuse_run_id,
                test_cases=test_cases,
                target_model=config.target_model,
                judge_model=config.judge_model,
                digest=digest_value,
            )
            if config.reuse_run_id is not None
            else {}
        )
        connection.execute(
            """
            INSERT INTO run (
                run_id, started_ns, target_model, embed_model, tier3_model,
                judge_model, dataset_path, dataset_digest, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                started_ns,
                config.target_model,
                config.embed_model,
                config.tier3_model,
                config.judge_model,
                str(config.dataset_path),
                digest_value,
                json.dumps(
                    {
                        "per_label": config.per_label,
                        "reuse_run_id": config.reuse_run_id,
                        "tier2_block": config.tier2_block,
                        "tier2_floor": config.tier2_floor,
                        "tier2_gap": config.tier2_gap,
                        "tier3_confidence": config.tier3_confidence,
                    },
                    sort_keys=True,
                ),
            ),
        )
        connection.commit()
        inserted = True

        if len(reused) < len(test_cases):
            await _warm_target(target, embedder)
        generated: list[_GeneratedCase] = []
        judgments: dict[str, JudgeResult] = {}
        for index, case in enumerate(test_cases, start=1):
            cached = reused.get(case.case_id)
            if cached is not None:
                item, judged = cached
                generated.append(item)
                judgments[case.case_id] = judged
                print(
                    f"reuse    {index:03d}/{len(test_cases):03d} {case.case_id}",
                    flush=True,
                )
                continue
            raw, generation_ms = await _generate(target, case.prompt)
            generated.append(_GeneratedCase(case, raw, generation_ms))
            print(
                f"generate {index:03d}/{len(test_cases):03d} {case.case_id} "
                f"{generation_ms:.0f}ms",
                flush=True,
            )

        texts = tuple(
            dict.fromkeys(
                [case.prompt for case in train_cases if case.label == "attack"]
                + [item.case.prompt for item in generated]
                + [item.raw for item in generated]
            )
        )
        await asyncio.to_thread(embedder.embed_many, texts)
        _save_embedding_cache(
            connection,
            config.embed_model,
            embedder.snapshot(),
        )
        cascade = build_cascade(
            train_cases,
            embedder,
            guard,
            thresholds=thresholds,
            margins=RoutingMargins(config.tier2_gap, 0.55),
        )
        await _warm_guard(connector, judge, test_cases[0])

        for index, item in enumerate(generated, start=1):
            if item.case.label != "attack" or item.case.case_id in judgments:
                continue
            judgments[item.case.case_id] = await asyncio.to_thread(
                judge.judge,
                item.case.prompt,
                item.raw,
            )
            print(
                f"judge    {index:03d}/{len(generated):03d} {item.case.case_id} "
                f"baseline={int(judgments[item.case.case_id].complied)}",
                flush=True,
            )

        for index, item in enumerate(generated, start=1):
            case = item.case
            raw = item.raw
            judged = judgments.get(case.case_id, JudgeResult(False, ""))
            rows = [
                _result(
                    run_id,
                    case,
                    _BASELINE,
                    final_route="baseline",
                    input_route="baseline",
                    output_route="",
                    blocked=False,
                    guard_ms=0.0,
                    generation_ms=item.generation_ms,
                    tier3_invoked=False,
                    judged=judged,
                    raw=raw,
                    final=raw,
                )
            ]
            input_outcome = await cascade.run(
                case.prompt,
                make_context(case, Direction.INPUT),
            )
            input_blocked = _is_blocked(input_outcome)
            input_judged = _guarded_judge(judged, input_blocked)
            rows.append(
                _result(
                    run_id,
                    case,
                    _INPUT,
                    final_route=input_outcome.route,
                    input_route=input_outcome.route,
                    output_route="",
                    blocked=input_blocked,
                    guard_ms=input_outcome.latency_ms,
                    generation_ms=item.generation_ms,
                    tier3_invoked=input_outcome.tier3_invoked,
                    judged=input_judged,
                    raw=raw,
                    final=_REFUSAL if input_blocked else raw,
                )
            )

            output_outcome: CascadeOutcome | None = None
            if not input_blocked:
                output_outcome = await cascade.run(
                    raw,
                    make_context(case, Direction.OUTPUT),
                )
            output_blocked = output_outcome is not None and _is_blocked(output_outcome)
            full_blocked = input_blocked or output_blocked
            full_judged = _guarded_judge(judged, full_blocked)
            full_route = (
                output_outcome.route
                if output_outcome is not None
                else input_outcome.route
            )
            rows.append(
                _result(
                    run_id,
                    case,
                    _FULL,
                    final_route=full_route,
                    input_route=input_outcome.route,
                    output_route="" if output_outcome is None else output_outcome.route,
                    blocked=full_blocked,
                    guard_ms=input_outcome.latency_ms
                    + (0.0 if output_outcome is None else output_outcome.latency_ms),
                    generation_ms=item.generation_ms,
                    tier3_invoked=input_outcome.tier3_invoked
                    or (
                        False
                        if output_outcome is None
                        else output_outcome.tier3_invoked
                    ),
                    judged=full_judged,
                    raw=raw,
                    final=_REFUSAL if full_blocked else raw,
                )
            )
            connection.executemany(
                """
                INSERT INTO run_result (
                    run_id, case_id, system, label, category, final_route,
                    input_route, output_route, blocked, guard_ms, generation_ms,
                    tier3_invoked, judge_complied, judge_error, raw_response,
                    final_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
            print(
                f"cascade  {index:03d}/{len(generated):03d} {case.case_id} "
                f"input={input_outcome.route} full={full_route}",
                flush=True,
            )
    finally:
        if inserted:
            connection.execute(
                "UPDATE run SET completed_ns = ? WHERE run_id = ?",
                (time.time_ns(), run_id),
            )
            connection.commit()
        connection.close()
    return run_id


def _load_reusable_baseline(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    test_cases: Sequence[EvaluationCase],
    target_model: str,
    judge_model: str,
    digest: str,
) -> dict[str, tuple[_GeneratedCase, JudgeResult]]:
    meta = connection.execute(
        "SELECT target_model, judge_model, dataset_digest FROM run WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if meta is None:
        raise ValueError(f"unknown reuse_run_id: {run_id}")
    if str(meta[0]) != target_model:
        raise ValueError("reuse run target model does not match")
    if str(meta[1]) != judge_model:
        raise ValueError("reuse run judge model does not match")
    if str(meta[2]) != digest:
        raise ValueError("reuse run dataset digest does not match")

    rows = connection.execute(
        """
        SELECT case_id, label, category, raw_response, generation_ms,
               judge_complied, judge_error
        FROM run_result
        WHERE run_id = ? AND system = 'baseline'
        """,
        (run_id,),
    ).fetchall()
    by_case = {str(row[0]): row for row in rows}
    reused: dict[str, tuple[_GeneratedCase, JudgeResult]] = {}
    for case in test_cases:
        row = by_case.get(case.case_id)
        if row is None:
            continue
        if str(row[1]) != case.label or str(row[2]) != case.category:
            raise ValueError(f"reuse row metadata mismatch: {case.case_id}")
        raw = row[3]
        generation_ms = row[4]
        complied = row[5]
        judge_error = row[6]
        if not isinstance(raw, str):
            raise ValueError(f"reuse raw response is not text: {case.case_id}")
        if isinstance(generation_ms, bool) or not isinstance(
            generation_ms, int | float
        ):
            raise ValueError(f"reuse generation time is invalid: {case.case_id}")
        if complied not in (0, 1):
            raise ValueError(f"reuse judge verdict is invalid: {case.case_id}")
        if judge_error is not None and not isinstance(judge_error, str):
            raise ValueError(f"reuse judge error is invalid: {case.case_id}")
        reused[case.case_id] = (
            _GeneratedCase(case, raw, float(generation_ms)),
            JudgeResult(bool(complied), "", judge_error),
        )
    return reused


def _load_embedding_cache(
    connection: sqlite3.Connection,
    model: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    rows = connection.execute(
        """
        SELECT text_sha256, text, dimensions, embedding
        FROM embedding_cache
        WHERE model = ?
        """,
        (model,),
    ).fetchall()
    entries: list[tuple[str, tuple[float, ...]]] = []
    for row in rows:
        stored_hash = row[0]
        text = row[1]
        dimensions = row[2]
        blob = row[3]
        if not isinstance(stored_hash, str) or not isinstance(text, str):
            raise ValueError("embedding cache metadata is invalid")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise ValueError("embedding cache dimensions are invalid")
        if not isinstance(blob, bytes) or len(blob) != dimensions * 8:
            raise ValueError("embedding cache payload is invalid")
        if stored_hash != _text_sha256(text):
            raise ValueError("embedding cache digest does not match")
        entries.append((text, struct.unpack(f"<{dimensions}d", blob)))
    return tuple(entries)


def _save_embedding_cache(
    connection: sqlite3.Connection,
    model: str,
    entries: Sequence[tuple[str, tuple[float, ...]]],
) -> None:
    rows: list[tuple[str, str, str, int, bytes]] = []
    for text, values in entries:
        if not text or not values:
            raise ValueError("embedding cache entry is invalid")
        rows.append(
            (
                model,
                _text_sha256(text),
                text,
                len(values),
                struct.pack(f"<{len(values)}d", *values),
            )
        )
    if not rows:
        return
    connection.executemany(
        """
        INSERT OR REPLACE INTO embedding_cache (
            model, text_sha256, text, dimensions, embedding
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_test(
    cases: Sequence[EvaluationCase],
    per_label: int | None,
) -> tuple[EvaluationCase, ...]:
    selected: list[EvaluationCase] = []
    for label in ("attack", "benign"):
        subset = tuple(
            case for case in cases if case.split == "test" and case.label == label
        )
        if per_label is None:
            selected.extend(subset)
        else:
            selected.extend(subset[:per_label])
    return tuple(selected)


async def _warm_target(
    target: OllamaClient,
    embedder: OllamaEmbedder,
) -> None:
    await asyncio.to_thread(
        target.generate,
        "Reply with OK.",
        num_predict=1,
    )
    await asyncio.to_thread(embedder.embed, "warmup")


async def _warm_guard(
    connector: OllamaLlamaGuardConnector,
    judge: ComplianceJudge,
    case: EvaluationCase,
) -> None:
    await connector.score("Hello", make_context(case, Direction.INPUT))
    await asyncio.to_thread(judge.judge, "Reply with OK.", "OK")


async def _generate(
    client: OllamaClient,
    prompt: str,
) -> tuple[str, float]:
    started = time.perf_counter_ns()
    response = await asyncio.to_thread(
        client.generate,
        prompt,
        num_predict=256,
    )
    return response, max(0.0, (time.perf_counter_ns() - started) / 1_000_000)


def _guarded_judge(judged: JudgeResult, blocked: bool) -> JudgeResult:
    if blocked:
        return JudgeResult(False, "guardrail_blocked")
    return judged


def _is_blocked(outcome: CascadeOutcome) -> bool:
    return outcome.route == "refused"


def _result(
    run_id: str,
    case: EvaluationCase,
    system: str,
    *,
    final_route: str,
    input_route: str,
    output_route: str,
    blocked: bool,
    guard_ms: float,
    generation_ms: float,
    tier3_invoked: bool,
    judged: JudgeResult,
    raw: str,
    final: str,
) -> tuple[object, ...]:
    return (
        run_id,
        case.case_id,
        system,
        case.label,
        case.category,
        final_route,
        input_route,
        output_route,
        int(blocked),
        guard_ms,
        generation_ms,
        int(tier3_invoked),
        int(judged.complied),
        judged.error,
        raw,
        final,
    )


__all__ = ["RunConfig", "run_evaluation"]
