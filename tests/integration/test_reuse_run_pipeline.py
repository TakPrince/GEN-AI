"""End-to-end verification of --reuse-run through the full evaluation pipeline."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

import safety_platform.evaluation.runner as runner_module
from safety_platform.config.registry import ExecutionContext
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.runner import RunConfig, run_evaluation

_CASE_FIELDS = (
    "case_id",
    "prompt",
    "label",
    "category",
    "behavior",
    "split",
    "source",
)


def _case(case_id: str, prompt: str, label: str, split: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        prompt=prompt,
        label=label,
        category="injection" if label == "attack" else "qa",
        behavior="behavior",
        split=split,
        source="jbb-v2",
    )


_DATASET = (
    _case("a0", "exploit prompt train probe", "attack", "train"),
    _case("a1", "exploit prompt one", "attack", "test"),
    _case("b1", "write a haiku one", "benign", "test"),
)

_JUDGE_REPLY = '{"complied": true, "reason": "harmless probe"}'


class ScriptedOllamaClient:
    generated: int = 0

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:9",
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        think: bool | None = None,
        num_predict: int = 256,
        format: str | None = None,
    ) -> str:
        if think is False:
            return _JUDGE_REPLY
        type(self).generated += 1
        return f"resp-{prompt}"

    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((float(len(text) % 3) + 1.0, 1.0) for text in texts)


class ScoreConnectorStub:
    scored: int = 0

    def __init__(self, config: object) -> None:
        self.config = config

    async def score(self, text: str, context: ExecutionContext) -> float:
        type(self).scored += 1
        return 0.05


@pytest.fixture()
def runner_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> RunConfig:
    ScriptedOllamaClient.generated = 0
    ScoreConnectorStub.scored = 0
    monkeypatch.setattr(runner_module, "OllamaClient", ScriptedOllamaClient)
    monkeypatch.setattr(
        runner_module,
        "OllamaLlamaGuardConnector",
        ScoreConnectorStub,
    )
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({field: getattr(case, field) for field in _CASE_FIELDS}) + "\n"
            for case in _DATASET
        ),
        encoding="utf-8",
    )
    return RunConfig(
        dataset_path=dataset,
        database_path=tmp_path / "results.sqlite3",
        per_label=1,
        reuse_run_id=None,
    )


def _rows(database: Path, run_id: str) -> dict[tuple[str, str], tuple[object, ...]]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT case_id, system, final_route, input_route, output_route,
                   raw_response, generation_ms
            FROM run_result
            WHERE run_id = ?
            ORDER BY case_id, system
            """,
            (run_id,),
        ).fetchall()
        completed = connection.execute(
            "SELECT COUNT(*) FROM run WHERE completed_ns IS NOT NULL"
        ).fetchone()
    finally:
        connection.close()
    assert completed is not None and completed[0] >= 1
    return {(str(row[0]), str(row[1])): row[2:] for row in rows}


def test_reuse_run_skips_generation_and_replays_results(runner_env: RunConfig) -> None:
    run_one = asyncio.run(run_evaluation(runner_env))
    assert run_one
    # warm-up target call + one baseline per test case
    assert ScriptedOllamaClient.generated == 3
    first_rows = _rows(runner_env.database_path, run_one)
    assert set(first_rows) == {
        ("a1", "baseline"),
        ("a1", "input"),
        ("a1", "full"),
        ("b1", "baseline"),
        ("b1", "input"),
        ("b1", "full"),
    }

    run_two = asyncio.run(
        run_evaluation(
            RunConfig(
                dataset_path=runner_env.dataset_path,
                database_path=runner_env.database_path,
                per_label=1,
                reuse_run_id=run_one,
            )
        )
    )

    assert run_two != run_one
    # full reuse: no regeneration and even the target warm-up is skipped
    assert ScriptedOllamaClient.generated == 3
    second_rows = _rows(runner_env.database_path, run_two)
    assert set(second_rows) == set(first_rows)
    for key, first_values in first_rows.items():
        assert second_rows[key] == first_values, key
