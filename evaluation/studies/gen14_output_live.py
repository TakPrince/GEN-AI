"""GEN-14 / GEN-10: live output path on the stored JBB-v2 target responses.

Runs the *dashboard* output cascade (always-check output Tier 3, Ollama, default
Tier 3 model) on every stored baseline response, with the original user prompt
as context, and compares the output verdict with the stored judge label. It
does not retrain or tune anything; it measures the live configuration on the
responses that already exist.

Run from the repository root (one Tier 3 model loaded at a time):
    PYTHONPATH=src python3 evaluation/studies/gen14_output_live.py

Writes evaluation/results/phase3/gen14-output-live.json (new file).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ.pop("SAFETY_UI_TIER3", None)  # live Ollama Tier 3, not the heuristic
os.environ["SAFETY_UI_SPECULATIVE_ECHO"] = "0"
os.environ.pop("SAFETY_UI_CANARY", None)
sys.path.insert(0, "src")

from safety_platform import app
from safety_platform.config.registry import Direction, ExecutionContext
from safety_platform.evaluation.dataset import load_cases

DB = Path("evaluation/results/jbb-v2-repro.sqlite3")
OUT = Path("evaluation/results/phase3/gen14-output-live.jsonl")


async def main() -> None:
    cases = {c.case_id: c for c in load_cases("evaluation/data/jbb-v2/cases.jsonl")}
    rows = (
        sqlite3.connect(DB)
        .execute(
            "select case_id, label, raw_response, judge_complied from run_result "
            "where system = 'baseline' order by case_id"
        )
        .fetchall()
    )
    runtime = app._build_pipeline(os.path.join(tempfile.mkdtemp(), "gen14.sqlite3"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if OUT.exists():  # resume after an interruption
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["case_id"]] = record
    results = list(done.values())
    for case_id, label, raw, judged in rows:
        if case_id in done:
            continue
        case = cases[case_id]
        context = ExecutionContext(
            trace_id=f"gen14-{case_id}",
            turn_id="1",
            conversation_id=f"gen14-{case_id}",
            direction=Direction.OUTPUT,
            user_input=case.prompt,
        )
        started = time.perf_counter()
        outcome = await runtime.cascade.run(raw, context)
        record = {
            "case_id": case_id,
            "split": case.split,
            "label": label,
            "judge_complied": bool(judged) if label == "attack" else None,
            "output_route": outcome.route,
            "tier3_invoked": outcome.tier3_invoked,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        results.append(record)
        with OUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(case_id, outcome.route, outcome.tier3_invoked, flush=True)
    print("complete", len(results), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
