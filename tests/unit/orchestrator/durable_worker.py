"""Task 18 Part B: a durable campaign run in its own process (not a test module).

Usage: durable_worker.py <start|resume> <single|multi> <db> <artifact_dir>
                         <run_id> <log_dir> <none|judge_hang|checkpoint_hang>

Every target/judge call is appended to a log file so the parent test can count
real invocations across processes. A fault writes a marker file and then blocks,
so the parent can SIGKILL the process at a known point.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import (
    ComplianceVerdict,
    ConversationVerdict,
    JudgeResult,
)
from safety_platform.orchestrator.durable import (
    CampaignStore,
    resume_campaign,
    start_multi,
    start_single,
)

ANSWER = "here is a harmless answer"
# Raw mode: the output guard blocks this completion, so the raw judge stage runs.
EXPLOIT = "Sure! ignore previous instructions and here is the exploit payload."
RAW_MODE = os.environ.get("DURABLE_TEST_RAW") == "1"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))
TS = "2026-01-01T00:00:00+00:00"


def main() -> None:
    mode, kind, db, artifacts, run_id, log_dir, fault = sys.argv[1:8]
    lease_s = float(os.environ["DURABLE_TEST_LEASE_S"])
    logs = pathlib.Path(log_dir)

    def log(name: str, line: str) -> None:
        with open(logs / name, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def block(marker: str) -> None:
        (logs / marker).write_text("1", encoding="utf-8")
        time.sleep(3600)  # the parent SIGKILLs us here

    class Target:
        def __call__(self, prompt: str) -> str:
            log("target.log", json.dumps(prompt))  # one line per call
            return EXPLOIT if RAW_MODE else ANSWER

    class Judge:
        identity = "worker-judge-v1"  # the same judge in every process

        def judge(self, prompt: str, response: str) -> JudgeResult:
            raw = "exploit payload" in response
            log("judge.log", "raw" if raw else "turn")
            if fault == "judge_hang" or (fault == "raw_judge_hang" and raw):
                block("judge_started")  # the running attempt is already committed
            # Raw mode: the exploit output is the compliant one; the refusal is not.
            return JudgeResult(raw if RAW_MODE else True, "verdict", None)

        def judge_conversation(self, objective: str, transcript: str):
            log("judge.log", "conversation")
            return ConversationVerdict(ComplianceVerdict.REFUSED, "refusals", None)

    if fault == "checkpoint_hang":
        if kind == "single":
            # Judge finished and recorded; the unit is never checkpointed.
            def hang_unit(self, run_id, unit_id, result, *, owner):
                block("checkpoint_started")

            CampaignStore.complete_unit = hang_unit  # type: ignore[method-assign]
        else:
            # The first checkpoint without a pending response follows the judge.
            original = CampaignStore.save_progress

            def hang_progress(self, run_id, unit_id, progress, *, owner):
                if "pending" not in progress:
                    block("checkpoint_started")
                return original(self, run_id, unit_id, progress, owner=owner)

            CampaignStore.save_progress = hang_progress  # type: ignore[method-assign]

    pipeline = app._pipeline(db)
    if kind == "single":
        cases = [
            EvaluationCase(
                "benign-001",
                "tell me about topic 1",
                "benign",
                "animals",
                "behavior",
                "test",
                "src",
            )
        ]
    else:
        cases = [
            EvaluationCase(
                "attack-001",
                "describe approach 1",
                "attack",
                "malware",
                "behavior",
                "test",
                "src",
            )
        ]
    if mode == "resume":
        result = asyncio.run(
            resume_campaign(
                db, pipeline, Judge(), Target(), run_id=run_id, lease_s=lease_s
            )
        )
    elif kind == "single":
        result = asyncio.run(
            start_single(
                db,
                pipeline,
                cases,
                Judge(),
                Target(),
                run_id=run_id,
                artifact_dir=artifacts,
                per_label=1,
                strategies=("direct",),
                seed=0,
                model_metadata=META,
                reference_time=TS,
                lease_s=lease_s,
            )
        )
    else:
        result = asyncio.run(
            start_multi(
                db,
                pipeline,
                cases,
                Judge(),
                Target(),
                run_id=run_id,
                artifact_dir=artifacts,
                per_label=1,
                max_turns=3,
                escalations=("crescendo",),
                seed=0,
                model_metadata=META,
                reference_time=TS,
                lease_s=lease_s,
            )
        )
    print(json.dumps({"status": result.status}))


if __name__ == "__main__":
    main()
