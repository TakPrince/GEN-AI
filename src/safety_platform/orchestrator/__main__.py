"""CLI entrypoint: run the single-turn red-team campaign end to end.

Example:
    python -m safety_platform.orchestrator --dataset \
        evaluation/data/jbb-v2/cases.jsonl --attack-per-label 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from os import environ
from pathlib import Path

from ..app import _live_completion, _pipeline
from ..evaluation.dataset import load_cases
from ..evaluation.llm import ComplianceJudge, OllamaClient
from .loop import _run_id as _run_stamp
from .loop import run_campaign, write_jsonl, write_summary
from .mutators import STRATEGIES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety_platform.orchestrator",
        description="Single-turn red-team orchestration harness",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/data/jbb-v2/cases.jsonl",
        help="evaluation cases JSONL (label=attack|benign)",
    )
    parser.add_argument("--attack-per-label", type=int, default=5)
    parser.add_argument(
        "--strategies",
        default="direct,role_wrap,encoding,payload_split",
        help="comma-separated mutation strategies",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="evaluation/results",
        help="output directory for JSONL + summary artifacts",
    )
    parser.add_argument(
        "--judge-model",
        default="llama3.1:8b",
        help="independent compliance judge model tag",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="telemetry sqlite path (defaults to a temp file outside the repo)",
    )
    return parser


async def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    for strategy in strategies:
        if strategy not in STRATEGIES:
            parser = _parser()
            parser.error(f"unknown strategy {strategy!r}")

    cases = tuple(load_cases(Path(args.dataset)))
    attacks = sum(1 for case in cases if case.label == "attack")
    benign_cases = len(cases) - attacks
    print(
        f"loaded {len(cases)} cases from {args.dataset}: "
        f"{attacks} attack, {benign_cases} benign"
    )
    dataset_name = Path(args.dataset).parent.name

    environ["SAFETY_UI_SPECULATIVE_ECHO"] = "0"
    db_path = Path(args.db or (f"{Path(tempfile_prefix())}-redteam.sqlite3"))
    runtime = _pipeline(str(db_path))
    judge = ComplianceJudge(OllamaClient(args.judge_model))
    records, summary = await run_campaign(
        runtime,
        cases,
        judge,
        _live_completion,
        run_id=_run_stamp(),
        per_label=args.attack_per_label,
        strategies=strategies,
        seed=args.seed,
        model_metadata=(
            ("target", "qwen3:8b"),
            ("judge", args.judge_model),
            ("dataset", dataset_name),
        ),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"redteam-singleturn-{summary['run_id']}.jsonl"
    summary_path = out_dir / f"redteam-singleturn-{summary['run_id']}-summary.json"
    write_jsonl(records, jsonl_path)
    write_summary(summary, summary_path)
    metrics = summary["metrics"]
    totals = summary["totals"]
    print(
        f"scorable {totals['scorable']}/{totals['attempted']}: "
        f"ASR {metrics['asr_pct'] if metrics['asr_pct'] is not None else 'N/A'}% "
        f"(CI {metrics['asr_low_pct']}-{metrics['asr_high_pct']}%), "
        f"block {metrics['block_rate_pct']}%, "
        f"FPR {metrics['fpr_pct']}%, "
        f"judge errors {totals['judge_errors']}"
    )
    print(f"records: {jsonl_path}")
    print(f"summary: {summary_path}")
    return 0


def tempfile_prefix() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="safety-platform-")


def _cli() -> None:
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    _cli()
