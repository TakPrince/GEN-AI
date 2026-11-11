"""CLI entrypoint: durable single- and multi-turn red-team campaigns.

Example:
    python -m safety_platform.orchestrator --dataset evaluation/data/jbb-v2/cases.jsonl
    python -m safety_platform.orchestrator --campaign-status <run_id>
    python -m safety_platform.orchestrator --resume <run_id>

Exit codes: 0 completed; 1 incomplete (uncertain or pending work remains);
2 usage error; 3 campaign lease held by another live process.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from os import environ
from pathlib import Path

from ..app import _pipeline, _target_completion
from ..evaluation.dataset import load_cases
from ..evaluation.llm import ComplianceJudge, OllamaClient
from .durable import (
    CampaignError,
    CampaignResult,
    CampaignStore,
    LeaseHeldError,
    describe_campaign,
    resume_campaign,
    start_multi,
    start_single,
)
from .loop import _run_id as _run_stamp
from .mutators import STRATEGIES

EXIT_COMPLETED, EXIT_INCOMPLETE, EXIT_USAGE, EXIT_LEASE_HELD = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety_platform.orchestrator",
        description="Durable red-team orchestration (single- and multi-turn)",
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
        help="output directory for JSONL + summary artifacts and the campaign ledger",
    )
    parser.add_argument(
        "--judge-model",
        default="llama3.1:8b",
        help="independent compliance judge model tag",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="campaign ledger + telemetry sqlite (default: <out>/orchestrator.sqlite3)",
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=("single", "multi"),
        help="single: single-turn campaigns; multi: conversation campaigns",
    )
    parser.add_argument(
        "--escalations",
        default="crescendo",
        help="comma-separated multi-turn escalation strategies (mode=multi)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="per-conversation turn cap (mode=multi)",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue a campaign from its durable state (use the same --db/--out)",
    )
    parser.add_argument(
        "--campaign-status",
        metavar="RUN_ID",
        help="print a campaign's durable progress and exit",
    )
    parser.add_argument(
        "--takeover",
        action="store_true",
        help="with --resume: override a lease that may still be live "
        "(only if that process is dead)",
    )
    return parser


def _report(result: CampaignResult) -> int:
    campaign = result.summary["campaign"]
    units = campaign["units"]
    print(
        f"campaign {result.run_id}: {result.status}; units total={units['total']} "
        f"completed={units['completed']} uncertain={units['uncertain']} "
        f"pending={units['pending']}"
    )
    summary = result.summary
    if campaign["kind"] == "single":
        metrics, totals = summary["metrics"], summary["totals"]
        asr = metrics["asr_pct"] if metrics["asr_pct"] is not None else "N/A"
        print(
            f"scorable {totals['scorable']}/{totals['attempted']}: ASR {asr}% "
            f"(CI {metrics['asr_low_pct']}-{metrics['asr_high_pct']}%), "
            f"block {metrics['block_rate_pct']}%, FPR {metrics['fpr_pct']}%, "
            f"judge errors {totals['judge_errors']}"
        )
    else:
        totals = summary["totals"]
        conversation = summary["conversation_metrics"]
        print(
            f"conversations {totals['conversations']}, scorable {totals['scorable']}: "
            f"ASR {conversation['asr_pct']}% "
            f"partial {conversation['partial_compliance_pct']}%, "
            f"turn-ASR {summary['turn_metrics']['asr_pct']}%"
        )
    print(f"records: {result.paths[0]}")
    print(f"summary: {result.paths[1]}")
    return EXIT_COMPLETED if result.status == "completed" else EXIT_INCOMPLETE


def _status(db: str, run_id: str) -> int:
    if not Path(db).exists():  # never create a ledger just to report on it
        print(f"error: no campaign database at {db}", file=sys.stderr)
        return EXIT_USAGE
    store = CampaignStore(db)
    try:
        info = describe_campaign(store, run_id)
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        store.close()
    units = info["units"]
    print(f"campaign {run_id} ({info['kind']}): {info['label']}")
    print(
        f"units: total={units['total']} completed={units['completed']} "
        f"uncertain={units['uncertain']} in_flight={units['in_flight']} "
        f"pending={units['pending']}"
    )
    print(f"segments: {info['segments']}")
    if info["label"] == "interrupted":
        print(f"resume with: --resume {run_id} --db {db}")
    return EXIT_COMPLETED


async def _resume(db: str, run_id: str, *, takeover: bool) -> int:
    if not Path(db).exists():
        print(f"error: no campaign database at {db}", file=sys.stderr)
        return EXIT_USAGE
    store = CampaignStore(db)
    try:
        campaign = store.campaign(run_id)
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        store.close()
    environ["SAFETY_UI_SPECULATIVE_ECHO"] = "0"
    judge_model = dict(campaign["config"]["model_metadata"])["judge"]
    print(f"resuming campaign {run_id} ({campaign['status']})", flush=True)
    try:
        result = await resume_campaign(
            db,
            _pipeline(db),
            ComplianceJudge(OllamaClient(judge_model)),
            _target_completion,
            run_id=run_id,
            takeover=takeover,
        )
    except LeaseHeldError as exc:
        print(
            f"error: {exc}. Wait for the lease to lapse; pass --takeover only if "
            "that process is known to be dead.",
            file=sys.stderr,
        )
        return EXIT_LEASE_HELD
    return _report(result)


async def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    db = args.db or str(Path(args.out) / "orchestrator.sqlite3")
    if args.campaign_status:
        return _status(db, args.campaign_status)
    if args.resume:
        return await _resume(db, args.resume, takeover=args.takeover)

    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    for strategy in strategies:
        if strategy not in STRATEGIES:
            parser.error(f"unknown strategy {strategy!r}")
    escalations = tuple(s.strip() for s in args.escalations.split(",") if s.strip())

    cases = tuple(load_cases(Path(args.dataset)))
    attacks = sum(1 for case in cases if case.label == "attack")
    print(
        f"loaded {len(cases)} cases from {args.dataset}: "
        f"{attacks} attack, {len(cases) - attacks} benign"
    )
    dataset_name = Path(args.dataset).parent.name
    environ["SAFETY_UI_SPECULATIVE_ECHO"] = "0"
    judge = ComplianceJudge(OllamaClient(args.judge_model))
    run_id = _run_stamp()
    runtime = _pipeline(db)
    # Announced before any external request so an interrupted run can be resumed.
    print(f"campaign: {run_id}", flush=True)
    base_metadata = (
        ("target", "qwen3:8b"),
        ("judge", args.judge_model),
        ("dataset", dataset_name),
        ("tier3_mode", runtime.tier3_mode),
    )
    if args.mode == "multi":
        result = await start_multi(
            db,
            runtime,
            cases,
            judge,
            _target_completion,
            run_id=run_id,
            artifact_dir=args.out,
            per_label=args.attack_per_label,
            max_turns=args.max_turns,
            escalations=escalations,
            seed=args.seed,
            model_metadata=(*base_metadata, ("mode", "multi")),
        )
    else:
        result = await start_single(
            db,
            runtime,
            cases,
            judge,
            _target_completion,
            run_id=run_id,
            artifact_dir=args.out,
            per_label=args.attack_per_label,
            strategies=strategies,
            seed=args.seed,
            model_metadata=base_metadata,
        )
    return _report(result)


def _cli() -> None:
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    _cli()
