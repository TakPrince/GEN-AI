from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .dataset import fetch_jbb_dataset
from .report import EvaluationReport, load_report, render_markdown
from .runner import RunConfig, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(prog="safety-eval")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch")
    fetch.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/data/jbb-v1"),
    )
    fetch.add_argument("--train-per-label", type=int, default=10)
    fetch.add_argument("--test-per-label", type=int, default=30)

    run = commands.add_parser("run")
    run.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/data/jbb-v1/cases.jsonl"),
    )
    run.add_argument(
        "--db",
        type=Path,
        default=Path("evaluation/results/jbb-v1.sqlite3"),
    )
    run.add_argument("--reuse-run", metavar="RUN_ID")
    run.add_argument("--target", default="llama3.2:latest")
    run.add_argument("--embed", default="llama3.2:latest")
    run.add_argument("--tier3", default="qwen3:8b")
    run.add_argument("--judge", default="qwen3:8b")
    run.add_argument("--per-label", type=int, default=20)
    run.add_argument("--tier2-block", type=float, default=0.80)
    run.add_argument("--tier2-floor", type=float, default=0.45)
    run.add_argument("--tier2-gap", type=float, default=0.01)
    run.add_argument("--tier3-confidence", type=float, default=0.78)
    run.add_argument("--report-json", type=Path)

    report = commands.add_parser("report")
    report.add_argument(
        "--db",
        type=Path,
        default=Path("evaluation/results/jbb-v1.sqlite3"),
    )
    report.add_argument("--run-id")
    report.add_argument("--json", type=Path)

    arguments = parser.parse_args()
    if arguments.command == "fetch":
        path = fetch_jbb_dataset(
            arguments.out,
            train_per_label=arguments.train_per_label,
            test_per_label=arguments.test_per_label,
        )
        print(path)
        return
    if arguments.command == "run":
        config = RunConfig(
            dataset_path=arguments.dataset,
            database_path=arguments.db,
            reuse_run_id=arguments.reuse_run,
            target_model=arguments.target,
            embed_model=arguments.embed,
            tier3_model=arguments.tier3,
            judge_model=arguments.judge,
            per_label=arguments.per_label if arguments.per_label > 0 else None,
            tier2_block=arguments.tier2_block,
            tier2_floor=arguments.tier2_floor,
            tier2_gap=arguments.tier2_gap,
            tier3_confidence=arguments.tier3_confidence,
        )
        run_id = asyncio.run(run_evaluation(config))
        active = load_report(str(arguments.db), run_id=run_id)
        print(render_markdown(active))
        if arguments.report_json is not None:
            _write_json(active, arguments.report_json)
        return
    active = load_report(str(arguments.db), run_id=arguments.run_id)
    print(render_markdown(active))
    if arguments.json is not None:
        _write_json(active, arguments.json)


def _write_json(report: EvaluationReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
