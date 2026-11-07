from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .dataset import fetch_jbb_dataset
from .report import EvaluationReport, load_report, render_markdown
from .runner import RunConfig, run_evaluation
from .tier3_calibration import PILOT_RUN_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safety-eval")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch")
    fetch.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/data/jbb-v2"),
    )
    fetch.add_argument("--train-per-label", type=int, default=10)
    fetch.add_argument("--test-per-label", type=int, default=30)
    fetch.add_argument(
        "--force",
        action="store_true",
        help="overwrite an already-frozen dataset directory",
    )

    run = commands.add_parser("run")
    run.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/data/jbb-v2/cases.jsonl"),
    )
    run.add_argument(
        "--db",
        type=Path,
        default=Path("evaluation/results/jbb-v2.sqlite3"),
    )
    run.add_argument("--reuse-run", metavar="RUN_ID")
    run.add_argument(
        "--speculative-echo",
        action="store_true",
        help="enable speculative context echo probe (fail-closed early refusal)",
    )
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
        default=Path("evaluation/results/jbb-v2.sqlite3"),
    )
    report.add_argument("--run-id")
    report.add_argument("--json", type=Path)

    replay = commands.add_parser("replay")
    replay.add_argument("dead_letter", type=Path)
    replay.add_argument(
        "--db",
        type=Path,
        default=Path("evaluation/results/jbb-v2.sqlite3"),
    )

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument(
        "--db",
        type=Path,
        default=Path("evaluation/results/jbb-v2.sqlite3"),
    )
    calibrate.add_argument("--run-id", default=None)
    calibrate.add_argument("--system", default="baseline")
    calibrate.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/data/jbb-v2/cases.jsonl"),
    )
    calibrate.add_argument("--tier3", default="qwen3:8b")
    calibrate.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/calibration/tier3-arbitration.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "fetch":
        path = fetch_jbb_dataset(
            arguments.out,
            train_per_label=arguments.train_per_label,
            test_per_label=arguments.test_per_label,
            overwrite=arguments.force,
        )
        print(path)
        return
    if arguments.command == "run":
        config = RunConfig(
            dataset_path=arguments.dataset,
            database_path=arguments.db,
            reuse_run_id=arguments.reuse_run,
            speculative_echo=arguments.speculative_echo,
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
    if arguments.command == "replay":
        from ..telemetry.db_sink import SqliteTelemetrySink
        from ..telemetry.replay import replay_dead_letter

        if not arguments.dead_letter.is_file():
            raise SystemExit(2)
        report = replay_dead_letter(
            SqliteTelemetrySink(str(arguments.db)),
            arguments.dead_letter,
        )
        print(
            f"replay read={report.records_read} recovered={report.events_recovered}"
            f" duplicated={report.events_duplicated}"
            f" malformed={report.malformed_records}"
            f" invalid={report.invalid_records} failed={report.events_failed}"
        )
        for failure in report.failures:
            print(f"  line {failure.line_no}: {failure.error}: {failure.detail}")
        if report.malformed_records or report.invalid_records or report.events_failed:
            raise SystemExit(1)
        return

    if arguments.command == "calibrate":
        from ..guardrails.tier3.guard_classifier import OllamaGuardConfig
        from .tier3_calibration import (
            build_artifact,
            decide,
            load_stored_responses,
            score_stored_responses,
            sweep,
        )

        responses = load_stored_responses(
            arguments.db,
            arguments.run_id or PILOT_RUN_ID,
            arguments.dataset,
            system=arguments.system,
        )
        scored = score_stored_responses(
            responses,
            OllamaGuardConfig(model=arguments.tier3, timeout_s=120.0),
        )
        entries = sweep(scored)
        decision = decide(entries)
        artifact = build_artifact(
            arguments.run_id or PILOT_RUN_ID,
            arguments.system,
            scored,
            entries,
            decision,
        )
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(artifact["decision"]))
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
