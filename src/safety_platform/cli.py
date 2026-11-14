"""CLI interface over the shared defense runtime (P2-04).

The CLI is an interface, not the engine: it constructs the same runtime as
the dashboard (runtime.build_runtime) and executes every prompt through the
shared defense boundary (execute_defense).

Usage:
    python -m safety_platform check "prompt" [--json]
    python -m safety_platform interactive

Exit codes: 0 allowed, 1 blocked, 2 error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

from .config.registry import DEFAULT_PAYLOAD_LIMITS, ExecutionContext
from .evaluation.llm import OllamaClient
from .execution import DefenseRequest, DefenseResult, execute_defense
from .runtime import _COMPLETION_MODEL, build_runtime

# Interactive input shares the entry-boundary bound; tier1 rejects larger input.
_MAX_INTERACTIVE_LINE_BYTES = DEFAULT_PAYLOAD_LIMITS.prompt_bytes


def default_database() -> str:
    return os.environ.get("SAFETY_CLI_DB", "safety_cli.sqlite3")


def default_completion(prompt: str) -> str:
    return OllamaClient(_COMPLETION_MODEL).generate(prompt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety_platform",
        description="Run the shared guardrail runtime from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="defend one prompt and print the verdict (exit 0/1/2)"
    )
    check.add_argument("prompt", help="prompt text to defend")
    check.add_argument("--json", action="store_true", help="machine-readable result")
    sub.add_parser("interactive", help="read prompts line by line; `exit` quits")
    return parser


def _decision(result: DefenseResult) -> str:
    if result.outcome == "error":
        return "error"
    return "block" if result.blocked else "allow"


def _print_human(result: DefenseResult, tier3_mode: str) -> None:
    decision = _decision(result).upper()
    print(f"DECISION: {decision}")
    print(f"route: {result.final_route} (stage {result.final_stage})")
    if result.target_status == "executed":
        print(f"response: {result.final_response}")
    print(f"latency: {result.latency_ms:.1f} ms · tier3: {tier3_mode}")


def _print_json(result: DefenseResult, tier3_mode: str) -> None:
    payload = {
        "decision": _decision(result),
        "route": result.final_route,
        "stage": result.final_stage,
        "latency_ms": result.latency_ms,
        "tier3_mode": tier3_mode,
        "tier3_invoked": result.tier3_invoked,
        "target_status": result.target_status,
        "response": (
            result.final_response if result.target_status == "executed" else None
        ),
        "telemetry_persisted": result.telemetry_persisted,
    }
    print(json.dumps(payload))


async def _defend(prompt: str) -> tuple[DefenseResult, str]:
    runtime = build_runtime(str(default_database()))
    request = DefenseRequest(prompt=prompt, request_id=uuid.uuid4().hex)
    context = ExecutionContext("cli", request.request_id, "cli-conversation")
    result = await execute_defense(
        runtime, request, context, completion=default_completion
    )
    return result, runtime.tier3_mode


def _exit_code(result: DefenseResult) -> int:
    if result.outcome == "error":
        return 2
    return 1 if result.blocked else 0


async def arun(args: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(args)
    if parsed.command == "check":
        result, tier3_mode = await _defend(parsed.prompt)
        if parsed.json:
            _print_json(result, tier3_mode)
        else:
            _print_human(result, tier3_mode)
        return _exit_code(result)

    # Interactive: one defended prompt per line until EOF or `exit`.
    stdin_iter = getattr(arun, "_stdin_iter", None) or (line for line in sys.stdin)
    while True:
        try:
            line = next(stdin_iter)
        except StopIteration:
            return 0
        prompt = line.rstrip("\n")
        if prompt.strip().lower() in ("exit", "quit"):
            return 0
        if not prompt.strip():
            continue
        if len(prompt.encode("utf-8")) > _MAX_INTERACTIVE_LINE_BYTES:
            # Entry-boundary parity: do not buffer or evaluate oversized input.
            print("DECISION: BLOCK")
            print("route: payload_limit (input exceeds the prompt byte bound)")
            continue
        result, tier3_mode = await _defend(prompt)
        _print_human(result, tier3_mode)
        if result.outcome == "error":
            code = _exit_code(result)
            print(f"exiting with error code {code}")
            return code


def main(args: list[str] | None = None, stdin_iter=None) -> int:
    if stdin_iter is not None:
        arun._stdin_iter = stdin_iter  # type: ignore[attr-defined]
    try:
        return asyncio.run(arun(args))
    finally:
        if stdin_iter is not None:
            del arun._stdin_iter  # type: ignore[attr-defined]


if __name__ == "__main__":
    sys.exit(main())
