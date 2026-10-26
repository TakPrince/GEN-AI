import time

import pytest

from safety_platform.guardrails.tier1.regex_filter import RegexAction, RegexFilter, RegexRule
from safety_platform.middleware.regex_filter import FilterContext, RegexFilterMiddleware
from safety_platform.telemetry import AsyncTelemetryBus


def p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[index]


@pytest.mark.asyncio
async def test_tier1_regex_middleware_p95_under_60ms() -> None:
    rules = [
        RegexRule("ignore", r"ignore\s+previous"),
        RegexRule("system", r"system\s+prompt"),
        RegexRule("role", r"role\s*:\s*system"),
        RegexRule("secret", r"secret=\w+", RegexAction.REDACT),
        RegexRule("key", r"key-\d+", RegexAction.REDACT),
    ]
    bus = AsyncTelemetryBus()
    middleware = RegexFilterMiddleware(RegexFilter(rules), bus)
    context = FilterContext("trace", "turn", "conversation")
    text = ("benign local prompt " * 300)[:4096]

    for _ in range(25):
        await middleware(text, context)

    samples: list[float] = []
    for _ in range(200):
        started = time.perf_counter_ns()
        await middleware(text, context)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)

    assert p95(samples) < 60.0
