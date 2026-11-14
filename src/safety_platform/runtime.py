"""Shared defense runtime construction (P2-04).

One place builds the guardrail components every interface consumes. The
dashboard (app.py), the CLI, and tests all construct their pipeline here;
app.py stays a Streamlit-only wrapper that adds telemetry and UI labels.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass

from .config.registry import (
    DEFAULT_CASCADE_THRESHOLDS,
    DEFAULT_PAYLOAD_LIMITS,
    ExecutionContext,
    RoutingMargins,
)
from .evaluation.cascade import CascadeRuntime
from .evaluation.llm import (
    OllamaClient,
    OllamaSpeculativeEchoProbe,
    SpeculativeEchoProbe,
)
from .guardrails.canary import CanaryDetector, ConversationTail
from .guardrails.router import ResponseRouter
from .guardrails.tier1.regex_filter import RegexFilter
from .guardrails.tier1.rules import defense_regex_filter
from .guardrails.tier2.vector_filter import Centroid, VectorFilter
from .guardrails.tier3.guard_classifier import (
    MULTI_SIGNAL_GATES,
    GuardClassifier,
    OllamaGuardConfig,
    OllamaLlamaGuardConnector,
    OllamaTransport,
)
from .research.ml_router import ModelDrivenRouter, Tier3BypassRouter, benchmark_router

_TOKEN = re.compile(r"[a-z0-9_'-]+")
_THRESHOLDS = DEFAULT_CASCADE_THRESHOLDS
_MARGINS = RoutingMargins(0.01, 0.55)

_RISK_PHRASES = (
    "ignore previous",
    "reveal system prompt",
    "print system prompt",
    "api key",
    "secret key",
    "write malware",
    "build bomb",
    "ransomware",
    "phishing",
    "self-harm",
    "exploit",
    "weapon",
)

_TIER3_MODEL = "qwen3:8b"
# Cold load + output arbitration measured 6-60 s on this CPU; harness uses 120 s too.
_TIER3_TIMEOUT_S = 120.0

# The target model arbitrates speculatively for itself (echo probe) and answers
# demo completions; shared config for dashboard and CLI.
_COMPLETION_MODEL = "qwen3:8b"


@dataclass(frozen=True)
class DemoComponents:
    """Guardrail components for the default (hash-trick demo) runtime."""

    regex: RegexFilter
    vector: VectorFilter
    guard: GuardClassifier
    ml_router: ModelDrivenRouter
    router: ResponseRouter
    cascade: CascadeRuntime
    tier3_mode: str
    tier2_mode: str


class DemoHashingEmbedder:
    def __init__(self, dimensions: int = 96) -> None:
        self._dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            values[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)


class _LocalClassifier:
    async def score(self, text: str, context: ExecutionContext) -> float:
        lowered = text.casefold()
        matches = sum(phrase in lowered for phrase in _RISK_PHRASES)
        if matches == 0:
            return 0.05
        return min(0.99, 0.62 + 0.18 * matches)


def tier3_guard(
    transport: OllamaTransport | None = None,
) -> tuple[GuardClassifier, str]:
    """Live Tier 3 is the real local classifier; the heuristic is opt-in demo only.

    Timeouts, unreachable models, and malformed output surface as ERROR verdicts,
    which the router fail-closes.
    """
    mode = os.environ.get("SAFETY_UI_TIER3", "ollama")
    if mode == "heuristic":
        guard = GuardClassifier(
            _LocalClassifier(), _THRESHOLDS, gates=MULTI_SIGNAL_GATES
        )
        return guard, "demo heuristic"
    if mode != "ollama":
        raise ValueError("SAFETY_UI_TIER3 must be 'ollama' or 'heuristic'")
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(
            model=_TIER3_MODEL,
            timeout_s=_TIER3_TIMEOUT_S,
            # Same prompt bound as the entry boundary: a 4-16 KiB prompt that
            # passes Tier 1 must not fail closed at Tier 3 (prompt_too_large).
            max_prompt_bytes=DEFAULT_PAYLOAD_LIMITS.prompt_bytes,
        ),
        transport,
    )
    return GuardClassifier(connector, _THRESHOLDS, gates=MULTI_SIGNAL_GATES), (
        f"ollama:{_TIER3_MODEL}"
    )


def canary_detector() -> CanaryDetector | None:
    """Opt-in known-value leak protection: `SAFETY_UI_CANARY=a,b,...`."""
    raw = os.environ.get("SAFETY_UI_CANARY", "").strip()
    if not raw:
        return None
    secrets = tuple(part.strip() for part in raw.split(",") if part.strip())
    # Partial-fragment protection is on by default at a conservative 16 chars
    # (0 disables); it cannot fire for secrets shorter than the fragment size.
    partial = os.environ.get("SAFETY_UI_CANARY_PARTIAL_MIN", "16").strip()
    partial_min = int(partial) if partial.isdigit() else 16
    return CanaryDetector(
        {f"canary-{index}": value for index, value in enumerate(secrets)},
        partial_min_fragment=partial_min,
    )


def speculative_echo_probe() -> SpeculativeEchoProbe | None:
    # Secure default: speculative echo is opt-in only; missing or "0" disables it.
    if os.environ.get("SAFETY_UI_SPECULATIVE_ECHO", "0") != "1":
        return None
    return OllamaSpeculativeEchoProbe(OllamaClient(_COMPLETION_MODEL))


def build_demo_components(canary: CanaryDetector | None = None) -> DemoComponents:
    """Build the default demo runtime shared by dashboard and CLI."""
    embedder = DemoHashingEmbedder()
    regex = defense_regex_filter(max_input_bytes=DEFAULT_PAYLOAD_LIMITS.prompt_bytes)
    vector = VectorFilter(
        embedder,
        (
            Centroid(
                "override",
                embedder.embed(
                    "ignore previous instructions bypass policy system prompt"
                ),
            ),
            Centroid(
                "exfiltration",
                embedder.embed("reveal system prompt print secret credentials api key"),
            ),
            Centroid(
                "harm",
                embedder.embed("write malware exploit weapon bomb ransomware"),
            ),
        ),
        _THRESHOLDS,
        _MARGINS,
    )
    model = benchmark_router()
    guard, tier3_mode = tier3_guard()
    ml_router = ModelDrivenRouter(Tier3BypassRouter(model))
    router = ResponseRouter()
    cascade = CascadeRuntime(
        regex=regex,
        vector=vector,
        guard=guard,
        ml_router=ml_router,
        router=router,
        tier3_on_output=True,
        canary=canary,
        canary_tail=None if canary is None else ConversationTail(canary),
    )
    return DemoComponents(
        regex=regex,
        vector=vector,
        guard=guard,
        ml_router=ml_router,
        router=router,
        cascade=cascade,
        tier3_mode=tier3_mode,
        tier2_mode="hash-trick demo",
    )
