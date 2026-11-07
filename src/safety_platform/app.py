from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import streamlit as st

from metrics.db_adapter import default_database_path, load_metrics
from safety_platform.config.registry import (
    DEFAULT_CASCADE_THRESHOLDS,
    ExecutionContext,
    RoutingMargins,
    TelemetryEvent,
)
from safety_platform.evaluation.cascade import CascadeRuntime
from safety_platform.evaluation.llm import (
    OllamaClient,
    OllamaError,
    OllamaSpeculativeEchoProbe,
    SpeculativeEchoProbe,
)
from safety_platform.execution import (
    DefenseRequest,
    execute_defense,
    persist_telemetry,
    stage_latency,
    tier3_signal,
    vector_signal,
)
from safety_platform.execution import _digest as _digest_imported
from safety_platform.guardrails.router import ResponseRouter
from safety_platform.guardrails.tier1.regex_filter import (
    RegexAction,
    RegexFilter,
    RegexRule,
)
from safety_platform.guardrails.tier2.vector_filter import (
    Centroid,
    VectorFilter,
)
from safety_platform.guardrails.tier3.guard_classifier import GuardClassifier
from safety_platform.research.ml_router import (
    ModelDrivenRouter,
    Tier3BypassRouter,
    benchmark_router,
)
from safety_platform.telemetry import AsyncTelemetryBus, TelemetryConsumer
from safety_platform.telemetry.db_sink import SqliteTelemetrySink

# Test-facing aliases (tests/unit/test_app.py uses app._persist and friends).
_persist = persist_telemetry
_stage_latency = stage_latency
_tier3_signal = tier3_signal
_vector_signal = vector_signal
_digest = _digest_imported
_REFUSAL = "I cannot help with that request."

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
_DENSE_CSS = """
<style>
.block-container {padding-top: .5rem; padding-bottom: 5.5rem; max-width: 100%;}
[data-testid="stMetricValue"] {font-size: 1.15rem; line-height: 1.1;}
[data-testid="stMetricLabel"] {font-size: .62rem; letter-spacing: .07em;}
[data-testid="stMetric"] {padding: .2rem .4rem;}
[data-testid="stChatMessage"] {padding: .2rem .35rem;}
[data-testid="stVerticalBlock"] {gap: .25rem;}
[data-testid="stDataFrame"] {font-size: .72rem;}
</style>
"""

_VERIFIED_TEST_CASES = 193
_GLOBAL_ASR_PCT = 6.7
_GLOBAL_FPR_PCT = 10.0
_STAGE_ICONS = {
    "PASS": "🟢",
    "AMBIGUOUS": "🟡",
    "REDACT": "🟡",
    "BLOCK": "🔴",
    "ERROR": "🔴",
    "SKIP": "⚡",
    "BYPASS": "⚡",
}


@dataclass(frozen=True, slots=True)
class _StageRow:
    stage: str
    verdict: str
    latency_ms: float
    signal: str


@dataclass(frozen=True, slots=True)
class _Turn:
    prompt: str
    raw: str
    route: str
    stage: str
    stages: tuple[_StageRow, ...]
    router_probability: float
    tier3_invoked: bool
    persisted: bool
    routing_drift_score: float = 0.0
    token_anomaly_score: float = 0.0
    speculative_echo_status: str = "unavailable"
    db_error: str | None = None


@dataclass(frozen=True, slots=True)
class _Pipeline:
    regex: RegexFilter
    vector: VectorFilter
    guard: GuardClassifier
    ml_router: ModelDrivenRouter
    router: ResponseRouter
    sink: SqliteTelemetrySink
    bus: AsyncTelemetryBus
    consumer: TelemetryConsumer
    cascade: CascadeRuntime
    echo_probe: SpeculativeEchoProbe | None = None


class _HashingEmbedder:
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


class _Collector:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit_nowait(self, event: TelemetryEvent) -> bool:
        self.events.append(event)
        return True


def _speculative_echo_probe() -> SpeculativeEchoProbe | None:
    if os.environ.get("SAFETY_UI_SPECULATIVE_ECHO", "1") == "0":
        return None
    return OllamaSpeculativeEchoProbe(OllamaClient(_COMPLETION_MODEL))


@st.cache_resource(show_spinner=False)
def _pipeline(database: str) -> _Pipeline:
    embedder = _HashingEmbedder()
    regex = RegexFilter(
        (
            RegexRule("block.ignore_previous", r"ignore\s+(all\s+)?previous"),
            RegexRule(
                "block.system_prompt",
                r"(reveal|show|print).{0,24}(system prompt|hidden instructions)",
            ),
            RegexRule("redact.secret", r"secret=\w+", action=RegexAction.REDACT),
            RegexRule(
                "redact.email",
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                action=RegexAction.REDACT,
            ),
        )
    )
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
    guard = GuardClassifier(_LocalClassifier(), _THRESHOLDS)
    ml_router = ModelDrivenRouter(Tier3BypassRouter(model))
    router = ResponseRouter()
    sink = SqliteTelemetrySink(database)
    # Per-turn drain (max_batch_wait_ms=0): Streamlit asyncio.run()s each turn on a
    # fresh loop, so a long-lived consumer task cannot survive between reruns.
    bus = AsyncTelemetryBus()
    consumer = TelemetryConsumer(
        bus,
        sink,
        max_batch_wait_ms=0,
        # Retry once, then append failed rows to a recoverable dead-letter file.
        dead_letter_path=f"{database}.dead-letter.jsonl",
    )
    return _Pipeline(
        regex=regex,
        vector=vector,
        guard=guard,
        ml_router=ml_router,
        router=router,
        sink=sink,
        bus=bus,
        consumer=consumer,
        cascade=CascadeRuntime(
            regex=regex,
            vector=vector,
            guard=guard,
            ml_router=ml_router,
            router=router,
        ),
        echo_probe=_speculative_echo_probe(),
    )


@st.cache_data(ttl=10.0, show_spinner=False)
def _ollama_status() -> str:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=1) as response:
            models = len(json.loads(response.read().decode("utf-8"))["models"])
        return f"🟢 Local-First Ollama · {models} models"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "🟡 Local-First · stub pipeline"


def _render_status_ribbon() -> None:
    columns = st.columns(4, gap="small")
    columns[0].metric("VERIFIED TEST CASES", str(_VERIFIED_TEST_CASES))
    columns[1].metric("GLOBAL ASR", f"{_GLOBAL_ASR_PCT:.1f}%")
    columns[2].metric("GLOBAL FPR", f"{_GLOBAL_FPR_PCT:.1f}%")
    columns[3].metric("SERVER STATUS", _ollama_status())


@st.fragment(run_every=1.0)
def _render_metrics(database: str) -> None:
    metrics = load_metrics(database)
    columns = st.columns(6, gap="small")
    columns[0].metric("PASSED", f"{metrics.passed_cases}/{metrics.total_cases}")
    columns[1].metric("ASR", f"{metrics.asr_pct:.1f}%")
    columns[2].metric("FPR", f"{metrics.fpr_pct:.1f}%")
    columns[3].metric("AVG MS", f"{metrics.avg_overhead_ms:.2f}")
    columns[4].metric("P95 MS", f"{metrics.p95_overhead_ms:.2f}")
    columns[5].metric("P99 MS", f"{metrics.p99_overhead_ms:.2f}")


async def _evaluate(
    runtime: _Pipeline,
    prompt: str,
    context: ExecutionContext,
) -> _Turn:
    result = await execute_defense(
        runtime,
        DefenseRequest(prompt=prompt, request_id=uuid.uuid4().hex),
        context,
        completion=_live_completion,
    )
    stages = tuple(
        _StageRow(row.stage, row.verdict, row.latency_ms, row.signal)
        for row in result.stage_observations
    )
    return _Turn(
        prompt=prompt,
        raw=result.final_response,
        route=result.final_route,
        stage=result.final_stage,
        stages=stages,
        router_probability=result.router_probability,
        tier3_invoked=result.tier3_invoked,
        persisted=result.telemetry_persisted,
        routing_drift_score=result.routing_drift_score,
        token_anomaly_score=result.token_anomaly_score,
        speculative_echo_status=result.speculative_echo_status,
        db_error=result.telemetry_error,
    )


_COMPLETION_MODEL = "qwen3:8b"


@lru_cache(maxsize=64)
def _live_completion(prompt: str) -> str:
    # A failed or empty model call must never be masked by an echo of the
    # prompt: let the exception propagate so execute_defense records a real
    # target failure (outcome=error, failure_stage=target).
    completion = OllamaClient(_COMPLETION_MODEL).generate(
        prompt, think=False, num_predict=16
    )
    if not completion.strip():
        raise OllamaError("empty_target_completion")
    return completion.strip()


def _turns() -> list[_Turn]:
    value = st.session_state.get("turns")
    if isinstance(value, list):
        return cast(list[_Turn], value)
    turns: list[_Turn] = []
    st.session_state["turns"] = turns
    return turns


def _session_id() -> str:
    value = st.session_state.get("session_id")
    if isinstance(value, str):
        return value
    session_id = uuid.uuid4().hex[:12]
    st.session_state["session_id"] = session_id
    return session_id


def _next_case() -> int:
    value = st.session_state.get("case_seq", 0)
    sequence = value if isinstance(value, int) and not isinstance(value, bool) else 0
    st.session_state["case_seq"] = sequence + 1
    return sequence


def _render_raw(turns: Sequence[_Turn]) -> None:
    with st.container(height=480, border=True):
        for turn in turns:
            with st.chat_message("user"):
                st.write(turn.prompt)
            with st.chat_message("assistant"):
                st.code(turn.raw, language="text")


def _telemetry_strip(turn: _Turn) -> str:
    return (
        f"DRIFT {turn.routing_drift_score:.2f} · "
        f"TOKEN {turn.token_anomaly_score:.2f} · "
        f"ECHO {turn.speculative_echo_status}"
    )


def _render_cascade(turns: Sequence[_Turn]) -> None:
    with st.container(height=480, border=True):
        for index, turn in enumerate(turns, start=1):
            lines = [f"`{index:02d}` **{turn.route.upper()}**"]
            for row in turn.stages:
                icon = _STAGE_ICONS.get(row.verdict.upper(), "⚪")
                latency = (
                    f" · {row.latency_ms:.2f} ms" if row.latency_ms else " · <1 ms"
                )
                signal = f" · {row.signal}" if row.signal and row.signal != "-" else ""
                lines.append(
                    f"{icon} `{row.stage}` **{row.verdict.upper()}**{latency}{signal}"
                )
            lines.append(_telemetry_strip(turn))
            st.markdown("\n\n".join(lines))
            if not turn.persisted:
                st.caption(f"DB {turn.db_error or 'ERROR'}")


def _demo_database_path() -> Path:
    return default_database_path().with_name("dashboard_demo.sqlite3")


def main() -> None:
    st.set_page_config(
        page_title="Safety Cascade",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_DENSE_CSS, unsafe_allow_html=True)
    database = default_database_path().as_posix()
    _render_status_ribbon()
    _render_metrics(database)
    st.caption(
        "Metrics are a benchmark snapshot read from metrics/telemetry.sqlite3. "
        "Demo turns persist to dashboard_demo.sqlite3 and do not update them."
    )

    left, right = st.columns(2, gap="small")
    with left:
        st.markdown(":blue-background[COLUMN 1: PROMPT // GUARDED MODEL OUTPUT]")
        _render_raw(_turns())
    with right:
        st.markdown(":green-background[COLUMN 2: GUARDED PIPELINE // ML ROUTER]")
        st.caption(
            "Demo pipeline: input guards (Tier 1/2 + echo probe + ML router) run "
            "before the target model; blocked, refused, or errored requests "
            "make zero target calls, and every completion passes the output "
            "guard before display. Tier 2 embeddings and Tier 3 verdicts use "
            "local stub models (not the calibrated qwen3 benchmark components)."
        )
        _render_cascade(_turns())

    label = st.radio(
        "CASE",
        ("attack", "benign"),
        horizontal=True,
        label_visibility="collapsed",
        width="stretch",
    )
    prompt = st.chat_input("message")
    if prompt:
        selected = "benign" if label == "benign" else "attack"
        sequence = _next_case()
        context = ExecutionContext(
            trace_id=uuid.uuid4().hex,
            turn_id=f"turn-{sequence}",
            conversation_id=f"{selected}-{_session_id()}-{sequence}",
            guardrail_version="dashboard-v1",
        )
        turn = asyncio.run(
            _evaluate(_pipeline(str(_demo_database_path())), prompt, context)
        )
        _turns().append(turn)
        st.rerun()


if __name__ == "__main__":
    main()
