from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import streamlit as st

from metrics.db_adapter import default_database_path, load_metrics
from safety_platform.config.registry import (
    DEFAULT_CASCADE_THRESHOLDS,
    ExecutionContext,
    RoutingMargins,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)
from safety_platform.guardrails.router import ResponseRouter, StageDecision
from safety_platform.guardrails.tier1.regex_filter import (
    RegexAction,
    RegexFilter,
    RegexRule,
)
from safety_platform.guardrails.tier2.vector_filter import (
    Centroid,
    VectorFilter,
    VectorFilterResult,
)
from safety_platform.guardrails.tier3.guard_classifier import GuardClassifier
from safety_platform.middleware.regex_filter import RegexFilterMiddleware
from safety_platform.middleware.vector_filter import VectorFilterMiddleware
from safety_platform.research.ml_router import (
    LogisticRegressionRouter,
    RouterFeatures,
    Tier3BypassRouter,
)
from safety_platform.telemetry.db_sink import SqliteTelemetrySink

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
    db_error: str | None = None


@dataclass(frozen=True, slots=True)
class _Pipeline:
    regex: RegexFilter
    vector: VectorFilter
    guard: GuardClassifier
    ml_router: Tier3BypassRouter
    router: ResponseRouter
    sink: SqliteTelemetrySink


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
    model = LogisticRegressionRouter(
        weights=(1.5, 1.0, -0.3, 3.0, 0.2),
        bias=-2.5,
        threshold=0.50,
    )
    return _Pipeline(
        regex=regex,
        vector=vector,
        guard=GuardClassifier(_LocalClassifier(), _THRESHOLDS),
        ml_router=Tier3BypassRouter(model),
        router=ResponseRouter(),
        sink=SqliteTelemetrySink(database),
    )


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
    collector = _Collector()
    tier1 = RegexFilterMiddleware(runtime.regex, collector)
    tier2 = VectorFilterMiddleware(runtime.vector, collector)
    rows: list[_StageRow] = []

    first = await tier1(prompt, context)
    rows.append(
        _StageRow(
            "TIER1",
            first.verdict.value,
            _stage_latency(collector.events, Stage.TIER1),
            ",".join(first.rule_ids) or "-",
        )
    )
    decisions = [
        StageDecision(
            stage=Stage.TIER1,
            verdict=first.verdict,
            reason_codes=first.rule_ids,
            redacted_text=first.redacted_text,
            error=first.error,
        )
    ]

    if first.verdict in (SafetyVerdict.BLOCK, SafetyVerdict.ERROR):
        route = runtime.router.route(context, decisions)
        rows.append(_StageRow("ML", "SKIP", 0.0, "-"))
        persisted, db_error = await _persist(runtime, collector.events)
        return _Turn(
            prompt=prompt,
            raw=_raw_completion(prompt),
            route=route.route.value,
            stage=route.stage.value,
            stages=tuple(rows),
            router_probability=0.0,
            tier3_invoked=False,
            persisted=persisted,
            db_error=db_error,
        )

    second = await tier2(prompt, context)
    rows.append(
        _StageRow(
            "TIER2",
            second.verdict.value,
            _stage_latency(collector.events, Stage.TIER2),
            _vector_signal(second),
        )
    )
    decisions.append(
        StageDecision(
            stage=Stage.TIER2,
            verdict=second.verdict,
            reason_codes=(
                (f"centroid:{second.top_centroid_id}",)
                if second.top_centroid_id is not None
                and second.verdict is not SafetyVerdict.PASS
                else ()
            ),
            error=second.error,
        )
    )

    decision = runtime.ml_router.route(
        RouterFeatures(
            regex_hit=bool(first.rule_ids),
            top_similarity=second.top_similarity,
            top_gap=second.top_gap,
            tier2_ambiguous=second.verdict is SafetyVerdict.AMBIGUOUS,
            prompt_bytes=len(prompt.encode("utf-8", "surrogatepass")),
        )
    )
    rows.append(
        _StageRow(
            "ML",
            "TIER3" if decision.invoke_tier3 else "BYPASS",
            0.0,
            f"p={decision.probability:.3f}",
        )
    )

    tier3_invoked = False
    if second.verdict is SafetyVerdict.AMBIGUOUS and decision.invoke_tier3:
        started = time.perf_counter_ns()
        third = await runtime.guard.classify_if_ambiguous(
            prompt,
            context,
            second.verdict,
        )
        latency_ms = max(0.0, (time.perf_counter_ns() - started) / 1_000_000)
        if third is not None:
            tier3_invoked = True
            rows.append(
                _StageRow(
                    "TIER3",
                    third.verdict.value,
                    latency_ms,
                    _tier3_signal(third.confidence, third.error),
                )
            )
            decisions.append(
                StageDecision(
                    stage=Stage.TIER3,
                    verdict=third.verdict,
                    reason_codes=third.reason_codes,
                    error=third.error,
                )
            )
            collector.emit_nowait(
                TelemetryEvent(
                    event_id=uuid.uuid4().hex,
                    context=context,
                    stage=Stage.TIER3,
                    verdict=third.verdict,
                    latency_ms=latency_ms,
                    prompt_hash=_digest(prompt),
                    timestamp_ns=time.time_ns(),
                    rule_ids=third.reason_codes,
                    error=third.error,
                )
            )

    route = runtime.router.route(context, decisions)
    persisted, db_error = await _persist(runtime, collector.events)
    return _Turn(
        prompt=prompt,
        raw=_raw_completion(prompt),
        route=route.route.value,
        stage=route.stage.value,
        stages=tuple(rows),
        router_probability=decision.probability,
        tier3_invoked=tier3_invoked,
        persisted=persisted,
        db_error=db_error,
    )


async def _persist(
    runtime: _Pipeline,
    events: Sequence[TelemetryEvent],
) -> tuple[bool, str | None]:
    try:
        await asyncio.to_thread(runtime.sink.append_many, events)
    except Exception as exc:
        return False, type(exc).__name__
    return True, None


def _stage_latency(events: Sequence[TelemetryEvent], stage: Stage) -> float:
    for event in reversed(events):
        if event.stage is stage:
            return event.latency_ms
    return 0.0


def _vector_signal(result: VectorFilterResult) -> str:
    centroid = result.top_centroid_id or "-"
    return f"{centroid} {result.top_similarity:.2f}/{result.top_gap:.2f}"


def _tier3_signal(confidence: float | None, error: str | None) -> str:
    if error is not None:
        return error
    if confidence is None:
        return "-"
    return f"conf={confidence:.3f}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _raw_completion(prompt: str) -> str:
    return prompt


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


def _frames(turn: _Turn) -> list[dict[str, str]]:
    return [
        {
            "STAGE": row.stage,
            "VERDICT": row.verdict.upper(),
            "MS": f"{row.latency_ms:.2f}" if row.latency_ms else "-",
            "SIGNAL": row.signal,
        }
        for row in turn.stages
    ]


def _render_raw(turns: Sequence[_Turn]) -> None:
    with st.container(height=480, border=True):
        for turn in turns:
            with st.chat_message("user"):
                st.write(turn.prompt)
            with st.chat_message("assistant"):
                st.code(turn.raw, language="text")


def _render_cascade(turns: Sequence[_Turn]) -> None:
    with st.container(height=480, border=True):
        for index, turn in enumerate(turns, start=1):
            st.markdown(
                f"`{index:02d}` **{turn.route.upper()}** `{turn.stage.upper()}`"
            )
            st.dataframe(
                _frames(turn),
                hide_index=True,
                width="stretch",
            )
            if not turn.persisted:
                st.caption(f"DB {turn.db_error or 'ERROR'}")


def main() -> None:
    st.set_page_config(
        page_title="Safety Cascade",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_DENSE_CSS, unsafe_allow_html=True)
    database = default_database_path().as_posix()
    _render_metrics(database)

    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("##### RAW // UNSHIELDED")
        _render_raw(_turns())
    with right:
        st.markdown("##### CASCADE // ML ROUTER")
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
        turn = asyncio.run(_evaluate(_pipeline(database), prompt, context))
        _turns().append(turn)
        st.rerun()


if __name__ == "__main__":
    main()
