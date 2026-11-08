from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import streamlit as st

# Self-locate the src/ tree and the repo root (metrics package) so the dashboard
# launches from any cwd or invocation, not only via `make ui`'s PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
for _bootstrap in (str(_repo_root / "src"), str(_repo_root)):
    if _bootstrap not in sys.path:
        sys.path.append(_bootstrap)

from metrics.db_adapter import default_database_path, load_metrics  # noqa: E402
from safety_platform.config.registry import (  # noqa: E402
    DEFAULT_CASCADE_THRESHOLDS,
    ExecutionContext,
    RoutingMargins,
    TelemetryEvent,
)
from safety_platform.evaluation.cascade import CascadeRuntime  # noqa: E402
from safety_platform.evaluation.dataset import EvaluationCase, load_cases  # noqa: E402
from safety_platform.evaluation.llm import (  # noqa: E402
    ComplianceJudge,
    OllamaClient,
    OllamaError,
    OllamaSpeculativeEchoProbe,
    SpeculativeEchoProbe,
)
from safety_platform.execution import (  # noqa: E402
    DefenseRequest,
    execute_defense,
    persist_telemetry,
    stage_latency,
    tier3_signal,
    vector_signal,
)
from safety_platform.execution import _digest as _digest_imported  # noqa: E402
from safety_platform.guardrails.router import ResponseRouter  # noqa: E402
from safety_platform.guardrails.tier1.regex_filter import (  # noqa: E402
    RegexAction,
    RegexFilter,
    RegexRule,
)
from safety_platform.guardrails.tier2.vector_filter import (  # noqa: E402
    Centroid,
    VectorFilter,
)
from safety_platform.guardrails.tier3.guard_classifier import (  # noqa: E402
    GuardClassifier,
)
from safety_platform.orchestrator.durable import (  # noqa: E402
    CampaignStore,
    describe_campaign,
    start_multi,
    start_single,
)
from safety_platform.research.ml_router import (  # noqa: E402
    ModelDrivenRouter,
    Tier3BypassRouter,
    benchmark_router,
)
from safety_platform.telemetry import AsyncTelemetryBus, TelemetryConsumer  # noqa: E402
from safety_platform.telemetry.db_sink import SqliteTelemetrySink  # noqa: E402

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


@dataclass(frozen=True, slots=True)
class _StageRow:
    stage: str
    verdict: str
    latency_ms: float
    signal: str


@dataclass(frozen=True, slots=True)
class _ConversationTurn:
    turn_number: int
    prompt: str
    outcome: str
    blocked: bool
    final_route: str
    judge_complied: bool | None
    target_input: str | None


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
    outcome: str = "safe"
    blocked: bool = False
    target_status: str = "skipped"
    target_input: str | None = None
    raw_response: str | None = None
    error: str | None = None
    failure_stage: str | None = None
    latency_ms: float = 0.0
    decisions: tuple[tuple[str, str, str, str], ...] = ()
    judge_complied: bool | None = None
    judge_error: str | None = None
    termination_reason: str | None = None
    conversation_turns: tuple[  # payload dicts, never dataclasses (AppTest reruns)
        dict[str, int | str | bool | None], ...
    ] = ()


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
    # Secure default: speculative echo is opt-in only; missing or "0" disables it.
    if os.environ.get("SAFETY_UI_SPECULATIVE_ECHO", "0") != "1":
        return None
    return OllamaSpeculativeEchoProbe(OllamaClient(_COMPLETION_MODEL))


def _build_pipeline(database: str) -> _Pipeline:
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


@st.cache_resource(show_spinner=False)
def _pipeline(database: str) -> _Pipeline:
    return _build_pipeline(database)


@st.cache_data(ttl=10.0, show_spinner=False)
def _ollama_status() -> str:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=1) as response:
            models = len(json.loads(response.read().decode("utf-8"))["models"])
        return f"🟢 Local-First Ollama · {models} models"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "🟡 Local-First · stub pipeline"


def _render_metrics_row(database: str) -> None:
    """Benchmark snapshot metrics (real data from the telemetry projection)."""
    metrics = load_metrics(database)
    columns = st.columns(4, gap="small")
    columns[0].metric(
        "BENCHMARK CASES",
        f"{metrics.total_cases}" if metrics.total_cases else "—",
    )
    columns[1].metric("ASR", f"{metrics.asr_pct:.1f}%" if metrics.total_cases else "—")
    columns[2].metric("FPR", f"{metrics.fpr_pct:.1f}%" if metrics.total_cases else "—")
    columns[3].metric(
        "AVG OVERHEAD",
        f"{metrics.avg_overhead_ms:.2f} ms" if metrics.total_cases else "—",
    )


_COMPLETION_MODEL = "qwen3:8b"
_JUDGE_MODEL = "llama3.1:8b"
_PIPELINE_STAGES = (
    "Tier 1",
    "Tier 2",
    "ML Router",
    "Tier 3",
    "Target LLM",
    "Output Guard",
)

_RESULTS_ROOT = Path(__file__).parents[2] / "evaluation" / "results"


def _campaign_db_path() -> Path:
    override = os.environ.get("SAFETY_CAMPAIGN_DB")
    if override:
        return Path(override).expanduser()
    return _RESULTS_ROOT / "orchestrator.sqlite3"


def _dataset_files() -> list[str]:
    root = Path(__file__).parents[2] / "evaluation" / "data"
    return sorted(str(p) for p in root.glob("*/cases.jsonl"))


def _target_completion(prompt: str) -> str:
    completion = OllamaClient(_COMPLETION_MODEL).generate(
        prompt, think=False, num_predict=16
    )
    if not completion or not completion.strip():
        raise OllamaError("empty_target_completion")
    return completion.strip()


@lru_cache(maxsize=64)
def _live_completion(prompt: str) -> str:
    # Interactive chat only: identical prompts reuse the last response.
    return _target_completion(prompt)


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
    decisions = tuple(
        (
            d.stage.value,
            d.verdict.value,
            ";".join(d.reason_codes) or "-",
            d.error or "-",
        )
        for d in result.decisions
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
        outcome=result.outcome,
        blocked=result.blocked,
        target_status=result.target_status,
        target_input=result.target_input,
        raw_response=result.raw_response,
        error=result.error,
        failure_stage=result.failure_stage,
        latency_ms=result.latency_ms,
        decisions=decisions,
    )


def _build_judge():
    return ComplianceJudge(OllamaClient(_JUDGE_MODEL))


def _multi_turn(prompt: str, max_turns: int) -> _Turn:
    """Adversarial conversation via the real orchestrator run_conversation."""
    from safety_platform.orchestrator.multi import run_conversation

    sequence = _next_case()
    runtime = _pipeline(str(_demo_database_path()))
    case = EvaluationCase(
        f"live-{sequence}", prompt, "attack", "adversarial", "behavior", "test", "live"
    )
    outcome = asyncio.run(
        run_conversation(
            runtime,
            case,
            _build_judge(),
            _target_completion,
            run_id=f"live-{_session_id()}-{sequence}",
            escalation="crescendo",
            max_turns=max_turns,
            seed=0,
        )
    )
    conversation = tuple(
        {
            "turn_number": t.turn_number,
            "prompt": t.prompt,
            "outcome": t.outcome,
            "blocked": t.blocked,
            "final_route": t.final_route,
            "judge_complied": t.judge_complied,
            "target_input": t.target_input,
        }
        for t in outcome.turns
    )
    return _Turn(
        prompt=prompt,
        raw=outcome.turns[-1].final_response or "" if outcome.turns else "",
        route=outcome.status,
        stage=outcome.termination_reason or outcome.status,
        stages=(),
        router_probability=0.0,
        tier3_invoked=bool(outcome.turns),
        persisted=True,
        outcome=outcome.status,
        blocked=any(t.blocked for t in outcome.turns),
        target_status="executed" if outcome.turns else "skipped",
        raw_response=outcome.turns[-1].raw_response if outcome.turns else None,
        target_input=outcome.turns[-1].target_input if outcome.turns else None,
        error=outcome.conversation_error or None,
        judge_complied=outcome.conversation_verdict == "partial_compliance",
        judge_error=outcome.conversation_error or None,
        termination_reason=outcome.termination_reason,
        conversation_turns=conversation,
    )


def _turns() -> list[dict[str, object]]:
    # Plain dicts: Streamlit re-executes the script every run, so dataclass
    # class objects cannot be re-identified across reruns.
    value = st.session_state.get("turns")
    if isinstance(value, list):
        return cast(list[dict[str, object]], value)
    turns: list[dict[str, object]] = []
    st.session_state["turns"] = turns
    return turns


def _turn_payload(turn: _Turn) -> dict[str, object]:
    import dataclasses

    payload = dataclasses.asdict(turn)
    payload["stages"] = [
        {
            "stage": row.stage,
            "verdict": row.verdict,
            "latency_ms": row.latency_ms,
            "signal": row.signal,
        }
        for row in turn.stages
    ]
    return payload


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _turn_from_payload(payload: dict[str, object]) -> _Turn:
    fields = dict(payload)
    stages: list[_StageRow] = []
    for raw in cast("Iterable[object]", fields.pop("stages", ())):
        if isinstance(raw, _StageRow):
            stages.append(raw)
            continue
        d = cast("dict[str, str | float]", raw)
        stages.append(
            _StageRow(
                stage=str(d["stage"]),
                verdict=str(d["verdict"]),
                latency_ms=float(d["latency_ms"]),
                signal=str(d["signal"]),
            )
        )
    fields["stages"] = tuple(stages)
    return _Turn(**fields)  # type: ignore[arg-type]


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


def _demo_database_path() -> Path:
    return default_database_path().with_name("dashboard_demo.sqlite3")


# --- shared rendering --------------------------------------------------------


def _render_topbar() -> None:
    st.columns(4, gap="small")  # keeps the main body width stable
    st.markdown(
        f"**AI Safety Lab** · *LLM Red-Teaming* — "
        f"{_ollama_status()} · target `{_COMPLETION_MODEL}` · judge `{_JUDGE_MODEL}`",
    )


_SESSION_OUTCOME_STYLE = {
    "safe": "🟢 Allowed",
    "blocked": "🔴 Blocked",
    "error": "🔴 Error",
    "partial": "🟡 Partial",
}


def _outcome_label(turn: dict[str, object]) -> str:
    outcome = str(turn.get("outcome", ""))
    return _SESSION_OUTCOME_STYLE.get(outcome, outcome.title())


def _pipeline_row(turn: _Turn) -> None:
    """Truthful stage states: only stages that actually produced observations
    show a verdict; everything else is gray 'skipped', which is honest."""
    observed = {row.stage: row for row in turn.stages}
    columns = st.columns(6, gap="small")
    tier1 = observed.get("TIER1")
    tier2 = observed.get("TIER2")
    tier3 = observed.get("TIER3")
    oguard = observed.get("OGUARD")
    target_label = "Executed" if turn.target_status == "executed" else "Skipped"
    parts = (
        ("Tier 1", tier1.verdict if tier1 else "skipped"),
        ("Tier 2", tier2.verdict if tier2 else "skipped"),
        ("ML Router", f"p={turn.router_probability:.2f}"),
        ("Tier 3", tier3.verdict if tier3 else "bypass"),
        ("Target", target_label),
        ("Output Guard", oguard.verdict if oguard else "skipped"),
    )
    for (name, label), column in zip(parts, columns, strict=True):
        verdict_upper = label.upper()
        if verdict_upper in {"PASS", "SAFE", "EXECUTED"}:
            color, marker = "green", "🟢"
        elif verdict_upper in {"BLOCK", "REFUSED", "ERROR"}:
            color, marker = "red", "🔴"
        elif verdict_upper in {"AMBIGUOUS", "REDACT", "PARTIAL"}:
            color, marker = "yellow", "🟡"
        else:
            color, marker = "gray", "⚪"
        column.markdown(f"{marker} **{name}**")
        column.markdown(f":{color}-background[{label}]", unsafe_allow_html=True)


def _render_result_banner(turn: _Turn) -> None:
    outcome = turn.outcome
    if outcome == "partial":
        color, label = "yellow", "🟡 ALLOWED WITH REDACTION"
    elif outcome == "safe":
        color, label = "green", "🟢 ALLOWED"
    elif outcome == "blocked":
        color, label = "red", "🔴 BLOCKED"
    elif outcome == "error":
        color, label = "red", "🔴 ERROR"
    else:
        color, label = "gray", f"⚪ {outcome.upper()}"
    st.markdown(f":{color}-background[{label}]", unsafe_allow_html=True)


def _render_execution_details(turn: _Turn) -> None:
    with st.expander("Execution details"):
        st.caption("Original prompt received by the application")
        st.code(turn.prompt, language="text")
        if turn.target_status == "executed" and turn.target_input is not None:
            st.caption("Exact text sent to the target model (after redaction)")
            st.code(turn.target_input, language="text")
        else:
            st.caption("Target was not invoked (prompt).")
        if turn.raw_response is not None:
            st.caption("Raw model response (before the output guard)")
            st.code(turn.raw_response, language="text")
        if turn.error:
            st.markdown(f"**Error:** `{turn.error}` (stage `{turn.failure_stage}`)")
        if turn.decisions:
            st.dataframe(
                [
                    {
                        "Stage": stage,
                        "Verdict": verdict,
                        "Reason": reason,
                        "Error": error,
                    }
                    for stage, verdict, reason, error in turn.decisions
                ],
                width="stretch",
                hide_index=True,
            )


def _recent_tests_table(turns: Sequence[dict[str, object]]) -> None:
    if not turns:
        st.info("No tests run yet.")
        return
    rows = [
        {
            "Prompt": str(turn.get("prompt", "")),
            "Result": _outcome_label(turn),
            "Route": str(turn.get("route", "")),
            "Latency ms": f"{_as_float(turn.get("latency_ms")):.0f}",
            "Target": "✓" if turn.get("target_status") == "executed" else "—",
        }
        for turn in turns[-8:]
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


# --- pages --------------------------------------------------------------------


def _page_overview() -> None:
    st.subheader("Overview")
    _render_metrics_row(str(default_database_path()))
    turns = _turns()
    allowed = sum(1 for t in turns if t.get("outcome") == "safe")
    blocked = sum(1 for t in turns if t.get("outcome") == "blocked")
    errored = sum(1 for t in turns if t.get("outcome") == "error")
    columns = st.columns(4, gap="small")
    columns[0].metric("THIS SESSION:", str(len(turns)))
    columns[1].metric("ALLOWED", str(allowed))
    columns[2].metric("BLOCKED", str(blocked))
    columns[3].metric("ERRORS", str(errored))
    st.button(
        "Run a live test",
        type="primary",
        on_click=lambda: st.session_state.update({"page": "Live Test"}),
    )
    st.markdown("**Defense pipeline**")
    pipeline = st.columns(6, gap="small")
    for name, column in zip(_PIPELINE_STAGES, pipeline, strict=True):
        column.markdown(f"`{name}`", unsafe_allow_html=True)
    st.markdown("**Recent tests** (this session)")
    _recent_tests_table(turns)


def _page_live_test() -> None:
    st.subheader("Live Test")
    mode = st.radio("Mode", ("single", "multi"), horizontal=True)
    prompt = st.text_area("Prompt", height=100)
    max_turns = None
    if mode == "multi":
        max_turns = st.slider("Max turns", 1, 5, 3)
    run = st.button("Run Test", type="primary", disabled=not prompt.strip())

    if run and prompt.strip():
        runtime = _pipeline(str(_demo_database_path()))
        with st.spinner("Running defense pipeline..."):
            if mode == "single":
                sequence = _next_case()
                context = ExecutionContext(
                    trace_id=uuid.uuid4().hex,
                    turn_id=f"turn-{sequence}",
                    conversation_id=f"live-{_session_id()}-{sequence}",
                    guardrail_version="dashboard-v1",
                    user_input=prompt,
                )
                turn = asyncio.run(_evaluate(runtime, prompt, context))
            else:
                turn = _multi_turn(prompt, max_turns or 3)
        _turns().append(_turn_payload(turn))
        st.session_state["last_run"] = _turn_payload(turn)
        st.rerun()

    payload = st.session_state.get("last_run")
    if not isinstance(payload, dict):
        return
    turn = _turn_from_payload(payload)
    st.markdown("---")
    if turn.conversation_turns:
        st.markdown(f"**Status:** {turn.outcome} · {turn.termination_reason or ''}")
        for t in turn.conversation_turns:
            judgment = (
                "complied"
                if t.get("judge_complied") is True
                else "refused" if t.get("judge_complied") is False else "—"
            )
            st.markdown(
                f"🟡 turn {t.get('turn_number')} · `{t.get('final_route')}` · "
                f"judge {judgment}"
            )
        st.markdown("**Model response (last turn)**")
        st.code(turn.raw or "—", language="text")
    else:
        _render_result_banner(turn)
        _pipeline_row(turn)
        st.markdown("**Model response**")
        st.code(turn.raw or "—", language="text")
    _render_execution_details(turn)
    if not turn.persisted:
        st.caption(f"Telemetry: {turn.db_error or 'not persisted'}")


def _describe_run(run: dict[str, object]) -> str:
    return f"{run['run_id']} · {run['kind']} · {run['status']}"


def _campaign_result_rows(units: list[dict[str, object]]) -> list[dict[str, str]]:
    rows = []
    for unit in units:
        result = unit["result"]
        if not isinstance(result, dict):
            continue
        if "turns" in result and isinstance(result["turns"], list):
            rows.append(
                {
                    "Case": str(unit.get("unit_id", "?")),
                    "Status": str(unit.get("status", "?")),
                    "Outcome": "conversation",
                    "Target": "✓" if result.get("turns") else "—",
                    "Turns": str(len(result["turns"])),
                }
            )
            continue
        rows.append(
            {
                "Case": str(result.get("case_id") or unit.get("unit_id", "?")),
                "Status": str(unit.get("status", "?")),
                "Outcome": str(result.get("outcome") or "—"),
                "Target": "✓" if result.get("target_input") else "—",
                "Turns": "1",
            }
        )
    return rows


def _page_campaigns() -> None:
    st.subheader("Campaigns")
    database = _campaign_db_path()
    store = CampaignStore(str(database))
    try:
        runs = store.list_runs()
    finally:
        store.close()
    with st.expander("Launch a new campaign", expanded=not runs):
        datasets = _dataset_files()
        dataset = st.selectbox(
            "Dataset",
            datasets or ["(no datasets found under evaluation/data)"],
            index=0 if datasets else None,
        )
        mode = st.radio("Mode", ("single", "multi"), horizontal=True)
        per_label = st.slider("Attack cases per label", 1, 5, 2)
        strategies = st.multiselect(
            "Strategies (single-turn)",
            ("direct", "role_wrap", "encoding", "payload_split"),
            default=("direct",),
        )
        escalations = ("crescendo",)
        max_turns = st.slider("Max turns (multi)", 1, 5, 3)
        seed = st.number_input("Seed", value=0, step=1)
        if st.button("Run campaign", type="primary", disabled=not datasets):
            cases = load_cases(Path(str(dataset)))
            judge = _build_judge()
            os.environ["SAFETY_UI_SPECULATIVE_ECHO"] = "0"
            runtime = _build_pipeline(str(database))
            run_id = f"ui-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            base_metadata = (
                ("target", _COMPLETION_MODEL),
                ("judge", _JUDGE_MODEL),
                ("dataset", Path(str(dataset)).parent.name),
            )
            with st.spinner(f"Running campaign {run_id} ..."):
                if mode == "multi":
                    result = asyncio.run(
                        start_multi(
                            str(database),
                            runtime,
                            cases,
                            judge,
                            _target_completion,
                            run_id=run_id,
                            artifact_dir=str(_RESULTS_ROOT),
                            per_label=int(per_label),
                            max_turns=int(max_turns),
                            escalations=escalations,
                            seed=int(seed),
                            model_metadata=(*base_metadata, ("mode", "multi")),
                        )
                    )
                else:
                    result = asyncio.run(
                        start_single(
                            str(database),
                            runtime,
                            cases,
                            judge,
                            _target_completion,
                            run_id=run_id,
                            artifact_dir=str(_RESULTS_ROOT),
                            per_label=int(per_label),
                            strategies=tuple(strategies or ("direct",)),
                            seed=int(seed),
                            model_metadata=base_metadata,
                        )
                    )
            st.session_state["last_campaign"] = result.run_id
            st.rerun()

    if not runs:
        st.info("No campaigns yet. Launch one above, or use the CLI.")
        return
    labels = [_describe_run(run) for run in runs]
    selected = st.selectbox("Campaign", labels, index=len(labels) - 1)
    run = runs[labels.index(selected)]
    database = _campaign_db_path()
    store = CampaignStore(str(database))
    try:
        info = describe_campaign(store, str(run["run_id"]))
        units = store.units(str(run["run_id"]))
    finally:
        store.close()
    completed = info["units"]
    st.markdown(
        f"**{run['run_id']}** — `{run['kind']}` · status `{run['status']}` · "
        f"units {completed['completed']}/{completed['total']} "
        f"(uncertain {completed['uncertain']})"
    )
    rows = _campaign_result_rows(units)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("This campaign has no completed unit results yet.")


def _render_campaign_record(result: dict[str, object]) -> None:
    st.caption("Original prompt / executed prompt")
    st.code(str(result.get("original_prompt") or result.get("executed_prompt") or "—"))
    if result.get("target_input"):
        st.caption("Exact text sent to the target")
        st.code(str(result["target_input"]))
    else:
        st.caption("Target was not invoked for this record.")
    if result.get("final_response"):
        st.caption("Final response")
        st.code(str(result["final_response"]))
    columns = st.columns(4, gap="small")
    columns[0].caption(f"outcome {result.get('outcome') or '—'}")
    columns[1].caption(f"route {result.get('final_route') or '—'}")
    columns[2].caption(f"latency {result.get('latency_ms') or 0:.0f} ms")
    judge = result.get("judge_complied")
    columns[3].caption(f"judge {judge if judge is not None else '—'}")
    if result.get("execution_error") or result.get("failure_stage"):
        st.caption(
            f"error {result.get('execution_error') or '—'} "
            f"stage {result.get('failure_stage') or '—'}"
        )


def _page_results() -> None:
    st.subheader("Results")
    database = _campaign_db_path()
    if database.exists():
        store = CampaignStore(str(database))
        try:
            runs = store.list_runs()
        finally:
            store.close()
    else:
        runs = []
    st.markdown("**Session tests**")
    _recent_tests_table(_turns())

    st.markdown("**Campaign results**")
    if not runs:
        st.info("No campaigns recorded yet.")
        return
    labels = [_describe_run(run) for run in runs]
    selected = st.selectbox("Campaign", labels, index=len(labels) - 1)
    store = CampaignStore(str(database))
    try:
        units = store.units(str(runs[labels.index(selected)]["run_id"]))
    finally:
        store.close()
    for unit in units:
        result = unit["result"]
        if not isinstance(result, dict):
            continue
        if "turns" in result and isinstance(result["turns"], list):
            for index, turn in enumerate(result["turns"], start=1):
                with st.expander(f"{unit.get('unit_id')} · turn {index}"):
                    _render_campaign_record(turn)
            continue
        with st.expander(str(result.get("case_id") or unit.get("unit_id", "—"))):
            _render_campaign_record(result)


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### AI Safety Lab")
        st.caption("LLM Red-Teaming")
        st.radio(
            "Navigate",
            ("Overview", "Live Test", "Campaigns", "Results"),
            key="page",
            label_visibility="collapsed",
        )


def main() -> None:
    st.set_page_config(
        page_title="AI Safety Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_DENSE_CSS, unsafe_allow_html=True)
    _sidebar()
    _render_topbar()
    _pipeline_row_placeholder = st.empty()  # keep the analysis column stable
    page = st.session_state.get("page", "Overview")
    if page == "Overview":
        _page_overview()
    elif page == "Live Test":
        _page_live_test()
    elif page == "Campaigns":
        _page_campaigns()
    elif page == "Results":
        _page_results()
    else:
        _page_overview()


if __name__ == "__main__":
    main()
