"""Canonical safety configuration and cross-layer interface contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class Direction(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class SafetyVerdict(str, Enum):
    PASS = "pass"
    AMBIGUOUS = "ambiguous"
    REDACT = "redact"
    BLOCK = "block"
    ERROR = "error"


class Route(str, Enum):
    SAFE = "safe"
    PARTIAL = "partial"
    REFUSED = "refused"
    ERROR = "error"


class Stage(str, Enum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    ROUTER = "router"


class TelemetryLane(str, Enum):
    DECISION = "decision"
    EVIDENCE = "evidence"
    CAPTURE = "capture"


_ROUTE_VERDICTS: Mapping[Route, SafetyVerdict] = MappingProxyType(
    {
        Route.SAFE: SafetyVerdict.PASS,
        Route.PARTIAL: SafetyVerdict.REDACT,
        Route.REFUSED: SafetyVerdict.BLOCK,
        Route.ERROR: SafetyVerdict.ERROR,
    }
)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _unit(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class CascadeThresholds:
    """Shared cosine bands and Tier-3 confidence gate."""

    tier2_block_similarity: float
    tier2_ambiguity_floor: float
    tier3_block_confidence: float

    def __post_init__(self) -> None:
        _unit(self.tier2_block_similarity, "tier2_block_similarity")
        _unit(self.tier2_ambiguity_floor, "tier2_ambiguity_floor")
        _unit(self.tier3_block_confidence, "tier3_block_confidence")
        if self.tier2_ambiguity_floor >= self.tier2_block_similarity:
            raise ValueError(
                "tier2_ambiguity_floor must be below tier2_block_similarity"
            )

    @property
    def ambiguity_band(self) -> tuple[float, float]:
        return (self.tier2_ambiguity_floor, self.tier2_block_similarity)


DEFAULT_CASCADE_THRESHOLDS = CascadeThresholds(
    tier2_block_similarity=0.80,
    tier2_ambiguity_floor=0.45,
    tier3_block_confidence=0.78,
)


@dataclass(frozen=True, slots=True)
class TierGates:
    """Activation and short-circuit rules for one cascade direction."""

    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier3_enabled: bool = True
    tier3_requires_ambiguity: bool = True
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if not any((self.tier1_enabled, self.tier2_enabled, self.tier3_enabled)):
            raise ValueError("at least one tier must be enabled")
        if (
            self.tier3_enabled
            and self.tier3_requires_ambiguity
            and not self.tier2_enabled
        ):
            raise ValueError("tier3 ambiguity gate requires tier2")


@dataclass(frozen=True, slots=True)
class RoutingMargins:
    """Fast-path, ambiguity-band, and Tier-3 escalation margins."""

    min_top_gap: float
    tier3_gate_probability: float

    def __post_init__(self) -> None:
        _unit(self.min_top_gap, "min_top_gap")
        _unit(self.tier3_gate_probability, "tier3_gate_probability")


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """Centralized p95 latency ceilings for the guardrail path."""

    tier1_ms: float = 60.0
    tier2_ms: float = 20.0
    cascade_ms: float = 400.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.tier1_ms, "tier1_ms"),
            (self.tier2_ms, "tier2_ms"),
            (self.cascade_ms, "cascade_ms"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive number")
        if self.tier1_ms + self.tier2_ms >= self.cascade_ms:
            raise ValueError("tier budgets must leave cascade headroom")


@dataclass(frozen=True, slots=True)
class PayloadLimits:
    """Centralized, immutable request payload bounds.

    Values are UTF-8 encoded bytes (never Python character counts) except
    `request_id_chars`, which follows the existing character contract.
    Trust boundaries (HTTP API, dashboard entry) enforce these; internal
    tiers keep their own conservative defaults (e.g. Tier 1 / Tier 3
    standalone limit of 4096 bytes) as defense in depth.
    """

    prompt_bytes: int = 16384
    request_body_bytes: int = 32768
    metadata_entries: int = 16
    metadata_key_bytes: int = 128
    metadata_value_bytes: int = 512
    request_id_chars: int = 64

    def __post_init__(self) -> None:
        for value, name in (
            (self.prompt_bytes, "prompt_bytes"),
            (self.request_body_bytes, "request_body_bytes"),
            (self.metadata_entries, "metadata_entries"),
            (self.metadata_key_bytes, "metadata_key_bytes"),
            (self.metadata_value_bytes, "metadata_value_bytes"),
            (self.request_id_chars, "request_id_chars"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_PAYLOAD_LIMITS = PayloadLimits()


def utf8_byte_length(text: str) -> int:
    """UTF-8 encoded byte length; invalid unicode (e.g. lone surrogates) is
    rejected rather than silently counted with a lossy encoding."""
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("text contains invalid unicode") from exc


@dataclass(frozen=True, slots=True)
class DirectionProfile:
    thresholds: CascadeThresholds
    gates: TierGates
    margins: RoutingMargins
    budgets: LatencyBudget = field(default_factory=LatencyBudget)


@dataclass(frozen=True, slots=True)
class SafetyProfile:
    """Versioned source of truth for all cascading safety controls."""

    version: str
    input: DirectionProfile
    output: DirectionProfile

    def __post_init__(self) -> None:
        _text(self.version, "version")

    def for_direction(self, direction: Direction) -> DirectionProfile:
        if direction is Direction.INPUT:
            return self.input
        if direction is Direction.OUTPUT:
            return self.output
        raise ValueError(f"unsupported direction: {direction!r}")


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Request-scoped identity and direction shared by logs and telemetry."""

    trace_id: str
    turn_id: str
    conversation_id: str
    direction: Direction = Direction.INPUT
    guardrail_version: str = "unversioned"
    user_input: str | None = None

    def __post_init__(self) -> None:
        _text(self.trace_id, "trace_id")
        _text(self.turn_id, "turn_id")
        _text(self.conversation_id, "conversation_id")
        _text(self.guardrail_version, "guardrail_version")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")
        if self.user_input is not None:
            _text(self.user_input, "user_input")


@dataclass(frozen=True, slots=True)
class RouteVerdict:
    """Terminal decision handed to the downstream response path."""

    context: ExecutionContext
    route: Route
    verdict: SafetyVerdict
    stage: Stage
    reason_codes: tuple[str, ...] = ()
    safe_text: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        if not isinstance(self.route, Route):
            raise TypeError("route must be a Route")
        if not isinstance(self.verdict, SafetyVerdict):
            raise TypeError("verdict must be a SafetyVerdict")
        if self.verdict is not _ROUTE_VERDICTS[self.route]:
            raise ValueError(
                f"route {self.route.value} requires {_ROUTE_VERDICTS[self.route].value}"
            )
        if not isinstance(self.stage, Stage):
            raise TypeError("stage must be a Stage")
        if self.route is Route.PARTIAL:
            if not self.safe_text:
                raise ValueError("partial route requires safe_text")
        elif self.safe_text is not None:
            raise ValueError("safe_text is only valid for partial routes")
        if self.route is Route.ERROR:
            if not self.error:
                raise ValueError("error route requires error")
        elif self.error is not None:
            raise ValueError("error is only valid for error routes")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Immutable asynchronous stage event shared by all producers."""

    event_id: str
    context: ExecutionContext
    stage: Stage
    verdict: SafetyVerdict
    latency_ms: float
    prompt_hash: str
    timestamp_ns: int
    lane: TelemetryLane = TelemetryLane.DECISION
    rule_ids: tuple[str, ...] = ()
    redaction_spans: tuple[tuple[int, int], ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id")
        _text(self.prompt_hash, "prompt_hash")
        if not isinstance(self.context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        if not isinstance(self.stage, Stage):
            raise TypeError("stage must be a Stage")
        if not isinstance(self.verdict, SafetyVerdict):
            raise TypeError("verdict must be a SafetyVerdict")
        if not isinstance(self.lane, TelemetryLane):
            raise TypeError("lane must be a TelemetryLane")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if self.verdict is SafetyVerdict.ERROR:
            if not self.error:
                raise ValueError("error verdict requires error")
        elif self.error is not None:
            raise ValueError("error is only valid for error verdicts")
        for start, end in self.redaction_spans:
            if start < 0 or end <= start:
                raise ValueError("redaction spans must be positive ranges")


__all__ = [
    "DEFAULT_CASCADE_THRESHOLDS",
    "DEFAULT_PAYLOAD_LIMITS",
    "CascadeThresholds",
    "Direction",
    "DirectionProfile",
    "ExecutionContext",
    "LatencyBudget",
    "PayloadLimits",
    "Route",
    "RouteVerdict",
    "RoutingMargins",
    "SafetyProfile",
    "SafetyVerdict",
    "Stage",
    "TelemetryEvent",
    "TelemetryLane",
    "TierGates",
    "utf8_byte_length",
]
