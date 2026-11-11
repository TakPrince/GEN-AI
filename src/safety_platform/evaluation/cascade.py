"""Real three-tier cascade execution for frozen evaluation cases."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..config.registry import (
    DEFAULT_CASCADE_THRESHOLDS,
    CascadeThresholds,
    Direction,
    ExecutionContext,
    RouteVerdict,
    RoutingMargins,
    SafetyVerdict,
    Stage,
)
from ..guardrails.canary import CanaryDetector, CanaryMatch, ConversationTail
from ..guardrails.router import ResponseRouter, StageDecision
from ..guardrails.tier1.regex_filter import RegexFilter
from ..guardrails.tier1.rules import defense_regex_filter
from ..guardrails.tier2.vector_filter import Centroid, VectorFilter
from ..guardrails.tier3.guard_classifier import GuardClassifier
from ..research.ml_router import (
    ModelDrivenRouter,
    RouterFeatures,
    Tier3BypassRouter,
    benchmark_router,
)
from .dataset import EvaluationCase
from .llm import (
    SPECULATIVE_VIOLATING,
    OllamaEmbedder,
    SpeculativeEchoProbe,
)


@dataclass(frozen=True, slots=True)
class CascadeOutcome:
    route: str
    verdict: str
    stage: str
    latency_ms: float
    tier3_invoked: bool
    tier2_similarity: float
    speculative_echo_status: str | None = None


@dataclass(frozen=True, slots=True)
class CascadeRuntime:
    regex: RegexFilter
    vector: VectorFilter
    guard: GuardClassifier
    ml_router: ModelDrivenRouter
    router: ResponseRouter
    # Optional output-direction detector profile: output text is
    # compared against centroids built from labeled *output* examples, never
    # silently against the input attack centroids. None keeps the legacy
    # shared-centroid behavior and must be reported honestly.
    vector_output: VectorFilter | None = None
    # Live output guard: a model response is the only unvetted artifact, so the
    # live path always consults Tier 3 on it. The harness keeps the measured
    # conditional escalation (default False).
    tier3_on_output: bool = False
    # Known confidential values: an exact/transformed echo in OUTPUT is a leak,
    # blocked fail-closed. None disables the check (evaluation harness default).
    canary: CanaryDetector | None = None
    # Cross-turn carry of delivered output, so a secret split across turns is
    # seen. Requires `canary`; committed only by execute_defense after delivery.
    canary_tail: ConversationTail | None = None

    def commit_delivered(self, context: ExecutionContext, delivered: str) -> None:
        """Record text the caller actually returned, for cross-turn canary checks."""
        if self.canary_tail is not None:
            self.canary_tail.commit(context.conversation_id, delivered)

    async def run(
        self,
        text: str,
        context: ExecutionContext,
        *,
        probe: SpeculativeEchoProbe | None = None,
    ) -> CascadeOutcome:
        started = time.perf_counter_ns()
        tier3_invoked = False
        echo_status: str | None = None
        vector = (
            self.vector_output
            if context.direction is Direction.OUTPUT and self.vector_output
            else self.vector
        )
        first = self.regex.evaluate(text)
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
            route = self.router.route(context, decisions)
            return _outcome(route, started, False, 0.0)

        # Known-value leak: a canary in a generated response is never returned.
        leaks: tuple[CanaryMatch, ...] = ()
        if self.canary is not None and context.direction is Direction.OUTPUT:
            prior = ""
            cumulative: list[CanaryMatch] = []
            if self.canary_tail is not None:
                prior = self.canary_tail.prefix(context.conversation_id)
                cumulative = [
                    CanaryMatch(canary_id, "coverage")
                    for canary_id in self.canary_tail.blocked_canaries(
                        context.conversation_id, text
                    )
                ]
            leaks = (*self.canary.find(prior + text), *cumulative)
        if leaks:
            decisions.append(
                StageDecision(
                    stage=Stage.TIER1,
                    verdict=SafetyVerdict.BLOCK,
                    reason_codes=tuple(f"canary:{match.canary_id}" for match in leaks),
                )
            )
            route = self.router.route(context, decisions)
            return _outcome(route, started, False, 0.0)

        # Opt-in echo probe: never before input screening, never unscreened.
        if probe is not None:
            echo = await probe.probe(first.redacted_text or text)
            echo_status = echo.status
            if echo.status == SPECULATIVE_VIOLATING:
                route = self.router.route(
                    context,
                    (
                        StageDecision(
                            stage=Stage.ROUTER,
                            verdict=SafetyVerdict.BLOCK,
                            reason_codes=("speculative_echo:violating",),
                        ),
                    ),
                )
                return _outcome(route, started, False, 0.0, echo_status)

        second = vector.evaluate(text)
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
        decision = self.ml_router.route(
            RouterFeatures(
                regex_hit=bool(first.rule_ids),
                top_similarity=second.top_similarity,
                top_gap=second.top_gap,
                tier2_ambiguous=second.verdict is SafetyVerdict.AMBIGUOUS,
                prompt_bytes=len(text.encode("utf-8", "surrogatepass")),
            )
        )
        always_check = self.tier3_on_output and context.direction is Direction.OUTPUT
        if (
            decision.invoke_tier3
            or second.verdict is SafetyVerdict.AMBIGUOUS
            or always_check
        ):
            # Multi-signal escalation: independent signals, not a
            # single Tier-2 ambiguity gate.
            third = await self.guard.classify_if_ambiguous(
                text,
                context,
                second.verdict,
            )
            if third is not None:
                tier3_invoked = True
                decisions.append(
                    StageDecision(
                        stage=Stage.TIER3,
                        verdict=third.verdict,
                        reason_codes=third.reason_codes,
                        error=third.error,
                    )
                )
        route = self.router.route(context, decisions)
        return _outcome(
            route,
            started,
            tier3_invoked,
            second.top_similarity,
            echo_status,
        )


def build_cascade(
    train_cases: Sequence[EvaluationCase],
    embedder: OllamaEmbedder,
    guard: GuardClassifier,
    *,
    thresholds: CascadeThresholds | None = None,
    margins: RoutingMargins | None = None,
    output_train_cases: Sequence[EvaluationCase] | None = None,
    output_thresholds: CascadeThresholds | None = None,
) -> CascadeRuntime:
    """Build the cascade runtime.

    `output_train_cases` carries labeled *output-direction* examples (unsafe
    model-response text); when provided, the output direction gets its own
    VectorFilter and centroids, and the output cascade stops reusing input
    attack centroids. Omitting it keeps the documented legacy behavior.
    """
    attacks = tuple(case for case in train_cases if case.label == "attack")
    if not attacks:
        raise ValueError("training split must contain attack cases")
    active_thresholds = DEFAULT_CASCADE_THRESHOLDS if thresholds is None else thresholds
    active_margins = RoutingMargins(0.01, 0.55) if margins is None else margins
    centroids = tuple(
        Centroid(case.case_id, embedder.embed(case.prompt)) for case in attacks
    )
    vector = VectorFilter(embedder, centroids, active_thresholds, active_margins)
    profile: VectorFilter | None = None
    if output_train_cases:
        output_attacks = tuple(
            case for case in output_train_cases if case.label == "attack"
        )
        if output_attacks:
            output_centroids = tuple(
                Centroid(case.case_id, embedder.embed(case.prompt))
                for case in output_attacks
            )
            profile = VectorFilter(
                embedder,
                output_centroids,
                (
                    DEFAULT_CASCADE_THRESHOLDS
                    if output_thresholds is None
                    else (output_thresholds)
                ),
                active_margins,
            )
    regex = defense_regex_filter(max_input_bytes=16384)
    router_model = benchmark_router()
    return CascadeRuntime(
        regex=regex,
        vector=vector,
        guard=guard,
        ml_router=ModelDrivenRouter(Tier3BypassRouter(router_model)),
        router=ResponseRouter(),
        vector_output=profile,
    )


def make_context(case: EvaluationCase, direction: Direction) -> ExecutionContext:
    return ExecutionContext(
        trace_id=f"eval-{case.case_id}",
        turn_id="1",
        conversation_id=f"{case.label}-{case.case_id}",
        direction=direction,
        guardrail_version="evaluation-v1",
        user_input=case.prompt,
    )


def _outcome(
    route: RouteVerdict,
    started: int,
    tier3_invoked: bool,
    tier2_similarity: float,
    speculative_echo_status: str | None = None,
) -> CascadeOutcome:
    return CascadeOutcome(
        route=route.route.value,
        verdict=route.verdict.value,
        stage=route.stage.value,
        latency_ms=max(0.0, (time.perf_counter_ns() - started) / 1_000_000),
        tier3_invoked=tier3_invoked,
        tier2_similarity=tier2_similarity,
        speculative_echo_status=speculative_echo_status,
    )


__all__ = ["CascadeOutcome", "CascadeRuntime", "build_cascade", "make_context"]
