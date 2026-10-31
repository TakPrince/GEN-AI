"""Real three-tier cascade execution for frozen evaluation cases."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..config.registry import (
    CascadeThresholds,
    DEFAULT_CASCADE_THRESHOLDS,
    Direction,
    ExecutionContext,
    RouteVerdict,
    RoutingMargins,
    SafetyVerdict,
    Stage,
)
from ..guardrails.router import ResponseRouter, StageDecision
from ..guardrails.tier1.regex_filter import RegexAction, RegexFilter, RegexRule
from ..guardrails.tier2.vector_filter import Centroid, VectorFilter
from ..guardrails.tier3.guard_classifier import GuardClassifier
from ..research.ml_router import (
    LogisticRegressionRouter,
    RouterFeatures,
    Tier3BypassRouter,
)
from .dataset import EvaluationCase
from .llm import OllamaEmbedder


@dataclass(frozen=True, slots=True)
class CascadeOutcome:
    route: str
    verdict: str
    stage: str
    latency_ms: float
    tier3_invoked: bool
    tier2_similarity: float


@dataclass(frozen=True, slots=True)
class CascadeRuntime:
    regex: RegexFilter
    vector: VectorFilter
    guard: GuardClassifier
    ml_router: Tier3BypassRouter
    router: ResponseRouter

    async def run(self, text: str, context: ExecutionContext) -> CascadeOutcome:
        started = time.perf_counter_ns()
        tier3_invoked = False
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

        second = self.vector.evaluate(text)
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
        if second.verdict is SafetyVerdict.AMBIGUOUS and decision.invoke_tier3:
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
        )


def build_cascade(
    train_cases: Sequence[EvaluationCase],
    embedder: OllamaEmbedder,
    guard: GuardClassifier,
    *,
    thresholds: CascadeThresholds | None = None,
    margins: RoutingMargins | None = None,
) -> CascadeRuntime:
    attacks = tuple(case for case in train_cases if case.label == "attack")
    if not attacks:
        raise ValueError("training split must contain attack cases")
    active_thresholds = (
        DEFAULT_CASCADE_THRESHOLDS if thresholds is None else thresholds
    )
    active_margins = RoutingMargins(0.01, 0.55) if margins is None else margins
    centroids = tuple(
        Centroid(case.case_id, embedder.embed(case.prompt)) for case in attacks
    )
    regex = RegexFilter(
        (
            RegexRule("block.ignore_previous", r"ignore\s+(all\s+)?previous"),
            RegexRule(
                "block.system_prompt",
                r"(reveal|show|print).{0,24}(system prompt|hidden instructions)",
            ),
            RegexRule("redact.secret", r"secret=\w+", action=RegexAction.REDACT),
        ),
        max_input_bytes=16384,
    )
    vector = VectorFilter(embedder, centroids, active_thresholds, active_margins)
    router_model = LogisticRegressionRouter(
        weights=(1.5, 1.0, -0.3, 3.0, 0.2),
        bias=-2.5,
        threshold=0.50,
    )
    return CascadeRuntime(
        regex=regex,
        vector=vector,
        guard=guard,
        ml_router=Tier3BypassRouter(router_model),
        router=ResponseRouter(),
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
) -> CascadeOutcome:
    return CascadeOutcome(
        route=route.route.value,
        verdict=route.verdict.value,
        stage=route.stage.value,
        latency_ms=max(0.0, (time.perf_counter_ns() - started) / 1_000_000),
        tier3_invoked=tier3_invoked,
        tier2_similarity=tier2_similarity,
    )


__all__ = ["CascadeOutcome", "CascadeRuntime", "build_cascade", "make_context"]
