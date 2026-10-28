"""Conditional local Ollama Llama Guard connector."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ...config.registry import (
    CascadeThresholds,
    ExecutionContext,
    SafetyVerdict,
    TierGates,
)
from .classifier import ClassifierClient, Tier3Classifier, Tier3Result

_INSTRUCTION = (
    "Classify the user message as safe or unsafe. "
    "Return only one word: safe or unsafe."
)


class OllamaGuardError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OllamaGuardConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama-guard3:8b"
    timeout_s: float = 5.0
    max_prompt_bytes: int = 4096

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP URL")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, int | float)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        if (
            isinstance(self.max_prompt_bytes, bool)
            or not isinstance(self.max_prompt_bytes, int)
            or self.max_prompt_bytes <= 0
        ):
            raise ValueError("max_prompt_bytes must be a positive integer")


@dataclass(frozen=True, slots=True)
class OllamaGenerateRequest:
    base_url: str
    model: str
    prompt: str
    timeout_s: float


class OllamaTransport(Protocol):
    async def generate(self, request: OllamaGenerateRequest) -> str: ...


class UrllibOllamaTransport:
    async def generate(self, request: OllamaGenerateRequest) -> str:
        return await asyncio.to_thread(self._generate, request)

    def _generate(self, request: OllamaGenerateRequest) -> str:
        payload = json.dumps(
            {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        url = f"{request.base_url.rstrip('/')}/api/generate"
        http_request = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=request.timeout_s) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise OllamaGuardError(type(exc).__name__) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaGuardError("malformed_json") from exc
        if not isinstance(decoded, dict):
            raise OllamaGuardError("malformed_response")
        value = decoded.get("response")
        if not isinstance(value, str) or not value.strip():
            raise OllamaGuardError("malformed_response")
        return value


class OllamaLlamaGuardConnector:
    def __init__(
        self,
        config: OllamaGuardConfig | None = None,
        transport: OllamaTransport | None = None,
    ) -> None:
        self._config = OllamaGuardConfig() if config is None else config
        self._transport = UrllibOllamaTransport() if transport is None else transport

    async def score(self, text: str, context: ExecutionContext) -> float:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        if len(text.encode("utf-8", "surrogatepass")) > self._config.max_prompt_bytes:
            raise OllamaGuardError("prompt_too_large")

        prompt = f"{_INSTRUCTION}\n\nUser message:\n{text}\n\nVerdict:"
        output = await self._transport.generate(
            OllamaGenerateRequest(
                base_url=self._config.base_url,
                model=self._config.model,
                prompt=prompt,
                timeout_s=self._config.timeout_s,
            )
        )
        return _score_output(output)


class GuardClassifier:
    def __init__(
        self,
        client: ClassifierClient,
        thresholds: CascadeThresholds,
        *,
        gates: TierGates | None = None,
    ) -> None:
        self._classifier = Tier3Classifier(client, thresholds)
        self._gates = TierGates() if gates is None else gates

    async def classify_if_ambiguous(
        self,
        text: str,
        context: ExecutionContext,
        tier2_verdict: SafetyVerdict,
    ) -> Tier3Result | None:
        if not isinstance(tier2_verdict, SafetyVerdict):
            raise TypeError("tier2_verdict must be a SafetyVerdict")
        if not self._gates.tier3_enabled:
            return None
        if (
            self._gates.tier3_requires_ambiguity
            and tier2_verdict is not SafetyVerdict.AMBIGUOUS
        ):
            return None
        return await self._classifier.classify(text, context)


def _score_output(output: str) -> float:
    token = output.strip().splitlines()[0].strip().lower()
    if token.startswith("unsafe"):
        return 1.0
    if token.startswith("safe"):
        return 0.0
    raise OllamaGuardError("malformed_verdict")


__all__ = [
    "GuardClassifier",
    "OllamaGenerateRequest",
    "OllamaGuardConfig",
    "OllamaGuardError",
    "OllamaLlamaGuardConnector",
    "OllamaTransport",
    "UrllibOllamaTransport",
]
