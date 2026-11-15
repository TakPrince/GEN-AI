"""Groq hosted-target adapter for the Red-Team pipeline.

Mirrors `OllamaClient`: stdlib urllib, temperature 0, bounded retries.
Credentials come exclusively from `GROQ_API_KEY`; the key is never
logged, embedded in exceptions, or reported. Bounded retry policy:
429/5xx/timeouts retried once (honoring Retry-After capped like the
Ollama runtime), 401/400 fail immediately. Usage counters (requests,
tokens) are recorded for benchmark reporting.
"""

from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MAX_ATTEMPTS = 2
_RETRY_AFTER_CAP_S = 5.0


class GroqError(RuntimeError):
    """Groq API failure; messages never contain credentials."""


class MissingApiKeyError(GroqError):
    def __init__(self) -> None:
        super().__init__(
            "GROQ_API_KEY is not set; export it in your shell or .env file"
        )


def _redact(message: str) -> str:
    return message.replace(os.environ.get("GROQ_API_KEY", ""), "")


class GroqClient:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        num_predict: int = 256,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._model = model
        self._api_key = api_key if api_key is not None else None
        self._timeout_s = timeout_s
        self._num_predict = num_predict
        self.requests = 0
        self.errors = 0
        self.rate_limited = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def model(self) -> str:
        return self._model

    def _key(self) -> str:
        if self._api_key is not None and self._api_key.strip():
            return self._api_key
        key = os.environ.get("GROQ_API_KEY", "")
        if not key.strip():
            raise MissingApiKeyError()
        return key

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self._num_predict,
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            request = Request(
                _GROQ_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key()}",
                },
                method="POST",
            )
            try:
                self.requests += 1
                with urlopen(request, timeout=self._timeout_s) as response:
                    return self._parse(response.read().decode("utf-8"))
            except HTTPError as exc:
                self.errors += 1
                detail = _redact(str(exc))
                if exc.code in (400, 401, 403):
                    raise GroqError(
                        f"groq request rejected: {exc.code} {detail}"
                    ) from exc
                last_error = exc
                if exc.code == 429:
                    self.rate_limited += 1
                    retry_after = (
                        exc.headers.get("Retry-After") if exc.headers else None
                    )
                    try:
                        delay = min(
                            float(retry_after or "1.0"), _RETRY_AFTER_CAP_S
                        )
                    except (TypeError, ValueError):
                        delay = 1.0
                    if attempt + 1 < _MAX_ATTEMPTS:
                        time.sleep(delay)
            except (URLError, TimeoutError) as exc:
                self.errors += 1
                last_error = exc
        raise GroqError(
            f"groq request failed: {_redact(str(last_error))}"
        ) from last_error

    def _parse(self, raw: str) -> str:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GroqError("groq returned malformed JSON") from exc
        usage = data.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqError("groq response missing completion content") from exc
        if not isinstance(content, str) or not content.strip():
            raise GroqError("groq returned an empty completion")
        return content


__all__ = ["GroqClient", "GroqError", "MissingApiKeyError"]
