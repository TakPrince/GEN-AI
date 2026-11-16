"""Minimal production-facing HTTP API for the safety platform.

Every request goes through `execute_defense`, the same unified boundary used by
the dashboard and the orchestrator. The API adds transport concerns only:
authentication, request limits, concurrency bounds, stable error envelopes and
health checks. It does not duplicate guardrail or telemetry logic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from safety_platform.config.registry import (
    DEFAULT_PAYLOAD_LIMITS,
    ExecutionContext,
    utf8_byte_length,
)
from safety_platform.evaluation.llm import OllamaClient
from safety_platform.execution import REFUSAL_TEXT, DefenseRequest, execute_defense

logger = logging.getLogger("safety_platform.api")

_LIMITS = DEFAULT_PAYLOAD_LIMITS
_MAX_PROMPT_BYTES = int(
    os.environ.get("SAFETY_API_MAX_PROMPT_BYTES", _LIMITS.prompt_bytes)
)
_MAX_BODY_BYTES = int(
    os.environ.get("SAFETY_API_MAX_BODY_BYTES", _LIMITS.request_body_bytes)
)
_MAX_METADATA = _LIMITS.metadata_entries
_MAX_METADATA_KEY = _LIMITS.metadata_key_bytes
_MAX_METADATA_VALUE = _LIMITS.metadata_value_bytes
_MAX_REQUEST_ID = _LIMITS.request_id_chars


class ApiError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    request_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def _bounded_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        try:
            length = utf8_byte_length(value)
        except ValueError as exc:
            raise ValueError("prompt contains invalid unicode") from exc
        if length > _MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the configured limit")
        return value

    @field_validator("request_id")
    @classmethod
    def _bounded_request_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value
            or len(value) > _MAX_REQUEST_ID
            or not all(c.isalnum() or c in "._-" for c in value)
        ):
            raise ValueError("request_id is invalid")
        return value

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > _MAX_METADATA:
            raise ValueError("too many metadata entries")
        for key, item in value.items():
            try:
                key_bytes = utf8_byte_length(key)
                item_bytes = utf8_byte_length(item)
            except ValueError as exc:
                raise ValueError("metadata contains invalid unicode") from exc
            if key_bytes > _MAX_METADATA_KEY or item_bytes > _MAX_METADATA_VALUE:
                raise ValueError("metadata entry is too large")
        return value


class ChatResponse(BaseModel):
    request_id: str
    trace_id: str
    outcome: str
    blocked: bool
    response: str
    route: str
    stage: str
    tier3_invoked: bool
    latency_ms: float
    telemetry_persisted: bool


def _default_completion() -> Callable[[str], str]:
    client = OllamaClient(
        os.environ.get("SAFETY_API_TARGET_MODEL", "qwen3:8b"),
        timeout_s=float(os.environ.get("SAFETY_API_EXECUTION_TIMEOUT_S", "120")),
    )
    num_predict = int(os.environ.get("SAFETY_API_NUM_PREDICT", "256"))

    def complete(prompt: str) -> str:
        return client.generate(prompt, think=False, num_predict=num_predict)

    return complete


def create_app(
    *,
    runtime: Any | None = None,
    completion: Callable[[str], str] | None = None,
    api_key: str | None = None,
    db_path: str | None = None,
    max_concurrency: int | None = None,
    queue_timeout_s: float | None = None,
    execution_timeout_s: float | None = None,
) -> FastAPI:
    key = api_key if api_key is not None else os.environ.get("SAFETY_API_KEY")
    allow_unauth = os.environ.get("SAFETY_API_ALLOW_UNAUTHENTICATED") == "1"
    database = (
        Path(db_path)
        if db_path
        else Path(os.environ.get("SAFETY_API_DB", "metrics/api.sqlite3"))
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    limit = max_concurrency or int(os.environ.get("SAFETY_API_MAX_CONCURRENCY", "2"))
    queue_wait = (
        queue_timeout_s
        if queue_timeout_s is not None
        else float(os.environ.get("SAFETY_API_QUEUE_TIMEOUT_S", "30"))
    )
    exec_wait = (
        execution_timeout_s
        if execution_timeout_s is not None
        else float(os.environ.get("SAFETY_API_EXECUTION_TIMEOUT_S", "120"))
    )
    target_model = os.environ.get("SAFETY_API_TARGET_MODEL", "qwen3:8b")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if runtime is not None:
            application.state.runtime = runtime
        else:
            from safety_platform.app import _build_pipeline

            application.state.runtime = _build_pipeline(str(database))
        application.state.completion = completion or _default_completion()
        application.state.semaphore = asyncio.Semaphore(limit)
        application.state.db_path = str(database)
        yield

    application = FastAPI(
        title="Safety Platform API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    async def require_auth(x_api_key: str | None = Header(default=None)) -> None:
        if allow_unauth:
            return
        if not key:
            raise ApiError(503, "auth_not_configured")
        if x_api_key is None or not secrets.compare_digest(x_api_key, key):
            raise ApiError(401, "unauthorized")

    @application.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"code": exc.code}}
        )

    @application.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": {"code": "invalid_request"}}
        )

    @application.middleware("http")
    async def _transport_limits(request: Request, call_next):
        length = request.headers.get("content-length")
        if length is not None:
            try:
                too_large = int(length) > _MAX_BODY_BYTES
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"error": {"code": "invalid_request"}}
                )
            if too_large:
                return JSONResponse(
                    status_code=413, content={"error": {"code": "request_too_large"}}
                )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        if getattr(request.app.state, "runtime", None) is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        try:
            with sqlite3.connect(request.app.state.db_path) as connection:
                connection.execute("select 1")
        except sqlite3.Error:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "target_model": target_model},
        )

    @application.post(
        "/v1/chat",
        response_model=ChatResponse,
        dependencies=[Depends(require_auth)],
    )
    async def chat(body: ChatRequest, request: Request):
        request_id = body.request_id or uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        context = ExecutionContext(
            trace_id=trace_id,
            turn_id=f"api-{uuid.uuid4().hex[:12]}",
            conversation_id=request_id,
            guardrail_version="api-v1",
            user_input=body.prompt,
        )
        semaphore: asyncio.Semaphore = request.app.state.semaphore
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=queue_wait)
        except TimeoutError as exc:
            raise ApiError(503, "server_busy") from exc
        try:
            result = await asyncio.wait_for(
                execute_defense(
                    request.app.state.runtime,
                    DefenseRequest(
                        prompt=body.prompt,
                        request_id=request_id,
                        metadata=tuple(body.metadata.items()),
                    ),
                    context,
                    completion=request.app.state.completion,
                ),
                timeout=exec_wait,
            )
        except TimeoutError as exc:
            raise ApiError(504, "execution_timeout") from exc
        except ApiError:
            raise
        except Exception as exc:  # never leak internals to the caller
            logger.exception("execution failed request_id=%s", request_id)
            raise ApiError(503, "execution_error") from exc
        finally:
            semaphore.release()

        if result.outcome == "error":
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"code": "execution_error"},
                    "request_id": request_id,
                    "outcome": "error",
                    "response": REFUSAL_TEXT,
                },
            )
        return ChatResponse(
            request_id=request_id,
            trace_id=trace_id,
            outcome=result.outcome,
            blocked=result.blocked,
            response=result.final_response,
            route=result.final_route,
            stage=result.final_stage,
            tier3_invoked=result.tier3_invoked,
            latency_ms=result.latency_ms,
            telemetry_persisted=result.telemetry_persisted,
        )

    return application


app = create_app()
