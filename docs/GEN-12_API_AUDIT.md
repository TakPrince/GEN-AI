# GEN-12 — Safety Platform API Audit

Status: **PARTIAL**

## 1. Existing architecture and routes

There is **no HTTP API in the repository**. `pyproject.toml` has no FastAPI,
Starlette, Flask, or uvicorn dependency; `src/safety_platform/api/` does not
exist. The documented FastAPI `/v1/chat` service is listed as **planned** in
`Readme.md` (§9 "Planned", §10 interface table). `docs/PROJECT_STRUCTURE.md`
describes an `api/v1/` layer, `api/deps.py` authentication, `api/errors.py`,
`deploy/`, and `health.py` as if implemented; none of those paths exist.

The actual interfaces are:

- **Streamlit dashboard** (`src/safety_platform/app.py`): four pages —
  Overview (`:782`), Live Test (`:807`), Campaigns (`:897`), Results (`:1020`).
- **Programmatic execution boundary** (`src/safety_platform/execution.py`):
  `DefenseRequest` → `execute_defense` → `DefenseResult`. Used by the dashboard
  (`app.py:465`) and the single/multi-turn orchestrator
  (`orchestrator/loop.py:220`, `orchestrator/multi.py:263`).
- **CLI evaluation harness** (`python -m safety_platform.evaluation`), which
  drives `CascadeRuntime` directly to measure baseline/input/full outcomes.

## 2. Confirmed weaknesses (not missing features)

1. **Documentation/reality mismatch.** `docs/PROJECT_STRUCTURE.md` presents the
   HTTP API, auth dependencies, deploy assets and health endpoint as present.
   They are not. `Readme.md` correctly labels the FastAPI service as planned.
2. **Boundary docstring overstates evaluation usage.** `execution.py` says the
   boundary is shared by "dashboard, evaluation and future red-team callers";
   the evaluation runner uses `cascade.run` directly. This is an offline
   measurement path, not a serving bypass, but the comment is inaccurate.
3. **No schema-level validation on `DefenseRequest`.** Validation relies on
   Tier 1: oversized input raises `RegexInputTooLarge`, non-string input raises
   `TypeError`, and malformed Unicode raises `UnicodeEncodeError`; the
   `RegexFilterMiddleware` converts these to `SafetyVerdict.ERROR` and the
   boundary fails closed. Empty prompts are not rejected, and a caller can
   inject a `RegexFilter` with a larger `max_input_bytes`.
4. **No timeout inside `execute_defense` for the caller-supplied completion.**
   The Streamlit path bounds the target with `OllamaClient` timeouts, a
   `BoundedSemaphore`, and a queue timeout (`app.py:440`); a future HTTP
   adapter must supply equivalent bounds. The boundary itself only catches
   exceptions.

## 3. Missing production features (deployment-dependent)

HTTP routing/versioning, request/response schemas, authentication and
authorization, CORS, security headers, HTTP body-size enforcement, rate
limiting, health/readiness endpoints, TLS, graceful shutdown, multi-worker
telemetry transport, and deploy assets. These are not implemented and are not
claimed. No credentials or auth mechanism was invented.

## 4. Changes made

- `tests/unit/test_execution.py`:
  - `test_oversized_input_fails_closed_before_target` — a 16,385-byte prompt is
    rejected at Tier 1, the target is never called, and the caller receives the
    refusal text with `outcome="error"`.
  - `test_malformed_unicode_input_fails_closed_before_target` — a lone surrogate
    fails closed at Tier 1 without leaking the exception text in
    `DefenseResult.error`.
- This audit report: `docs/GEN-12_API_AUDIT.md`.

No production API service was added. Adding FastAPI, auth, or deploy
infrastructure would be a new feature and a deployment decision, not a minimal
GEN-12 fix. No GEN-10/GEN-15 logic or artifacts were touched.

## 5. Security and execution-boundary findings

- **Unified boundary preserved:** all serving paths (dashboard, single-turn and
  multi-turn orchestrator) call `execute_defense`; the target completion is
  invoked only after input guards pass; the output guard runs before delivery.
- **Fail-closed:** Tier 1/2/3, output-guard, and target exceptions become
  `outcome="error"`; target failures are never reported as safe; telemetry
  persistence failures do not change the defense outcome.
- **Bounded input:** Tier 1 is configured with `max_input_bytes=16384` in
  `app.py:290` and `evaluation/cascade.py:251`; the middleware converts
  `RegexInputTooLarge` and `UnicodeEncodeError` into `ERROR` decisions.
- **Resource controls:** target concurrency is capped by `_TARGET_SLOTS` and
  queue wait by `_TARGET_QUEUE_TIMEOUT_S` (`app.py:430-449`); the telemetry bus
  is bounded and drained per turn.
- **Error disclosure:** `DefenseResult.error` and `telemetry_error` carry
  exception class names only; no stack traces, filesystem paths, prompts, or
  secrets. Prompt telemetry stores SHA-256 hashes (`_digest`).
- **No auth in the local dashboard:** acceptable for a localhost demo, not for
  a network deployment. The required decision is documented below.

## 6. Validation evidence

- Targeted: `tests/unit/test_execution.py` — **35 passed**.
- Full suite: **623 passed, 3 skipped, 3 xfailed**.
- `black --check src metrics tests`: clean (123 files).
- `ruff check src metrics tests`: clean.
- `mypy src`: clean (49 files). (`mypy src metrics tests` still reports the
  pre-existing GEN-03 errors in `tests/unit/execution/test_output_security.py`,
  untouched here.)
- These are unit/integration checks with mocked or local components; no
  production load or concurrency claim is made.

## 7. Remaining deployment requirements

Before exposing this beyond localhost, decide and implement:

1. Authentication/authorization model (API keys, OIDC, mTLS) and tenancy.
2. TLS termination and trusted-proxy configuration.
3. CORS policy and security headers.
4. HTTP request body limit, rate limiting, and per-tenant quotas.
5. Health/readiness endpoints and graceful shutdown.
6. Multi-worker telemetry transport (the in-process bus is single-node).
7. Secret management and model-availability startup checks.
8. A stable error envelope that never exposes internal exception names.

## 8. Status

**PARTIAL.** Core execution API functionality exists, is unified, bounded, and
fail-closed, with regression coverage for oversized and malformed input. The
production-facing HTTP contract, auth boundary, health/readiness surface, and
deploy assets do not exist; they depend on deployment decisions and were not
invented. The documentation mismatch is recorded here rather than silently
"fixed" by expanding scope.

## 9. New issues discovered (not fixed)

- `docs/PROJECT_STRUCTURE.md` describes unimplemented API/deploy/health paths.
- `execution.py` docstring says evaluation shares the boundary; the evaluation
  runner intentionally uses `cascade.run` for per-stage measurement.
- `DefenseRequest` has no empty-prompt/size validation of its own; the limit is
  runtime-configured in Tier 1.
- No timeout parameter exists on `execute_defense` for caller-supplied
  completions.
