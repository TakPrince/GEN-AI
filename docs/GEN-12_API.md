# GEN-12 — Safety Platform HTTP API

The API is a thin transport layer over `execute_defense`. It adds
authentication, request limits, concurrency bounds, stable errors and health
checks; guardrails, telemetry and the target call remain in the existing
execution boundary.

## Run

```bash
export SAFETY_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
PYTHONPATH=src python -m safety_platform.api
# or: make api
```

Defaults: `127.0.0.1:8000`. Set `SAFETY_API_HOST=0.0.0.0` only behind a
TLS-terminating reverse proxy.

Local development without a key (explicit, never the default):

```bash
SAFETY_API_ALLOW_UNAUTHENTICATED=1 PYTHONPATH=src python -m safety_platform.api
```

## Routes

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/healthz` | no | Liveness. |
| `GET` | `/readyz` | no | Runtime initialized and telemetry DB reachable. Does not call Ollama. |
| `POST` | `/v1/chat` | yes | One prompt through the full defense cascade. |
| `GET` | `/docs` | no | OpenAPI UI (disable in a hardened deployment). |

Request:

```json
{"prompt": "hello", "request_id": "optional-id", "metadata": {"tenant": "a"}}
```

`prompt` is required, non-blank, and limited to 16,384 UTF-8 bytes. Unknown
fields are rejected. `metadata` is limited to 16 entries (key ≤128, value ≤512
bytes). `request_id` is optional, ≤64 chars, `[A-Za-z0-9._-]`.

Response (200):

```json
{
  "request_id": "…",
  "trace_id": "…",
  "outcome": "safe|partial|blocked",
  "blocked": false,
  "response": "…",
  "route": "safe|partial|refused|error",
  "stage": "tier1|tier2|tier3|router|output",
  "tier3_invoked": true,
  "latency_ms": 12.3,
  "telemetry_persisted": true
}
```

A defense failure returns `503` with a generic envelope and the refusal text; it
never returns the target output:

```json
{"error": {"code": "execution_error"}, "outcome": "error", "response": "I cannot help with that request."}
```

## Error codes

| Status | Code | Meaning |
| --- | --- | --- |
| 401 | `unauthorized` | Missing/wrong `X-API-Key`. |
| 413 | `request_too_large` | `Content-Length` above 32,768 bytes. |
| 422 | `invalid_request` | Schema/limit violation; no input echoed. |
| 503 | `auth_not_configured` | No server key and unauthenticated mode not enabled. |
| 503 | `server_busy` | Concurrency queue wait exceeded. |
| 503 | `execution_error` | Guard/target failure; fail-closed. |
| 504 | `execution_timeout` | Execution exceeded the configured timeout. |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAFETY_API_KEY` | unset | Required `X-API-Key` value. Unset fails closed unless local mode is enabled. |
| `SAFETY_API_ALLOW_UNAUTHENTICATED` | `0` | Explicit local-only bypass; never use in a network deployment. |
| `SAFETY_API_HOST` / `SAFETY_API_PORT` | `127.0.0.1` / `8000` | Bind address. |
| `SAFETY_API_DB` | `metrics/api.sqlite3` | Telemetry SQLite path. |
| `SAFETY_API_TARGET_MODEL` | `qwen3:8b` | Ollama target model. |
| `SAFETY_API_NUM_PREDICT` | `256` | Target generation cap. |
| `SAFETY_API_MAX_PROMPT_BYTES` | `16384` | Prompt limit (also enforced by Tier 1). |
| `SAFETY_API_MAX_BODY_BYTES` | `32768` | HTTP body limit. |
| `SAFETY_API_MAX_CONCURRENCY` | `2` | In-process execution slots. |
| `SAFETY_API_QUEUE_TIMEOUT_S` | `30` | Wait for a slot before `server_busy`. |
| `SAFETY_API_EXECUTION_TIMEOUT_S` | `120` | Execution deadline. |

## Security properties

- All requests pass through `execute_defense`; the target is invoked only after
  input guards pass, and the output guard runs before delivery.
- Guard, target and output failures are fail-closed (`outcome=error`, refusal
  text, `503`).
- Errors expose only stable codes; no stack traces, paths, prompts, model output
  or exception class names.
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, `Cache-Control: no-store`.
- API keys are compared with `secrets.compare_digest`.
- Telemetry failures do not turn a defense outcome into a success.

## Limitations / deployment requirements

- Single-node only: the concurrency semaphore and SQLite telemetry are
  in-process. Multi-worker deployments need a shared limiter and telemetry
  transport.
- The body limit uses `Content-Length`; a reverse proxy must also cap chunked
  bodies.
- No built-in rate limiting beyond concurrency; add per-key limits at the proxy
  or a shared store.
- TLS, CORS, key rotation, tenancy, audit retention and OpenAPI exposure are
  deployment decisions. `/docs` is enabled by default for development.
- `asyncio.wait_for` bounds the response, but a blocking target thread cannot be
  forcibly killed; the Ollama client timeout and concurrency cap bound it.
- Readiness checks the runtime and telemetry DB, not Ollama model availability;
  per-request failures fail closed. Monitor Ollama separately.
