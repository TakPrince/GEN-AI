# Phase 1 Implementation Report — Reliability, Concurrency & Payload Hardening

Branch: `phase1-reliability-hardening` · Base: `4ad726f` · Date: 2026-11-12

## Scope

Phase 1 of the reliability plan (from `ARCHITECTURE_ANALYSIS_REPORT.md`):
P1-01 retry transient Ollama errors, P1-02 unify concurrency admission,
P1-03 centralize payload limits, P1-04 assess HTTP connection reuse.
Method: TDD (RED → GREEN → commit per slice), gates per `CONSTRAINTS.md`.

## Status

| Item | Status | Commit |
|------|--------|--------|
| P1-01 Retry transient 429/503 with bounded backoff + Retry-After | FIXED AND VERIFIED | `155f1f3` |
| P1-02 Unified thread-safe admission gate for all Ollama calls | FIXED AND VERIFIED | `b412ca9` |
| P1-03 Centralized payload limits at every trust boundary | FIXED AND VERIFIED | `27259d0` |
| P1-04 HTTP connection reuse (keep-alive) | DEFERRED WITH EVIDENCE | — |

## What was built

### P1-01 — Retry (`src/safety_platform/ollama_runtime.py`)

Previously `_retryable()` rejected *all* `HTTPError`, so a transient 429/503
immediately surfaced as a guard failure. Now both Ollama clients (`OllamaClient._post`,
`UrllibOllamaTransport._generate`) use one policy:

- 429 and 503 are transient → retried up to 2 attempts total;
- 4xx/5xx others are permanent → immediate failure (500 enshrined as permanent);
- `Retry-After` honored as numeric seconds only, capped at 5.0 s (HTTP-date form
  deliberately ignored — documented shortcut);
- exponential backoff otherwise (1 s, 2 s).

### P1-02 — Admission (`OllamaAdmission`, `threading.BoundedSemaphore`)

Three uncoordinated semaphores (API `asyncio.Semaphore(2)`, dashboard
`_TARGET_SLOTS(2)`, tier3 transport `BoundedSemaphore(1)`) allowed
2+2+1 = 5 concurrent Ollama requests → OOM pressure. One shared process-wide
gate now bounds every Ollama call: target, judge, embeddings, echo probe,
tier3 classifier. Not asyncio, because the dashboard creates a fresh event
loop per turn. Env: `SAFETY_OLLAMA_MAX_CONCURRENT` (default 1),
`SAFETY_OLLAMA_QUEUE_TIMEOUT_S` (default 60).

- Gate acquired per urlopen attempt, released during retry backoff.
- Per-callsite private slots (tier3) nest *inside* the shared gate; ordering
  is never reversed, so no deadlock.
- Queue timeout maps to existing contracts: `OllamaError("admission_timeout")`
  / `OllamaGuardError("capacity_exhausted")` — callers see failure without
  blocking forever, and cancellation does not leak permits.
- Known documented limitation: judge calls in `orchestrator/loop.py`/`multi.py`
  run synchronously on the event loop (pre-existing); admission wait there can
  block the loop up to the queue timeout. Accepted; orchestrator redesign is
  Phase 2 scope. The health probe (`/api/tags`, app.py) stays ungated so the
  status display never queues behind inference.

### P1-03 — Payload limits (`config/registry.py`)

Single frozen source of truth: `DEFAULT_PAYLOAD_LIMITS`
(prompt 16384 B, body 32768 B, metadata 16×(128 key/512 value) B, request-id 64
chars) and `utf8_byte_length()` — strict UTF-8 byte counting that *rejects*
lone surrogates instead of encoding them with `surrogatepass` and storing
garbage. Wired at the trust boundaries:

- **API**: prompt/metadata/request-id validators count UTF-8 bytes (multibyte
  characters no longer slip through character counts) and reject invalid
  unicode with 422; `SAFETY_API_MAX_PROMPT_BYTES` / `SAFETY_API_MAX_BODY_BYTES`
  remain env-overridable; body-middleware bound unchanged (413).
- **Tier 2**: `VectorFilter(max_input_bytes=…)` fails closed (`ERROR` with
  `input_too_large` / `invalid_unicode`) before any embedding is computed.
- **Dashboard**: regex filter and tier3 guard config both use the central
  prompt limit — this fixes a real bug where a 4–16 KiB prompt that passed
  Tier 1 failed closed at Tier 3 with `prompt_too_large`.
- Tier 1 / Tier 3 *library* defaults stay conservative at 4096 B (defense in
  depth; existing tests enshrine this).

### P1-04 — Connection reuse: DEFERRED WITH EVIDENCE

Micro-benchmark on the live local Ollama server (50 sequential `/api/tags`,
median): urllib fresh-connect **1.89 ms** vs `http.client` keep-alive
**1.39 ms** → **≈0.5 ms per-request connect overhead**. Reasons to defer:

1. Localhost connect cost is ~0.04% of the 9.88 s smoke-test inference
   latency; reuse saves nothing user-visible.
2. `httpx` (async, keep-alive capable) is a dev-only dependency in this
   codebase; adding it to production paths for 0.5 ms is not justified.
3. `http.client` persistent connections are not thread-safe; our call sites
   are multi-threaded, so reuse would need a connection pool — a non-trivial
   new component.

Revisit only if Ollama moves off localhost (network RTT would change the
calculus) or call rates per process rise by orders of magnitude.

## Verification

- Unit: **656 passed, 3 xfailed**, coverage **93.03%** (gate ≥80%)
- New tests: `tests/unit/test_payload_limits.py` (12), API boundary tests (7
  added), registry additions, plus the P1-01 retry suite and P1-02 admission
  suite (13) from earlier slices
- Performance gates: **4 passed** (tier1 p95 < 60 ms, tier2 p95 < 20 ms,
  envelope p95 < 400 ms per `CONSTRAINTS.md`)
- Integration: **18 passed, 3 skipped**
- Static: `ruff check` clean, `black` clean, `mypy src` clean
- Live smoke: real `llama3.2:latest` call through shared admission + retry
  path returned `ok` in 9.88 s; dashboard tier3 path accepts a
  boundary-sized (16384 B) prompt

## Out of scope / follow-ups for Phase 2

- Sync-on-loop judge calls in the orchestrator (pre-existing; redesigned there)
- Connection pooling (P1-04, deferred with evidence above)
- `Retry-After` HTTP-date parsing (cosmetic; Ollama sends numeric seconds)
