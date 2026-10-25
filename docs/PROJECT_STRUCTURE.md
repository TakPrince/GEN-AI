# Project Structure

**Status:** Prescriptive target layout. This workspace currently contains the OpenCode runtime and this document, not the application source. The tree below is the placement contract for every scaffold, package, and dataset added from this point forward.

**Scope:** The local-first LLM red-teaming and three-tier guardrail platform described in the technical blueprint: FastAPI request handling, Ollama/vLLM/Hugging Face inference, an asynchronous telemetry bus, a SQLite/Postgres/DuckDB evaluation store, and a closed calibration loop.

**Reading rule:** Each directory's opening definition below is exactly two sentences. The first sentence states what lives there; the second states its structural role in the safety platform loop.

---

## Requested Components at a Glance

| Concern | Canonical home | Placement rule |
|---|---|---|
| Core middleware filters | `src/safety_platform/middleware/filters/` | Transport-scoped filters only. They delegate all safety judgment to `guardrails/`. |
| Asynchronous collection endpoints | `src/safety_platform/api/v1/collections/` | Async REST handlers for runs, conversations, events, datasets, versions, and evaluations. Bulk event ingestion returns `202` after enqueue. |
| Database configuration models | `src/safety_platform/db/config/` | Typed settings, DSN parsing, pool sizing, WAL and PRAGMA policy, and backend selection. |
| Database entity models | `src/safety_platform/db/models/` | SQLAlchemy models only. Schema changes also require a migration in `migrations/versions/`. |
| Testing datasets | `datasets/` and `tests/data/` | Full or licensed corpora go to `datasets/` with a manifest. Small deterministic fixtures go to `tests/data/`. |

---

## The Safety Loop in One Pass

```text
red-team orchestrator
  -> execution endpoint
  -> request context + body/rate filters
  -> input filter
  -> Tier 1 deterministic checks
  -> Tier 2 embedding-centroid score
  -> Tier 3 classifier (conditional)
  -> local inference adapter
  -> output filter
  -> Tier 3 output policy
  -> response router
  -> safe response / canned refusal / redaction

           |
           +--> telemetry bus (async, bounded, fire-and-forget)
                    |
                    +--> batch consumer --> evaluation store
                                              |
                                              +--> ASR / FPR / latency / partial compliance
                                              +--> threshold calibration
                                              +--> learned Tier 3 router
                                              +--> next red-team run
```

The loop closes only when every stage emits a correlated event and the evaluation store can reconstruct the full trajectory: prompt, per-tier verdicts, model output, final verdict, and latency. A change that breaks that reconstruction breaks the product, not just observability.

### Stage Budgets

| Stage | Owning package | p95 budget from the blueprint |
|---|---|---|
| Input guardrail | `guardrails/tier1/`, `guardrails/tier2/`, `guardrails/tier3/` | `< 60 ms` |
| Local inference | `inference/` | `300 ms` to `4 s`, model and hardware dependent |
| Output guardrail | `guardrails/tier3/`, `middleware/filters/` | `< 80 ms` |
| Response router | `guardrails/response_router.py` | `< 5 ms` |
| Telemetry emission | `telemetry/` | `~0 ms` added to the critical path |

---

## Non-Negotiable Invariants

1. The request path never awaits telemetry persistence.
2. Middleware does not classify content; it delegates to `guardrails/`.
3. Collection endpoints do not execute SQL; they call repositories or services.
4. Promoted guardrail versions are immutable and recorded with a configuration hash.
5. Datasets are referenced by manifest, never by an ad hoc local path.
6. Every safety event carries `turn_id`, `conversation_id`, `stage`, `verdict`, `latency_ms`, and `prompt_hash`.
7. Secrets live in environment variables; only `.env.example` is committed.

---

## Repository Tree

```text
safety-platform/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── Makefile
├── README.md
├── CONSTRAINTS.md
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── .github/
│   └── workflows/
├── configs/
│   ├── guardrails/
│   ├── policies/
│   └── telemetry/
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── domain/
│   ├── manifests/
│   ├── schemas/
│   └── scripts/
├── deploy/
│   ├── compose/
│   ├── docker/
│   └── systemd/
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── adr/
│   ├── architecture/
│   ├── api/
│   └── runbooks/
├── migrations/
│   └── versions/
├── scripts/
├── src/
│   └── safety_platform/
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── main.py
│       ├── api/
│       │   ├── deps.py
│       │   ├── errors.py
│       │   └── v1/
│       │       ├── router.py
│       │       ├── execution/
│       │       │   └── cascade.py
│       │       └── collections/
│       │           ├── conversations.py
│       │           ├── datasets.py
│       │           ├── evaluations.py
│       │           ├── events.py
│       │           ├── guardrail_versions.py
│       │           └── runs.py
│       ├── middleware/
│       │   ├── registry.py
│       │   ├── request_context.py
│       │   ├── error_shield.py
│       │   └── filters/
│       │       ├── audit_log.py
│       │       ├── body_limit.py
│       │       ├── input_filter.py
│       │       ├── output_filter.py
│       │       └── rate_limit.py
│       ├── guardrails/
│       │   ├── cascade.py
│       │   ├── response_router.py
│       │   ├── router.py
│       │   ├── thresholds.py
│       │   ├── verdicts.py
│       │   ├── tier1/
│       │   │   ├── canary.py
│       │   │   ├── denylist.py
│       │   │   └── regex_filter.py
│       │   ├── tier2/
│       │   │   ├── centroid_index.py
│       │   │   ├── embedder.py
│       │   │   └── similarity.py
│       │   └── tier3/
│       │       ├── classifier.py
│       │       ├── client.py
│       │       └── policy_filter.py
│       ├── orchestrator/
│       │   ├── attack_state.py
│       │   ├── budgets.py
│       │   ├── loop.py
│       │   ├── replay.py
│       │   ├── scorers.py
│       │   ├── mutators/
│       │   │   ├── crescendo.py
│       │   │   ├── encoding.py
│       │   │   ├── pair.py
│       │   │   ├── roleplay.py
│       │   │   └── tap.py
│       │   └── strategies/
│       │       └── registry.py
│       ├── inference/
│       │   ├── base.py
│       │   ├── health.py
│       │   ├── huggingface.py
│       │   ├── ollama.py
│       │   ├── registry.py
│       │   ├── vllm.py
│       │   └── templates/
│       │       ├── refusal.jinja2
│       │       └── sealed_system.jinja2
│       ├── telemetry/
│       │   ├── backpressure.py
│       │   ├── bus.py
│       │   ├── consumer.py
│       │   ├── emitters.py
│       │   ├── events.py
│       │   ├── metrics.py
│       │   └── transports/
│       │       ├── asyncio_queue.py
│       │       └── kafka.py
│       ├── db/
│       │   ├── base.py
│       │   ├── session.py
│       │   ├── types.py
│       │   ├── config/
│       │   │   ├── backends.py
│       │   │   ├── engine.py
│       │   │   └── settings.py
│       │   ├── models/
│       │   │   ├── attack_state.py
│       │   │   ├── conversation.py
│       │   │   ├── dataset_manifest.py
│       │   │   ├── evaluation_result.py
│       │   │   ├── guardrail_trace.py
│       │   │   ├── guardrail_version.py
│       │   │   └── run.py
│       │   └── repositories/
│       │       ├── conversations.py
│       │       ├── evaluations.py
│       │       ├── runs.py
│       │       ├── traces.py
│       │       └── versions.py
│       ├── evaluation/
│       │   ├── calibration/
│       │   │   ├── job.py
│       │   │   ├── promotion.py
│       │   │   └── search.py
│       │   ├── harness/
│       │   │   ├── circleguardbench.py
│       │   │   ├── domain.py
│       │   │   └── tensor_trust.py
│       │   ├── metrics/
│       │   │   ├── asr.py
│       │   │   ├── fpr.py
│       │   │   ├── latency.py
│       │   │   └── partial.py
│       │   ├── reports/
│       │   │   └── render.py
│       │   └── routing/
│       │       ├── features.py
│       │       ├── policy.py
│       │       └── train.py
│       ├── schemas/
│       │   ├── events.py
│       │   ├── pagination.py
│       │   ├── requests.py
│       │   ├── responses.py
│       │   └── verdicts.py
│       ├── config/
│       │   ├── feature_flags.py
│       │   ├── loader.py
│       │   ├── logging.py
│       │   └── settings.py
│       └── core/
│           ├── clock.py
│           ├── errors.py
│           ├── hashing.py
│           ├── ids.py
│           ├── result.py
│           └── security.py
└── tests/
    ├── conftest.py
    ├── unit/
    ├── integration/
    ├── contract/
    ├── e2e/
    ├── regression/
    ├── performance/
    ├── fixtures/
    │   ├── model_stubs/
    │   ├── prompts/
    │   └── traces/
    └── data/
```

Every Python package directory contains `__init__.py`; those files are omitted from the tree for readability.

---

## Repository-Level Directories

### `configs/`

Declarative, reviewable configuration lives here, including guardrail thresholds, policy text, and telemetry limits. It is the versioned control surface that changes safety-loop behavior without editing Python.

### `configs/guardrails/`

Each YAML file pins a named guardrail version with Tier 1 rules, Tier 2 cosine bands, Tier 3 thresholds, and routing policy. Promoted calibration output lands here as an immutable version so every decision can be replayed exactly.

### `configs/policies/`

Domain policy definitions, refusal copy, harm-category maps, and redaction rules live here as reviewable artifacts. They define what the cascade enforces while keeping policy changes separate from detection mechanics.

### `configs/telemetry/`

Queue bounds, batch sizes, drop thresholds, and exporter settings live here. This is where backpressure policy is tuned so logging trouble cannot become safety trouble.

### `datasets/`

The permanent home for versioned evaluation corpora and their provenance metadata. It keeps benchmark and domain data out of application code while giving the safety loop a reproducible input surface.

### `datasets/raw/`

Immutable upstream downloads from CircleGuardBench, Tensor Trust, and other sources are stored here and never edited in place. This preserves provenance so every reported metric can be traced to an exact source snapshot.

### `datasets/processed/`

Normalized, deduplicated, and split datasets are generated here from raw inputs by deterministic scripts. It is the only dataset form the evaluation harness may read during a run.

### `datasets/domain/`

Hand-curated benign and adversarial examples for the target domain live here, including security code review, CTF write-ups, fiction, and medical or legal text. This is the calibration set that exposes domain-specific false positives.

### `datasets/manifests/`

Each manifest records source URL, license, content hash, row counts, split rules, and dataset version. It is the contract that makes dataset drift and accidental leakage visible.

### `datasets/schemas/`

JSON Schema or Pydantic-compatible definitions for dataset rows live here. They let ingestion reject malformed examples before those examples reach metrics or training.

### `datasets/scripts/`

Fetch, verify, transform, and split utilities live here and are invoked only through documented commands. They keep dataset lineage reproducible instead of burying transformations in notebooks.

### `deploy/`

Docker, Compose, and systemd assets live here alongside environment templates. This is the boundary where the local safety loop becomes a repeatable service without leaking deployment assumptions into the package.

### `docs/`

Architecture notes, ADRs, API references, and operational runbooks live here. It is the decision memory that explains why the loop is shaped this way, beyond what the code can show.

### `docs/adr/`

Numbered Architecture Decision Records capture expensive choices such as tier boundaries, storage-engine roles, and dataset licensing. Each ADR preserves the alternatives and consequences that future changes must respect.

### `docs/architecture/`

Living diagrams and prose for dataflow, trust boundaries, and failure modes live here. They are the shared map used when a change crosses more than one package.

### `docs/api/`

OpenAPI snapshots, endpoint contracts, and example request and response payloads live here. They make the asynchronous collection surface reviewable independently of implementation.

### `docs/runbooks/`

Operational procedures for queue saturation, model outage, dataset refresh, and guardrail rollback live here. They turn safety-loop incidents into repeatable recovery steps.

### `migrations/`

Alembic environment configuration and ordered schema revisions live here. Every persisted trace, run, and version record must arrive through a migration so the evaluation store remains reproducible.

### `migrations/versions/`

The append-only sequence of database revisions lives here. This is the only place schema history is allowed to accumulate.

### `scripts/`

Thin operator entry points for red-team runs, calibration, evaluation, replay, and dataset refresh live here. They compose package APIs into repeatable commands and contain no safety logic of their own.

---

## Application Package

### `src/`

All importable application code lives under the src layout so tests run against the installed package rather than the working directory. It is the single code root from which the safety loop is assembled.

### `src/safety_platform/`

The top-level application package owns composition, lifecycle, and the public seams between API, guardrails, orchestration, telemetry, and storage. It is the assembly point where the closed loop is wired together without circular imports.

---

## API Layer

### `src/safety_platform/api/`

The HTTP contract layer translates requests into validated commands and responses into stable envelopes. It is the outer membrane of the safety loop and contains no detection, SQL, or provider logic.

### `src/safety_platform/api/v1/`

Versioned routing and resource namespaces live here so breaking contract changes can ship beside a stable surface. It is the compatibility boundary for every client and red-team harness.

### `src/safety_platform/api/v1/execution/`

The request-time execution endpoint that runs one prompt through the defense cascade and local inference lives here. This is the hot path where the end-to-end latency budget is enforced.

### `src/safety_platform/api/v1/collections/`

Asynchronous REST collection endpoints for runs, conversations, events, datasets, guardrail versions, and evaluations live here. Handlers validate, authorize, and delegate, with bulk telemetry ingestion returning `202` after enqueue so persistence never blocks the caller.

#### Collection Modules

| Module | Owns |
|---|---|
| `runs.py` | `/v1/runs` lifecycle: create, list, inspect, cancel, and status transitions. |
| `conversations.py` | `/v1/conversations` multi-turn state, turn history, and replay prefixes. |
| `events.py` | `/v1/events` bulk telemetry ingestion; validates batches, enqueues to the async bus, returns `202`. |
| `datasets.py` | `/v1/datasets` manifest registration, validation status, and split metadata. |
| `guardrail_versions.py` | `/v1/guardrail-versions` immutable versions, promotion state, and configuration hashes. |
| `evaluations.py` | `/v1/evaluations` metric results keyed by guardrail version and dataset manifest. |

`api/deps.py` supplies authentication, pagination, repository, and service dependencies. `api/errors.py` maps domain failures into a single stable error envelope.

---

## Middleware Layer

### `src/safety_platform/middleware/`

Ordered request-scoped filters live here, covering correlation, body limits, rate limits, audit logging, and the input and output guard seams. This is the enforcement corridor that every prompt and response crosses before a business handler runs.

### `src/safety_platform/middleware/filters/`

Individual filter implementations live here, including body-size rejection, rate limiting, input-guard delegation, output-guard delegation, and audit emission. Each filter does one transport concern and delegates every safety judgment to the guardrails package.

#### Middleware Modules

| Module | Owns |
|---|---|
| `registry.py` | The single ordered registration point for the middleware stack. |
| `request_context.py` | Assignment and propagation of `turn_id`, `conversation_id`, and `trace_id`. |
| `filters/body_limit.py` | Rejection of oversized prompts before parsing or model work. |
| `filters/rate_limit.py` | Per-client and global limits that protect local inference capacity. |
| `filters/input_filter.py` | Delegation to the input side of the cascade and short-circuit responses. |
| `filters/output_filter.py` | Delegation to output policy, redaction, and refusal rewriting. |
| `filters/audit_log.py` | Access events and configuration changes, never safety verdict truth. |
| `error_shield.py` | Mapping unexpected failures to safe, non-leaking responses. |

**Boundary rule:** if a change needs to know whether text is harmful, it belongs in `guardrails/`. If it needs to know about the HTTP lifecycle, it belongs in `middleware/`.

---

## Guardrails

### `src/safety_platform/guardrails/`

The safety decision domain lives here: tier implementations, cascade sequencing, verdict types, thresholds, and the response router. It is the policy engine of the loop and must remain callable from both the live API and offline replay.

### `src/safety_platform/guardrails/tier1/`

Cheap deterministic checks such as deny-list regex, canary-token echo detection, and input normalization live here. This tier is the first latency gate and must resolve obvious attacks in milliseconds.

### `src/safety_platform/guardrails/tier2/`

Embedding generation, the known-attack centroid index, and cosine-distance scoring live here. This tier converts ambiguous text into a calibrated similarity signal before the expensive classifier is considered.

### `src/safety_platform/guardrails/tier3/`

The local Llama-Guard-class classifier client, policy-specific semantic filters, and borderline-case logic live here. This tier is the expensive escalation path and is invoked only when cheaper tiers cannot decide.

#### Guardrail Modules

| Module | Owns |
|---|---|
| `cascade.py` | Tier order, short-circuit rules, verdict aggregation, and per-stage latency capture. |
| `router.py` | The learned minimum-sufficient-defense decision for whether Tier 3 is invoked. |
| `response_router.py` | Final routing to `safe`, `refuse`, or `redact`, with the canned refusal fallback. |
| `verdicts.py` | Shared verdict enums and value objects used by API, telemetry, and evaluation. |
| `thresholds.py` | Loading and validation of the active guardrail version's threshold set. |

---

## Orchestrator

### `src/safety_platform/orchestrator/`

The attacker-side loop lives here: persistent attack state, mutation strategies, turn budgets, scoring, and replay. It is the half of the system that turns stored telemetry into the next adversarial probe.

### `src/safety_platform/orchestrator/mutators/`

Individual mutation techniques such as PAIR, TAP, encoding, role-play wrapping, and crescendo steps live here. Each mutator is a small composable strategy so the loop can record which technique produced which outcome.

### `src/safety_platform/orchestrator/strategies/`

The strategy registry and selection policy live here alongside the technique metadata used for stratified reporting. This is where the attacker decides what to try next based on current run state.

---

## Inference

### `src/safety_platform/inference/`

Provider-agnostic model invocation lives here, with one adapter per local runtime and a sealed prompt-template boundary. It is the only package allowed to call Ollama, vLLM, or Hugging Face, which keeps model switching out of the safety logic.

### `src/safety_platform/inference/templates/`

Sealed system prompts and canned refusal templates live here as versioned text assets. They isolate system instructions from untrusted input and give the response router a consistent fallback.

---

## Telemetry

### `src/safety_platform/telemetry/`

Structured event definitions, the async bus, fire-and-forget emitters, a batch consumer, and drop metrics live here. It is the non-blocking observation layer that makes defense latency independent of persistence latency.

### `src/safety_platform/telemetry/transports/`

Concrete bus adapters live here, including the in-process asyncio queue for single-node runs and Kafka or Redpanda for multi-worker deployments. This is the swap point between local development and scaled transport without touching emitters or consumers.

**Boundary rule:** emitters never await the consumer, and the consumer never decides whether a request is safe. Saturation drops telemetry and increments `telemetry_dropped_total`; it never blocks or alters a guardrail verdict.

---

## Database

### `src/safety_platform/db/`

The persistence boundary lives here: engine configuration, session management, ORM models, and repositories. It is the only package that speaks to SQLite, Postgres, or DuckDB, keeping storage choice out of the request path.

### `src/safety_platform/db/config/`

Typed database configuration models and engine construction live here, covering DSN parsing, pool sizing, SQLite WAL pragmas, and backend selection. This is the only place connection policy is decided, keeping storage configuration out of models and endpoints.

#### Database Configuration Modules

| Module | Owns |
|---|---|
| `settings.py` | The typed `DatabaseSettings` model: backend, DSN, pool sizes, timeouts, and WAL policy. |
| `engine.py` | Async engine and session-factory construction from validated settings. |
| `backends.py` | Backend-specific capabilities and pragmas for SQLite, Postgres, and DuckDB. |

### `src/safety_platform/db/models/`

SQLAlchemy models for conversations, runs, attack states, guardrail traces, guardrail versions, evaluation results, and dataset manifests live here. This is the canonical persisted shape of the safety loop and the source of truth for migrations.

### `src/safety_platform/db/repositories/`

All query and transaction logic lives here behind narrow repository interfaces. It is the only layer the API, telemetry consumer, and evaluation jobs use to read or write loop state.

**Boundary rule:** Pydantic contracts in `schemas/` are not ORM models. SQLAlchemy models stay in `db/models/`, and every schema change ships with a migration.

---

## Evaluation and Calibration

### `src/safety_platform/evaluation/`

Offline measurement and learning live here: ASR, FPR, latency, partial-compliance metrics, calibration, learned routing, benchmark harnesses, and reports. It is the feedback half of the loop that converts persisted traces into the next guardrail version.

### `src/safety_platform/evaluation/metrics/`

Metric implementations for attack success rate, false-positive rate, latency percentiles, and partial compliance live here. Each metric reads the same trace schema and can be stratified by category, turn depth, and tier reached.

### `src/safety_platform/evaluation/calibration/`

Threshold search, held-out validation, promotion gates, and the scheduled calibration job live here. This is the mechanism that turns red-team transcripts into a safer, versioned guardrail configuration.

### `src/safety_platform/evaluation/routing/`

Feature extraction, training, and export for the learned minimum-sufficient-defense router live here. It produces the model artifact consumed by the request-time router without adding an LLM call to the hot path.

### `src/safety_platform/evaluation/harness/`

Adapters for CircleGuardBench, Tensor Trust, and the in-house domain corpus live here. They normalize each benchmark into the project's common dataset and metric contracts.

### `src/safety_platform/evaluation/reports/`

Report rendering and trend tables keyed by guardrail version, dataset, and metric live here. This is where ASR and FPR movement becomes reviewable evidence rather than raw rows.

---

## Shared Contracts and Runtime

### `src/safety_platform/schemas/`

Pydantic request, response, event, verdict, and pagination contracts live here. They are the shared language between the HTTP layer, telemetry bus, persistence adapters, and evaluation jobs.

### `src/safety_platform/config/`

Typed runtime settings and their loaders live here, sourced from environment variables and checked at startup. This is the boundary between deploy-time configuration and code, distinct from the declarative guardrail content in top-level `configs/`.

### `src/safety_platform/core/`

Small cross-cutting primitives such as clocks, ID generation, hashing, error types, and result wrappers live here. They carry no domain policy and keep the safety loop's dependencies acyclic.

### Entry Points

| File | Owns |
|---|---|
| `app.py` | The `create_app()` factory, lifespan wiring, middleware registration, and dependency composition. |
| `main.py` | The production ASGI entry point used by the process manager. |
| `__main__.py` | The `python -m safety_platform` developer entry point. |

---

## Test Layout

### `tests/`

The verification suite lives here, split by confidence level and runtime cost. It is the guardrail against silent safety regressions in every part of the loop.

### `tests/unit/`

Fast, deterministic tests for pure functions and single components live here, with model and network calls replaced by stubs. They pin the behavior of tiers, metrics, and routing logic before integration costs are paid.

### `tests/integration/`

Tests that exercise real database sessions, the async bus, provider adapters, and middleware ordering live here. They verify the seams where the safety loop most often breaks.

### `tests/contract/`

Schema, OpenAPI, event-envelope, and provider-adapter contract tests live here. They catch breaking changes in the asynchronous collection API and telemetry format before clients do.

### `tests/e2e/`

Full-loop tests that run an adversarial prompt through mutator, cascade, inference stub or local model, telemetry, and evaluation live here. They prove the closed loop works as an assembled system, not just as isolated packages.

### `tests/regression/`

Golden-set tests pinned to specific guardrail versions and dataset manifests live here. They fail when ASR, FPR, or latency moves beyond the accepted tolerance, which is the core safety regression surface.

### `tests/performance/`

Latency and throughput tests for the request path, telemetry backpressure, and batch consumer live here. They enforce the p95 budgets and prove that dropping telemetry never blocks a safety decision.

### `tests/fixtures/`

Reusable model stubs, prompt corpora, trace snapshots, and dependency overrides live here. They make tests readable and keep setup noise out of individual test files.

### `tests/data/`

Small, deterministic, committed datasets for unit, contract, and golden tests live here. Large or licensed corpora belong in top-level `datasets/`, keeping the test suite fast and reproducible.

---

## Placement Decision Rules

| If the artifact is... | It belongs in... | It must not go in... |
|---|---|---|
| A transport concern such as body size, rate limit, correlation, or audit | `middleware/filters/` or `middleware/` | `guardrails/` or endpoint handlers |
| A safety decision, threshold, or filter | `guardrails/tierN/` and `configs/guardrails/` | `middleware/` |
| A new asynchronous REST collection | `api/v1/collections/` | `router.py` or a service module |
| A persisted entity | `db/models/` plus `migrations/versions/` | `schemas/` |
| A database connection or pool concern | `db/config/` | `config/settings.py` or endpoint code |
| A full or licensed dataset | `datasets/` with a manifest | `src/`, `tests/data/`, or a notebook |
| A small deterministic test corpus | `tests/data/` | `datasets/raw/` |
| A metric or benchmark adapter | `evaluation/metrics/` or `evaluation/harness/` | an ad hoc notebook |
| A prompt or refusal template | `inference/templates/` | `configs/policies/` |
| A one-off operator command | `scripts/` | importable application code |
| A new top-level directory | An ADR in `docs/adr/` first | the tree by convenience |

---

## Dependency Direction

| Layer | May depend on | Must not depend on |
|---|---|---|
| `api` | `schemas`, `guardrails`, `orchestrator`, `telemetry`, `db.repositories`, `config`, `core` | `db.models`, provider SDKs |
| `middleware` | `guardrails.cascade`, `guardrails.verdicts`, `schemas`, `telemetry`, `core` | `db`, `api.v1` |
| `guardrails` | `inference`, `telemetry.emitters`, `schemas`, `config`, `core` | `api`, `db`, `orchestrator` |
| `orchestrator` | `schemas`, `db.repositories`, an HTTP client to the execution API | guardrail tier internals |
| `inference` | `config`, `core` | `guardrails`, `api`, `db` |
| `telemetry` | `schemas.events`, `core` | `db`, `guardrails` |
| `db` | `core`, `schemas` in repositories only | `api`, `guardrails`, `telemetry` |
| `evaluation` | `db.repositories`, `datasets`, `schemas`, `guardrails`, `inference` | `api.v1` handlers or the request path |
| `core` | Standard library and small primitives | any application package |

The dependency graph is intentionally acyclic. If a change requires a back-edge, it is a design change and needs an ADR.

---

## Permanence and Versioning Rules

1. `configs/guardrails/` and the `guardrail_versions` table move together; a promoted version is never edited in place.
2. `datasets/manifests/` pins every corpus by hash, license, and split rule; evaluation results without a manifest are invalid.
3. `migrations/versions/` is append-only; merged revisions are never rewritten.
4. Telemetry records are append-only, with retention handled only by documented jobs in `scripts/`.
5. `tests/data/` contains only small committed fixtures; large corpora are fetched into `datasets/`.
6. Secrets never enter the repository; `.env.example` documents required variable names without values.
7. Any deviation from this layout requires an ADR and an update to this document.

---

## Change Protocol

When adding a component:

1. Place it using the decision table above before writing code.
2. Update the tree and the relevant two-sentence definition if a new component directory is introduced.
3. Add an ADR for a new tier, storage backend, public endpoint family, or dataset source.
4. Extend the dependency table if the new component introduces a new allowed edge.
5. Add or update tests at the lowest layer that can prove the change, then the regression test that would catch a safety regression.

The goal is not a tidy tree. The goal is that any engineer or agent can answer three questions without guessing: where does this code live, which layer may import it, and what part of the safety loop does it serve.
