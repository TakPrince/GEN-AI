# Implementation Roadmap

## Foundation

- [ ] Package namespace scaffold | AC: "`import safety_platform` succeeds without starting services or reading secrets" | Deps: `src/safety_platform/`, `pyproject.toml`
- [ ] ASGI entrypoints | AC: "`safety_platform.app:app` loads under uvicorn with empty configuration" | Deps: `src/safety_platform/app.py`, `src/safety_platform/main.py`, `src/safety_platform/__main__.py`
- [ ] Typed runtime settings | AC: "invalid environment values fail startup with a typed configuration error" | Deps: `src/safety_platform/config/`, `.env.example`
- [ ] Structured logging context | AC: "request logs include correlation identifiers and guardrail version without raw prompt content" | Deps: `src/safety_platform/config/`, `src/safety_platform/core/`
- [ ] Core identity primitives | AC: "clock, hash, ID, result, and security helpers pass deterministic unit tests" | Deps: `src/safety_platform/core/`, `tests/unit/`
- [ ] Shared schemas | AC: "request, response, event, and verdict models reject incompatible payloads" | Deps: `src/safety_platform/schemas/`, `tests/contract/`
- [ ] Project tooling baseline | AC: "`make lint` and `make test` run from a clean checkout" | Deps: `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml`, `.python-version`
- [ ] Secret-free environment template | AC: "`.env.example` lists every required setting without live credentials" | Deps: `.env.example`, `.gitignore`, `src/safety_platform/config/`

## API

- [ ] Versioned router composition | AC: "`/v1` mounts every execution and collection router" | Deps: `src/safety_platform/api/v1/`, `src/safety_platform/app.py`
- [ ] Request dependency providers | AC: "dependencies resolve settings, DB adapter, cascade, and telemetry bus per request" | Deps: `src/safety_platform/api/deps.py`
- [ ] Uniform API error envelope | AC: "validation and domain failures return distinct stable error codes" | Deps: `src/safety_platform/api/errors.py`, `src/safety_platform/schemas/`
- [ ] Guarded execution endpoint | AC: "the endpoint returns one terminal route without leaking blocked content" | Deps: `src/safety_platform/api/v1/execution/`, `src/safety_platform/guardrails/`
- [ ] Async event ingestion endpoint | AC: "event ingestion returns `202` only after bounded enqueue" | Deps: `src/safety_platform/api/v1/collections/`, `src/safety_platform/telemetry/`
- [ ] Collection read endpoints | AC: "runs, conversations, datasets, versions, and evaluations read exclusively through repositories" | Deps: `src/safety_platform/api/v1/collections/`, `src/safety_platform/db/repositories/`
- [ ] Pagination contract | AC: "cursor pagination returns stable next-page tokens across concurrent inserts" | Deps: `src/safety_platform/schemas/`, `src/safety_platform/api/v1/collections/`

## Middleware

- [ ] Middleware registry | AC: "registered filters execute in one deterministic documented order" | Deps: `src/safety_platform/middleware/registry.py`, `src/safety_platform/middleware/filters/`
- [ ] Request context propagation | AC: "request, conversation, and turn identifiers reach logs and telemetry" | Deps: `src/safety_platform/middleware/request_context.py`, `src/safety_platform/schemas/`
- [ ] Error shield | AC: "unhandled middleware failures produce the canonical error envelope" | Deps: `src/safety_platform/middleware/error_shield.py`, `src/safety_platform/api/errors.py`
- [ ] Body and rate limits | AC: "oversized or excess requests are rejected before cascade execution" | Deps: `src/safety_platform/middleware/filters/`
- [ ] Audit logging boundary | AC: "audit records contain metadata and content hashes but no raw prompt text" | Deps: `src/safety_platform/middleware/filters/audit_log.py`, `src/safety_platform/core/`
- [ ] Guardrail delegation filters | AC: "input and output filters call the cascade with zero local classification rules" | Deps: `src/safety_platform/middleware/filters/input_filter.py`, `src/safety_platform/middleware/filters/output_filter.py`, `src/safety_platform/guardrails/`

## Guardrails

- [ ] Verdict and threshold types | AC: "policy validation rejects ambiguity floors above block thresholds" | Deps: `src/safety_platform/guardrails/verdicts.py`, `src/safety_platform/guardrails/thresholds.py`, `configs/policies/`
- [ ] Shared cascade engine | AC: "input and output directions execute tiers through one orchestration contract" | Deps: `src/safety_platform/guardrails/cascade.py`, `src/safety_platform/guardrails/router.py`
- [ ] Tier 1 deterministic detectors | AC: "regex, deny-list, canary, and redaction checks run without model or network calls" | Deps: `src/safety_platform/guardrails/tier1/`, `configs/guardrails/`
- [ ] Tier 2 embedding scorer | AC: "cosine scores resolve to pass, ambiguous, or block under validated thresholds" | Deps: `src/safety_platform/guardrails/tier2/`, `configs/guardrails/`
- [ ] Tier 3 conditional classifier | AC: "Tier 3 executes only for ambiguous cases and returns schema-valid verdicts" | Deps: `src/safety_platform/guardrails/tier3/`, `src/safety_platform/inference/`
- [ ] Response router | AC: "terminal decisions map to blocked, partial, safe, or error without re-running detection" | Deps: `src/safety_platform/guardrails/response_router.py`
- [ ] Fail-closed error posture | AC: "exceptions, timeouts, and malformed classifier output produce recorded blocks" | Deps: `src/safety_platform/guardrails/cascade.py`, `src/safety_platform/core/errors.py`
- [ ] Skipped-tier records | AC: "every short-circuited tier emits an explicit skipped stage event" | Deps: `src/safety_platform/guardrails/cascade.py`, `src/safety_platform/telemetry/`
- [ ] Direction-specific profiles | AC: "input and output profiles load independent detectors, thresholds, and templates" | Deps: `configs/guardrails/`, `configs/policies/`, `src/safety_platform/guardrails/`

## Inference

- [ ] Adapter contract and registry | AC: "backend resolution imports one adapter without loading optional clients" | Deps: `src/safety_platform/inference/`, `src/safety_platform/config/`
- [ ] Local backend adapters | AC: "Ollama, vLLM, and Hugging Face implementations satisfy the same async interface" | Deps: `src/safety_platform/inference/`
- [ ] Backend health checks | AC: "health probes report backend availability without accepting inference traffic" | Deps: `src/safety_platform/inference/`, `src/safety_platform/api/`
- [ ] Versioned inference templates | AC: "refusal and sealed-system templates load from versioned policy files" | Deps: `src/safety_platform/inference/templates/`, `configs/policies/`

## Orchestrator

- [ ] Run and attack state | AC: "run, conversation, and turn identifiers survive process restart" | Deps: `src/safety_platform/orchestrator/attack_state.py`, `src/safety_platform/db/models/`
- [ ] Multi-turn attack loop | AC: "each turn traverses the guarded execution path before the next mutation" | Deps: `src/safety_platform/orchestrator/loop.py`, `src/safety_platform/api/v1/execution/`
- [ ] Budget enforcement | AC: "turn, token, and time limits stop runs before overrun" | Deps: `src/safety_platform/orchestrator/budgets.py`
- [ ] Mutator and strategy registries | AC: "each mutator and strategy resolves by stable configured name" | Deps: `src/safety_platform/orchestrator/mutators/`, `src/safety_platform/orchestrator/strategies/`, `configs/`
- [ ] Replay and scoring hooks | AC: "fixed-trajectory and closed-loop runs share persisted scoring interfaces" | Deps: `src/safety_platform/orchestrator/replay.py`, `src/safety_platform/orchestrator/scorers.py`
- [ ] Guardrail bypass prevention | AC: "measured runs fail when any cascade stage is skipped" | Deps: `src/safety_platform/orchestrator/loop.py`, `src/safety_platform/guardrails/cascade.py`

## Telemetry

- [ ] Event envelope schema | AC: "events validate identifiers, versions, verdict, and latency fields" | Deps: `src/safety_platform/telemetry/events.py`, `src/safety_platform/schemas/events.py`
- [ ] Non-blocking telemetry bus | AC: "enqueue never awaits capacity, storage, or consumer progress" | Deps: `src/safety_platform/telemetry/bus.py`, `src/safety_platform/telemetry/backpressure.py`
- [ ] Bounded lane isolation | AC: "decision, evidence, and capture lanes enforce independent limits" | Deps: `src/safety_platform/telemetry/`, `configs/telemetry/`
- [ ] Metrics collectors | AC: "collectors emit stage, latency, and loss summaries without raw content" | Deps: `src/safety_platform/telemetry/collectors/`
- [ ] Single consumer and batch writer | AC: "consumer commits 256-event or 250-ms batches through one writer" | Deps: `src/safety_platform/telemetry/consumer.py`, `src/safety_platform/db/adapters/`
- [ ] Independent loss accounting | AC: "queue-full, serialization, commit, and shutdown losses persist independently" | Deps: `src/safety_platform/telemetry/`, `src/safety_platform/db/models/`
- [ ] Transport adapters | AC: "asyncio and Kafka transports preserve one event contract" | Deps: `src/safety_platform/telemetry/transports/`
- [ ] Completeness and shutdown drain | AC: "shutdown drains bounded lanes and marks missing critical events incomplete" | Deps: `src/safety_platform/telemetry/`, `src/safety_platform/telemetry/collectors/`

## Persistence

- [ ] Async adapter interface | AC: "all adapters expose identical engine and session contracts" | Deps: `src/safety_platform/db/adapters/`, `src/safety_platform/db/base.py`
- [ ] SQLite WAL adapter | AC: "reference mode enables WAL, foreign keys, NORMAL sync, and one writer" | Deps: `src/safety_platform/db/adapters/`, `src/safety_platform/db/config/`
- [ ] Postgres and DuckDB adapters | AC: "DSN selection switches adapters without repository changes" | Deps: `src/safety_platform/db/adapters/`, `src/safety_platform/db/config/`
- [ ] Versioning and run models | AC: "pipeline, dataset, run, conversation, and turn foreign keys enforce integrity" | Deps: `src/safety_platform/db/models/`, `migrations/versions/`
- [ ] Decision and outcome models | AC: "stage verdicts, observations, outcomes, and transcripts satisfy uniqueness constraints" | Deps: `src/safety_platform/db/models/`, `migrations/versions/`
- [ ] Metric snapshot model | AC: "snapshots persist numerator, denominator, sample, completeness, versions, and split" | Deps: `src/safety_platform/db/models/`, `src/safety_platform/evaluation/`
- [ ] Repository boundaries | AC: "API and evaluation code access persistence only through repositories" | Deps: `src/safety_platform/db/repositories/`, `src/safety_platform/api/`, `src/safety_platform/evaluation/`
- [ ] Idempotent ingestion and migrations | AC: "duplicate event and batch identifiers create neither duplicate rows nor metric counts" | Deps: `src/safety_platform/db/`, `migrations/versions/`
- [ ] Guardrail write isolation | AC: "guardrail processes open no writable evaluation connection" | Deps: `src/safety_platform/guardrails/`, `src/safety_platform/db/`

## Evaluation

- [ ] ASR and FPR metrics | AC: "run-level ASR and benign-turn FPR use explicit eligible denominators" | Deps: `src/safety_platform/evaluation/metrics/`
- [ ] Latency and partial compliance | AC: "p50/p95/p99 overhead and partial compliance report sample counts" | Deps: `src/safety_platform/evaluation/metrics/`
- [ ] Completeness and exclusions | AC: "incomplete turns are excluded and surfaced with loss counts" | Deps: `src/safety_platform/evaluation/metrics/`, `src/safety_platform/telemetry/collectors/`
- [ ] Immutable metric snapshots | AC: "each report row resolves to one immutable snapshot with query parameters" | Deps: `src/safety_platform/evaluation/reports/`, `src/safety_platform/db/models/`
- [ ] Calibration search | AC: "threshold search enforces FPR and Tier-3 budgets on calibration data only" | Deps: `src/safety_platform/evaluation/calibration/`, `configs/evaluation/`
- [ ] Calibration promotion | AC: "promotion writes an immutable guardrail version and configuration hash" | Deps: `src/safety_platform/evaluation/calibration/`, `src/safety_platform/db/models/`
- [ ] Router training and policy | AC: "router features exclude identifiers, labels, family, split, and turn index" | Deps: `src/safety_platform/evaluation/routing/`, `configs/evaluation/`
- [ ] Benchmark harness adapters | AC: "CircleGuardBench, Tensor Trust, and domain datasets load through one harness interface" | Deps: `src/safety_platform/evaluation/harness/`, `datasets/`
- [ ] Version and split stratification | AC: "reports separate guardrail version, backend, model, and sealed-test scope" | Deps: `src/safety_platform/evaluation/reports/`, `src/safety_platform/evaluation/metrics/`

## Datasets

- [ ] Baseline corpus layout | AC: "attack and benign baselines resolve only through manifest entries" | Deps: `datasets/baselines/`, `datasets/manifests/`
- [ ] Dataset manifest schema | AC: "manifests validate source, license, hash, split, and label schema" | Deps: `datasets/schemas/`, `datasets/manifests/`
- [ ] Raw and processed lanes | AC: "raw, processed, and domain corpora remain physically separated" | Deps: `datasets/raw/`, `datasets/processed/`, `datasets/domain/`
- [ ] Dataset preparation scripts | AC: "ingest, deduplication, and split scripts produce reproducible manifest hashes" | Deps: `datasets/scripts/`, `datasets/schemas/`
- [ ] Sealed held-out split | AC: "sealed manifests are unavailable to calibration and router training jobs" | Deps: `datasets/manifests/`, `datasets/baselines/`, `configs/evaluation/`

## Quality and Delivery

- [ ] Unit test baseline | AC: "core, guardrail, and telemetry units meet the configured coverage gate" | Deps: `tests/unit/`, `pyproject.toml`
- [ ] Contract tests | AC: "API, event, verdict, and dataset schemas reject incompatible versions" | Deps: `tests/contract/`, `src/safety_platform/schemas/`
- [ ] Integration and e2e tests | AC: "one guarded run persists a complete reconstructable trajectory" | Deps: `tests/integration/`, `tests/e2e/`, `tests/fixtures/`
- [ ] Regression and performance tests | AC: "baseline ASR, FPR, and p95 latency remain within frozen thresholds" | Deps: `tests/regression/`, `tests/performance/`
- [ ] Offline test fixtures | AC: "model stubs, prompts, and traces run without external services" | Deps: `tests/fixtures/`, `tests/data/`
- [ ] CI quality gates | AC: "CI runs lint, typecheck, unit, contract, and integration suites on every change" | Deps: `.github/workflows/`, `Makefile`, `pyproject.toml`
- [ ] Container runtime | AC: "compose starts the app, local inference, and evaluation store with health checks" | Deps: `Dockerfile`, `docker-compose.yml`, `deploy/compose/`, `deploy/docker/`
- [ ] Systemd deployment unit | AC: "systemd restart preserves environment-file references and service health" | Deps: `deploy/systemd/`, `.env.example`
- [ ] Operational runbooks | AC: "runbooks cover startup, telemetry loss, migration rollback, and sealed-test unlock" | Deps: `docs/runbooks/`, `docs/architecture/`
- [ ] Documentation path reconciliation | AC: "`ARCHITECTURE.md` has one canonical path referenced by `docs/TREE.md`" | Deps: `docs/ARCHITECTURE.md`, `docs/TREE.md`
- [ ] Release documentation baseline | AC: "README, SECURITY, CHANGELOG, and CONTRIBUTING reflect current commands" | Deps: `README.md`, `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`
