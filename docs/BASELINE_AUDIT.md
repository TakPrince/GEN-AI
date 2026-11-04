# GEN-AI Baseline Audit — Pre-V2

**Date:** 2026-11-04
**Scope:** Read-only audit of the current repository before V2 implementation. No source was modified. All test, lint, and live-model results below were executed during the audit.
**Repo note:** the working directory is `/home/asus/Desktop/GEN-AI ` — the trailing space is significant (also flagged in `CHECKPOINT.md`).

---

## 1. Executive Summary

The repository is a **working two-half prototype** of the red-team/guardrail loop:

- The **defense half is real and verified**: three-tier cascade (regex → cosine centroid → conditional Ollama classifier), fail-closed response router, calibration/ML-router research modules, a 60-case JBB v2 pilot with a honestly-reported negative finding, and a live-verified output-arbitration Tier 3. All 214 tests pass, coverage is 93% (gate 80%), black/ruff/mypy are clean.
- The **offense half does not exist**: no red-team orchestrator, no prompt mutators, no multi-turn state. The loop is never closed — the orchestrator box in every architecture diagram is empty.
- The **planned V2 platform tree is ~5% real**: `docs/PROJECT_STRUCTURE.md` describes an API/middleware/db/orchestrator/inference platform of ~60 modules; ~20 modules exist. The doc is a target, not a status — but nothing in the repo marks it as such.
- The committed pilot (`20261029T015710-587cb28a`) reproduces exactly via the CLI: ASR 6.7%, FPR 10.0%, guard p95 ≈ 21 s. Dataset digest matches the manifest.

**Biggest risk going into V2:** building against `docs/PROJECT_STRUCTURE.md` as if it were the implemented baseline. It is not. V2 should build on the ~20 modules that exist and deliberately ignore most of the aspirational tree (see §5).

---

## 2. What Was Inspected

| Area | Files / artifacts | Evidence |
|---|---|---|
| Docs | `Readme.md` (spec + status), `docs/ARCHITECTURE.md`, `docs/PROJECT_STRUCTURE.md`, `docs/TREE.md`, `docs/RESEARCH_INNOVATION.md`, `docs/INNOVATIONS.md`, `docs/QA_PROMTTS.md` (QA_PROMPTS.md), `CONSTRAINTS.md`, `CHECKPOINT.md`, `tests/STRATEGY.md`, `status/TEMPLATE.md`, `tasks/todo.md` | read in full or in part |
| Source | all 31 modules under `src/safety_platform/` (~5,060 LOC incl. `metrics/db_adapter.py`) | read in full |
| Tests | 44 test files: unit (194), integration (3), performance (4); `pytest.ini`, `pyproject.toml` | executed |
| Evaluation | `evaluation/data/jbb-v2/{cases.jsonl,manifest.json}`, `evaluation/results/jbb-v2.{sqlite3,json}` + 4 smoke run DBs, `evaluation/calibration/*.json` (3) | inspected, digest-verified, CLI re-executed |
| Telemetry | `src/safety_platform/telemetry/{bus,consumer,db_sink,events}.py`, `metrics/telemetry.sqlite3`, `metrics/dashboard_demo.sqlite3` | read; DBs inspected |
| Model integration | Ollama 127.0.0.1:11434 — 5 models (`llama3.2:latest`, `qwen3:8b`, `qwen3:4b`, `llama3.1:8b`, `llama3:latest`) | live calls executed |
| Git | last commit `4910e13`; 13 dirty files (uncommitted speculative-echo + drift-guard work, +557/−14) | `git log`, `git diff` |

Uncommitted working-tree state (audited as-is, all gates pass on it): `ModelDrivenRouter` drift guard, speculative echo probe wiring in the dashboard and `CascadeRuntime`, plus their tests.

---

## 3. Actual Architecture (traced from code, not docs)

**Verified live path — evaluation/CLI (`CascadeRuntime`, src/safety_platform/evaluation/cascade.py:57):**

```
case.prompt ──▶ [optional SpeculativeEchoProbe: Ollama num_predict=10, fail-closed early block on "violating" prefix]
  ▶ Tier 1 RegexFilter.evaluate (block rules first, then redact spans, max 16 KB)
  ▶ Tier 2 VectorFilter.evaluate (cosine vs attack-prompt centroids from the 10-case train split; ≥0.80+gap→block, ≥0.45→ambiguous)
  ▶ ML router: LogisticRegressionRouter(hardcoded weights (1.5,1.0,-0.3,3.0,0.2), bias −2.5) wrapped in Tier3BypassRouter, then ModelDrivenRouter drift guard (≥0.60 forces escalation)
  ▶ Tier 3 GuardClassifier — ONLY if Tier 2 == AMBIGUOUS and router says invoke
      · input direction: "Classify the user message as safe or unsafe" → safe/unsafe token
      · output direction: JSON arbitration prompt {violation, reason} on User Input + Model Response
  ▶ ResponseRouter.route (fail-closed: error > block > unresolved-ambiguity→error > redact > pass)
  ▶ rows written to run_result (baseline/input/full × 3 systems per case)
```

**Verified live path — dashboard (`app._evaluate`, src/safety_platform/app.py:261):**

```
prompt ──▶ Ollama completion FIRST (asyncio.to_thread, lru_cache 64)        ← before the guard (§6 F-CRIT-1)
  ▶ [optional echo probe] ▶ Tier 1 middleware ▶ Tier 2 middleware
  ▶ ModelDrivenRouter ▶ Tier 3 (hash-embedder + keyword _LocalClassifier stub persist to metrics/dashboard_demo.sqlite3 (direct sink call; AsyncTelemetryBus + TelemetryConsumer exist but are NOT on this path)
```

**Telemetry:** `AsyncTelemetryBus` (bounded 4096, drop-newest), `TelemetryConsumer` (batch 256/250 ms, WAL SQLite sink, idempotent `INSERT OR IGNORE`, stats + error callback) — implemented and unit-tested, but only exercised in tests. Neither the dashboard nor the evaluation runner uses the bus.

**Ollama clients:** synchronous `urllib` wrappers (`OllamaClient`, `UrllibOllamaTransport`) — no retry, no streaming, no connection reuse.

---

## 4. Implemented vs Planned

### Implemented and verified working
- Tier 1 regex filter (block/redact spans, size limit, ReDoS-safe simple patterns) ✓ tested + perf-gated p95 <60 ms
- Tier 2 vector filter (unit-vector cosines, ambiguity band, top-gap margin) ✓ tested + perf-gated p95 <20 ms
- Tier 3: input classifier prompt, output JSON arbitration prompt (live-verified: unsafe exploit response → 1.0; benign SSH answer → 0.0), fail-closed malformed-JSON handling ✓
- Response router fail-closed aggregation ✓ tested
- ComplianceJudge (ASR judge), OllamaEmbedder with persistent SHA-256-keyed SQLite cache, `--reuse-run` replay ✓ unit + integration tested (reuse smoke verified by tests; CHECKPOINT's "verify end-to-end" caveat addressed by tests/integration/test_reuse_run_pipeline.py)
- LogisticRegressionRouter + fit path, ModelDrivenRouter drift guard, CalibrationSweep ✓ tested
- JBB v2 frozen dataset (80 cases, digest verified: `8bde5da5…`) and 4 completed smoke runs + the 60-case pilot ✓
- Report generation: ASR/FPR with Wilson CIs, latency p50/p95/p99, tier-3 rate, per-category ASR ✓ reproduced exactly during audit
- Streamlit split-pane dashboard with live metrics ribbon ✓
- Quality gates per CONSTRAINTS.md: 214 tests, 93% coverage, black/ruff/mypy clean, 4 perf tests pass ✓

### Not implemented (planned but absent)
- Red-team orchestrator, `AttackState`, prompt mutators (role-play/encoding/payload-split/crescendo), multi-turn + partial-compliance metrics — Readme §9 "Planned"; zero code.
- FastAPI `/v1/chat` + collections API, middleware registry, db/models + repositories + migrations, `inference/` package, `configs/` YAML guardrail versions, `datasets/` manifests tree, `core/` primitives, Kafka transport, Postgres/DuckDB backends.
- CircleGuardBench / Tensor Trust loaders — specified in `tests/STRATEGY.md`, absent.
- Per-tier FPR attribution; canary/deny-list/PII **output** scanning (Tier 1 redacts emails/`secret=` in whatever direction, but no canary-token echo detection exists despite the spec naming it).
- Trained ML-router artifact (weights are hardcoded in two places, not loaded from calibration output).

### Documentation vs reality gaps (LOW/visibility)
- `docs/PROJECT_STRUCTURE.md` and `docs/TREE.md` read as present tense but are a placement contract only; Readme §9 is the only accurate status map.
- `tasks/todo.md` (ASGI/API/db roadmap) has unchecked items for work that was never started; it conflicts with the actual V2 needs and should be rewritten, not extended.
- `AGEN.md` is an empty placeholder file at the repo root.

---

## 5. Technical Issues, Security, Tests, Performance

Priorities: **CRITICAL** = core-functionality failure; **HIGH** = missing essential functionality; **MEDIUM** = reliability/testability; **LOW** = docs/optional.

### CRITICAL

**F-CRIT-1 — Dashboard calls the target model before the input guard.**
- Files: src/safety_platform/app.py:271 (completion), :276 (Tier 1), :577-583 (turn handling).
- Behaviour: `_evaluate` awaits `_live_completion(prompt)` (lru_cache, qwen3:8b, `num_predict=16`) **before** Tier 1/2/3 run. A blocked prompt has already been executed by the LLM. The "guarded" pane shows the raw model output regardless of route; the echo probe then generates a *second* completion (app.py:273-274).
- Impact: the demo does not demonstrate the defense it displays; a specifically crafted jailbreak is executed with zero protection. Violates the spec invariant "input guardrails stop the request before the model is ever called."
- Fix (smallest): run Tier 1 → Tier 2 → routing decision first; only request a completion when route ≠ REFUSED. Reuse `CascadeRuntime` (already implemented and tested) instead of the hand-rolled `_evaluate` — it is the same logic, fixed.
- Tests: integration test asserting a stub client raises 0 calls on a Tier-1-blocked prompt; existing tests/integration/test_cascade_runtime.py covers the runtime.
- Dependencies: none blocking.

### HIGH

**F-HIGH-1 — Red-team half of the loop does not exist.**
- Readme §3 component 1/2/9, `docs/ARCHITECTURE.md` decision path. Side effect: all evaluation is single-shot static datasets; no adaptive attacks, so ASR is a floor, not a defense test.
- Fix: minimal `orchestrator/` (turn loop over N mutators reusing `EvaluationCase` + `CascadeRuntime`); do not build `db/`/`api/` around it yet.

**F-HIGH-2 — ML router weights are hardcoded, never trained.**
- src/safety_platform/evaluation/cascade.py:173-177 and src/safety_platform/app.py:215-219 duplicate `LogisticRegressionRouter(weights=(1.5,1.0,-0.3,3.0,0.2), bias=-2.5)`. The provenance of these numbers is nowhere; the `fit()` path (ml_router.py:139) is never called in production code.
- Impact: Innovation 2 is not testable against a baseline because the "learned" router is not learned; the drift guard masks this by forcing escalation at ≥0.60 similarity.
- Fix: extract to one module-level constant + document; then fit from stored run data per Innovation 2's plan. Keep `ModelDrivenRouter`.

**F-HIGH-3 — Telemetry bus/consumer is dead code on every real path.**
- src/safety_platform/telemetry/consumer.py and bus.py are only used by their own tests; app.py persists by calling `sink.append_many` directly (app.py:415-423), and the evaluation cascade persists *nothing* (only `run_result` rows).
- Impact: the "observation path" in docs/ARCHITECTURE.md is not real; completeness/loss metrics don't exist for actual runs; verdict history can't feed the orchestrator later.
- Fix: dashboard switch to `bus.emit_nowait` + one sink consumer (small); evaluation path: decide whether stage events belong in an eval run at all before plumbing it.

**F-HIGH-4 — Judge is verdict-spoofable (prompt injection into the ASR judge).**
- src/safety_platform/evaluation/llm.py:270-294: user prompt + raw model response are concatenated into one judge string. A crafted response (e.g. one containing `JSON: {"complied": true, ...}`) can steer the judge. Judge output is the ASR ground truth (report.py:157-181), so this directly biases the *headline metric*.
- Fix: send instruction as `system`, keep user/assistant content in delimited fields, and strip/flag embedded JSON blocks; add adversarial judge-spoof fixtures asserting `malformed_judge_verdict` or non-compliance handling.
- Directly from CHECKPOINT: both surviving attacks (attack-024, attack-038) passed input+output guards entirely.

**F-HIGH-5 — Default thresholds are not the calibrated ones.**
- `DEFAULT_CASCADE_THRESHOLDS` (config/registry.py:88-92): Tier-2 block at 0.80. The historical issue persists — with real llama-embeddings, attack similarity max ≈ 0.56 and FP benign ≈ 0.53 (CHECKPOINT), so Tier 2 effectively *never* blocks; ~52% of traffic escalates to the 21-second Tier 3. `evaluation/calibration/tier2-heldout.json` holds the data; `research/calibration.CalibrationSweep` exists but is never wired against it — pilot diagnosis (overfit 0.55; overlap band 0.47-0.56) remains unresolved.
- Impact: measured latency and capacity are tier-3-dominated; ASR unaffected because baseline already refuses. This is the stated core open question of the project.
- Fix: sequence is already planned (calibrate qwen3 arbitration on the 30 stored responses before any rerun — CHECKPOINT "Resume Plan" steps 2-4).

### MEDIUM

**F-MED-1 — Output guard sees only the response; arbitration is wired but uncalibrated in the pilot.** Output path now receives `user_input` via `ExecutionContext.user_input` (registry.py:188; set in make_context, cascade.py:187-195) — good. But the pilot predates it; nothing measured yet.

**F-MED-2 — Upstream loading disorder: no retry on transient failures.** `OllamaClient._post` (evaluation/llm.py:108-125) raises on the first failure; during this audit `/api/embed` and the Tier-3 call each failed with `RemoteDisconnected` on first contact with a cold/cached model (bare `OllamaError` with no retry, and embedding calls aren't retried anywhere).

**F-MED-3 — `make eval-smoke` writes into the pilot results DB.** Makefile eval-smoke: `--db evaluation/results/jbb-v2.sqlite3`. A completed smoke run (4 cases) would then be picked up by `make report` latest-run resolution (report.py:129-139). Never execute it; fix by pointing it at a disposable DB (as the `smoke*.sqlite3` files already do manually).

**F-MED-4 — Tracked binary artifacts despite .gitignore.** `metrics/dashboard_demo.sqlite3` (currently dirty in `git status`) and `evaluation/results/jbb-v2.sqlite3` are committed; .gitignore lists them but does not untrack. The demo DB will churn on every dashboard session. Fix: `git rm --cached` when committing the V2 baseline (requires user approval per constraints — record here only).

**F-MED-5 — Duplicated JSON parser.** `_json_object` exists identically at evaluation/llm.py:303 and tier3/guard_classifier.py:268. Consolidate to one home (guard_classifier imports from llm or both from a shared util) when touching Tier 3.

**F-MED-6 — Private cross-module import.** evaluation/cascade.py:32 imports `_SPECULATIVE_VIOLATING` (underscore) from evaluation/llm.py. Rename to a public constant when touching it.

**F-MED-7 — Config duplication between pytest.ini and pyproject.toml.** `pytest.ini` sets `testpaths = tests`, `pythonpath = src`; pyproject `[tool.pytest.ini_options]` also sets `testpaths`. Harmless today, confusing tomorrow; keep one.

**F-MED-8 — Coordinator stub in the dashboard misrepresents Tier 2/3 components.** src/safety_platform/app.py:135-158: `_HashingEmbedder` (hashing-trick, 96-dim) + `_LocalClassifier` (phrase matching). The dashboard caption now says so, but `OllamaClient.embed` and the real `OllamaLlamaGuardConnector` already exist — the stub only exists to keep the demo fast/offline. Set an env flag to choose, or bundle the calibrated components as default with a cache; not urgent while the banner stays honest.

### Security review (beyond the findings above)
- Fail-closed posture is genuinely implemented at every boundary: middleware wraps detector exceptions as ERROR verdicts; the router maps unreconciled AMBIGUOUS to an ERROR route (`unresolved_ambiguity`); Tier-3 malformed JSON raises; `Tier3Result` validates non-negative-latency, unknown staged error semantics. Good.
- Telemetry stores *hashes*, not prompts; redaction spans only. No raw user content leaks into DB sinks. `st.markdown(unsafe_allow_html=True)` only receives a fixed stylesheet constant, never user content — not exploitable.
- RegexFilter rejects patterns matching empty strings and enforces UTF-8 size limits (4 KB UI / 16 KB eval); patterns used are linear-time — no ReDoS.
- `metrics/db_adapter.load_metrics` opens SQLite read-only (`mode=ro`, `query_only`), and the whole computation is wrapped (`except` → default metrics). Good defensive read.
- Weaknesses: no authentication on the (future) API; dashboard is a single-user, localhost-only Streamlit (acceptable today but none of the invariants from PROJECT_STRUCTURE.md §9 exist — no rate/body limit code, no request context middleware). The `evaluation/results/*sqlite3` holds full harmful prompts and responses in plaintext (fine for research; flag before any publication of artifacts).

### Tests and evaluation findings
- Full suite: `pytest -q` → **214 passed in 67.7 s**; unit-only with cov `194 passed`, **93% line coverage** (gate 80%). Weakest coverage: `evaluation/cascade.py` 49% (covered via smoke/CLI paths instead), `evaluation/__main__.py` 75%.
- Performance gates: all 4 pass in 1.76 s (tier1 p95, tier2 p95, envelope p95 ≤400 ms, 10k-loop embedding RSS ≤512 MiB).
- Lint/type: `black --check` clean, `ruff check` clean, `mypy` clean (59 files).
- Actual command evidence:
  - `PYTHONPATH=src:. python -m safety_platform.evaluation report --db evaluation/results/jbb-v2.sqlite3` → reproduced baseline ASR 6.7% / FPR 10.0% / p95 21005.92 ms / Tier-3 53.3% full & 51.7% input — identical to `Readme.md` §9 and `jbb-v2.json`.
  - `sha256sum evaluation/data/jbb-v2/cases.jsonl` → `8bde5da5…5b5a`, equals manifest digest. `wc -l` → 80 cases (10+30 train/test per label).
  - Live: target `llama3.2:latest` generate OK ("18" for 9+9), `embed` OK (3072 dims, second attempt), `qwen3:8b` generate OK, output-arbitration score OK (1.0 unsafe / 0.0 benign).
  - Smoke DBs: all 4 runs completed with 12 result rows each. Pilot DB has exactly one completed run (`20261029T015710-587cb28a`), 60 rows/system, blocked sums 29/29/0 — matches the documented figures.
- Missing tests: none of the 214 test Θ-attaches `_live_completion`'s pre-guard ordering (F-CRIT-1), no judge-spoof fixtures (F-HIGH-4), no retry-required network failure tests (they pass by stubbing success instead), no test loads pilot DB to assert ASR invariants against the report (regression guard).

### Performance concerns
- Guard p95 ≈ 21 s, p99 ≈ 28.7 s (as committed). Breakdown per model: embeddings ~16 min, judge ~13 min, guard ~6 min of a 56-min run. qwen3:8b on this hardware is the dominant cost, by design (52-53% escalation rate) and by threshold looseness (F-HIGH-5). `num_predict=128` and `timeout_s=120` per Tier-3 call bound the worst case but not the typical case.
- Cold-start `RemoteDisconnected` (first embed after model unload) is a real failure mode on a laptop; retry/fallback (F-MED-2) or keep-alive warmup is required before any unattended rerun.
- `OllamaEmbedder` keeps the full cache in memory forever (fine: 80 cases + 3072-dim floats ≈ MB scale; the 10k-loop soak test proves bounded RSS).

---

## 6. Architecture Review — Current vs Spec vs V2

| Element | `docs/ARCHITECTURE.md` (proposed V2) | `Readme.md` §9 | Reality (audited) | Verdict for V2 |
|---|---|---|---|---|
| Three-tier cascade (both directions, fail-closed) | specified | "implemented, routed by evaluation" | **real** | keep as-is; calibrate Tier 2/3 |
| Async telemetry bus/consumer/WAL sink | lanes/bounds/backpressure | "implemented (2)" | **exists, not wired** | wire into dashboard; skip Kafka/lanes |
| Red-team orchestrator + mutators | core of the loop | "planned" | **absent** | build minimal |
| Multi-turn + partial-compliance eval | ordered | "planned" | absent | depends on orchestrator |
| FastAPI /v1 + collections + db repositories + migrations | large surface | "planned" | absent | **defer / cut** — the Streamlit demo + CLI cover every current use case; nothing consumes a REST API today |
| `inference/` package w/ multi-backend | adapter-only | Ollama only, vLLM planned | thin 2-client module | don't build; current `OllamaClient` suffices until vLLM is actually needed |
| Versioned YAML guardrail configs + DB versions | immutability rule | partly (guardrail_version constant) | hardcoded thresholds in two Python sites | fold into F-HIGH-2/5 as a single constants-or-JSON artifact; no full configs/ layer yet |
| Calibration/routing research modules | "separable" | Innovation 1/2 | **exist, unused in prod path** | wire (that is the research goal) |

**Unnecessary complexity already present (keep, don't expand):** `ModelDrivenRouter` wrapping `Tier3BypassRouter` wrapping `LogisticRegressionRouter` — three objects where one trained table + one drift-floor constant would do; keep because both halves are tested and no new abstraction costs anything to change later.

**Recommended smallest V2 scope:** (1) fix F-CRIT-1 by reusing `CascadeRuntime` in the dashboard; (2) calibrate Tier-3 output arbitration on the 30 stored responses (uses existing code paths); (3) land the uncommitted drift-guard + echo work as-is; (4) minimal orchestrator/mutators reusing dataset + cascade APIs. Everything else in PROJECT_STRUCTURE.md is deferred until a REST consumer actually exists.
