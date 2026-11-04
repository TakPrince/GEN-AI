# Implementation Plan — V2 (post baseline audit, 2026-11-04)

Basis: `docs/BASELINE_AUDIT.md`. Read-only findings converted into ordered, verifiable tasks. Task IDs reference audit finding IDs. Sequencing follows the dependency graph: correctness fixes first, then the research questions, then new capability. Each task ≤5 files; every task ends with the repo's gates (`make test`, `make lint`).

**Recommended implementation sequence:** Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13.

---

## Phase A — Correctness (fix the broken loop before extending it)

### Task 1: Commit or discard the uncommitted working tree
**Description:** 13 dirty files (ModelDrivenRouter drift guard, speculative echo probe, CascadeRuntime probe param + their tests) currently sit uncommitted at HEAD `4910e13`. All gates pass on them. Land them so the V2 baseline is a clean commit.
- **Acceptance:** `git status` shows only intentional artifacts (`metrics/dashboard_demo.sqlite3` churn aside); work tree matches HEAD.
- **Verification:** `make test && make lint` green before commit; commit message covering drift-guard + echo probe.
- **Files:** the 13 already-modified files.
- **Deps:** none. **Size:** XS.

### Task 2 (F-MED-4): Untrack binary SQLite artifacts
**Description:** `metrics/dashboard_demo.sqlite3` and `evaluation/results/jbb-v2.sqlite3` are tracked despite .gitignore. Untrack them (keep pilot artifacts on disk; they are reproducible from the committed json/manifest).
- **Acceptance:** `git ls-files | grep sqlite` returns nothing; pilot re-verification instructions documented in Readme §9 remain valid via `make report`.
- **Verification:** `git ls-files`; `make report` still reproduces §9 numbers from the local file.
- **Files:** `.gitignore` (no change needed), index only.
- **Deps:** Task 1. **Size:** XS. ⚠ ask-first: removes files from tracking (keeps them on disk).

### Task 3 (F-HIGH-3): Wire real telemetry through the bus in the dashboard
**Description:** Replace the direct `sink.append_many` call in `app._evaluate` with `AsyncTelemetryBus.emit_nowait` + one `SqliteTelemetrySink.consume` task started at app startup. All tested components already exist; this makes the documented observation path real.
- **Acceptance:** dashboard turns persist via the bus; a saturation test drops events without blocking; existing dashboard tests still pass.
- **Verification:** `pytest tests/unit/test_app.py tests/unit/telemetry -q`; manual `make ui` + QA_PROMPTS row 11 (row count check).
- **Files:** `src/safety_platform/app.py`, `tests/unit/test_app.py`.
- **Deps:** Task 1 (else merge conflict). **Size:** S.

### Task 4 (F-CRIT-1): Stop calling the target model before the guard (dashboard)
**Description:** Rewrite `app._evaluate` to reuse the tested `CascadeRuntime.run(...)` for input-direction decisions. Request a completion only when the route is not REFUSED/ERROR; display the canned refusal (`guardrails` router already produces the route) in the guarded pane for blocked prompts. Keep column 1 as the intentionally-unshielded baseline (that is its purpose).
- **Acceptance:** blocked prompts generate zero target-model calls; guarded pane shows refusal text on block; telemetry events unchanged.
- **Verification:** new integration test with a counting stub client (0 calls on Tier-1 block, 1 call on pass); `pytest tests/integration tests/unit/test_app.py -q`.
- **Files:** `src/safety_platform/app.py`, `tests/unit/test_app.py`, `tests/integration/` (1 new file).
- **Deps:** Task 1, Task 3 (same file). **Size:** M.

### Task 5 (F-MED-2): Retry + typed recovery in Ollama HTTP calls
**Description:** Add one bounded retry (e.g. 1 retry with short backoff) to `OllamaClient._post` and `UrllibOllamaTransport._generate` for transient `RemoteDisconnected`/`URLError` during model load. Keep errors typed (`OllamaError`).
- **Acceptance:** a stubbed transport that fails once then succeeds returns the result; permanent failures still raise once, typed.
- **Verification:** unit tests with counting stub transports; `pytest tests/unit/evaluation/test_llm.py tests/unit/guardrails/tier3 -q`.
- **Files:** `src/safety_platform/evaluation/llm.py`, `src/safety_platform/guardrails/tier3/guard_classifier.py`, 2 test files.
- **Deps:** none (parallelizable with 3/4). **Size:** S.

### Checkpoint A
- [ ] `make test && make lint` green; dashboard makes no unprotected model calls; telemetry path real.

## Phase B — The open research question (calibration before any rerun)

### Task 6 (F-HIGH-4): Harden the compliance judge against prompt injection
**Description:** Move `_JUDGE_INSTRUCTION` to the `system` field; delimit user prompt and assistant response in separate labeled blocks; reject/strip embedded JSON blocks inside the quoted content before handing to the model (fail to `malformed_judge_verdict`).
- **Acceptance:** a fixture response containing `{"complied": true}` inside a refusal is not scored compliant; existing judge tests pass.
- **Verification:** `pytest tests/unit/evaluation/test_llm.py tests/unit/evaluation/test_runner.py -q`; adversarial fixtures from the 30 stored baseline responses.
- **Files:** `src/safety_platform/evaluation/llm.py`, `tests/unit/evaluation/test_llm.py`.
- **Deps:** Task 5 (same file). **Size:** S.

### Task 7 (F-HIGH-2): Single source of truth for router weights + config
**Description:** Extract the duplicated `LogisticRegressionRouter(weights=(1.5,1.0,-0.3,3.0,0.2), bias=-2.5, threshold=0.50)` + `DEFAULT_CASCADE_THRESHOLDS`/margins into one module-level definition (and document their provenance/plan), then dedupe `_json_object` (F-MED-5) and make `_SPECULATIVE_VIOLATING` public (F-MED-6) while touching these files.
- **Acceptance:** one definition; dashboard + cascade import it; no behavior change.
- **Verification:** `pytest -q`, `make lint`.
- **Files:** `src/safety_platform/evaluation/cascade.py`, `src/safety_platform/app.py`, `src/safety_platform/research/ml_router.py`, `src/safety_platform/evaluation/llm.py`, `src/safety_platform/guardrails/tier3/guard_classifier.py`.
- **Deps:** Task 1. **Size:** M.

### Task 8 (F-HIGH-5): Calibrate Tier-3 output arbitration on stored responses, then decide thresholds
**Description:** Run the calibration the CHECKPOINT prescribes: use the 30 attack + 30 benign stored responses from run `20261029T015710-587cb28a`, score them with `OllamaLlamaGuardConnector` (output direction, qwen3:8b + arbitration prompt), compute recall/FPR per threshold (reuse `research/calibration.CalibrationSweep`心智 rather than a new script — feed `CalibrationSample`s from tier2-heldout.json + the new Tier-3 scores). Then decide: keep 0.80/0.45/0.78 or adopt calibrated values. Report honestly; negative result acceptable.
- **Acceptance:** calibration artifact JSON committed (`evaluation/calibration/tier3-arbitration.json`); decision recorded in Readme §9; NO full 30/30 rerun unless separation improves.
- **Verification:** sweep script run recorded in the artifact; `make lint`.
- **Files:** new small calibration entry module (or extend `evaluation/__main__.py` with a `calibrate` subcommand), `evaluation/calibration/` artifact, `Readme.md`.
- **Deps:** Task 5 (retry), Task 7 (dedup) optional. **Size:** M.

### Checkpoint B
- [ ] Judge hardened; calibration decision recorded; weights single-sourced.

## Phase C — Missing capability (the offense half)

### Task 9 (F-HIGH-1): Minimal red-team orchestrator (single-turn)
**Description:** New `src/safety_platform/orchestrator/` package: a loop that takes an `EvaluationCase`-like attack set and reusable mutators (role-play wrap, encoding, payload-split — pure string transforms first), and runs each mutated prompt through the existing `CascadeRuntime`. Record `{case_id, mutator, verdict, route}` rows. No state, no API, no DB schema beyond JSONL output.
- **Acceptance:** `python -m safety_platform.orchestrator --dataset evaluation/data/jbb-v2/cases.jsonl --attack-per-label 5` produces a JSONL ASR summary per mutator; unit tests for mutator functions and loop wiring.
- **Verification:** `pytest tests/unit/orchestrator -q`; smoke run against live Ollama with 2 cases.
- **Files:** new `orchestrator/{__init__,mutators,loop}.py`, `tests/unit/orchestrator/*`.
- **Deps:** Tasks 1-8 (uses cascade + hardened judge). **Size:** M.

### Task 10: Multi-turn state + partial-compliance metric
**Description:** Add `AttackState` (turn history, prior verdicts) to the orchestrator and a crescendo/escalation mutator that reads prior-blocked verdicts (spec §3 component 9, feedback). Add partial-compliance counting across turns.
- **Acceptance:** a scripted multi-turn fixture (stub target) shows escalation reacts to blocked-turn verdicts; partial metric computed in summary.
- **Verification:** `pytest tests/unit/orchestrator -q`.
- **Files:** `src/safety_platform/orchestrator/{loop,attack_state}.py`, test files.
- **Deps:** Task 9. **Size:** M.

### Task 11 (F-MED-3): Makefile hygiene
**Description:** Point `make eval-smoke` at a disposable DB (`evaluation/results/smoke.sqlite3`); align `pytest.ini` vs `pyproject` F-MED-7 (keep pytest.ini: it is what CI uses); add a roadmap refresh of `tasks/todo.md` (or replace content) to reflect the audit plan rather than the abandoned ASGI roadmap.
- **Acceptance:** running `make eval-smoke` cannot pollute the pilot DB's "latest completed run"; todo matches this plan.
- **Verification:** grep Makefile; run `make eval-smoke` with Ollama optional (skips gracefully) then `make report` still shows the pilot.
- **Files:** `Makefile`, `pytest.ini`, `tasks/todo.md`.
- **Deps:** Task 2. **Size:** XS.

### Checkpoint C
- [ ] Orchestrator runs single-turn and multi-turn attacks end-to-end against Ollama; ASR summary per mutator.

## Phase D — Deferred (explicitly not planned for V2 unless a consumer appears)

| Deferred | Why |
|---|---|
| FastAPI `/v1` collections + repositories + migrations + OpenAPI | no API consumer; Streamlit + CLI cover the loop today |
| `inference/` multi-backend package | only Ollama used; vLLM not promised |
| `configs/guardrails` YAML versions | Task 7 covers single-sourcing until calibration produces a *versioned* artifact |
| CircleGuardBench / Tensor Trust loaders | blocked on orchestrator tasks 9-10 making use meaningful |
| Postgres/DuckDB + Kafka transports | single-node is sufficient for the current experiment scale |
| Canary-token / PII output scanning tier-1 rules in eval cascade | cheap to add later; keep the audit item open |

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Full eval rerun consumes ~1 h + model churn on this host | High | Tasks 1-8 avoid reruns; calibration reuses stored responses; only rerun if separation improves (Checkpoint B gate) |
| Calibrated thresholds worsen held-out results | Medium | Honest reporting; negative result is a valid finding (already the pilot's precedent) |
| Orchestrator work expands into "the whole V2 tree" | High | Phases D are explicitly out of scope; orchestrator stays on JSONL + existing cascade |
| Trailing-space directory confuses scripts/agents | Low | Repeated warning in docs; prompt before any `git add -A` |

## Open questions for the user (before Task 8 runs)
1. Adopt calibrated thresholds if they only *tie* on held-out separation but halve latency? (equivalence-margin decision belongs to you)
2. Untrack the binary DBs (Task 2) — yes/no?
3. Is a minimal `calibrate` CLI subcommand acceptable, or do you want a standalone `scripts/calibrate_tier3.py`?
