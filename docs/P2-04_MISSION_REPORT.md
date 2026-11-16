# Phase 2 — P2-04: CLI Migration and Shared Runtime Convergence — Final Report

Branch: `phase1-reliability-hardening` · Slices: P2-04.1 audit, P2-04.2 runtime
parity, P2-04.3 shared construction, P2-04.4 CLI, P2-04.5 parity tests.
Commits: `ee76f37` (escalation/output-Tier3 unification), `90f7610`
(runtime.py extraction), `7ab6ef1` (CLI; amended `d7dd859`).

## A) Architecture audit (P2-04.1)

| Component | Dashboard (`app.py`) | Evaluation (`runner.py`/`build_cascade`) | Final state |
|---|---|---|---|
| Runtime construction | `_build_pipeline` (private) | `build_cascade` | **shared** `runtime.py`; evaluation keeps dataset-trained centroids by design |
| Tier 1 limit | `DEFAULT_PAYLOAD_LIMITS.prompt_bytes` | hardcoded `16384` | registry-sourced in both |
| Tier 2 thresholds/margins | `DEFAULT_CASCADE_THRESHOLDS`, `RoutingMargins(0.01, 0.55)` | same values | identical (kept) |
| Tier 2 backend | hashing-trick demo embedder (accepted Task 20) | semantic Ollama embeddings | kept separate — different, documented purpose |
| Tier 3 gates | `MULTI_SIGNAL_GATES`, `qwen3:8b` | same | identical |
| Escalation | `escalation_for` (P2-03) | **old rule** (`invoke_tier3 or AMBIGUOUS or always_check`) | **unified** on `escalation_for` |
| Output Tier 3 | always-on (`tier3_on_output=True`) | **False**, contradicting runner comment | **always-on** in `build_cascade` |
| Router / retry / admission / breaker / echo / canary | shared | shared | kept |
| Telemetry | full sink/bus/consumer | none (results only) | interface concern, kept per-interface |

Existing CLI check (mission step 3): `evaluation/__main__.py`,
`orchestrator/__main__.py`, `api/__main__.py` exist; none executes the defense
path, so a top-level `python -m safety_platform` was added rather than a second
campaign/evaluation CLI.

## B) Changes

1. **CascadeRuntime.run** now derives escalation from `escalation_for`; the
   always-on output check (`tier3_on_output`, OUTPUT direction only) remains an
   additional ESCALATE condition. Tier 2 BLOCK and ERROR no longer buy a Tier 3
   call in evaluation, matching `execute_defense`.
2. **build_cascade** sets `tier3_on_output=True` (closing the parity hole:
   evaluation responses were vetted less strictly than live responses despite
   the runner comment claiming always-on) and sources the Tier 1 limit from
   `DEFAULT_PAYLOAD_LIMITS` instead of a hardcoded 16384.
3. **New `src/safety_platform/runtime.py`**: `DemoHashingEmbedder`, demo
   centroids, `tier3_guard`, `canary_detector`, `speculative_echo_probe`,
   `DemoComponents`, `DefensePipeline`, `build_runtime(database)`. app.py
   delegates (`_build_pipeline = build_runtime`) and keeps its test-facing
   aliases; it now owns only Streamlit UI and labels.
4. **New CLI** (`cli.py` + `__main__.py`): `check PROMPT [--json]`,
   `interactive`. Exit 0 allowed / 1 blocked / 2 error. Interactive oversized
   lines are refused at the CLI boundary with the Tier 1 payload limit.
   Telemetry persists through the same `SqliteTelemetrySink` wiring.
5. Parity test pins CLI and dashboard to one `DefensePipeline` construction.

## C) Tests added

- `tests/unit/evaluation/test_cascade_parity.py` (6): escalation unification
  (BLOCK/ERROR skip Tier 3, ambiguity escalates, output always-check) +
  construction parity (payload limit, always-on output Tier 3).
- `tests/unit/cli/test_check_cli.py` (6): allowed → exit 0, blocked → exit 1,
  JSON fields, interactive multi-line, `exit` stop, CLI↔dashboard parity.
- One real regression caught by the existing suite during extraction: my
  rewritten `_RISK_PHRASES` tuple had dropped "ransomware" (heuristic Tier 3
  scored unsafe output 0.05). Restored the exact original tuple — lesson:
  move verbatim, never retype.

## D) Verification

- Unit: **702 passed, 3 xfailed**, coverage **93.00%** (baseline 690/93.12%).
- Integration: 18 passed, 3 skipped. Performance: 4 passed.
- ruff, black, mypy: clean (56 source files).
- Smoke (real Ollama): `check` benign → ALLOW with qwen3:8b response;
  attack → BLOCK at tier1 in 4.2 ms, exit 1; `--json` emits machine fields;
  tier3 label `ollama:qwen3:8b` end-to-end.

## E) Deviations / notes

- Dashboard kept as a thin wrapper per the mission's allowed alternative to
  deletion: construction is fully delegated; 33 test files keep importing
  `app.*` aliases unchanged. Full deletion was judged higher-churn /
  lower-value than the wrapper (UI remains a maintained interface).
- Env var names (`SAFETY_UI_TIER3`, `SAFETY_UI_CANARY`, …) are dashboard-era
  names now shared by the CLI; not renamed to avoid docs/test churn.
- One smoke observation: the qwen3:8b target occasionally returns an empty
  string for a prompt (observed once); defense output contract is unaffected —
  flagged as a demo-target quirk, not a CLI defect.

## F) Status: **COMPLETE** (per the thin-wrapper option for dashboard removal)
