# GEN Issue Status (final)

Baseline: `635 passed, 3 skipped, 3 xfailed`; `black`/`ruff` clean; `mypy src`
clean (52 files). `mypy src metrics tests` retains 14 pre-existing GEN-03 errors
in `tests/unit/execution/test_output_security.py`.

| ID | Status | Evidence / remaining limit |
| --- | --- | --- |
| GEN-01 | COMPLETE | CPU-first path restored; GPU path removed/rolled back. |
| GEN-02 | PARTIAL | Per-layer replay + JBB held-out (P .897 / R .867 / F1 .881). `attack-037` remains a known FN (similarity .392, gap .009); no safe threshold/policy fix found. |
| GEN-03 | COMPLETE | Output security boundary and canary/leakage tests. |
| GEN-04 | COMPLETE | Telemetry and evaluation reproducibility. |
| GEN-05 | COMPLETE | Dataset/manifest integrity. |
| GEN-06 | COMPLETE | Evaluation harness/reporting. |
| GEN-07 | PARTIAL | Router audit: pass-band dead zone; no policy safely improves FPR/TPR. Behavior preserved; regression test added. |
| GEN-08 | COMPLETE | Dashboard/session isolation. |
| GEN-09 | COMPLETE | Indirect-injection rules; 0/250 XSTest false positives. |
| GEN-10 | PARTIAL | Judge repeatable (κ 1.0) and 10/11 human cases; one scaffold-benign FP remains. Structured-verdict experiment failed (0 TP / 5 FN) and was not shipped. |
| GEN-11 | COMPLETE | Deterministic, seed-controlled campaign artifacts. |
| GEN-12 | COMPLETE | Authenticated FastAPI service: `POST /v1/chat`, `/healthz`, `/readyz`; API-key auth, 16 KiB prompt / 32 KiB body limits, concurrency/queue/execution timeouts, stable errors, fail-closed target failures, security headers. Tests: `tests/unit/api/` (12). Docs: `docs/GEN-12_API.md`. Single-node only; proxy rate limiting/TLS are deployment concerns. |
| GEN-13 | COMPLETE | Manifests and provenance. |
| GEN-14 | COMPLETE | Drain fast path; target semaphore/queue timeout. |
| GEN-15 | PARTIAL | Frozen Tier-3 candidate: dev P 1.0 / R .933 / F1 .966; independent XSTest P .781 / R 1.0 / F1 .877 (7 homonym FPs, 0 FN). Figurative-language and incitement experiments rejected (dev regressions). |

## Final evidence pointers

- GEN-10: `docs/GEN-10_JUDGE_RELIABILITY_FINAL.md`,
  `evaluation/results/phase3/gen10-human-adjudicated.json`,
  `gen10-structured-judge-20261111T215246.json`.
- GEN-12: `docs/GEN-12_API.md`, `docs/GEN-12_API_AUDIT.md`,
  `tests/unit/api/test_api.py`.
- GEN-15: `docs/GEN-15_TIER3_CLASSIFIER_REPORT.md`,
  `evaluation/results/phase3/gen15-independent-eval-offset0-20261111T105628.json`.
- GEN-02/07: `docs/GUARDRAIL_EVALUATION.md`,
  `evaluation/results/jbb-v2-repro.json`.

No prompt-only change was shipped without independent evidence. Historical
artifacts were preserved.
