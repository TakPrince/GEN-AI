# TODO / Roadmap

Implementation status is tracked in `docs/IMPLEMENTATION_PLAN_V2.md`; this file
only mirrors the short list. The former ASGI/API service roadmap was abandoned
(see Task plan §scope) and was removed.

## Implementation tasks (see `docs/IMPLEMENTATION_PLAN_V2.md` for detail)

- [x] Tasks 1–10: guardrails hardening, unified defense interface, judge
  hardening, single-turn red-team orchestrator, multi-turn conversations with
  partial-compliance evaluation, conversation metrics, CLI + artifacts.
- [x] Task 11: Makefile hygiene (disposable `make eval-smoke` DB), pytest
  config consolidated into `pytest.ini`, unused `ollama` SDK dependency
  removed (stdlib urllib client), docs refreshed, SQLite untracking plan
  documented (see below).
- [ ] Task 2 follow-up: untrack local SQLite databases (needs owner approval —
  do not run destructively without review):
  ```bash
  git rm --cached evaluation/results/jbb-v2.sqlite3 metrics/dashboard_demo.sqlite3
  # then add to .gitignore:  evaluation/results/jbb-v2.sqlite3  metrics/*.sqlite3
  ```
  These files stay on disk; only git tracking is removed.

## Outstanding engineering work (recommendations, not started)

- [ ] Full 30/30 held-out rerun only if tier-3 separation improves (Checkpoint B gate).
- [ ] Larger multi-turn live campaign with the ConversationJudge verdict set
  (pairwise calibration of `partial_compliance` labels on stored transcripts).
- [ ] Per-tier FPR attribution and a drift-aware router-training exercise
  (Innovation 2 A/B on an artifact, not hardcoded weights).
- [ ] Optional: consolidate the three locally-duplicated Ollama base-URL
  defaults (`127.0.0.1:11434`) into one config constant.

## Future research ideas (non-implementation)

- CircleGuardBench / Tensor Trust loaders as additional datasets.
- Postgres/DuckDB/Kafka telemetry backends behind the existing adapter seam.
- vLLM as an alternative inference backend to Ollama.
