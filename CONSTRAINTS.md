# Constraints

Last reviewed: 2026-10-28 | Status: binding | Enforcement: scaffold + CI

## Floor

- No new `# type: ignore`, `# noqa`, `# pylint: disable`, `# pragma: no cover`
- No stubs: `NotImplementedError`, `pass`-only handlers, empty `except`
- No skipped or deleted tests without commit reason
- No secrets in source, logs, fixtures, traces
- No weakening this file to pass a change
- No standing exceptions to the numeric gates

## Enforced with numbers

| Dimension | Rule | Why | Checked by | Runs at |
|---|---|---|---|---|
| Tier 1 latency | p95 <60ms; warm; 1-4KiB prompt; zero model/network calls | Cheap-first budget; preserves cascade headroom | `pytest tests/performance/test_tier1_latency.py -q` | task end + CI |
| Tier 2 latency | p95 <20ms; warm; local embedder, 128-centroid index; zero model/network calls | Keeps ambiguity scoring inside the fast local path | `pytest tests/performance/test_tier2_latency.py -q` | task end + CI |
| Guardrail envelope | p95 <400ms; Tier 1 + Tier 2 + router + non-blocking telemetry enqueue | Bounds the complete pre-inference gate | `pytest tests/performance/test_guardrail_envelope.py -q` | task end + CI |
| Embedding-loop memory | peak RSS <=512MiB per worker; 10k-loop soak | Bounds local embedder + index + batch headroom; blocks leak-driven OOM | `pytest tests/performance/test_embedding_memory.py -q` | task end + CI |
| Unit coverage | >=80% line coverage; `src/safety_platform`; unit tests only | Mandatory regression floor | `pytest tests/unit --cov=safety_platform --cov-fail-under=80` | task end + CI |
