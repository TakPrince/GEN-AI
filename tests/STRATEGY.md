# Test Strategy

- Scope: Tier 1 regex; Tier 2 cosine; CircleGuardBench and Tensor Trust loaders.
- Unit: pure logic; no I/O, network, model, or DB.
- Contract: schemas, manifests, adapter interfaces.
- Integration: policy -> loader -> cascade; local fixtures only.
- Regression: frozen labeled cases; exact counts.
- Performance: Tier 1 p95 <60ms; embedding RSS <=512MiB.
- Dataset large: nightly; pinned external snapshots.

## Regex Filtering

- Unit: match/no-match; anchors; flags; unicode; multiline; escaping; empty; 4KiB limit.
- Unit: deny-list precedence; canary echo; normalization before match; redaction spans.
- Negative: invalid pattern rejected at load; compile errors typed; no silent skip.
- Security: ReDoS fixture bounded; no catastrophic backtracking; no model or network call.
- Contract: YAML rule schema accepts named rules and expected verdict.
- Integration: policy load -> compiled matcher -> Tier 1 verdict; first block short-circuits.
- Regression: frozen corpus per rule; expected block/allow labels; zero drift.
- Performance: `tests/performance/test_tier1_latency.py`; p95 <60ms.

## Cosine Distance

- Unit: dot, norm, similarity; distance = 1 - similarity; range [-1,1]; epsilon clamp.
- Unit: zero vector, NaN/Inf, dimension mismatch -> typed error.
- Unit: normalized centroid = mean of unit vectors; empty index -> no-match.
- Unit: top-1/top-2; margin = s1 - s2; ties resolved by stable ID.
- Property: scale invariance; permutation invariance; symmetry.
- Contract: embedding dimension, model ID, and index hash pinned.
- Integration: index load -> query -> pass/ambiguous/block bands; boundary inclusivity pinned.
- Regression: frozen vectors; expected scores within 1e-6.
- Performance: `tests/performance/test_embedding_memory.py`; RSS <=512MiB over 10k loops.

## CircleGuardBench Loader

- Contract: manifest source, commit/version, SHA256, license, row count.
- Contract: required fields; labels/categories map to project taxonomy; unknown labels rejected.
- Data: unique IDs; no empty text; no duplicate normalized text; category coverage report.
- Integration: pinned fixture -> normalized records -> metric harness; network disabled.
- Fail-closed: checksum, schema, or version mismatch aborts whole load; no partial ingest.
- Split: deterministic; sealed test excluded from calibration.

## Tensor Trust Loader

- Contract: manifest source, commit/version, SHA256, license, row count.
- Contract: attack/defense pairs parsed; prompt + injection required; map to attack/benign.
- Data: unique IDs; dedupe normalized text; extract injection spans; dropped rows counted.
- Integration: pinned fixture -> normalized attack set -> ASR harness; network disabled.
- Security: no secret or password values in logs, fixtures, or errors.
- Fail-closed: checksum, schema, or version mismatch aborts whole load; no partial ingest.
- Split: deterministic; sealed test excluded from calibration.

## Shared Dataset Gates

- Immutable raw snapshot; no in-place mutation.
- Record source URL, version, license, SHA256, load timestamp.
- Loader returns typed records or typed error; no mixed success.
- CI: small fixtures only; nightly: pinned full snapshots.
