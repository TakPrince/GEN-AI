# Pipeline Folder Structure

```text
safety-platform/
|-- .github/
|   `-- workflows/
|-- configs/
|   |-- evaluation/
|   |-- guardrails/
|   |-- policies/
|   `-- telemetry/
|-- datasets/
|   |-- baselines/
|   |   |-- attacks/
|   |   |-- benign/
|   |   `-- manifests/
|   |-- domain/
|   |-- manifests/
|   |-- processed/
|   |-- raw/
|   |-- schemas/
|   `-- scripts/
|-- deploy/
|   |-- compose/
|   |-- docker/
|   `-- systemd/
|-- docs/
|   |-- adr/
|   |-- api/
|   |-- architecture/
|   |-- runbooks/
|   |-- INNOVATIONS.md
|   |-- PROJECT_STRUCTURE.md
|   |-- RESEARCH_INNOVATION.md
|   `-- TREE.md
|-- migrations/
|   `-- versions/
|-- scripts/
|-- src/
|   `-- safety_platform/
|       |-- api/
|       |   |-- v1/
|       |   |   |-- collections/
|       |   |   `-- execution/
|       |   |-- deps.py
|       |   `-- errors.py
|       |-- config/
|       |-- core/
|       |-- db/
|       |   |-- adapters/
|       |   |-- config/
|       |   |-- models/
|       |   `-- repositories/
|       |-- evaluation/
|       |   |-- calibration/
|       |   |-- harness/
|       |   |-- metrics/
|       |   |-- reports/
|       |   `-- routing/
|       |-- guardrails/
|       |   |-- tier1/
|       |   |-- tier2/
|       |   `-- tier3/
|       |-- inference/
|       |   `-- templates/
|       |-- middleware/
|       |   `-- filters/
|       |-- orchestrator/
|       |   |-- mutators/
|       |   `-- strategies/
|       |-- schemas/
|       |-- telemetry/
|       |   |-- collectors/
|       |   `-- transports/
|       |-- __main__.py
|       |-- app.py
|       `-- main.py
|-- tests/
|   |-- contract/
|   |-- data/
|   |-- e2e/
|   |-- fixtures/
|   |   |-- model_stubs/
|   |   |-- prompts/
|   |   `-- traces/
|   |-- integration/
|   |-- performance/
|   |-- regression/
|   `-- unit/
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- .pre-commit-config.yaml
|-- .python-version
|-- ARCHITECTURE.md
|-- CHANGELOG.md
|-- CONSTRAINTS.md
|-- CONTRIBUTING.md
|-- Dockerfile
|-- LICENSE
|-- Makefile
|-- README.md
|-- SECURITY.md
|-- alembic.ini
|-- docker-compose.yml
|-- pyproject.toml
`-- uv.lock
```

## Key Components

- `src/safety_platform/middleware/` - transport-scoped request/response filters delegating all safety decisions to `guardrails/`
- `src/safety_platform/telemetry/collectors/` - bounded non-blocking collectors for stage, latency, and loss metrics
- `src/safety_platform/db/adapters/` - async engine and session adapters for SQLite, Postgres, and DuckDB
- `datasets/baselines/` - immutable attack and benign reference corpora pinned by manifest
