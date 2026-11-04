# GEN-AI Development Rules

## Project

LLM Red-Teaming and Guardrails Platform.

## Engineering Rules

- Inspect existing code before modifying it.
- Reuse existing implementations.
- Avoid unnecessary dependencies and abstractions.
- Preserve existing architecture unless a change is justified.
- Never claim a test passed without running it.
- Never fabricate evaluation results.
- Never silently skip failing tests.
- Keep changes small and focused.

## Skill Usage

- Coding task: ponytail
- Feature planning: spec-driven-development
- Multi-file changes: incremental-implementation
- Logic changes: test-driven-development
- Failures: debugging-and-error-recovery
- Security: security-and-hardening
- Final review: code-review-and-quality
- Git changes: git-workflow-and-versioning

Load skills only when relevant.

Do not load multiple overlapping skills unnecessarily.

## Execution

Implement, test, inspect results and commit each

verified implementation slice.

Do not rewrite working modules without evidence.
