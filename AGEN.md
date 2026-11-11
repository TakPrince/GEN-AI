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

# Engineering Rules

1. Never rebuild an existing working component without evidence that replacement is necessary.
2. Never add a dependency before checking whether the existing stack can solve the problem.
3. Never download a model or dataset without checking whether it already exists.
4. Never overwrite historical evaluation reports.
5. Never count failed or unscorable security checks as successful defenses.
6. Never claim a security improvement without measurable evaluation results.
7. Never tune against the final held-out evaluation dataset.
8. Keep input and output security responsibilities clearly separated.
9. Keep real classifier mode separate from heuristic demonstration mode.
10. Record exact model, dataset, configuration and run versions.
11. Preserve fail-closed behavior at security-critical boundaries.
12. Add regression tests for every security-sensitive bug fix.
13. Prefer CPU-efficient and storage-efficient implementations.
14. Do not run multiple large local models simultaneously unless required.
15. Keep changes small, reviewable and reversible.
16. Do not commit or push without explicit authorization.
17. Record limitations honestly; passing tests does not prove complete security.
18. Prioritize preventing sensitive information leakage and detecting real attacks over adding features.
