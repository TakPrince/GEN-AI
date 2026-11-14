# P2-04: CLI Migration and Shared Runtime Convergence

## Overview

Migrate GEN-AI from a Streamlit dashboard to a CLI-first architecture while consolidating dashboard and evaluation execution around one shared guardrail runtime.

## Target Architecture

```text
                  GEN-AI
                     |
              Entry Interfaces
                 /       \
               CLI     Evaluation
                 \       /
                  \
               Shared Runtime
                     |
               Defense Pipeline
              /       |       \
           Tier 1   Tier 2   Tier 3
              \       |       /
               Escalation Policy
                     |
                Response Router
                     |
                Ollama Runtime
                     |
               Result + Telemetry
```

**Key Principle:** The CLI is an interface, not the security engine. All guardrail decisions remain inside reusable core modules.

## Current State Analysis

### Dashboard (src/safety_platform/app.py)
- `_build_pipeline(database: str) -> _Pipeline`
- Uses `Streamlit` for UI
- `_tier3_guard()` with `MULTI_SIGNAL_GATES` (tier3_requires_ambiguity=False)
- Uses `SqliteTelemetrySink` for telemetry
- Creates `CascadeRuntime` with `tier3_on_output=True`
- Uses hardcoded centroids (override, exfiltration, harm)
- Uses `_HashingEmbedder`
- Uses `defense_regex_filter()`

### Evaluation (src/safety_platform/evaluation/runner.py)
- `run_evaluation(config: RunConfig) -> str`
- Uses `build_cascade()`
- Uses `OllamaEmbedder`
- Uses `GuardClassifier`
- Uses `OllamaSpeculativeEchoProbe` if enabled
- Uses `_warm_guard()`
- Uses configurable thresholds and margins
- Uses `OllamaClient` for judging

## Convergence Opportunities

### Common Components
- `GuardClassifier` used by both
- `MULTI_SIGNAL_GATES` constant
- `CascadeRuntime` used by both
- `ResponseRouter` used by both
- `Tier3BypassRouter` shared
- `ModelDrivenRouter` shared
- `RouterFeatures`, `RouterDecision` classes
- `escalation_for` function

### Differences
- **Dashboard**: Hardcoded test centroids, Streamlit UI, SQLite telemetry, Streamlit async loops
- **Evaluation**: `build_cascade()` for centroid creation, `OllamaEmbedder`, configurable, no UI, no SQLite

## Implementation Plan

### P2-04.1 — Runtime Architecture Audit

**Tasks:**
1. Audit current dashboard execution path
2. Audit evaluation execution path
3. Produce comparison table
4. Identify duplicated code
5. Identify different configurations
6. Identify components that should remain independent

**Deliverable:** Convergence analysis report

### P2-04.2 — Shared Runtime Interface

**Create/consolidate:**
- Minimal shared runtime interface that both CLI and evaluation can use
- Reusable core modules for security decisions
- Common configuration loading
- Consistent telemetry handling

**Requirements:**
- Both CLI and evaluation use the same guardrail execution logic
- Consistent runtime construction
- Shared Tier 1-3 configurations
- Consistent error handling

### P2-04.3 — CLI Implementation

**Requirements:**
1. Single-prompt execution
2. Interactive mode
3. Result display with structured output
4. Meaningful exit codes
5. Optional structured output (JSON)

**Commands:**
- `gen-ai check <prompt>` - Single prompt evaluation
- `gen-ai interactive` - Interactive mode
- `gen-ai --help` - Help documentation

### P2-04.4 — Evaluation Integration

**Requirements:**
- Evaluation calls the shared runtime
- Maintains existing datasets and reporting behavior
- No terminal output parsing by evaluation
- Consistent security decisions across interfaces
- Same input and security configuration produce equivalent decisions

### P2-04.5 — Dashboard Functionality Migration

**Classification:**
- Core guardrail execution → Move/reuse in shared runtime
- Prompt submission → CLI equivalent
- Tier decision display → CLI equivalent
- Runtime configuration → Preserve through existing configuration
- Reports → Existing evaluation reports
- UI/UX → Remove unless required for security

### P2-04.6 — Dashboard Removal

**After successful tests:**
1. Remove Streamlit-specific UI code
2. Remove unused dashboard modules
3. Remove Streamlit dependencies if no longer required
4. Remove obsolete configuration
5. Update documentation and commands
6. Remove dashboard-specific tests
7. Move reusable tests to appropriate modules
8. Verify all import and execution paths

## Verification Criteria

### Runtime Parity
- Identical security decisions with same inputs and configuration
- consistent tier1-tier3 behavior
- shared admission control effectiveness
- circuit breaker behavior consistency
- configuration parity

### CLI Tests
- Single-prompt execution with result display
- Interactive mode with proper input handling
- Meaningful exit codes
- Optional JSON output
- Error handling and validation

### Security Regression
- No fail-open behavior changes
- Escalation policy consistency
- No admission control bypass
- Circuit breaker functionality
- Payload limit enforcement

### Evaluation Regression
- Dataset compatibility
- Metric preservation
- Reporting consistency
- Execution decision alignment

### Dashboard Removal
- No remaining Streamlit imports
- No broken package imports
- No unaddressed functionality dependencies
- No obsolete dependencies

## Technical Implementation Plan

### 1. Create Shared Runtime Interface

**File:** `src/safety_platform/runtime.py` (new)

**Components:**
- `GuardClassifier` factory function
- `CascadeRuntime` factory function
- `create_runtime()` main entry point
- Common configuration loading
- Consistent error handling

### 2. Create CLI Module

**File:** `src/safety_platform/cli.py` (new)

**Components:**
- `execute_prompt()` single prompt execution
- `interactive_mode()` interactive interface
- Result formatting functions
- Argument parsing
- Exit code handling

### 3. Update Evaluation

**Changes:**
- Modify `src/safety_platform/evaluation/runner.py`
- Use shared runtime instead of direct component construction
- Preserve existing API and behavior
- Remove direct component instantiation where appropriate

### 4. Create Dashboard Migration

**File:** `src/safety_platform/dashboard_migration.py` (new/transitional)

**Purpose:** Temporary bridge during migration

### 5. Remove Dashboard

**Files to remove/revise:**
- `src/safety_platform/app.py` → Keep `runtime.py` export, remove UI code
- `src/safety_platform/__pycache__/app.cpython-313.pyc` → Remove
- `tests/unit/test_app.py` → Revise
- `tests/integration/test_dashboard_resource_isolation.py` → Remove
- Any remaining Streamlit dependencies

## Testing Strategy

### Unit Tests
- Runtime parity tests (shared runtime)
- CLI functionality tests
- Integration tests between CLI and runtime
- Security regression tests
- Evaluation compatibility tests

### Integration Tests
- Full runtime construction tests
- CLI- runtime interaction tests
- End-to-end scenario tests

### Performance Tests
- Latency and throughput measurements
- Memory usage validation
- Concurrency handling tests

## Risk Assessment

### High Risk
- Shared runtime interface design
- CLI implementation completeness
- Evaluation compatibility maintenance

### Medium Risk
- Breaking existing tests during migration
- Performance regression
- Error handling changes

### Low Risk
- Code organization improvements
- Documentation updates
- Minor refactoring

## Timeline

**Week 1-2:** Runtime architecture audit and shared interface design
**Week 3-4:** Shared runtime implementation and CLI development
**Week 5-6:** Evaluation integration and testing
**Week 7:** Dashboard migration and cleanup
**Week 8:** Final testing and documentation

## Success Criteria

✅ CLI and evaluation use shared runtime
✅ Identical security decisions with same inputs
✅ CLI provides essential functionality
✅ Evaluation maintains existing behavior
✅ All tests pass
✅ No security regressions
✅ Clean codebase without Streamlit

## Cost-Benefit Analysis

### Benefits
- Reduced maintenance burden
- Consistent security decisions across interfaces
- Simplified codebase
- Easier testing and validation
- Better resource utilization

### Costs
- Migration effort and risk
- Temporary functionality gaps
- Testing and validation overhead
- Potential short-term disruption

## Conclusion

This migration will consolidate GEN-AI around a single, reusable guardrail runtime while providing a modern CLI interface. The approach preserves all existing security functionality while improving maintainability and reducing technical debt.

The migration is technically feasible with careful testing and incremental implementation. The benefits in terms of code quality and maintainability justify the investment.
