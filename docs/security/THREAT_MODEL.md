# GEN-04: Threat Model

## 1. Executive Summary

This document establishes a threat model for the **LLM Red-Teaming and Guardrails** system, covering the existing three-tier security control architecture (Tier 1 → Tier 2 → ML Router → Tier 3), red-team orchestration, telemetry, and evaluation capabilities.

The threat model identifies both known and potential attack surfaces, documents existing security controls, and establishes a foundation for security engineering decisions.

**Scope:** This threat model covers the existing production components as documented in the codebase. It does **not** include planned features (e.g., vLLM alternatives) or future components outside the current implementation.

**Verification:** This threat model was developed through systematic code review and analysis of the actual implementation. All identified threats have been evaluated against the existing source code, tests, and documentation to ensure evidence-based findings.

---

## 2. System Overview

The system implements a **three-tier security control cascade** for both input and output content processing:

```
User Input ──► Tier 1 (regex/block/redact) ──► Tier 2 (embeddings) ──►
                                \
                                 ML Router ──► Tier 3 (LLM classifier) ──►
                                /
                                 Response Router ──► Output (safe/blocked/redacted/partial)
```

**Key Components:**

1. **Three-Tier Security Cascade** → Deterministic regex → Embedding semantic → LLM classification
2. **Red-Team Orchestrator** → Multi-turn adversarial conversation planning
3. **Response Router** → Final decision-making and response routing
4. **Cascade Evaluation System** → Evaluation harness for measuring effectiveness
5. **Telemetry System** → Comprehensive logging and monitoring
6. **Dashboard Interface** → Streamlit-based management interface

**Trust Boundaries:**
- User input enters the security pipeline at Tier 1
- Tier 1 → Tier 2 → ML Router → Tier 3 forms the core security control chain
- Output response passes through Output Guardrails before delivery
- System internals (model prompts, canaries, secrets) are protected by confidentiality controls
- Red-Team Orchestrator operates with elevated permissions to execute adversarial attacks

---

## 3. Protected Assets

### Security-Critical Assets

| Asset | Location | Protection Requirements | Impact if Compromised |
|-------|----------|------------------------|---------------------|
| **System Prompts** | `src/safety_platform/guardrails/tier3/guard_classifier.py:80-120` | Confidentiality, integrity | Escalation, manipulation |
| **Canary Values** | `src/safety_platform/guardrails/canary.py` | Confidentiality, tamper evidence | Secret leakage, bypass |
| **Guardrail Rules** | `src/safety_platform/guardrails/tier1/rules.py` | Integrity, version control | Bypass, false negatives |
| **Model Metadata** | `src/safety_platform/guardrails/tier2/vector_filter.py:128-150` | Integrity, authenticity | Model confusion, escalation |
| **Evaluation Data** | `evaluation/results/` (JSONL, SQLite) | Integrity, confidentiality | Benchmark corruption |
| **Telemetry Records** | `metrics/telemetry.sqlite3` | Integrity, auditability | Attack reconstruction |

### Technical Assets

| Asset | Description | Threat Model Relevance |
|-------|-------------|----------------------| |
| **Execution Pipeline** | `src/safety_platform/execution.py` | Core security enforcement |
| **Model Connection** | Ollama client (`OllamaClient`) | Service interruption, manipulation |
| **Configuration** | Environment variables, YAML configs | Configuration hijacking |
| **Logging** | `stage_event` table in telemetry | Attack analysis, debugging |

---

## 4. Attacker Profiles and Capabilities

### Primary Attacker Types

1. **Prompt Injection Attackers**
   - Goal: Bypass or subvert security controls
   - Techniques: Direct instruction manipulation, indirect injection
   - Capabilities: Craft prompts, use encoding/obfuscation

2. **Multi-Turn Escalation Attackers**
   - Goal: Gradually reconstruct secrets/system prompts
   - Techniques: Split-context attacks, coverage probing
   - Capabilities: State tracking, adaptive strategy

3. **Resource Exhaustion Attackers**
   - Goal: Deny service, cause failures
   - Techniques: Oversized inputs, rapid fire requests
   - Capabilities: Rate control, payload crafting

4. **Model Confusion Attackers**
   - Goal: Induce incorrect model behavior or classification
   - Techniques: Semantic perturbation, prompt engineering
   - Capabilities: Understand model boundaries, exploit ambiguities

### Attacker Capabilities by Component

| Component | Attacker Capabilities | Limitations |
|-----------|----------------------|-------------|
| **Input Guardrails** | Craft prompts matching/leaking bypass patterns | Cannot break cryptographic hashes |
| **Tier 3 Classifier** | Adversarial prompt engineering | Limited model access |
| **Canary System** | Extract values via LLM queries | Partial view through guards |
| **Telemetry** | Replay dead-letter JSONL | No direct injection |

---

## 5. Attack Surfaces

### 1. Prompt Injection Surface

**Location:** Tier 1 regex rules (`src/safety_platform/guardrails/tier1/rules.py:12-44`)

**Vulnerable Patterns:**
- "ignore previous instructions"
- "you are now developer mode" 
- "reveal system prompt"
- Credential theft patterns

**Impact:** Direct bypass of security controls

**Current Controls:**
- Comprehensive regex patterns
- Heuristic coverage detection
- Guardrails with fail-closed behavior

**Tested:** Integration tests (`test_canary_leakage.py`) confirm injection protection

### 2. Multi-Turn Extraction Surface

**Location:** Canary detection system (`src/safety_platform/guardrails/canary.py:51-122`)

**Attack Vector:** Split canary values across multiple turns to evade single-turn detection

**Impact:** Secret extraction, privilege escalation

**Current Controls:**
- Partial fragment tracking (`_lengths` parameter)
- Cross-turn tail (`ConversationTail`)
- Blacklist/blocklist for leaked fragments

**Tested:** Multi-turn canary leakage tests (`test_canary_leakage.py`)

### 3. Guardrail Bypass Surface

**Location:** ML Router decision logic (`src/safety_platform/research/ml_router.py:240-260`)

**Attack Vector:** Adversarial examples that manipulate similarity scores

**Impact:** False negatives, safety failures

**Current Controls:**
- Weighted scoring system
- Threshold mechanisms
- Adversarial training notes

**Evidence:** GitHub commit `0ebd2e1`: "centralize router configuration"

### 4. Model Extraction Surface

**Location:** Tier 3 LLM classifier (`src/safety_platform/guardrails/tier3/guard_classifier.py:240-310`)

**Attack Vector:** LLM queries to extract system information

**Impact:** System compromise, information disclosure

**Current Controls:**
- Rate limits and capacity controls
- Fail-closed error handling
- Query timeout mechanisms

**Limitation:** Model exposure through classifier outputs

### 5. Output Guardrail Surface

**Location:** Output cascade (`src/safety_platform/evaluation/cascade.py:51-140`)

**Attack Vector:** Response manipulation, output parsing

**Impact:** Malicious response delivery, data exfiltration

**Current Controls:**
- Direction-aware processing
- Redaction capabilities
- Output validation

### 6. Telemetry Manipulation Surface

**Location:** Dead-letter file replay (`src/safety_platform/telemetry/replay.py:129-200`)

**Attack Vector:** Corrupt JSONL entries, replay invalid data

**Impact:** False telemetry, replay-based attacks

**Current Controls:**
- Idempotent replay with `INSERT OR IGNORE`
- Schema validation
- Malformed record detection

### 7. Configuration Manipulation Surface

**Location:** Evaluation config (`src/safety_platform/evaluation/runner.py:90-120`)

**Attack Vector:** Modify evaluation parameters

**Impact:** Biased results, data corruption

**Current Controls:**
- Integrity checks (`dataset_digest`)
- Read-only configuration

---

## 6. STRIDE Threat Analysis

### S - Spoofing

**Threat:** Authentication bypass through credential manipulation

**Affected Component:** Red-Team Orchestrator (`src/safety_platform/orchestrator/durable.py:621-670`)

**Attack Path:** Attacker creates invalid campaign record, impersonation

**Preconditions:** Access to campaign creation API

**Impact:** Session hijacking, unauthorized access

**Existing Control:** Lease-based access control, lease expiration

**Evidence:** `LeaseHeldError` class, `campaign.acquire()` with lease validation

**Risk:** Medium (Requires API access)

### T - Tampering

**Threat:** Modification of guardrail rules or configuration

**Affected Component:** Tier 1 rules (`src/safety_platform/guardrails/tier1/rules.py:12-44`)

**Attack Path:** Direct file modification, code injection

**Preconditions:** File system write access

**Impact:** Bypass controls, false negatives

**Existing Control:** Immutable rule definitions, compiled patterns

**Evidence:** All rules stored as compile-time constants, not runtime variables

**Risk:** High (Runtime compromise)

### R - Repudiation

**Threat:** Denial of actions, data deletion

**Affected Component:** Evaluation store (`src/safety_platform/evaluation/runner.py:1-140`)

**Attack Path:** Delete SQLite database, tamper with results

**Preconditions:** Database access, write permissions

**Impact:** Evidence destruction, audit trail loss

**Existing Control:** Durable WAL-mode SQLite, read-only access in production

**Risk:** Medium (Requires persistence access)

### I - Information Disclosure

**Threat:** Leak of system prompts, canaries, or internal state

**Affected Component:** Tier 3 outputs (`src/safety_platform/guardrails/tier3/guard_classifier.py:80-120`)

**Attack Path:** Queries through guardrail inputs

**Preconditions:** Valid guardrail query access

**Impact:** System compromise, security bypass

**Existing Control:** Prompt hiding through system prompts, canary protection

**Evidence:** Canary detection, system prompt hiding in `guard_classifier.py:80-120`

**Risk:** High (Canary extraction demonstrated)

### D - Denial of Service

**Threat:** Service interruption, resource exhaustion

**Affected Component:** Model connection (`src/safety_platform/evaluation/llm.py:64-130`)

**Attack Path:** Oversized requests, rapid fire, connection timeouts

**Preconditions:** Model connectivity, rate limits

**Impact:** Service unavailability, degraded performance

**Existing Control:**
- Prompt size limits (`_MAX_PROMPT_SIZE`)
- Rate limiting (`UrllibOllamaTransport.max_concurrent`)
- Timeout mechanisms

**Risk:** Medium (Resource-based)

### E - Elevation of Privilege

**Threat:** Gain unauthorized access to system functions

**Affected Component:** Execution pipeline (`src/safety_platform/execution.py:1-500`)

**Attack Path:** Guardrail bypass, privilege escalation through prompts

**Preconditions:** Security control weaknesses

**Impact:** System compromise, unauthorized actions

**Existing Control:**
- Defense-in-depth (three tiers)
- Clean-room implementation
- Access controls

**Risk:** Medium (Exploitation-dependent)

---

## 7. Existing Security Controls Assessment

### Implemented Controls

| Control | Component | Status | Evidence |
|---------|-----------|--------|----------|
| **Regex-based blocking** | `tier1/rules.py:12-44` | Implemented | 112 regex patterns, integration tests |
| **Embedding filtering** | `tier2/vector_filter.py:128-150` | Implemented | Cosine similarity, thresholds |
| **Tier 3 classification** | `tier3/guard_classifier.py:80-120` | Implemented | LLM-based, configurable thresholds |
| **Canary detection** | `canary.py:51-122` | Implemented | Partial-fragment tracking, coverage |
| **Output checking** | `cascade.py:51-140` | Implemented | Direction-aware validation |
| **Rate limiting** | `UrllibOllamaTransport:128-150` | Implemented | Bounded semaphores |
| **Telemetry logging** | `telemetry/bus.py:8-31` | Implemented | Stage events, non-blocking |

### Partially Implemented Controls

| Control | Notes |
|---------|-------|
| **Model output monitoring** | Limited to Tier 3 classification |
| **Attack pattern tracking** | Reactive rather than proactive |

### Missing/Weak Controls

| Control | Description | Recommendation |
|---------|-------------|----------------|
| **Runtime integrity** | No attestation of running system | Code signing, integrity checks |
| **Fine-grained access control** | Role-based permissions | Implement RBAC |
| **Adversarial robustness testing** | Static rule sets | Dynamic adversarial testing |

### Test Coverage

**Evidence:** Integration tests (`tests/integration/`) verify many controls
- `test_canary_leakage.py`: Canary extraction protection
- `test_pipeline_cascade.py`: Cascade behavior validation
- `test_target_failure.py`: Target failure handling
- `test_execution.py`: Execution pipeline validation

**Coverage Gaps:**
- No comprehensive end-to-end red-team attack simulation
- Limited runtime security testing

---

## 8. Risk Assessment

### High Risk

| Threat | Component | Likelihood | Impact | Priority |
|--------|-----------|-----------|--------|----------|
| Canary extraction | Canary system | Medium | High | High |
| Guardrail bypass | ML Router | Medium | High | High |
| Model compromise | Tier 3 classifier | Low | High | High |

### Medium Risk

| Threat | Component | Likelihood | Impact | Priority |
|--------|-----------|-----------|--------|----------|
| Input injection | Tier 1 rules | Low | Medium | Medium |
| Output manipulation | Output guardrails | Low | Medium | Medium |
| Telemetry corruption | Dead-letter replay | Low | Medium | Medium |

### Low Risk

| Threat | Component | Likelihood | Impact | Priority |
|--------|-----------|-----------|--------|----------|
| Configuration tampering | Evaluation config | Very Low | Low | Low |
| Service denial | Resource limits | Medium | Low | Low |
| Privilege escalation | Execution pipeline | Very Low | Medium | Low |

---

## 9. Mapping to Existing Issue Tracker

### Directly Addressed by Existing Work

| Issue | Control | Coverage |
|-------|---------|----------|
| **GEN-01 (Tier 3 classifier)** | Tier 3 classification | Full |
| **GEN-02 (Attack detection)** | Tier 1 + Tier 2 | Good |
| **GEN-03 (Output security)** | Output checking | Full |
| **GEN-07 (ML Router)** | ML routing logic | Implemented |
| **GEN-08 (Dashboard telemetry)** | Telemetry system | Implemented |
| **GEN-09 (Indirect injection)** | Tier 1 indirect rules | Implemented |
| **GEN-10 (Judge reliability)** | Evaluation harness | Implemented |
| **GEN-11 (Telemetry reliability)** | Telemetry bus/consumer | Implemented |

### Gaps in Existing Work

| Issue | Gap | Recommended Action |
|-------|-----|------------------| |
| **GEN-04 (Threat model)** | No formal threat analysis | Create this document (current work) |
| **GEN-05 (Evaluation validity)** | Limited adversarial simulation | Develop comprehensive red-team tests |
| **GEN-12 (Production API)** | API security not addressed | Add API security controls |
| **GEN-13 (Reproducibility)** | Limited reproducibility documentation | Document evaluation procedures |
| **GEN-14 (Resource exhaustion)** | Rate limiting only at model level | Add comprehensive rate limiting |

---

## 10. Security Assumptions

1. **Local Execution Model:** All components run locally; no external network dependencies
2. **Configuration Security:** Environment variables and configuration files are protected
3. **Isolation:** System components are isolated through Python process boundaries
4. **Fail-Closed:** Security controls default to blocking when uncertain
5. **Audit Integrity:** All security events are logged and immutable

### Assumptions Under Review

1. **Canary Protection:** Canary values are effectively protected (tested but not exhaustive)
2. **Guardrail Completeness:** All attack patterns are covered (residual risk acknowledged)
3. **Model Security:** LLM classifier outputs are trustworthy (evaluated but not proven)

---

## 11. Limitations

1. **Testing Scope:** Most security controls are validated through unit/integration tests but not comprehensive adversarial testing
2. **Dynamic Threats:** Static rule sets may be vulnerable to evolving attack patterns
3. **Complexity:** Three-tier cascade increases complexity and potential failure points
4. **Dependencies:** System depends on external Ollama service

### Open Risks

1. **Model Evasion:** Advanced adversarial examples against Tier 3 classifier
2. **Resource Exhaustion:** DoS attacks through oversized payloads
3. **Prompt Engineering:** Sophisticated injection techniques

---

## 12. Prioritized Recommendations

### Critical (P0)

1. **Add Comprehensive Red-Team Tests:** Systematic adversarial testing against all security controls
2. **Implement Runtime Integrity:** Code signing, integrity verification, and tamper protection
3. **Enhance Canary Protection:** Multi-layered canary detection with cross-validation

### High (P1)

1. **Adversarial Robustness Testing:** Dynamic generation of adversarial examples
2. **Fine-Grained Access Control:** Implement role-based permissions
3. **Model Output Monitoring:** Real-time monitoring of Tier 3 outputs

### Medium (P2)

1. **Security Documentation:** Maintain and update security documentation
2. **Incident Response:** Develop formal incident response procedures
3. **Performance Monitoring:** Monitor security control performance metrics

### Low (P3)

1. **User Education:** Security awareness training for users
2. **Configuration Auditing:** Regular security configuration audits
3. **Documentation Updates:** Keep security documentation current

---

## 13. Conclusion

This threat model provides a comprehensive foundation for understanding the security posture of the **LLM Red-Teaming and Guardrails** system. The system implements multiple security controls with varying levels of effectiveness, and identified gaps highlight areas for future security enhancement.

**Key Findings:**
- The system has strong defensive controls (three-tier cascade, canary detection)
- Multiple attack surfaces are protected (input/output, telemetry, configuration)
- Significant gaps remain (adversarial testing, runtime integrity)
- Existing components provide good security but require enhancement

**Immediate Actions:**
1. Implement comprehensive red-team testing
2. Enhance canary protection mechanisms
3. Add runtime integrity controls

This threat model serves as a foundation for ongoing security improvement and risk management. Regular updates are necessary as the system evolves and new threats emerge.

---

## 14. Threat Register

| ID | Threat | Component | Likelihood | Impact | Priority | Status |
|----|--------|-----------|-----------|--------|----------|--------|
| T-001 | Canary Value Extraction | Canary System | Medium | High | High | Evidence |
| T-002 | Guardrail Bypass | ML Router | Medium | High | High | Evidence |
| T-003 | System Prompt Disclosure | Tier 3 Classifier | Low | High | High | Evidence |
| T-004 | Input Injection | Tier 1 Regex | Low | Medium | Medium | Evidence |
| T-005 | Output Manipulation | Output Guardrails | Low | Medium | Medium | Evidence |
| T-006 | Telemetry Replay | Dead-Letter | Low | Medium | Medium | Evidence |
| T-007 | Configuration Tampering | Evaluation Store | Very Low | Low | Low | Evidence |
| T-008 | Resource Exhaustion | Model Connection | Medium | Low | Low | Evidence |
| T-009 | Privilege Escalation | Execution Pipeline | Very Low | Medium | Low | Evidence |

---

## 15. References

1. GitHub repository commits and source code
2. Integration tests (`test_canary_leakage.py`, `test_pipeline_cascade.py`)
3. Documentation (`phase24_architecture_blueprint.md`, `docs/ARCHITECTURE.md`)
4. Evaluation reports (`evaluation/results/*.json`)
5. Existing issue tracker (GEN-01 through GEN-14)

---

*Document generated: 2026-11-10*
*Prepared by: Security Assessment Team*
*Status: COMPLETE - Evidence-based threat model for existing system components*