# Research Innovation Validation Plan

> **Status:** Proposed validation architecture. No experiments have been run and no results are reported here.
>
> **Relationship to project design:** This document refines the two research contributions in `Readme.md` §6 and uses the runtime contracts, metric definitions, and versioning rules in `ARCHITECTURE.md`. It does not modify the architecture; it specifies how the innovations must be tested if their claims are to be credible.
>
> **Date:** 2026-10-25

---

## 0. Refined One-Page Brief

### Problem statement — How Might We

**How might we determine whether attack-driven threshold calibration and a learned Tier-3 invocation router produce real, reproducible safety-cost improvements on unseen attack and benign distributions, rather than artifacts of search, data leakage, selective labels, or distribution shift?**

The intended user is a local-first LLM security researcher or defense engineer who needs to decide whether a guardrail change is worth deploying, not merely whether it looks better on the data used to design it. Success means a pre-registered, sufficiently powered, held-out result that survives realistic perturbations and reports the cost side of the trade-off.

### Recommended direction

1. **Reframe Innovation 1 as constrained, nested threshold calibration.** The calibration workflow should select thresholds on a calibration split under an explicit FPR and Tier-3-cost budget, then be evaluated once on sealed held-out attack families, held-out time windows, and held-out benign categories. The primary comparison must include a benign-only FPR-matched baseline so that the value of _attack-driven_ calibration is separated from the value of merely moving a threshold.
2. **Reframe Innovation 2 as selective Tier-3 invocation, not a "classifier bypass."** A router that skips Tier 3 must specify a fallback decision, a safety loss, and how labels are obtained for cases where Tier 3 is not called. The learned router must improve false-safe-skip behavior at a matched invocation rate, with ASR and FPR noninferior to a static gate, not merely beat the original static threshold.
3. **Test both innovations in one 2 × 2 factorial design.** Threshold policy (static vs calibrated) and routing policy (static gate vs learned router) are separate interventions. The interaction matters because calibration changes the Tier-2 score distribution on which the router is trained.
4. **Use fixed-trajectory replay for cost and decision analysis, but closed-loop replay for run-level ASR.** A multi-turn policy change alters which responses are delivered, which alters later prompts. Re-scoring a frozen transcript cannot establish run-level attack success.
5. **Pre-register margins, sample sizes, splits, search spaces, and analysis.** Treat the held-out set as sealed until the protocol and analysis code are frozen.

### Key assumptions to validate

- [ ] Held-out attack families and time windows exist and are genuinely disjoint from development data.
- [ ] Benign traffic is representative enough to estimate category-specific FPR, and a benign-only FPR/cost-matched calibration baseline is feasible.
- [ ] Attack success and partial-compliance labels are reliable enough for run-level inference; label noise is bounded and reported.
- [ ] Tier-3 labels can be obtained on a randomized audit sample or exhaustive offline replay, avoiding selective-label bias, with enough Tier-3-unsafe cases to bound false-safe-skip.
- [ ] The learned router's features are available before Tier 3, are stable at inference time, and do not encode dataset identity, attack family, split, or ground-truth labels.
- [ ] Tier-3 latency/cost is large enough that invocation reduction matters relative to router overhead.
- [ ] Multi-turn closed-loop replay is feasible within the project's compute budget.
- [ ] The candidate effect is large enough to power a noninferiority test on ASR and FPR; otherwise the claim must be weakened or the study expanded.

### MVP scope

A defensible first experiment contains:

- One frozen local model and one frozen inference backend.
- A static-threshold cascade with a static Tier-3 gate.
- A calibration policy and one learned router.
- A benign-only FPR/cost-matched control and a matched-invocation static gate.
- Three data partitions: development, calibration, and sealed test.
- At least one held-out attack-family split, one held-out benign-category split, and one temporal split.
- Fixed-trajectory replay for decision, invocation, and latency analysis.
- Closed-loop replay for the primary run-level ASR comparison.
- Oracle Tier-3 labels on the test set for evaluating routing counterfactuals.
- Cluster-bootstrap confidence intervals and a pre-registered analysis plan.

### Not doing

- No production deployment and no claims of production safety.
- No tuning on the sealed test set, including threshold, feature, or hyperparameter selection.
- No unvalidated LLM judge as the sole source of attack-success labels.
- No "equivalence" claim inferred merely from a non-significant test.
- No pooling of attack families or benign categories without reporting the constituent results.
- No mixing of guardrail versions, model versions, or inference backends inside a confirmatory comparison.
- No claim that a lower Tier-3 invocation rate is an improvement unless ASR, FPR, and latency are jointly characterized.
- No use of online outcomes to retrain the router during the primary evaluation window.

---

## 1. Critique of the Original Proposals

### 1.1 Innovation 1: Attack-Driven Calibration

**Original claim:** Re-tune Tier-2/Tier-3 thresholds using accumulated red-team transcripts, then validate on a held-out split. Success is better ASR at equal FPR, or better FPR at equal ASR, ideally across several calibration cycles.

**What is strong**

- It uses the project's own attack distribution, which is more relevant than a generic benchmark.
- It treats the defense as an operating-point choice rather than a single pass/fail configuration.
- It can be implemented entirely locally and is compatible with the existing policy-version model.

**What is underspecified or risky**

| Risk                        | Why it matters                                                                                                                                         | Required refinement                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| Temporal leakage            | "Recent runs" used for calibration may overlap the attack distribution being used for evaluation; the attacker may also adapt to the defense over time | Split by attack family and by time; seal test windows before threshold selection            |
| Search overfitting          | Trying many threshold pairs and reporting the best on the same data overstates the effect                                                              | Nested selection; bootstrap upper confidence bounds; held-out evaluation; optimism estimate |
| Confounded baseline         | A static threshold may simply be poorly chosen, so any re-tuning looks good                                                                            | Include a benign-only FPR-matched baseline and a search-matched control                     |
| Attack-family overfitting   | Thresholds may fit the phrasing of the calibration attacks, not the concept of harm                                                                    | Leave-one-attack-family-out and external/OOD evaluation                                     |
| Benign under-representation | FPR is a multi-category property; one aggregate FPR can hide severe over-blocking                                                                      | Stratify by benign category; report worst-group FPR and category-level CIs                  |
| Metric mismatch             | Turn-level thresholds do not directly determine run-level ASR                                                                                          | Primary evaluation uses closed-loop, run-level ASR; turn-level metrics are secondary        |
| Cost neglect                | Lower ASR may be achieved by widening the ambiguity band and calling Tier 3 more often                                                                 | Add Tier-3 invocation rate and latency to the acceptance criteria                           |
| Multiple cycles             | Repeatedly re-tuning until a favorable cycle appears inflates false positives                                                                          | Pre-register the number of cycles and report all cycles, including failures                 |
| Label circularity           | Calibration may use success labels produced by an unreliable judge                                                                                     | Validate labels against human adjudication; freeze label rules                              |
| Adaptive attacker           | An attacker that sees the calibration cycle changes the test distribution                                                                              | Treat adaptive attacks as a separate exploratory arm, not the confirmatory test             |

**Refined claim**

> On sealed held-out attack families and time windows, a calibration procedure selected under a pre-registered FPR budget will reduce run-level ASR at a matched FPR budget relative to a static policy re-selected under the same budget and relative to a benign-only FPR-matched policy, without exceeding the pre-registered Tier-3 budget. The deployed policy is reported alongside these comparisons.

This is deliberately narrower than "calibration improves the defense." It specifies the population, the comparator, the operating point, and the cost boundary.

**Falsifiers**

The claim is not supported if any of the following holds:

- The held-out ASR difference is not favorable at the matched FPR, or its confidence interval includes the noninferiority/superiority boundary.
- The apparent gain disappears under leave-one-family-out or temporal backtesting.
- The calibration reduces ASR only by increasing FPR beyond the margin or by increasing Tier-3 calls beyond the cost budget.
- The gain is no better than the benign-only FPR-matched baseline, meaning attack-driven calibration added no value over threshold placement.
- The calibration is unstable across cycles, with thresholds or decisions changing materially without corresponding performance changes.
- The test set was used for any selection decision.

### 1.2 Innovation 2: Learned Tier-3 Invocation Router

**Original claim:** Replace the fixed Tier-2 threshold that decides when Tier 3 is invoked with a lightweight learned router over Tier-1/Tier-2 features, trained to predict whether Tier 3 would change the verdict. Success is a materially lower Tier-3 invocation rate with ASR and FPR statistically indistinguishable from the static-threshold baseline, within a pre-registered equivalence margin.

**Operational target.** “Would Tier 3 change the verdict?” is not identifiable unless the comparison verdict is specified. Define the label as the frozen Tier-3 terminal verdict or action on the case, and define “change” relative to the router’s fallback action. The router is then trained to reproduce the Tier-3-consistent decision while avoiding calls where the fallback already produces it. This document uses noninferiority for safety, not equivalence, unless a separate equivalence test is explicitly powered.

**What is strong**

- It targets an actual cost driver: conditional Tier-3 calls dominate the slow tail.
- It can use cheap Tier-1/Tier-2 features already computed by the cascade.
- It is directly comparable to a static gate at the same invocation rate.

**What is underspecified or risky**

| Risk                                        | Why it matters                                                                                                                                    | Required refinement                                                                                                                          |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Conflation of routing and classification    | "Skip Tier 3" does not itself produce a verdict for an ambiguous case                                                                             | Define an explicit fallback action: invoke, skip-to-safe, or skip-to-block; evaluate the resulting selective classifier                      |
| Selective labels                            | The target "Tier 3 would change the verdict" is unobserved when Tier 3 is skipped                                                                 | Collect counterfactual Tier-3 labels via exhaustive offline replay or randomized online audits; use inverse-probability weighting for audits |
| Leaky features                              | Dataset source, attack family, turn index, policy version, or label-correlated artifacts can make the router look predictive without generalizing | Freeze a minimal feature list; prohibit identifiers and labels; run leakage and ablation checks                                              |
| Threshold movement masquerading as learning | A static gate can also reduce Tier-3 calls by moving its threshold                                                                                | Add a **matched-invocation static gate** baseline; require lower false-safe-skip with noninferior ASR/FPR at the same call rate              |
| Class imbalance                             | Unsafe cases may be rare in the ambiguity band; accuracy is the wrong metric                                                                      | Report unsafe recall, false-safe-skip rate, selective risk, and lower confidence bounds                                                      |
| Safety validity                             | Lower call rate can hide lost detections                                                                                                          | Co-primary noninferiority on ASR and FPR; safety stop rules; no claim based on invocation reduction alone                                    |
| Distribution shift                          | A deployed router changes which cases receive Tier 3, so future training labels disappear                                                         | Shadow deployment with randomized Tier-3 audit; drift monitoring; retraining under a new frozen version                                      |
| Interaction with calibration                | Calibration changes the score distribution and the ambiguity band, invalidating a router trained on the static regime                             | Train and evaluate routers under each threshold policy; include the 2 × 2 interaction                                                        |
| Ignoring router overhead                    | Router inference may consume the latency saved by skipping Tier 3                                                                                 | Net latency analysis, including router inference and feature computation                                                                     |
| "Bypass" framing                            | The phrase implies skipping a safety layer without accounting for the fallback decision                                                           | Rename to "selective Tier-3 invocation" and document the fallback and loss                                                                   |

**Refined claim**

> On sealed held-out data, a selective Tier-3 invocation policy that uses only pre-Tier-3 features will reduce Tier-3 calls by at least the pre-registered relative margin relative to the static gate, while (a) run-level ASR and FPR are noninferior within pre-registered margins, (b) the false-safe-skip rate is below its pre-registered upper bound, and (c) net p95 latency is noninferior. At a matched invocation rate, it will also have a lower false-safe-skip rate than a static gate, with ASR and FPR noninferior to that gate.

**Falsifiers**

The claim is not supported if any of the following holds:

- The invocation reduction relative to the deployed static gate disappears or falls below the pre-registered margin.
- At matched invocation, the router shows no false-safe-skip advantage over the static gate, or ASR/FPR regress relative to that gate.
- ASR or FPR exceeds the noninferiority margin, even if the invocation rate falls.
- False-safe-skip rate is above the safety bound; this is a stop condition.
- The router relies on features that are unavailable, unstable, or label-correlated in deployment.
- The effect does not survive family, temporal, or benign-category holdouts.
- Net p95 latency is worse because router overhead offsets saved Tier-3 calls.
- The router is trained on Tier-3 labels from the test distribution or on selectively observed labels without correction.

### 1.3 Interaction and Shared Threats

Both innovations change the Tier-2 ambiguity region. Calibration changes the score thresholds; the router changes which ambiguous cases receive Tier 3. Testing them separately can therefore misestimate either effect. The confirmatory design must be factorial:

| Arm | Threshold policy | Tier-3 gate policy       |
| --- | ---------------- | ------------------------ |
| A   | Deployed static  | Static threshold gate    |
| B   | Calibrated       | Static threshold gate    |
| C   | Deployed static  | Learned selective router |
| D   | Calibrated       | Learned selective router |

Shared threats:

- **Same-data reuse:** development, calibration, and test must be disjoint.
- **Multiple comparisons:** many attack families, benign categories, metrics, and thresholds create false-positive risk.
- **Path dependence:** multi-turn policies change future prompts and responses; fixed-transcript replay is not sufficient for run-level ASR.
- **Label noise:** attack success, partial compliance, and benign labels can be ambiguous.
- **Telemetry incompleteness:** missing stage or terminal events can bias both numerator and denominator.
- **Adaptive red-teaming:** attack runs produced against one policy may not represent attacks against another.
- **Version drift:** model updates, embedding updates, classifier updates, or backend changes can invalidate comparisons.
- **Metric gaming:** optimizing a proxy (Tier-3 agreement, accuracy, or call rate) can worsen the actual safety outcome.

The validation architecture below treats these as first-class design constraints rather than post-hoc caveats.

---

## 2. Estimands, Units, and Policy Arms

### 2.1 Units of analysis

| Unit                                      | Used for                                       | Independence concern                         |
| ----------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| Attack case / family                      | Attack sampling, family holdouts               | Prompt phrasing correlation                  |
| Attack run                                | Primary ASR and partial compliance             | Multiple turns per run                       |
| Conversation turn                         | FPR, per-tier block rate, latency              | Repeated turns per conversation              |
| Benign case                               | FPR by category                                | Near-duplicate domain text                   |
| Policy evaluation window                  | Temporal stability                             | Shared model/policy version                  |
| Cluster (family, source, or conversation) | Cluster bootstrap and design-effect adjustment | Observations within a cluster are correlated |

Primary ASR is run-level. Primary FPR is turn-level over benign requests, because a benign request that is blocked is the operational false positive. Report both units where relevant and do not mix their denominators.

### 2.2 Policy definition

A policy is a frozen tuple:

- Guardrail version (Tier-1 rules, Tier-2 index and thresholds, Tier-3 model and templates).
- Tier-3 invocation rule.
- Fallback rule for ambiguous cases when Tier 3 is skipped.
- Inference backend and model identity.
- Router feature list and model parameters, if present.

Only the threshold policy and routing policy vary across the factorial arms. All other elements are held constant.

### 2.3 Notation

Let:

- \(p \in \{A,B,C,D\}\) index the policy arm.
- \(N_A\) be the number of eligible attack runs.
- \(N_B\) be the number of eligible benign turns.
- \(S_i^{(p)} \in \{0,1\}\) indicate that attack run \(i\) succeeds under policy \(p\).
- \(B_j^{(p)} \in \{0,1\}\) indicate that benign turn \(j\) is blocked under policy \(p\).
- \(D_i^{(p)} \in \{0,1\}\) indicate that Tier 3 is invoked for attack turn/run \(i\).
- \(L_i^{(p)}\) be guardrail latency for a turn or run, as specified by the metric being reported.
- \(Y_i^{\mathrm{T3}}\) be the frozen Tier-3 verdict on the audit set.
- \(\alpha\_{alloc}\) be the one-sided alpha allocated to a confirmatory composite after family-wise adjustment.

“Success” of an attack always means satisfaction of the case’s pre-registered attack predicate, not merely that the prompt reached the model or that a guardrail passed. “Blocked” means a terminal `blocked_input` or `blocked_output` route; `partial` interventions are reported separately.

---

## 3. Validation Architecture Step by Step

### Step 0 — Define the estimands and decision

Before collecting data, write down:

- The population (attack families, benign categories, language/domain, model/backend, multi-turn depth).
- The policy arms and the exact contrast for each hypothesis.
- The unit of analysis for each metric.
- Whether the claim is superiority, noninferiority, or equivalence.
- The deployment decision the experiment is meant to support.

**Gate G0:** The estimand, comparator, and decision rule are written and reviewed. If “does calibration help?” has no fixed comparator, stop and refine.

### Step 1 — Freeze data sources and labeling protocol

1. Freeze attack datasets: internal probe families, Tensor Trust-derived cases if used, and any other source. Record versions, seeds, generation parameters, and deduplication rules.
2. Freeze benign datasets by category: security code review, penetration-test write-ups, role-play fiction, and domain technical text. Record source provenance and category labels.
3. Define attack success predicates per case. Deterministic predicates are preferred. If a human label is required, define the rubric, adjudication process, and inter-rater agreement target.
4. Define partial-compliance labels at run level. Do not infer partial compliance from a per-turn verdict.
5. Freeze the Tier-3 label-generation protocol and classifier version for the test set. Generate labels after the policy and analysis freeze via exhaustive offline replay if feasible; otherwise pre-register a randomized audit. Never use test labels for threshold or router selection.
6. Deduplicate exact and near-duplicate prompts across splits using a pre-registered method and threshold.

**Gate G1:** Label agreement is adequate, data splits are disjoint by source/family/time, and the test set has not been inspected for policy selection.

### Step 2 — Replicate the baseline

Run the deployed static cascade (Arm A) with the exact model, embedding index, classifier, router templates, and telemetry configuration. Report:

- ASR and FPR with confidence intervals.
- Tier-2 score distributions.
- Tier-3 invocation rate by direction.
- p50/p95/p99 guardrail, inference, and total latency.
- Completeness ratio and missingness.
- Per-family and per-benign-category breakdowns.

**Gate G2:** Baseline metrics are reproducible from stored data, and telemetry completeness meets the pre-registered threshold. If the baseline is incomplete, no innovation comparison is confirmatory.

### Step 3 — Create the partitions

Use three logical partitions, with additional outer folds:

| Partition           | Purpose                                                            | Access rule                                                               |
| ------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Development `D_dev` | Pipeline implementation, feature engineering, exploratory analysis | May be inspected repeatedly                                               |
| Calibration `D_cal` | Threshold selection, router fitting, operating-point selection     | Used for selection only                                                   |
| Test `D_test`       | Confirmatory evaluation                                            | Sealed until the protocol and analysis are frozen                         |
| Audit `D_audit`     | Randomized or exhaustive Tier-3 labels for skipped cases           | Label protocol frozen; labels generated after freeze or held by custodian |

Outer folds must include:

- **Attack-family holdout:** all runs from one family are absent from development/calibration and appear only in test.
- **Temporal holdout:** calibration data precedes the test window.
- **Benign-category holdout:** at least one benign category is absent from calibration and appears in test.
- **Adaptive-attacker arm:** optional and clearly separate; attacks generated after the defense is frozen, not used in the confirmatory primary analysis.

Random row-level splits are not sufficient and must not be the sole basis for generalization claims.

**Gate G3:** The test set is sealed and access is logged. Any accidental inspection is disclosed and may invalidate the confirmatory claim.

### Step 4 — Develop candidate policies on development data

For Innovation 1:

- Define the threshold search space, including the relationship `ambiguity_floor <= block_at`.
- Define the FPR budget, Tier-3 invocation budget, and minimum per-group coverage.
- Use nested selection: an inner split chooses thresholds; an outer development fold estimates selection stability.
- Pre-register whether the objective is run-level ASR, turn-level ASR, or a constrained combination. Run-level ASR is primary for safety claims.

For Innovation 2:

- Define the action set: `invoke_t3`, `skip_to_safe`, `skip_to_block`.
- Define the fallback rule for each skip action.
- Freeze the feature list to pre-Tier-3 signals: Tier-1 match indicators, Tier-2 similarity, nearest-centroid margin, top-k distances, artifact length, and direction. Prohibit dataset identifiers, attack family, split, labels, run identifiers, and policy-version identifiers.
- Pre-register the model class. Logistic regression is the primary learner for interpretability; tree/boosting models are secondary robustness checks, not a post-hoc upgrade.
- Define class weights or a safety-first loss. Pre-register the maximum acceptable false-safe-skip risk.
- Train only on `D_dev`/`D_cal`; never on `D_test`.

**Gate G4:** Candidate policies can be reproduced from a manifest, and no candidate uses test data or forbidden features.

### Step 5 — Select and freeze the operating point

For calibration:

- Select thresholds that minimize the calibration objective subject to the FPR, invocation, and per-group constraints.
- Use bootstrap upper confidence bounds for FPR rather than the point estimate.
- Record the full search trace, not just the winning threshold, for the optimism analysis.

For the router:

- Fit the model on a development/training partition.
- Calibrate the routing probabilities on a disjoint validation partition.
- Choose the operating point on validation data, not test data.
- Freeze the router coefficients, feature preprocessing, and fallback rules.

**Gate G5:** Operating points, hyperparameters, and thresholds are frozen in a versioned manifest before the test set is unlocked.

### Step 6 — Pre-register the confirmatory protocol

The pre-registration must include every parameter in §4 and must be time-stamped and content-hashed. The analysis code, data manifest, and policy manifests are frozen with it.

**Gate G6:** The registration, manifests, and analysis plan are immutable. Any change creates a new registration and a new test window.

### Step 7 — Run fixed-trajectory replay

Apply each arm to the same stored prompt/response trajectories. This is a **turn-level re-scoring analysis**. It is valid for:

- Per-turn verdict changes.
- Tier-3 invocation counts.
- Guardrail latency and cost.
- Tier-level false positives and detection.
- Sensitivity to counterfactual labels.

It is not, by itself, a valid estimate of run-level ASR under a changed policy, because later prompts and model responses may differ when earlier responses are blocked or altered.

**Gate G7:** Fixed-trajectory results are labeled as such and are not substituted for the confirmatory run-level ASR.

### Step 8 — Run closed-loop policy replay

For each arm, execute the full multi-turn conversation with the frozen model, attack generator, orchestrator policy, and defense policy. Use the same attack seeds across arms where possible, but allow the conversation trajectory to diverge because the defense is part of the environment.

Record:

- Run-level attack success and partial compliance.
- Which turn first succeeded.
- Blocked and partial routes.
- Tier reached and Tier-3 invocation.
- Total and guardrail latency.
- Telemetry completeness.

For nondeterministic backends, pre-register the number of repetitions and cluster by attack case or conversation seed. If multiple model samples are used, define whether success is any-sample or majority-sample before looking at the test set.

**Gate G8:** The primary ASR comparison comes from closed-loop replay under the sealed test protocol.

### Step 9 — Analyze once, then stress-test

Perform the pre-registered primary analysis exactly once. After that, run the pre-registered stability and sensitivity analyses:

- Family and temporal holdouts.
- Benign-category holdouts.
- Calibration-cycle stability.
- Label-noise perturbation.
- Telemetry-missingness bounds.
- Negative controls and oracle bounds.
- Router activation and feature drift.

Label any analysis that was not pre-registered as exploratory.

### Step 10 — Shadow deployment and randomized audit

If the offline result passes, run a shadow deployment:

- The current production policy remains authoritative.
- The candidate policy scores the same live turns without affecting delivery.
- A pre-registered random fraction of router-skipped cases is still sent to Tier 3 for audit.
- Compare shadow decisions, latency, and resource use.

Only after shadow evidence is stable should a live A/B or staged rollout be considered. Online randomization should occur at the conversation/run level, not the turn level, to preserve multi-turn coherence and avoid contamination.

**Gate G9:** No live deployment if false-safe-skip rate, ASR noninferiority, or FPR noninferiority fails in shadow.

### Validation gates summary

| Gate | Requirement                                 | Failure action                   |
| ---- | ------------------------------------------- | -------------------------------- |
| G0   | Estimand and decision rule defined          | Do not collect confirmatory data |
| G1   | Labels and splits valid                     | Redesign data or labeling        |
| G2   | Baseline complete and reproducible          | Fix instrumentation first        |
| G3   | Test set sealed                             | Disclose and restart             |
| G4   | Candidate policies use only allowed data    | Rebuild candidate                |
| G5   | Operating points frozen                     | Restart registration             |
| G6   | Pre-registration immutable                  | New registration                 |
| G7   | Fixed-trajectory analysis labeled correctly | Do not infer run-level ASR       |
| G8   | Closed-loop confirmatory result available   | Do not claim policy effect       |
| G9   | Shadow safety passes                        | Do not deploy                    |

---

## 4. Pre-Registration Protocol

### 4.1 Confirmatory hypotheses

The study has two co-primary innovation claims, H1 and H2. Each is a conjunction tested by intersection–union logic: every required component must pass, and a mechanism component cannot rescue a failed primary component. To control family-wise error across the two claims, test each composite at \(\alpha/2\) (Bonferroni) or use a pre-registered gatekeeping procedure; do not claim both at a full \(\alpha=0.05\) without adjustment. Here \(\alpha\) denotes a one-sided error rate for directional and noninferiority tests. Within a conjunction, the intersection–union test does not require splitting alpha among its components.

**H1 — Attack-driven calibration.** H1 is a conjunction of two required comparisons:

- **H1a (primary):** On sealed held-out attack families and time windows, the calibrated threshold policy (Arm B) reduces run-level ASR at a matched FPR budget relative to a static policy re-selected under the same budget (\(\theta_s\), defined in §6.2), without exceeding the pre-registered Tier-3 budget \(i^_\). The deployed Arm A is reported as the practical baseline; when the deployed point already targets \(f^_\), it can serve as \(\theta_s\), otherwise \(\theta_s\) is a separate comparator.
- **H1b (mechanism):** Arm B also reduces run-level ASR relative to a benign-only policy selected with the same threshold grid, search budget, FPR target, and Tier-3 budget \(i^\*\), and evaluated at matched FPR and cost. This comparison isolates the value of attack labels from the value of moving the threshold or buying more Tier-3 calls.

H1a is the calibration package effect: threshold changes may alter the ambiguity band and therefore Tier-3 cost, as long as the budget is respected. H1b is the label-specific mechanism contrast at matched FPR and cost.

If H1a passes but H1b fails, the result may support a better operating point, but it does not support the distinctive “attack-driven calibration” claim. If H1b passes but H1a fails, the calibrated policy may beat a benign-only control without beating the static policy at the same FPR budget; that is not a deployment claim.

**H2 — Selective Tier-3 invocation.** H2 is a conjunction of two required comparisons and a safety set:

- **H2a (primary):** On sealed held-out data, the learned router (Arm C) reduces Tier-3 invocation by at least the relative margin relative to the static gate (Arm A), with ASR and FPR noninferior within their margins, false-safe-skip rate below its bound, and net p95 latency noninferior.
- **H2b (mechanism):** At a matched invocation rate, Arm C has a lower false-safe-skip rate than a static gate, with ASR and FPR noninferior to that gate. This comparison isolates the value of the learned router from simply moving the static gate.
- **Safety set:** The false-safe-skip bound, selective-risk constraint, and audit coverage must all pass. A call-rate gain alone is not success.

H2a is the overall router package effect relative to the deployed gate; H2b is the mechanism contrast at matched invocation. A router can pass H2a by being a better operating point without proving that learned features add value; H2b is required for the learned-routing claim.

**H3 — Interaction.**

The benefit of calibration persists under learned routing, and the benefit of routing persists under calibrated thresholds. The interaction is reported; a harmful interaction is a failure of the combined policy even if one main effect is positive.

H3 is secondary unless the study is explicitly powered for interaction. The combined policy Arm D can still be evaluated against Arm A as a practical deployment comparison, but that comparison must be labeled as a package effect, not as evidence for either mechanism.

Within each conjunction, apply the same pre-registered component-level decision rule at the alpha allocated to that composite. Report every component, including a passing primary with a failing mechanism check. Secondary families use Holm correction; exploratory analyses are labeled as such.

### 4.2 Candidate pre-registration parameters

The following are **provisional working values** for review. They must be justified against deployment tolerance and frozen before the test unlock. They are not results and must not be presented as tuned.

| Parameter                      | Candidate value or rule                                                        | Notes                                                             |
| ------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Registration ID                | Timestamped, content-hashed identifier                                         | Store with manifests and analysis code                            |
| Target population              | Local single-node deployment, declared model/backend, declared language/domain | No generalization beyond this population                          |
| Primary attack unit            | Attack run                                                                     | Run-level ASR                                                     |
| Primary benign unit            | Benign turn                                                                    | Hard blocked rate                                                 |
| α (confirmatory)               | 0.05 one-sided, family-wise across H1/H2                                       | Bonferroni split α/2 per composite, or pre-registered gatekeeping |
| Power                          | 0.80 minimum                                                                   | Higher preferred for safety claims                                |
| δ_ASR noninferiority           | 0.02 absolute (2 percentage points)                                            | Positive difference means router/calibration worse                |
| δ_FPR noninferiority           | 0.005 absolute (0.5 percentage points)                                         | Must be justified by benign-volume tolerance                      |
| δ_invocation superiority       | 25% relative reduction                                                         | Compared with static gate                                         |
| δ_FPR matching                 | 0.005 absolute                                                                 | For matched-FPR analysis                                          |
| δ_invocation matching          | 0.02 absolute                                                                  | Verify matched call rate for H2b and matched cost for H1b         |
| δ_p95 latency                  | 10 ms absolute                                                                 | Net of router overhead                                            |
| Minimum cycle sign consistency | 2 of 3 pre-registered cycles                                                   | Descriptive requirement for the stability claim                   |
| False-safe-skip bound          | Pre-registered upper bound, e.g. 0.5% of T3-unsafe ambiguous cases             | Safety-critical; exact value must be reviewed                     |
| FPR budget `f*`                | Pre-registered target, e.g. 5%                                                 | Used for calibration selection                                    |
| Tier-3 budget `i*`             | Pre-registered maximum invocation rate                                         | Prevent cost-shifting                                             |
| Calibration cycles `K`         | 3 pre-registered temporal/family cycles                                        | Report every cycle                                                |
| Completeness threshold         | 99.0% of expected critical events                                              | From `ARCHITECTURE.md` §6.6.7                                     |
| Label agreement                | Cohen's κ ≥ 0.80 for subjective labels                                         | Below this, do not run confirmatory inference                     |
| Audit rate `q`                 | 5% of router-skipped cases in shadow/live                                      | Starting point; must be powered for the FSR bound                 |
| Bootstrap replicates           | 10,000 cluster-bootstrap resamples                                             | Pre-register clustering unit                                      |
| Multiplicity                   | Holm–Bonferroni within pre-registered families                                 | H3 secondary family separate                                      |
| Test-set access                | One unlock after freeze; pre-registered analyses only                          | Any other peek is a deviation                                     |
| Stopping rule                  | No early stopping for efficacy                                                 | Safety stop only under pre-registered bound                       |
| Missing data                   | Report bounds; no silent imputation                                            | See §6.6                                                          |
| Deviation log                  | Required                                                                       | Any deviation reported with impact                                |

### 4.3 Data splits and sealing

- The test set is split by attack family, time, and benign category, not by random prompt rows.
- Near-duplicates are removed before splitting, using a pre-registered semantic similarity threshold.
- Development data may be reused for engineering, but not for confirmatory claims after the test is unlocked.
- Calibration may use development data, but the final operating point must be selected without test access.
- Tier-3 oracle labels for test cases are produced after the policy and analysis freeze, under the frozen classifier version, and stored separately. If they must be pre-generated, an independent custodian holds them and all access is logged. They are never used to fit thresholds or routers.
- The attack generator version, seed manifest, model version, embedding index, classifier version, and policy manifest are frozen with the registration.
- Adaptive attacks generated against the calibrated or routed defense are a separate arm. They are not mixed into the primary held-out estimate.

### 4.4 Sample size and power

The final sample size must be calculated from pilot estimates of discordance and intra-cluster correlation. The formulas are in §6.5. The pre-registration should record:

- Expected ASR and FPR under the static policy.
- Expected paired discordance for ASR and FPR.
- Expected false-safe-skip rate and the number of Tier-3-unsafe ambiguous cases available for audit.
- Expected Tier-3 invocation rates for Arms A and C; power for the relative invocation reduction.
- Expected contrast for the benign-only calibration control and for the matched-invocation gate.
- Number of runs per attack family and turns per conversation.
- Cluster design effect and number of independent clusters.
- Minimum detectable effect at the Bonferroni-allocated alpha.
- Whether FPR equivalence is adequately powered; if not, no equivalence claim will be made.
- Whether the H1b and H2b mechanism contrasts are adequately powered; if not, the corresponding composite claim is not confirmable.
- Number of model repetitions per attack run, if the backend is stochastic.
- Whether the test uses fixed-trajectory or closed-loop replay for each endpoint.
- Latency precision target and number of repeated measurements.

A study that cannot power the safety noninferiority test may still produce a useful pilot, but it must not be reported as evidence of equivalence.

### 4.5 Analysis freeze

Before test unlock, freeze:

- The analysis script or executable artifact.
- The metric definitions and denominator rules.
- The list of primary and secondary endpoints.
- The multiple-comparison families.
- The exclusion rules.
- The missing-data handling.
- The bootstrap method and clustering unit.
- The reporting tables and figures.
- The decision rule for each arm.

Any post-unlock analysis is exploratory and must be labeled as such.

---

## 5. Experimental Test Matrix

### 5.1 Factorial design

| Arm | Thresholds      | Tier-3 gate                | Purpose                         |
| --- | --------------- | -------------------------- | ------------------------------- |
| A   | Static deployed | Static ambiguity-band gate | Baseline                        |
| B   | Calibrated      | Static ambiguity-band gate | Isolates calibration            |
| C   | Static deployed | Learned selective router   | Isolates routing                |
| D   | Calibrated      | Learned selective router   | Combined policy and interaction |

For each arm, train the router under that arm’s threshold regime. Do not train one router on static scores and apply it naively to calibrated scores without re-validating the score distribution.

### 5.2 Experiment matrix

| ID  | Experiment                               | Comparison                                                            | Data / unit                             | Primary readout                                                       | Decision use                    |
| --- | ---------------------------------------- | --------------------------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------- | ------------------------------- |
| E1  | Baseline replication                     | A vs stored baseline                                                  | Test; turns and runs                    | ASR, FPR, invocation, latency, completeness                           | Gate G2                         |
| E2  | Fixed-trajectory decision analysis       | A, B, C, D on identical transcripts                                   | Test; turns                             | Verdict flips, invocation, per-turn block rate, latency               | Secondary mechanism check       |
| E3  | Calibration primary                      | B vs θ_s (same-budget static); A reported                             | Sealed closed-loop test; runs           | ΔASR at matched FPR, ΔFPR, Tier-3 rate                                | H1a confirmatory                |
| E4  | Attack-driven vs benign-only calibration | B vs benign-only FPR/cost-matched policy                              | Calibration then sealed test; runs      | ΔASR at matched FPR and cost                                          | H1b mechanism                   |
| E5  | Calibration OOD robustness               | B vs θ_s within each held-out family/time/benign category; A reported | Outer folds; runs and turns             | Worst-group ΔASR, worst-group ΔFPR                                    | Robustness claim                |
| E6  | Calibration cycle stability              | Repeated calibration cycles \(t=1..K\)                                | Temporal/family cycles                  | Mean and dispersion of ΔASR; threshold drift; decision flip rate      | Stability claim                 |
| E7  | Router primary                           | C vs A                                                                | Sealed closed-loop test; runs and turns | Relative invocation reduction, ΔASR, ΔFPR, false-safe-skip rate       | H2a confirmatory                |
| E8  | Router vs matched static gate            | C vs static gate tuned to same invocation rate                        | Calibration then sealed test            | ΔASR, ΔFPR, ΔFSR at matched invocation                                | H2b mechanism                   |
| E9  | Router negative controls                 | C vs random router at same invocation                                 | Sealed test                             | ΔASR, ΔFPR, invocation                                                | Rules out chance call avoidance |
| E10 | Router oracle and ablation               | C vs oracle router vs feature-ablation router                         | Sealed test                             | Invocation gap, unsafe recall, feature contribution                   | Mechanism and upper bound       |
| E11 | Factorial interaction                    | D vs A, plus B−A, C−A, D−C, D−B                                       | Sealed test                             | Difference-in-differences and interaction CI                          | H3                              |
| E12 | Cost and latency                         | A, B, C, D                                                            | Test; turns                             | Tier-3 rate, router overhead, p50/p95/p99 total and guardrail latency | Cost acceptance                 |
| E13 | Missingness and label sensitivity        | All arms under perturbation                                           | Test; runs and turns                    | Worst-case bounds; Δ after label flips; CIs                           | Robustness                      |
| E14 | Shadow deployment and audit              | Candidate vs live policy                                              | Live shadow; conversations              | False-safe-skip, decision agreement, invocation, latency, drift       | Pre-deployment gate             |
| E15 | Adaptive attacker stress                 | Candidate vs attacker aware of policy                                 | Separate exploratory test               | ASR, partial compliance, time-to-success                              | External validity only          |

### 5.3 Required baselines and controls

| Control                                    | Purpose                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| Deployed static policy                     | Practical counterfactual and reproducibility anchor                          |
| Same-FPR-budget static policy \(\theta_s\) | Matched primary comparator for H1a                                           |
| Benign-only FPR-matched policy             | Separates attack-driven calibration from threshold movement                  |
| Search-matched threshold control           | Controls for the degrees of freedom used by threshold search                 |
| Static gate at matched invocation rate     | Separates learned routing from simply moving the Tier-2 gate                 |
| Random router at matched invocation        | Tests whether any call reduction produces the apparent benefit               |
| Oracle router with Tier-3 verdicts         | Ceiling on invocation reduction without changing Tier-3-consistent decisions |
| Feature-ablation router                    | Tests whether the ML model adds value beyond a single feature                |
| Leaky-feature probe                        | Detects dataset-source or attack-family shortcuts                            |
| Test-tuned threshold oracle                | Descriptive upper bound only; never deployed or claimed as a valid policy    |

### 5.4 Replay modes

| Mode                        | What varies                                          | Valid claims                                                                           |
| --------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Fixed-trajectory re-scoring | Defense policy only, with prompts/responses frozen   | Per-turn decisions, invocation, cost, counterfactual T3 labels; not full run-level ASR |
| Closed-loop policy replay   | Defense policy and resulting multi-turn trajectory   | Run-level ASR, partial compliance, first-success depth                                 |
| Shadow deployment           | Candidate policy observes live turns without control | Operational false-skip rate, latency, drift, decision agreement                        |
| Live A/B                    | Conversation/run-level randomized policy assignment  | Causal operational effect, subject to safety review                                    |

The confirmatory ASR claim uses closed-loop policy replay. Fixed-trajectory results are a mechanism and cost analysis, not a substitute.

### 5.5 Required stratification

Report every primary metric by:

- Attack family and attack category.
- Turn depth and first-success depth.
- Benign category.
- Direction (input/output).
- Tier reached and first blocking tier.
- Guardrail version and model/backend version.
- Calibration cycle or temporal window.
- Completeness status.

A pooled number without these strata is insufficient for either innovation. Stratified results are descriptive unless the stratum was pre-registered and adequately powered; do not claim subgroup superiority from an unpowered slice.

### 6. Metric Stability Checks

Stability is not a single test. It is a battery of pre-registered perturbations that asks whether the claimed effect survives reasonable changes in data, labels, and runtime conditions.

| Check                         | Procedure                                                                                | Statistic                                  | Stability rule                                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Attack-family holdout         | Leave out one family at a time; calibrate/train on the rest; evaluate on held-out family | ΔASR, ΔFPR per fold                        | Claim requires no catastrophic fold; report all folds                                                                    |
| Temporal backtest             | Rolling-origin calibration and test windows                                              | ΔASR, ΔFPR, threshold drift, decision flip | Effect must not reverse solely because of time                                                                           |
| Benign-category holdout       | Hold out each benign category in turn                                                    | Category FPR, worst-group FPR              | Aggregate FPR cannot hide a category regression                                                                          |
| Calibration-cycle replication | Repeat threshold selection across pre-registered cycles                                  | Mean, SD, sign consistency                 | A single winning cycle is insufficient                                                                                   |
| Split stability               | Repeated family/group splits where statistically valid                                   | Distribution of ΔASR, ΔFPR                 | Report the interval across splits; no cherry-picking                                                                     |
| Label noise                   | Flip pre-registered fractions of success/benign labels                                   | Range of effects and decision status       | Conclusion must survive plausible noise                                                                                  |
| Telemetry missingness         | Worst-case bounds and complete-case analysis                                             | ASR/FPR lower/upper bounds                 | Inconclusive if decision boundary is inside bounds                                                                       |
| Router audit coverage         | Measure audit fraction in shadow/live                                                    | Audit coverage, IPW interval               | Low coverage invalidates false-skip claims                                                                               |
| Router feature drift          | PSI/KS on features and score distributions                                               | Drift statistic, ECE, Brier                | Drift past threshold triggers investigation, not silent retuning                                                         |
| Router calibration            | Reliability bins on held-out audit labels                                                | ECE, Brier, unsafe recall                  | Probability quality must be sufficient for the chosen operating rule                                                     |
| Latency repeatability         | Randomized run order with warm-up discarded                                              | Bootstrap p50/p95/p99                      | Net latency claim must hold across repetitions                                                                           |
| Negative controls             | Random router, shuffled labels, leaky features                                           | ΔASR, invocation, feature importance       | A control must not reproduce the claimed advantage                                                                       |
| Oracle gap                    | Compare to Tier-3-consistent oracle                                                      | Invocation gap, ASR gap                    | Large gap indicates remaining opportunity; small gap indicates limited headroom under the frozen Tier-3 label definition |
| Inter-rater reliability       | Human/structured partial-compliance labels                                               | Cohen's κ                                  | Below threshold, partial-compliance claim is withheld                                                                    |

### 6.1 Calibration-specific stability

- Report the selected threshold pair for every cycle and the distribution of selected thresholds.
- Report the fraction of cases whose decision flips across cycles.
- Report the overlap of blocked sets at matched ASR.
- Check whether performance gains concentrate in one attack family. If the worst-family result is unfavorable, the claim must be limited to the families where the effect holds.
- Check whether calibration merely shifts the ambiguity band and thereby changes Tier-3 cost. Cost-shifting is not a free safety improvement.

### 6.2 Router-specific stability

- Report invocation rate, unsafe recall, false-safe-skip rate, and selective risk at every operating point on the risk–coverage curve.
- Run a feature-ablation study: Tier-2 similarity only, plus Tier-1 flags, plus margins, plus length. If a single feature reproduces the learned model, the complexity is not justified.
- Run a leakage probe: train a classifier to predict attack family or dataset source from router features. High predictability is a warning that the router may be learning provenance rather than safety.
- Audit skipped cases in shadow deployment. The audit set must be sampled independently of router confidence; otherwise the estimate is biased.
- Check calibration of probabilities if the router uses a threshold on predicted risk. A well-ranked but poorly calibrated score can choose the wrong operating point.

### 6.3 Shared metric stability

- Report metric results separately for input and output directions.
- Report ASR by turn depth and first-success depth; a policy that merely delays success is not equivalent to one that prevents it.
- Report partial compliance with annotation coverage.
- Check that telemetry completeness is above 99% and that missingness bounds do not change the decision.
- Re-run the complete primary analysis on at least one independent data snapshot if resources permit.

---

## . Reporting and Decision Rules

### 7.1 Required reporting table

Every confirmatory report should include:

| Field               | Content                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| Registration        | ID, hash, timestamp, deviations                                         |
| Data                | Dataset versions, split definitions, deduplication rules, sample counts |
| Policies            | Guardrail versions, thresholds, router feature list, fallback rule      |
| Environment         | Model, backend, embedding index, classifier, hardware                   |
| Primary endpoints   | Effect size, interval, margin, test result                              |
| Secondary endpoints | Effect size, interval, multiplicity treatment                           |
| Strata              | Family, depth, direction, benign category, tier                         |
| Cost                | Tier-3 invocation, router overhead, p50/p95/p99                         |
| Data quality        | Completeness, label agreement, audit coverage                           |
| Stability           | Holdouts, cycles, noise, missingness                                    |
| Deviations          | What changed, when, and impact                                          |
| Verdict             | Supported, not supported, or inconclusive                               |

### 7.2 Decision matrix

| Calibration result                       | Router result | Decision                                                                                    |
| ---------------------------------------- | ------------- | ------------------------------------------------------------------------------------------- |
| Passes H1                                | Passes H2     | Consider combined policy after shadow validation                                            |
| Passes H1                                | Fails H2      | Adopt calibration only if operationally justified; keep static gate                         |
| Fails H1                                 | Passes H2     | Adopt router only if it beats matched static gate and safety passes; keep static thresholds |
| Fails H1                                 | Fails H2      | Do not deploy either innovation as claimed                                                  |
| Safety regression in any arm             | Any           | Stop; do not trade safety for call reduction                                                |
| Inconclusive due to power or missingness | Any           | Do not claim; expand or redesign with a new registration                                    |

### 7.3 Stop conditions

Stop or hold the evaluation immediately if:

- The false-safe-skip rate exceeds its pre-registered upper bound.
- The ASR noninferiority upper bound exceeds the margin in the unfavorable direction.
- The FPR noninferiority upper bound exceeds the margin.
- The test set is contaminated or inspected before freeze.
- The router cannot obtain labels for an adequate audit sample.
- Telemetry completeness falls below the threshold and missingness bounds straddle the decision boundary.
- A policy or model version changes during the evaluation window.

### 7.4 What counts as success

For calibration, a credible success requires all of:

- Favorable held-out ΔASR against \(\theta_s\) at the pre-registered FPR operating point.
- No FPR regression beyond margin, including worst-group categories.
- Tier-3 cost within the declared budget.
- Improvement beyond the benign-only FPR-matched baseline.
- Reasonable stability across families, time, and cycles.

For the router, a credible success requires all of:

- Pre-registered relative invocation reduction.
- ASR and FPR noninferiority.
- False-safe-skip and selective-risk bounds satisfied.
- Lower false-safe-skip rate than the matched-invocation static gate, with noninferior ASR and FPR.
- No feature leakage and no reliance on forbidden inputs.
- Net p95 latency noninferior.
- Stable shadow behavior and adequate audit coverage.

For the combined policy, the interaction must be reported and the package effect must be evaluated directly. Two individually positive effects do not guarantee a safe combined policy.

---

## 8. Threats to Validity and Mitigations

| Threat                                | Type              | Mitigation                                                                        |
| ------------------------------------- | ----------------- | --------------------------------------------------------------------------------- |
| Test-set leakage                      | Internal          | Seal test; freeze manifests; log access; one unlock for pre-registered analyses   |
| Temporal drift                        | Internal/external | Temporal holdout; report window; no pooling across policy versions                |
| Attack-family overfitting             | External          | Leave-one-family-out; report worst family; external benchmark as OOD check        |
| Benign-set narrowness                 | External          | Category holdouts; worst-group FPR; domain review                                 |
| Label noise                           | Construct         | Adjudication, κ, sensitivity analysis, human audit                                |
| Selective labels for router           | Internal          | Oracle replay; randomized audits; IPW; audit coverage                             |
| Feature leakage                       | Construct         | Minimal feature allowlist; leakage probe; ablation                                |
| Path dependence in multi-turn attacks | Internal          | Closed-loop replay for confirmatory ASR; fixed-trajectory only for mechanism/cost |
| Model nondeterminism                  | Internal          | Frozen model, deterministic settings where possible, repetition and clustering    |
| Multiple comparisons                  | Statistical       | Pre-registered families; Holm; exploratory labels                                 |
| Underpowered equivalence              | Statistical       | Power calculation; no equivalence claim if not powered                            |
| Missing telemetry                     | Measurement       | Completeness threshold; worst-case bounds; no imputation                          |
| Cost shifting                         | Construct         | Co-report Tier-3 rate and latency; cost budget in calibration                     |
| Adaptive attacker                     | External          | Separate exploratory arm; do not mix into primary                                 |
| Publication bias                      | Reporting         | Pre-registration; report all cycles and endpoints                                 |
| Version drift                         | Internal          | Immutable guardrail/pipeline manifests; archive everything                        |
| Router feedback loop                  | External/internal | Shadow audit; retraining only under new version and new window                    |

---

## 9. MVP Execution Plan and Not Doing

### 9.1 Minimum viable study

1. Freeze one model/backend, one guardrail manifest, and one dataset manifest.
2. Create `D_dev`, `D_cal`, and sealed `D_test` with at least one family and one temporal holdout.
3. Implement and validate the static baseline, including telemetry completeness.
4. Implement the calibration search on `D_cal` with FPR and Tier-3 budgets.
5. Implement a logistic-regression router with an explicit fallback and safety loss.
6. After freezing policies and analysis, generate Tier-3 oracle labels for all ambiguous cases in `D_test` without using them for selection.
7. Run fixed-trajectory replay for all four arms to validate decisions, invocation, and cost.
8. Run closed-loop replay for Arms A, B, C, and D for the primary ASR comparison.
9. Compute cluster-bootstrap intervals, matched-FPR and matched-invocation contrasts, and stability checks.
10. Report the confirmatory result and all pre-registered secondary analyses.

### 9.2 Explicitly not doing in the MVP

- No production deployment.
- No adaptive attacker in the confirmatory arm.
- No LLM-as-judge without human calibration.
- No threshold search on the test set.
- No router feature engineering on test outcomes.
- No multi-model comparison.
- No claim that a lower Tier-3 call rate is safe without ASR/FPR evidence.
- No claim of equivalence from a failed superiority test.
- No pooled result that hides a failing family or benign category.

---

## 10. Relationship to the Architecture

This plan assumes the following contracts from `ARCHITECTURE.md`:

- Run-level ASR and turn-level FPR definitions (§6.6.2 and §6.6.4).
- Tier-3 invocation rate and nearest-rank latency reporting (§6.6.5 and §6.6.6).
- Completeness ratio and provisional-window rules (§6.6.7).
- Immutable guardrail and pipeline versions (§3.5 and §6.6.8).
- Policy-conditioned transcripts and metric snapshots (§6.7).
- Non-blocking telemetry and loss accounting (§5).

The innovations must not weaken these contracts. In particular, a router may not bypass telemetry completeness, a calibration result may not be compared across guardrail versions, and a run-level ASR claim may not be derived from fixed-trajectory replay alone
