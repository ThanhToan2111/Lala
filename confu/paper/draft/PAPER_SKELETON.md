# ConFu++ paper skeleton — v6.1

## Title candidates

1. **When Is Multimodal Interaction Learnable? Identifiability, Task Relevance, and Accessibility in Multimodal Representations**
2. **Beyond Multimodal Fusion: Diagnosing Identifiable and Task-Relevant Cross-Modal Interaction**
3. **Before We Fuse: Diagnosing Predictive Interaction in Multimodal Representations**

Preferred working title: **When Is Multimodal Interaction Learnable?**

## Abstract

See `ABSTRACT_V1.md`. The abstract must state the problem, the four-stage diagnostic hierarchy, controlled evidence, natural evidence, the decoupling result, and the evidence-driven takeaway without claiming SOTA.

## 1. Introduction

See `INTRODUCTION_V1.md`.

Required logical sequence: multimodality does not imply useful interaction; define the conceptual gap; introduce identifiability/task relevance/fidelity/accessibility; summarize the diagnostic machinery; summarize controlled and natural evidence; list contributions.

## 2. Related Work

### 2.1 Multimodal fusion
### 2.2 Higher-order multimodal interaction
### 2.3 Information-theoretic and synergy decomposition
### 2.4 Functional interaction decomposition
### 2.5 Representation identifiability
### 2.6 Multimodal interaction benchmarks

Source matrix: `paper/related_work_matrix.md`. Original ConFu is the closest lineage; DMIL/PID-style work is discussed as a different object from predictive hypothesis-class advantage.

## 3. Framework

### 3.1 Problem setup
Frozen modality representations `r_m = E_m(X_m)` and a source pair `(r_i,r_j)` predicting target representation `r_k`.

### 3.2 Additive and joint predictors
`q_A(r_i,r_j)=q_i(r_i)+q_j(r_j)` versus joint `q_J(r_i,r_j)`.

### 3.3 Joint predictive advantage
`J_hat = R²(q_J,r_k) - R²(q_A,r_k)`. State representation, hypothesis-class, optimizer, and finite-sample dependence.

### 3.4 Task joint headroom
`H_task = Perf(c_J)-Perf(c_A)`. Keep this separate from `J_hat`.

### 3.5 Joint-Advantage Distillation
`d=q_J-q_A`; canonical low-rank multiplicative JAD trained against `d` with cosine loss.

### 3.6 Fidelity and accessibility
Direct/JAD recovery probes and downstream task probes are separate measurements.

### 3.7 Two-gate natural screening
Conceptual `J>0` and `H_task>0`; historical `τ=0.01` is an operational threshold, not a universal significance threshold.

## 4. Formal analysis

### 4.1 Population squared-risk setup
### 4.2 Additive and joint function spaces
### 4.3 Projection-gap proposition
### 4.4 Vector-output extension
### 4.5 Relation to empirical neural fits
### 4.6 Non-claims

Primary artifact: `paper/theory/projection_theory.md`.

## 5. Experiments

### 5.1 Questions and protocol

- Q1: Does `J` detect known non-additive structure?
- Q2: Can JAD represent detected structure?
- Q3: Does detected interaction imply task headroom?
- Q4: Does fidelity imply accessibility?
- Q5: Are natural negative results explained by one restrictive joint architecture?
- Q6: How calibrated is the historical gate?

### 5.2 Controlled synthetic hierarchy
Use existing synthetic oracle and IPIB results as controlled validation.

### 5.3 IPIB
Use known interaction and non-identifiable controls.

### 5.4 Naturalized IPIB
Use canonical MOSEI V/A source geometry, fixed generators, β in `{0, .25, .5, 1}`, five seeds.

### 5.5 Natural two-gate audit
MOSEI, MUStARD, and MELD; report empirical `J`, task headroom, JAD, and interpretation separately.

### 5.6 JAD fidelity and accessibility
Include MOSEI target cosine and sentiment result, plus Naturalized IPIB recovery/accessibility separation.

### 5.7 Hypothesis-class robustness
Include generic MLP caveat and v5.6.1 nested residual audit.

### 5.8 Null calibration
Include v6.0 B=10 stress test and v6.1-C1 confirmation if completed. Never call the historical threshold universal.

## 6. Limitations

Include: no natural Type-IV confirmation; representation/hypothesis-class/optimization dependence; historical threshold calibration limits; finite-replicate nulls; partial JAD recovery; task/probe dependence; no PID or causal interpretation.

## 7. Conclusion

Multimodal interaction learning should be evidence-driven. Non-additive predictive structure must be distinguished from task relevance, fidelity, and accessibility before complex fusion mechanisms are interpreted as useful.

## Main figures

1. Framework hierarchy: `results/figures/v60/fig_framework.pdf`.
2. Naturalized IPIB detection: regenerate from `paper/scripts/fig_naturalized_ipib.py`.
3. Naturalized IPIB recovery: required new paper figure from `jad_recovery.json`.
4. Natural regimes: `results/figures/v60/fig_two_gate.pdf` with missing task headroom left explicit.
5. Fidelity/accessibility separation: `results/figures/v60/fig_fidelity_accessibility.pdf`.

## Main tables

1. Controlled evidence: synthetic / IPIB / Naturalized IPIB.
2. Natural audit: MOSEI / MUStARD / MELD.
3. Hypothesis-class robustness: Product / Generic MLP / Nested residual.
4. Null calibration: observed values, null summaries, and interpretation.

## Supplement

A. Full proof; B. synthetic construction; C. IPIB; D. Naturalized IPIB; E. natural audits; F. task headroom; G. JAD diagnostics; H. hypothesis-class audits; I. calibration; J. closed branches; K. reproducibility.

## Evidence policy

Every numerical claim maps to a JSON artifact, experiment ID, seed list, and configuration in `results/registry.json`. No architecture search, task-aware JAD, new natural dataset search, pooling sweep, rank sweep, or loss sweep is admissible in this version.
