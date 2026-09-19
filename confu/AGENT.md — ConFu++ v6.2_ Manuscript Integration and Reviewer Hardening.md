# AGENT.md

# ConFu++ v6.2

## Manuscript Integration, Reviewer Hardening, and CVPR Submission Freeze

---

# 0. Mission

This repository develops **ConFu++**, a scientific framework for diagnosing multimodal predictive interaction.

The target is:

\[
\boxed{\textbf{CVPR Main Track}}
\]

The research and architecture-development phases are now complete.

From v6.2 onward, the project objective is:

\[
\boxed{
\textbf{
convert the frozen scientific evidence into the strongest truthful, reproducible, reviewer-resistant CVPR submission.
}
}
\]

No additional method-development experiment is permitted.

No additional architecture search is permitted.

No additional dataset search is permitted.

No performance-rescue experiment is permitted.

The remaining work is:

```text
MANUSCRIPT
THEORY VERIFICATION
RELATED WORK
FIGURES
TABLES
SUPPLEMENT
REPRODUCIBILITY
ANONYMIZATION
REVIEWER HARDENING
```

---

# 1. Final Project State

```text
Original ConFu lineage:
ESTABLISHED

Synthetic Oracle hierarchy:
COMPLETE

NIC non-identifiable control:
COMPLETE

IPIB:
COMPLETE

Canonical JAD:
FROZEN

MOSEI exact-source G1:
COMPLETE

MOSEI task-headroom audit:
COMPLETE

MOSEI JAD:
COMPLETE

MOSEI shortcut localization:
COMPLETE

MOSEI purification audit:
CLOSED / FAIL

MUStARD Two-Gate screening:
COMPLETE

MELD Two-Gate screening:
COMPLETE

Generic hypothesis-class audit:
COMPLETE / INCONCLUSIVE

Nested joint-class audit:
COMPLETE / NESTED_HEADROOM_ABSENT

T1 formal projection theory:
COMPLETE

T2 null calibration:
COMPLETE

T3 Naturalized IPIB:
COMPLETE

v6.1 C1 calibration confirmation:
COMPLETE

Architecture phase:
HARD CLOSED

Dataset search:
HARD CLOSED

Experimental phase:
HARD CLOSED

Current phase:
MANUSCRIPT INTEGRATION + REVIEWER HARDENING
```

---

# 2. Final Scientific Thesis

The paper's central claim is:

\[
\boxed{
\textbf{
Multimodal interaction learning should be evidence-driven rather than assumed from modality count.
}
}
\]

Useful interaction requires multiple distinct questions to be answered.

The final diagnostic hierarchy is:

\[
\boxed{\text{Measurement}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{Calibration}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{Task Relevance}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{Representation Fidelity}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{Downstream Accessibility}}.
\]

These stages are empirically distinct.

---

# 3. Stage 1 — Measurement

For source modalities:

\[
r_i,r_j
\]

and target representation:

\[
r_k,
\]

train an additive predictor:

\[
q_A(r_i,r_j)
=
q_i(r_i)+q_j(r_j)
\]

and a joint predictor:

\[
q_J(r_i,r_j).
\]

Define empirical joint predictive advantage:

\[
\boxed{
\hat J_{ij\rightarrow k}
=
R^2(q_J,r_k)
-
R^2(q_A,r_k).
}
\]

This measures predictive advantage under the frozen representation and hypothesis classes.

---

# 4. Stage 2 — Calibration

A positive:

\[
\hat J>0
\]

does NOT automatically establish a calibrated interaction discovery.

The empirical score must be interpreted relative to:

```text
finite sample variation
optimization variation
hypothesis class
representation
an explicit null construction
```

Therefore:

\[
\boxed{
\text{operational predictive advantage}
\neq
\text{calibrated discovery}.
}
\]

---

# 5. Stage 3 — Task Relevance

For downstream task \(Y\), compare:

\[
c_A(r_i,r_j)
\]

against:

\[
c_J(r_i,r_j).
\]

Define task headroom:

\[
\boxed{
H_{\text{task}}
=
Perf(c_J)
-
Perf(c_A).
}
\]

Cross-modal predictive structure need not help the downstream task.

Thus:

\[
\boxed{
\text{predictive advantage}
\nRightarrow
\text{task relevance}.
}
\]

---

# 6. Stage 4 — Representation Fidelity

JAD distills the empirical predictive difference:

\[
\boxed{
d=q_J-q_A.
}
\]

Interaction representation:

\[
u_i=LN(U_ir_i)
\]

\[
u_j=LN(U_jr_j)
\]

\[
\boxed{
h_{ij}
=
W
\left(
\frac{u_i\odot u_j}{\sqrt R}
\right).
}
\]

Canonical:

\[
R=64.
\]

Loss:

\[
\boxed{
L_{JAD}
=
1-\cos(h_{ij},d).
}
\]

Fidelity asks:

> Does \(h_{ij}\) recover the intended predictive interaction target?

---

# 7. Stage 5 — Accessibility

Even a faithful interaction representation may not improve a downstream task.

Let:

\[
Z_{\le1}
=
[r_i,r_j]
\]

and:

\[
Z_{\le2}
=
[r_i,r_j,h_{ij}].
\]

Define:

\[
\boxed{
\Delta_{\text{access}}
=
Perf(Z_{\le2})
-
Perf(Z_{\le1}).
}
\]

Then:

\[
\boxed{
\text{fidelity}
\nRightarrow
\text{accessibility}.
}
\]

---

# 8. Final Conceptual Non-Implications

The paper should make the following distinctions explicit.

### A

\[
\boxed{
\text{Multimodality}
\nRightarrow
\text{Operational Predictive Interaction}
}
\]

Evidence:

```text
MUStARD
MELD
```

under frozen settings.

### B

\[
\boxed{
\text{Operational Predictive Advantage}
\nRightarrow
\text{Calibrated Discovery}
}
\]

Evidence:

```text
MOSEI C1 alignment-null calibration
```

### C

\[
\boxed{
\text{Predictive Advantage}
\nRightarrow
\text{Task Headroom}
}
\]

Evidence:

```text
MOSEI
```

### D

\[
\boxed{
\text{Interaction Fidelity}
\nRightarrow
\text{Downstream Accessibility}
}
\]

Evidence:

```text
IPIB
MOSEI
Naturalized IPIB diagnostics
```

---

# 9. Canonical Terminology

Preferred terms:

```text
joint predictive advantage
predictive interaction
non-additive predictive structure
cross-modal predictive advantage
operational identifiability
representation-conditional identifiability
hypothesis-class-relative identifiability
task joint headroom
interaction-target fidelity
downstream accessibility
alignment-breaking null
```

---

# 10. Forbidden Overclaims

Do NOT claim:

```text
true synergy
PID synergy
pure interaction
causal interaction
new information created by fusion
statistically significant natural interaction
universal discovery threshold
universal identifiability
SOTA multimodal fusion
```

unless separately proven.

---

# 11. Information-Theoretic Constraint

For deterministic fusion:

\[
r_{ij}=F(r_i,r_j),
\]

\[
H(r_{ij}\mid r_i,r_j)=0.
\]

Also:

\[
I(Y;r_{ij}\mid r_i,r_j)=0.
\]

Therefore deterministic fusion does not create new Shannon information.

Do not write otherwise.

---

# 12. Final MOSEI Status

Frozen observed validation:

\[
\boxed{
\hat J_{VA\to T}
=
0.01983.
}
\]

This is stable and positive under the frozen Product-vs-Additive predictor comparison.

However C1 calibration gives:

\[
q_{95}^{null}
=
0.02031
\]

\[
q_{99}^{null}
=
0.02202
\]

and:

\[
\boxed{
\hat p(
J_{null}\ge J_{obs}
)
=
0.09804.
}
\]

Therefore MOSEI must be described as:

\[
\boxed{
\textbf{
operationally positive but not separated from the alignment-breaking null.
}
}
\]

---

# 13. MOSEI Null Interpretation

Do NOT write:

> \(p=0.098\), therefore the interaction is statistically insignificant.

This is not a classical parametric hypothesis test.

Instead write:

> Four of fifty alignment-breaking null replicates exceeded the frozen observed statistic; with the preregistered +1 correction, the empirical exceedance estimate is \(5/51=0.098\).

Interpretation is specific to the chosen operational null.

---

# 14. Historical Threshold

Historical screening threshold:

\[
\boxed{
\tau=0.01.
}
\]

This remains frozen for historical comparability.

It is:

\[
\boxed{
\textbf{an operational progression threshold}
}
\]

not:

\[
\boxed{
\textbf{a calibrated statistical significance threshold}.
}
\]

---

# 15. Why \(\tau=0.01\) Cannot Be Called Calibrated

For the C1 MOSEI null:

\[
\hat p(J_{null}\ge0.01)
=
0.64706.
\]

Therefore many alignment-breaking runs exceed the historical gate.

The paper must state this clearly.

---

# 16. Final Natural Dataset Roles

## MOSEI

Role:

\[
\boxed{
\text{operational-positive / calibration-ambiguous natural stress case}
}
\]

Important facts:

```text
stable positive Product-vs-Additive J
not separated from alignment null
high JAD target fidelity
no stable downstream sentiment accuracy gain
shortcut/selectivity issues
```

---

# 17. MUStARD Role

Role:

\[
\boxed{
\text{small-data negative natural screening case}.
}
\]

All G1 mappings fail under the frozen protocol.

Do not claim MUStARD has no multimodal interaction.

Correct:

> No stable non-additive predictive headroom was detected under the tested pooled representation and predictor classes.

---

# 18. MELD Role

Role:

\[
\boxed{
\text{larger natural negative G1 setting}.
}
\]

Frozen utterance-level screening found no G1 pass.

Nested residual extension also failed to reveal substantial additional headroom.

---

# 19. MELD Nested Robustness

Nested model:

\[
q_N
=
q_P
+
g([r_i,r_j]).
\]

Product is preserved and frozen.

Largest mean incremental MELD headroom:

\[
\Delta_N
\approx0.0038
\]

for:

\[
VT\rightarrow A.
\]

This remains below:

\[
0.01.
\]

Correct interpretation:

> A modest nonlinear residual extension did not expose substantial extra predictive headroom beyond Product Joint.

---

# 20. Generic MLP Audit

Generic MLP experiment is NOT evidence that generic nonlinear prediction is weaker than Product Joint.

Reason:

Generic MLP failed the MOSEI positive control.

Official status:

\[
\boxed{
\text{HYPOTHESIS\_CLASS\_INCONCLUSIVE}.
}
\]

Use only as supporting diagnostic history.

---

# 21. Nested Audit Supersedes Generic MLP as Main Robustness Evidence

The nested audit is stronger because:

\[
q_N=q_P+g.
\]

The Product solution is preserved exactly.

Thus the residual only needs to discover additional structure.

This is the preferred reviewer-facing robustness analysis.

---

# 22. Natural Type-IV Status

No natural setting has established all three:

\[
\boxed{
\text{calibrated predictive evidence}
}
\]

+

\[
\boxed{
\text{stable task headroom}
}
\]

+

\[
\boxed{
\text{positive downstream accessibility}.
}
\]

Therefore:

\[
\boxed{
\textbf{NO NATURAL TYPE-IV CLAIM}
}
\]

for this submission.

---

# 23. Do Not Hide Natural Type-IV Absence

This limitation must appear explicitly in:

```text
Abstract wording
Experiments
Limitations
Reviewer-response preparation
```

Do not search post hoc for a favorable dataset.

---

# 24. Controlled Synthetic Hierarchy

Synthetic experiments establish known-order interaction behavior.

Oracle results show:

```text
S0: first-order
S1: one pairwise interaction
S2: all pairwise interactions
S3: triple interaction
S4: mixed pair/triple structure
```

Explicit known interaction features make corresponding nonlinear structure linearly accessible.

Do not claim generic shallow networks can never learn such structure.

---

# 25. NIC Role

NIC demonstrates:

\[
\boxed{
\text{task interaction can exist without cross-modal predictability}.
}
\]

This motivates separation between:

\[
\text{cross-modal identifiability}
\]

and:

\[
\text{task interaction}.
\]

---

# 26. IPIB Role

IPIB provides controlled identifiable predictive interaction.

It includes:

```text
additive component
joint-predictable component
private component
known task interaction
```

It establishes a controlled environment for JAD analysis.

---

# 27. Naturalized IPIB Role

Naturalized IPIB is the main controlled bridge between:

\[
\text{synthetic ground truth}
\]

and:

\[
\text{natural representation geometry}.
\]

Sources come from canonical MOSEI V/A representations.

Injected target contains:

\[
a(x_1,x_2)
+
\beta j(x_1,x_2)
+
\lambda p.
\]

---

# 28. Naturalized IPIB Frozen Ladder

\[
\beta
\in
\{0,0.25,0.5,1.0\}.
\]

Five seeds.

Frozen interaction rank:

\[
32.
\]

No post-result tuning.

---

# 29. Naturalized IPIB Predictor Results

Validation joint predictive advantage:

\[
\beta=0:
-0.01427
\]

\[
\beta=.25:
0.00262
\]

\[
\beta=.5:
0.05787
\]

\[
\beta=1:
0.22625.
\]

This is one of the strongest controlled results.

---

# 30. Naturalized IPIB Main Claim

Allowed:

> Under frozen natural MOSEI source geometry, increasing known injected interaction strength produces increasing joint predictive advantage.

Do NOT write:

> J universally measures true interaction strength.

---

# 31. Naturalized Direct Recovery

Known injected interaction recovery increases with \(\beta\).

At:

\[
\beta=1,
\]

direct:

\[
d\rightarrow j
\]

recovery reaches approximately:

\[
R^2=0.415.
\]

Thus the empirical difference contains substantial but incomplete ground-truth interaction information.

---

# 32. Naturalized JAD Recovery

At:

\[
\beta=1,
\]

JAD:

\[
h\rightarrow j
\]

recovery reaches approximately:

\[
R^2=0.260.
\]

Correct:

> JAD recovers part of the known interaction structure.

Incorrect:

> JAD recovers the true interaction.

---

# 33. Naturalized Accessibility Caveat

Accessibility task labels are constructed from the interaction latent.

Therefore positive accessibility at:

\[
\beta=0
\]

is not a contradiction.

At \(\beta=0\):

target-side injected interaction is absent.

But task-side interaction relevance remains defined independently.

This demonstrates that target-side interaction and task-side relevance are separate axes.

---

# 34. Figure Rule for Naturalized IPIB

Do NOT place:

\[
J
\]

and:

\[
\Delta_{\text{access}}
\]

on the same semantic axis.

Use separate panels.

---

# 35. T1 Formal Theory

Let:

\[
X=(X_i,X_j)
\]

and:

\[
Y\in\mathbb R^d.
\]

Define:

\[
m(X)=E[Y\mid X].
\]

Work in:

\[
L^2(P_X;\mathbb R^d).
\]

---

# 36. Hilbert Inner Product

\[
\langle f,g\rangle
=
E[
f(X)^\top g(X)
].
\]

Norm:

\[
\|f\|^2
=
E\|f(X)\|_2^2.
\]

---

# 37. Population Risk

\[
\mathcal R(f)
=
E\|Y-f(X)\|_2^2.
\]

The conditional mean:

\[
m(X)
\]

is the unrestricted population minimizer under squared loss.

---

# 38. Additive Space

Define appropriate closed additive space:

\[
\mathcal H_A.
\]

Conceptually:

\[
\mathcal H_A
=
\{
f_i(X_i)+f_j(X_j)
\}.
\]

Use closure where mathematically required.

---

# 39. Joint Space

For the population proposition assume:

\[
\boxed{
\mathcal H_A
\subseteq
\mathcal H_J.
}
\]

This nested theoretical formulation is cleaner than the practical capacity-matched neural comparison.

Do not conflate them.

---

# 40. Population Optima

\[
f_A^*
=
\Pi_A m
\]

\[
f_J^*
=
\Pi_Jm.
\]

---

# 41. Projection Proposition

Under the stated assumptions:

\[
\boxed{
\mathcal R(f_A^*)
-
\mathcal R(f_J^*)
=
\|
\Pi_Jm-\Pi_Am
\|^2.
}
\]

---

# 42. Population \(R^2\)

Define:

\[
V_Y
=
E\|Y-EY\|_2^2.
\]

Then:

\[
R^2(f)
=
1-
\frac{
\mathcal R(f)
}{
V_Y
}.
\]

Thus:

\[
\boxed{
J^*
=
\frac{
\|\Pi_Jm-\Pi_Am\|^2
}{
V_Y
}.
}
\]

---

# 43. Formal Interpretation

Population \(J^*\) measures:

> conditional-mean predictive structure accessible in the richer joint function space but not in the additive projection.

It does NOT identify:

```text
PID atoms
causal effects
semantic interaction
new information
```

---

# 44. Empirical-Theory Gap

Practical models:

\[
\hat q_A,\hat q_J
\]

are finite-trained neural predictors.

Therefore:

\[
\hat J
\]

depends on:

```text
dataset
representation
architecture
capacity
optimization
regularization
sample size
```

The paper must distinguish:

\[
J^*
\]

from:

\[
\hat J.
\]

---

# 45. Final Framework Variables

Operationally:

\[
\boxed{
\hat J
=
\hat J(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J,
\mathcal O
).
}
\]

Where:

```text
D = data distribution
φ = representation
H_A = additive hypothesis class
H_J = joint hypothesis class
O = optimization/training procedure
```

---

# 46. Final Contribution 1

## Formal Predictive Diagnostic

Define joint predictive advantage and provide a population projection-gap interpretation under squared loss and nested function-space assumptions.

---

# 47. Final Contribution 2

## Diagnostic Hierarchy

Separate:

```text
operational measurement
null calibration
task relevance
representation fidelity
downstream accessibility
```

and show these need not imply one another.

---

# 48. Final Contribution 3

## Controlled Interaction Benchmarks

Introduce/use:

```text
IPIB
Naturalized IPIB
```

to study known interaction structure under synthetic and natural representation geometry.

---

# 49. Final Contribution 4

## Natural Stress Tests

Audit:

```text
MOSEI
MUStARD
MELD
```

and show that neither multimodality nor an operationally positive predictive score alone establishes calibrated, task-useful interaction.

---

# 50. Paper Identity

The paper is:

\[
\boxed{
\textbf{a scientific diagnostic framework paper}
}
\]

with JAD as a concrete representation mechanism.

It is NOT primarily:

\[
\boxed{
\text{a fusion-performance paper}.
}
\]

---

# 51. Preferred Title Candidate A

**When Is Multimodal Interaction Learnable? Identifiability, Calibration, Task Relevance, and Accessibility**

---

# 52. Preferred Title Candidate B

**Beyond Multimodal Fusion: Diagnosing Predictive Interaction, Task Relevance, and Accessibility**

---

# 53. Preferred Title Candidate C

**Before We Fuse: Diagnosing Predictive Interaction in Multimodal Representations**

Final title depends on final related-work differentiation.

---

# 54. Abstract Required Structure

The Abstract should follow:

```text
1. Problem
2. Missing scientific question
3. Diagnostic hierarchy
4. Formal score
5. Controlled evidence
6. Natural evidence
7. Main result
8. Takeaway
```

---

# 55. Abstract Problem Sentence

Desired idea:

> Multimodal models increasingly employ sophisticated fusion mechanisms, yet multiple modalities alone do not establish that useful non-additive interaction is present.

Do not begin with architecture details.

---

# 56. Abstract Framework Sentence

Explain that the paper separates:

```text
measurement
calibration
task relevance
fidelity
accessibility
```

---

# 57. Abstract Controlled Result

Mention Naturalized IPIB:

> Under natural source-representation geometry with known injected interaction, joint predictive advantage increases with interaction strength while interaction recovery remains incomplete.

---

# 58. Abstract Natural Result

Use cautious language:

> Natural experiments on MOSEI, MUStARD, and MELD reveal that positive-looking predictive structure does not consistently translate into calibrated evidence or downstream task gains.

---

# 59. Abstract Takeaway

Preferred:

> These results suggest that multimodal interaction learning should be evidence-driven rather than assumed from modality count or fusion complexity.

---

# 60. Introduction Structure

Six-paragraph structure:

```text
P1 — multimodal fusion motivation
P2 — missing question
P3 — hierarchy
P4 — operational framework
P5 — evidence
P6 — contributions
```

---

# 61. Introduction P1

Discuss growth of increasingly complex multimodal fusion.

Avoid claiming existing methods are conceptually wrong.

Instead:

> They often focus on how to model interaction rather than whether interaction is demonstrably present and useful in a given representation/task setting.

---

# 62. Introduction P2

Key question:

\[
\boxed{
\text{Should interaction be learned at all in this setting?}
}
\]

This question precedes architecture design.

---

# 63. Introduction P3

Introduce the five-stage hierarchy:

\[
M\rightarrow C\rightarrow T\rightarrow F\rightarrow A
\]

where:

```text
M = measurement
C = calibration
T = task relevance
F = fidelity
A = accessibility
```

---

# 64. Introduction P4

Introduce:

```text
Additive predictor
Joint predictor
J
Task headroom
JAD
Fidelity probes
Accessibility probes
Null calibration
```

---

# 65. Introduction P5

Evidence summary:

```text
Synthetic hierarchy
IPIB
Naturalized IPIB
MOSEI
MUStARD
MELD
Nested robustness
Calibration confirmation
```

---

# 66. Introduction P6

Exactly four contributions.

Do not list eight mini-contributions.

---

# 67. Related Work Structure

Recommended subsections:

```text
Multimodal Fusion

Higher-Order Multimodal Alignment

Information-Theoretic Decomposition

Functional / Statistical Interaction Analysis

Multimodal Synergy

Representation Identifiability

Diagnostic and Controlled Multimodal Benchmarks
```

---

# 68. Mandatory Related-Work Matrix

Maintain:

```text
paper/related_work_matrix.md
```

Columns:

```text
Paper
Year
Venue
Interaction definition
Task supervision
Pair/higher-order
Information theoretic?
Explicit interaction representation?
Synthetic ground truth?
Natural benchmark?
Identifiability analysis?
Primary objective
Difference from ours
```

---

# 69. Original ConFu Positioning

ConFu is direct lineage.

Do not frame ConFu as flawed.

Distinction:

ConFu asks how higher-order multimodal alignment can be learned.

Current paper asks:

\[
\boxed{
\text{when interaction learning is empirically justified}.
}
\]

---

# 70. DMIL/PID Positioning

Information-theoretic methods decompose information quantities or synergy-related components.

Our score is:

\[
\boxed{
\text{hypothesis-class-relative predictive advantage}.
}
\]

It is not PID.

---

# 71. Functional ANOVA Positioning

Functional decomposition literature may define interaction components mathematically.

Our focus differs:

```text
multimodal representation spaces
learned predictor classes
task relevance
distillation
accessibility
```

Do not claim functional decomposition is new.

---

# 72. Method Section Structure

Recommended:

```text
3.1 Problem Setup
3.2 Joint Predictive Advantage
3.3 Calibration
3.4 Task Joint Headroom
3.5 Joint-Advantage Distillation
3.6 Fidelity and Accessibility
3.7 Diagnostic Protocol
```

---

# 73. Theory Section Structure

Recommended:

```text
4.1 Population Squared-Risk Setup
4.2 Additive and Joint Spaces
4.3 Projection-Gap Proposition
4.4 Vector-Valued Extension
4.5 Empirical Approximation
4.6 Interpretation and Non-Claims
```

---

# 74. Experiment Section Structure

Recommended:

```text
5.1 Experimental Questions
5.2 Controlled Synthetic Hierarchy
5.3 IPIB
5.4 Naturalized IPIB
5.5 Natural Multimodal Stress Tests
5.6 Fidelity and Accessibility
5.7 Hypothesis-Class Robustness
5.8 Null Calibration
```

---

# 75. Experimental Question Q1

> Does joint predictive advantage respond to known non-additive predictive structure?

Evidence:

```text
Synthetic
IPIB
Naturalized IPIB
```

---

# 76. Experimental Question Q2

> Can the predictive difference be faithfully represented?

Evidence:

```text
IPIB
Naturalized IPIB
MOSEI JAD
```

---

# 77. Experimental Question Q3

> Does positive predictive advantage imply downstream task headroom?

Evidence:

```text
MOSEI
```

Answer:

\[
\boxed{
\text{No guarantee.}
}
\]

---

# 78. Experimental Question Q4

> Does fidelity imply accessibility?

Evidence:

```text
MOSEI
controlled experiments
```

Answer:

\[
\boxed{
\text{No guarantee.}
}
\]

---

# 79. Experimental Question Q5

> Could natural negative results arise only because Product Joint is restrictive?

Evidence:

```text
Generic MLP audit
Nested residual audit
```

Nested audit is primary.

---

# 80. Experimental Question Q6

> Is the operational threshold statistically calibrated?

Evidence:

```text
v6.0 null audit
v6.1 C1 B=50
```

Answer:

\[
\boxed{
\text{Not on MOSEI under the alignment-breaking null.}
}
\]

---

# 81. Main Figure 1 — Final Diagnostic Funnel

The final framework figure should show:

```text
Multimodal Representations
          |
          v
Additive vs Joint Prediction
          |
          v
Operational Predictive Advantage
          |
          v
Alignment / Null Calibration
          |
          v
Task Joint Headroom
          |
          v
Interaction Distillation
          |
          v
Representation Fidelity
          |
          v
Downstream Accessibility
```

---

# 82. Figure 1 Visual Principle

Use broken implication markers between stages where supported.

Do not imply each stage automatically follows from the previous one.

---

# 83. Main Figure 2 — Naturalized Detection

Plot:

\[
\beta
\]

against:

\[
\hat J.
\]

Show five-seed mean and uncertainty.

This is the main construct-validity figure.

---

# 84. Main Figure 3 — Recovery

Plot:

\[
\beta
\]

against:

```text
Direct d→ground-truth R²
JAD h→ground-truth R²
```

This demonstrates incomplete but increasing fidelity.

---

# 85. Main Figure 4 — Natural Stress Cases

Preferred design:

```text
MOSEI
MUStARD
MELD
```

showing separate columns/panels for:

```text
operational J
calibration status
task headroom
JAD/accessibility
```

Do not collapse into one score.

---

# 86. Main Table 1 — Controlled Benchmarks

Columns:

```text
Benchmark
Natural source geometry?
Known interaction?
Operational J?
Ground-truth recovery?
JAD recovery?
Task accessibility?
```

Rows:

```text
Synthetic Oracle
IPIB
Naturalized IPIB
```

---

# 87. Main Table 2 — Natural Audit

Columns:

```text
Dataset
Source pair
Operational J
Calibration
Task headroom
JAD fidelity
Accessibility
Interpretation
```

---

# 88. Main Table 3 — Hypothesis-Class Audit

Columns:

```text
Dataset
Mapping
Product J
Generic MLP J
Nested Δ
Interpretation
```

Clearly mark Generic MLP as:

```text
positive-control failure / inconclusive
```

---

# 89. Main Table 4 — Calibration

Recommended compact table:

```text
Setting
Observed J
Null mean
q95
q99
Empirical exceedance
Interpretation
```

MOSEI C1 should be the primary row.

---

# 90. Do Not Overemphasize q99

With:

\[
B=50,
\]

q99 remains a finite empirical tail summary.

The more interpretable statistic is:

\[
\hat p_{obs}=0.09804.
\]

Report both.

---

# 91. Why No More Calibration Runs

B=50 is sufficient to establish the qualitative conclusion:

\[
\boxed{
J_{obs}
\text{ is not clearly separated from this null}.
}
\]

Further B expansion would refine precision but is not expected to change the paper's qualitative claim.

Therefore:

\[
\boxed{
\textbf{NO MORE CALIBRATION EXPERIMENTS}.
}
\]

---

# 92. Experimental Phase Final Status

\[
\boxed{
\textbf{PERMANENTLY CLOSED FOR THIS SUBMISSION}
}
\]

No:

```text
v6.3 experiment
v7 method
new dataset
new task
new architecture
new threshold
new representation
```

---

# 93. Permitted Work From Now On

Only:

```text
LaTeX integration
writing
citation checking
proof checking
figure refinement
table refinement
supplement
reproducibility
repository cleanup
anonymization
reviewer simulation
```

---

# 94. Prohibited Work

```text
NO new training experiment
NO architecture modification
NO hyperparameter sweep
NO dataset search
NO task reformulation
NO threshold replacement
NO post-hoc subgroup search
NO representation change
NO JAD modification
NO natural Type-IV fishing
```

---

# 95. Paper Skeleton

Required file:

```text
paper/draft/PAPER_SKELETON.md
```

Must map every paper section to:

```text
claim
evidence
figure/table
artifact
limitation
```

---

# 96. Claims Matrix

Required:

```text
paper/claims.md
```

Columns:

```text
Claim
Evidence
Assumptions
Counter-evidence
Allowed wording
Forbidden wording
Limitations
Status
```

---

# 97. Claim 1

> Joint predictive advantage tracks increasing injected interaction strength in Naturalized IPIB.

Status:

\[
\boxed{
SUPPORTED.
}
\]

---

# 98. Claim 2

> A positive operational joint predictive advantage guarantees calibrated interaction evidence.

Status:

\[
\boxed{
REFUTED.
}
\]

MOSEI calibration demonstrates the distinction.

---

# 99. Claim 3

> A positive predictive advantage guarantees downstream task gain.

Status:

\[
\boxed{
REFUTED.
}
\]

---

# 100. Claim 4

> High interaction-target fidelity guarantees accessibility.

Status:

\[
\boxed{
REFUTED.
}
\]

---

# 101. Claim 5

> Product Joint is universally sufficient.

Status:

\[
\boxed{
NOT CLAIMED.
}
\]

Nested audit supports only a narrow robustness conclusion.

---

# 102. Claim 6

> JAD completely recovers known injected interaction.

Status:

\[
\boxed{
REFUTED.
}
\]

Naturalized IPIB shows partial recovery.

---

# 103. Claim 7

> \(\tau=0.01\) is a calibrated false-positive threshold.

Status:

\[
\boxed{
REFUTED.
}
\]

---

# 104. Reviewer Questions Matrix

Maintain:

```text
paper/reviewer_questions.md
```

Priority questions:

```text
Q1 Why call this interaction?
Q2 Why not PID synergy?
Q3 Why another-modality prediction?
Q4 Why no natural Type IV?
Q5 Why CVPR without SOTA gain?
Q6 Why does MOSEI overlap the null?
Q7 Is the null appropriate?
Q8 Could representation erase interaction?
Q9 Could Product Joint be too weak?
Q10 Why not use Transformers?
Q11 What does Naturalized IPIB add?
Q12 Why is accessibility task-dependent?
Q13 What exactly does the theorem prove?
Q14 How does this differ from ConFu?
Q15 How does this differ from DMIL/PID work?
```

---

# 105. Reviewer Defense — Why Call It Interaction?

Answer:

The paper uses:

\[
\boxed{
\text{predictive interaction}
}
\]

to mean predictive structure available under joint computation beyond an additive reference class.

It is explicitly hypothesis-class-relative.

It is not claimed to be PID synergy or causality.

---

# 106. Reviewer Defense — Why Predict Another Modality?

Cross-modal prediction provides a label-free operational target.

It asks whether:

\[
(r_i,r_j)
\]

jointly expose structure in:

\[
r_k
\]

beyond independent additive contributions.

This is different from downstream task interaction.

---

# 107. Reviewer Defense — Why No Natural Type IV?

Answer directly:

> We did not find a natural setting satisfying calibrated predictive evidence, stable task headroom, and downstream accessibility simultaneously.

Then:

> Rather than continue benchmark search until a favorable case appeared, we freeze this as a limitation and validate the positive mechanism using controlled Naturalized IPIB.

---

# 108. Reviewer Defense — Why CVPR Without Performance Gain?

Focus on:

```text
multimodal representation diagnosis
controlled benchmark
formalization
failure modes
interaction validity
```

The paper answers:

> when complex multimodal interaction modeling is actually supported by evidence.

Do not answer merely:

> negative results are valuable.

---

# 109. Reviewer Defense — MOSEI Calibration

State:

> MOSEI is operationally positive under the frozen predictor comparison but does not separate from the preregistered alignment-breaking null.

This distinction is a central result.

---

# 110. Reviewer Defense — Null Limitation

The alignment-breaking shuffle:

```text
breaks source alignment
does not preserve every lower-order dependency
is operational rather than universal
```

Therefore it is a stress-test null, not a unique mathematically canonical null.

State this explicitly.

---

# 111. Reviewer Defense — Product Weakness

Use nested audit:

\[
q_N=q_P+g.
\]

Product is frozen and preserved.

A modest residual does not expose substantial MELD headroom.

Limit claim to this tested extension.

---

# 112. Reviewer Defense — Representation Dependence

Representation dependence is acknowledged.

The score is:

\[
\hat J(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J,
\mathcal O
).
\]

The paper does not claim dataset-intrinsic universal interaction.

---

# 113. Limitations Section

Must explicitly state:

```text
No natural Type-IV setting
MOSEI operational signal does not separate from the null
Historical τ=0.01 is not calibrated
Alignment-breaking null is operational
Representation dependence
Hypothesis-class dependence
Optimization dependence
Partial JAD recovery
Accessibility is task/probe dependent
No causal claim
No PID claim
No information-creation claim
```

---

# 114. Limitations Philosophy

Each limitation should answer:

> What does this prevent us from claiming?

Not:

> Why our paper is bad.

---

# 115. Supplement Structure

Recommended:

```text
A Formal Proof
B Synthetic Hierarchy
C IPIB
D Naturalized IPIB
E Natural Data Audits
F Task Headroom
G JAD Diagnostics
H Shortcut and Purification Audits
I Hypothesis-Class Audits
J Nested Residual Audit
K Calibration
L Closed Branches
M Reproducibility
```

---

# 116. Main Paper Exclusions

Move to supplement:

```text
all loss-tuning variants
purification experiments
reliability weighting
full Generic MLP diagnostics
full data audits
all per-seed historical tables
development chronology
```

---

# 117. Main Paper Inclusions

Must stay:

```text
framework
J definition
projection theorem
Naturalized IPIB
natural stress-test table
MOSEI calibration result
MOSEI fidelity/accessibility separation
nested robustness
limitations
```

---

# 118. Reproducibility Source of Truth

Maintain:

```text
results/registry.json
```

Every reported number must link to:

```text
experiment ID
artifact
config
seeds
commit
```

---

# 119. No Manual Number Drift

No table number should be copied manually without checking machine-readable artifact.

Final manuscript audit must compare:

```text
LaTeX numbers
vs
registry/artifact numbers
```

---

# 120. Figure Scripts

All figures must remain script-generated.

Required categories:

```text
framework
naturalized detection
naturalized recovery
natural stress cases
calibration
```

---

# 121. Figure Quality Rules

Every main figure must satisfy:

```text
readable at single-column width
no tiny text
consistent terminology
uncertainty clearly identified
no unsupported causal arrows
color not required to distinguish meaning
```

---

# 122. Theory Verification

Independently re-derive:

\[
\mathcal R(f_A^*)-\mathcal R(f_J^*)
=
\|\Pi_Jm-\Pi_Am\|^2.
\]

Verify:

```text
closedness
nestedness
orthogonality
vector-valued setting
finite second moment
common R² denominator
```

If a condition is questionable:

weaken theorem.

Do not force elegance over correctness.

---

# 123. Citation Audit

Before submission manually verify every important related-work statement.

Particularly:

```text
ConFu
DMIL
PID/synergy methods
functional ANOVA interaction work
multimodal fusion methods
representation identifiability work
```

No citation should support a stronger claim than the source actually makes.

---

# 124. Novelty Audit

Before final submission ask:

> If a reviewer removes JAD, what remains novel?

Desired answer:

```text
formal predictive diagnostic
calibration distinction
multi-stage interaction hierarchy
IPIB/Naturalized IPIB
natural stress-test evidence
```

The paper must not depend entirely on JAD novelty.

---

# 125. Abstract Kill Test

A reader should understand:

```text
problem
diagnostic stages
controlled benchmark
natural finding
main limitation
takeaway
```

without reading the paper.

---

# 126. Introduction Kill Test

After the Introduction, reviewer must know:

```text
what interaction means here
why measurement is needed
why calibration is separate
why task relevance is separate
what the paper contributes
```

---

# 127. Figure Kill Test

Figure 1 should communicate:

\[
\boxed{
\text{interaction evidence is a funnel, not a single score}.
}
\]

---

# 128. Naturalized Figure Kill Test

Without caption details, reviewer should see:

\[
\beta\uparrow
\Rightarrow
J\uparrow
\]

and increasing but incomplete recovery.

---

# 129. Table Kill Test

No table may merge:

```text
J
calibration
task headroom
fidelity
accessibility
```

into one composite result.

Keep stages separate.

---

# 130. Manuscript Tone

Preferred tone:

```text
precise
scientific
conservative
mechanistic
```

Avoid:

```text
revolutionary
breakthrough
surprisingly
proves universally
fundamentally solves
```

unless genuinely justified.

---

# 131. Anonymization

Before submission inspect:

```text
author names
lab names
institution names
personal paths
GitHub usernames
repository URLs
W&B entities
machine names
acknowledgments
PDF metadata
git metadata exposed in supplement
```

Remove identifying information according to CVPR rules.

---

# 132. Local Paths

Do NOT expose paths such as:

```text
/home/linhkastner/...
```

inside the anonymous manuscript or supplementary material.

Convert to generic repository-relative paths.

---

# 133. Anonymous Repository

If an anonymous code repository is provided:

```text
remove commit author identities
remove issue history revealing identity
remove personal config files
remove cloud credentials
remove logs containing usernames
remove tracking links
```

---

# 134. Secret Audit

Before repository release scan for:

```text
API keys
tokens
credentials
W&B keys
HuggingFace tokens
cloud credentials
SSH material
personal emails
```

Never include them.

---

# 135. Submission Readiness States

Use:

```text
EXPERIMENTALLY_READY
MANUSCRIPT_IN_PROGRESS
REVIEWER_HARDENED
SUBMISSION_READY
```

Current state:

\[
\boxed{
\textbf{EXPERIMENTALLY\_READY}
}
\]

and:

\[
\boxed{
\textbf{MANUSCRIPT\_IN\_PROGRESS}.
}
\]

---

# 136. Reviewer Hardening Pass 1

Run an internal review pretending score:

```text
Weak Reject
```

Ask:

> What is the strongest reason to reject this paper?

Likely answers:

```text
no natural Type IV
limited practical gain
diagnostic score not uniquely interaction
null construction imperfect
closest-work overlap
```

Prepare explicit answers.

---

# 137. Reviewer Hardening Pass 2

Pretend reviewer says:

> This is just additive vs nonlinear prediction.

Response must explain:

```text
formal object
controlled injection results
task-relevance separation
fidelity/accessibility diagnostics
calibration stage
```

---

# 138. Reviewer Hardening Pass 3

Pretend reviewer says:

> Natural results are mostly negative.

Response:

> The paper's question is not whether one architecture wins, but whether interaction modeling is justified. The natural experiments reveal that this prerequisite frequently fails or remains ambiguous.

Support with controlled positive benchmark.

---

# 139. Reviewer Hardening Pass 4

Pretend reviewer says:

> Why should CVPR care?

Answer around:

```text
multimodal representation learning
fusion architecture interpretation
avoiding unsupported interaction claims
controlled multimodal evaluation
diagnostic methodology
```

---

# 140. Reviewer Hardening Pass 5

Pretend reviewer says:

> Your MOSEI positive result disappears under calibration.

Correct response:

> That is precisely why measurement and calibration are separate stages in the final framework.

Do not defend the old interpretation.

---

# 141. Paper Narrative Rule

Never write the chronological story:

```text
we tried AV-MNIST
then MOSI
then UR-FUNNY
then ...
```

Instead organize by scientific questions.

---

# 142. Final Narrative

Preferred flow:

```text
1. Complex multimodal interaction is often assumed.

2. Define operational predictive advantage.

3. Formalize its population interpretation.

4. Show controlled construct validity.

5. Separate calibration from measurement.

6. Separate task relevance from predictive advantage.

7. Separate fidelity from accessibility.

8. Stress-test on natural datasets.

9. Conclude interaction learning should be evidence-driven.
```

---

# 143. Final Main Message

\[
\boxed{
\textbf{
A multimodal model should not be interpreted as exploiting meaningful interaction merely because a complex fusion mechanism is present.
}
}
\]

Instead evaluate:

\[
\boxed{
\text{measurement}
\rightarrow
\text{calibration}
\rightarrow
\text{relevance}
\rightarrow
\text{fidelity}
\rightarrow
\text{accessibility}.
}
\]

---

# 144. Final Natural Result Wording

Preferred:

> Across MOSEI, MUStARD, and MELD, we do not find evidence that multimodality alone reliably implies calibrated, task-useful non-additive interaction under the frozen representations and hypothesis classes.

---

# 145. Final Controlled Result Wording

Preferred:

> Under Naturalized IPIB, where interaction strength is known by construction while source geometry is inherited from natural MOSEI representations, joint predictive advantage increases with injected interaction strength and interaction recovery improves, though remains incomplete.

---

# 146. Final Calibration Wording

Preferred:

> A preregistered 50-replicate alignment-breaking calibration on MOSEI yields an empirical exceedance estimate of 0.098 for the observed score, showing that the frozen operational positive is not clearly separated from this null.

---

# 147. Final Fidelity Wording

Preferred:

> High agreement with the predictive interaction target does not guarantee downstream task improvement.

---

# 148. Final JAD Wording

Preferred:

> JAD is an operational mechanism for representing the predictive difference exposed by joint relative to additive models.

---

# 149. Final Non-Claim

Do not say:

> JAD discovers true multimodal synergy.

---

# 150. Final Contribution Bullets

Use exactly four.

### 1.

A formal hypothesis-class-relative diagnostic for non-additive multimodal predictive advantage.

### 2.

A diagnostic hierarchy separating measurement, calibration, task relevance, fidelity, and accessibility.

### 3.

Controlled IPIB and Naturalized IPIB benchmarks for interaction detection and recovery under known structure.

### 4.

Natural stress tests across MOSEI, MUStARD, and MELD exposing the gap between multimodality, predictive advantage, calibrated evidence, and task utility.

---

# 151. Submission Checklist

Before final submission:

```text
[ ] Full LaTeX draft complete
[ ] Abstract frozen
[ ] Introduction frozen
[ ] Related Work audited
[ ] Theory proof independently checked
[ ] Method notation consistent
[ ] Main figures regenerated
[ ] Main tables regenerated
[ ] Naturalized IPIB numbers verified
[ ] MOSEI C1 numbers verified
[ ] Natural benchmark numbers verified
[ ] Claims matrix complete
[ ] Reviewer matrix complete
[ ] Limitations explicit
[ ] Supplement complete
[ ] Repository anonymized
[ ] Personal paths removed
[ ] Secrets scanned
[ ] PDF metadata anonymized
[ ] Citation compilation clean
[ ] No undefined references
[ ] No test-driven selection claim
[ ] No unsupported novelty claim
```

---

# 152. Final Experiment Rule

\[
\boxed{
\textbf{NO MORE EXPERIMENTS.}
}
\]

This includes:

```text
no architecture
no dataset
no task
no null extension
no B=100
no new baseline training unless strictly required by an already frozen paper comparison and approved before seeing results
```

Default action is always:

\[
\boxed{
\text{write / verify / clarify}
}
\]

not:

\[
\boxed{
\text{train}.
}
\]

---

# 153. Exception Rule

A new computation is allowed only if it is:

```text
bug correction
table regeneration
figure regeneration
artifact verification
reproducibility validation
```

and does NOT alter the scientific hypothesis or model.

Any result-changing fix must be documented explicitly.

---

# 154. Bug Discovery Rule

If a material bug is discovered:

1. stop manuscript integration for affected result;
2. document bug;
3. identify all affected artifacts;
4. rerun only affected frozen protocol;
5. update registry;
6. update claims;
7. do not use the bug as justification for architecture changes.

---

# 155. Final Evidence Freeze

The experimental evidence freeze consists of:

```text
Synthetic Oracle
NIC
IPIB
MOSEI
MUStARD
MELD
v5.6 Generic audit
v5.6.1 Nested audit
v6.0 T1
v6.0 T2
v6.0 Naturalized IPIB
v6.1 C1 calibration
```

No new scientific branch enters the submission.

---

# 156. Final Paper Files

Recommended structure:

```text
paper/
    main.tex
    sections/
        introduction.tex
        related_work.tex
        method.tex
        theory.tex
        experiments.tex
        limitations.tex
        conclusion.tex

    figures/
    tables/
    scripts/

    supplement/
        supplement.tex

    claims.md
    reviewer_questions.md
    related_work_matrix.md

    draft/
        PAPER_SKELETON.md
        ABSTRACT_V1.md
        INTRODUCTION_V1.md
        SUBMISSION_READINESS.md
```

---

# 157. Final Repository Structure

Keep final experiment artifacts under:

```text
results/
    synthetic/
    ipib/
    mosei/
    mustard/
    meld/
    hypothesis_class/
    calibration/
    naturalized_ipib/
    registry.json
```

No manual result copies.

---

# 158. Immediate Next Action

The next action is:

\[
\boxed{
\textbf{
integrate the frozen manuscript into the official CVPR LaTeX template.
}
}
\]

Order:

```text
1. Abstract
2. Introduction
3. Main Figure 1
4. Method
5. Theory
6. Naturalized IPIB section
7. Natural stress-test section
8. Calibration section
9. Limitations
10. Related Work
11. Supplement
```

---

# 159. After Full Draft

Perform:

\[
\boxed{
\textbf{Hostile Reviewer Pass}
}
\]

for:

```text
Novelty
Technical correctness
Experimental validity
Claims
Calibration
Related work
Practical significance
Limitations
Presentation
```

Every major objection must have either:

```text
evidence
qualification
or explicit limitation
```

---

# 160. Final v6.2 Principle

\[
\boxed{
\textbf{
Do not make the evidence fit the method.
Make the paper faithfully explain the evidence.
}
}
\]

---

# 161. Final Submission Principle

\[
\boxed{
\textbf{
The strength of this paper is not that every interaction experiment succeeds.
The strength is that it precisely identifies when interaction evidence succeeds, fails, or becomes ambiguous.
}
}
\]

---

# 162. Final Rule

\[
\boxed{
\textbf{
Research phase closed.
Evidence frozen.
No more result hunting.
Write the paper.
}
}
\]