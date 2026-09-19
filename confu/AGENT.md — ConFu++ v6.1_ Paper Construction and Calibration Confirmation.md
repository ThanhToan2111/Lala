# AGENT.md

# ConFu++ v6.1

## Paper Construction, Calibration Confirmation, and Submission Readiness

---

# 0. Mission

The experimental method-development phase of ConFu++ is complete.

The project now enters:

\[
\boxed{
\textbf{v6.1 — Paper Construction and Calibration Confirmation}
}
\]

The objective is no longer:

> improve the multimodal model.

The objective is:

\[
\boxed{
\textbf{
turn the frozen evidence into the strongest scientifically defensible CVPR main-track submission.
}
}
\]

All work must now belong to one of three categories:

```text
P1 — Paper construction

P2 — Reproducibility / presentation

P3 — One preregistered calibration-only confirmation
```

No architecture development is permitted.

---

# 1. Frozen Scientific State

The current framework is:

\[
\boxed{
\text{Data + Representation}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Cross-Modal Predictive Identifiability}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Task Joint Headroom}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Interaction Distillation}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Representation Fidelity}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Downstream Accessibility}
}
\]

The central empirical finding is that these stages can decouple.

---

# 2. Architecture Phase Status

```text
Canonical Product Joint:
FROZEN

Canonical JAD:
FROZEN

Generic Joint MLP audit:
COMPLETE / INCONCLUSIVE

Nested residual audit:
NESTED_HEADROOM_ABSENT

Task-Aware JAD:
BLOCKED

Third-order natural JAD:
BLOCKED

New fusion architecture:
BLOCKED

New natural dataset search:
BLOCKED

Pooling sweep:
BLOCKED

Rank sweep:
BLOCKED

Loss sweep:
BLOCKED

Context rescue:
BLOCKED
```

---

# 3. Current Evidence Chain

## Controlled synthetic hierarchy

Known first-, second-, and third-order structure demonstrates that explicit interaction representations can make known nonlinear structure accessible.

## IPIB

Controlled cross-modal predictive interaction demonstrates the difference between identifiable interaction structure and non-identifiable task interaction.

## Naturalized IPIB

Uses real MOSEI representation geometry with injected known interaction.

It establishes:

\[
\beta\uparrow
\Rightarrow
\hat J\uparrow
\]

under the frozen benchmark.

## Natural benchmarks

MOSEI shows a positive operational joint predictive advantage under the frozen historical protocol.

MUStARD and MELD do not produce a stable positive natural screen under the frozen protocol.

---

# 4. Important Calibration Correction

Historical threshold:

\[
\tau=0.01.
\]

It must now be described as:

\[
\boxed{
\textbf{an operational historical threshold}
}
\]

not:

\[
\boxed{
\text{a statistically calibrated discovery threshold}.
}
\]

---

# 5. Why This Wording Is Mandatory

The compute-limited MOSEI alignment-breaking null produced high-tail values above:

\[
0.01.
\]

Therefore:

\[
\hat J>0.01
\]

cannot currently be interpreted as a universally controlled low-false-positive discovery criterion.

---

# 6. MOSEI Wording

Do NOT write:

> MOSEI contains statistically significant cross-modal interaction.

Preferred:

> Under the frozen Product-Joint predictor and historical protocol, MOSEI VA→T exhibits a stable positive empirical joint predictive advantage across seeds.

Then immediately state:

> The alignment-breaking calibration audit overlaps this magnitude, so we treat the signal as operational rather than as a calibrated statistical interaction discovery.

---

# 7. Current Strongest Natural Claim

\[
\boxed{
\textbf{
A positive empirical joint predictive advantage does not guarantee downstream task improvement.
}
}
\]

MOSEI provides the principal natural evidence.

---

# 8. Current Strongest Controlled Claim

Naturalized IPIB shows that the operational predictor score responds systematically to increasing known interaction strength under natural source-representation geometry.

This is:

\[
\boxed{
\textbf{construct-validity evidence}
}
\]

for the diagnostic.

---

# 9. Naturalized IPIB Frozen Results

Interaction ladder:

\[
\beta
\in
\{
0,
0.25,
0.5,
1
\}.
\]

Validation \(J\):

\[
-0.01427
\]

\[
0.00262
\]

\[
0.05787
\]

\[
0.22625.
\]

Direct ground-truth recovery rises with \(\beta\).

JAD ground-truth recovery also rises but remains incomplete.

---

# 10. Naturalized IPIB Interpretation

Correct:

> Increasing injected interaction strength produces increasing measurable joint predictive headroom and increasing recovery of the known interaction component.

Do NOT write:

> J perfectly recovers interaction.

Do NOT write:

> JAD recovers the true synergy.

---

# 11. Accessibility Caveat

Naturalized IPIB accessibility labels are deliberately interaction-derived.

Therefore:

\[
\beta=0
\]

is a null for:

\[
\text{target-side injected interaction}
\]

but not a null for:

\[
\text{task-side interaction relevance}.
\]

---

# 12. Figure Separation Rule

Do NOT visualize:

\[
J
\]

and:

\[
\Delta_{\text{access}}
\]

as though both are controlled by the same null.

Use separate conceptual panels.

---

# 13. Main Paper Thesis

Use:

> **Multimodal interaction learning should be evidence-driven rather than assumed from modality count.**

A candidate extended statement:

> Before introducing explicit interaction representations, one should distinguish whether non-additive structure is predictively measurable, relevant to the downstream task, faithfully representable, and accessible to the task learner.

---

# 14. Main Research Question

\[
\boxed{
\textbf{
When is multimodal interaction measurable, task-relevant, faithfully represented, and downstream accessible?
}
}
\]

---

# 15. Preferred Paper Identity

The paper is primarily:

\[
\boxed{
\textbf{a diagnostic and scientific framework paper}
}
\]

not:

\[
\boxed{
\text{a state-of-the-art fusion architecture paper}.
}
\]

---

# 16. JAD Positioning

JAD is:

\[
\boxed{
\text{a concrete interaction-distillation mechanism}
}
\]

used to study:

```text
predictive interaction targets
fidelity
shortcut behavior
accessibility
```

JAD is NOT positioned as:

> universally superior fusion.

---

# 17. Contribution Structure

Use at most four main contribution bullets.

## Contribution 1 — Formal diagnostic

Define joint predictive advantage and formalize its population counterpart as a squared-risk projection gap under explicit assumptions.

---

# 18. Contribution 2 — Interaction hierarchy

Separate:

```text
cross-modal identifiability
task relevance
representation fidelity
downstream accessibility
```

and empirically demonstrate that these stages need not imply one another.

---

# 19. Contribution 3 — Controlled interaction benchmarks

Introduce:

```text
IPIB
Naturalized IPIB
```

to evaluate interaction detection and representation recovery when the interaction structure is known.

---

# 20. Contribution 4 — Natural multimodal audit

Study:

```text
MOSEI
MUStARD
MELD
```

and show that natural multimodal representations do not automatically provide robust evidence for useful non-additive interaction under the tested settings.

---

# 21. Contribution Language Restrictions

Do NOT claim:

```text
SOTA multimodal fusion
true synergy recovery
universal interaction identifiability
causal decomposition
information creation
universal calibration
```

---

# 22. P1 — Paper Construction

The main task is now to write the paper.

Recommended structure:

```text
1. Abstract
2. Introduction
3. Related Work
4. Framework / Method
5. Formal Analysis
6. Experiments
7. Limitations
8. Conclusion
```

---

# 23. Abstract Structure

The abstract should contain exactly the following logical sequence.

### Problem

Multimodal models frequently introduce increasingly complex fusion mechanisms without first establishing whether useful non-additive interaction is present.

### Question

When is multimodal interaction:

```text
measurable
task-relevant
faithfully represented
accessible
```

### Framework

Introduce the four-stage diagnostic hierarchy.

### Controlled evidence

Mention IPIB / Naturalized IPIB.

### Natural evidence

Mention MOSEI, MUStARD, MELD.

### Finding

These properties frequently decouple.

### Takeaway

Interaction learning should be evidence-driven.

---

# 24. Abstract — Avoid

Do not begin with:

> We propose a powerful new architecture...

Do not claim:

> ConFu++ significantly outperforms previous methods.

Evidence does not support that framing.

---

# 25. Introduction Paragraph 1

Motivate multimodal fusion.

Core observation:

\[
\text{multiple modalities}
\neq
\text{evidence of useful interaction}.
\]

Complex fusion modules are often justified by architecture rather than measured interaction structure.

---

# 26. Introduction Paragraph 2

Introduce the conceptual gap.

Existing questions such as:

```text
How should modalities be fused?
How should synergy be decomposed?
How should higher-order representations be learned?
```

are downstream of another question:

\[
\boxed{
\text{Is non-additive interaction measurably present and useful here?}
}
\]

---

# 27. Introduction Paragraph 3

Introduce four levels:

\[
I
\rightarrow
T
\rightarrow
F
\rightarrow
A.
\]

Where:

```text
I = identifiability
T = task relevance
F = fidelity
A = accessibility
```

State explicitly:

\[
I\nRightarrow T
\]

and:

\[
F\nRightarrow A.
\]

---

# 28. Introduction Paragraph 4

Introduce diagnostic machinery.

Describe:

```text
Additive predictor
Joint predictor
Joint predictive advantage J
Task headroom
JAD
Fidelity probes
Accessibility probes
```

---

# 29. Introduction Paragraph 5

Summarize evidence.

Controlled:

```text
Synthetic
IPIB
Naturalized IPIB
```

Natural:

```text
MOSEI
MUStARD
MELD
```

---

# 30. Introduction Paragraph 6

State contributions.

Do not include development-history details.

---

# 31. Related Work Structure

Organize by scientific topic, not chronological bibliography.

Suggested subsections:

```text
Multimodal Fusion

Higher-Order Multimodal Interaction

Information-Theoretic / Synergy Decomposition

Functional Interaction Decomposition

Representation Identifiability

Multimodal Interaction Benchmarks
```

---

# 32. Mandatory Closest-Work Matrix

Maintain:

```text
paper/related_work_matrix.md
```

Fields:

```text
Paper
Year
Venue
Interaction definition
Uses task labels?
Pair / higher-order?
Explicit interaction representation?
Information-theoretic?
Identifiability analysis?
Synthetic ground truth?
Natural benchmark?
Primary objective
Difference from this paper
```

---

# 33. Original ConFu Positioning

Original ConFu is mandatory.

Frame the distinction around:

```text
higher-order alignment
versus
measuring whether non-additive interaction is operationally supported
```

Do not attack ConFu.

Use it as the direct lineage.

---

# 34. DMIL / PID-Style Work

Clearly distinguish:

\[
\boxed{
\text{information decomposition}
}
\]

from:

\[
\boxed{
\text{predictive hypothesis-class advantage}.
}
\]

The current paper does NOT estimate PID atoms.

---

# 35. Method Section Structure

Use:

```text
4.1 Problem Setup

4.2 Additive and Joint Predictors

4.3 Cross-Modal Predictive Advantage

4.4 Task Joint Headroom

4.5 Joint-Advantage Distillation

4.6 Fidelity Diagnostics

4.7 Accessibility Diagnostics

4.8 Two-Gate Natural Screening
```

---

# 36. Problem Setup

Given:

\[
(X_1,\ldots,X_M,Y),
\]

frozen representations:

\[
r_m=E_m(X_m).
\]

For source pair:

\[
(r_i,r_j),
\]

and target modality:

\[
r_k.
\]

---

# 37. Additive Predictor

\[
q_A(r_i,r_j)
=
q_i(r_i)+q_j(r_j).
\]

Interpretation:

> predictive structure accessible without joint feature computation.

---

# 38. Joint Predictor

Canonical:

\[
q_J(r_i,r_j).
\]

Interpretation:

> predictive structure accessible when joint source computation is permitted.

Do not call all differences “synergy.”

---

# 39. Operational Score

\[
\boxed{
\hat J_{ij\rightarrow k}
=
R^2(q_J,r_k)
-
R^2(q_A,r_k).
}
\]

Use the hat notation when useful to distinguish empirical score from population object.

---

# 40. Task Headroom

For downstream task \(Y\):

\[
\boxed{
\hat H_{\text{task}}
=
Perf(c_J)
-
Perf(c_A).
}
\]

Task metrics depend on task type.

Do not collapse all metrics into one universal scalar if not necessary.

---

# 41. Two-Gate Principle

Conceptually:

\[
G_1:
\hat J>0
\]

\[
G_2:
\hat H_{\text{task}}>0.
\]

Historical experiments additionally used an operational:

\[
\tau=0.01
\]

screen.

The paper must clearly distinguish:

```text
conceptual positivity
historical operational threshold
statistical calibration
```

---

# 42. Historical Gate Presentation

Do NOT elevate:

\[
0.01
\]

into a theoretical constant.

Describe it as:

> a fixed preregistered operational threshold used for experiment progression.

---

# 43. JAD Section

Define:

\[
d=q_J-q_A.
\]

Then:

\[
h_{ij}
=
W
\left(
\frac{
LN(U_ir_i)\odot LN(U_jr_j)
}{
\sqrt R
}
\right).
\]

Loss:

\[
L=
1-\cos(h_{ij},d).
\]

---

# 44. JAD Interpretation

Use:

> JAD attempts to represent the empirical predictive difference exposed by the chosen joint predictor relative to the additive predictor.

Not:

> JAD extracts pure synergy.

---

# 45. Theory Section

Recommended:

```text
5.1 Population Squared-Risk Setup

5.2 Additive and Joint Function Spaces

5.3 Projection-Gap Proposition

5.4 Vector-Output Extension

5.5 Relation to Empirical Neural Fits

5.6 Non-Claims
```

---

# 46. Population Setup

Let:

\[
X=(X_i,X_j),
\]

\[
Y\in\mathbb R^d,
\]

and:

\[
m(X)=\mathbb E[Y|X].
\]

---

# 47. Function Spaces

Assume appropriate closed subspaces:

\[
\mathcal H_A
\subseteq
\mathcal H_J
\subseteq
L^2(P_X;\mathbb R^d).
\]

---

# 48. Population Optima

\[
f_A^*
=
\Pi_A m,
\]

\[
f_J^*
=
\Pi_J m.
\]

---

# 49. Main Proposition

Under stated assumptions:

\[
\boxed{
\mathcal R(f_A^*)
-
\mathcal R(f_J^*)
=
\|
\Pi_Jm-\Pi_Am
\|^2
}
\]

and therefore, for common variance normalization:

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

# 50. Theory Interpretation

This establishes a population predictive meaning:

> the additional conditional-mean structure accessible in the richer joint space beyond the additive projection.

---

# 51. Theory Non-Claims

Explicitly state that this does NOT imply:

```text
PID synergy
causal interaction
unique semantic interaction
new Shannon information
ground-truth natural interaction
```

---

# 52. Empirical-Theory Gap

Finite neural models estimate:

\[
\hat q_A,\hat q_J,
\]

not exact:

\[
\Pi_A m,\Pi_Jm.
\]

Therefore empirical \(\hat J\) additionally depends on:

```text
sample size
architecture
optimization
regularization
representation
```

---

# 53. Experiments Section

Recommended ordering:

```text
6.1 Experimental Questions

6.2 Controlled Synthetic Hierarchy

6.3 IPIB

6.4 Naturalized IPIB

6.5 Natural Two-Gate Audit

6.6 JAD Fidelity and Accessibility

6.7 Hypothesis-Class Robustness

6.8 Null Calibration
```

---

# 54. Experimental Questions

Explicitly organize experiments around:

```text
Q1 — Does J detect known non-additive predictive structure?

Q2 — Can JAD represent the detected structure?

Q3 — Does detected interaction imply task headroom?

Q4 — Does high representation fidelity imply accessibility?

Q5 — Are natural negative results explained by one restrictive joint architecture?

Q6 — How calibrated is the historical interaction gate?
```

---

# 55. Q1 Evidence

Use:

```text
Synthetic Oracle
IPIB
Naturalized IPIB
```

Naturalized IPIB should be the main controlled evidence.

---

# 56. Q2 Evidence

Use:

```text
IPIB target recovery
Naturalized IPIB direct recovery
Naturalized IPIB JAD recovery
MOSEI target cosine
```

---

# 57. Q3 Evidence

Use:

```text
MOSEI cross-modal operational J
MOSEI sentiment headroom
```

Phrase:

> positive empirical predictive advantage does not guarantee demonstrated task headroom.

---

# 58. Q4 Evidence

Use:

```text
MOSEI D2 high target cosine
no stable fixed-threshold sentiment gain
controlled synthetic accessibility analyses
```

---

# 59. Q5 Evidence

Use:

```text
Generic MLP audit
Nested Joint-Class audit
```

The nested audit is the stronger result.

---

# 60. Q6 Evidence

Use:

```text
IPIB null
MOSEI null
MELD null
threshold sensitivity
```

---

# 61. Main Figure 1 — Framework

Create one polished conceptual figure:

```text
Multimodal representations
        |
        v
Additive vs Joint Prediction
        |
        v
Identifiability
        |
        v
Task Additive vs Joint
        |
        v
Task Relevance
        |
        v
JAD
        |
        v
Fidelity
        |
        v
Accessibility
```

Show broken implication arrows where supported.

---

# 62. Main Figure 2 — Naturalized IPIB Detection

Plot:

\[
x=\beta
\]

\[
y=\hat J.
\]

Show mean ± uncertainty over five seeds.

Do not combine accessibility.

---

# 63. Main Figure 3 — Naturalized IPIB Recovery

Plot:

\[
x=\beta
\]

with:

```text
Direct d → ground-truth R²
JAD h → ground-truth R²
```

This visually separates:

\[
\text{detection}
\]

from:

\[
\text{representation fidelity}.
\]

---

# 64. Main Figure 4 — Natural Regimes

Represent natural settings using:

\[
x=\hat J
\]

and task headroom on \(y\), where meaningful.

Do not put hard Type labels on statistically ambiguous settings.

Use:

```text
operational positive
weak
negative
inconclusive
```

where appropriate.

---

# 65. Main Table 1 — Controlled Evidence

| Benchmark | Ground-truth interaction | Natural geometry | J screen | Direct recovery | JAD recovery |
|---|---|---|---|---|---|

Rows:

```text
Synthetic
IPIB
Naturalized IPIB
```

---

# 66. Main Table 2 — Natural Audit

| Dataset | Pair | Empirical \(J\) | Task headroom | JAD | Interpretation |
|---|---|---:|---:|---|---|

Rows should include:

```text
MOSEI VA
MUStARD VA / VT / AT
MELD VA / VT / AT
```

---

# 67. Main Table 3 — Architecture Robustness

| Dataset | Mapping | Product J | Generic MLP J | Nested Δ |
|---|---|---:|---:|---:|

Emphasize:

> generic MLP audit was inconclusive because positive control failed.

---

# 68. Main Table 4 — Null Calibration

Keep small:

| Setting | observed J | null mean | null high tail | interpretation |
|---|---:|---:|---:|---|

Do not overemphasize q99 with B=10.

---

# 69. P2 — Reproducibility

From now onward every paper number must map to:

```text
JSON artifact
experiment ID
commit hash
seed list
configuration
```

No manual transcription without verification.

---

# 70. Results Registry

Maintain:

```text
results/registry.json
```

as the source of truth.

Every paper table cell must trace back to it or another frozen machine-readable artifact.

---

# 71. Figure Scripts

All figures must be reproducible.

Required:

```text
paper/scripts/fig_framework.py
paper/scripts/fig_naturalized_detection.py
paper/scripts/fig_naturalized_recovery.py
paper/scripts/fig_natural_regimes.py
paper/scripts/table_natural.py
paper/scripts/table_calibration.py
```

---

# 72. Numerical Precision

Use consistent reporting.

Recommended:

```text
J / R²: 3–4 decimals
percentage points: 2–3 decimals
cosine: 3 decimals
```

Do not imply precision unsupported by seed count.

---

# 73. Development vs Confirmatory Results

Tag experiments explicitly as:

```text
CONTROLLED
SCREENING
CONFIRMATORY
DIAGNOSTIC
ROBUSTNESS
CALIBRATION
```

This is important because the project contains extensive exploratory development history.

---

# 74. Supplement Structure

Recommended:

```text
A. Full Formal Proof

B. Synthetic Construction Details

C. IPIB

D. Naturalized IPIB

E. Natural Dataset Audits

F. Task Headroom

G. JAD Diagnostics

H. Hypothesis-Class Audits

I. Null Calibration

J. Closed Experimental Branches

K. Reproducibility Details
```

---

# 75. Closed Branches Go to Supplement

Include concise summaries of:

```text
purification failure
loss variants
reliability weighting
generic MLP positive-control failure
MUStARD details
MELD data mismatches
```

Do not overload main paper.

---

# 76. P3 — Optional Calibration Confirmation

This is the ONLY new experiment allowed by default.

Name:

\[
\boxed{
\textbf{v6.1-C1 — MOSEI Null Calibration Confirmation}
}
\]

---

# 77. C1 Motivation

Current MOSEI calibration used:

\[
B=10.
\]

This is too small for a precise estimate of extreme null quantiles.

Therefore C1 may increase:

\[
B
\]

without changing model architecture or interpretation rules.

---

# 78. C1 Allowed Scope

Primary:

\[
\boxed{
\text{MOSEI VA}\rightarrow\text{T}
}
\]

alignment-breaking null.

Optional controls:

```text
IPIB I0
MELD VA→T
```

only if cheap.

---

# 79. C1 Replicates

Preferred:

\[
\boxed{
B=100
}
\]

Minimum worthwhile:

\[
\boxed{
B=50.
}
\]

Do not call B=10 a reliable 99th-percentile calibration.

---

# 80. C1 Preregistration

Before execution create:

```text
results/calibration/v61/C1_PROTOCOL.md
```

containing:

```text
dataset
mapping
shuffle definition
B
seeds
training protocol
reported statistics
decision-independent interpretation
```

---

# 81. C1 Frozen Shuffle

Use exactly the previously defined alignment-breaking procedure unless there is a documented bug.

Do not redesign the null after seeing existing results.

---

# 82. C1 Statistics

Report:

```text
mean
std
median
q90
q95
q99
maximum
empirical exceedance of 0.01
empirical exceedance of observed MOSEI J
```

---

# 83. More Important Than q99

With finite B, report empirical exceedance probability:

\[
\boxed{
\hat p
=
\frac{
1+\sum_b\mathbf 1[J_b\ge J_{\text{obs}}]
}{
B+1
}.
}
\]

Use the +1 correction.

This is more interpretable than relying only on interpolated \(q_{0.99}\).

---

# 84. C1 Interpretation

Possible outcome A:

\[
\hat p
\]

small.

Then say:

> the observed MOSEI advantage exceeds most alignment-breaking null replicates under this operational null.

Do NOT call it universal interaction significance.

---

# 85. C1 Outcome B

If:

\[
\hat p
\]

is not small:

MOSEI becomes:

\[
\boxed{
\text{operational positive under frozen predictor comparison, but not separated from the alignment-breaking null}.
}
\]

This does not damage the main framework.

It changes only the strength of the natural evidence.

---

# 86. C1 Cannot Change Architecture

Forbidden:

```text
new predictor
new residual
different rank
different optimizer
different pooling
new MOSEI feature source
```

---

# 87. C1 Cannot Change Historical Gate

Even after B=100:

\[
\tau=0.01
\]

remains historical.

If C1 suggests another threshold, discuss it descriptively.

Do not retroactively relabel all experiments.

---

# 88. C1 Decision

C1 is not:

```text
PASS / FAIL method
```

It is:

```text
CALIBRATION CONFIRMATION
```

Its role is reviewer-facing uncertainty quantification.

---

# 89. If C1 Is Not Run

Paper must explicitly state:

> The alignment-breaking calibration uses only ten replicates and should be interpreted as a finite-sample stress test rather than precise tail calibration.

Then move on.

Do not delay the paper indefinitely.

---

# 90. Claims Matrix

Maintain:

```text
paper/claims.md
```

Every claim has:

```text
Claim
Evidence
Assumptions
Counter-evidence
Limitations
Allowed wording
Forbidden wording
Status
```

---

# 91. Claim C1

Claim:

> Joint predictive advantage increases with known injected interaction strength under natural MOSEI representation geometry.

Evidence:

Naturalized IPIB.

Status:

\[
\boxed{
SUPPORTED
}
\]

---

# 92. Claim C2

Claim:

> A positive empirical joint predictive advantage guarantees downstream utility.

Status:

\[
\boxed{
REFUTED
}
\]

by MOSEI evidence.

Use careful operational wording.

---

# 93. Claim C3

Claim:

> Product Joint is universally sufficient.

Status:

\[
\boxed{
NOT CLAIMED.
}
\]

Nested audit only tests one modest residual extension.

---

# 94. Claim C4

Claim:

> Historical \(J>0.01\) is a calibrated significance threshold.

Status:

\[
\boxed{
REFUTED / NOT CLAIMED.
}
\]

---

# 95. Claim C5

Claim:

> JAD faithfully reconstructs all injected interaction.

Status:

\[
\boxed{
REFUTED.
}
\]

Naturalized IPIB shows partial recovery.

---

# 96. Claim C6

Claim:

> High interaction-target fidelity guarantees task improvement.

Status:

\[
\boxed{
REFUTED.
}
\]

---

# 97. Reviewer Matrix

Maintain:

```text
paper/reviewer_questions.md
```

Highest-risk reviewer questions:

```text
1. Why is this interaction rather than generic nonlinear prediction?

2. Why isn't J a synergy measure?

3. Why does the natural positive MOSEI score overlap the null?

4. Why no natural Type-IV success?

5. Why should this be a CVPR paper if accuracy does not improve?

6. Why is another-modality prediction meaningful?

7. Could your representation destroy the interaction?

8. Could the Product predictor be too weak?

9. Why not use stronger fusion / Transformer models?

10. What does Naturalized IPIB demonstrate that synthetic data does not?
```

---

# 98. Reviewer Answer — Why CVPR Without SOTA?

Answer around:

```text
scientific diagnosis
multimodal representation analysis
controlled interaction benchmark
formal grounding
failure-mode discovery
```

Do not answer with:

> negative results are valuable.

Be specific about what was learned.

---

# 99. Reviewer Answer — Representation Dependence

State explicitly:

\[
\hat J
=
\hat J(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J,
\mathcal O
).
\]

Representation dependence is part of the framework, not a hidden assumption.

---

# 100. Reviewer Answer — Product Weakness

Use nested audit.

Explain:

\[
q_N=q_P+g.
\]

Product is preserved by construction.

A modest residual extension did not reveal substantial extra MELD headroom.

Do not generalize beyond the tested extension.

---

# 101. Reviewer Answer — MOSEI Null

Be transparent:

> The alignment-breaking null exposes that the fixed historical threshold is not transportably calibrated.

This motivates the distinction between:

```text
operational screen
statistical discovery
```

This is a feature of the scientific framework, not something to conceal.

---

# 102. Reviewer Answer — No Natural Type IV

State:

> We did not find a natural setting that simultaneously establishes calibrated identifiability, stable task headroom, and downstream accessibility under the frozen protocols.

Then explain:

> Rather than search post hoc for a favorable benchmark, we preserve this as a limitation and use controlled Naturalized IPIB to study the positive mechanism.

This is the strongest defensible positioning.

---

# 103. Limitations Section Must Include

```text
No natural Type-IV confirmation

Historical threshold not universally calibrated

Finite B calibration unless C1 is run

Representation dependence

Hypothesis-class dependence

Optimization dependence

Partial JAD ground-truth recovery

Task/probe dependence of accessibility

No causal interpretation

No PID interpretation
```

---

# 104. Limitations Tone

Do not frame limitations as apologies.

Explain exactly what each limitation prevents the paper from claiming.

---

# 105. Title Candidates

### Preferred A

**When Is Multimodal Interaction Learnable? Identifiability, Task Relevance, and Accessibility in Multimodal Representations**

### Preferred B

**Beyond Multimodal Fusion: Diagnosing Identifiable and Task-Relevant Cross-Modal Interaction**

### Preferred C

**Before We Fuse: Diagnosing Predictive Interaction in Multimodal Representations**

Final selection waits for the related-work audit.

---

# 106. Terminology Consistency

Use one phrase consistently for \(J\):

Preferred:

\[
\boxed{
\text{joint predictive advantage}
}
\]

Use:

\[
\boxed{
\text{cross-modal predictive identifiability}
}
\]

for the broader operational property.

---

# 107. Avoid Overusing “Interaction”

Whenever possible qualify it:

```text
predictive interaction
non-additive predictive structure
joint predictive headroom
```

This reduces semantic overclaiming.

---

# 108. Introduction Kill Test

After writing the Introduction, verify a reviewer can answer:

```text
What is the problem?

Why is existing fusion insufficient conceptually?

What is J?

What are the four stages?

What are the main findings?

Why should CVPR readers care?
```

without reading the method.

---

# 109. Abstract Kill Test

The abstract must NOT require knowing ConFu.

A reader unfamiliar with the previous paper should understand the problem.

---

# 110. Figure Kill Test

Figure 1 alone should convey:

\[
\text{interaction learning requires multiple independent conditions}.
\]

Figure 2 alone should convey:

\[
\text{the diagnostic responds to controlled interaction strength}.
\]

---

# 111. Table Kill Test

The natural-data table must make clear that:

```text
positive J
positive task headroom
positive accessibility
```

are separate columns.

Never combine them into one “interaction score.”

---

# 112. Paper Narrative Order

Use:

\[
\boxed{
\text{Problem}
\rightarrow
\text{Definition}
\rightarrow
\text{Controlled validation}
\rightarrow
\text{Natural diagnosis}
\rightarrow
\text{Limitations}.
}
\]

Do not use:

\[
\text{chronological experiment history}.
\]

---

# 113. What Stays Out of Main Paper

Move to supplement:

```text
all failed purification variants
all loss tuning branches
all reliability weighting
full Generic MLP diagnostic tables
full per-seed MUStARD tables
full MELD data audit
all implementation-debug history
```

---

# 114. What Must Stay in Main Paper

```text
projection proposition
definition of J
two-gate distinction
Naturalized IPIB
natural screening summary
MOSEI fidelity/accessibility separation
nested residual robustness
calibration limitation
```

---

# 115. Paper Freeze Integrity

The existing experiment freeze remains binding.

Any new C1 calibration artifact must be appended as:

```text
calibration-only extension
```

not treated as reopening v6.0 method development.

---

# 116. New Freeze After C1

If C1 runs, create:

```text
PAPER_EXPERIMENT_FREEZE_V61.md
```

containing:

```text
previous freeze reference
C1 protocol
C1 result
updated calibration wording
no method changes
```

---

# 117. Submission Readiness Checklist

Before submission require:

```text
[ ] Abstract frozen
[ ] Introduction frozen
[ ] Related-work matrix complete
[ ] Theory proof independently checked
[ ] All main figures regenerated
[ ] All tables regenerated
[ ] Naturalized IPIB artifact verified
[ ] Natural benchmark values verified
[ ] Null calibration wording verified
[ ] Claims matrix complete
[ ] Reviewer matrix complete
[ ] Limitations complete
[ ] Supplement complete
[ ] Anonymous repository sanitized
[ ] No hidden test-driven selection
[ ] No unsupported novelty wording
```

---

# 118. Theory Verification Gate

Before paper freeze, independently re-derive:

\[
\mathcal R(f_A^*)-\mathcal R(f_J^*)
=
\|\Pi_Jm-\Pi_Am\|^2.
\]

Check:

```text
nestedness
closedness
orthogonality
vector-valued Hilbert setting
common R² denominator
```

If any assumption is too strong, weaken the theorem.

---

# 119. Naturalized IPIB Verification Gate

Verify:

```text
beta generator fixed
same source IDs
no target leakage
projection matrices frozen
private noise independent
JAD labels never used
five seeds complete
```

---

# 120. Calibration Verification Gate

Verify:

```text
shuffle done only inside split
new random permutation per replicate
same model protocol
no test-based selection
historical threshold unchanged
B explicitly reported
```

---

# 121. Statistical Language

Preferred:

```text
stable across seeds
operationally positive
below historical screen
alignment-breaking null
finite-replicate calibration audit
```

Avoid without justification:

```text
statistically significant
confirmed interaction
discovery threshold
1% FPR
```

---

# 122. Natural Type-IV Search Status

\[
\boxed{
\textbf{CLOSED FOR THIS SUBMISSION}
}
\]

unless all paper work is complete and a new test is explicitly preregistered for a strong scientific reason.

---

# 123. Architecture Search Status

\[
\boxed{
\textbf{PERMANENTLY CLOSED FOR THIS SUBMISSION}
}
\]

---

# 124. Third-Order Status

No natural third-order method.

Synthetic triple result remains a controlled demonstration only.

---

# 125. Task-Aware JAD Status

Do not add task supervision to JAD.

It would change the central scientific question.

---

# 126. Main Result to Emphasize

Not:

\[
\text{ConFu++ beats baseline}.
\]

Instead:

\[
\boxed{
\textbf{
interaction detection, task relevance, fidelity, and accessibility are empirically distinct.
}
}
\]

---

# 127. Main Controlled Result

\[
\boxed{
\textbf{
Naturalized IPIB shows that joint predictive advantage tracks injected interaction strength under natural representation geometry.
}
}
\]

---

# 128. Main Natural Result

\[
\boxed{
\textbf{
The natural benchmarks do not support treating multimodality itself as evidence for useful higher-order interaction.
}
}
\]

Always qualify this by the tested representations and model classes.

---

# 129. Main Negative Result

\[
\boxed{
\textbf{
MOSEI interaction-target fidelity does not translate into stable sentiment accuracy improvement.
}
}
\]

This is scientifically useful because it breaks an often implicit assumption.

---

# 130. Main Methodological Result

\[
\boxed{
\textbf{
interaction screening requires calibration and explicit separation from task utility.
}
}
\]

---

# 131. Final Paper Claim

Preferred final claim:

> Multimodal interaction learning should be evidence-driven: non-additive predictive structure must be distinguished from task relevance, representation fidelity, and downstream accessibility before complex interaction mechanisms are interpreted as useful multimodal reasoning.

---

# 132. Final Paper Non-Claim

The paper does not claim:

> every multimodal task lacks useful interaction.

It claims:

> useful interaction cannot be assumed and must be empirically established under explicit representations, hypotheses, and tasks.

---

# 133. Immediate Execution Plan

```text
1. Lock paper title candidates.

2. Write complete Abstract v1.

3. Write Introduction v1.

4. Complete related-work matrix.

5. Verify T1 theorem independently.

6. Build Figure 1 framework.

7. Build Naturalized IPIB detection/recovery figures.

8. Build natural benchmark summary table.

9. Build hypothesis-class robustness table.

10. Build calibration table.

11. Decide whether C1 B=100 is computationally feasible.

12A. If feasible:
     preregister C1;
     run only calibration;
     update wording.

12B. If not feasible:
     retain B=10 limitation;
     do not delay writing.

13. Draft Method.

14. Draft Theory.

15. Draft Experiments by Q1–Q6.

16. Draft Limitations.

17. Draft Supplement.

18. Run claims audit.

19. Run reviewer-question audit.

20. Freeze manuscript evidence.
```

---

# 134. Priority Order

\[
\boxed{
\text{Paper writing}
>
\text{figure/table construction}
>
\text{C1 calibration}
>
\text{everything else}.
}
\]

C1 should not block manuscript construction.

---

# 135. Hard Stop Rule

If any proposed new work asks:

> Can we make performance better?

Reject it.

If it asks:

> Does this resolve an explicit reviewer-facing uncertainty in the frozen scientific claim?

It may be considered.

---

# 136. Next Required Artifact

Create:

```text
paper/draft/PAPER_SKELETON.md
```

containing:

```text
Title candidates

Abstract

Introduction

Contributions

Related Work headings

Method headings

Theory headings

Experiment questions

Main tables

Main figures

Limitations

Conclusion
```

---

# 137. Artifact After Skeleton

Then create:

```text
paper/draft/ABSTRACT_V1.md
```

and:

```text
paper/draft/INTRODUCTION_V1.md
```

before any new non-calibration experiment.

---

# 138. Final v6.1 Principle

\[
\boxed{
\textbf{
The research question is now more important than the model.
}
}
\]

---

# 139. Final v6.1 Rule

\[
\boxed{
\textbf{
Do not search for a better result.
Write the strongest truthful paper supported by the frozen evidence.
}
}
\]

---

# 140. Next Action

\[
\boxed{
\textbf{
Construct the full CVPR paper skeleton and draft the Abstract + Introduction from the frozen evidence.
}
}
\]

The only experiment that remains admissible by default is the preregistered calibration-only confirmation:

\[
\boxed{
\textbf{C1: MOSEI alignment-breaking null with }B\ge50\textbf{, preferably }B=100.
}
\]