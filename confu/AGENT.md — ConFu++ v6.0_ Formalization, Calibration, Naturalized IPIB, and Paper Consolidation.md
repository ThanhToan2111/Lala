# AGENT.md

# ConFu++ v6.0

## Formalization, Calibration, Naturalized IPIB, and Paper Consolidation

---

# 0. Mission

This repository develops **ConFu++**, a scientific framework for diagnosing and learning higher-order multimodal predictive interaction.

The target remains:

\[
\boxed{
\textbf{CVPR Main Track}
}
\]

The architecture-development phase is complete.

v6.0 does **not** search for a stronger fusion model.

v6.0 converts the accumulated empirical results into a formally grounded, calibrated, reproducible scientific paper.

The four mandatory workstreams are:

\[
\boxed{
T1:\text{ Formal Theory}
}
\]

\[
\boxed{
T2:\text{ Gate / Null Calibration}
}
\]

\[
\boxed{
T3:\text{ Naturalized IPIB}
}
\]

\[
\boxed{
T4:\text{ Paper Consolidation + Closest-Method Comparison}
}
\]

No architecture branch may be reopened before these four workstreams are completed.

---

# 1. Current Project Status

```text
Synthetic Oracle:
PASS

NIC non-identifiable control:
PASS

IPIB controlled identifiable benchmark:
PASS

Canonical JAD:
FROZEN

Synthetic pair interaction recovery:
PASS

Synthetic triple oracle:
PASS

Fidelity–Accessibility separation:
ESTABLISHED

MOSEI canonical VA→T G1:
PASS

MOSEI sentiment G2:
INCONCLUSIVE

MOSEI JAD target fidelity:
PASS

MOSEI downstream sentiment gain:
FAIL

MOSEI shortcut localization:
MIXED

MOSEI additive purification:
FAIL

MUStARD G1:
FAIL

MUStARD G2:
FAIL / WEAK_UNSTABLE

MELD G1:
FAIL

MELD G2:
FAIL / WEAK_UNSTABLE

Generic Joint MLP H0:
HYPOTHESIS_CLASS_INCONCLUSIVE

Nested Joint-Class Audit:
NESTED_HEADROOM_ABSENT

Architecture development:
HARD FROZEN

New dataset search:
BLOCKED

Task-Aware JAD:
BLOCKED

Third-order natural interaction:
BLOCKED

Current phase:
FORMALIZATION + CALIBRATION + CONTROLLED BRIDGING + PAPER
```

---

# 2. Final Architecture Audit Result

v5.6.1 used:

\[
q_N(r_i,r_j)
=
q_P(r_i,r_j)
+
g([r_i,r_j]),
\]

where:

- \(q_P\) is frozen Product Joint;
- \(g\) is a small nonlinear residual;
- epoch-zero Product is a valid checkpoint;
- only \(g\) is trainable.

Official result:

\[
\boxed{
\textbf{NESTED\_HEADROOM\_ABSENT}
}
\]

---

# 3. Nested Validation Results

### MOSEI VA→T

\[
J_P
=
0.0198\pm0.0019
\]

\[
J_N
=
0.0200\pm0.0021
\]

\[
\boxed{
\Delta_N
=
0.0001\pm0.0003
}
\]

### MELD VA→T

\[
\Delta_N
=
0.0003\pm0.0004
\]

### MELD VT→A

\[
\Delta_N
=
0.0038\pm0.0014
\]

### MELD AT→V

\[
\Delta_N
\approx0.
\]

No MELD mapping satisfies:

\[
\Delta_N>0.01
\]

for all screening seeds.

---

# 4. Interpretation of Nested Audit

Correct statement:

> Under the frozen representations, Product predictor, training protocol, and modest nonlinear residual extension, no substantial additional cross-modal predictive headroom was detected on MELD.

Do NOT claim:

> MELD contains no interaction.

Do NOT claim:

> Product Joint is universally sufficient.

Do NOT claim:

> nonlinear models cannot detect multimodal interaction.

The result is conditional on:

\[
\boxed{
(\mathcal D,\phi,\mathcal H_P,\mathcal H_N,\mathcal O)
}
\]

where:

- \(\mathcal D\): data distribution;
- \(\phi\): representations;
- \(\mathcal H\): hypothesis classes;
- \(\mathcal O\): optimization/training protocol.

---

# 5. Architecture Hard Freeze

From v6.0 onward, do NOT introduce:

```text
new product architecture
deeper MLP
wider MLP
Transformer predictor
cross-attention
bilinear attention
new JAD loss
new JAD target
rank sweep
pooling sweep
dialogue-context rescue
purification
reliability weighting
Task-Aware JAD
third-order natural model
new natural benchmark search
```

Architecture research is complete for this submission.

---

# 6. Why Architecture Work Stops

The project has already tested:

```text
canonical Product Joint
independent Generic Joint MLP
nested nonlinear residual
```

with controlled protocols.

Continuing architecture search now would have lower scientific information gain and substantially higher risk of:

```text
architecture fishing
dataset fishing
post-hoc hypothesis changes
```

The paper bottleneck is no longer model capacity.

The bottleneck is:

\[
\boxed{
\text{formal grounding + calibration + controlled validation + positioning}.
}
\]

---

# 7. Main Scientific Thesis

Current thesis:

\[
\boxed{
\textbf{
Multimodal interaction should not be assumed from the presence of multiple modalities.
}
}
\]

Interaction learning should be separated into distinct questions:

\[
\boxed{
\text{Cross-Modal Identifiability}
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
\text{Representation Fidelity}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Downstream Accessibility}.
}
\]

These properties are not equivalent.

---

# 8. Empirical Non-Implications

The paper should organize results around three empirical non-implications.

## A

\[
\boxed{
\text{multimodality}
\nRightarrow
\text{cross-modal identifiability}
}
\]

Evidence:

```text
MUStARD
MELD
```

under tested representations and predictors.

---

# 9. Empirical Non-Implication B

\[
\boxed{
\text{cross-modal identifiability}
\nRightarrow
\text{demonstrated task headroom}
}
\]

Primary evidence:

```text
MOSEI canonical VA→T
```

Cross-modal:

\[
J_{\text{cross}}>0
\]

but downstream sentiment joint-headroom was not demonstrated.

---

# 10. Empirical Non-Implication C

\[
\boxed{
\text{interaction fidelity}
\nRightarrow
\text{downstream accessibility}
}
\]

Evidence:

```text
synthetic IPIB
MOSEI D2
```

A representation may accurately match an interaction target without delivering downstream task improvement.

---

# 11. Canonical JAD Definition

Canonical JAD remains frozen.

Given modalities:

\[
r_i,r_j
\]

and target:

\[
r_k,
\]

define additive predictor:

\[
q_A(r_i,r_j)
=
q_i(r_i)+q_j(r_j).
\]

Joint predictor:

\[
q_J(r_i,r_j).
\]

Define interaction target:

\[
\boxed{
d_{ij\to k}
=
q_J(r_i,r_j)
-
q_A(r_i,r_j).
}
\]

---

# 12. JAD Representation

Canonical interaction representation:

\[
u_i=LN(U_ir_i),
\]

\[
u_j=LN(U_jr_j),
\]

\[
\boxed{
h_{ij}
=
W
\left(
\frac{
u_i\odot u_j
}{
\sqrt R
}
\right)
}
\]

with:

\[
R=64.
\]

Loss:

\[
\boxed{
L_{JAD}
=
1-\cos(h_{ij},d_{ij\to k}).
}
\]

No modifications in v6.0.

---

# 13. Terminology

Preferred:

```text
joint predictive advantage
predictive interaction
non-additive predictive structure
cross-modal predictive interaction
task joint headroom
representation-conditional identifiability
hypothesis-class-relative identifiability
interaction fidelity
downstream accessibility
```

---

# 14. Forbidden Terminology

Do NOT use as unqualified claims:

```text
PID synergy
true synergy
pure interaction
causal interaction
new Shannon information
fusion creates information
ground-truth natural interaction
```

unless formally proven.

---

# 15. Information-Theoretic Constraint

For deterministic fusion:

\[
r_{ij}=F(r_i,r_j),
\]

we have:

\[
H(r_{ij}\mid r_i,r_j)=0.
\]

Therefore deterministic fusion does not create new Shannon information.

Similarly:

\[
I(Y;r_{ij}\mid r_i,r_j)=0.
\]

Do not imply otherwise.

---

# 16. v6.0 Structure

Execute strictly:

```text
T1 — Formal Projection Theory

T2 — Null and Threshold Calibration

T3 — Naturalized IPIB

T4 — Literature, Baselines, Claims, Figures, Paper
```

T1 should conceptually precede final T3 design.

T2 and T3 may be implemented in parallel once T1 definitions are frozen.

---

# 17. T1 — Formal Projection Theory

## Goal

Give a population interpretation of:

\[
J
=
R^2(q_J,Y)
-
R^2(q_A,Y).
\]

The theorem must explain exactly what the additive-vs-joint improvement measures.

---

# 18. Formal Setup

Let:

\[
X=(X_i,X_j)
\]

and target:

\[
Y\in\mathbb R^d.
\]

Define conditional mean:

\[
\boxed{
m(X)=\mathbb E[Y\mid X].
}
\]

Work in the Hilbert space:

\[
\boxed{
\mathcal L
=
L^2(P_X;\mathbb R^d).
}
\]

---

# 19. Hilbert Inner Product

Define:

\[
\langle f,g\rangle
=
\mathbb E[
f(X)^\top g(X)
].
\]

Norm:

\[
\boxed{
\|f\|_{\mathcal L}^2
=
\mathbb E\|f(X)\|_2^2.
}
\]

---

# 20. Population Risk

For predictor \(f\):

\[
\boxed{
\mathcal R(f)
=
\mathbb E
\|
Y-f(X)
\|_2^2.
}
\]

Then:

\[
m(X)
\]

is the unrestricted population squared-risk minimizer.

---

# 21. Additive Function Space

Define:

\[
\boxed{
\mathcal H_A
=
\left\{
f_i(X_i)+f_j(X_j)
\right\}.
}
\]

For formal projection theorem, assumptions must ensure the relevant closure is a closed subspace or otherwise specify closure:

\[
\overline{\mathcal H_A}.
\]

Do not hand-wave this point.

---

# 22. Joint Function Space

Let:

\[
\mathcal H_J
\]

be a richer function class satisfying, for the cleanest theorem:

\[
\boxed{
\mathcal H_A\subseteq\mathcal H_J.
}
\]

If practical Product Joint is not literally nested, distinguish:

```text
population theoretical nested formulation
finite practical capacity-matched formulation
```

Do not pretend they are identical.

---

# 23. Population Optima

Define:

\[
f_A^*
=
\arg\min_{f\in\mathcal H_A}
\mathcal R(f),
\]

\[
f_J^*
=
\arg\min_{f\in\mathcal H_J}
\mathcal R(f).
\]

Under Hilbert projection assumptions:

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

# 24. Core Proposition Target

Derive:

\[
\boxed{
\mathcal R(f_A^*)
-
\mathcal R(f_J^*)
=
\|
\Pi_Jm-\Pi_Am
\|_{\mathcal L}^2
}
\]

when:

\[
\mathcal H_A\subseteq\mathcal H_J
\]

are appropriate closed subspaces.

Verify all assumptions formally.

---

# 25. Why the Proposition Matters

The proposition gives a precise interpretation:

\[
\boxed{
\text{joint predictive advantage}
}
\]

measures predictable conditional-mean structure available to the joint function class but outside the additive projection.

This is NOT:

```text
PID synergy
causal interaction
new information
```

---

# 26. Residual Object

Define population interaction residual:

\[
\boxed{
d^*(X)
=
f_J^*(X)-f_A^*(X).
}
\]

Under nested orthogonal projection assumptions:

\[
d^*
\]

is orthogonal to:

\[
\mathcal H_A
\]

within the relevant projection geometry.

Verify before claiming.

---

# 27. Important Distinction

Canonical empirical:

\[
d=q_J-q_A
\]

is a finite-data approximation.

Do NOT automatically claim:

\[
d=d^*.
\]

Write:

> JAD distills an empirical estimate of the predictive difference between the trained joint and additive predictors.

---

# 28. Vector \(R^2\)

Freeze one exact definition.

Recommended:

\[
\boxed{
R^2
=
1-
\frac{
\mathbb E\|Y-\hat Y\|_2^2
}{
\mathbb E\|Y-\mathbb E[Y]\|_2^2
}.
}
\]

Empirical implementation must match.

Document whether denominator is:

```text
global vector variance
mean per-dimension variance
sum of squared deviations
```

Use one convention everywhere.

---

# 29. Connection Between Risk and R²

Under a common target denominator:

\[
V_Y
=
\mathbb E
\|Y-\mathbb E[Y]\|^2,
\]

derive:

\[
\boxed{
J^*
=
R^2(f_J^*)-R^2(f_A^*)
=
\frac{
\mathcal R(f_A^*)-\mathcal R(f_J^*)
}{
V_Y
}.
}
\]

Then under projection assumptions:

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

# 30. Theorem Scope

The theorem should explicitly state:

```text
population setting
squared loss
finite second moments
nested function spaces
projection assumptions
common target variance denominator
```

Do not overgeneralize to arbitrary losses.

---

# 31. Practical Finite-Sample Statistic

Empirical:

\[
\boxed{
\hat J
=
R^2(\hat q_J,Y)
-
R^2(\hat q_A,Y).
}
\]

It reflects:

```text
population structure
finite sample effects
optimization
regularization
model misspecification
```

Therefore call it:

\[
\boxed{
\text{operational / hypothesis-class-relative interaction score}.
}
\]

---

# 32. T1 Deliverables

Create:

```text
paper/theory/
    projection_theory.md
    assumptions.md
    proof_notes.md
```

Final `projection_theory.md` must include:

```text
definitions
proposition
proof
vector-output treatment
finite-sample caveat
relationship to JAD
non-claims
```

---

# 33. T1 Unit Tests

Where applicable verify numerical identities using synthetic data:

```text
R²-risk identity
nested linear projection example
zero interaction additive example
known interaction example
vector-output example
```

Theory tests are sanity checks, not theorem proofs.

---

# 34. T1 Hard Gate

Do not proceed to final paper claim wording until:

\[
\boxed{
\text{all theorem assumptions are explicitly written}.
}
\]

If proof fails:

weaken proposition.

Never force the theorem.

---

# 35. T2 — Null / Threshold Calibration

## Goal

Determine whether the operational threshold:

\[
\boxed{
J>0.01
}
\]

is reasonably separated from finite-sample null fluctuations.

Do NOT change historical decisions.

---

# 36. Historical Gate

The existing gate:

\[
\tau=0.01
\]

has already been used for:

```text
IPIB
MOSEI
MUStARD
MELD
```

Treat it as frozen historical protocol.

---

# 37. T2 Question

Ask:

> Under deliberately broken cross-modal interaction structure, what values of \(\hat J\) arise from finite sample, optimization, and model variance?

---

# 38. Null Construction Principles

A valid null should:

```text
destroy the relevant cross-modal correspondence
preserve sample count
preserve individual modality distributions when possible
avoid leaking validation/test selection
```

---

# 39. Primary Null Candidate

Preferred first null:

shuffle one source modality across samples within a split:

\[
(r_i,r_j,r_k)
\rightarrow
(r_i,\pi(r_j),r_k).
\]

Then retrain additive and joint predictors.

This breaks alignment involving \(r_j\).

---

# 40. Alternative Null

A target permutation:

\[
r_k\rightarrow\pi(r_k)
\]

destroys all predictability and is therefore a stronger null.

It may be included as a secondary sanity check.

Do not conflate the two nulls.

---

# 41. Conditional Shuffle Warning

Do not claim a shuffle preserves all lower-order dependencies unless mathematically true.

Simple permutations change the data-generating structure.

Describe them operationally:

> alignment-breaking null.

---

# 42. T2 Dataset Scope

Use a small but representative set:

```text
IPIB I0 or NIC
MOSEI VA→T
MELD VA→T
```

Optionally MELD VT→A only if computationally cheap.

Do not rerun every historical experiment.

---

# 43. Null Replications

Prefer enough null replicates to characterize the distribution.

Example:

\[
B=50
\]

or:

\[
B=100.
\]

Choose before examining results.

If compute is prohibitive, use a smaller preregistered number and report limitation.

---

# 44. Null Statistics

For each null replicate compute:

\[
\hat J_b.
\]

Report:

```text
mean
std
median
95th percentile
99th percentile
maximum
```

---

# 45. Empirical Null Quantile

Define:

\[
q_{0.95}^{null}
\]

and:

\[
q_{0.99}^{null}.
\]

Compare:

\[
0.01
\]

against those values.

---

# 46. Threshold Sensitivity

Without changing decisions, report the natural-screening classification under:

\[
\boxed{
\tau\in
\{0.005,0.01,0.02\}.
}
\]

Purpose:

measure sensitivity.

Not model selection.

---

# 47. Threshold Robustness Table

Produce:

| Dataset | Mapping | \(J_{val}\) | τ=.005 | τ=.01 | τ=.02 |
|---|---|---:|---|---|---|

Include:

```text
MOSEI
MUStARD
MELD
```

---

# 48. Desired Calibration Outcome

Ideal:

\[
q_{0.99}^{null}
<
0.01
\]

while canonical MOSEI:

\[
J\approx0.02.
\]

But do not tune null construction to obtain this.

---

# 49. If 0.01 Is Not Well Calibrated

Do NOT rewrite history.

Report:

> The fixed 0.01 gate is an operational threshold rather than a statistically calibrated universal cutoff.

Then emphasize continuous \(J\) values and confidence intervals in the paper.

---

# 50. T2 Deliverables

Create:

```text
results/calibration/v60/
    NULL_CALIBRATION_REPORT.md
    threshold_sensitivity.json
    null_distributions.json
```

Suggested code:

```text
src/experiments/calibration/
    interaction_null.py
    threshold_sensitivity.py
```

---

# 51. T2 Forbidden Behavior

Do NOT:

```text
pick a new threshold that maximizes paper story
change old PASS/FAIL labels silently
use test sets to calibrate threshold
search null constructions until 0.01 looks good
```

---

# 52. T3 — Naturalized IPIB

## Goal

Bridge the gap between:

\[
\boxed{
\text{clean synthetic ground truth}
}
\]

and:

\[
\boxed{
\text{natural representation geometry}.
}
\]

---

# 53. Why Naturalized IPIB Is Needed

Existing IPIB provides:

```text
known additive structure
known interaction structure
known private component
known task target
```

but uses idealized latent variables.

Natural datasets provide:

```text
natural correlations
anisotropy
representation geometry
real nuisance structure
```

but interaction ground truth is unknown.

Naturalized IPIB combines both.

---

# 54. Naturalized IPIB Principle

Take frozen real source representations:

\[
x_1,x_2
\]

from a natural dataset.

Do NOT use the natural target modality as the synthetic target.

Construct:

\[
\boxed{
y_{\beta}
=
a(x_1,x_2)
+
\beta j(x_1,x_2)
+
\lambda p
+
\epsilon.
}
\]

---

# 55. Real Geometry

Sources should come from a frozen real representation distribution.

Primary recommended source:

\[
\boxed{
\text{MOSEI canonical }(V,A)
}
\]

because:

```text
dataset is sufficiently large
canonical alignment is already audited
existing G1 positive mapping exists
```

Optionally repeat one setting on MELD after the primary benchmark is complete.

---

# 56. Additive Component

Construct:

\[
\boxed{
a(x_1,x_2)
=
A_1\tilde x_1
+
A_2\tilde x_2.
}
\]

Where:

\[
\tilde x_m
\]

may be a frozen standardized or projected representation.

---

# 57. Interaction Component

Use a known product interaction:

\[
z_1=P_1x_1
\]

\[
z_2=P_2x_2
\]

\[
\boxed{
j(x_1,x_2)
=
B(z_1\odot z_2).
}
\]

Freeze random matrices.

Do not learn the injected target generator.

---

# 58. Dimensionality

Choose a moderate latent interaction rank:

\[
r
\]

once.

Example:

\[
r=32
\]

or existing IPIB-compatible dimension.

No rank sweep.

---

# 59. Private Component

Sample:

\[
p
\]

independently of source modalities.

For example:

\[
p\sim\mathcal N(0,I)
\]

or frozen Rademacher.

Map through:

\[
P_p.
\]

---

# 60. Noise Component

Optional:

\[
\epsilon
\sim\mathcal N(0,\sigma^2I).
\]

Choose one fixed small noise level.

Do not sweep noise unless explicitly preregistered as a robustness axis.

---

# 61. Interaction Strength Ladder

Use one fixed ladder.

Preferred:

\[
\boxed{
\beta\in\{0,0.25,0.5,1.0\}.
}
\]

If scale requires normalization, calibrate components before locking the benchmark.

Do not modify after seeing JAD results.

---

# 62. Component Normalization

Before mixing components, standardize expected scale so that \(\beta\) has interpretable effect.

For example enforce approximately:

\[
Var(a)
\approx
Var(j)
\]

at \(\beta=1\).

Document exact procedure.

---

# 63. Control Condition

At:

\[
\beta=0,
\]

target contains no injected interaction component.

Expected:

\[
J\approx0
\]

up to finite-sample/model effects.

This is the critical false-positive control.

---

# 64. Increasing Interaction

As:

\[
\beta
\uparrow,
\]

desired:

\[
J
\uparrow.
\]

Do not require perfect monotonicity per seed, but inspect monotonic trend.

---

# 65. Naturalized IPIB Predictor Audit

For every \(\beta\):

train:

\[
q_A
\]

and:

\[
q_J
\]

with frozen architecture/protocol.

Report:

\[
J_{\beta}.
\]

---

# 66. Oracle Interaction

Because injected \(j(x_1,x_2)\) is known, compute direct recovery metrics.

For target:

\[
d=q_J-q_A,
\]

measure against:

\[
\beta j(x_1,x_2).
\]

---

# 67. Target Recovery Metrics

Report:

```text
cosine similarity
standardized MSE
linear R²
effective rank
variance
```

where meaningful.

---

# 68. JAD Recovery

Train canonical JAD:

\[
h_{12}
\]

against empirical:

\[
d.
\]

Then evaluate both:

\[
h_{12}
\leftrightarrow d
\]

and, because ground truth exists:

\[
h_{12}
\leftrightarrow j.
\]

---

# 69. Why Ground-Truth Recovery Matters

Natural datasets cannot tell whether:

\[
d
\]

corresponds to meaningful interaction.

Naturalized IPIB can.

This provides a controlled bridge:

\[
\boxed{
\text{real representation geometry}
+
\text{known interaction target}.
}
\]

---

# 70. Optional Task Label

Naturalized IPIB may include a controlled task:

\[
Y_{task}
=
\mathbf 1[
w^\top j(x_1,x_2)>0
].
\]

But only if needed to test accessibility.

Do not use task labels during JAD representation training.

---

# 71. Task Headroom in Naturalized IPIB

If task included:

compare:

\[
c_A
\]

vs:

\[
c_J.
\]

Verify:

\[
H_{\text{task}}
\]

increases as task-relevant interaction becomes stronger.

---

# 72. Accessibility Test

Compare:

\[
Z_{\le1}
=
[x_1,x_2]
\]

vs:

\[
Z_{\le2}
=
[x_1,x_2,h_{12}].
\]

Measure:

\[
\boxed{
\Delta_{access}
=
Perf(Z_{\le2})
-
Perf(Z_{\le1}).
}
\]

---

# 73. Naturalized IPIB Main Question

The benchmark should answer:

> Does the screen-discover-represent pipeline behave correctly when interaction ground truth is controlled but source geometry is inherited from real multimodal representations?

---

# 74. Naturalized IPIB Desired Pattern

Ideal:

\[
\beta=0
\Rightarrow
J\approx0
\]

and:

\[
\beta\uparrow
\Rightarrow
J\uparrow.
\]

Also:

\[
\beta\uparrow
\Rightarrow
\text{interaction target recovery improves}.
\]

---

# 75. No Need for Downstream SOTA

Naturalized IPIB is a mechanistic benchmark.

Its purpose is not classification SOTA.

---

# 76. T3 Seeds

Use:

\[
\boxed{
5\text{ seeds}
}
\]

for final benchmark.

Naturalized benchmark is intended for the paper, so stronger seed coverage is appropriate.

---

# 77. T3 Artifact Structure

```text
results/naturalized_ipib/v60/
    data_spec.json
    predictor_screen.json
    jad_recovery.json
    accessibility.json
    NATURALIZED_IPIB_REPORT.md
```

---

# 78. Naturalized IPIB Implementation

Suggested:

```text
src/experiments/naturalized_ipib/
    data.py
    generator.py
    screen.py
    jad.py
    accessibility.py
    report.py
```

---

# 79. Naturalized IPIB Reproducibility

Save:

```text
source sample IDs
projection matrices/seeds
component scales
beta
private/noise seeds
train/val/test split
```

The benchmark must be exactly reproducible.

---

# 80. Naturalized IPIB Anti-Tuning Rule

Once benchmark generation protocol is locked:

```text
do not change beta values
do not change interaction rank
do not change JAD rank
do not change task construction
do not change private noise
```

to improve results.

---

# 81. T4 — Paper Consolidation

v6.0 ultimately exists to produce the paper.

No experiment is justified unless it supports:

```text
a theorem
a central claim
a reviewer objection
a controlled benchmark
```

---

# 82. Paper Positioning

Preferred:

\[
\boxed{
\textbf{diagnostic/scientific framework}
}
\]

not:

\[
\boxed{
\textbf{SOTA fusion architecture}.
}
\]

---

# 83. Main Research Question

Use:

\[
\boxed{
\textbf{
When is multimodal interaction identifiable, task-relevant, faithfully represented, and downstream accessible?
}
}
\]

---

# 84. Candidate Paper Title A

> **When Is Multimodal Interaction Learnable? Identifiability, Task Relevance, and Accessibility in Multimodal Representations**

---

# 85. Candidate Paper Title B

> **Beyond Multimodal Fusion: Diagnosing Identifiable and Task-Relevant Cross-Modal Interactions**

---

# 86. Candidate Paper Title C

> **Do Multimodal Models Need Interaction? A Diagnostic Framework for Identifiability, Task Relevance, and Accessibility**

Do not finalize title until related-work positioning is complete.

---

# 87. Main Contribution 1

A formal distinction between:

\[
\boxed{
\text{cross-modal predictive identifiability}
}
\]

and:

\[
\boxed{
\text{task joint headroom}.
}
\]

---

# 88. Main Contribution 2

A projection-based interpretation of joint predictive advantage:

\[
J.
\]

Subject to T1 proof completion.

---

# 89. Main Contribution 3

IPIB and Naturalized IPIB as controlled interaction benchmarks.

The latter bridges synthetic interaction ground truth and natural representation geometry.

---

# 90. Main Contribution 4

JAD as a concrete label-free mechanism for distilling:

\[
q_J-q_A.
\]

Do not oversell JAD as universally task-improving.

---

# 91. Main Contribution 5

Natural evidence that:

\[
\text{multimodality}
\nRightarrow
\text{identifiability},
\]

and:

\[
\text{identifiability}
\nRightarrow
\text{task headroom}.
\]

---

# 92. Main Contribution 6

Evidence that high interaction-target fidelity does not necessarily imply downstream accessibility.

---

# 93. Related-Work Audit

Create:

```text
paper/related_work_matrix.md
```

Columns:

```text
Paper
Venue/year
Interaction definition
Requires labels?
Pair/triple?
Information-theoretic?
Explicit interaction representation?
Natural benchmarks
Synthetic ground truth?
Task-performance objective?
Identifiability analysis?
Closest difference to our work
```

---

# 94. Mandatory Related-Work Categories

Search and position against:

```text
ConFu
multimodal synergy / PID methods
DMIL
multimodal decomposition
tensor/bilinear fusion
interaction discovery
ANOVA-style interaction decomposition
representation identifiability
multimodal information decomposition
```

Use up-to-date literature before final claims.

---

# 95. Novelty Claim Restriction

Do NOT claim:

```text
first multimodal interaction method
first synergy decomposition
first higher-order multimodal representation
```

unless exhaustive literature audit proves it.

---

# 96. Preferred Novelty Position

Potential:

> Existing multimodal interaction methods generally optimize or decompose interaction once a multimodal task is defined. We instead ask whether non-additive structure is operationally identifiable in the representation space, whether the downstream task exhibits corresponding joint headroom, and whether the resulting interaction representation is faithful and accessible.

This wording must be revised after literature audit.

---

# 97. Main Figure 1

Create conceptual pipeline:

```text
Raw modalities
      |
      v
Frozen representations φ
      |
      v
Additive vs Joint Prediction
      |
      v
Cross-Modal Identifiability
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
Interaction Fidelity
      |
      v
Downstream Accessibility
```

---

# 98. Main Figure 2

Natural-regime plot.

Axes:

\[
x=J_{\text{cross}}
\]

\[
y=H_{\text{task}}.
\]

Place:

```text
MOSEI
MUStARD
MELD
```

with uncertainty.

Do not falsely force categorical labels if results are inconclusive.

---

# 99. Main Figure 3

Synthetic / Naturalized interaction-strength curve:

\[
\beta
\]

vs:

\[
J.
\]

Also possibly:

\[
\beta
\]

vs interaction recovery.

---

# 100. Main Figure 4

Fidelity–Accessibility diagram.

Show that:

\[
\text{high target similarity}
\]

does not guarantee:

\[
\text{task gain}.
\]

Use synthetic and MOSEI.

---

# 101. Main Table 1

Controlled benchmarks:

| Benchmark | Identifiable? | Ground truth interaction? | JAD recovery | Accessibility |
|---|---|---|---|---|

Include:

```text
Synthetic
IPIB
Naturalized IPIB
```

---

# 102. Main Table 2

Natural screening:

| Dataset | Representation | Pair | \(J_{cross}\) | G1 | \(H_{task}\) | G2 | JAD |
|---|---|---|---:|---|---:|---|---|

---

# 103. Main Table 3

Hypothesis-class robustness:

| Dataset | Mapping | Product J | Generic MLP J | Nested Δ |
|---|---|---:|---:|---:|

Keep generic MLP result explicitly labeled as inconclusive positive-control audit.

Nested result is the stronger robustness evidence.

---

# 104. Claims Matrix

Create:

```text
paper/claims.md
```

Required fields:

| Claim | Evidence | Assumptions | Limitation | Status |
|---|---|---|---|---|

---

# 105. Example Claim

Claim:

> Natural multimodal data does not automatically exhibit measurable non-additive cross-modal predictive headroom.

Evidence:

```text
MUStARD
MELD
```

Limitation:

```text
representation-conditional
hypothesis-class-relative
finite-sample
```

Status:

```text
SUPPORTED UNDER TESTED SETTINGS
```

---

# 106. Another Claim

Claim:

> Cross-modal predictive interaction does not guarantee downstream task headroom.

Evidence:

```text
MOSEI VA→T G1 positive
MOSEI sentiment T0 inconclusive/no positive headroom
```

Use careful wording:

```text
not demonstrated
```

not:

```text
does not exist
```

---

# 107. Reviewer Question Matrix

Create:

```text
paper/reviewer_questions.md
```

At minimum answer:

```text
What exactly is interaction?

Why isn't J PID synergy?

Why use another modality as prediction target?

Why should cross-modal predictability matter?

Why does JAD not improve MOSEI sentiment?

Why does MELD fail G1?

Could Product Joint be too restrictive?

Why is 0.01 the threshold?

Why not train a stronger model?

Why no dialogue context?

Why no natural Type IV?

Why is this useful without SOTA gains?

What does Naturalized IPIB add beyond synthetic IPIB?

How is this different from ConFu?

How is this different from DMIL/PID approaches?
```

---

# 108. Critical Reviewer Defense — Product Capacity

Response structure:

1. Product finds stable positive MOSEI signal.
2. Generic independent MLP audit was inconclusive because it failed that positive control.
3. Therefore we did not claim sufficiency from generic MLP.
4. We constructed a nested nonlinear residual preserving Product exactly.
5. Nested residual added little MELD headroom.
6. Conclusion remains hypothesis-class conditional.

This is the scientifically correct response.

---

# 109. Critical Reviewer Defense — Negative Results

Do not apologize for negative results.

Explain:

> The framework is designed to distinguish settings where higher-order interaction learning is justified from settings where it is not.

Negative datasets are therefore diagnostic evidence.

---

# 110. Critical Reviewer Defense — No Type IV

Current limitation:

\[
\boxed{
\text{no natural Type-IV setting has been established}.
}
\]

State directly.

Do not hide it.

Naturalized IPIB partially addresses the positive mechanistic side but does not replace a natural Type-IV task.

---

# 111. Optional Natural Type-IV Search

Blocked by default.

May only reopen if:

```text
T1–T4 are complete
paper reviewers/coauthors identify a specific theoretically motivated task
the hypothesis is written before running the experiment
```

No broad dataset search.

---

# 112. Baseline Philosophy

Do not compare only task accuracy.

For diagnostic methods, compare:

```text
false-positive interaction detection
interaction recovery
joint predictive advantage
target fidelity
shortcut reconstructability
downstream accessibility
```

where baselines permit.

---

# 113. ConFu Comparison

Original ConFu is mandatory.

Compare conceptually and experimentally where possible:

```text
higher-order alignment
interaction identifiability
explicit interaction target
downstream representation
```

Do not misrepresent ConFu as interaction decomposition if that is not its formal objective.

---

# 114. DMIL / Decomposition Comparison

If implementation is available/reproducible:

include at least one controlled comparison.

If not:

provide conceptual distinction with precise citations and clearly state implementation limitation.

Do not build a poor unofficial baseline and present it as authoritative.

---

# 115. Statistical Reporting

For final main tables:

```text
mean
sample std
number of seeds
per-seed values in supplement
```

Use confidence intervals where useful.

Do not rely on p-values alone.

---

# 116. Multiple Comparisons

Because many historical experiments exist:

do not retrospectively turn every run into a hypothesis test.

Separate:

```text
development experiments
confirmatory experiments
diagnostic audits
```

clearly.

---

# 117. Experiment Taxonomy

Label every experiment as one of:

```text
CONTROLLED CONSTRUCTION
SCREENING
CONFIRMATORY
DIAGNOSTIC
NEGATIVE CONTROL
ROBUSTNESS AUDIT
```

This improves paper clarity.

---

# 118. Final Natural Dataset Roles

### MOSEI

Role:

\[
\boxed{
\text{natural cross-modal identifiability positive}
}
\]

plus:

\[
\boxed{
\text{task relevance/accessibility negative or inconclusive}
}
\]

---

# 119. MUStARD Role

Role:

\[
\boxed{
\text{small-data natural negative screening}
}
\]

Use cautiously due sample size.

---

# 120. MELD Role

Role:

\[
\boxed{
\text{larger natural negative G1 setting}
}
\]

plus nested robustness audit.

---

# 121. Synthetic Role

Role:

\[
\boxed{
\text{known interaction hierarchy}
}
\]

including pair and triple Oracle behavior.

---

# 122. IPIB Role

Role:

\[
\boxed{
\text{controlled cross-modal identifiable predictive interaction}
}
\]

with adjustable interaction/private components.

---

# 123. Naturalized IPIB Role

Role:

\[
\boxed{
\text{controlled interaction under natural source geometry}.
}
\]

This should become one of the paper's strongest experiments.

---

# 124. Do Not Reopen Triple Learning

Synthetic triple Oracle can remain evidence that higher-order interaction may require explicit order-specific representation.

Do NOT create natural third-order JAD in this submission.

---

# 125. Paper Method Section Structure

Suggested:

```text
3.1 Problem Setup

3.2 Cross-Modal Predictive Identifiability

3.3 Task Joint Headroom

3.4 Joint-Advantage Distillation

3.5 Fidelity and Accessibility Diagnostics

3.6 Two-Gate Screening Protocol
```

---

# 126. Theory Section Structure

Suggested:

```text
4.1 Population Squared-Risk Formulation

4.2 Additive and Joint Projection Spaces

4.3 Joint Predictive Advantage Proposition

4.4 Empirical Approximation and Limitations
```

---

# 127. Experiments Section Structure

Suggested:

```text
5.1 Synthetic Interaction Hierarchy

5.2 IPIB

5.3 Naturalized IPIB

5.4 Natural Two-Gate Screening

5.5 MOSEI JAD Fidelity and Accessibility

5.6 Hypothesis-Class Robustness

5.7 Threshold / Null Calibration
```

---

# 128. Avoid Chronological Paper Writing

Do NOT write paper as:

```text
v3
v4
v4.2
v5.0
v5.1
...
```

Those are development history.

Paper must organize evidence by scientific question.

---

# 129. Supplement Role

Move detailed:

```text
all seed-level tables
closed branches
P1 purification
B/C loss experiments
Generic MLP failure diagnostics
full capacity audits
data audits
unit-test details
```

to supplementary.

---

# 130. Main Paper Only Keeps

Main paper should contain:

```text
central theorem
core framework
one controlled synthetic hierarchy
IPIB/Naturalized IPIB
natural screening summary
MOSEI positive G1 + accessibility result
nested robustness summary
```

---

# 131. Limitations Section

Must explicitly include:

```text
hypothesis-class-relative identifiability
representation dependence
finite-data optimization dependence
no natural Type-IV setting established
JAD not shown to improve all downstream tasks
threshold is operational unless calibration supports stronger claim
deterministic interaction representation does not create information
```

---

# 132. Reproducibility Checklist

Before paper freeze verify:

```text
all seeds saved
all configs saved
all data-source provenance documented
all split IDs saved
all checkpoints reproducible
all tables regenerable from JSON
all figures regenerable from scripts
no manual table transcription
```

---

# 133. Results Registry

Create:

```text
results/registry.json
```

containing every final paper result with:

```text
experiment_id
dataset
version
artifact
commit hash
config
seeds
status
paper table/figure
```

---

# 134. Figure Regeneration

All figures must be script-generated.

Suggested:

```text
paper/scripts/
    fig_framework.py
    fig_two_gate.py
    fig_naturalized_ipib.py
    fig_fidelity_accessibility.py
    table_natural_screening.py
```

---

# 135. Experimental Freeze Date

Once T1–T3 final results are complete:

create:

```text
PAPER_EXPERIMENT_FREEZE.md
```

containing:

```text
commit hash
date
included experiments
excluded exploratory experiments
frozen claims
```

---

# 136. No Post-Freeze Tuning

After paper experiment freeze:

only:

```text
bug fixes
figure regeneration
clarity checks
reproducibility fixes
```

No performance-driven modifications.

---

# 137. v6.0 Priority

Priority order:

\[
\boxed{
T1>T2\approx T3>T4.
}
\]

T4 literature audit can start immediately, but final positioning waits for T1–T3.

---

# 138. Immediate T1 Execution Order

```text
1. Freeze vector R² definition.

2. Define Hilbert-space setting.

3. Define additive space.

4. Define joint space.

5. State nestedness assumptions.

6. Derive population risk decomposition.

7. Derive projection proposition.

8. Extend to vector outputs.

9. Connect risk gap to R² gap.

10. Define empirical J.

11. State finite-sample limitations.

12. State what theorem does NOT prove.

13. Validate on small numerical examples.
```

---

# 139. Immediate T2 Execution Order

```text
1. Freeze null construction before runs.

2. Choose datasets/settings.

3. Choose number of null replicates.

4. Run alignment-breaking nulls.

5. Compute null J distributions.

6. Compare 0.01 to null quantiles.

7. Generate threshold sensitivity table.

8. Preserve historical decisions.

9. Write calibration limitations.
```

---

# 140. Immediate T3 Execution Order

```text
1. Freeze source representation distribution.

2. Freeze additive generator.

3. Freeze interaction generator.

4. Freeze private/noise generator.

5. Normalize component scales.

6. Freeze beta ladder.

7. Generate train/val/test targets.

8. Run additive/joint predictor screen.

9. Measure ground-truth interaction recovery.

10. Train canonical JAD.

11. Measure h ↔ d and h ↔ ground-truth interaction.

12. Run accessibility probe if task target included.

13. Produce five-seed report.

14. Do not retune benchmark after results.
```

---

# 141. Immediate T4 Execution Order

```text
1. Build related-work matrix.

2. Build claims matrix.

3. Build reviewer-question matrix.

4. Draft central figures.

5. Draft method section.

6. Draft theory section.

7. Insert final T1–T3 results.

8. Draft limitations.

9. Build supplementary result registry.

10. Freeze experiments.
```

---

# 142. Hard Stop Conditions

Immediately stop an experiment branch if it requires:

```text
changing architecture to rescue result
changing threshold after observing data
switching datasets to find a positive
using test set for method selection
changing benchmark construction after seeing JAD performance
```

---

# 143. v6.0 Success Condition

v6.0 succeeds if the project has:

\[
\boxed{
\text{formal definition}
}
\]

+

\[
\boxed{
\text{calibrated operational diagnostic}
}
\]

+

\[
\boxed{
\text{controlled natural-geometry benchmark}
}
\]

+

\[
\boxed{
\text{coherent natural evidence}
}
\]

+

\[
\boxed{
\text{clear novelty positioning}.
}
\]

It does NOT require a new architecture.

---

# 144. Minimum CVPR Package

Before submission, require:

```text
T1 theorem/proposition complete

T2 threshold/null calibration complete

T3 Naturalized IPIB complete

Original ConFu comparison complete

closest decomposition/synergy positioning complete

natural-regime summary complete

nested robustness result included

claims/limitations matrix complete

all core figures reproducible
```

---

# 145. Stronger CVPR Package

Desirable but not mandatory:

```text
one independently motivated natural Type-IV setting
one additional reproducible decomposition baseline
strong Naturalized IPIB monotonic interaction ladder
clean theoretical corollary connecting d* to additive orthogonal complement
```

Do not sacrifice rigor to obtain these.

---

# 146. Main Paper Message

Do not write:

> More complex fusion is better.

Write:

> Multimodal interaction learning is conditional: interaction must first be measurable beyond additive prediction, relevant to the downstream task, representable with sufficient fidelity, and accessible to the task learner.

---

# 147. Final Framework

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
\text{Additive vs Joint Predictability}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Cross-Modal Identifiability}
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
\text{Fidelity}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Accessibility}.
}
\]

---

# 148. Final Scientific Principle

\[
\boxed{
\textbf{
Interaction learning should be evidence-driven rather than architecture-driven.
}
}
\]

---

# 149. Final v6.0 Rule

\[
\boxed{
\textbf{
No more architecture fishing.
Formalize the object.
Calibrate the diagnostic.
Bridge synthetic and natural geometry.
Then write the paper.
}
}
\]

---

# 150. Next Action

Begin:

\[
\boxed{
\textbf{
T1 — Formal Projection Theory
}
}
\]

before any additional model-development experiment.

The next required artifact is:

```text
paper/theory/projection_theory.md
```

containing the precise population definition, projection proposition, vector-output extension, proof, empirical approximation, assumptions, and non-claims.