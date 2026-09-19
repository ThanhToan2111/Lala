# AGENT.md

# ConFu++ v4.3

## Accessibility Diagnosis and Calibration for Explicit Multimodal Interaction Discovery

---

# 0. Mission

This repository develops **ConFu++**, an extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The project has already established:

\[
\boxed{
Q1:\ Architecture\ Capability = PASS
}
\]

and:

\[
\boxed{
Q2a:\ Interaction\ Identifiability = PASS
}
\]

and has shown that:

\[
\boxed{
\text{Joint-Advantage Distillation can discover known pair interaction}
}
\]

on the Identifiable Predictive-Interaction Benchmark.

The remaining problem is:

\[
\boxed{
Q2b:\ Interaction\ Accessibility
}
\]

Specifically:

> Why does D2 recover a cleaner and more selective interaction target than D1, while D1 remains more useful to a linear downstream classifier in weak and medium interaction regimes?

This is the sole high-priority research question of v4.3.

---

# 1. Current Project Status

```text
Oracle architecture:
PASS

Oracle order recovery:
PASS

NIC non-identifiable control:
PASS

IPIB identifiability:
PASS

D0 Original ConFu discovery:
PASS as baseline

D1 residual discovery:
STRONG ACCESSIBILITY

D2 joint-advantage discovery:
STRONG SELECTIVITY

D2 > D0:
PASS

D2 > D1:
PARTIAL

D2 I0 false-positive control:
PASS

D2 accessibility calibration:
NEXT

Third-order discovery:
BLOCKED

Real-world transfer:
BLOCKED
```

---

# 2. Canonical v4.2 Result

Five-seed linear downstream gains:

| Regime | D0 | D1 | D2 |
|---|---:|---:|---:|
| I0 | 4.12 | 6.12 | **0.00** |
| I1 | 13.52 | **39.43** | 25.19 |
| I2 | 31.88 | **44.83** | 38.78 |
| I3 | 46.40 | 47.00 | **47.12** |
| I4 | 44.06 | 44.66 | **44.78** |

D2 therefore:

- beats D0 in every positive interaction regime;
- perfectly controls I0 false-positive utility;
- approaches or slightly exceeds D1 for strong interaction;
- loses substantially to D1 in weak and medium regimes.

---

# 3. Canonical Recovery Result

Target recovery:

\[
R^2(target\rightarrow\tilde g_{12})
\]

shows:

### I1

\[
D1=0.058
\]

\[
D2=0.573.
\]

### I2

\[
D1=0.199
\]

\[
D2=0.870.
\]

### I3

\[
D1=0.498
\]

\[
D2=0.984.
\]

### I4

\[
D1=0.199
\]

\[
D2=0.956.
\]

Thus:

\[
\boxed{
D2\text{ is dramatically more selective for the known interaction target.}
}
\]

---

# 4. Interaction Embedding Recovery

For learned interaction embedding:

\[
h_{12},
\]

interaction recovery also remains strong.

Example:

### I1

\[
D1=0.845
\]

\[
D2=0.562.
\]

### I2

\[
D1=0.959
\]

\[
D2=0.841.
\]

### I3

\[
D1=0.983
\]

\[
D2=0.960.
\]

The key observation is therefore:

\[
\boxed{
\text{target purity, embedding recovery, and task accessibility are different properties}.
}
\]

---

# 5. Central Scientific Observation

The project must no longer assume:

\[
\boxed{
\text{better interaction recovery}
\Rightarrow
\text{better task accessibility}.
}
\]

Current evidence shows this implication is false.

D2 can recover a much cleaner approximation of the known joint interaction while D1 produces a representation that a linear classifier uses more effectively.

This is now a central research finding.

---

# 6. New Core Distinction

From v4.3 onward, always distinguish:

### Interaction fidelity

How accurately the target or embedding corresponds to known joint structure.

### Task-direction fidelity

How much the representation preserves the particular interaction direction relevant to the downstream label.

### Accessibility

How easily a restricted downstream model can exploit that representation.

These are not equivalent.

---

# 7. v4.3 Research Question

The immediate question is:

\[
\boxed{
\text{Where does D2 lose downstream task utility?}
}
\]

There are two primary possibilities.

---

# 8. Hypothesis H1 — Target Construction Bottleneck

D2 target:

\[
d=q_{joint}-q_{add}
\]

may already have lower task-direction accessibility than D1 target:

\[
e=x_3-q_{add}.
\]

Then:

\[
\boxed{
\text{utility is lost before interaction distillation}.
}
\]

If this is true, modifying:

\[
h_{12}
\]

or the interaction loss cannot fully solve the problem.

---

# 9. Hypothesis H2 — Distillation Bottleneck

D2 target may itself be strongly task-accessible, but training:

\[
h_{12}\rightarrow d
\]

may distort its useful geometry.

Then:

\[
\boxed{
\text{utility is lost during distillation}.
}
\]

If this is true, target construction should remain unchanged and only the projection/loss should be calibrated.

---

# 10. Immediate Experiment A0

The mandatory next experiment is:

\[
\boxed{
\text{A0 — Target-vs-Embedding Accessibility Audit}
}
\]

Do NOT train a new discovery method before completing A0.

---

# 11. Ground-Truth Objects

For each IPIB sample, retain:

\[
g=z_1\odot z_2.
\]

Projected joint target:

\[
\tilde g=B_3g.
\]

Task score:

\[
\boxed{
s_y=w^\top g
}
\]

where the downstream binary label is generated from:

\[
y=\mathbf1[s_y+\epsilon_y>0].
\]

---

# 12. Discovery Targets

D0 target:

\[
t_{D0}=x_3.
\]

D1 target:

\[
\boxed{
t_{D1}
=
e
=
x_3-q_{add}.
}
\]

D2 target:

\[
\boxed{
t_{D2}
=
d
=
q_{joint}-q_{add}.
}
\]

---

# 13. Learned Embeddings

Audit:

\[
h_{D0},
h_{D1},
h_{D2}.
\]

All must come from the existing frozen v4.2 runs initially.

Do not retrain them for A0.

---

# 14. A0 Required Measurements

For each of:

```text
g
g_tilde
D0 target
D1 target
D2 target
D0 embedding
D1 embedding
D2 embedding
```

measure:

### Full interaction recovery

\[
R^2(feature\rightarrow g)
\]

or projected equivalent.

### Task-score recovery

\[
\boxed{
R^2(feature\rightarrow s_y)
}
\]

### Linear downstream accuracy

\[
Acc_{linear}(feature\rightarrow y).
\]

---

# 15. A0 Core Table

Produce:

| Feature | \(R^2\to g\) | \(R^2\to s_y\) | Linear Label Acc |
|---|---:|---:|---:|
| \(g\) | 1.0 | | |
| \(\tilde g\) | | | |
| D0 target | | | |
| D1 target | | | |
| D2 target | | | |
| D0 \(h\) | | | |
| D1 \(h\) | | | |
| D2 \(h\) | | | |

Produce separately for:

```text
I1
I2
I3
I4
```

---

# 16. Why Task-Score Recovery Is Critical

Current target recovery evaluates:

\[
\text{all dimensions of }g.
\]

But downstream task uses only:

\[
s_y=w^\top g.
\]

Therefore it is possible that:

\[
R^2(D2\rightarrow g)
>
R^2(D1\rightarrow g)
\]

while:

\[
\boxed{
R^2(D1\rightarrow s_y)
>
R^2(D2\rightarrow s_y).
}
\]

This would explain the current result directly.

---

# 17. Important Interpretation

If D2 recovers more of the total interaction but less of:

\[
s_y,
\]

then D2 is not necessarily worse as an interaction representation.

Instead:

\[
\boxed{
D2\text{ captures interaction broadly}
}
\]

while:

\[
\boxed{
D1\text{ happens to preserve the task-relevant direction more strongly}.
}
\]

This distinction may become a major paper insight.

---

# 18. A0 Decision Case A

If:

\[
Perf(t_{D2})
<
Perf(t_{D1})
\]

before distillation,

or:

\[
R^2(t_{D2}\rightarrow s_y)
<
R^2(t_{D1}\rightarrow s_y),
\]

then:

\[
\boxed{
\text{TARGET CONSTRUCTION IS THE PRIMARY BOTTLENECK}.
}
\]

Proceed to target calibration.

Do NOT modify the interaction loss first.

---

# 19. A0 Decision Case B

If:

\[
Perf(t_{D2})
\ge
Perf(t_{D1})
\]

but:

\[
Perf(h_{D2})
<
Perf(h_{D1}),
\]

then:

\[
\boxed{
\text{DISTILLATION IS THE PRIMARY BOTTLENECK}.
}
\]

Proceed to projection/loss calibration.

---

# 20. A0 Decision Case C

If both target and embedding D2 have lower task-score accessibility but higher full interaction recovery:

\[
\boxed{
\text{there is a fidelity–task-direction tradeoff}.
}
\]

This is not a bug.

It becomes a representation objective question.

---

# 21. A0 Uses Labels Only for Diagnosis

Task score:

\[
s_y
\]

and labels:

\[
y
\]

must NOT influence D0/D1/D2 training.

They may only be used after training to diagnose synthetic mechanism.

This preserves the unsupervised/self-supervised discovery claim.

---

# 22. No Task-Aware Training Yet

Do NOT optimize:

\[
R^2(h\rightarrow s_y)
\]

directly.

Do NOT train projection using:

\[
y.
\]

Task-aware Joint Advantage remains a future extension.

---

# 23. New Metric — Distillation Retention

If D2 target itself has positive task accessibility, define:

\[
\boxed{
Retention
=
\frac{
Perf(h)-Perf(Z_{lower})
}{
Perf(t)-Perf(Z_{lower})+\epsilon
}
}
\]

when denominator is safely positive.

This estimates how much target-level utility survives interaction distillation.

Diagnostic only.

---

# 24. Retention Interpretation

If:

\[
Retention_{D2}\ll Retention_{D1},
\]

then the interaction learner is losing D2 target utility.

Focus on distillation.

If:

\[
Retention_{D2}\approx1
\]

but D2 still loses to D1:

\[
\boxed{
\text{target construction explains the gap}.
}
\]

---

# 25. New Diagnostic — Accessibility Efficiency

Optionally report:

\[
AE=
\frac{
\Delta_{linear}
}{
R^2(feature\rightarrow g)+\epsilon
}.
\]

Use only as descriptive diagnostic.

Do not optimize or make formal claims from it.

---

# 26. Target Calibration Branch

Enter this branch only if A0 identifies target construction as the bottleneck.

First experiment:

\[
\boxed{
A1 — Standardized Joint Advantage
}
\]

---

# 27. A1 — Standardization

Current:

\[
d=q_J-q_A.
\]

Compute train-only:

\[
\mu_d,
\sigma_d.
\]

Define:

\[
\boxed{
d_{std}
=
\frac{
d-\mu_d
}{
\sigma_d+\epsilon
}.
}
\]

Statistics must be fitted on training data only.

Freeze them for validation/test.

---

# 28. Purpose of A1

Subtraction can produce dimensions with highly different scales.

Standardization asks:

> Is D2 accessibility poor because useful dimensions are badly calibrated relative to predictor-error dimensions?

A1 changes geometry, not information source.

---

# 29. A1 Hard Constraints

Keep fixed:

```text
IPIB generator
interaction rank
predictor architectures
predictor parameter counts
optimizer budget
interaction architecture
probe protocol
```

Only target normalization changes.

---

# 30. A1 Evaluation

Measure:

\[
TargetRecovery
\]

\[
TaskScoreRecovery
\]

\[
InteractionRecovery
\]

\[
\Delta_{linear}.
\]

Compare against canonical D2.

---

# 31. A2 — Whitened Joint Advantage

Only after A1.

Estimate training covariance:

\[
\Sigma_d.
\]

Use regularized whitening:

\[
W=(\Sigma_d+\lambda I)^{-1/2}.
\]

Define:

\[
\boxed{
d_{white}
=
W(d-\mu_d).
}
\]

---

# 32. Whitening Safety

Regularization:

\[
\lambda>0
\]

must prevent unstable amplification of low-variance directions.

Select:

\[
\lambda
\]

using validation only.

Do not run a broad sweep.

---

# 33. A2 Hypothesis

If predictor subtraction produces correlated error geometry, whitening may make useful joint directions easier for:

\[
h
\]

and the downstream linear probe to preserve.

---

# 34. A3 — Reliability-Weighted Joint Advantage

This is the preferred target-calibration experiment after simple normalization.

For target dimension:

\[
k,
\]

estimate:

\[
E_{A,k}
=
MSE(q_A^{(k)},x_3^{(k)})
\]

and:

\[
E_{J,k}
=
MSE(q_J^{(k)},x_3^{(k)}).
\]

Define improvement:

\[
r_k
=
\max(
0,
E_{A,k}-E_{J,k}
).
\]

---

# 35. Reliability Weights

Normalize:

\[
\bar r_k
=
\frac{
r_k
}{
mean(r)+\epsilon
}
\]

or equivalent stable normalization.

Construct:

\[
\boxed{
d^{rel}_k
=
\bar r_k
(q_{J,k}-q_{A,k}).
}
\]

---

# 36. Meaning of A3

A3 distills only difference dimensions where the joint predictor demonstrates measurable predictive advantage.

Thus:

\[
\boxed{
\text{difference magnitude}
}
\]

is no longer treated as equivalent to:

\[
\boxed{
\text{difference reliability}.
}
\]

---

# 37. Reliability Must Be Label-Free

Weights:

\[
r_k
\]

must be derived only from target prediction errors.

Never use:

```text
class labels
task score
downstream probe gradients
```

to define reliability.

---

# 38. A3 Primary Hypothesis

A3 should preserve:

\[
\text{D2 selectivity}
\]

while improving:

\[
\text{D2 accessibility}.
\]

Desired:

\[
TargetRecovery_{A3}
\approx
TargetRecovery_{D2}
\]

and:

\[
\Delta_{A3}
>
\Delta_{D2}.
\]

---

# 39. Projection Branch

Do NOT start here.

Use only if:

```text
standardization fails
reliability weighting fails
```

or A0 indicates clear dimensional geometry problems.

---

# 40. A4 — Unsupervised Target Projection

Optional later experiment.

Fit:

\[
P_d:
\mathbb R^D\rightarrow\mathbb R^m
\]

using only:

\[
d.
\]

Allowed examples:

```text
PCA
linear autoencoder
```

No labels.

---

# 41. A4 Projection Dimensions

Test at most a small predefined set:

\[
m\in\{16,32,64\}.
\]

Use validation only.

Do not broad-sweep dimensionality.

---

# 42. Purpose of A4

Reduce predictor-difference noise while preserving stable interaction structure.

Train:

\[
P_h(h)
\leftrightarrow
stopgrad(P_d(d)).
\]

---

# 43. Distillation Branch

Use this branch only if A0 shows:

\[
t_{D2}
\]

is already sufficiently accessible but:

\[
h_{D2}
\]

is not.

---

# 44. B1 — Cosine Regression

Canonical:

\[
L_{cos}
=
1-
cos(
P_h(h),
P_d(d)
).
\]

This remains baseline.

---

# 45. B2 — Standardized MSE

Define standardized features:

\[
\hat h
=
Std(h)
\]

\[
\hat d
=
Std(d).
\]

Then:

\[
\boxed{
L_{MSE}
=
\|
\hat h-\hat d
\|_2^2.
}
\]

---

# 46. B3 — Cosine + MSE

Test:

\[
\boxed{
L
=
L_{cos}
+
\lambda L_{MSE}.
}
\]

Select one small fixed:

\[
\lambda
\]

or a minimal validation comparison.

Do not broad-sweep.

---

# 47. No InfoNCE Yet

Do NOT add InfoNCE in v4.3 core.

Reason:

instance discrimination adds another mechanism and complicates diagnosis.

Only reconsider after simple regression geometry is exhausted.

---

# 48. No VICReg / Barlow Twins

Do not add:

```text
VICReg
Barlow Twins
CCA losses
large contrastive frameworks
```

during the current mechanism stage.

---

# 49. v4.3 Experimental Order

Follow exactly:

```text
A0:
Target-vs-Embedding Accessibility Audit

IF target bottleneck:
    A1 Standardization
    A2 Whitening if needed
    A3 Reliability Weighting
    A4 Projection if needed

IF distillation bottleneck:
    B1 Cosine baseline
    B2 Standardized MSE
    B3 Cosine + MSE
```

Do not run every branch automatically.

---

# 50. Do Not Conflate Calibration With Architecture

The following remain frozen:

\[
h_{12}
=
W[
LN(U_1x_1)
\odot
LN(U_2x_2)
].
\]

Rank:

\[
R=64.
\]

Do not modify interaction architecture during v4.3.

---

# 51. No Rank Sweep

Oracle and IPIB results have already ruled out rank as the primary bottleneck.

Do not sweep:

\[
R.
\]

---

# 52. No Cross-Attention

Cross-attention is not justified.

The current issue is:

\[
\boxed{
\text{geometry/accessibility}
}
\]

not lack of interaction capacity.

---

# 53. No Encoder Fine-Tuning

The synthetic benchmark does not require encoder adaptation.

Do not add it.

---

# 54. No Task Loss

Do NOT train:

\[
h
\]

using:

\[
y.
\]

This would bypass the current scientific question.

---

# 55. No Third-Order Discovery Yet

Even though Oracle third-order passes, discovery:

\[
h_{123}
\]

remains blocked.

Pair discovery/accessibility must be resolved first.

---

# 56. I0 Is a Hard Safety Gate

Canonical D2:

\[
\Delta_{I0}=0.
\]

Every v4.3 candidate must preserve:

\[
\boxed{
\Delta_{I0}\approx0.
}
\]

Also preserve:

\[
J_{I0}\approx0.
\]

---

# 57. No False-Positive Tradeoff

A method is NOT successful if it improves I1/I2 by creating significant:

\[
\Delta_{I0}>0.
\]

Interaction accessibility must remain selective.

---

# 58. I1 Is the Most Informative Regime

I1 exposes weak interaction.

Canonical:

\[
D1=39.43
\]

while:

\[
D2=25.19.
\]

This is the largest accessibility gap.

Therefore v4.3 should focus diagnosis first on I1.

---

# 59. I2 Is the Second Primary Regime

Canonical:

\[
D1=44.83
\]

\[
D2=38.78.
\]

A valid calibration should reduce this gap without harming selectivity.

---

# 60. I3 Role

I3 verifies that strong signal remains recoverable.

Canonical D2 already performs well.

Do not optimize I1/I2 at the cost of large I3 degradation.

---

# 61. I4 Role

I4 remains the private-target robustness test.

Canonical:

\[
D1\approx D2.
\]

New methods must not degrade strongly under private target noise.

---

# 62. v4.3 Primary Success Pattern

Candidate must satisfy:

### I0

\[
\Delta\approx0.
\]

### I1/I2

\[
\Delta_{candidate}>
\Delta_{D2}.
\]

Prefer:

\[
\Delta_{candidate}
\ge
\Delta_{D1}.
\]

### I3/I4

No material degradation from canonical D2.

---

# 63. Selectivity Preservation

Require:

\[
TargetRecovery_{candidate}
\]

to remain close to D2.

Do not improve accuracy by simply reverting to D1-like contamination.

---

# 64. Proposed Selectivity Preservation Ratio

Define diagnostic:

\[
SPR
=
\frac{
TargetRecovery_{candidate}
}{
TargetRecovery_{D2}+\epsilon
}.
\]

Prefer:

\[
SPR\approx1.
\]

This is a diagnostic, not a universal metric.

---

# 65. Utility Gain

Define:

\[
UG
=
\Delta_{candidate}
-
\Delta_{D2}.
\]

Goal on I1/I2:

\[
UG>0.
\]

---

# 66. Ideal v4.3 Region

The preferred candidate lies in:

\[
\boxed{
SPR\approx1
}
\]

and:

\[
\boxed{
UG>0.
}
\]

This corresponds to:

> preserve D2 interaction purity while gaining downstream accessibility.

---

# 67. Main v4.3 Plot

Plot:

X-axis:

\[
TargetRecovery
\]

Y-axis:

\[
\Delta_{linear}.
\]

Points:

```text
D0
D1
D2
A1
A2
A3
```

for I1/I2.

This directly visualizes the:

\[
\boxed{
\text{selectivity-accessibility frontier}.
}
\]

---

# 68. Important New Concept

Call this empirically:

\[
\boxed{
\text{Interaction Fidelity–Accessibility Tradeoff}
}
\]

Do NOT treat it as an information-theoretic law.

It is an empirical representation-learning phenomenon.

---

# 69. Target-Level Frontier

Also plot:

\[
R^2(target\rightarrow g)
\]

against:

\[
R^2(target\rightarrow s_y).
\]

This determines whether the tradeoff already exists before interaction distillation.

---

# 70. Embedding-Level Frontier

Plot:

\[
R^2(h\rightarrow g)
\]

against:

\[
R^2(h\rightarrow s_y).
\]

Compare D1 vs D2.

This determines how much the learner changes the frontier.

---

# 71. If D1 Has Higher Task-Score Recovery

Suppose:

\[
R^2(e\rightarrow s_y)
>
R^2(d\rightarrow s_y).
\]

Then report:

\[
\boxed{
\text{D1 preserves the task-relevant interaction direction more strongly}
}
\]

even though:

\[
D2
\]

better recovers the complete known joint component.

This is a meaningful result.

---

# 72. If D2 Target Has Higher Task-Score Recovery

Suppose:

\[
R^2(d\rightarrow s_y)
>
R^2(e\rightarrow s_y)
\]

but D2 embedding utility is lower.

Then the bottleneck is clearly:

\[
\boxed{
\text{distillation geometry}.
}
\]

Focus only on B1–B3.

---

# 73. If Calibration Cannot Beat D1

Do not force D2 to become universal.

A valid scientific outcome is:

\[
\boxed{
D1\text{ optimizes accessibility,}
\quad
D2\text{ optimizes selectivity.}
}
\]

The paper may then argue for different representations depending on the downstream requirement.

---

# 74. Possible Future Hybrid

Only after v4.3 diagnosis.

Potential:

\[
h=
[h_{D1},h_{D2}]
\]

or gated mixture.

Do NOT implement now.

A hybrid architecture would confound the current mechanism question.

---

# 75. Task-Aware Extension

Later, after unsupervised calibration is exhausted, one may define:

\[
d^{task}
=
q_{joint}^{task}
-
q_{add}^{task}.
\]

This explicitly focuses on label-relevant joint advantage.

This is a separate method family.

---

# 76. Do Not Introduce Task-Aware D2 Yet

Reason:

it would make comparison with original ConFu fundamentally different.

v4.3 must first determine the limit of representation-level calibration.

---

# 77. Statistical Protocol

A0 diagnostics:

Use all existing five seeded models where available.

Calibration exploration:

```text
seed 1
```

Validation selects candidate.

Then freeze.

Confirm:

```text
seeds 1–5
```

---

# 78. No Test-Driven Calibration

All:

```text
normalization choice
whitening regularizer
reliability formula
loss choice
projection size
```

must be selected by validation only.

Test remains final.

---

# 79. Candidate Confirmation

Once a candidate is selected:

run:

```text
I0
I1
I2
I3
I4
```

for:

\[
5\text{ seeds}.
\]

---

# 80. Report Paired Differences

For candidate vs D2:

\[
\Delta^{candidate}
-
\Delta^{D2}.
\]

For candidate vs D1:

\[
\Delta^{candidate}
-
\Delta^{D1}.
\]

Report per seed.

---

# 81. Confidence Reporting

Report:

```text
mean
sample std
paired difference
95% CI
effect size
```

where appropriate.

---

# 82. Target Geometry Diagnostics

For each D2 target variant report:

```text
variance
effective rank
mean norm
dimension std
condition number
top PCA explained variance
```

These help identify subtraction geometry issues.

---

# 83. Error Geometry Diagnostics

Compute:

\[
\epsilon_A=x_3-q_A
\]

\[
\epsilon_J=x_3-q_J.
\]

Then:

\[
d=q_J-q_A
=
\epsilon_A-\epsilon_J.
\]

Analyze covariance:

\[
Cov(\epsilon_A,\epsilon_J).
\]

This may explain why predictor subtraction amplifies or cancels specific directions.

---

# 84. Predictor Error Decomposition

Measure error separately on:

```text
additive component
joint component
private component
```

where synthetic ground truth allows.

This is highly informative.

---

# 85. Desired Predictor Behavior

Ideally:

\[
q_A\approx a_3
\]

and:

\[
q_J\approx a_3+\beta j_3.
\]

Therefore:

\[
q_J-q_A\approx\beta j_3.
\]

Measure each deviation directly.

---

# 86. Component Reconstruction Audit

Compute:

\[
R^2(q_A\rightarrow a_3)
\]

\[
R^2(q_J\rightarrow a_3)
\]

\[
R^2(q_J\rightarrow j_3)
\]

and:

\[
R^2(d\rightarrow j_3).
\]

This helps localize predictor-subtraction errors.

---

# 87. Weak-Regime Diagnostic

I1 is particularly sensitive to:

\[
\epsilon_J-\epsilon_A.
\]

When:

\[
\beta
\]

is small, predictor-error magnitude may be comparable to true:

\[
\beta j_3.
\]

This likely explains part of the D2 weakness.

---

# 88. Signal-to-Error Ratio

Define diagnostic:

\[
SER
=
\frac{
Var(\beta j_3)
}{
Var(d-\beta j_3)+\epsilon
}.
\]

Compute for I1–I4.

Do not call it information-theoretic SNR.

---

# 89. SER Hypothesis

Expected:

\[
SER_{I1}<SER_{I2}<SER_{I3}.
\]

If D2 accessibility follows this pattern, predictor-error contamination is strongly implicated.

---

# 90. Reliability Weighting Motivation

A3 is particularly justified if dimensions with low joint predictive improvement also exhibit large:

\[
d-\beta j
\]

errors.

Reliability weighting should suppress them.

---

# 91. Avoid Ground-Truth Use During Calibration

Although synthetic ground truth is available for diagnosis:

\[
g,j,s_y
\]

must not be used to train A1/A2/A3.

They are evaluation-only.

Otherwise calibration becomes Oracle-assisted.

---

# 92. Ground Truth Is Diagnostic Only

Permitted:

```text
measure R² to g
measure R² to s_y
measure component reconstruction
```

Forbidden:

```text
choose target weights using g
choose projection using y
optimize loss directly against g
```

for discovery candidates.

---

# 93. A3 Validation Rule

Reliability weights must derive from ordinary predictor validation errors only.

Do not choose per-dimension weights using Oracle recovery.

---

# 94. Reproducibility

Every v4.3 run must save:

```text
dataset seed
model seed
predictor seed
interaction seed
config
git commit
target variant
loss variant
normalization statistics
predictor metrics
target diagnostics
embedding diagnostics
accessibility metrics
```

---

# 95. Seed Rule

Seed all:

```text
dataset generation
data loader
predictor initialization
interaction initialization
probe initialization
```

where relevant.

The v4.2 reproducibility correction is now canonical.

---

# 96. Required Unit Tests

Add tests for:

```text
train-only standardization
train-only whitening
frozen normalization statistics
reliability weight computation
no label leakage
deterministic initialization
I0 zero-gate preservation
```

---

# 97. A0 Unit Test

Verify task-score diagnostic:

\[
s_y=w^\top g
\]

matches the score used by the synthetic label generator before label noise/threshold.

---

# 98. Required A0 Artifact

Save:

```text
results/synthetic_interaction/v43/a0_accessibility_audit.json
```

containing every representation × regime diagnostic.

---

# 99. Suggested New Modules

Extend:

```text
src/experiments/synthetic_interaction/
```

with:

```text
accessibility_audit.py
target_calibration.py
distillation_losses.py
v43_runner.py
```

Do not duplicate dataset logic.

---

# 100. Canonical Existing Code

Continue using:

```text
src/datasets/identifiable_interaction.py
src/experiments/synthetic_interaction/ipib.py
```

as benchmark source of truth.

---

# 101. Immediate Coding Order

Implement exactly:

```text
1. Load canonical seeded v4.2 artifacts.

2. Implement A0 feature extraction.

3. Compute target → g recovery.

4. Compute target → task-score recovery.

5. Compute target linear label accuracy.

6. Compute embedding → g recovery.

7. Compute embedding → task-score recovery.

8. Compute embedding linear label accuracy.

9. Compute Distillation Retention.

10. Produce A0 report.

11. Classify bottleneck:
    target or distillation.

12. Only then implement one calibration branch.
```

---

# 102. If Target Bottleneck Is Confirmed

Run:

```text
A1 standardized D2
```

first.

Then, only if needed:

```text
A3 reliability-weighted D2
```

Whitening is optional between them depending on geometry diagnostics.

---

# 103. If Distillation Bottleneck Is Confirmed

Run:

```text
B1 cosine baseline
B2 standardized MSE
B3 cosine + MSE
```

Do NOT change target.

---

# 104. Do Not Optimize Two Layers Simultaneously

Do not change:

```text
target construction
+
distillation loss
```

in the same first experiment.

Otherwise cause of improvement becomes uninterpretable.

---

# 105. v4.3 Candidate Naming

Use:

```text
D2-C0 — canonical
D2-C1 — standardized
D2-C2 — whitened
D2-C3 — reliability weighted
```

and:

```text
D2-L1 — cosine
D2-L2 — standardized MSE
D2-L3 — cosine + MSE
```

---

# 106. Candidate Comparison Table

Required:

| Variant | I0 | I1 | I2 | I3 | I4 | Target Recovery I1/I2 |
|---|---:|---:|---:|---:|---:|---:|
| D1 | | | | | | |
| D2-C0 | | | | | | |
| D2-C1 | | | | | | |
| D2-C3 | | | | | | |

Only include variants actually justified by A0.

---

# 107. Primary v4.3 Gate

A candidate passes if:

\[
\boxed{
\Delta_{I1/I2}>
\Delta_{D2}
}
\]

while preserving:

\[
\boxed{
\Delta_{I0}\approx0
}
\]

and:

\[
\boxed{
TargetRecovery
}
\]

near canonical D2.

---

# 108. Strong v4.3 Gate

Strong success:

\[
\Delta_{candidate}\ge\Delta_{D1}
\]

on I1/I2,

while:

\[
TargetRecovery_{candidate}
>
TargetRecovery_{D1}.
\]

This would combine:

\[
\boxed{
\text{D2 selectivity}
+
\text{D1 accessibility}.
}
\]

---

# 109. Do Not Require Universal D2 Dominance

The original gate:

\[
D2>D1
\]

in every regime is no longer mandatory as a conceptual assumption.

It becomes an empirical target.

Scientific validity does not depend on forcing it.

---

# 110. If No Calibration Beats D1

Conclude:

\[
\boxed{
\text{selectivity and accessibility trade off under current unsupervised objective}.
}
\]

This is still a meaningful result.

Do not add architecture simply to make the ranking change.

---

# 111. Third-Order Discovery Trigger

Third-order discovery remains blocked until one of:

### Condition A

v4.3 substantially closes accessibility gap;

or:

### Condition B

the selectivity-accessibility tradeoff is characterized well enough to freeze pair discovery formulation.

---

# 112. Real-World Return Trigger

Do not return to UR-FUNNY/MOSEI until:

```text
A0 diagnosis complete
v4.3 pair formulation frozen
5-seed confirmation complete
I0 control retained
```

---

# 113. Why Not Return Yet

Natural data does not provide:

\[
g
\]

or:

\[
s_y
\]

ground truth interaction decomposition.

The synthetic benchmark is currently the only place where the D1/D2 accessibility discrepancy can be mechanistically localized.

Use this advantage first.

---

# 114. Real-World Objective Later

Once frozen:

\[
d_{ij\rightarrow k}
=
q_J-q_A
\]

or calibrated version becomes the interaction discovery target.

Then screen natural datasets for:

\[
J>0.
\]

---

# 115. Real-World Screening Still Required

For each:

\[
ij\rightarrow k,
\]

measure:

\[
R^2_{add}
\]

\[
R^2_{joint}
\]

and:

\[
J.
\]

Only identifiable directions should use D2 discovery.

---

# 116. Potential Final Paper Insight

If v4.3 confirms the current pattern:

> Recovering the full multimodal interaction more faithfully does not necessarily maximize downstream task accessibility.

Then ConFu++ contributes not only a new discovery target, but a distinction between:

\[
\boxed{
\text{interaction fidelity}
}
\]

and:

\[
\boxed{
\text{interaction accessibility}.
}
\]

---

# 117. Potential Paper Framework

The final paper could organize interaction representation quality along three axes:

\[
\boxed{
\text{Identifiability}
}
\]

\[
\boxed{
\text{Fidelity}
}
\]

\[
\boxed{
\text{Accessibility}.
}
\]

This is a cleaner framework than simply reporting classification accuracy.

---

# 118. Definitions

## Identifiability

Does the target modality contain predictable non-additive structure from the source subset?

## Fidelity

Does the learned representation recover the known interaction?

## Accessibility

Can a restricted downstream model exploit that representation?

---

# 119. Three Failure Modes

A model can therefore fail because:

### Failure I

No identifiable interaction exists.

### Failure II

Interaction exists but discovery target fails to recover it.

### Failure III

Interaction is recovered but is not accessible/useful downstream.

v4.2 currently exposes Failure III most clearly.

---

# 120. v4.3 Goal

v4.3 is specifically aimed at:

\[
\boxed{
\text{Failure III}.
}
\]

Do not reopen Failure I or II unless diagnostics demand it.

---

# 121. Current Forbidden Work

Until v4.3 diagnosis/calibration is finished:

```text
NO third-order discovery
NO UR-FUNNY optimization
NO MOSEI training
NO cross-attention
NO rank sweep
NO dynamic routing
NO task loss
NO large encoder fine-tuning
NO broad contrastive loss sweep
```

---

# 122. Immediate Priority

The immediate experiment is:

\[
\boxed{
\textbf{A0 — Target-vs-Embedding Accessibility Audit}
}
\]

for:

```text
I1
I2
I3
I4
```

using canonical v4.2 artifacts.

---

# 123. A0 Must Answer One Question

\[
\boxed{
\textbf{Does D2 lose task utility before or after interaction distillation?}
}
\]

Everything else is secondary.

---

# 124. Decision Tree

```text
                 v4.2 D2
                    │
                    ▼
              A0 accessibility
                    │
          ┌─────────┴─────────┐
          │                   │
   D2 target weak       D2 target strong
   for task score       but h is weak
          │                   │
          ▼                   ▼
   Target calibration    Loss/projection
          │                calibration
     A1 / A2 / A3         B1 / B2 / B3
          │                   │
          └─────────┬─────────┘
                    ▼
              validation winner
                    │
                    ▼
                 5 seeds
                    │
                    ▼
         I0 selectivity preserved?
                    │
            ┌───────┴───────┐
            │               │
           NO              YES
            │               │
          reject         freeze v4.3
                            │
                            ▼
                    third-order / real
```

---

# 125. Current Milestone

The next milestone is NOT:

\[
D2>D1.
\]

The next milestone is:

\[
\boxed{
\text{locate the D2 accessibility loss precisely}.
}
\]

Only after localization should the project attempt to close the gap.

---

# 126. Strong Immediate Result

A strong diagnostic result would show either:

\[
\boxed{
d\text{ itself loses the task-relevant direction}
}
\]

or:

\[
\boxed{
d\text{ contains it, but distillation loses it}.
}
\]

Either outcome substantially clarifies the mechanism.

---

# 127. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ v4.3 studies why faithful interaction discovery does not automatically yield accessible downstream representations, and calibrates the discovery pipeline without sacrificing interaction selectivity.}
}
\]

---

# 128. Final Rule

Do not improve classification accuracy by sacrificing the property that v4.2 finally achieved:

\[
\boxed{
\text{selective interaction discovery}.
}
\]

The v4.3 objective is:

\[
\boxed{
\textbf{preserve D2 fidelity while recovering D1-level accessibility.}
}
\]

The next required experiment is:

\[
\boxed{
\textbf{A0 Target-vs-Embedding Accessibility Audit.}
}
\]