# AGENT.md

# ConFu++ v4.4

## Reliability Audit and Pair-Phase Freeze Decision

---

# 0. Mission

This repository develops **ConFu++**, an extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The project has now validated three distinct properties:

\[
\boxed{
\text{Architecture Capability}
}
\]

\[
\boxed{
\text{Interaction Identifiability}
}
\]

\[
\boxed{
\text{Interaction Discovery}
}
\]

The remaining pair-level question is narrower:

\[
\boxed{
\text{Can predictor reliability improve weak-interaction accessibility without sacrificing D2 selectivity?}
}
\]

This is the final planned synthetic pair-discovery question before freezing the pair formulation.

---

# 1. Current Project State

Current state:

```text
Q1 — Architecture capability:
PASS

Q2a — Target identifiability:
PASS

Q2b — Pair interaction discovery:
PASS

Q2c — Pair interaction selectivity:
PASS for D2

Q2d — Weak/medium accessibility:
PARTIAL

D2 > D0:
PASS

D2 > D1:
PARTIAL

v4.3 geometry calibration:
INSUFFICIENT

C0 reliability audit:
NEXT

C1 reliability-weighted D2:
CONDITIONAL

Third-order discovery:
BLOCKED

Real-world validation:
BLOCKED UNTIL PAIR FREEZE
```

---

# 2. Core Scientific Separation

The project must distinguish:

\[
\boxed{
\text{Identifiability}
}
\]

from:

\[
\boxed{
\text{Interaction Fidelity}
}
\]

from:

\[
\boxed{
\text{Task Accessibility}.
}
\]

These are separate properties.

A representation may:

- recover the true interaction well;
- be highly selective;
- avoid false interaction;

while still being less useful to a linear downstream classifier than a less selective representation.

This is now empirically established.

---

# 3. Architecture Capability

The Oracle benchmark has already shown that the low-rank multiplicative architecture can represent:

\[
h_{12}
\]

\[
h_{13}
\]

\[
h_{23}
\]

and:

\[
h_{123}.
\]

Oracle pair gains are approximately:

\[
\Delta_2\approx29\text{ pp}.
\]

Oracle pure triple:

\[
Z_{\le1}\approx50\%
\]

\[
Z_{\le2}\approx50\%
\]

\[
Z_{\le3}\approx93.5\%.
\]

Thus:

\[
\boxed{
\text{architecture is not the current bottleneck}.
}
\]

---

# 4. Non-Identifiable Control

The original IID benchmark has independent:

\[
z_1,z_2,z_3.
\]

For:

\[
12\rightarrow3,
\]

there is no population-level predictive relation:

\[
R^2((z_1,z_2)\rightarrow z_3)\approx0.
\]

This benchmark is retained as:

\[
\boxed{
\text{NIC — Non-Identifiable Control}.
}
\]

It must not be interpreted as model failure.

---

# 5. IPIB

The valid discovery benchmark is:

\[
\boxed{
\text{Identifiable Predictive-Interaction Benchmark}
}
\]

with:

\[
x_3
=
a_3
+
\beta j_3
+
\lambda_p p_3.
\]

Where:

\[
a_3
=
A_{31}z_1+A_{32}z_2
\]

is additive predictable structure,

\[
j_3
=
B_3(z_1\odot z_2)
\]

is joint predictable structure,

and:

\[
p_3=P_3u_3
\]

is private unpredictable structure.

---

# 6. Canonical Pair Ground Truth

Define:

\[
g_{12}=z_1\odot z_2.
\]

Projected interaction:

\[
\tilde g_{12}=B_3g_{12}.
\]

Task score:

\[
s_y=w^\top g_{12}.
\]

Labels:

\[
y=\mathbf1[s_y+\epsilon_y>0].
\]

Important:

\[
y
\]

and:

\[
s_y
\]

must never be used during discovery training.

They are synthetic diagnostics only.

---

# 7. IPIB Regimes

Canonical regimes:

| Regime | \(\beta\) | Private strength | Purpose |
|---|---:|---:|---|
| I0 | 0 | 1 | false-positive control |
| I1 | 0.25 | 1 | weak interaction |
| I2 | 0.5 | 1 | medium interaction |
| I3 | 1.0 | 1 | strong interaction |
| I4 | 1.0 | 2 | private-target robustness |

---

# 8. Discovery Methods

## D0 — Original ConFu

Target:

\[
t_{D0}=x_3.
\]

Train:

\[
h_{12}\leftrightarrow x_3.
\]

---

# 9. D1 — Residual Alignment

Define:

\[
q_A=q_1(x_1)+q_2(x_2).
\]

Target:

\[
\boxed{
e=x_3-q_A.
}
\]

Train:

\[
h_{12}\leftrightarrow e.
\]

D1 removes additive predictable structure but retains:

\[
\text{joint}
+
\text{private}
+
\text{predictor error}.
\]

---

# 10. D2 — Joint-Advantage Distillation

Train capacity-matched joint predictor:

\[
q_J(x_1,x_2).
\]

Target:

\[
\boxed{
d=q_J-q_A.
}
\]

Train:

\[
h_{12}\leftrightarrow stopgrad(d).
\]

D2 is designed to retain:

\[
\boxed{
\text{predictable non-additive structure}.
}
\]

---

# 11. Canonical v4.2 Result

Five-seed linear gains:

| Regime | D0 | D1 | D2 |
|---|---:|---:|---:|
| I0 | 4.12 | 6.12 | **0.00** |
| I1 | 13.52 | **39.43** | 25.19 |
| I2 | 31.88 | **44.83** | 38.78 |
| I3 | 46.40 | 47.00 | **47.12** |
| I4 | 44.06 | 44.66 | **44.78** |

Interpretation:

\[
\boxed{
D2>D0
}
\]

in every positive regime.

But:

\[
\boxed{
D1>D2
}
\]

substantially in I1 and I2.

---

# 12. D2 False-Positive Control

I0:

\[
\boxed{
\Delta_{D2}=0.00
}
\]

after validation identifiability gating.

This is a major strength.

Any new pair-discovery variant must preserve:

\[
\boxed{
\Delta_{I0}\approx0.
}
\]

---

# 13. Target Recovery

Canonical target recovery:

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
D2\text{ is dramatically more selective for the known interaction}.
}
\]

---

# 14. Interaction Recovery

Learned D2 embeddings also recover the true pair interaction.

Canonical approximate D2:

\[
R^2(h_{D2}\rightarrow g)
=
0.562,\;
0.841,\;
0.960,\;
0.934
\]

for I1–I4.

Therefore:

\[
\boxed{
D2\text{ successfully discovers pair interaction}.
}
\]

---

# 15. v4.3 A0 Diagnosis

A0 measured:

\[
R^2(feature\rightarrow s_y)
\]

for targets and embeddings.

Five-seed task-score recovery:

| Regime | D1 target | D2 target | D1 embedding | D2 embedding |
|---|---:|---:|---:|---:|
| I1 | 0.054 | 0.470 | **0.854** | 0.498 |
| I2 | 0.194 | 0.858 | **0.963** | 0.849 |
| I3 | 0.493 | 0.980 | **0.984** | 0.960 |
| I4 | 0.194 | 0.955 | **0.962** | 0.942 |

---

# 16. A0 Key Finding

D2 target contains strong task-relevant signal.

Furthermore:

\[
R^2(d\rightarrow s_y)
\approx
R^2(h_{D2}\rightarrow s_y).
\]

Thus:

\[
\boxed{
\text{canonical D2 does not catastrophically lose its own target utility during distillation}.
}
\]

The D2 target itself is meaningful.

---

# 17. D1 Paradox

D1 target itself has low linear task-score recovery:

\[
R^2(e\rightarrow s_y)
\ll
R^2(d\rightarrow s_y).
\]

Yet:

\[
h_{D1}
\]

has extremely high task-score recovery.

Therefore:

\[
\boxed{
\text{the nonlinear D1 interaction learner extracts highly useful geometry from a contaminated residual target}.
}
\]

This is a major empirical finding.

---

# 18. Consequence

The previous simple hypothesis:

```text
D2 target good
→ distillation bad
```

is incomplete.

Instead:

\[
\boxed{
D1\text{ receives a favorable nonlinear inductive bias}
}
\]

that D2 does not reproduce.

---

# 19. v4.3 B-Branch

v4.3 tested distillation-loss calibration.

Variants:

### B1

Canonical cosine regression.

### B2

Standardized MSE.

### B3

Cosine + standardized MSE.

---

# 20. B2 Selection

Seed-1 validation target recovery selected B2 in I1–I4.

B2 therefore received full five-seed confirmation.

Selection used:

\[
\boxed{
\text{validation target recovery only}
}
\]

with no label/task-score selection.

---

# 21. B2 Result

Five-seed linear gains:

| Regime | Canonical D2 | B2 | D1 |
|---|---:|---:|---:|
| I0 | 0.00 | **0.00** | 6.12 |
| I1 | 25.19 | **25.44** | 39.43 |
| I2 | 38.78 | **38.86** | 44.83 |
| I3 | **47.12** | 47.07 | 47.00 |
| I4 | **44.78** | 44.69 | 44.66 |

---

# 22. B2 Effect

Paired B2 − D2:

### I1

\[
+0.252\text{ pp}
\]

with:

\[
p=0.0283.
\]

### I2

\[
+0.076\text{ pp}.
\]

### I3

\[
-0.048\text{ pp}.
\]

### I4

\[
-0.092\text{ pp}.
\]

---

# 23. B2 Interpretation

Although I1 is statistically detectable:

\[
+0.252\text{ pp}
\]

is tiny relative to the approximately:

\[
14\text{ pp}
\]

D1–D2 gap.

Therefore:

\[
\boxed{
\text{standardized MSE does not solve the accessibility problem}.
}
\]

---

# 24. B2 Fidelity

B2 improves embedding fidelity:

\[
R^2(h_{B2}\rightarrow g)
\]

to approximately:

\[
0.570,\;
0.863,\;
0.978,\;
0.952.
\]

But classification utility barely changes.

Thus:

\[
\boxed{
\text{better interaction fidelity does not automatically yield better accessibility}.
}
\]

---

# 25. B2 Rank

Effective rank:

\[
18.1,\;
19.7,\;
23.4,\;
22.9
\]

for I1–I4.

This is lower than canonical D2.

Therefore B2 changes representation geometry substantially.

It is not a clean universal improvement.

---

# 26. v4.3 Decision

The B-branch is now CLOSED.

Do NOT continue:

```text
additional MSE weights
large loss grids
InfoNCE
VICReg
Barlow Twins
CCA losses
contrastive temperature sweeps
```

The loss is not the main unresolved mechanism.

---

# 27. Current Bottleneck

Evidence now suggests:

\[
\boxed{
\text{weak-regime predictor subtraction reliability}
}
\]

plus:

\[
\boxed{
\text{fidelity–accessibility tradeoff}.
}
\]

---

# 28. Signal-to-Error Ratio

Measured D2 signal-to-error diagnostic:

\[
SER_{I1}=1.584
\]

\[
SER_{I2}=5.823
\]

\[
SER_{I3}=37.340
\]

\[
SER_{I4}=14.753.
\]

This strongly matches D2's behavior.

---

# 29. SER Interpretation

When:

\[
\beta
\]

is weak,

\[
d=q_J-q_A
\]

contains a true interaction signal whose magnitude is comparable to predictor-difference error.

In strong regimes:

\[
\beta j
\]

dominates predictor error.

This provides a plausible explanation for:

\[
D2_{I1/I2}<D1_{I1/I2}.
\]

---

# 30. New Final Pair-Level Question

The final synthetic pair question is:

\[
\boxed{
\textbf{Can predictor reliability identify which D2 target dimensions are trustworthy?}
}
\]

This is tested by:

\[
\boxed{
C0\text{ — Predictor Reliability Audit}.
}
\]

---

# 31. C0 Is Diagnostic Only

C0 must NOT train a new interaction representation.

Use already trained:

\[
q_A
\]

and:

\[
q_J.
\]

Purpose:

> determine whether an entirely label-free reliability signal predicts actual D2 target fidelity.

---

# 32. Per-Dimension Additive Error

For target dimension:

\[
k,
\]

compute on validation data:

\[
E_{A,k}
=
\frac1N
\sum_n
(
q_{A,k}^{(n)}-x_{3,k}^{(n)}
)^2.
\]

---

# 33. Per-Dimension Joint Error

Compute:

\[
E_{J,k}
=
\frac1N
\sum_n
(
q_{J,k}^{(n)}-x_{3,k}^{(n)}
)^2.
\]

---

# 34. Reliability Score

Define:

\[
\boxed{
R_k
=
\frac{
\max(0,E_{A,k}-E_{J,k})
}{
E_{A,k}+\epsilon
}.
}
\]

Interpretation:

\[
R_k
\]

measures how much the joint predictor improves over the additive predictor for target dimension \(k\).

---

# 35. Reliability Is Label-Free

C0 reliability computation may use only:

```text
x3
q_add
q_joint
```

It must NOT use:

```text
y
s_y
g12
projected oracle interaction
```

during score construction.

---

# 36. Ground Truth Is Allowed for Audit

Because this is synthetic mechanism diagnosis, after computing:

\[
R_k,
\]

ground truth may be used for evaluation.

This does not affect discovery training.

---

# 37. Per-Dimension True Fidelity

Define:

\[
d_k=q_{J,k}-q_{A,k}.
\]

Known projected joint component:

\[
\tilde g_k.
\]

Define:

\[
\boxed{
F_k
=
1-
\frac{
MSE(d_k,\tilde g_k)
}{
Var(\tilde g_k)+\epsilon
}.
}
\]

This is diagnostic only.

---

# 38. Alternative Fidelity Metric

Also compute:

\[
corr(d_k,\tilde g_k)
\]

where numerically meaningful.

Do not rely on one metric alone.

---

# 39. Main C0 Test

Measure:

\[
\boxed{
Corr(R_k,F_k).
}
\]

Use:

```text
Pearson correlation
Spearman rank correlation
```

across output dimensions.

---

# 40. Why Spearman Matters

C1 requires relative ranking of target dimensions more than exact linear relationship.

Therefore:

\[
\boxed{
Spearman(R,F)
}
\]

is particularly important.

---

# 41. C0 Across Seeds

Compute:

\[
Corr(R,F)
\]

for every:

```text
seed × regime
```

for:

\[
I1,I2,I3,I4.
\]

Report per seed and mean.

---

# 42. I1 Is Primary

I1 has the lowest SER.

Therefore if reliability scoring is meaningful, it should reveal the clearest distinction between:

```text
trustworthy D2 dimensions
unreliable D2 dimensions
```

in I1.

---

# 43. C0 Quantile Audit

Divide dimensions into:

```text
top 25% reliability
middle 50%
bottom 25%
```

or quartiles.

For each group report:

\[
R^2(d_{group}\rightarrow\tilde g_{group})
\]

or equivalent fidelity.

---

# 44. Expected Pattern

Desired:

\[
F_{top}>F_{middle}>F_{bottom}.
\]

This is stronger evidence than correlation alone.

---

# 45. Top-vs-Bottom Fidelity Gap

Define:

\[
\boxed{
FG
=
F_{top25}
-
F_{bottom25}.
}
\]

A consistently positive:

\[
FG
\]

supports reliability weighting.

---

# 46. Reliability Concentration

Report distribution of:

\[
R_k.
\]

Diagnostics:

```text
mean
median
std
min
max
fraction exactly zero
top-10% mass
```

This detects degenerate weighting.

---

# 47. Avoid Single-Dimension Noise

Reliability estimates can be noisy.

Do not make claims from individual dimensions.

Use aggregate behavior across:

```text
dimensions
seeds
regimes
```

---

# 48. C0 Pass Criterion

C0 passes if reliability score demonstrates a stable positive relationship with ground-truth fidelity.

Prefer:

\[
Spearman(R,F)>0
\]

for most:

```text
seed × positive regime
```

and:

\[
FG>0
\]

consistently.

Do not require an arbitrary huge correlation.

---

# 49. C0 Strong Pass

Strong evidence:

\[
Spearman(R,F)
\]

positive in:

```text
≥ 4/5 seeds
```

for I1 and I2,

with positive top-bottom fidelity gap.

---

# 50. C0 Fail

If reliability does not predict true fidelity:

\[
\boxed{
\text{DO NOT RUN C1}.
}
\]

This is mandatory.

Do not train a reliability-weighted model merely because the method sounds reasonable.

---

# 51. C0 Failure Interpretation

If C0 fails:

\[
\boxed{
\text{simple predictor advantage is not sufficient to identify reliable D2 target dimensions}.
}
\]

Then pair-level unsupervised calibration is frozen.

---

# 52. C1 Trigger

Run C1 only if:

\[
\boxed{
C0=PASS.
}
\]

C1 is:

\[
\boxed{
\text{Reliability-Weighted Joint-Advantage Distillation}.
}
\]

---

# 53. C1 Target

Start from:

\[
d=q_J-q_A.
\]

Define reliability:

\[
R_k.
\]

Normalize:

\[
\bar R_k
=
\frac{
R_k
}{
mean(R)+\epsilon
}.
\]

Then:

\[
\boxed{
d_k^{rel}
=
\bar R_k d_k.
}
\]

---

# 54. Why Mean Normalization

Without normalization, weighting can reduce target magnitude globally.

Mean normalization aims to change:

\[
\boxed{
\text{relative dimensional emphasis}
}
\]

rather than simply shrink the target.

---

# 55. C1 Exponent

Initial:

\[
\boxed{
\alpha=1.
}
\]

General form:

\[
d_k^{rel}
=
\bar R_k^\alpha d_k.
\]

Do NOT sweep:

\[
\alpha
\]

initially.

---

# 56. No Ground-Truth Weighting

Do NOT define:

\[
R_k
\]

using:

```text
g12
g_tilde
task score
label
downstream accuracy
```

C1 must remain label-free and Oracle-free.

---

# 57. C1 Loss

Use canonical D2 cosine regression.

Do NOT combine C1 with B2.

Therefore:

\[
\boxed{
\text{only the target changes}.
}
\]

This isolates target reliability as the experimental factor.

---

# 58. C1 Architecture

Keep unchanged:

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

No architectural changes.

---

# 59. C1 Seed-1 Protocol

Run seed 1 first on:

```text
I0
I1
I2
I3
I4
```

but use validation only for decision.

---

# 60. I0 Behavior

Because D2 identifiability gate already disables:

\[
d
\]

when joint advantage is absent,

C1 must preserve:

\[
\boxed{
\Delta_{I0}=0.
}
\]

Any non-zero I0 behavior is immediate rejection.

---

# 61. C1 Primary Objective

Improve weak/medium accessibility:

\[
\Delta_{I1}
\]

and:

\[
\Delta_{I2}.
\]

Without sacrificing:

\[
\text{selectivity}
\]

or strong-regime performance.

---

# 62. C1 Minimum Practical Effect

Do not promote C1 for tiny effects comparable to B2.

A meaningful candidate should ideally improve weak/medium gain by approximately:

\[
\boxed{
>1\text{ pp}
}
\]

on validation/test confirmation.

This is a practical heuristic, not a statistical law.

---

# 63. Why Practical Magnitude Matters

B2 showed that:

\[
p<0.05
\]

can coexist with an effect too small to matter scientifically.

Therefore prioritize:

\[
\boxed{
\text{effect magnitude}
}
\]

over thresholded significance.

---

# 64. C1 Seed-1 Gate

Proceed to five seeds only if:

### I0

\[
\Delta=0.
\]

### I1 or I2

meaningful positive improvement over canonical D2.

### I3/I4

no obvious degradation.

### Fidelity

interaction recovery remains healthy.

---

# 65. Five-Seed C1 Confirmation

If seed-1 gate passes:

freeze:

```text
reliability formula
normalization
alpha
loss
rank
optimizer
training budget
```

then run:

\[
5\text{ seeds}.
\]

---

# 66. Five-Seed Primary Table

Required:

| Regime | D1 | D2 | C1 | C1−D2 | C1−D1 |
|---|---:|---:|---:|---:|---:|
| I0 | | | | | |
| I1 | | | | | |
| I2 | | | | | |
| I3 | | | | | |
| I4 | | | | | |

---

# 67. C1 Fidelity Table

Report:

| Regime | D2 Target Recovery | C1 Target Recovery | D2 Embedding Recovery | C1 Embedding Recovery |
|---|---:|---:|---:|---:|

---

# 68. C1 Task-Score Table

Report:

\[
R^2(h\rightarrow s_y)
\]

for D1, D2 and C1.

This remains diagnostic only.

---

# 69. C1 Pass

C1 is considered useful if it:

1. preserves I0 zero control;
2. improves D2 accessibility meaningfully in I1/I2;
3. preserves interaction fidelity;
4. avoids material degradation in I3/I4.

---

# 70. Strong C1 Success

Strong success means:

\[
\Delta_{C1}\approx\Delta_{D1}
\]

or exceeds it on I1/I2,

while:

\[
TargetRecovery_{C1}
\]

remains close to D2 and far above D1.

This would achieve:

\[
\boxed{
\text{D2 selectivity + D1 accessibility}.
}
\]

---

# 71. C1 Partial Success

If C1 improves D2 by a meaningful amount but does not reach D1:

\[
\boxed{
\text{reliability contributes to the gap but does not fully explain it}.
}
\]

This is still useful.

---

# 72. C1 Failure

If C1 gives:

```text
tiny gain
no gain
unstable gain
I0 leakage
strong-regime degradation
```

then C1 fails.

Do NOT create C2/C3/C4 weighting variants.

---

# 73. Pair-Phase Freeze Rule

If either:

```text
C0 fails
```

or:

```text
C1 fails to produce meaningful improvement
```

then:

\[
\boxed{
\textbf{FREEZE SYNTHETIC PAIR DISCOVERY}.
}
\]

No further unsupervised pair hyperparameter search.

---

# 74. Why Freeze Is Important

The project already has enough evidence to establish:

```text
architecture works
identifiability matters
D2 discovers interaction
D2 is selective
D1 is more accessible in weak regimes
simple loss calibration is insufficient
```

Continuing endless calibration would reduce scientific clarity.

---

# 75. Frozen Pair Conclusion

If C1 fails, canonical conclusion:

\[
\boxed{
\text{D2 is a selectivity-oriented interaction discovery objective}.
}
\]

And:

\[
\boxed{
\text{D1 is a less selective but more accessibility-oriented residual objective}.
}
\]

---

# 76. Fidelity–Accessibility Tradeoff

The project may then formally report an empirical:

\[
\boxed{
\text{Interaction Fidelity–Accessibility Tradeoff}.
}
\]

Meaning:

> A representation that more faithfully isolates the full ground-truth interaction does not necessarily maximize linear accessibility for a specific downstream label.

This is empirical, not information-theoretic.

---

# 77. Three-Axis Framework

The project should characterize methods using:

### Identifiability

Can the interaction be predicted from the source subset?

### Fidelity

Does the learned representation recover the interaction?

### Accessibility

Can a restricted downstream model exploit it?

---

# 78. Method Positioning

Approximate synthetic behavior:

### D0

```text
moderate fidelity
low weak-regime accessibility
false-positive behavior
```

### D1

```text
low target selectivity
very high downstream accessibility
false-positive behavior
```

### D2

```text
high target selectivity
strong interaction fidelity
perfect I0 control
moderate weak-regime accessibility
```

---

# 79. Important Scientific Result

The project has demonstrated:

\[
\boxed{
\text{higher interaction fidelity}
\not\Rightarrow
\text{higher task accessibility}.
}
\]

This should remain central regardless of C1 outcome.

---

# 80. Do Not Optimize Against Synthetic Labels

Even after seeing D1's task advantage, do NOT modify C1 based on:

\[
y
\]

or:

\[
s_y.
\]

Otherwise the objective becomes supervised.

---

# 81. Task-Aware Future Branch

A later method may explicitly define:

\[
d^{task}
=
q_J^{task}-q_A^{task}.
\]

This would target:

\[
\text{label-relevant joint advantage}.
\]

It is a separate method family.

Do not mix it into v4.4.

---

# 82. Third-Order Discovery

Third-order discovery remains blocked until the pair phase is frozen.

After freeze:

\[
\boxed{
\text{do NOT immediately implement }h_{123}\text{ discovery}.
}
\]

First validate the frozen pair objective on natural data.

---

# 83. Why Real-World Before Third-Order

Pair discovery now has extensive synthetic evidence.

The next important question becomes:

\[
\boxed{
\text{Does this behavior transfer to natural multimodal representations?}
}
\]

before increasing order complexity.

---

# 84. Real-World Phase Trigger

Enter real-world phase after:

```text
C0 complete
C1 either confirmed or rejected
pair formulation frozen
five-seed synthetic report complete
```

---

# 85. Preferred Real-World First Step

Do NOT train ConFu++ immediately.

Run:

\[
\boxed{
\text{Real-World Identifiability Screening}
}
\]

first.

---

# 86. Preferred Dataset

First candidate:

\[
\boxed{
CMU\text{-}MOSEI\ Emotion
}
\]

before returning to UR-FUNNY.

This is a screening hypothesis, not an assumption of better performance.

---

# 87. Why MOSEI Emotion

UR-FUNNY has already shown:

```text
text dominance
small multimodal headroom
unstable pair accessibility
```

MOSEI Emotion may contain stronger:

```text
audio
visual
cross-modal
```

predictive structure for some emotion labels.

This must be measured, not assumed.

---

# 88. MOSEI Identifiability Audit

For each mapping:

\[
12\rightarrow3
\]

\[
13\rightarrow2
\]

\[
23\rightarrow1,
\]

train:

### Nonlinear additive predictor

\[
q_A.
\]

### Capacity-matched joint predictor

\[
q_J.
\]

---

# 89. Real-World Joint Advantage

Compute:

\[
\boxed{
J_{ij\rightarrow k}
=
R^2(q_J,r_k)
-
R^2(q_A,r_k).
}
\]

Use validation and test separately.

---

# 90. Real-World Mapping Categories

Classify each direction:

### Non-identifiable

\[
R^2_J\approx0.
\]

### Additively identifiable

\[
R^2_A>0
\]

but:

\[
J\approx0.
\]

### Jointly identifiable

\[
\boxed{
J>0.
}
\]

---

# 91. Only Jointly Identifiable Directions Matter for D2

Do not train D2 on mappings where:

\[
J\approx0.
\]

This avoids repeating the v4.1 identifiability mistake.

---

# 92. MOSEI Emotion Screening

Audit individual emotions where possible:

```text
happiness
sadness
anger
surprise
disgust
fear
```

Do not assume sentiment-style text dominance applies equally.

---

# 93. Screening Output

Required table:

| Task | Mapping | Additive R² | Joint R² | J |
|---|---|---:|---:|---:|
| happiness | V+A→T | | | |
| happiness | V+T→A | | | |
| ... | ... | | | |

---

# 94. Dataset Selection Rule

Choose real-world development settings only where:

\[
J
\]

is:

```text
positive
stable across seeds
not purely train-set overfit
```

---

# 95. Do Not Rank Tasks by “Synergy”

Do not claim:

> emotion X is more synergistic.

Use:

\[
\boxed{
\text{higher measured joint predictive advantage under this predictor family}.
}
\]

---

# 96. Real-World D1/D2 Comparison

Once mappings are selected:

compare:

```text
Original ConFu
D1 residual
D2 canonical or frozen C1
```

under identical encoders and interaction architecture.

---

# 97. Real-World Metrics

Evaluate:

```text
classification accuracy
F1 where appropriate
linear accessibility
nonlinear probe
retrieval
interaction health
modality shuffle
joint predictor advantage
```

---

# 98. Real-World Selectivity

Ground-truth interaction:

\[
g
\]

is unavailable.

Therefore synthetic target fidelity cannot be measured.

Use indirect diagnostics:

```text
J
single-modality predictability
shuffle dependence
linear utility
retrieval
```

---

# 99. Preserve Original ConFu Evaluation

Where available retain:

```text
1→1 retrieval
2→1 retrieval
```

so ConFu++ does not improve classification by destroying cross-modal alignment.

---

# 100. UR-FUNNY Return

UR-FUNNY is not abandoned.

It becomes:

\[
\boxed{
\text{later transfer/generalization benchmark}.
}
\]

Return after:

```text
pair formulation frozen
MOSEI screening complete
at least one jointly identifiable natural setting found
```

---

# 101. No UR-FUNNY Tuning Before Screening

Do not directly optimize:

\[
64.462\%\rightarrow X.
\]

First determine whether relevant mappings have:

\[
J>0.
\]

---

# 102. No Third-Order Real-World Claim Yet

Even though Oracle:

\[
h_{123}
\]

works, real-world order-3 discovery remains future work until pair transfer is demonstrated.

---

# 103. Current Forbidden Work

Until C0/C1 decision:

```text
NO InfoNCE
NO VICReg
NO Barlow Twins
NO rank sweep
NO cross-attention
NO dynamic routing
NO task loss
NO encoder fine-tuning
NO h123 discovery
NO UR-FUNNY tuning
NO MOSEI ConFu++ training
```

---

# 104. C0 Repository Task

Create or extend:

```text
src/experiments/synthetic_interaction/reliability_audit.py
```

---

# 105. C1 Repository Task

Only if C0 passes:

```text
src/experiments/synthetic_interaction/reliability_weighted.py
```

or add a clearly isolated target variant to the canonical runner.

---

# 106. Avoid Code Duplication

Reuse:

```text
identifiable_interaction.py
ipib.py
predictor checkpoints
canonical probe utilities
```

Do not create a second benchmark implementation.

---

# 107. Required C0 Artifact

Save:

```text
results/synthetic_interaction/v44/
    C0_RELIABILITY_AUDIT.md
```

and machine-readable:

```text
c0_reliability_audit.json
```

---

# 108. Required C0 Fields

For each:

```text
regime
seed
dimension
```

save:

```text
E_add
E_joint
R
oracle_fidelity
oracle_correlation
```

where Oracle fields are diagnostic only.

---

# 109. Required C0 Aggregate

Report:

```text
Pearson R-vs-F
Spearman R-vs-F
top-quartile fidelity
bottom-quartile fidelity
fidelity gap
fraction reliability zero
```

for each regime.

---

# 110. C1 Artifact

If run:

```text
results/synthetic_interaction/v44/c1_seed1/
```

then if confirmed:

```text
results/synthetic_interaction/v44/c1_final/
```

---

# 111. Reproducibility

Every run saves:

```text
dataset seed
predictor seed
interaction seed
probe seed
git commit
config
validation selection
reliability formula
normalization
```

---

# 112. Seed Discipline

Seed:

```text
Python
NumPy
PyTorch CPU
PyTorch CUDA
DataLoader workers
model initialization
```

where applicable.

The v4.2 seed correction remains canonical.

---

# 113. Unit Tests

Add:

```text
reliability uses validation only
no label leakage
no oracle leakage into weighting
same seed reproduces R
mean reliability normalization correct
I0 remains gated
```

---

# 114. Leakage Test

Explicitly assert that:

\[
R_k
\]

computation has no access to:

```text
label
task score
g12
B3g12
test set
```

---

# 115. C0 Statistical Caution

Dimensions inside one model are not independent experimental replicates.

Therefore:

\[
Corr(R,F)
\]

is a diagnostic.

Primary robustness comes from consistency across:

\[
\boxed{
\text{seeds and regimes}.
}
\]

---

# 116. C1 Statistical Reporting

For five-seed C1 report:

```text
mean
sample std
paired C1−D2
paired C1−D1
95% CI
effect size
```

Do not overemphasize p-values with \(n=5\).

---

# 117. Freeze Decision Artifact

After C0/C1:

create:

```text
PAIR_DISCOVERY_FREEZE_REPORT.md
```

This is mandatory.

---

# 118. Freeze Report Contents

Include:

```text
Architecture findings
Identifiability findings
D0 results
D1 results
D2 results
A0 diagnosis
B2 result
C0 result
C1 result if run
Final frozen formulation
Known limitations
Real-world entry criteria
```

---

# 119. Frozen Method Naming

If canonical D2 remains best scientific choice:

\[
\boxed{
\text{Joint-Advantage Distillation}
}
\]

remains the main method.

---

# 120. If C1 Wins

Rename only if the reliability mechanism is clearly supported.

Possible:

\[
\boxed{
\text{Reliability-Weighted Joint-Advantage Distillation}
}
\]

Do not rename for tiny empirical differences.

---

# 121. Main Paper Claim If Canonical D2 Is Frozen

Safe claim:

> Joint-Advantage Distillation isolates predictable non-additive multimodal structure more selectively than direct or residual alignment on a controlled identifiable benchmark.

---

# 122. Additional Safe Claim

> More faithful interaction recovery does not necessarily translate into greater downstream linear accessibility.

---

# 123. Unsafe Claim

Do NOT claim:

```text
D2 universally outperforms residual alignment
D2 always improves classification
D2 recovers formal information-theoretic synergy
D2 creates new information
```

---

# 124. D1 Interpretation

D1 should not be dismissed as “bad” because it is contaminated.

Its contaminated residual may provide a useful inductive bias for downstream task geometry.

This is empirically important.

---

# 125. D2 Interpretation

D2 should not be judged only by classification gain.

Its strengths include:

```text
target selectivity
false-positive suppression
interaction fidelity
predictable non-additive isolation
```

---

# 126. Research Framework

The emerging research thesis is:

\[
\boxed{
\text{Higher-order multimodal representation learning requires distinguishing identifiability, fidelity, and accessibility.}
}
\]

Original higher-order alignment alone does not make this distinction.

---

# 127. Pair-Phase Final Question

The only remaining synthetic pair question is:

\[
\boxed{
\textbf{Can label-free predictor reliability improve D2 accessibility in weak-interaction regimes?}
}
\]

---

# 128. Immediate Execution Order

Run exactly:

```text
1. Implement C0 reliability audit.

2. Use existing five-seed D2 predictor artifacts.

3. Compute per-dimension additive error.

4. Compute per-dimension joint error.

5. Compute reliability R.

6. Compute Oracle fidelity only after R is fixed.

7. Measure Pearson/Spearman R-vs-F.

8. Measure top/bottom reliability fidelity gap.

9. Produce C0 report.

10. Decide PASS/FAIL.

11. If FAIL:
    freeze pair phase.

12. If PASS:
    implement C1 with alpha=1.

13. Run C1 seed 1.

14. Require meaningful validation improvement.

15. If weak:
    freeze pair phase.

16. If strong:
    freeze C1 configuration.

17. Run C1 seeds 1–5.

18. Produce pair freeze report.

19. Begin real-world identifiability screening.
```

---

# 129. No Extra C1 Search

If:

\[
\alpha=1
\]

fails:

\[
\boxed{
\text{STOP}.
}
\]

Do not immediately test:

```text
alpha 0.25
alpha 0.5
alpha 2
softmax weighting
top-k weighting
threshold weighting
```

unless a new theoretical reason emerges.

---

# 130. Reason for Hard Stop

The purpose of C1 is to test a mechanism:

\[
\boxed{
\text{reliability weighting}
}
\]

not to search for a lucky hyperparameter.

---

# 131. Pair Phase Success Is Already Partial

Even if C1 fails, the pair phase has established:

1. architecture can represent pair interaction;
2. cross-modal identifiability is necessary;
3. IPIB provides a valid controlled benchmark;
4. D2 isolates known interaction;
5. D2 suppresses false positive interaction;
6. D1 and D2 occupy different fidelity/accessibility regimes.

Therefore C1 failure does not invalidate the project.

---

# 132. Real-World Objective After Freeze

Once pair formulation is frozen:

\[
\boxed{
\text{screen first, train second}.
}
\]

Do not assume every multimodal mapping contains discoverable joint structure.

---

# 133. Real-World Entry Question

For each dataset:

\[
\boxed{
\text{Does }q_J\text{ predict target modality better than capacity-matched }q_A?
}
\]

Only then run interaction discovery.

---

# 134. Main Real-World Hypothesis

Synthetic evidence predicts:

- D2 may be especially useful where false interaction is a concern;
- D1 may remain strong where task accessibility dominates;
- stronger joint-identifiable mappings may reduce the D1–D2 gap.

This must be tested empirically.

---

# 135. Third-Order Future Hypothesis

After real-world pair transfer:

future order-3 discovery may compare:

\[
q_{123}
\]

against a predictor containing all validated:

\[
\text{order-1}
+
\text{order-2}
\]

components.

Potential target:

\[
d_{123}=q_{123}-q_{\le2}.
\]

Do not implement yet.

---

# 136. Final Scientific Principle

A higher-order representation should never be judged solely by:

```text
variance
rank
classification accuracy
target similarity
```

Instead evaluate:

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

---

# 137. Immediate Milestone

The next milestone is NOT:

\[
D2>D1.
\]

It is:

\[
\boxed{
\textbf{determine whether predictor reliability explains D2's weak-regime gap.}
}
\]

---

# 138. Stop Condition

If C0 or C1 does not provide convincing evidence:

\[
\boxed{
\textbf{freeze pair discovery and move to real-world identifiability screening.}
}
\]

Do not continue synthetic pair optimization.

---

# 139. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ separates whether multimodal interaction is identifiable, whether it is faithfully recovered, and whether that recovered structure is accessible to downstream tasks.}
}
\]

---

# 140. Current Required Experiment

\[
\boxed{
\textbf{C0 — Label-Free Predictor Reliability Audit}
}
\]

followed conditionally by:

\[
\boxed{
\textbf{C1 — Reliability-Weighted Joint-Advantage Distillation}
}
\]

and then:

\[
\boxed{
\textbf{PAIR-PHASE FREEZE}.
}
\]