# AGENT.md

# ConFu++ v4.2

## Identifiable Higher-Order Multimodal Interaction Discovery

---

# 0. Mission

This repository develops **ConFu++**, an extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The project has already established that the low-rank multiplicative interaction architecture is capable of representing explicit pairwise and third-order interaction structure under Oracle supervision.

Therefore:

\[
\boxed{
Q1:\ Architecture\ Capability = PASS
}
\]

The current research problem is:

\[
\boxed{
Q2:\ Discovery\ Objective\ Capability
}
\]

The central question is:

> Can an unsupervised cross-modal discovery objective identify interaction structure when that structure is both present and identifiable from the target modality?

The project must now distinguish:

\[
\boxed{
\text{architecture capability}
}
\]

from:

\[
\boxed{
\text{target identifiability}
}
\]

from:

\[
\boxed{
\text{objective quality}.
}
\]

---

# 1. Current Project Status

```text
Oracle architecture:
PASS

Synthetic order recovery:
PASS

Oracle health:
PASS

Oracle modality dependence:
PASS

IID discovery benchmark:
NON-IDENTIFIABLE

Generic MLP sanity:
WARNING

D0 Original ConFu:
BLOCKED on IID benchmark

D1 Residual Alignment:
BLOCKED on IID benchmark

D2 Joint-Advantage Distillation:
BLOCKED on IID benchmark

Identifiable discovery benchmark:
NEXT

Real-world transfer:
BLOCKED
```

---

# 2. Core Scientific Principle

A deterministic interaction representation:

\[
h_{12}=F(r_1,r_2)
\]

does not create new information beyond:

\[
(r_1,r_2).
\]

Its purpose is:

\[
\boxed{
\text{to expose nonlinear joint structure in an explicit representation}
}
\]

that is easier for a downstream model to use.

The main concept is therefore:

\[
\boxed{
\text{interaction accessibility}
}
\]

not new information creation.

---

# 3. Three Separate Scientific Questions

The research must keep the following questions separate.

## Q1 — Architecture capability

Can:

\[
h_{12},
h_{13},
h_{23},
h_{123}
\]

represent known interaction structure?

Current answer:

\[
\boxed{YES}
\]

via Oracle synthetic experiments.

---

# 4. Q2 — Discovery capability

Can an objective discover those interactions without being given:

\[
z_i\odot z_j
\]

or:

\[
z_1\odot z_2\odot z_3
\]

as targets?

Current answer:

\[
\boxed{UNKNOWN}
\]

because the original IID synthetic discovery protocol is non-identifiable.

---

# 5. Q3 — Real-world usefulness

Does the validated discovery mechanism improve:

```text
classification
retrieval
linear accessibility
interaction selectivity
```

on natural multimodal data?

Current answer:

\[
\boxed{UNKNOWN}
\]

and testing is blocked until Q2 is validated.

---

# 6. Historical Real-World Findings

The project already established several important controls.

## AV-MNIST

Observed strong single-modality shortcut.

Interpretation:

\[
\boxed{
\text{joint computation does not guarantee joint dependence}.
}
\]

---

# 7. CMU-MOSI

Observed multimodal dependence but weak task accessibility and text dominance.

Interpretation:

\[
\boxed{
\text{joint dependence does not guarantee task-useful interaction}.
}
\]

---

# 8. UR-FUNNY

Original ConFu:

\[
64.462\pm0.713\%.
\]

Multiple ConFu++ variants failed to produce stable positive linear accessibility.

This motivated controlled synthetic analysis.

UR-FUNNY remains a later transfer benchmark, not the current development benchmark.

---

# 9. Oracle Synthetic Benchmark

The original S0–S4 benchmark uses independent:

\[
z_1,z_2,z_3
\in
\{-1,+1\}^{32}.
\]

It contains:

```text
first-order signal
pairwise signal
third-order signal
mixed-order signal
```

and has successfully validated architecture capability.

---

# 10. Oracle Pair Result

Pair Oracle interactions produce approximately:

\[
\Delta_2\approx+29\text{ pp}
\]

on pairwise regimes.

Correct pair selectivity is recovered.

Therefore:

\[
\boxed{
\text{pair interaction architecture is valid}.
}
\]

---

# 11. Oracle Triple Result

On pure triple S3:

\[
Z_{\le1}\approx50\%
\]

\[
Z_{\le2}\approx50\%
\]

while:

\[
Z_{\le3}\approx93.5\%.
\]

Thus:

\[
\boxed{
\Delta_3\approx+43.4\text{ pp}.
}
\]

Therefore:

\[
\boxed{
\text{triple interaction architecture is valid}.
}
\]

---

# 12. Oracle Health

Pair and triple interaction representations exhibit:

```text
non-zero variance
healthy effective rank
healthy dimension-wise std
high target cosine
large shuffle sensitivity
```

Therefore architecture failure, collapse, and source-modality ignorance are not the current bottlenecks.

---

# 13. Discovery Identifiability Failure

The original IID generator defines:

\[
z_1,z_2,z_3
\]

independently.

But ConFu-style discovery uses mappings such as:

\[
h_{12}\rightarrow r_3.
\]

If:

\[
r_3=f(z_3)
\]

and:

\[
z_3\perp(z_1,z_2),
\]

then:

\[
I((z_1,z_2);z_3)=0.
\]

Operationally:

\[
R^2((z_1,z_2)\rightarrow z_3)\approx0.
\]

Therefore:

\[
q_{joint}\approx q_{add}\approx E[r_3].
\]

---

# 14. Consequence

On the IID benchmark:

\[
q_{joint}-q_{add}\approx0.
\]

Therefore D0, D1, and D2 have no valid discovery signal.

Any failure would primarily demonstrate:

\[
\boxed{
\text{target non-identifiability}
}
\]

rather than objective failure.

---

# 15. IID Benchmark Reclassification

Do NOT delete the original IID synthetic benchmark.

Reclassify it as:

\[
\boxed{
\text{Non-Identifiable Discovery Control}
}
\]

or:

```text
NIC — Non-Identifiable Control
```

Its purpose is to verify that a discovery method does not invent interaction where the target modality contains no predictable source-pair information.

---

# 16. Expected NIC Behavior

For:

\[
12\rightarrow3,
\]

require:

\[
R^2_{joint}\approx0
\]

and:

\[
R^2_{add}\approx0.
\]

Therefore:

\[
J=R^2_{joint}-R^2_{add}\approx0.
\]

D2 target magnitude should also be close to zero.

This is a negative identifiability control.

---

# 17. New Main Benchmark

Create a separate:

\[
\boxed{
\text{Identifiable Predictive-Interaction Benchmark}
}
\]

abbreviated:

```text
IPIB
```

The target modality must contain three explicitly controlled components:

\[
\boxed{
\text{additive predictable}
+
\text{joint predictable}
+
\text{private unpredictable}.
}
\]

---

# 18. Why IPIB Is Necessary

D0/D1/D2 can only be meaningfully compared if the source pair contains information that predicts the target modality.

The benchmark must make it possible to know exactly:

```text
what additive predictors can learn
what only joint predictors can learn
what no predictor can learn
```

This gives a clean identification test.

---

# 19. Initial Scope

Do NOT build a symmetric three-target benchmark first.

Start only with:

\[
\boxed{
12\rightarrow3.
}
\]

That means:

```text
source modality 1
source modality 2
target modality 3
```

Once validated, permute to:

\[
13\rightarrow2
\]

and:

\[
23\rightarrow1.
\]

---

# 20. Latent Variables

Sample:

\[
z_1,z_2,u_3
\overset{iid}{\sim}
\{-1,+1\}^{d}.
\]

Recommended:

\[
d=32.
\]

Where:

- \(z_1\): source-1 latent;
- \(z_2\): source-2 latent;
- \(u_3\): private target latent.

---

# 21. Source Modalities

Generate:

\[
x_1=P_1z_1+\epsilon_1
\]

\[
x_2=P_2z_2+\epsilon_2.
\]

Use fixed random projections.

Start noise-free:

\[
\epsilon_1=\epsilon_2=0.
\]

---

# 22. Target Modality

Generate:

\[
\boxed{
x_3
=
A_{31}z_1
+
A_{32}z_2
+
\beta B_3(z_1\odot z_2)
+
\lambda_pP_3u_3
+
\epsilon_3.
}
\]

This is the core new benchmark definition.

---

# 23. Target Decomposition

Define:

### Additive component

\[
a_3
=
A_{31}z_1
+
A_{32}z_2.
\]

### Joint component

\[
j_3
=
B_3(z_1\odot z_2).
\]

### Private component

\[
p_3
=
P_3u_3.
\]

Thus:

\[
\boxed{
x_3
=
a_3+\beta j_3+\lambda_p p_3+\epsilon_3.
}
\]

---

# 24. Interpretation

The benchmark is designed such that:

\[
q_{add}(x_1,x_2)
\]

should approximate:

\[
a_3.
\]

The joint predictor:

\[
q_{joint}(x_1,x_2)
\]

should approximate:

\[
a_3+\beta j_3.
\]

Neither can predict:

\[
p_3.
\]

Therefore:

\[
\boxed{
q_{joint}-q_{add}
\approx
\beta j_3.
}
\]

This is exactly the structure D2 is intended to recover.

---

# 25. Ground-Truth Pair Interaction

Retain latent ground truth:

\[
g_{12}
=
z_1\odot z_2.
\]

Projected target-space interaction:

\[
\boxed{
\tilde g_{12}
=
B_3g_{12}.
}
\]

This allows direct target-quality measurement.

---

# 26. Downstream Label

Define downstream task:

\[
y
=
\mathbf1[
w^\top g_{12}+\eta>0
].
\]

Important:

\[
\boxed{
D0,D1,D2\text{ must never use }y\text{ during representation training}.
}
\]

Labels exist only for downstream accessibility evaluation.

---

# 27. Why This Label Is Useful

The discovery objective uses only cross-modal prediction.

The downstream task measures whether:

\[
h_{12}
\]

exposes the discovered interaction in a task-relevant form.

Therefore:

\[
\boxed{
\text{discovery remains representation-level, not task-supervised}.
}
\]

---

# 28. IPIB Regimes

Define the following regimes.

---

# 29. I0 — No Joint Interaction

Set:

\[
\beta=0.
\]

Target:

\[
x_3=a_3+\lambda_p p_3.
\]

Expected:

\[
R^2_{joint}
\approx
R^2_{add}.
\]

Therefore:

\[
J\approx0.
\]

And:

\[
\Delta_{12}\approx0.
\]

This is the primary false-positive discovery control.

---

# 30. I1 — Weak Interaction

Set:

\[
\beta=0.25.
\]

Expected:

\[
J>0
\]

but small.

A good discovery method should show weak but measurable interaction recovery.

---

# 31. I2 — Medium Interaction

Set:

\[
\beta=0.5.
\]

This is the default development regime.

Expected:

\[
J>0
\]

and interaction recovery should be clear.

---

# 32. I3 — Strong Interaction

Set:

\[
\beta=1.0.
\]

Expected:

\[
J
\]

and:

\[
\Delta_{12}
\]

to increase relative to I1/I2.

---

# 33. I4 — Strong Private Noise

Set:

\[
\beta=1
\]

with larger:

\[
\lambda_p.
\]

Purpose:

> test whether D2 isolates predictable joint structure while D1 becomes contaminated by unpredictable target variation.

---

# 34. Recommended Initial Private Strength

Start with:

\[
\lambda_p=1.
\]

Later sweep:

\[
\lambda_p\in\{0,0.5,1,2\}.
\]

Do not sweep initially.

---

# 35. Component Scaling

Ensure:

\[
Var(a_3)
\]

\[
Var(j_3)
\]

\[
Var(p_3)
\]

are explicitly measured.

Do not allow one component to dominate by accidental projection scale.

Normalize where necessary.

---

# 36. Required Dataset Diagnostics

For every regime report:

```text
Var(additive)
Var(joint)
Var(private)

effective rank(additive)
effective rank(joint)
effective rank(private)

cross-correlation between components
```

Components should be approximately disentangled by construction.

---

# 37. Target Identifiability Check

Before running discovery methods, verify:

\[
R^2_{add}
\]

and:

\[
R^2_{joint}.
\]

Require expected ordering:

### I0

\[
R^2_{joint}
\approx
R^2_{add}.
\]

### I1–I3

\[
\boxed{
R^2_{joint}>R^2_{add}.
}
\]

---

# 38. Joint Prediction Advantage

Define:

\[
\boxed{
J_{12\rightarrow3}
=
R^2_{joint}
-
R^2_{add}.
}
\]

This is the benchmark identifiability metric.

Do NOT call this formal synergy.

---

# 39. Predictor Capacity Matching

Require:

\[
Params(q_{joint})
\approx
Params(q_1)+Params(q_2).
\]

Target difference:

\[
<5\%.
\]

Prefer:

\[
<1\%.
\]

Always log exact counts.

---

# 40. Additive Predictor

Use independent predictors:

\[
q_1(x_1)
\]

and:

\[
q_2(x_2).
\]

Define:

\[
q_{add}
=
q_1(x_1)+q_2(x_2).
\]

No cross-modal access is allowed.

---

# 41. Joint Predictor

Use:

\[
q_{joint}([x_1,x_2]).
\]

The architecture should have approximately matched parameter capacity to the additive system.

---

# 42. Freeze Predictor Targets

Training order:

```text
1. Train additive predictors.
2. Freeze additive predictors.
3. Train joint predictor.
4. Freeze joint predictor.
5. Construct discovery target.
6. Train interaction representation.
```

No co-adaptation initially.

---

# 43. D0 — Original ConFu Discovery

Train:

\[
h_{12}=F(x_1,x_2)
\]

toward:

\[
x_3.
\]

Conceptually:

\[
\boxed{
h_{12}\leftrightarrow x_3.
}
\]

This target contains:

\[
a_3+\beta j_3+\lambda_pp_3.
\]

---

# 44. D0 Research Question

Can direct cross-modal alignment recover the pair interaction when the target also contains additive and private content?

Expected risk:

\[
h_{12}
\]

may preferentially encode:

\[
a_3
\]

because additive shared structure may be easier.

---

# 45. D1 — Residual Alignment

Construct:

\[
\boxed{
e_3
=
x_3-q_{add}(x_1,x_2).
}
\]

Ideally:

\[
e_3
\approx
\beta j_3+\lambda_p p_3.
\]

Train:

\[
h_{12}\leftrightarrow e_3.
\]

---

# 46. D1 Limitation

D1 removes additive predictable structure but preserves:

\[
p_3,
\]

which is unpredictable from:

\[
(x_1,x_2).
\]

Therefore D1 target remains contaminated by target-private content.

---

# 47. D2 — Joint-Advantage Distillation

Construct:

\[
\boxed{
d_{12\rightarrow3}
=
q_{joint}(x_1,x_2)
-
q_{add}(x_1,x_2).
}
\]

With good predictors:

\[
d_{12\rightarrow3}
\approx
\beta j_3.
\]

Train:

\[
\boxed{
h_{12}
\leftrightarrow
stopgrad(d_{12\rightarrow3}).
}
\]

This is the primary ConFu++ v4.2 hypothesis.

---

# 48. Central D2 Hypothesis

D2 should remove both:

### additive predictable information

because it is shared by:

\[
q_{joint}
\]

and:

\[
q_{add};
\]

and:

### unpredictable target-private information

because neither predictor can predict it.

Therefore D2 isolates:

\[
\boxed{
\text{predictable non-additive target structure}.
}
\]

---

# 49. Important Terminology

Preferred:

```text
Joint Predictive Advantage
Non-Additive Predictive Residual
Joint-Advantage Distillation
Interaction Discovery Target
```

Avoid:

```text
exact synergy
PID synergy
pure information synergy
new information
```

---

# 50. Discovery Architecture

Keep the Oracle-validated interaction architecture fixed.

\[
u_1=LN(U_1x_1)
\]

\[
u_2=LN(U_2x_2)
\]

\[
h_{12}
=
W_{12}
\left[
\frac{
u_1\odot u_2
}{
\sqrt R
}
\right].
\]

Default:

\[
R=64.
\]

---

# 51. Architecture Freeze Rule

During D0/D1/D2 comparison do NOT change:

```text
rank
output dimension
normalization
interaction operator
encoder architecture
```

Only discovery target/objective changes.

---

# 52. Primary Target Quality Metric

Because:

\[
\tilde g_{12}
\]

is known, measure:

\[
\boxed{
R^2(
d_{12\rightarrow3}
\rightarrow
\tilde g_{12}
).
}
\]

This tests target construction directly.

---

# 53. D1 Target Quality

Also measure:

\[
R^2(
e_3
\rightarrow
\tilde g_{12}
).
\]

This allows direct D1 vs D2 comparison before interaction training.

---

# 54. D0 Target Quality

For completeness measure how much:

\[
x_3
\]

linearly or nonlinearly exposes:

\[
\tilde g_{12}.
\]

But do not interpret raw target recoverability as interaction selectivity.

---

# 55. Target Quality Table

Required:

| Regime | Target | Target→Joint GT R² |
|---|---|---:|
| I0 | D0 raw target | |
| I0 | D1 residual | |
| I0 | D2 joint advantage | |
| I1 | D0 | |
| I1 | D1 | |
| I1 | D2 | |
| I2 | D0 | |
| I2 | D1 | |
| I2 | D2 | |
| I3 | D0 | |
| I3 | D1 | |
| I3 | D2 | |

---

# 56. Interaction Recovery Metric

After training:

\[
h_{12},
\]

measure:

\[
\boxed{
R^2(h_{12}\rightarrow g_{12}).
}
\]

If dimensions differ, use a simple linear recovery probe.

Do not use a powerful MLP for the primary recovery metric.

---

# 57. Interaction Recovery Cosine

If representations are dimension-aligned, also report:

\[
cos(h_{12},P(g_{12})).
\]

Use this as a secondary metric.

---

# 58. Accessibility Metric

Define:

\[
Z_1=[x_1,x_2].
\]

Define:

\[
Z_2=[x_1,x_2,h_{12}].
\]

Then:

\[
\boxed{
\Delta_{12}^{linear}
=
Perf_{linear}(Z_2)
-
Perf_{linear}(Z_1).
}
\]

This remains the primary task accessibility metric.

---

# 59. Nonlinear Accessibility

Also compute:

\[
\Delta_{12}^{MLP}.
\]

Secondary only.

Primary goal:

\[
\Delta_{12}^{linear}>0.
\]

---

# 60. Linearization Gap

Define:

\[
LG_1
=
Perf_{MLP}(Z_1)
-
Perf_{linear}(Z_1)
\]

and:

\[
LG_2
=
Perf_{MLP}(Z_2)
-
Perf_{linear}(Z_2).
\]

Desired:

\[
LG_2<LG_1.
\]

---

# 61. False-Positive Control I0

On:

\[
\beta=0,
\]

require:

\[
J\approx0.
\]

D2 target should satisfy:

\[
Var(d)\approx0.
\]

And:

\[
\Delta_{12}\approx0.
\]

This must happen through discovery, not hard-coded branch suppression.

---

# 62. Interaction Strength Response

Across:

\[
\beta=0,0.25,0.5,1,
\]

measure:

\[
J(\beta)
\]

\[
TargetRecovery(\beta)
\]

\[
InteractionRecovery(\beta)
\]

\[
\Delta_{12}(\beta).
\]

Desired monotonic or generally increasing trend.

---

# 63. Main Mechanism Figure

X-axis:

\[
\beta
\]

Y-axis:

\[
\Delta_{12}^{linear}
\]

or:

\[
R^2(h_{12}\rightarrow g_{12}).
\]

Curves:

```text
Oracle
D0 Original ConFu
D1 Residual
D2 Joint Advantage
```

This is a key planned paper figure.

---

# 64. Main Mechanism Hypothesis

Expected ordering:

\[
\boxed{
D2>D1>D0
}
\]

in joint-interaction recovery.

This is a hypothesis, not an assumption.

Report actual outcome faithfully.

---

# 65. Private Noise Stress Test

I4 specifically tests:

\[
\lambda_p\uparrow.
\]

Prediction:

D0 may degrade because target contains more irrelevant private content.

D1 may also degrade because:

\[
e_3
\]

still contains:

\[
p_3.
\]

D2 should be more robust because predictors cannot model:

\[
p_3.
\]

---

# 66. Key I4 Scientific Test

Measure recovery as:

\[
\lambda_p
\]

increases.

Strong result:

\[
Recovery_{D2}
\]

remains more stable than:

\[
Recovery_{D1}.
\]

This directly supports the D2 formulation.

---

# 67. Distillation Failure Localization

If:

\[
R^2(d\rightarrow\tilde g_{12})
\]

is high but:

\[
R^2(h_{12}\rightarrow g_{12})
\]

is low:

\[
\boxed{
\text{distillation failure}.
}
\]

Do not redesign target construction.

---

# 68. Target Construction Failure

If:

\[
R^2(d\rightarrow\tilde g_{12})
\]

is low despite:

\[
J>0,
\]

then:

\[
\boxed{
\text{joint-advantage target construction is insufficient}.
}
\]

Investigate predictor representation or subtraction geometry.

---

# 69. Identifiability Failure

If:

\[
J\approx0
\]

despite:

\[
\beta>0,
\]

then the benchmark/predictor pair does not expose the intended joint component.

Do not blame D2.

Fix identifiability first.

---

# 70. D0 Evaluation

D0 answers:

> Does Original ConFu-style alignment naturally recover the known predictable interaction?

Measure:

```text
target recovery
interaction recovery
Delta12
linearization
```

---

# 71. D1 Evaluation

D1 answers:

> Does subtracting additive prediction isolate interaction better than direct ConFu alignment?

Do not add extra objectives.

---

# 72. D2 Evaluation

D2 answers:

> Does distilling the difference between matched joint and additive predictors isolate predictable interaction better than either D0 or D1?

This is the primary research question.

---

# 73. Initial Loss for D2

Start with normalized cosine regression:

\[
L_{D2}
=
1-
cos(
P_h(h_{12}),
P_d(d_{12\rightarrow3})
).
\]

Optionally add small MSE only after baseline characterization.

---

# 74. No InfoNCE Initially

Do not introduce InfoNCE in the first D2 experiment.

Reason:

instance discrimination may introduce an additional confound.

First validate target quality using direct regression.

---

# 75. D0/D1 Loss Matching

Where practical, use similar projection dimensionality and optimization budgets.

The discovery target should be the main difference between D0/D1/D2.

---

# 76. Exploratory Protocol

First run:

```text
seed 1
```

on:

```text
I0
I2
I3
I4
```

Start with:

\[
12\rightarrow3.
\]

I1 can follow after the mechanism behaves correctly.

---

# 77. Exploratory Gate

Proceed if D2 shows:

### I0

\[
\Delta_{12}\approx0
\]

and target variance near zero.

### I2/I3

\[
J>0
\]

\[
R^2(d\rightarrow g)>0
\]

\[
R^2(h\rightarrow g)>0
\]

\[
\Delta_{12}^{linear}>0.
\]

---

# 78. Five-Seed Trigger

If exploratory D2 passes:

```text
freeze benchmark
freeze predictor capacity
freeze interaction architecture
freeze D2 loss
freeze optimizer
```

Then run:

\[
5\text{ seeds}.
\]

---

# 79. Confirmatory Metrics

Report:

```text
mean
sample std
per-seed values
paired differences
95% CI
effect size
```

for:

\[
J
\]

\[
TargetRecovery
\]

\[
InteractionRecovery
\]

\[
\Delta_{12}.
\]

---

# 80. Predictor Generalization

Always log:

```text
train R²
validation R²
test R²
```

for both additive and joint predictors.

Large train-test gaps indicate predictor overfitting and invalidate the interpretation of:

\[
J.
\]

---

# 81. No Test-Driven Tuning

All architecture and predictor choices must be selected by validation.

Test is used only for final reporting.

---

# 82. Expand to Other Pair Directions

Only after:

\[
12\rightarrow3
\]

passes.

Then generate separate benchmarks:

```text
13 → 2
23 → 1
```

Do not create circular multimodal dependence initially.

---

# 83. Pair Permutation Goal

D2 should recover comparable behavior after changing the identity of source/target modalities.

This tests whether performance is architectural rather than index-specific.

---

# 84. Symmetric Multi-Pair Benchmark

Only after all three directional benchmarks pass individually.

Then construct a more complex multimodal system with simultaneous predictable pair interactions.

This is a later stage.

---

# 85. Third-Order Discovery Remains Blocked

Even though Oracle:

\[
h_{123}
\]

passes, do not implement third-order discovery yet.

Pair discovery must work first.

---

# 86. Third-Order Identifiability Requirement

Future order-3 benchmark must provide:

\[
\boxed{
\text{predictable triple structure}
}
\]

that is not explainable by:

```text
order-1 predictors
order-2 predictors
```

before a triple discovery objective can be tested.

---

# 87. Future Third-Order Target

Conceptually:

\[
d_{123}
=
q_{joint123}
-
q_{\le2}.
\]

Where:

\[
q_{\le2}
\]

contains all validated order-1 and order-2 components.

Do not implement yet.

---

# 88. Task-Aware Discovery

Task-aware discovery remains a later extension.

Possible target:

\[
d^{task}
=
q_{joint}^{task}
-
q_{add}^{task}.
\]

This may directly isolate label-relevant interaction.

---

# 89. Task-Aware Discovery Is Not Core Yet

Do not use labels in the core D2 method.

Reason:

the current research goal is to preserve ConFu's representation-learning character.

Task-aware discovery becomes an ablation or downstream extension.

---

# 90. Generic MLP Control

Current shallow MLP fails to optimize the dense third-order S3 rule.

Do not claim universal superiority of explicit multiplicative interactions.

Valid claim:

> Oracle higher-order features make the synthetic interaction linearly accessible.

---

# 91. Future Stronger MLP Controls

Later consider:

```text
deeper MLP
gated MLP
polynomial network
multiplicative MLP
```

with matched capacity.

Not current priority.

---

# 92. MLP Sanity Priority

Generic MLP sanity is now secondary to discovery identifiability.

Do not spend large experiment budget optimizing generic MLP before D2 is tested on IPIB.

---

# 93. NIC Negative Control

The old IID benchmark should remain in the final experimental suite.

Expected:

\[
J\approx0.
\]

D2 should not discover meaningful target interaction.

This proves D2 respects identifiability constraints.

---

# 94. Two Synthetic Benchmark Families

The final synthetic suite should contain:

### Family A

```text
Oracle Order Benchmark
```

Purpose:

\[
\text{architecture capability}
\]

### Family B

```text
Identifiable Predictive-Interaction Benchmark
```

Purpose:

\[
\text{discovery capability}
\]

### Family C

```text
Non-Identifiable Control
```

Purpose:

\[
\text{identifiability negative control}
\]

---

# 95. Scientific Separation

The project must never conflate:

\[
\text{can represent}
\]

with:

\[
\text{can discover}
\]

with:

\[
\text{can transfer}.
\]

These correspond to:

\[
Q1,
Q2,
Q3.
\]

---

# 96. Main Result Table for Discovery

Required:

| Regime | Method | J | Target→GT R² | h→GT R² | Linear Δ12 |
|---|---|---:|---:|---:|---:|
| I0 | D0 | | | | |
| I0 | D1 | | | | |
| I0 | D2 | | | | |
| I1 | D0 | | | | |
| I1 | D1 | | | | |
| I1 | D2 | | | | |
| I2 | D0 | | | | |
| I2 | D1 | | | | |
| I2 | D2 | | | | |
| I3 | D0 | | | | |
| I3 | D1 | | | | |
| I3 | D2 | | | | |
| I4 | D0 | | | | |
| I4 | D1 | | | | |
| I4 | D2 | | | | |

---

# 97. Oracle Reference

For I1–I4, also train an Oracle:

\[
h_{12}\rightarrow g_{12}.
\]

This defines an accessibility upper-bound-style reference.

---

# 98. Oracle Recovery Ratio

Define:

\[
Recovery_2
=
\frac{
\Delta_{12}^{method}
}{
\Delta_{12}^{oracle}+\epsilon
}.
\]

Only report when Oracle gain is clearly positive.

---

# 99. Target Recovery Ratio

Also define:

\[
TargetRecovery
=
R^2(
d_{method}
\rightarrow
\tilde g_{12}
).
\]

This is especially important for D1/D2.

---

# 100. Discovery Success Criteria

D2 passes if:

### I0

\[
J\approx0
\]

and:

\[
\Delta_{12}\approx0.
\]

### I1–I3

\[
J>0.
\]

\[
TargetRecovery>0.
\]

\[
InteractionRecovery>0.
\]

\[
\Delta_{12}^{linear}>0.
\]

### Relative performance

Prefer:

\[
D2>D0
\]

and:

\[
D2>D1.
\]

---

# 101. Robustness Success

In I4:

\[
D2
\]

should degrade less than D1 as:

\[
\lambda_p
\]

increases.

This is a central robustness hypothesis.

---

# 102. Discovery Failure Case A

If:

\[
J\approx0
\]

on I2/I3:

```text
BENCHMARK IDENTIFIABILITY FAILURE
```

Fix dataset or predictor design.

Do not modify interaction architecture.

---

# 103. Discovery Failure Case B

If:

\[
J>0
\]

but:

\[
TargetRecovery\approx0:
\]

```text
TARGET CONSTRUCTION FAILURE
```

D2 subtraction is not isolating ground-truth interaction.

---

# 104. Discovery Failure Case C

If:

\[
TargetRecovery>0
\]

but:

\[
InteractionRecovery\approx0:
\]

```text
DISTILLATION FAILURE
```

Focus on alignment loss/projection.

---

# 105. Discovery Failure Case D

If:

\[
InteractionRecovery>0
\]

but:

\[
\Delta_{12}\le0:
\]

```text
ACCESSIBILITY / TASK RELEVANCE FAILURE
```

The learned interaction exists but is not useful for the downstream task.

---

# 106. Failure Localization Is Mandatory

Every failed experiment must identify whether the failure lies in:

```text
identifiability
target construction
distillation
representation
accessibility
task relevance
```

Do not respond to all failures with larger architecture.

---

# 107. Current Forbidden Work

Until D2 passes IPIB:

```text
do not return to UR-FUNNY tuning
do not implement discovered h123
do not add cross-attention
do not sweep rank
do not add dynamic routing
do not add task loss
do not fine-tune large encoders
```

---

# 108. Real-World Return Gate

Return to natural data only after:

```text
IPIB identifiability passes
D2 target recovery passes
D2 interaction recovery passes
D2 linear accessibility passes
D2 false-positive control passes
5-seed confirmation passes
```

---

# 109. Preferred Real-World Dataset

After Q2 passes:

\[
\boxed{
CMU\text{-}MOSEI\ Emotion
}
\]

is the preferred first candidate.

But screen it first.

---

# 110. Real-World Screening

Use:

\[
P0=\text{linear additive}
\]

\[
P1=\text{nonlinear additive}
\]

\[
P2=\text{capacity-matched joint}.
\]

Require:

\[
P2>P1
\]

on validation for relevant target/task settings before investing in ConFu++.

---

# 111. UR-FUNNY Return

UR-FUNNY remains valuable as:

```text
historical negative result
real-world transfer test
Original ConFu comparison
```

Return only after synthetic discovery is validated.

---

# 112. Real-World D2

For natural modalities:

\[
d_{ij\rightarrow k}
=
q_{joint}(r_i,r_j)
-
q_{add}(r_i,r_j).
\]

Train:

\[
h_{ij}
\]

to distill this target.

Then evaluate:

```text
classification
retrieval
linear accessibility
shuffle dependence
single-modality predictability
```

---

# 113. Preserve ConFu Alignment Evaluation

Where possible, retain:

```text
1→1 retrieval
2→1 retrieval
```

so improvements in classification do not destroy ConFu's original alignment behavior.

---

# 114. Main ConFu++ Claim if Successful

Potential claim:

> ConFu++ identifies predictable non-additive cross-modal structure by distilling the advantage of a capacity-matched joint predictor over independent additive predictors into explicit interaction embeddings.

---

# 115. Stronger Claim if Real-World Transfer Works

> Explicit interaction discovery improves downstream linear accessibility while preserving multimodal alignment.

Do not make this claim before real-world validation.

---

# 116. Paper Story

Potential final story:

```text
1. Higher-order alignment does not guarantee explicit interaction recovery.

2. Oracle experiments prove interaction architecture capability.

3. IID discovery benchmark reveals an identifiability requirement.

4. We introduce an identifiable predictive-interaction benchmark.

5. Original ConFu and residual alignment are compared under known ground truth.

6. Joint-Advantage Distillation isolates predictable non-additive structure.

7. Interaction recovery is measured directly.

8. The mechanism transfers to natural multimodal tasks.
```

---

# 117. Scientific Contribution of Identifiability

The identifiability finding is itself important:

\[
\boxed{
\text{a discovery objective cannot recover cross-modal interaction if its target contains no predictable information from the source subset}.
}
\]

This should be stated explicitly in future writing.

---

# 118. Important Distinction

A dataset may contain task-relevant pair interaction:

\[
Y=f(z_1,z_2)
\]

while target modality:

\[
z_3
\]

is independent.

In that case:

\[
\boxed{
\text{task interaction exists}
}
\]

but:

\[
\boxed{
\text{cross-modal discovery signal does not}.
}
\]

This is a major conceptual distinction.

---

# 119. Implication for ConFu

Original ConFu alignment:

\[
F(r_i,r_j)\leftrightarrow r_k
\]

implicitly assumes that:

\[
r_k
\]

contains structure predictable from:

\[
(r_i,r_j).
\]

This assumption must be audited on real datasets.

---

# 120. Real-World Identifiability Audit

Before applying D0/D1/D2 to a natural dataset, measure:

\[
R^2_{add}
\]

and:

\[
R^2_{joint}.
\]

If both are near zero:

\[
\boxed{
\text{cross-modal discovery is not identifiable under that target mapping}.
}
\]

Do not interpret failure as objective weakness.

---

# 121. New Dataset Screening Rule

For each:

\[
ij\rightarrow k,
\]

classify mapping as:

### Non-identifiable

\[
R^2_{joint}\approx0.
\]

### Additively identifiable

\[
R^2_{add}>0
\]

and:

\[
J\approx0.
\]

### Jointly identifiable

\[
\boxed{
J>0.
}
\]

Only the third class is ideal for D2.

---

# 122. Mapping Selection

On natural datasets, do not assume:

```text
12 → 3
13 → 2
23 → 1
```

are equally identifiable.

Audit all three directions.

A pair-target direction may be informative while another is not.

---

# 123. Mapping-Specific Training

Future ConFu++ may activate interaction discovery only where:

\[
J_{ij\rightarrow k}
\]

is sufficiently supported by validation.

Do not implement routing yet.

This is a future implication.

---

# 124. Immediate Repository Tasks

Implement:

```text
src/datasets/identifiable_interaction.py
```

and:

```text
src/experiments/synthetic_interaction/
```

with:

```text
dataset.py
predictors.py
d0_confu.py
d1_residual.py
d2_joint_advantage.py
oracle.py
probes.py
diagnostics.py
metrics.py
runner.py
```

---

# 125. Dataset Unit Tests

Required:

```text
correct beta=0 behavior
correct additive component
correct joint component
correct private component
source/private independence
component scaling
seed reproducibility
no split leakage
```

---

# 126. Identifiability Unit Tests

For ideal/noiseless settings, verify:

```text
I0:
joint advantage ≈ 0

I2/I3:
joint advantage > 0
```

within empirical tolerance.

---

# 127. Mapping Unit Tests

Use explicit names:

```text
12_to_3
13_to_2
23_to_1
```

Never infer direction from list ordering.

---

# 128. Predictor Unit Tests

Verify:

```text
q1 sees source 1 only
q2 sees source 2 only
q_joint sees both
all target same x3
predictors frozen before D2
capacity counts recorded
```

---

# 129. Immediate Experiment Sequence

Run exactly:

```text
1. Implement IPIB 12→3.

2. Verify component decomposition.

3. Verify I0 identifiability control.

4. Verify I2/I3 joint advantage.

5. Train Oracle h12.

6. Run D0.

7. Run D1.

8. Build D2 target.

9. Measure D2 target→ground-truth recovery.

10. Train D2 h12.

11. Measure h12→ground-truth recovery.

12. Measure linear Delta12.

13. Compare D0/D1/D2.

14. Run I4 private-noise test.

15. If D2 passes:
    freeze configuration.

16. Run five seeds.

17. Expand to 13→2.

18. Expand to 23→1.
```

---

# 130. Do Not Skip Target Analysis

Before every D2 interaction training run, inspect:

\[
d=q_{joint}-q_{add}.
\]

If it does not recover:

\[
\tilde g_{12},
\]

do not waste compute training:

\[
h_{12}.
\]

---

# 131. Primary Immediate Milestone

The next milestone is:

\[
\boxed{
R^2(
q_{joint}-q_{add}
\rightarrow
\tilde g_{12}
)>0
}
\]

on an identifiable benchmark.

Only after this target-level result should D2 distillation be considered valid.

---

# 132. Second Immediate Milestone

Then require:

\[
\boxed{
R^2(h_{12}\rightarrow g_{12})>0.
}
\]

This demonstrates interaction recovery.

---

# 133. Third Immediate Milestone

Then require:

\[
\boxed{
\Delta_{12}^{linear}>0.
}
\]

This demonstrates downstream accessibility.

---

# 134. Full Discovery Chain

A successful run must establish:

\[
\boxed{
\text{identifiable target}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{joint advantage target}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{interaction recovery}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{linear accessibility}.
}
\]

---

# 135. Current Scientific Bottleneck

The current bottleneck is no longer:

```text
rank
architecture
collapse
modality dependence
```

It is:

\[
\boxed{
\text{Discovery Target Construction Under Identifiability}
}
\]

---

# 136. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ learns explicit multimodal interactions by distilling predictable structure that is accessible to a joint predictor but not to capacity-matched additive predictors.}
}
\]

---

# 137. Final Development Rule

Before declaring any discovery objective a failure, first verify:

\[
\boxed{
\text{the target is identifiable from the source modalities}.
}
\]

Before changing architecture, first verify:

\[
\boxed{
\text{the target itself contains the intended interaction}.
}
\]

Before returning to real data, first verify:

\[
\boxed{
\text{the interaction can be discovered and made linearly accessible on controlled identifiable data}.
}
\]

The immediate next experiment is:

\[
\boxed{
\textbf{IPIB 12→3 + D0/D1/D2 comparison.}
}
\]