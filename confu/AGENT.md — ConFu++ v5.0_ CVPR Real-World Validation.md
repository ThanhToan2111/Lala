# AGENT.md

# ConFu++ v5.0

## From Controlled Interaction Discovery to Real-World Multimodal Validation

---

# 0. Mission

This repository develops **ConFu++**, an extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The project is now explicitly targeting a **CVPR main-track paper**.

The synthetic pair-discovery phase is complete and frozen.

The project has established:

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
\text{Interaction Fidelity}
}
\]

and has exposed a non-trivial:

\[
\boxed{
\text{Fidelity–Accessibility Tradeoff}.
}
\]

The current objective is no longer to invent or tune another synthetic interaction module.

The objective is now:

\[
\boxed{
\textbf{validate whether these principles transfer to real multimodal data.}
}
\]

---

# 1. CVPR Research Goal

The target paper should answer:

> When can higher-order multimodal interaction be identified from natural multimodal representations, and can explicit interaction discovery improve upon standard higher-order alignment?

The intended high-level thesis is:

\[
\boxed{
\text{Higher-order alignment}
\neq
\text{interaction identifiability}
\neq
\text{interaction fidelity}
\neq
\text{interaction accessibility}.
}
\]

ConFu++ explicitly separates these properties.

---

# 2. Final Synthetic-Phase Status

Current state:

```text
Architecture capability:
PASS

Oracle pair recovery:
PASS

Oracle third-order recovery:
PASS

Non-identifiable control:
PASS

IPIB identifiability:
PASS

D0 baseline:
COMPLETE

D1 residual:
COMPLETE

D2 Joint-Advantage Distillation:
COMPLETE

D2 > D0:
PASS

D2 false-positive suppression:
PASS

D2 > D1 accessibility:
PARTIAL / NOT UNIVERSAL

v4.3 B2 loss calibration:
INSUFFICIENT

v4.4 C0 reliability audit:
PASS

v4.4 C1 reliability weighting:
FAIL

Synthetic pair discovery:
FROZEN

Real-world validation:
NEXT
```

---

# 3. Frozen Final Pair Method

The final synthetic pair method is:

\[
\boxed{
\textbf{Joint-Advantage Distillation}
}
\]

abbreviated:

\[
\boxed{\text{JAD}}
\]

or:

```text
D2
```

Canonical formulation must not be changed during the first real-world phase.

---

# 4. Canonical D2 Predictors

For source modalities:

\[
(r_i,r_j)
\]

and target modality:

\[
r_k,
\]

train independent additive predictors:

\[
q_i(r_i)
\]

\[
q_j(r_j).
\]

Define:

\[
\boxed{
q_A(r_i,r_j)
=
q_i(r_i)+q_j(r_j).
}
\]

---

# 5. Capacity-Matched Joint Predictor

Train:

\[
q_J(r_i,r_j).
\]

Require:

\[
Params(q_J)
\approx
Params(q_i)+Params(q_j).
\]

Preferred difference:

\[
<1\%.
\]

Maximum acceptable:

\[
<5\%.
\]

Always report exact parameter counts.

---

# 6. Joint Predictive Advantage

Define:

\[
\boxed{
J_{ij\rightarrow k}
=
R^2(q_J,r_k)
-
R^2(q_A,r_k).
}
\]

This is the first real-world screening metric.

Preferred terminology:

```text
Joint Predictive Advantage
Joint Prediction Advantage
Non-Additive Predictive Advantage
```

Do NOT call it formal synergy.

---

# 7. Identifiability Gate

Canonical D2 uses:

\[
\boxed{
J_{ij\rightarrow k}>0.01
}
\]

as the interaction-activation gate unless real-world validation provides a strong reason to revise the threshold.

Do NOT tune the threshold on task labels.

If:

\[
J\le0.01,
\]

then:

\[
\boxed{
d_{ij\rightarrow k}=0.
}
\]

---

# 8. Joint-Advantage Target

For identifiable directions:

\[
\boxed{
d_{ij\rightarrow k}
=
q_J(r_i,r_j)
-
q_A(r_i,r_j).
}
\]

This is the frozen D2 target.

---

# 9. Interaction Representation

Use the Oracle-validated low-rank multiplicative interaction architecture:

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
W_{ij}
\left(
\frac{u_i\odot u_j}{\sqrt R}
\right).
}
\]

Canonical rank:

\[
\boxed{R=64}.
\]

---

# 10. D2 Distillation Loss

Canonical D2 uses cosine regression:

\[
\boxed{
L_{JAD}
=
1-
\cos(
P_h(h_{ij}),
P_d(d_{ij\rightarrow k})
).
}
\]

Do not replace it with B2 standardized MSE in the main formulation.

---

# 11. Why B2 Is Not Final

B2 improved I1 by only approximately:

\[
+0.252\text{ pp}
\]

and failed to close the D1–D2 accessibility gap.

It also altered representation rank geometry.

Therefore B2 is retained as:

```text
geometry diagnostic
```

not promoted into the final method.

---

# 12. Why Reliability Weighting Is Not Final

C0 showed:

\[
R_k
\]

correlates with interaction fidelity.

However C1 produced approximately:

\[
-0.02\text{ pp}
\]

on I1 and:

\[
+0.02\text{ pp}
\]

on I2.

Therefore:

\[
\boxed{
\text{reliability predicts fidelity but does not automatically improve accessibility}.
}
\]

C1 is closed.

---

# 13. Hard Freeze Rule

Do NOT reopen synthetic pair tuning using:

```text
different alpha values
additional reliability functions
rank sweeps
loss sweeps
InfoNCE
VICReg
Barlow Twins
cross-attention
dynamic routing
task loss
larger interaction MLPs
```

without a new theoretically justified hypothesis.

---

# 14. Central Synthetic Finding

The current evidence establishes:

\[
\boxed{
\text{higher interaction fidelity}
\not\Rightarrow
\text{higher downstream accessibility}.
}
\]

This is a central scientific result.

---

# 15. Three-Axis Framework

Every higher-order representation must now be analyzed along:

## Identifiability

Can the target modality be predicted from the source subset?

## Fidelity

Does the representation isolate/recover the intended joint structure?

## Accessibility

Can a restricted downstream predictor exploit that representation?

These three concepts form the core conceptual framework of the paper.

---

# 16. Real-World Phase Objective

The next phase asks:

\[
\boxed{
\textbf{Do natural multimodal datasets contain jointly identifiable cross-modal mappings?}
}
\]

Before training JAD, screen the dataset.

Always:

\[
\boxed{
\text{screen first, train second}.
}
\]

---

# 17. First Real-World Dataset

Primary next dataset:

\[
\boxed{
\text{CMU-MOSEI Emotion}
}
\]

Do NOT start with sentiment.

Rationale:

emotion tasks may provide stronger acoustic and visual structure than text-dominated sentiment.

This is only a hypothesis.

It must be tested.

---

# 18. Initial MOSEI Modalities

Use:

```text
V = vision
A = audio
T = text
```

Representations:

\[
r_V,r_A,r_T.
\]

---

# 19. Mandatory MOSEI Directions

Audit all three:

\[
\boxed{
V+A\rightarrow T
}
\]

\[
\boxed{
V+T\rightarrow A
}
\]

\[
\boxed{
A+T\rightarrow V.
}
\]

Do not assume symmetry.

---

# 20. Direction Naming

Use explicit names everywhere:

```text
VA_to_T
VT_to_A
AT_to_V
```

Never infer target direction from list index.

---

# 21. MOSEI Emotion Tasks

Where supported by the dataset pipeline, screen:

```text
happiness
sadness
anger
surprise
disgust
fear
```

Treat each emotion/task separately.

---

# 22. Real-World Predictor Audit

For each:

```text
emotion × mapping × seed
```

train:

\[
q_A
\]

and:

\[
q_J.
\]

Report:

\[
R^2_A
\]

\[
R^2_J
\]

and:

\[
J.
\]

---

# 23. Train/Validation/Test Reporting

For every predictor report:

```text
train R²
validation R²
test R²
```

Do not use only validation or test.

The goal is to detect overfitting.

---

# 24. Screening Seeds

Initial screening:

\[
\boxed{3\text{ seeds}}
\]

per mapping/task.

Use five seeds only for mappings selected for final experiments.

---

# 25. Screening Pass Gate

A real-world mapping is considered jointly identifiable only if:

\[
J_{val}>0
\]

and preferably:

\[
J_{test}>0
\]

with consistent sign across seeds.

Preferred stronger gate:

\[
\boxed{
J>0.01
}
\]

consistent with synthetic D2.

---

# 26. Screening Failure

If:

\[
R^2_J\approx0,
\]

the mapping is:

\[
\boxed{
\text{non-identifiable}.
}
\]

Do not train JAD.

---

# 27. Additive-Only Mapping

If:

\[
R^2_A>0
\]

but:

\[
J\approx0,
\]

the mapping is:

\[
\boxed{
\text{additively identifiable but not jointly identifiable}.
}
\]

This is also not an ideal D2 development mapping.

---

# 28. Jointly Identifiable Mapping

If:

\[
\boxed{
R^2_J>R^2_A
}
\]

consistently,

then the mapping contains measurable non-additive predictive headroom under the selected hypothesis classes.

This is the preferred JAD setting.

---

# 29. Real-World Screening Table

Produce:

| Task | Direction | Additive R² | Joint R² | J | Seeds positive |
|---|---|---:|---:|---:|---:|
| happiness | VA→T | | | | |
| happiness | VT→A | | | | |
| happiness | AT→V | | | | |
| sadness | VA→T | | | | |
| ... | ... | | | | | |

---

# 30. Benchmark Selection Rule

Do NOT select a task because:

```text
classification accuracy is easy
D2 looks strong on test
one seed is positive
```

Select using:

\[
\boxed{
\text{validation joint-identifiability evidence}.
}
\]

---

# 31. No Test Leakage

The test set must not determine:

```text
which emotion is selected
which mapping is selected
J threshold
architecture
rank
loss
```

Use validation only.

---

# 32. Phase R1

The immediate experiment is:

\[
\boxed{
\textbf{R1 — MOSEI Emotion Identifiability Screening}
}
\]

This is now the highest-priority task.

---

# 33. R1 Output

Required artifacts:

```text
results/mosei/identifiability/
```

and:

```text
MOSEI_IDENTIFIABILITY_REPORT.md
```

---

# 34. R1 Report Must Answer

1. Which modality mappings are predictable?
2. Which mappings have additive predictability?
3. Which mappings have additional joint predictability?
4. Which emotions show stable \(J>0\)?
5. Are those results stable across seeds?
6. Is predictor overfitting controlled?

---

# 35. R1 No Interaction Training

During R1:

\[
\boxed{
\text{DO NOT train D0/D1/D2 interaction representations}.
}
\]

R1 is screening only.

---

# 36. R2 Trigger

Proceed to:

\[
\boxed{
\text{R2 — Natural Interaction Discovery}
}
\]

only if at least one:

\[
emotion\times mapping
\]

passes R1.

---

# 37. R2 Methods

Compare exactly:

\[
\boxed{
D0
}
\]

\[
\boxed{
D1
}
\]

\[
\boxed{
D2
}
\]

with identical interaction architecture.

---

# 38. R2 D0

Original ConFu-style target:

\[
t_{D0}=r_k.
\]

Train:

\[
h_{ij}\leftrightarrow r_k.
\]

---

# 39. R2 D1

Residual:

\[
\boxed{
e_{ij\rightarrow k}
=
r_k-q_A(r_i,r_j).
}
\]

Train:

\[
h_{ij}\leftrightarrow e.
\]

---

# 40. R2 D2

Joint advantage:

\[
\boxed{
d_{ij\rightarrow k}
=
q_J-q_A.
}
\]

Train:

\[
h_{ij}\leftrightarrow d.
\]

Use canonical cosine loss.

---

# 41. R2 Capacity Control

All D0/D1/D2 variants must share:

```text
encoder
interaction rank
interaction output size
probe
training epochs
optimizer class
early stopping rule
```

Only discovery target changes.

---

# 42. Real-World Accessibility Representation

Define lower-order:

\[
Z_{\le1}
=
[r_V,r_A,r_T].
\]

For selected pair:

\[
Z_{\le2}
=
[Z_{\le1},h_{ij}].
\]

---

# 43. Linear Accessibility

Primary real-world metric:

\[
\boxed{
\Delta_{ij}^{linear}
=
Perf_{linear}(Z_{\le2})
-
Perf_{linear}(Z_{\le1}).
}
\]

---

# 44. Nonlinear Probe

Secondary:

\[
\Delta_{ij}^{MLP}.
\]

If only nonlinear probe gains:

\[
\boxed{
\text{interaction exists but is not explicitly linearized}.
}
\]

---

# 45. Task Metrics

Depending on MOSEI emotion protocol, report appropriate:

```text
accuracy
F1
macro F1
AUC
```

Do not introduce a new metric solely because it favors D2.

---

# 46. Original ConFu Comparison

Retain original ConFu baseline under the same:

```text
split
encoder
probe
seed
training budget
```

Do not rely only on paper-reported numbers.

---

# 47. Retrieval Preservation

Where the ConFu pipeline supports retrieval, preserve:

```text
1→1 retrieval
2→1 retrieval
```

metrics.

JAD should not improve task performance by destroying alignment behavior.

---

# 48. Representation Health

For:

\[
h_{ij},
\]

report:

```text
variance
effective rank
dimension-wise std
mean L2 norm
```

These are health diagnostics only.

---

# 49. Modality Dependence

For:

\[
h_{ij},
\]

shuffle each source modality independently.

Measure:

\[
\Delta Perf_{shuffle-i}
\]

and:

\[
\Delta Perf_{shuffle-j}.
\]

Both source modalities should matter in a genuine pair interaction.

---

# 50. Single-Modality Shortcut

Measure:

\[
R^2(r_i\rightarrow h_{ij})
\]

and:

\[
R^2(r_j\rightarrow h_{ij}).
\]

Strong asymmetry indicates possible shortcutting.

Do NOT use arbitrary hard thresholds.

---

# 51. Real-World Fidelity Limitation

Natural datasets do not provide:

\[
g_{ij}.
\]

Therefore true interaction fidelity cannot be measured directly.

Do NOT claim exact interaction recovery on real data.

---

# 52. Real-World Proxy Evidence

Use:

```text
joint predictive advantage
source-modality shuffle
single-modality predictability
linear accessibility
retrieval
task performance
```

as indirect evidence.

---

# 53. R2 Five-Seed Protocol

Once a mapping/task is selected:

\[
\boxed{
5\text{ seeds}
}
\]

for:

```text
Original ConFu
D0 if distinct implementation
D1
D2
```

---

# 54. R2 Statistical Reporting

Report:

```text
mean
sample standard deviation
per-seed values
paired difference
95% CI
effect size
```

where appropriate.

Do not over-focus on \(p<0.05\) with only five seeds.

---

# 55. Real-World D2 Success

D2 does NOT need to dominate D1 on every metric.

Valid successful outcomes include:

### Outcome A

\[
D2>D0
\]

in task accessibility.

### Outcome B

D2 approximately matches D1 task performance but shows cleaner interaction behavior.

### Outcome C

D2 is more robust to modality intervention.

### Outcome D

D2 better preserves retrieval/alignment.

---

# 56. Strong Real-World Success

Ideal:

\[
\boxed{
D2>D0
}
\]

and:

\[
\boxed{
D2\ge D1
}
\]

on task utility,

with:

\[
\boxed{
\text{better selectivity / robustness}.
}
\]

But this is not required for scientific validity.

---

# 57. Real-World Negative Result

If D2 fails despite:

\[
J>0,
\]

this is still informative.

Then ask whether:

```text
target geometry
representation accessibility
task relevance
```

causes the gap.

Do NOT immediately modify architecture.

---

# 58. R3 — Second Real-World Dataset

After MOSEI:

\[
\boxed{
\text{UR-FUNNY}
}
\]

becomes the preferred second real-world benchmark.

---

# 59. UR-FUNNY Role

UR-FUNNY is no longer the primary development benchmark.

Its role:

```text
transfer benchmark
hard negative/control dataset
text-dominant benchmark
comparison to historical ConFu results
```

---

# 60. UR-FUNNY Screening First

Before D2:

run exactly the same identifiability screening.

Mappings:

\[
V+A\rightarrow T
\]

\[
V+T\rightarrow A
\]

\[
A+T\rightarrow V.
\]

---

# 61. UR-FUNNY Existing Context

Historical ConFu:

\[
64.462\pm0.713\%.
\]

Do not chase this number before identifiability analysis.

---

# 62. Potential Strong Paper Result

A particularly strong scientific pattern would be:

\[
J_{MOSEI}>0
\]

for selected directions,

while:

\[
J_{URFUNNY}\approx0
\]

or much weaker.

Then:

\[
\boxed{
\text{identifiability screening explains why interaction methods help some datasets more than others}.
}
\]

---

# 63. Do Not Force Universal Gains

The paper does not require:

\[
D2
\]

to improve every benchmark.

A method with clear conditions of applicability can be stronger scientifically than an unexplained universal claim.

---

# 64. R4 — Optional Third Dataset

Only if time permits.

Candidate:

\[
\boxed{\text{MUStARD}}
\]

Role:

```text
small-data
cross-modal incongruity
generalization
```

Do not tune heavily on MUStARD.

---

# 65. Third-Order Discovery Status

Still:

\[
\boxed{\text{BLOCKED / STRETCH GOAL}}
\]

for the CVPR paper.

---

# 66. Why Third-Order Is Not Required Yet

Oracle already proves:

\[
h_{123}
\]

can represent third-order structure.

The stronger current paper story is:

\[
\text{interaction discovery}
+
\text{identifiability}
+
\text{real-world validation}.
\]

Do not dilute the paper unless pair-level transfer succeeds early.

---

# 67. Third-Order Trigger

Only implement discovered:

\[
h_{123}
\]

if all are true:

```text
MOSEI pair screening succeeds
D2 pair transfer succeeds
main paper tables are already complete
remaining compute/time permits
```

---

# 68. Future Order-3 Principle

Potential future target:

\[
\boxed{
d_{123}
=
q_{123}
-
q_{\le2}.
}
\]

Where:

\[
q_{\le2}
\]

must contain all validated first- and second-order components.

Do not implement in current phase.

---

# 69. Theory Track

In parallel with experiments, formalize JAD.

Do not postpone theory until all experiments finish.

---

# 70. Basic Decomposition

Assume:

\[
Y
=
A(X_1,X_2)
+
J(X_1,X_2)
+
\epsilon.
\]

Where:

\[
A
\]

belongs to the additive hypothesis class,

while:

\[
A+J
\]

belongs to the joint hypothesis class.

---

# 71. Population-Optimal Additive Predictor

If additive predictor is optimal:

\[
\boxed{
q_A^*=A.
}
\]

---

# 72. Population-Optimal Joint Predictor

If joint predictor is optimal:

\[
\boxed{
q_J^*=A+J.
}
\]

---

# 73. Ideal Joint Advantage

Then:

\[
\boxed{
q_J^*-q_A^*=J.
}
\]

This provides the conceptual motivation for JAD.

---

# 74. Finite Predictor Case

In practice:

\[
q_A=A+\epsilon_A
\]

\[
q_J=A+J+\epsilon_J.
\]

Therefore:

\[
\boxed{
q_J-q_A
=
J+
(\epsilon_J-\epsilon_A).
}
\]

---

# 75. Connection to Weak Regimes

When:

\[
\|J\|
\]

is small,

predictor-error difference:

\[
\epsilon_J-\epsilon_A
\]

can become comparable to the true interaction.

This matches synthetic weak-regime observations.

---

# 76. SER Connection

Synthetic experiments observed substantially smaller signal-to-error ratio in weak interaction regimes.

This empirical behavior should be connected to the finite-predictor equation.

---

# 77. Theory Claim Boundary

Do NOT claim:

\[
q_J-q_A
\]

is exact statistical interaction for arbitrary predictors.

It is:

\[
\boxed{
\text{hypothesis-class-dependent predictive interaction}.
}
\]

---

# 78. Relation to Functional ANOVA

The paper should discuss functional decomposition literature.

But JAD should be positioned as:

\[
\boxed{
\text{predictive representation-level decomposition}
}
\]

rather than exact functional ANOVA.

---

# 79. Relation to Synergy

Do NOT claim formal PID synergy.

Use:

```text
joint predictive structure
non-additive predictive structure
interaction representation
joint predictive advantage
```

---

# 80. Core CVPR Contributions

Target contribution list:

### Contribution 1

Introduce a framework distinguishing:

\[
\boxed{
\text{identifiability, fidelity, accessibility}
}
\]

for multimodal interaction representations.

### Contribution 2

Introduce:

\[
\boxed{
\text{IPIB}
}
\]

for controlled additive/joint/private decomposition.

### Contribution 3

Introduce:

\[
\boxed{
\text{Joint-Advantage Distillation}
}
\]

for explicit interaction discovery.

### Contribution 4

Show an empirical:

\[
\boxed{
\text{Fidelity–Accessibility Tradeoff}.
}
\]

### Contribution 5

Validate interaction identifiability and discovery on natural multimodal tasks.

---

# 81. Paper Claim Priority

The paper should NOT primarily claim:

```text
highest classification accuracy
SOTA fusion model
largest neural architecture
```

The primary novelty is:

\[
\boxed{
\text{understanding and discovering explicit predictable multimodal interaction}.
}
\]

---

# 82. CVPR Figure 1

Start preparing immediately.

Suggested conceptual figure:

```text
Multimodal inputs
      │
      ├───────────────┐
      │               │
      ▼               ▼
 Additive         Joint
 Predictor        Predictor
      │               │
      └──────┬────────┘
             │
             ▼
       q_joint - q_add
             │
             ▼
       Interaction target
             │
             ▼
            h_ij
             │
             ▼
      downstream probe
```

Include:

```text
Identifiability gate J
```

before interaction distillation.

---

# 83. CVPR Figure 2

Synthetic concept:

X-axis:

\[
\beta
\]

interaction strength.

Y-axis:

\[
interaction recovery
\]

or:

\[
linear accessibility.
\]

Curves:

```text
Oracle
D0
D1
D2
```

---

# 84. CVPR Figure 3

Fidelity–Accessibility plot:

X-axis:

\[
interaction fidelity
\]

Y-axis:

\[
linear downstream accessibility.
\]

Show:

```text
D0
D1
D2
B2
```

This visually demonstrates the tradeoff.

---

# 85. CVPR Main Table 1

Controlled IPIB:

| Method | I0 | I1 | I2 | I3 | I4 | Target Fidelity |
|---|---:|---:|---:|---:|---:|---:|
| D0 | | | | | | |
| D1 | | | | | | |
| D2 | | | | | | |

Use frozen five-seed results.

---

# 86. CVPR Main Table 2

MOSEI screening:

| Task | Direction | Add R² | Joint R² | J |
|---|---|---:|---:|---:|

---

# 87. CVPR Main Table 3

Real-world comparison:

| Dataset/Task | ConFu | D1 | D2 |
|---|---:|---:|---:|
| MOSEI ... | | | |
| UR-FUNNY | | | |

Include primary task metric.

---

# 88. Supplementary

Move detailed negative explorations into supplementary:

```text
S1 adversarial de-shortcutting
v2 residual correction
v3 mapping failure
v3.1 predictor capacity audit
v4.1 identifiability failure
B2 calibration
C0 reliability
C1 failure
generic MLP detailed sweeps
```

These strengthen rigor without overloading the main paper.

---

# 89. Paper Writing Starts Now

Do NOT wait for all experiments before writing.

Immediately draft:

```text
Introduction
Related Work
Method
Synthetic Benchmark
Theory
Synthetic Results
```

Leave real-world result sections incomplete until experiments finish.

---

# 90. Introduction Narrative

Recommended structure:

1. Multimodal fusion increasingly models higher-order modality combinations.
2. Existing alignment objectives do not guarantee interaction discovery.
3. Interaction cannot be learned when cross-modal targets are non-identifiable.
4. Even faithful interaction recovery does not guarantee downstream accessibility.
5. We introduce a framework and JAD to explicitly study these issues.

---

# 91. Research Question 1

\[
\boxed{
\text{When is multimodal interaction identifiable?}
}
\]

Answer using NIC + IPIB + real-world screening.

---

# 92. Research Question 2

\[
\boxed{
\text{Can explicit interaction be discovered?}
}
\]

Answer using Oracle + D0/D1/D2.

---

# 93. Research Question 3

\[
\boxed{
\text{Does faithful interaction recovery imply downstream utility?}
}
\]

Answer:

\[
\boxed{\text{No, not necessarily.}}
\]

Supported by the D1/D2 tradeoff.

---

# 94. Research Question 4

\[
\boxed{
\text{Does the framework transfer to natural multimodal data?}
}
\]

This is the current open question.

---

# 95. CVPR Stop Rule

From now on, every new experiment must satisfy at least one:

```text
tests a main claim
answers a reviewer objection
provides required baseline
provides real-world transfer
provides robustness evidence
```

Otherwise:

\[
\boxed{\text{DO NOT RUN IT}.}
\]

---

# 96. Forbidden Work During R1

Do NOT:

```text
change D2 target
change D2 loss
change rank
add attention
add routing
add h123
use labels in predictors
optimize UR-FUNNY
```

R1 is screening only.

---

# 97. Immediate Code Task

Implement:

```text
src/experiments/multibench/mosei_identifiability.py
```

or equivalent module consistent with current repository organization.

---

# 98. MOSEI Predictor API

Create shared function:

```text
fit_additive_predictor(source_a, source_b, target)
```

and:

```text
fit_joint_predictor(source_a, source_b, target)
```

Return:

```text
train_r2
val_r2
test_r2
parameter_count
checkpoint
```

---

# 99. MOSEI Screening Config

Config should include:

```text
emotion
mapping
seed
predictor_hidden_dim
optimizer
learning_rate
weight_decay
batch_size
patience
```

---

# 100. Capacity-Matching Test

Unit-test:

\[
\frac{
|Params_J-Params_A|
}{
Params_A
}
<0.05.
\]

Prefer:

\[
<0.01.
\]

---

# 101. Screening Result JSON

Example:

```json
{
  "task": "happiness",
  "mapping": "VA_to_T",
  "seed": 1,
  "params_add": 0,
  "params_joint": 0,
  "add_train_r2": 0.0,
  "add_val_r2": 0.0,
  "add_test_r2": 0.0,
  "joint_train_r2": 0.0,
  "joint_val_r2": 0.0,
  "joint_test_r2": 0.0,
  "joint_advantage_val": 0.0,
  "joint_advantage_test": 0.0
}
```

---

# 102. Screening Summary

Produce ranking by:

\[
J_{val}.
\]

But use ranking only for experiment selection.

Do not call the highest-\(J\) emotion:

```text
most synergistic emotion
```

---

# 103. Selection Limit

Do not run full D0/D1/D2 on every MOSEI emotion.

Select:

\[
1\text{–}3
\]

strongest validated settings.

This keeps the paper focused.

---

# 104. If No MOSEI Mapping Passes

Do not force JAD.

Then immediately:

1. document the negative screening;
2. screen UR-FUNNY;
3. screen MUStARD or another natural multimodal benchmark.

The identifiability framework itself predicts such failures.

---

# 105. If MOSEI Passes

Immediately freeze selected mapping/task and run R2.

No further screening tuning.

---

# 106. R2 Development Sequence

```text
1. Original ConFu baseline
2. D0 matched implementation
3. D1
4. D2
5. Linear probes
6. Nonlinear probes
7. Shuffle diagnostics
8. Retrieval
9. Five-seed confirmation
```

---

# 107. Accuracy Chase Rule

Do not ask:

> How do I make D2 accuracy larger?

Ask:

> Does JAD behave as predicted by identifiability and fidelity/accessibility theory?

This distinction is mandatory.

---

# 108. Real-World Success Story A

Best case:

\[
J>0
\]

and:

\[
D2>D0
\]

and:

\[
D2\ge D1.
\]

Strongest empirical result.

---

# 109. Real-World Success Story B

Also valid:

\[
J>0
\]

\[
D2>D0
\]

but:

\[
D1>D2
\]

with the same synthetic accessibility tradeoff.

This demonstrates transfer of the phenomenon.

---

# 110. Real-World Success Story C

Also valuable:

MOSEI:

\[
J>0
\]

and D2 useful.

UR-FUNNY:

\[
J\approx0
\]

and D2 not useful.

This supports:

\[
\boxed{
\text{identifiability as a predictive criterion for interaction methods}.
}
\]

---

# 111. Real-World Failure Story

If:

\[
J>0
\]

but D2 fails everywhere,

the current CVPR story weakens.

Do not hide this.

Reassess task-accessibility mechanism before submission.

---

# 112. Current Main Method Is Frozen

Until real-world R2 is complete:

\[
\boxed{
\textbf{NO METHOD CHANGES}.
}
\]

This protects scientific validity.

---

# 113. Current Paper Title Candidates

Primary:

**Beyond Higher-Order Alignment: Identifiability, Fidelity, and Accessibility in Multimodal Interaction Learning**

Alternative:

**Discovering Predictable Multimodal Interactions with Joint-Advantage Distillation**

Alternative:

**When Is Multimodal Interaction Learnable? Identifiability and Joint-Advantage Distillation**

Alternative:

**Fusion Is Not Interaction: Discovering Explicit Multimodal Structure Beyond Higher-Order Alignment**

---

# 114. Preferred Current Title

\[
\boxed{
\textbf{Beyond Higher-Order Alignment: Identifiability, Fidelity, and Accessibility in Multimodal Interaction Learning}
}
\]

because the paper contribution is broader than JAD alone.

---

# 115. Main Method Name

Keep:

\[
\boxed{
\textbf{Joint-Advantage Distillation (JAD)}
}
\]

unless later novelty search identifies a naming conflict.

---

# 116. Main Benchmark Name

Keep:

\[
\boxed{
\textbf{Identifiable Predictive-Interaction Benchmark (IPIB)}
}
\]

unless a clearer paper-ready name is chosen later.

---

# 117. Reproducibility Requirements

Every real-world run must save:

```text
git commit
dataset split
seed
encoder config
predictor config
parameter counts
optimizer
checkpoint
validation metrics
test metrics
interaction health
probe results
```

---

# 118. Seed Discipline

Seed:

```text
Python
NumPy
PyTorch CPU
CUDA
model initialization
DataLoader
probe initialization
```

where applicable.

---

# 119. Unit Tests

Required:

```text
mapping correctness
predictor capacity matching
no target leakage
no test-set model selection
deterministic initialization
J computation
identifiability gate
D2 zero-target behavior
```

---

# 120. Paper Artifact Folder

Create:

```text
paper/
```

Suggested structure:

```text
paper/
    figures/
    tables/
    notes/
    claims.md
    experiment_matrix.md
    reviewer_questions.md
```

---

# 121. claims.md

Maintain a live file containing:

```text
Claim
Evidence
Datasets
Experiment IDs
Limitations
Status
```

Every paper claim must map to evidence.

---

# 122. reviewer_questions.md

Track likely reviewer objections:

```text
Why not generic fusion?
Why is q_joint - q_add interaction?
What if predictor capacity differs?
What if target is not identifiable?
Does interaction fidelity imply utility?
Does the method work outside synthetic data?
Why not use task supervision?
Why not PID synergy?
Does D2 preserve original ConFu retrieval?
```

Each must have an experiment or explicit limitation.

---

# 123. Main Reviewer Risk

The largest current risk is:

\[
\boxed{
\text{synthetic-only evidence}.
}
\]

R1/R2 are designed to eliminate this risk.

---

# 124. Second Reviewer Risk

Potential objection:

> JAD is only predictor subtraction.

Response must include:

```text
capacity-controlled predictors
identifiability gating
Oracle mechanism validation
IPIB ground-truth recovery
false-positive controls
real-world transfer
```

---

# 125. Third Reviewer Risk

Potential objection:

> Why not use D1 if D1 has better classification?

Response:

D1 and D2 optimize different properties.

D1:

\[
\text{accessibility-oriented residual}.
\]

D2:

\[
\text{selective predictable interaction}.
\]

The paper must show why this distinction matters in natural data.

---

# 126. Fourth Reviewer Risk

Potential objection:

> JAD is not higher-order because it is deterministic.

Response:

The claim is not new Shannon information.

The claim is:

\[
\boxed{
\text{explicit reparameterization of non-additive predictive structure}.
}
\]

---

# 127. Fifth Reviewer Risk

Potential objection:

> Interaction depends on hypothesis class.

Correct response:

\[
\boxed{\text{Yes.}}
\]

JAD identifies:

\[
\boxed{
\text{predictive interaction relative to specified additive and joint hypothesis classes}.
}
\]

This limitation must be stated.

---

# 128. Current Research Priority

Priority order:

\[
\boxed{
R1:\ MOSEI\ Identifiability
}
\]

\[
\downarrow
\]

\[
\boxed{
R2:\ MOSEI\ D0/D1/D2
}
\]

\[
\downarrow
\]

\[
\boxed{
R3:\ UR\text{-}FUNNY\ Screening/Transfer
}
\]

\[
\downarrow
\]

\[
\boxed{
Theory + Main Tables
}
\]

\[
\downarrow
\]

\[
\boxed{
Optional\ third\ dataset
}
\]

\[
\downarrow
\]

\[
\boxed{
CVPR\ submission
}
\]

---

# 129. Immediate Milestone

The next milestone is:

\[
\boxed{
\textbf{find whether CMU-MOSEI contains stable jointly identifiable modality mappings.}
}
\]

Not:

```text
increase UR-FUNNY accuracy
discover h123
design another interaction block
```

---

# 130. Current Hard Stop

If an experiment does not contribute to:

```text
identifiability
real-world transfer
required baseline
robustness
theory
reviewer defense
```

do not run it.

---

# 131. Paper-Level Success Condition

Before CVPR submission, aim to have:

```text
Oracle mechanism validation
NIC negative control
IPIB benchmark
D0/D1/D2 comparison
JAD method
fidelity-accessibility finding
theoretical derivation
MOSEI real-world screening
MOSEI real-world interaction experiment
UR-FUNNY screening/transfer
matched-capacity baselines
retrieval/alignment comparison
five-seed core results
```

---

# 132. Minimum CVPR Gate

The project should not be considered ready for CVPR main track until:

\[
\boxed{
\text{at least one natural-data experiment supports the interaction-discovery story}.
}
\]

Synthetic results alone are insufficient.

---

# 133. Strong CVPR Gate

Preferred:

\[
\boxed{
\text{at least two natural-data settings}
}
\]

show interpretable behavior consistent with the framework.

They do not both need positive classification improvements.

---

# 134. Final Scientific Thesis

\[
\boxed{
\textbf{Multimodal interaction learning is not determined by fusion capacity alone: interaction must first be identifiable, then faithfully recovered, and finally made accessible to downstream tasks.}
}
\]

---

# 135. Final Method Thesis

\[
\boxed{
\textbf{Joint-Advantage Distillation isolates predictable non-additive multimodal structure by distilling the advantage of a capacity-matched joint predictor over additive predictors.}
}
\]

---

# 136. Current Required Experiment

\[
\boxed{
\textbf{R1 — CMU-MOSEI Emotion Identifiability Screening}
}
\]

using:

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V
\]

for each available emotion task.

---

# 137. Immediate Execution Checklist

```text
[ ] Audit current MOSEI pipeline and splits.

[ ] Confirm modality shapes.

[ ] Confirm emotion labels.

[ ] Implement capacity-matched additive predictor.

[ ] Implement capacity-matched joint predictor.

[ ] Unit-test parameter counts.

[ ] Unit-test mapping directions.

[ ] Run one seed smoke test.

[ ] Verify train/val/test R² logging.

[ ] Run 3-seed screening.

[ ] Produce MOSEI_IDENTIFIABILITY_REPORT.md.

[ ] Select only validation-supported mappings.

[ ] Freeze selected mappings.

[ ] Proceed to R2 D0/D1/D2.
```

---

# 138. Final Rule

From v5.0 onward:

\[
\boxed{
\textbf{the project is paper-driven, not tuning-driven.}
}
\]

Every new experiment must either:

\[
\boxed{
\text{strengthen a CVPR claim}
}
\]

or:

\[
\boxed{
\text{eliminate a plausible reviewer objection}.
}
\]

The next action is:

\[
\boxed{
\textbf{CMU-MOSEI Emotion real-world identifiability screening.}
}
\]