# AGENT.md

# ConFu++ v3.1

## Capacity-Controlled Order-Decomposed Higher-Order Multimodal Representation Learning

---

# 0. Mission

This repository develops a direct extension of **ConFu: Contrastive Fusion for Higher-Order Multimodal Alignment**.

The current objective is:

\[
\boxed{
\text{learn explicit order-specific multimodal embeddings}
}
\]

that expose non-additive multimodal structure in a form that is easier for simple downstream models to access.

The project is NOT trying to create new Shannon information.

Instead:

\[
\boxed{
\text{higher-order embedding}
=
\text{reparameterization of existing joint information}
}
\]

into a representation with improved:

- accessibility,
- order specificity,
- modality dependence,
- downstream usefulness.

---

# 1. Current Research Status

Completed stages:

```text
AV-MNIST diagnosis
DONE

CMU-MOSI diagnosis
DONE

UR-FUNNY Original ConFu reproduction
DONE

ConFu++ S1
REJECTED

ConFu++ v2 residual correction
REJECTED

ConFu++ v3 E4 linear-additive residual alignment
REJECTED
```

The next stage is:

\[
\boxed{
\text{E4.1 — Predictor Capacity Audit}
}
\]

---

# 2. Canonical UR-FUNNY Baseline

Official matched Original ConFu baseline:

\[
\boxed{
64.462\pm0.713\%
}
\]

across five seeds.

This result must remain frozen.

Do not modify or overwrite canonical baseline artifacts.

---

# 3. v2 Result

ConFu++ v2 demonstrated that post-hoc use of frozen ConFu pair representations does not provide reliable utility.

E0:

\[
\Delta_{pair}^{linear}
\approx0
\]

while:

\[
\Delta_{pair}^{nonlinear}>0
\]

on average but is highly unstable.

E1–E3 produced:

\[
NetCorrection<0.
\]

Conclusion:

\[
\boxed{
\text{classifier-level correction is not the main bottleneck}.
}
\]

---

# 4. v3 E4 Result

ConFu++ v3 introduced:

\[
h_{12},h_{13},h_{23}
\]

as explicit second-order multiplicative representations.

The representations were trained using residual targets produced from additive linear first-order predictors.

E4 failed the main accessibility gate.

Validation:

\[
\Delta_2^{linear}
=
-1.832\text{ pp}.
\]

Test:

\[
\Delta_2^{linear}
=
-2.662\text{ pp}.
\]

Nonlinear exploratory:

\[
\Delta_2^{MLP}
=
+1.512\text{ pp}.
\]

Therefore:

\[
\boxed{
\text{E4 does not improve explicit linear accessibility}.
}
\]

---

# 5. E4 Did Not Fail Because of Collapse

Interaction representations remain active.

They have:

- non-zero variance;
- substantial effective rank;
- positive modality-shuffle sensitivity;
- non-trivial dependence on both inputs.

Therefore:

\[
\boxed{
\text{representation collapse is not the current bottleneck}.
}
\]

---

# 6. Most Important New Bottleneck

The current residual target may be incorrectly defined.

E4 uses:

\[
\hat r_j^{(1)}
=
q_i^{linear}(r_i)
+
q_k^{linear}(r_k)
\]

and:

\[
e_j^{(2)}
=
r_j-\hat r_j^{(1)}.
\]

But a linear predictor cannot represent nonlinear relationships such as:

\[
r_j=f_i(r_i)
\]

where:

\[
f_i
\]

is nonlinear.

Therefore:

\[
e_j^{(2)}
\]

may contain:

\[
\boxed{
\text{unimodal nonlinear residue}
}
\]

in addition to:

\[
\boxed{
\text{actual multimodal non-additive structure}.
}
\]

---

# 7. New Residual Interpretation

The current residual may be:

\[
e_j^{(2)}
=
N_i(r_i)
+
N_k(r_k)
+
I_{ik}(r_i,r_k)
+
\epsilon
\]

where:

- \(N_i\): nonlinear information from modality \(i\);
- \(N_k\): nonlinear information from modality \(k\);
- \(I_{ik}\): actual joint interaction;
- \(\epsilon\): noise/unmodelled structure.

The purpose of E4.1 is to separate:

\[
N_i+N_k
\]

from:

\[
I_{ik}.
\]

---

# 8. Critical Headroom Correction

Previous v3 headroom audit compared:

\[
\text{linear additive predictor}
\]

against:

\[
\text{joint nonlinear MLP}.
\]

This comparison changes two variables simultaneously:

1. additive → joint;
2. linear → nonlinear.

Therefore the previous quantity:

\[
R^2_{jointMLP}
-
R^2_{linearAdd}
\]

cannot be interpreted cleanly as non-additive multimodal headroom.

---

# 9. Correct Headroom Audit

The correct experiment requires three hypothesis classes:

\[
\boxed{
P0=\text{Linear Additive}
}
\]

\[
\boxed{
P1=\text{Nonlinear Additive}
}
\]

\[
\boxed{
P2=\text{Joint Nonlinear}
}
\]

The primary comparison becomes:

\[
\boxed{
P2-P1
}
\]

rather than:

\[
P2-P0.
\]

---

# 10. E4.1 — Predictor Capacity Audit

Implement:

```text
E4.1
Predictor Capacity Audit
```

No interaction retraining yet.

No architecture change.

No task loss.

No test-driven hyperparameter search.

---

# 11. P0 — Linear Additive Predictor

For target:

\[
r_j,
\]

from modalities:

\[
r_i,r_k,
\]

define:

\[
\hat r_j^{P0}
=
W_ir_i
+
W_kr_k
+
b.
\]

This is the existing v3 predictor.

---

# 12. P1 — Nonlinear Additive Predictor

Use independent networks:

\[
q_i(r_i)
\]

and:

\[
q_k(r_k).
\]

Then:

\[
\boxed{
\hat r_j^{P1}
=
q_i(r_i)
+
q_k(r_k)
+
b.
}
\]

Critically:

\[
q_i
\]

must never see:

\[
r_k.
\]

And:

\[
q_k
\]

must never see:

\[
r_i.
\]

Therefore P1 remains an order-1 additive hypothesis class.

---

# 13. P1 Architecture

Default starting architecture:

\[
256\rightarrow128\rightarrow256.
\]

Recommended:

```text
LayerNorm
Linear(256,128)
GELU
Linear(128,256)
```

No:

```text
cross-attention
concatenation
joint input
Transformer
deep residual MLP
```

---

# 14. P1 Smaller Ablation

Optionally evaluate:

\[
256\rightarrow64\rightarrow256.
\]

Call:

```text
P1-small
```

Only if predictor capacity sensitivity is unclear.

Do not perform large hidden-dimension sweeps.

---

# 15. P2 — Joint Nonlinear Predictor

Define:

\[
\hat r_j^{P2}
=
q_{joint}([r_i;r_k]).
\]

The predictor receives both modalities jointly.

This model is allowed to learn:

\[
f(r_i,r_k).
\]

---

# 16. Capacity Matching

P1 and P2 must have approximately equal parameter budgets.

Require:

\[
Params(P1)
\approx
Params(P2).
\]

Tolerance target:

\[
<5\%
\]

difference where practical.

Always report exact parameter counts.

---

# 17. Why Capacity Matching Is Mandatory

Otherwise:

\[
R^2_{P2}>R^2_{P1}
\]

could simply result from:

\[
\text{more parameters}.
\]

The purpose is to isolate:

\[
\boxed{
\text{joint functional capacity}
}
\]

rather than generic network size.

---

# 18. Primary E4.1 Metric

Define:

\[
\boxed{
J_{ik\rightarrow j}
=
R^2_{P2}
-
R^2_{P1}.
}
\]

Call this:

\[
\boxed{
\text{Joint Prediction Advantage}
}
\]

or:

\[
\boxed{
\text{Non-Additive Prediction Advantage}.
}
\]

Do NOT call it formal synergy.

---

# 19. E4.1 Targets

For three modalities:

\[
r_1,r_2,r_3,
\]

evaluate:

\[
J_{23\rightarrow1}
\]

\[
J_{13\rightarrow2}
\]

\[
J_{12\rightarrow3}.
\]

All three must be reported separately.

---

# 20. Required Headroom Table

Produce:

| Target | P0 Linear Additive R² | P1 Nonlinear Additive R² | P2 Joint MLP R² | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| \(r_1\leftarrow r_2,r_3\) | | | | | |
| \(r_2\leftarrow r_1,r_3\) | | | | | |
| \(r_3\leftarrow r_1,r_2\) | | | | | |

---

# 21. Pairwise Unimodal Nonlinearity Audit

Also measure every directed unimodal prediction.

For:

\[
r_i\rightarrow r_j,
\]

compare:

### Linear

\[
W_{ij}r_i
\]

against:

### MLP

\[
q_{ij}(r_i).
\]

---

# 22. Required Unimodal Audit Table

| Source → Target | Linear R² | MLP R² | Nonlinear gain |
|---|---:|---:|---:|
| Vision → Audio | | | |
| Vision → Text | | | |
| Audio → Vision | | | |
| Audio → Text | | | |
| Text → Vision | | | |
| Text → Audio | | | |

Define:

\[
N_{i\rightarrow j}
=
R^2_{MLP}
-
R^2_{linear}.
\]

---

# 23. Interpretation of Large Unimodal Nonlinear Gain

If:

\[
N_{i\rightarrow j}\gg0,
\]

then E4 linear residual likely contains significant nonlinear single-modality structure.

This supports rebuilding residual targets with P1.

---

# 24. E4.1 Decision Case A

Suppose:

\[
P1\approx P2.
\]

Then:

\[
J\approx0.
\]

Interpretation:

\[
\boxed{
\text{most previous P2-P0 headroom came from nonlinear unimodal functions}.
}
\]

Do NOT proceed with E4.2 on UR-FUNNY unless at least one target still has meaningful validated joint advantage.

---

# 25. Case A Action

If all:

\[
J_{ik\rightarrow j}
\]

are small:

```text
STOP UR-FUNNY mechanism development.
```

Move to:

\[
\boxed{
\text{Controlled Synthetic Higher-Order Benchmark}.
}
\]

Do not add architecture to compensate.

---

# 26. E4.1 Decision Case B

Suppose:

\[
P2>P1
\]

clearly for one or more targets.

Then the frozen representations contain structure that independent nonlinear unimodal predictors cannot explain.

Interpretation:

\[
\boxed{
\text{empirical non-additive representation headroom exists}.
}
\]

Proceed to:

\[
\boxed{
E4.2.
}
\]

---

# 27. No Arbitrary J Threshold Yet

Do not hard-code:

```text
J > 0.1
```

or another arbitrary numeric threshold.

Use:

- validation consistency;
- magnitude relative to predictor variance;
- bootstrap/seed uncertainty where available.

Seed 1 remains exploratory.

---

# 28. E4.2 — Nonlinear Additive Residual Alignment

If E4.1 passes, replace residual predictors.

Old:

\[
\hat r_j^{(1)}
=
q_i^{linear}(r_i)
+
q_k^{linear}(r_k).
\]

New:

\[
\boxed{
\hat r_j^{(1)}
=
q_i^{MLP}(r_i)
+
q_k^{MLP}(r_k).
}
\]

---

# 29. E4.2 Residual Target

Define:

\[
\boxed{
e_j^{(2)}
=
r_j
-
stopgrad(
q_i^{MLP}(r_i)
+
q_k^{MLP}(r_k)
)
}
\]

for all three target modalities.

---

# 30. Freeze P1 Predictors

Train P1 first.

Select using:

\[
validation\ reconstruction\ loss.
\]

Then freeze.

Do NOT jointly optimize P1 and interaction representations.

Otherwise P1 could intentionally deteriorate to enlarge the residual.

---

# 31. Residual Health Audit v2

For each new residual:

\[
e_j^{(2)},
\]

report:

```text
variance
effective rank
mean L2 norm
dimension std
R² reconstructed by P1
cosine to original target
```

---

# 32. Compare Old and New Residuals

Produce:

| Target | Linear-residual variance | Nonlinear-residual variance | Linear rank | Nonlinear rank |
|---|---:|---:|---:|---:|
| \(e_1\) | | | | |
| \(e_2\) | | | | |
| \(e_3\) | | | | |

The new residual will likely be smaller.

That is expected.

---

# 33. Interaction Architecture Must Remain Fixed

During E4.2 do NOT change:

```text
rank
interaction output dimension
temperature
optimizer family
pair architecture
LayerNorm placement
```

unless required for numerical stability.

The experiment must isolate residual-target quality.

---

# 34. E4.2 Interaction Branch

Keep:

\[
u_i=LN(U_ir_i)
\]

\[
u_j=LN(U_jr_j)
\]

and:

\[
h_{ij}
=
W_{ij}
\left(
\frac{
u_i\odot u_j
}{
\sqrt R
}
\right).
\]

Default:

\[
R=64.
\]

---

# 35. E4.2 Alignment

Train:

\[
h_{12}
\leftrightarrow
e_3^{(2)}
\]

\[
h_{13}
\leftrightarrow
e_2^{(2)}
\]

\[
h_{23}
\leftrightarrow
e_1^{(2)}.
\]

Use the same symmetric InfoNCE formulation as E4.

---

# 36. Do Not Add Task Loss

E4.2 must NOT use:

\[
L_{task}.
\]

Otherwise improved accessibility could come from direct label supervision rather than a better order decomposition.

---

# 37. No Encoder Fine-Tuning

Keep Original ConFu encoders frozen.

E4.2 is still a representation-mechanism experiment.

Do not enable end-to-end adaptation.

---

# 38. Primary Representation Sets

First order:

\[
Z_{\le1}
=
[r_1;r_2;r_3].
\]

Second order:

\[
Z_{\le2}
=
[
r_1;r_2;r_3;
h_{12};h_{13};h_{23}
].
\]

---

# 39. Primary E4.2 Metric

\[
\boxed{
\Delta_2^{linear}
=
Perf_{linear}(Z_{\le2})
-
Perf_{linear}(Z_{\le1})
}
\]

Primary selection uses validation.

Test result must not determine model design.

---

# 40. Secondary Metric

\[
\Delta_2^{MLP}
=
Perf_{MLP}(Z_{\le2})
-
Perf_{MLP}(Z_{\le1}).
\]

This is secondary.

A positive nonlinear result cannot override a negative validation linear gate.

---

# 41. New Linearization Metric

Define:

\[
LG_1
=
Perf_{MLP}(Z_{\le1})
-
Perf_{linear}(Z_{\le1})
\]

and:

\[
LG_2
=
Perf_{MLP}(Z_{\le2})
-
Perf_{linear}(Z_{\le2}).
\]

Call:

\[
LG
\]

the **Linearization Gap**.

---

# 42. Linearization Objective

A good explicit higher-order embedding ideally makes interaction structure easier for a simple model.

Therefore desirable behavior is:

\[
LG_2<LG_1
\]

while task performance remains competitive.

This is a secondary scientific metric.

---

# 43. Strong Accessibility Pattern

Ideal:

\[
\Delta_2^{linear}>0
\]

and:

\[
LG_2<LG_1.
\]

Interpretation:

> second-order embedding improves performance and reduces the amount of nonlinear downstream computation required.

---

# 44. Pair-Specific Accessibility

Measure:

\[
\Delta_{12}
=
Perf(Z_{\le1},h_{12})
-
Perf(Z_{\le1})
\]

and similarly:

\[
\Delta_{13},
\Delta_{23}.
\]

Use both linear and nonlinear probes.

---

# 45. h23 Is a Priority Diagnostic

Current E4 showed:

\[
h_{23}
\]

had the strongest exploratory nonlinear gain.

Therefore closely monitor:

\[
\Delta_{23}^{linear}
\]

after improved residual construction.

Do NOT give \(h_{23}\) a special loss or larger model.

Only analyze it separately.

---

# 46. Representation Scale Diagnostics

Log:

\[
\|r_i\|_2
\]

and:

\[
\|h_{ij}\|_2.
\]

Compute:

\[
S_{ij,i}
=
\frac{
E[\|h_{ij}\|_2]
}{
E[\|r_i\|_2]+\epsilon
}.
\]

---

# 47. Why Scale Matters

An interaction representation can be useful but badly calibrated in magnitude.

Concatenation may distort:

```text
feature scale
regularization
probe optimization
```

even with standardized inputs.

This is a diagnostic only in E4.2.

---

# 48. Do Not Add Scaling Gate Yet

Do NOT immediately add:

\[
\alpha_{ij}h_{ij}.
\]

Only consider learned scaling if E4.2 demonstrates:

```text
positive interaction signal
+
obvious scale mismatch
```

---

# 49. Interaction Health

For every:

\[
h_{ij},
\]

log:

```text
variance
effective rank
mean norm
dimension std
```

These are health metrics.

They do not prove order utility.

---

# 50. Modality Dependence

For:

\[
h_{12},
\]

compute:

\[
h_{12}(\pi(r_1),r_2)
\]

and:

\[
h_{12}(r_1,\pi(r_2)).
\]

Report:

\[
D_1,D_2.
\]

Repeat for all pairs.

---

# 51. Single-Modality Predictability

For each pair:

\[
R^2(r_i\rightarrow h_{ij})
\]

and:

\[
R^2(r_j\rightarrow h_{ij}).
\]

Use nonlinear predictors where practical.

This detects one-modality shortcuts.

---

# 52. Joint Predictability

You may report:

\[
R^2([r_i,r_j]\rightarrow h_{ij}).
\]

But:

\[
\boxed{
\text{joint R² is not an acceptance gate}.
}
\]

---

# 53. E4.2 Primary Gate

Pass if validation shows:

\[
\boxed{
\Delta_2^{linear}>0.
}
\]

Prefer positive pair-specific gains as supporting evidence.

---

# 54. E4.2 Failure Rule

If:

\[
\Delta_2^{linear}\le0
\]

after a valid nonlinear-additive residual target:

```text
STOP UR-FUNNY representation architecture development.
```

Do not add:

```text
task loss
cross-attention
deeper fusion
rank sweep
dynamic gates
h123
```

---

# 55. Interpretation of E4.2 Failure

If E4.1 showed:

\[
J>0
\]

but E4.2 still shows:

\[
\Delta_2\le0,
\]

the result means:

\[
\boxed{
\text{non-additive joint structure exists}
}
\]

but:

\[
\boxed{
\text{residual contrastive alignment does not make it task-accessible}.
}
\]

This is a meaningful negative result.

---

# 56. Five-Seed Trigger

Do not run E4.2 five seeds automatically.

Trigger only if:

```text
validation Delta2 > 0
seed-1 representation health passes
no obvious leakage
test exploratory result is not contradictory
```

---

# 57. Confirmatory E4.2

Once configuration is frozen:

```text
seeds = 1,2,3,4,5
```

Report:

\[
mean\pm sample\ std.
\]

Also report:

```text
per-seed Δ2
paired difference
95% CI
paired test
effect size
```

---

# 58. Preferred Confirmatory Pattern

Prefer:

\[
4/5
\]

or:

\[
5/5
\]

positive:

\[
\Delta_2.
\]

A positive mean driven by one outlier seed is insufficient.

---

# 59. No h123 Yet

Explicit:

\[
h_{123}
\]

remains blocked.

Trigger requires:

\[
\boxed{
\Delta_2^{linear}>0
}
\]

stably.

---

# 60. No E5 Task Loss Yet

Task-aware order alignment remains blocked.

Do not run E5 until E4.2 establishes that residual alignment itself produces useful structure.

---

# 61. No End-to-End Fine-Tuning Yet

Encoder adaptation remains blocked.

Reason:

> if the frozen mechanism does not work, end-to-end adaptation could hide the failure through generic supervised representation drift.

---

# 62. Synthetic Benchmark Trigger

Move to controlled synthetic benchmark if either:

### Condition A

\[
J=P2-P1\approx0
\]

for UR-FUNNY.

Or:

### Condition B

\[
J>0
\]

but E4.2 still fails:

\[
\Delta_2^{linear}\le0.
\]

---

# 63. Synthetic Benchmark Goal

The synthetic benchmark must explicitly control:

\[
\text{order-1}
\]

\[
\text{order-2}
\]

\[
\text{order-3}
\]

task structure.

The goal is to test whether the method recovers the correct representational order.

---

# 64. Synthetic Functional Decomposition

Construct:

\[
Y
=
\sum_i
\alpha_i f_i(X_i)
+
\sum_{i<j}
\beta_{ij}f_{ij}(X_i,X_j)
+
\gamma f_{123}(X_1,X_2,X_3)
+
\epsilon.
\]

Control:

\[
\alpha,
\beta,
\gamma.
\]

---

# 65. Synthetic Regime S0 — First Order Only

Set:

\[
\beta_{ij}=0
\]

and:

\[
\gamma=0.
\]

Desired:

\[
\Delta_2\approx0
\]

and:

\[
\Delta_3\approx0.
\]

The model must not invent higher-order utility.

---

# 66. Synthetic Regime S1 — One Pair Interaction

Example:

\[
\beta_{12}>0
\]

while:

\[
\beta_{13}=\beta_{23}=0
\]

and:

\[
\gamma=0.
\]

Desired:

\[
\Delta_{12}>0
\]

but:

\[
\Delta_{13}\approx0
\]

\[
\Delta_{23}\approx0.
\]

---

# 67. Synthetic Regime S2 — All Pairwise

Set:

\[
\beta_{12},
\beta_{13},
\beta_{23}>0
\]

and:

\[
\gamma=0.
\]

Desired:

\[
\Delta_2>0
\]

but:

\[
\Delta_3\approx0.
\]

---

# 68. Synthetic Regime S3 — Pure Third Order

Set:

\[
\beta_{ij}=0
\]

and:

\[
\gamma>0.
\]

Desired:

\[
\Delta_2\approx0
\]

but:

\[
\boxed{
\Delta_3>0.
}
\]

This is the cleanest test of explicit third-order embedding.

---

# 69. Synthetic Regime S4 — Mixed Order

Set:

\[
\alpha>0,
\beta>0,
\gamma>0.
\]

Desired:

\[
Perf(Z_{\le1})
<
Perf(Z_{\le2})
<
Perf(Z_{\le3}).
\]

---

# 70. Controlled XOR Is Not Enough

Do not use only:

\[
X_1\oplus X_2\oplus X_3.
\]

Synthetic benchmark must include:

```text
continuous variables
noise
redundancy
unique components
pair interactions
triple interactions
```

to avoid solving a single toy function.

---

# 71. Synthetic Difficulty Sweep

Control:

\[
SNR
\]

and:

\[
interaction\ strength.
\]

Example:

```text
weak
medium
strong
```

for:

\[
\beta,\gamma.
\]

---

# 72. Synthetic Success Criteria

A valid order-decomposed model should recover:

```text
no higher-order gain when none exists

pair gain when pair structure exists

third-order gain when triple structure exists
```

without producing false order gains.

---

# 73. After Synthetic Validation

Only after mechanism validation move back to natural benchmarks.

Priority:

\[
\boxed{
MOSEI\ Emotion
}
\]

then:

\[
\boxed{
MUStARD.
}
\]

---

# 74. Why MOSEI Emotion Next

Sentiment tasks often have strong text dominance.

Emotion may provide larger:

```text
audio prosody headroom
visual affect headroom
cross-modal interaction headroom
```

Audit first.

Do not assume it exists.

---

# 75. Natural Dataset Pre-Gate

Before ConFu++ training on any new dataset run:

\[
P0,
P1,
P2.
\]

Require evidence that:

\[
P2>P1
\]

for at least some modality targets.

This becomes the new higher-order benchmark screening procedure.

---

# 76. Benchmark Screening Pipeline

For each candidate dataset:

```text
1. Train/freeze modality encoders.

2. P0:
   linear additive prediction.

3. P1:
   nonlinear additive prediction.

4. P2:
   capacity-matched joint prediction.

5. Compute J = P2-P1.

6. If J approximately 0:
   not a strong higher-order development benchmark.

7. If J > 0:
   run order-decomposed representation experiment.
```

---

# 77. New Benchmark Definition

A useful higher-order benchmark should have:

\[
\boxed{
\text{non-additive predictive headroom}
}
\]

under controlled predictor capacity.

Not merely:

```text
multimodal data
three modalities
all-modal accuracy > unimodal
```

---

# 78. Important Terminology

Use:

```text
non-additive headroom
joint prediction advantage
order accessibility
explicit interaction embedding
order-specific residual
```

Do NOT use:

```text
formal synergy
PID synergy
new Shannon information
conditional information beyond deterministic inputs
```

without formal proof.

---

# 79. Revised Higher-Order Definition

Operationally, an order-\(k\) embedding is useful when:

\[
\boxed{
\text{adding an explicit order-}k\text{ representation improves accessibility}
}
\]

over all lower-order representations under matched downstream capacity.

---

# 80. Second-Order Metric

\[
\boxed{
\Delta_2
=
Perf(Z_{\le2})
-
Perf(Z_{\le1}).
}
\]

---

# 81. Third-Order Metric

Later:

\[
\boxed{
\Delta_3
=
Perf(Z_{\le3})
-
Perf(Z_{\le2}).
}
\]

---

# 82. Accessibility vs Capacity

Always distinguish:

\[
\text{representation improvement}
\]

from:

\[
\text{downstream model becoming larger}.
\]

Therefore capacity matching is mandatory.

---

# 83. Linear Probe Priority

Primary:

\[
\Delta_k^{linear}.
\]

Why?

Because a useful explicit interaction embedding should ideally expose structure such that a simple classifier can access it.

---

# 84. Nonlinear Probe Role

Secondary:

\[
\Delta_k^{MLP}.
\]

A pattern such as:

\[
\Delta^{linear}<0
\]

but:

\[
\Delta^{MLP}>0
\]

means:

> interaction structure may exist but is not explicitly linearized.

This does not satisfy the primary ConFu++ goal.

---

# 85. Linearization Gap

Always report:

\[
LG_k
=
Perf_{MLP}(Z_{\le k})
-
Perf_{linear}(Z_{\le k}).
\]

Track:

\[
LG_1,
LG_2,
LG_3.
\]

---

# 86. Desired Representation Geometry

Ideal:

\[
\Delta_2^{linear}>0
\]

and:

\[
LG_2\le LG_1.
\]

Later:

\[
\Delta_3^{linear}>0
\]

and:

\[
LG_3\le LG_2.
\]

---

# 87. Current UR-FUNNY Hypothesis

Current leading hypothesis:

\[
\boxed{
\text{E4 residual targets contain substantial unimodal nonlinear residue}.
}
\]

E4.1 must falsify or support this hypothesis.

---

# 88. Do Not Assume E4.1 Will Pass

If P1 closes the gap to P2, accept the negative result.

Do not redesign P1 to artificially preserve headroom.

The audit exists to test whether the dataset actually contains measurable non-additive structure under reasonable hypothesis classes.

---

# 89. Predictor Overcapacity Risk

If P1 is too expressive, independent networks could memorize sample-specific structure.

Therefore:

```text
small architecture
validation early stopping
weight decay
capacity matching
```

are required.

---

# 90. P1 Regularization

Default:

```yaml
predictor:
  hidden_dim: 128
  activation: gelu
  weight_decay: 1e-4
  early_stopping_patience: 5
```

Select using validation reconstruction performance.

---

# 91. Predictor Evaluation

Report:

```text
train R²
validation R²
test R²
```

Large:

\[
train-test
\]

gap indicates predictor overfit.

---

# 92. Representation Standardization

Use identical normalization protocol for:

```text
P0
P1
P2
```

Do not let preprocessing differences confound the capacity audit.

---

# 93. Residual Predictor Objective

Default reconstruction:

\[
MSE.
\]

Optionally compare normalized cosine reconstruction only if MSE behaves pathologically.

Do not combine multiple losses initially.

---

# 94. Fixed Data Splits

All E4.1/E4.2 models use exactly:

```text
official train
official validation
official test
```

No resampling.

No test-based tuning.

---

# 95. Seed Policy

E4.1:

```text
seed 1 exploratory
```

If capacity conclusion is ambiguous:

```text
run seeds 1–3
```

before rebuilding architecture.

Do not automatically run five seeds for predictor diagnostics.

---

# 96. Reproducibility Artifacts

Save:

```text
P0 checkpoints
P1 checkpoints
P2 checkpoints

predictor metrics
parameter counts
residual target metrics
E4.2 checkpoint
probe outputs
configuration
git commit
seed
```

---

# 97. Required E4.1 JSON

Example:

```json
{
  "seed": 1,

  "target_r1": {
    "p0_r2": 0.0,
    "p1_r2": 0.0,
    "p2_r2": 0.0,
    "nonlinear_unimodal_gain": 0.0,
    "joint_advantage": 0.0
  },

  "target_r2": {
    "p0_r2": 0.0,
    "p1_r2": 0.0,
    "p2_r2": 0.0,
    "joint_advantage": 0.0
  },

  "target_r3": {
    "p0_r2": 0.0,
    "p1_r2": 0.0,
    "p2_r2": 0.0,
    "joint_advantage": 0.0
  },

  "p1_params": 0,
  "p2_params": 0
}
```

---

# 98. Required E4.2 JSON

```json
{
  "seed": 1,

  "order1_linear": 0.0,
  "order2_linear": 0.0,
  "delta2_linear": 0.0,

  "order1_nonlinear": 0.0,
  "order2_nonlinear": 0.0,
  "delta2_nonlinear": 0.0,

  "linearization_gap_order1": 0.0,
  "linearization_gap_order2": 0.0,

  "delta12_linear": 0.0,
  "delta13_linear": 0.0,
  "delta23_linear": 0.0,

  "h12_variance": 0.0,
  "h13_variance": 0.0,
  "h23_variance": 0.0,

  "h12_effective_rank": 0.0,
  "h13_effective_rank": 0.0,
  "h23_effective_rank": 0.0
}
```

---

# 99. Experiment Report Template

Every report must contain:

```text
Research question

Hypothesis

Baseline

P0 definition

P1 definition

P2 definition

Parameter matching

Headroom result

Residual health

Interaction health

Linear accessibility

Nonlinear accessibility

Linearization gap

Pair-specific gains

Modality dependence

Single-modality predictability

Decision

Next experiment
```

---

# 100. Forbidden Actions Before E4.1

Do NOT:

```text
implement h123
add task labels to interaction loss
fine-tune original ConFu encoders
increase interaction rank
add cross-attention
add dynamic gates
add residual classifier
sweep temperature broadly
```

---

# 101. Forbidden Actions Before E4.2 Passes

Still do NOT:

```text
Global-Local
temporal routing
context/punchline fusion
Transformer interaction head
hypergraph fusion
third-order module
```

---

# 102. Why Architecture Complexity Is Blocked

The current failure is:

\[
\boxed{
\text{target definition / order isolation}
}
\]

not clearly:

\[
\boxed{
\text{insufficient model capacity}.
}
\]

Increasing architecture capacity before target validation makes the experiment uninterpretable.

---

# 103. Immediate Implementation Task

Implement:

\[
\boxed{
E4.1
}
\]

first.

Required modules:

```text
IndependentMLPPredictor

AdditiveNonlinearPredictor

JointMLPPredictor

CapacityMatcher

PredictorAudit
```

---

# 104. E4.1 Execution Order

```text
1. Load frozen ConFu features.

2. Reproduce P0.

3. Train all six independent unimodal MLP predictors.

4. Build P1 additive predictions.

5. Train capacity-matched P2 joint predictors.

6. Evaluate validation R².

7. Evaluate test R² only after checkpoint selection.

8. Compute P1-P0.

9. Compute P2-P1.

10. Produce directed unimodal nonlinearity matrix.

11. Save report.

12. Make gate decision.
```

---

# 105. E4.1 Gate

Proceed to E4.2 only if:

\[
\boxed{
P2>P1
}
\]

is non-trivial and validation-supported for relevant targets.

---

# 106. E4.2 Execution Order

```text
1. Freeze best P1 predictors.

2. Generate nonlinear additive residual targets.

3. Audit residual health.

4. Reinitialize h12/h13/h23.

5. Keep interaction architecture identical to E4.

6. Train residual alignment.

7. Select checkpoint by validation alignment metric.

8. Extract h representations.

9. Run matched linear accessibility probe.

10. Run matched nonlinear probe.

11. Compute Delta2.

12. Compute pair-specific gains.

13. Compute linearization gaps.

14. Run modality shuffles.

15. Run single-modality predictability.

16. Make gate decision.
```

---

# 107. E4.2 Success

Success requires:

\[
\boxed{
\Delta_2^{linear}>0
}
\]

on validation.

For stronger evidence:

```text
pair-specific positive gains
positive test exploratory Delta2
reduced linearization gap
balanced modality dependence
```

---

# 108. E4.2 Failure

If:

\[
\Delta_2^{linear}\le0,
\]

close the current UR-FUNNY mechanism branch.

Write the negative result.

Move to synthetic.

---

# 109. Synthetic Becomes Mandatory Next

Do not search for another architecture on UR-FUNNY after E4.2 failure.

The next research question becomes:

> Can the proposed order decomposition recover known order-specific structure when ground truth is controlled?

---

# 110. Post-Synthetic Real Dataset

If synthetic mechanism passes, next audit:

\[
\boxed{
MOSEI\ Emotion.
}
\]

Run P0/P1/P2 before ConFu++ training.

---

# 111. MUStARD Role

Use MUStARD later as:

```text
small-data
cross-modal incongruity
stress test
```

Do not use it as the primary model-development benchmark.

---

# 112. AV-MNIST Role

Keep as:

\[
\boxed{
\text{single-modality shortcut control}.
}
\]

---

# 113. MOSI Role

Keep as:

\[
\boxed{
\text{text-dominance / low-accessibility control}.
}
\]

---

# 114. UR-FUNNY Role

UR-FUNNY is now:

\[
\boxed{
\text{real-world non-additivity diagnostic benchmark}.
}
\]

Whether it remains the main model-development benchmark depends on E4.1/E4.2.

---

# 115. Scientific Story So Far

Current progression:

```text
AV-MNIST
fusion can collapse to one modality

MOSI
multimodal dependence does not guarantee accessibility

UR-FUNNY ConFu
pair representations are active but weakly useful

ConFu++ S1
de-redundancy does not create utility

ConFu++ v2
post-hoc residual classification does not create utility

ConFu++ v3 E4
linear-additive residual alignment does not create linear accessibility

ConFu++ v3.1
test whether the residual target itself is contaminated
by unimodal nonlinear structure
```

---

# 116. Core Research Question v3.1

\[
\boxed{
\text{After removing independently learnable nonlinear unimodal effects,}
}
\]

\[
\boxed{
\text{does measurable non-additive cross-modal structure remain?}
}
\]

This question must be answered before further architecture development.

---

# 117. If The Answer Is No

Then UR-FUNNY is not currently a strong benchmark for the desired higher-order claim under these representations.

Accept this.

Do not manufacture interaction.

---

# 118. If The Answer Is Yes

Then ConFu++ should attempt to transform:

\[
\boxed{
\text{non-additive predictive structure}
}
\]

into:

\[
\boxed{
\text{explicit accessible order-specific embeddings}.
}
\]

That is the purpose of E4.2.

---

# 119. Third-Order Trigger

Explicit:

\[
h_{123}
\]

becomes legal only after:

```text
E4.1 passes
E4.2 passes
Delta2 positive
pair dependence verified
```

and preferably synthetic pair-order validation succeeds.

---

# 120. Third-Order Principle

For future third-order work, repeat the same philosophy.

Remove:

\[
\text{order-1}
+
\text{order-2}
\]

predictable components before defining an order-3 residual.

Do not simply multiply all three modalities and call the result third-order information.

---

# 121. No Formal Information-Theoretic Claim

The project currently studies:

\[
\boxed{
\text{functional order decomposition}
}
\]

and:

\[
\boxed{
\text{representation accessibility}.
}
\]

It does not yet estimate exact:

```text
PID synergy
unique information
redundant information
```

in the formal information-theoretic sense.

---

# 122. Preferred Paper Language

Use:

```text
order-specific representation

non-additive headroom

joint prediction advantage

functional interaction

explicit higher-order embedding

representation accessibility

linearization of multimodal interaction
```

---

# 123. Avoid

Avoid claims such as:

```text
creates new information

extracts pure synergy

exactly isolates PID synergy

conditional mutual information of deterministic interaction is positive
```

---

# 124. Potential Contribution If v3.1 Works

A successful method could claim:

> Existing higher-order alignment may encode multimodal relationships without making interaction structure explicitly accessible. We introduce capacity-controlled functional order decomposition that separates independently predictable modality effects from non-additive multimodal structure and learns explicit interaction embeddings from the remaining residual.

---

# 125. Potential Negative Contribution If It Fails

Even failure can support:

> Apparent multimodal joint headroom can disappear after controlling for nonlinear unimodal predictor capacity, showing that linear-vs-joint comparisons overestimate higher-order structure.

This is scientifically valuable.

---

# 126. Main Method Evaluation Ladder

Always evaluate:

\[
P0
\]

then:

\[
P1
\]

then:

\[
P2.
\]

Then:

\[
Z_{\le1}
\]

then:

\[
Z_{\le2}.
\]

Later:

\[
Z_{\le3}.
\]

Never skip order controls.

---

# 127. Core Gate Hierarchy

## Gate H0 — Predictor validity

Independent predictors see only one modality.

## Gate H1 — Capacity matching

P1/P2 parameter budgets matched.

## Gate H2 — Non-additive headroom

\[
P2>P1.
\]

## Gate H3 — Residual health

Nonlinear-additive residual remains non-trivial.

## Gate H4 — Interaction health

\(h_{ij}\) does not collapse.

## Gate H5 — Accessibility

\[
\Delta_2^{linear}>0.
\]

## Gate H6 — Dependence

Both modalities affect each \(h_{ij}\).

## Gate H7 — Stability

Positive effect generalizes across seeds.

---

# 128. Development Rule

If a gate fails:

\[
\boxed{
\text{fix that gate}
}
\]

before moving upward.

Do not bypass failed gates with more architecture.

---

# 129. Final Immediate Priority

The only high-priority experiment now is:

\[
\boxed{
\textbf{E4.1 — capacity-controlled linear-additive vs nonlinear-additive vs joint-nonlinear prediction audit}
}
\]

Everything else is blocked until its result is known.

---

# 130. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ seeks explicit higher-order embeddings only after controlling for nonlinear effects that individual modalities can explain independently.}
}
\]

---

# 131. Final Rule

Do not ask:

> Can a larger multimodal network fit the data better?

Ask:

\[
\boxed{
\textbf{What predictive structure requires joint access to multiple modalities after independently learnable modality effects have been controlled?}
}
\]

Only that remaining structure justifies the next higher-order representation.