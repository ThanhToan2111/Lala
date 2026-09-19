# AGENT.md

# ConFu++ v4

## Synthetic-First, Oracle-Validated Higher-Order Multimodal Representation Learning

---

# 0. Mission

This repository develops a direct research extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The current scientific objective is:

\[
\boxed{
\text{learn explicit order-specific multimodal representations}
}
\]

that expose higher-order structure in a form that is:

- task-accessible;
- order-selective;
- modality-dependent;
- easy for simple downstream models to use;
- distinguishable from generic increases in model capacity.

The project must no longer optimize only for:

```text
higher fusion capacity
lower reconstruction R²
larger interaction modules
higher standalone interaction accuracy
```

The central question is:

> Can a learned representation make pairwise or third-order structure explicitly accessible when that structure is known to exist?

---

# 1. Core Scientific Principle

A deterministic interaction:

\[
h_{12}=F(r_1,r_2)
\]

does not create new Shannon information beyond:

\[
(r_1,r_2).
\]

Therefore the project does NOT claim that higher-order embeddings create information.

Instead:

\[
\boxed{
\text{higher-order embedding}
=
\text{reparameterization of implicit joint structure}
}
\]

into an explicit representation.

The practical objective is:

\[
\boxed{
\text{improved representational accessibility}
}
\]

rather than:

\[
\boxed{
\text{new information creation}.
}
\]

---

# 2. Current Empirical History

The project has already evaluated several increasingly targeted hypotheses.

## AV-MNIST

Finding:

\[
r_{12}\approx f(r_{image})
\]

despite the interaction module receiving image and audio.

Interpretation:

\[
\boxed{
\text{joint computation does not guarantee joint dependence}.
}
\]

Role:

```text
single-modality shortcut control
```

---

# 3. CMU-MOSI

MOSI exhibited:

- text dominance;
- pair representations sensitive to multiple inputs;
- little or negative downstream conditional utility.

Interpretation:

\[
\boxed{
\text{joint dependence does not guarantee useful accessibility}.
}
\]

Role:

```text
text-dominance / weak-utility control
```

---

# 4. UR-FUNNY Original ConFu

Canonical five-seed baseline:

\[
\boxed{
64.462\pm0.713\%
}
\]

The baseline must remain frozen and unchanged.

UR-FUNNY demonstrates some multimodal performance headroom over text alone, but existing pair embeddings do not reliably expose this headroom.

---

# 5. Rejected ConFu++ Branches

## S1 adversarial de-shortcutting

Rejected because:

\[
\text{de-redundancy}
\neq
\text{task complementarity}.
\]

It reduced interaction predictability without creating stable downstream utility.

---

# 6. ConFu++ v2

Frozen residual correction:

\[
\ell_F
=
\ell_L+\Delta\ell_{pairs}
\]

failed.

E1–E3 produced:

\[
NetCorrection<0.
\]

Interpretation:

\[
\boxed{
\text{existing ConFu pair embeddings are not reliably calibrated as residual task corrections}.
}
\]

---

# 7. ConFu++ v3

v3 introduced explicit:

\[
h_{12},h_{13},h_{23}
\]

and aligned them to residual targets generated from linear additive first-order predictors.

Primary linear accessibility remained negative.

However, the v3 residual mapping contained an implementation error and has been superseded.

Do not use v3 E4 as final evidence.

---

# 8. ConFu++ v3.1

v3.1 fixed:

```text
h12 → e3
h13 → e2
h23 → e1
```

and performed a capacity-controlled audit:

\[
P0=\text{linear additive}
\]

\[
P1=\text{nonlinear additive}
\]

\[
P2=\text{joint nonlinear}.
\]

P1 closed most of the apparent P0→P2 gap.

Conclusion:

\[
\boxed{
\text{much of the previous headroom was unimodal nonlinearity}
}
\]

rather than clearly demonstrated higher-order structure.

---

# 9. v3.1 E4.2

Using nonlinear additive residual targets:

\[
e_j
=
r_j-
[
q_i^{MLP}(r_i)+q_k^{MLP}(r_k)
],
\]

the model still failed linear accessibility.

Validation:

\[
\Delta_2^{linear}<0.
\]

Test:

\[
\Delta_2^{linear}<0.
\]

Nonlinear downstream gain remained positive.

Interpretation:

\[
\boxed{
\text{the representation may contain nonlinear structure}
}
\]

but:

\[
\boxed{
\text{the current objective does not make that structure explicitly accessible}.
}
\]

---

# 10. Current Decision

The UR-FUNNY representation-learning branch is now CLOSED temporarily.

Do NOT continue:

```text
rank sweep
cross-attention
dynamic gates
task-loss injection
encoder fine-tuning
larger MLP interaction
r123 on UR-FUNNY
```

until the mechanism is validated on a controlled benchmark.

---

# 11. New Development Philosophy

From now on, separate three questions.

## Q1 — Architecture capability

Can the proposed representation architecture encode known order-specific structure?

## Q2 — Objective capability

Can the learning objective discover that structure without access to ground-truth interaction labels?

## Q3 — Real-world usefulness

Does the validated mechanism improve representation accessibility and downstream performance on natural multimodal data?

These questions must be tested separately.

---

# 12. New Experimental Ladder

The project proceeds through:

```text
Controlled Synthetic
        ↓
Oracle Representation Test
        ↓
Discovery Objective Test
        ↓
Mechanism Validation
        ↓
Real-World Dataset Screening
        ↓
Real-World ConFu++
```

Do NOT skip stages.

---

# 13. Controlled Higher-Order Benchmark

Create a synthetic benchmark with known decomposition:

\[
f(x_1,x_2,x_3)
=
f_1(x_1)
+
f_2(x_2)
+
f_3(x_3)
\]

\[
+
f_{12}(x_1,x_2)
+
f_{13}(x_1,x_3)
+
f_{23}(x_2,x_3)
\]

\[
+
f_{123}(x_1,x_2,x_3).
\]

The dataset must expose explicit control over:

```text
order-1 contribution
order-2 contribution
order-3 contribution
noise
redundancy
interaction strength
```

---

# 14. Latent Variables

Initial clean setting:

\[
z_1,z_2,z_3
\overset{iid}{\sim}
\{-1,+1\}^{d}.
\]

Recommended starting dimension:

\[
d=32
\]

or:

\[
d=64.
\]

Rademacher variables are preferred initially because:

\[
E[z_i]=0
\]

and multiplicative interaction terms have clean moment properties.

---

# 15. Modality Observations

Do NOT expose latent variables directly.

Generate:

\[
x_i=A_i z_i+\epsilon_i
\]

where:

\[
A_i
\]

is a fixed random projection.

Optionally use orthogonal:

\[
A_i.
\]

Noise:

\[
\epsilon_i
\sim
\mathcal N(0,\sigma^2I).
\]

Start with:

\[
\sigma=0.
\]

Noise sweeps come later.

---

# 16. Ground-Truth First-Order Components

Define:

\[
s_1=w_1^\top z_1
\]

\[
s_2=w_2^\top z_2
\]

\[
s_3=w_3^\top z_3.
\]

Normalize so:

\[
Var(s_i)\approx1.
\]

---

# 17. Ground-Truth Pair Components

Define:

\[
s_{12}
=
w_{12}^\top
(z_1\odot z_2)
\]

\[
s_{13}
=
w_{13}^\top
(z_1\odot z_3)
\]

\[
s_{23}
=
w_{23}^\top
(z_2\odot z_3).
\]

Normalize:

\[
Var(s_{ij})\approx1.
\]

---

# 18. Ground-Truth Third-Order Component

Define:

\[
\boxed{
s_{123}
=
w_{123}^\top
(z_1\odot z_2\odot z_3)
}
\]

with:

\[
Var(s_{123})\approx1.
\]

This component is the ground-truth order-3 signal.

---

# 19. Label Generator

Define:

\[
S_1=s_1+s_2+s_3
\]

\[
S_2=s_{12}+s_{13}+s_{23}
\]

\[
S_3=s_{123}.
\]

Generate score:

\[
\eta
=
\alpha S_1
+
\beta S_2
+
\gamma S_3
+
\epsilon_y.
\]

Binary target:

\[
y=\mathbf1[\eta>0].
\]

---

# 20. Component Normalization

Before label generation:

\[
Var(S_1),
Var(S_2),
Var(S_3)
\]

must be controlled.

Do not allow one order to dominate because its raw scale is larger.

Normalize each order contribution where required.

---

# 21. Mandatory Synthetic Regimes

Implement exactly the following initial regimes.

---

# 22. S0 — First-Order Only

Set:

\[
\alpha>0
\]

\[
\beta=0
\]

\[
\gamma=0.
\]

Expected:

\[
\Delta_2\approx0
\]

and:

\[
\Delta_3\approx0.
\]

A higher-order model must NOT invent order utility.

---

# 23. S1 — Single Pair Interaction

Example:

\[
s_{12}\neq0
\]

while:

\[
s_{13}=0
\]

\[
s_{23}=0
\]

\[
s_{123}=0.
\]

Expected:

\[
\Delta_{12}>0
\]

and:

\[
\Delta_{13}\approx0
\]

\[
\Delta_{23}\approx0.
\]

---

# 24. S2 — All Pairwise Interactions

Set:

\[
\beta>0
\]

and:

\[
\gamma=0.
\]

Expected:

\[
\boxed{
\Delta_2>0
}
\]

while:

\[
\Delta_3\approx0.
\]

---

# 25. S3 — Pure Third-Order

Set:

\[
\alpha=0
\]

\[
\beta=0
\]

\[
\gamma>0.
\]

Expected:

\[
\Delta_2\approx0
\]

while:

\[
\boxed{
\Delta_3>0.
}
\]

This is the most important higher-order mechanism test.

---

# 26. S4 — Mixed Order

Set:

\[
\alpha>0
\]

\[
\beta>0
\]

\[
\gamma>0.
\]

Expected:

\[
Perf(Z_{\le1})
<
Perf(Z_{\le2})
<
Perf(Z_{\le3}).
\]

---

# 27. Representation Hierarchy

Define:

\[
Z_{\le1}
=
[r_1,r_2,r_3].
\]

Define:

\[
Z_{\le2}
=
[
r_1,r_2,r_3,
h_{12},h_{13},h_{23}
].
\]

Define:

\[
Z_{\le3}
=
[
Z_{\le2},
h_{123}
].
\]

---

# 28. Second-Order Accessibility

Define:

\[
\boxed{
\Delta_2
=
Perf(Z_{\le2})
-
Perf(Z_{\le1}).
}
\]

This is not information creation.

It measures:

\[
\boxed{
\text{order-2 representational accessibility}.
}
\]

---

# 29. Third-Order Accessibility

Define:

\[
\boxed{
\Delta_3
=
Perf(Z_{\le3})
-
Perf(Z_{\le2}).
}
\]

This is the primary explicit order-3 metric.

---

# 30. Pair-Specific Accessibility

Define:

\[
\Delta_{12}
=
Perf([Z_{\le1},h_{12}])
-
Perf(Z_{\le1}).
\]

Similarly:

\[
\Delta_{13}
\]

and:

\[
\Delta_{23}.
\]

---

# 31. Pair Selectivity

For S1 where only pair \(12\) is active:

\[
\boxed{
Selectivity_{12}
=
\Delta_{12}
-
\frac{
\Delta_{13}+\Delta_{23}
}{2}
}
\]

This is a diagnostic.

Do not call it information-theoretic synergy.

---

# 32. False Positive Order Test

An order-specific method must satisfy:

```text
no pair gain when pair interaction is absent
no triple gain when triple interaction is absent
```

False-positive order activation is a failure.

---

# 33. Linear Probe Priority

Primary evaluation:

\[
Perf_{linear}.
\]

Why?

Because explicit interaction embeddings are intended to expose nonlinear structure to simple downstream models.

---

# 34. Nonlinear Probe

Secondary evaluation:

\[
Perf_{MLP}.
\]

The nonlinear probe measures remaining inaccessible structure.

---

# 35. Linearization Gap

Define:

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

Desired:

\[
LG_3<LG_2<LG_1
\]

when higher-order structure exists.

---

# 36. Strong Linearization Result

A particularly strong result is:

\[
Perf_{linear}(Z_{\le3})
\approx
Perf_{MLP}(Z_{\le1}).
\]

This demonstrates that explicit interaction embeddings reduce downstream nonlinear computation requirements.

---

# 37. Generic MLP Baseline

Always train:

\[
MLP([r_1,r_2,r_3]).
\]

Purpose:

> verify that the task itself is learnable through generic nonlinear capacity.

If generic MLP fails on S3, the synthetic benchmark or optimization may be broken.

---

# 38. Oracle Stage

Before testing a discovery objective, directly supervise interaction embeddings using ground-truth interaction components.

This is mandatory.

---

# 39. Oracle Pair Target

Ground truth:

\[
g_{12}
=
z_1\odot z_2.
\]

Train:

\[
h_{12}
\rightarrow g_{12}.
\]

Similarly:

\[
h_{13}
\rightarrow g_{13}
\]

\[
h_{23}
\rightarrow g_{23}.
\]

---

# 40. Oracle Third-Order Target

Ground truth:

\[
\boxed{
g_{123}
=
z_1\odot z_2\odot z_3.
}
\]

Train:

\[
h_{123}
\rightarrow g_{123}.
\]

---

# 41. Oracle Purpose

The Oracle experiment answers:

> Is the architecture capable of representing the correct order if given the correct target?

This separates:

\[
\boxed{
\text{architecture failure}
}
\]

from:

\[
\boxed{
\text{objective failure}.
}
\]

---

# 42. Oracle Failure Rule

If Oracle pair embedding fails:

\[
\Delta_2\le0
\]

on pair regimes,

do NOT design a new discovery loss.

Fix:

```text
interaction architecture
optimization
representation dimension
synthetic generator
```

first.

---

# 43. Oracle Third-Order Failure Rule

If:

\[
h_{123}
\]

cannot generate:

\[
\Delta_3>0
\]

on S3:

```text
STOP third-order objective research.
```

The architecture itself is not validated.

---

# 44. Initial Pair Architecture

Use:

\[
u_1=LN(U_1r_1)
\]

\[
u_2=LN(U_2r_2)
\]

and:

\[
\boxed{
h_{12}
=
W_{12}
\left(
\frac{
u_1\odot u_2
}{
\sqrt R
}
\right).
}
\]

Start:

\[
R=64.
\]

---

# 45. Initial Third-Order Architecture

Use:

\[
v_1=LN(V_1r_1)
\]

\[
v_2=LN(V_2r_2)
\]

\[
v_3=LN(V_3r_3).
\]

Then:

\[
\boxed{
h_{123}
=
W_{123}
\left(
\frac{
v_1\odot v_2\odot v_3
}{
\sqrt R
}
\right).
}
\]

Do not start with attention.

---

# 46. No Rank Sweep Initially

Use:

\[
R=64.
\]

Only sweep rank if Oracle itself fails and diagnostics suggest representational bottleneck.

---

# 47. Oracle Loss

Possible starting loss:

\[
L_{oracle}
=
MSE(h_S,P(g_S)).
\]

Or normalized cosine/MSE.

Keep the initial experiment simple.

---

# 48. Oracle Probe Evaluation

Do NOT report only reconstruction loss.

Always evaluate:

\[
\Delta_2
\]

and:

\[
\Delta_3.
\]

A representation is useful only if it improves accessibility.

---

# 49. Oracle Pass Criteria

S0:

\[
\Delta_2\approx0,
\quad
\Delta_3\approx0.
\]

S1:

\[
\Delta_{correct-pair}>0.
\]

S2:

\[
\Delta_2>0.
\]

S3:

\[
\boxed{
\Delta_3>0
}
\]

with:

\[
\Delta_2\approx0.
\]

S4:

\[
Perf_1<Perf_2<Perf_3.
\]

---

# 50. Only After Oracle Pass

Once architecture is validated, proceed to discovery objectives.

Do NOT mix Oracle supervision into final real-world claims.

Oracle is mechanism validation only.

---

# 51. Discovery Objective Baseline A

Original ConFu-style alignment:

\[
h_{12}\leftrightarrow r_3.
\]

Similarly:

\[
h_{13}\leftrightarrow r_2
\]

\[
h_{23}\leftrightarrow r_1.
\]

This provides the original higher-order alignment reference.

---

# 52. Discovery Objective Baseline B

Residual target:

\[
e_3
=
r_3-q_{add}(r_1,r_2).
\]

This corresponds to the v3/v3.1 residual approach.

Keep only as a comparison.

Do not assume it is the final method.

---

# 53. Main New Discovery Hypothesis

The residual:

\[
r_3-q_{add}
\]

contains:

```text
noise
unpredictable target variation
non-additive predictable structure
```

Therefore it is not an ideal interaction target.

Instead identify:

\[
\boxed{
\text{what a joint model predicts beyond an additive model}.
}
\]

---

# 54. Additive Predictor

Define:

\[
q_{add}(r_1,r_2)
=
q_1(r_1)+q_2(r_2).
\]

Use independent nonlinear predictors.

---

# 55. Joint Predictor

Define:

\[
q_{joint}(r_1,r_2)
=
q([r_1,r_2]).
\]

Capacity-match:

\[
q_{add}
\]

and:

\[
q_{joint}.
\]

---

# 56. Joint Advantage Target

Define:

\[
\boxed{
d_{12\rightarrow3}
=
q_{joint}(r_1,r_2)
-
q_{add}(r_1,r_2).
}
\]

Similarly:

\[
d_{13\rightarrow2}
\]

and:

\[
d_{23\rightarrow1}.
\]

---

# 57. Interpretation

The quantity:

\[
d_{12\rightarrow3}
\]

represents:

> target-predictive structure available to a joint predictor but not to the matched additive predictor.

Preferred terminology:

\[
\boxed{
\text{Joint Predictive Advantage}
}
\]

or:

\[
\boxed{
\text{Non-Additive Predictive Residual}.
}
\]

---

# 58. No Formal Synergy Claim

Do NOT call:

\[
d_{12\rightarrow3}
\]

exact:

```text
PID synergy
conditional mutual information
pure higher-order information
```

It is hypothesis-class dependent.

---

# 59. Joint Advantage Distillation

Train:

\[
\boxed{
h_{12}
\leftrightarrow
stopgrad(d_{12\rightarrow3})
}
\]

and similarly for other pairs.

This becomes the leading ConFu++ discovery hypothesis.

---

# 60. Why Joint Advantage Is Cleaner Than Residual Target

Old:

\[
r_3-q_{add}.
\]

New:

\[
q_{joint}-q_{add}.
\]

The new target removes target variation that neither predictor can explain.

Conceptually:

```text
target
  │
  ├── predictable additively
  │
  ├── predictable only jointly
  │
  └── unpredictable / noise
```

ConFu++ should focus on:

```text
predictable only jointly
```

rather than:

```text
everything additive model misses
```

---

# 61. Discovery Objective Comparison

On synthetic benchmark compare:

```text
Original ConFu alignment
Residual alignment
Joint-Advantage Distillation
Oracle
```

Oracle is upper-bound-style mechanism reference.

---

# 62. Recovery Ratio

Define:

\[
\boxed{
Recovery_2
=
\frac{
\Delta_2^{discovery}
}{
\Delta_2^{oracle}+\epsilon
}
}
\]

and:

\[
Recovery_3
=
\frac{
\Delta_3^{discovery}
}{
\Delta_3^{oracle}+\epsilon
}.
\]

These quantify how much of the accessible Oracle interaction the unsupervised/self-supervised objective recovers.

Use carefully when Oracle gains are near zero.

---

# 63. Order Selectivity Under Discovery

For S1:

\[
\Delta_{12}>0
\]

and:

\[
\Delta_{13},
\Delta_{23}\approx0.
\]

If all pair embeddings improve equally:

\[
\boxed{
\text{order selectivity fails}.
}
\]

---

# 64. Triple Discovery Objective

Do NOT design a final third-order discovery objective until pairwise Joint-Advantage Distillation works.

Order-3 remains blocked until pair discovery succeeds.

---

# 65. Synthetic Noise Sweep

Only after clean S0–S4 pass.

Use:

\[
\sigma
\in
\{0,0.1,0.25,0.5,1.0\}.
\]

Measure degradation of:

\[
\Delta_2
\]

\[
\Delta_3
\]

and recovery ratios.

---

# 66. Interaction-Strength Sweep

Sweep:

\[
\beta
\]

for order-2.

Example:

\[
0,
0.25,
0.5,
1,
2.
\]

Desired:

\[
\Delta_2
\]

should generally increase with pair interaction strength.

---

# 67. Triple-Strength Sweep

Sweep:

\[
\gamma.
\]

Desired:

\[
\Delta_3
\]

should increase with triple interaction strength.

---

# 68. Strong Paper Plot

Plot:

\[
interaction\ strength
\]

against:

\[
\Delta_2
\]

or:

\[
\Delta_3.
\]

Curves:

```text
ConFu
Residual Alignment
Joint-Advantage Distillation
Oracle
```

This is a high-value mechanism figure.

---

# 69. Required Synthetic Baselines

Include:

```text
Unimodal linear
Unimodal MLP
Concatenation linear
Concatenation MLP
Additive fusion
Original ConFu
Low-rank multiplicative
Oracle order representation
ConFu++ discovery
```

---

# 70. Capacity Matching

Downstream comparisons must control model capacity.

Do not allow:

\[
Z_{\le3}
\]

probe to have a much larger trainable classifier than:

\[
Z_{\le1}.
\]

Use standardized linear probes and parameter-matched MLP probes.

---

# 71. Health Metrics

For each:

\[
h_S,
\]

report:

```text
variance
effective rank
mean norm
dimension standard deviation
```

These detect collapse only.

They are not order-utility metrics.

---

# 72. Dependence Metrics

For:

\[
h_{12},
\]

shuffle each modality independently.

Require both modalities to influence the representation or its downstream utility.

---

# 73. Triple Dependence

For:

\[
h_{123},
\]

independently shuffle:

\[
r_1
\]

\[
r_2
\]

\[
r_3.
\]

All three should matter in S3/S4.

---

# 74. False Triple Dependence

In S2, where:

\[
\gamma=0,
\]

the model should not produce meaningful:

\[
\Delta_3.
\]

This is a key negative control.

---

# 75. Five Seeds

Exploratory:

```text
seed 1
```

for implementation/debugging.

Mechanism-confirmatory synthetic results:

```text
seeds 1–5
```

once configurations are fixed.

---

# 76. Synthetic Dataset Size

Recommended initial:

```text
train: 20,000
validation: 5,000
test: 5,000
```

Large enough to reduce small-data variance.

May increase if necessary.

---

# 77. Label Balance

For binary classification, ensure approximately:

\[
50/50.
\]

Report actual distribution.

Do not let accuracy gains come from imbalance.

---

# 78. Synthetic Generator Tests

Unit-test:

```text
component variance
component independence/correlation
label balance
correct enabled interaction order
seed reproducibility
absence of disabled components
```

---

# 79. Oracle Unit Tests

Test:

```text
correct pair target
correct triple target
no swapped mappings
correct normalization
correct frozen latent target
```

Mapping bugs must be impossible by construction.

---

# 80. Suggested Repository Structure

Create:

```text
src/datasets/synthetic_order.py
```

and:

```text
src/experiments/synthetic_order/
```

with:

```text
dataset.py
components.py
models.py
oracle.py
predictors.py
discovery.py
probes.py
metrics.py
runner.py
```

---

# 81. Dataset Configuration

Example:

```yaml
synthetic_order:
  latent_dim: 32
  observation_dim: 64

  train_size: 20000
  val_size: 5000
  test_size: 5000

  noise_std: 0.0

  alpha_order1: 1.0
  beta_order2: 0.0
  gamma_order3: 0.0

  active_pairs:
    - "12"
    - "13"
    - "23"
```

---

# 82. Regime Configuration

Provide named configs:

```text
s0_first_order
s1_pair12
s2_all_pairs
s3_triple
s4_mixed
```

Do not edit coefficients manually for official runs.

---

# 83. Oracle Experiment IDs

Use:

```text
O0_first_order
O1_pair12
O2_pairs
O3_triple
O4_mixed
```

---

# 84. Discovery Experiment IDs

Use:

```text
D0_confu
D1_residual
D2_joint_advantage
```

for each regime.

---

# 85. Primary Synthetic Result Table

Required:

| Regime | Method | Linear Z1 | Linear Z2 | Linear Z3 | Δ2 | Δ3 |
|---|---|---:|---:|---:|---:|---:|
| S0 | | | | | | |
| S1 | | | | | | |
| S2 | | | | | | |
| S3 | | | | | | |
| S4 | | | | | | |

---

# 86. Secondary Result Table

| Regime | Method | MLP Z1 | MLP Z2 | MLP Z3 | LG1 | LG2 | LG3 |
|---|---|---:|---:|---:|---:|---:|---:|

---

# 87. Selectivity Table

| Regime | Δ12 | Δ13 | Δ23 | Correct-pair selectivity |
|---|---:|---:|---:|---:|

---

# 88. Order Recovery Table

| Method | Recovery₂ | Recovery₃ |
|---|---:|---:|
| ConFu | | |
| Residual | | |
| Joint Advantage | | |

---

# 89. Synthetic Pass Gate

The architecture passes only if Oracle produces the expected order pattern.

The discovery objective passes only if it reproduces a meaningful fraction of Oracle order gains without generating false-positive order gains.

---

# 90. Synthetic Failure Decision

If Oracle fails:

\[
\boxed{
\text{architecture problem}.
}
\]

If Oracle passes but all discovery objectives fail:

\[
\boxed{
\text{objective problem}.
}
\]

If synthetic discovery passes but real-world fails:

\[
\boxed{
\text{domain/data problem}.
}
\]

This separation is mandatory.

---

# 91. Real-World Return Gate

Do NOT return to heavy real-world ConFu++ development until:

```text
Oracle pair mechanism passes
Oracle triple mechanism passes
at least one discovery objective passes pair regime
false-positive order controls pass
```

---

# 92. First Real-World Candidate

After synthetic validation:

\[
\boxed{
CMU\text{-}MOSEI\ Emotion
}
\]

is the preferred next benchmark.

Do not start with sentiment.

---

# 93. MOSEI Screening

Before training ConFu++, perform:

\[
P0=\text{linear additive}
\]

\[
P1=\text{nonlinear additive}
\]

\[
P2=\text{joint nonlinear}.
\]

Capacity-match P1/P2.

---

# 94. Real-World Joint Advantage

Define:

\[
J
=
Metric(P2)-Metric(P1).
\]

Use representation reconstruction and task metrics separately.

Do not conflate them.

---

# 95. MOSEI Task Screening

Evaluate each emotion separately if possible:

```text
happiness
sadness
anger
surprise
disgust
fear
```

Identify tasks with stable non-additive headroom.

---

# 96. Benchmark Selection Rule

Only use a real task as a main ConFu++ development benchmark if:

\[
P2>P1
\]

is stable enough to justify a joint interaction mechanism.

---

# 97. UR-FUNNY Future Role

UR-FUNNY remains:

```text
negative real-world mechanism result
text-dominant humor benchmark
later generalization benchmark
```

Do not discard the existing results.

They form part of the research story.

---

# 98. MOSI Future Role

MOSI remains:

```text
text-dominance control
multimodal-dependence-with-low-utility example
```

---

# 99. AV-MNIST Future Role

AV-MNIST remains:

```text
single-modality shortcut negative control
```

---

# 100. MUStARD Future Role

MUStARD may be used later as:

```text
cross-modal incongruity stress test
small-data generalization benchmark
```

Do not tune large architectures on it initially.

---

# 101. Real-World Accuracy Goal

Once mechanism is validated, downstream accuracy becomes an explicit secondary goal.

For each benchmark compare:

```text
Original ConFu
ConFu++
task-specific multimodal baseline
```

---

# 102. Preserve ConFu Representation Goals

Do not improve classification by abandoning representation learning entirely.

ConFu++ should still support:

```text
cross-modal retrieval
pair-to-modality alignment
higher-order representation analysis
```

where applicable.

---

# 103. Classification Is Not Enough

A large supervised classifier can improve accuracy without learning explicit higher-order embeddings.

Therefore every real-world result must include:

```text
task performance
order accessibility
interaction dependence
linear/nonlinear probe comparison
```

---

# 104. Retrieval Evaluation

Where Original ConFu reports:

```text
1→1 retrieval
2→1 retrieval
```

retain comparable evaluation.

ConFu++ should not destroy the alignment property that motivated ConFu.

---

# 105. Main Future Objective

If synthetic tests support it, the preferred ConFu++ method becomes:

\[
\boxed{
\text{Joint-Advantage Distillation}
}
\]

where explicit interaction branches learn:

\[
q_{joint}-q_{add}.
\]

---

# 106. Proposed Pair Loss

For pair:

\[
(i,j)\rightarrow k,
\]

define:

\[
d_{ij\rightarrow k}
=
stopgrad(
q_{joint}(r_i,r_j)
-
q_i(r_i)
-
q_j(r_j)
).
\]

Train:

\[
h_{ij}
\]

toward:

\[
d_{ij\rightarrow k}.
\]

---

# 107. Predictor Training Order

1. Train independent predictors.
2. Freeze them.
3. Train capacity-matched joint predictor.
4. Freeze it.
5. Construct joint-advantage targets.
6. Train interaction representations.

Do not co-adapt predictors and interaction branch initially.

---

# 108. Prevent Target Gaming

All target-generating predictors must be frozen before interaction training.

Otherwise the training process could increase target magnitude artificially.

---

# 109. Joint Advantage Health

Report:

```text
target variance
target effective rank
target norm
joint-vs-additive prediction advantage
```

before training interaction representations.

---

# 110. Target Collapse Gate

If:

\[
Var(d_{ij\rightarrow k})\approx0,
\]

there is little measurable joint advantage under the chosen predictor classes.

Do not force an interaction embedding.

---

# 111. Task-Relevance Caveat

Joint predictive advantage of one representation target does not automatically imply task relevance.

Therefore downstream:

\[
\Delta_2
\]

remains mandatory.

---

# 112. Task-Aware Extension

Do NOT add task supervision to Joint-Advantage Distillation until the unsupervised/representation objective is characterized.

Task-aware variants are later ablations.

---

# 113. Third-Order Discovery

Blocked until pair discovery works.

Future idea:

remove all independently predictable order-1 and validated order-2 components before constructing an order-3 target.

Do not implement prematurely.

---

# 114. Explicit h123 Trigger

Require:

```text
Oracle h123 pass
pair discovery pass
positive Δ2
false pair-order controls pass
```

before real discovery of:

\[
h_{123}.
\]

---

# 115. Third-Order Acceptance

Require:

\[
\Delta_3>0
\]

on S3.

Also require positive dependence on:

\[
r_1,r_2,r_3.
\]

And no false:

\[
\Delta_3
\]

on S0/S2.

---

# 116. No Arbitrary R² Gate

Do NOT require:

\[
R^2<0.5
\]

for interaction representations.

Use predictability only as a shortcut diagnostic.

---

# 117. Single-Modality Shortcut Diagnostic

For:

\[
h_{12},
\]

measure:

\[
R^2(r_1\rightarrow h_{12})
\]

and:

\[
R^2(r_2\rightarrow h_{12}).
\]

Strong imbalance may indicate shortcutting.

---

# 118. Correct Joint Predictability Interpretation

High:

\[
R^2([r_1,r_2]\rightarrow h_{12})
\]

is not inherently bad because:

\[
h_{12}
\]

is a deterministic function of those inputs.

Do not use it as a redundancy failure criterion.

---

# 119. Statistical Protocol

Exploratory:

```text
1 seed
```

Mechanism confirmation:

```text
5 seeds
```

Report:

```text
mean
sample std
paired deltas
confidence interval
effect size
```

where meaningful.

---

# 120. Hyperparameter Rule

Select:

```text
rank
temperature
learning rate
loss weights
hidden size
```

using validation only.

Do not tune on test.

---

# 121. Negative Results

Preserve all negative results.

Required research log:

```text
hypothesis
experiment
result
why it failed
decision
```

Negative results are part of the scientific contribution.

---

# 122. Reproducibility

Every run must save:

```text
seed
git commit
config
dataset regime
parameter counts
checkpoint
validation metrics
test metrics
order metrics
health metrics
```

---

# 123. Unit Tests

Synthetic benchmark unit tests must verify:

```text
seed determinism
correct regime construction
disabled terms are exactly zero
pair mappings correct
triple mapping correct
component normalization
label balance
no split leakage
```

---

# 124. No Mapping-by-Index Fragility

Represent modality/pair targets using explicit names:

```text
"12_to_3"
"13_to_2"
"23_to_1"
```

not fragile integer ordering.

This prevents repetition of the earlier mapping error.

---

# 125. Immediate Coding Task

Implement only:

```text
Controlled S0–S4 generator
Oracle pair interaction
Oracle triple interaction
matched probes
Δ2 / Δ3 metrics
```

Do NOT implement Joint-Advantage Distillation yet.

---

# 126. Immediate Execution Sequence

Run exactly:

```text
1. Build synthetic generator.

2. Unit-test S0–S4.

3. Run raw-feature linear baseline.

4. Run raw-feature MLP baseline.

5. Train simple modality encoders if required.

6. Run Oracle h12/h13/h23.

7. Measure pair-specific Δ.

8. Run Oracle h123.

9. Measure Δ3.

10. Verify false-positive order controls.

11. Freeze architecture.

12. Only then implement discovery objectives.
```

---

# 127. First Decision Gate

If Oracle does not produce correct:

\[
\Delta_2
\]

on pair tasks:

```text
STOP.
```

Fix architecture.

---

# 128. Second Decision Gate

If Oracle pair passes but:

\[
\Delta_3\le0
\]

on S3:

```text
STOP third-order development.
```

Fix:

\[
h_{123}.
\]

---

# 129. Third Decision Gate

If Oracle pair/triple pass:

\[
\boxed{
\text{architecture validated}.
}
\]

Then implement discovery.

---

# 130. Discovery Order

Test:

```text
D0 Original ConFu Alignment

D1 Target Residual Alignment

D2 Joint-Advantage Distillation
```

Use exactly the same architecture.

Only objective changes.

---

# 131. Objective Comparison Rule

Do not modify architecture between D0/D1/D2.

Otherwise objective comparison is confounded.

---

# 132. Main Discovery Metric

Compare:

\[
\Delta_2^{D0}
\]

\[
\Delta_2^{D1}
\]

\[
\Delta_2^{D2}
\]

against:

\[
\Delta_2^{Oracle}.
\]

---

# 133. Strong D2 Result

Ideal:

\[
\Delta_2^{D2}
>
\Delta_2^{D0}
\]

and:

\[
\Delta_2^{D2}
>
\Delta_2^{D1}.
\]

Also:

\[
Recovery_2^{D2}
\]

should approach the Oracle performance.

---

# 134. Third-Order Discovery Later

Only after D2 pair success.

Do not invent order-3 discovery target before order-2 discovery is validated.

---

# 135. Real-World Return

After synthetic discovery passes:

```text
MOSEI Emotion screening
↓
select promising emotion/task
↓
Original ConFu
↓
ConFu++ D2
↓
compare accuracy + retrieval + accessibility
```

---

# 136. Accuracy Improvement Target

On real-world tasks, ConFu++ should ideally improve:

\[
\text{classification accuracy}
\]

without sacrificing:

\[
\text{cross-modal alignment}.
\]

But accuracy is secondary to proving the higher-order mechanism.

---

# 137. Capacity-Controlled Accuracy Baseline

Every ConFu++ improvement must be compared against a generic parameter-matched fusion model.

This controls for:

\[
\text{extra capacity}.
\]

---

# 138. No Architecture Escalation Rule

Do not add:

```text
cross-attention
Transformer fusion
Global-Local
hypergraph
large tensor fusion
```

unless:

1. Oracle proves interaction architecture is the bottleneck; or
2. simple validated mechanism reaches a clear capacity ceiling.

---

# 139. Research Thesis

Current thesis:

> Higher-order alignment does not guarantee that multimodal interaction structure becomes explicitly accessible. ConFu++ studies order-specific representations by separating architecture capability, target discovery, and real-world utility.

---

# 140. Stronger Thesis if D2 Works

> ConFu++ distills the predictive advantage of joint multimodal models over capacity-matched additive models into explicit order-specific embeddings.

This is the preferred future method story.

---

# 141. Potential Paper Title

Preferred:

**Beyond Higher-Order Alignment: Learning Explicit Order-Specific Multimodal Representations**

Alternative:

**Joint Advantage Distillation for Higher-Order Multimodal Representation Learning**

Alternative:

**Making Multimodal Interaction Explicit**

Alternative:

**Fusion Is Not Interaction: Learning Order-Specific Multimodal Embeddings**

---

# 142. Primary Contribution Candidate

If successful:

1. identify shortcomings of higher-order alignment as an interaction representation objective;

2. introduce controlled order-specific evaluation;

3. separate architecture capability from objective capability using Oracle interaction targets;

4. introduce Joint-Advantage Distillation;

5. demonstrate explicit pair and third-order accessibility;

6. validate on synthetic and real multimodal datasets.

---

# 143. Limitation Awareness

Even Joint-Advantage Distillation is hypothesis-class dependent.

The distinction:

\[
q_{joint}-q_{add}
\]

depends on:

```text
predictor capacity
optimization
representation quality
regularization
```

Therefore do not interpret it as exact information decomposition.

---

# 144. Synthetic-to-Real Gap

Success on controlled data does not guarantee success on:

```text
UR-FUNNY
MOSI
MOSEI
MUStARD
```

Synthetic validates mechanism.

Real data validates usefulness.

Both are required.

---

# 145. Final Development Priority

Current priority order:

\[
\boxed{
\text{Synthetic generator}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Oracle pair representation}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Oracle third-order representation}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Joint-Advantage discovery}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{MOSEI Emotion}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{real-world ConFu++ validation}.
}
\]

---

# 146. Immediate Forbidden Work

Until Oracle completes, do NOT:

```text
return to UR-FUNNY tuning
run h123 on UR-FUNNY
run large architecture sweeps
add task-aware fusion
run cross-attention
change ConFu encoders
```

---

# 147. Immediate Success Definition

The next milestone is NOT:

\[
accuracy>64.462\%.
\]

The next milestone is:

\[
\boxed{
\text{Oracle representations recover the correct known interaction order}.
}
\]

Specifically:

\[
S2:
\Delta_2>0
\]

and:

\[
S3:
\Delta_2\approx0,
\quad
\Delta_3>0.
\]

---

# 148. Final Success Definition

The full project succeeds when an unsupervised/self-supervised ConFu++ objective approaches Oracle order accessibility and transfers that advantage to natural multimodal tasks.

---

# 149. One-Sentence Research Question

\[
\boxed{
\textbf{Can we learn explicit multimodal representations that expose only the predictive structure requiring genuine joint access to multiple modalities?}
}
\]

---

# 150. Final Rule

When an experiment fails, identify whether the failure belongs to:

\[
\boxed{
\text{data}
}
\]

\[
\boxed{
\text{architecture}
}
\]

\[
\boxed{
\text{objective}
}
\]

or:

\[
\boxed{
\text{evaluation}.
}
\]

Never respond to failure by increasing model complexity before identifying which layer of the problem failed.

The next required experiment is:

\[
\boxed{
\textbf{Controlled Synthetic S0–S4 + Oracle Order Recovery}.
}
\]