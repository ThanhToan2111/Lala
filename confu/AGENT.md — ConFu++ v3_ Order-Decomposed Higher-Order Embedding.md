# AGENT.md

# ConFu++ v3

## Order-Decomposed Higher-Order Multimodal Representation Learning

---

# 0. Mission

This repository develops a direct research extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The new research objective is:

\[
\boxed{
\text{learn explicit order-specific multimodal representations}
}
\]

that make higher-order interaction structure:

\[
\boxed{
\text{accessible, measurable, and useful}
}
\]

to downstream models.

The project must NOT optimize for:

```text
low redundancy for its own sake
low R² for its own sake
different-looking interaction embeddings
larger fusion architectures
```

The primary objective is:

\[
\boxed{
\text{higher-order accessibility}
}
\]

---

# 1. Core Scientific Correction

Previous versions implicitly treated:

\[
I(Y;r_{12}\mid r_1,r_2)
\]

as a desirable notion of interaction utility.

This is no longer valid as the central formulation.

If:

\[
r_{12}=F(r_1,r_2)
\]

is deterministic, then:

\[
H(r_{12}\mid r_1,r_2)=0.
\]

Therefore:

\[
I(Y;r_{12}\mid r_1,r_2)=0.
\]

This does NOT mean that:

\[
r_{12}
\]

is useless.

It means that a deterministic interaction representation cannot create new Shannon information beyond its inputs.

Therefore the correct question is not:

> Does \(r_{12}\) contain new information outside \(r_1,r_2\)?

The correct question is:

> Does \(r_{12}\) make higher-order structure already implicit in \(r_1,r_2\) easier for downstream models to access?

---

# 2. New Thesis

The project thesis is now:

\[
\boxed{
\text{higher-order embedding}
=
\text{explicit reparameterization of implicit joint structure}
}
\]

rather than:

\[
\boxed{
\text{creation of new information}.
}
\]

The desired representation should:

1. require multiple modalities;
2. encode interaction structure;
3. improve task accessibility;
4. add predictive value at a controlled model complexity;
5. generalize across seeds and datasets.

---

# 3. Current Empirical State

The project has completed experiments on:

```text
AV-MNIST
CMU-MOSI
UR-FUNNY
```

These reveal three different failure patterns.

---

# 4. AV-MNIST Finding

AV-MNIST showed:

\[
r_{12}\approx f(r_{image}).
\]

Single-modality predictability:

\[
R^2(r_{image}\rightarrow r_{12})
\]

was extremely high.

Audio predictability was extremely low.

Interpretation:

\[
\boxed{
\text{joint computation without genuine joint dependence}
}
\]

AV-MNIST remains a:

```text
single-modality shortcut control
```

---

# 5. MOSI Finding

MOSI showed that some pair representations react to both modalities.

However:

```text
conditional linear utility ≈ negative
multimodal representation dominated by text
```

Interpretation:

\[
\boxed{
\text{joint dependence without stable task accessibility}
}
\]

MOSI remains a:

```text
text-dominance / redundancy control
```

---

# 6. UR-FUNNY Finding

Original ConFu:

\[
64.462\pm0.713\%.
\]

ConFu++ S1 + task:

\[
63.549\pm0.855\%.
\]

ConFu++ v2 E0:

### Linear

\[
\Delta_{pairs}^{linear}
=
-0.017\pm1.179\text{ pp}.
\]

### Nonlinear

\[
\Delta_{pairs}^{MLP}
=
+1.191\pm2.579\text{ pp}.
\]

The nonlinear signal is positive in expectation but unstable across seeds.

---

# 7. UR-FUNNY v2 Failure

Frozen residual correction:

```text
E1:
-1.229 pp vs lower

E2:
-0.851 pp

E3:
-0.756 pp
```

All stages produced:

\[
NetCorrection<0.
\]

Therefore:

\[
\boxed{
\text{post-hoc consumption of existing pair embeddings is insufficient}.
}
\]

The bottleneck now lies at:

\[
\boxed{
\text{representation formation}
}
\]

rather than:

\[
\boxed{
\text{classifier calibration alone}.
}
\]

---

# 8. What Is Rejected

Do NOT continue:

```text
S1 adversarial de-shortcutting
residual classifier sweeps
dynamic gates on frozen pair embeddings
rank sweeps
cross-attention
large MLP fusion
additional preserve-loss sweeps
```

The previous experiments already provide sufficient evidence that these directions do not address the current bottleneck.

---

# 9. Important Metric Correction

The following quantity:

\[
R^2([r_i,r_j]\rightarrow r_{ij})
\]

must NOT be used as a hard redundancy failure criterion.

Since:

\[
r_{ij}=F(r_i,r_j),
\]

joint predictability is expected.

What remains useful is:

\[
R^2(r_i\rightarrow r_{ij})
\]

and:

\[
R^2(r_j\rightarrow r_{ij}).
\]

These measure whether the interaction collapses toward a single modality.

---

# 10. New Representation Hierarchy

ConFu++ v3 explicitly decomposes representation order.

Define:

\[
\mathcal R^{(1)}
=
\{r_1,r_2,r_3\}
\]

as first-order modality representations.

Define:

\[
\mathcal R^{(2)}
=
\{h_{12},h_{13},h_{23}\}
\]

as explicit second-order interaction representations.

Later define:

\[
\mathcal R^{(3)}
=
\{h_{123}\}.
\]

---

# 11. Do Not Conflate Fused and Interaction Representations

Previous ConFu pair representation:

\[
r_{12}
\]

mixes:

```text
shared semantic information
first-order contributions
interaction information
```

ConFu++ v3 separates these concepts.

Define:

\[
z_{12}
=
s_{12}
+
g_{12}h_{12}.
\]

Where:

\[
s_{12}
\]

contains lower-order/shared content.

And:

\[
h_{12}
\]

is the explicit second-order interaction representation.

---

# 12. First-Order Pair Component

For modalities:

\[
r_1,r_2,
\]

define:

\[
s_{12}
=
P_1(r_1)
+
P_2(r_2).
\]

This branch is intentionally additive.

It represents information that can be expressed through first-order modality contributions.

---

# 13. Second-Order Interaction Component

Project:

\[
u_1
=
LN(U_1r_1)
\]

\[
u_2
=
LN(U_2r_2).
\]

Define:

\[
\boxed{
h_{12}
=
W_{12}
\left(
u_1\odot u_2
\right)
}
\]

or normalized form:

\[
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
\]

Similarly:

\[
h_{13},
h_{23}.
\]

---

# 14. Initial Interaction Operator

Use low-rank multiplicative interaction first.

Do NOT start with:

```text
cross-attention
tensor fusion
large bilinear maps
hypergraphs
```

The current scientific question is about interaction order, not architecture capacity.

---

# 15. Pair Composition

Pair fused representation:

\[
z_{12}
=
LN(
s_{12}
+
g_{12}h_{12}
).
\]

Initialize:

\[
g_{12}
\]

small.

Recommended:

\[
g_{12}\approx0.1.
\]

Same for other pairs.

---

# 16. Order Accessibility

Define first-order downstream representation:

\[
Z_{\le1}
=
[r_1,r_2,r_3].
\]

Define second-order augmented representation:

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
\boxed{
\Delta_2
=
Perf(Z_{\le2})
-
Perf(Z_{\le1})
}
\]

This is the primary pairwise higher-order metric.

---

# 17. Interpretation of Δ2

If:

\[
\Delta_2>0,
\]

then explicit second-order embeddings make joint structure more accessible to the downstream model.

This is the correct operational goal.

Do NOT interpret:

\[
\Delta_2
\]

as newly created information.

Interpret it as:

\[
\boxed{
\text{incremental representational accessibility}.
}
\]

---

# 18. Core ConFu Limitation

Original ConFu optimizes:

\[
r_{12}\leftrightarrow r_3.
\]

However:

\[
r_3
\]

contains:

```text
information predictable from r1
information predictable from r2
potential higher-order joint structure
```

Therefore the interaction branch has no explicit reason to specialize toward higher-order information.

It can align using shared semantic content.

---

# 19. ConFu++ v3 Main Idea

Instead of aligning:

\[
h_{12}
\]

to all of:

\[
r_3,
\]

align it to the part of:

\[
r_3
\]

that cannot be explained by first-order predictors.

This is:

\[
\boxed{
\text{Residual Higher-Order Contrastive Alignment}.
}
\]

---

# 20. First-Order Cross-Predictors

For target:

\[
r_3,
\]

train separate predictors:

\[
q_{1\rightarrow3}(r_1)
\]

and:

\[
q_{2\rightarrow3}(r_2).
\]

They must remain independent.

Do NOT use:

\[
q([r_1,r_2]).
\]

A joint predictor could itself learn second-order interaction.

---

# 21. Additive First-Order Prediction

Define:

\[
\hat r_3^{(1)}
=
q_{1\rightarrow3}(r_1)
+
q_{2\rightarrow3}(r_2)
+
b_3.
\]

This is a deliberately restricted order-1 approximation.

Similarly:

\[
\hat r_2^{(1)}
=
q_{1\rightarrow2}(r_1)
+
q_{3\rightarrow2}(r_3)
+
b_2.
\]

And:

\[
\hat r_1^{(1)}
=
q_{2\rightarrow1}(r_2)
+
q_{3\rightarrow1}(r_3)
+
b_1.
\]

---

# 22. Second-Order Residual Targets

Define:

\[
e_3^{(2)}
=
r_3
-
stopgrad(
\hat r_3^{(1)}
).
\]

Similarly:

\[
e_2^{(2)}
=
r_2
-
stopgrad(
\hat r_2^{(1)}
)
\]

and:

\[
e_1^{(2)}
=
r_1
-
stopgrad(
\hat r_1^{(1)}
).
\]

These are operational second-order residual targets.

---

# 23. Important Terminology

Do NOT call:

\[
e_i^{(2)}
\]

formal synergistic information.

Preferred terms:

```text
first-order residual
order-specific residual target
higher-order residual target
interaction-specific residual signal
```

Avoid:

```text
exact synergy
PID synergy
unique information
true information decomposition
```

unless formally proven.

---

# 24. Residual Alignment Objective

Align:

\[
h_{12}
\leftrightarrow
e_3^{(2)}.
\]

Similarly:

\[
h_{13}
\leftrightarrow
e_2^{(2)}
\]

and:

\[
h_{23}
\leftrightarrow
e_1^{(2)}.
\]

Define:

\[
L_{order2}
=
L_{12\rightarrow3}
+
L_{13\rightarrow2}
+
L_{23\rightarrow1}.
\]

---

# 25. Contrastive Form

For:

\[
h_{12},
e_3^{(2)},
\]

normalize:

\[
\bar h_{12}
=
\frac{h_{12}}{\|h_{12}\|}
\]

and:

\[
\bar e_3^{(2)}
=
\frac{e_3^{(2)}}{\|e_3^{(2)}\|}.
\]

Then use:

\[
InfoNCE(
\bar h_{12},
\bar e_3^{(2)}
).
\]

Use symmetric alignment if stable.

---

# 26. Why Residual Predictors Must Be Frozen

First-order predictors:

\[
q_{i\rightarrow j}
\]

must be trained before higher-order interaction training.

Then freeze them.

Otherwise optimization could artificially worsen:

\[
\hat r_j^{(1)}
\]

to create a larger residual.

This would produce fake higher-order signal.

---

# 27. Training Phase A — Baseline

Reproduce original ConFu.

Save:

```text
encoders
projection heads
pair fusion
validation metrics
test metrics
```

Do not modify canonical baseline artifacts.

---

# 28. Training Phase B — First-Order Predictors

Freeze modality encoders.

Train:

\[
q_{i\rightarrow j}.
\]

Objective:

\[
L_{pred}^{(1)}
=
\sum_{i\neq j}
\|q_{i\rightarrow j}(r_i)-r_j\|^2
\]

or a normalized cosine/MSE combination.

Select predictor checkpoints on validation reconstruction loss.

---

# 29. Predictor Capacity Rule

First-order predictors should remain simple.

Start with:

```text
Linear
or
2-layer MLP
```

Do NOT use large Transformers.

The predictors are intended to estimate accessible first-order relationships.

---

# 30. Additive Predictor Audit

Before training interaction branch, evaluate:

\[
R^2(
\hat r_j^{(1)}
\rightarrow
r_j
).
\]

Also measure residual variance:

\[
Var(e_j^{(2)}).
\]

Abort if residual variance collapses near zero.

---

# 31. Residual Health

For each:

\[
e_j^{(2)},
\]

report:

```text
variance
effective rank
norm
dimension std
cosine with original target
```

Residual signal must remain non-trivial.

---

# 32. Training Phase C — Interaction Branch Only

Freeze:

```text
modality encoders
first-order predictors
```

Train:

```text
U_i
W_ij
interaction LayerNorm
optional projection heads
```

using:

\[
L_{order2}.
\]

This is the cleanest mechanism test.

---

# 33. E4 Experiment

Name the first v3 experiment:

```text
E4 — Order-Decomposed Residual Alignment
```

Purpose:

> Can explicit second-order embeddings aligned to first-order residual targets produce positive order accessibility?

---

# 34. E4 Main Metric

Compare:

\[
Z_{\le1}
\]

versus:

\[
Z_{\le2}.
\]

Run:

```text
frozen linear probe
capacity-matched nonlinear probe
```

Compute:

\[
\Delta_2^{linear}
\]

and:

\[
\Delta_2^{nonlinear}.
\]

---

# 35. E4 Primary Gate

The preferred result is:

\[
\boxed{
\Delta_2^{linear}>0.
}
\]

Linear accessibility is preferred because it demonstrates that interaction structure has become explicit.

Nonlinear gain is secondary.

---

# 36. E4 Exploratory Criterion

Seed-1 can proceed if:

```text
validation Δ2 > 0
test exploratory Δ2 is not strongly negative
representation health passes
```

Do NOT run five seeds immediately if seed 1 fails.

---

# 37. Five-Seed Criterion

Run five seeds only if E4 passes exploratory validation.

Confirmatory success requires:

```text
positive mean Δ2
majority of seeds positive
stable variance
no major lower-order degradation
```

Preferred:

\[
4/5
\]

positive seeds.

---

# 38. Modality Dependence for h12

For:

\[
h_{12},
\]

evaluate:

\[
h_{12}(\pi(r_1),r_2)
\]

and:

\[
h_{12}(r_1,\pi(r_2)).
\]

Measure resulting:

```text
probe drop
alignment drop
```

Both modalities should matter.

---

# 39. Single-Modality Predictability

Train:

\[
q_1(r_1)\rightarrow h_{12}
\]

and:

\[
q_2(r_2)\rightarrow h_{12}.
\]

Report:

\[
R^2_1
\]

and:

\[
R^2_2.
\]

These should detect one-modality shortcuts.

---

# 40. Joint Predictability Is Not a Gate

You may still report:

\[
R^2([r_1,r_2]\rightarrow h_{12})
\]

for completeness.

But do NOT treat high joint predictability as evidence of failure.

---

# 41. Pair-Specific Order Gain

Also evaluate each pair separately.

Example:

\[
Z_{\le1}+h_{12}
\]

versus:

\[
Z_{\le1}.
\]

Define:

\[
\Delta_{12}.
\]

Similarly:

\[
\Delta_{13},
\Delta_{23}.
\]

---

# 42. Pair Utility Matrix

Mandatory:

| Pair | Linear Gain | Nonlinear Gain | Shuffle-1 | Shuffle-2 | R² single-1 | R² single-2 |
|---|---:|---:|---:|---:|---:|---:|
| V-A | | | | | | |
| V-T | | | | | | |
| A-T | | | | | | |

---

# 43. Comparison With Original ConFu Pair Embeddings

Compare:

```text
original r12
vs
new h12
```

using exactly the same probes.

The target is not necessarily:

\[
Perf(h_{12})>Perf(r_{12})
\]

standalone.

The target is:

\[
\Delta_{12}^{new}
>
\Delta_{12}^{ConFu}.
\]

---

# 44. Core ConFu++ v3 Loss

Initial:

\[
\boxed{
L
=
L_{ConFu}
+
\lambda_2L_{order2}
}
\]

during joint stage.

But E4 should first isolate:

\[
L_{order2}
\]

with frozen baseline representations.

---

# 45. Joint Training Trigger

Only after E4 passes.

Then enable:

\[
L_{ConFu}
+
\lambda_2L_{order2}.
\]

Initially update:

```text
interaction branch
projection heads
```

while keeping encoders frozen.

---

# 46. Low-LR Encoder Fine-Tuning

Only if joint interaction training improves validation.

Use:

\[
lr_{encoder}
=
0.1
\times
lr_{interaction}
\]

or:

\[
0.01
\times.
\]

Do not end-to-end fine-tune immediately.

---

# 47. Prevent Predictor Leakage

Residual predictors must never receive:

```text
pair interaction embeddings
other pair features
joint concatenation
```

They should represent lower-order approximations only.

---

# 48. Predictor Alternatives

Allowed ablations:

### P0

Linear predictor.

### P1

Two-layer MLP predictor.

### P2

Normalized additive predictor.

Do NOT use expressive cross-attention predictors in the core experiment.

---

# 49. Predictor Capacity Ablation

If residual target changes dramatically with predictor capacity, report this.

A too-weak predictor leaves first-order information in the residual.

A too-strong predictor may inadvertently approximate higher-order functions.

Therefore predictor capacity is a critical ablation.

---

# 50. Residual Orthogonality Diagnostic

Optional diagnostic:

\[
cos(
e_3^{(2)},
q_{1\rightarrow3}(r_1)
)
\]

and:

\[
cos(
e_3^{(2)},
q_{2\rightarrow3}(r_2)
).
\]

Lower similarity is desirable but not a hard constraint.

---

# 51. Do Not Add Orthogonality Loss Initially

Do NOT immediately add:

\[
L_{orth}.
\]

First observe natural residual behavior.

Additional regularization must be evidence-driven.

---

# 52. Task Supervision

Do not add task classification loss in E4.

First test whether residual contrastive alignment itself creates accessible interaction structure.

Task supervision may be added later only as an ablation.

---

# 53. Why No Task Loss Initially

Previous task auxiliary training increased text strength and damaged pair geometry.

Therefore:

\[
L_{task}
\]

must not confound the first v3 mechanism test.

---

# 54. E5 — Optional Task-Aware Order Alignment

Only if E4 succeeds.

Add:

\[
L_{task-order2}
\]

on:

\[
Z_{\le2}.
\]

Use small weight.

Test whether supervised task signal improves:

\[
\Delta_2
\]

without collapsing interaction into text shortcut.

---

# 55. E5 Acceptance

Require:

```text
Δ2 improves
single-modality shortcut does not worsen severely
pair dependence remains
```

---

# 56. Third-Order Representation

Do NOT implement:

\[
h_{123}
\]

until second-order embeddings pass.

Trigger requires:

\[
\Delta_2>0
\]

stably.

---

# 57. Third-Order Interaction

When triggered:

\[
u_1
=
LN(V_1r_1)
\]

\[
u_2
=
LN(V_2r_2)
\]

\[
u_3
=
LN(V_3r_3).
\]

Define:

\[
\boxed{
h_{123}
=
W_{123}
\left(
\frac{
u_1\odot u_2\odot u_3
}{
\sqrt R
}
\right).
}
\]

---

# 58. Third-Order Lower Representation

Define:

\[
Z_{\le2}
=
[
r_1,r_2,r_3,
h_{12},h_{13},h_{23}
].
\]

Then:

\[
Z_{\le3}
=
[
Z_{\le2},
h_{123}
].
\]

---

# 59. Third-Order Accessibility

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

This is the primary third-order metric.

---

# 60. Third-Order Residual Target

To train:

\[
h_{123},
\]

construct a target residual after removing lower-order predictions.

Conceptually:

\[
\hat z^{(\le2)}
=
\text{first-order}
+
\text{pairwise-order predictors}.
\]

Then:

\[
e^{(3)}
=
z
-
stopgrad(
\hat z^{(\le2)}
).
\]

Align:

\[
h_{123}
\leftrightarrow
e^{(3)}.
\]

---

# 61. Third-Order Warning

Do NOT implement third-order residual target naively until pairwise residual decomposition is validated.

Order-3 decomposition is substantially easier to mis-specify.

---

# 62. Controlled Synthetic Benchmark

A controlled dataset is mandatory before making strong higher-order claims.

Construct:

\[
X_1,X_2,X_3.
\]

Create:

```text
first-order information
redundant information
pairwise interaction
triple interaction
```

---

# 63. Synthetic Functional Form

A simple decomposition:

\[
Y
=
f_1(X_1)
+
f_2(X_2)
+
f_3(X_3)
\]

\[
+
f_{12}(X_1,X_2)
+
f_{13}(X_1,X_3)
+
f_{23}(X_2,X_3)
\]

\[
+
f_{123}(X_1,X_2,X_3).
\]

Control each coefficient independently.

---

# 64. Synthetic Pair Test

When only:

\[
f_{12}\neq0,
\]

desired behavior:

```text
h12 strong
h13 weak
h23 weak
h123 unnecessary
```

---

# 65. Synthetic Triple Test

When only:

\[
f_{123}\neq0,
\]

desired:

\[
\Delta_2\approx0
\]

but:

\[
\boxed{
\Delta_3>0.
}
\]

This is the strongest test of explicit third-order embedding.

---

# 66. Dataset Roles

Use:

### AV-MNIST

```text
one-modality shortcut control
```

### MOSI

```text
text-dominance / interaction-accessibility control
```

### UR-FUNNY

```text
main real-world development benchmark
```

### Synthetic

```text
ground-truth order benchmark
```

Later:

### MOSEI Emotion

```text
scale / generalization
```

### MUStARD

```text
incongruity stress test
```

---

# 67. Do Not Abandon UR-FUNNY Yet

UR-FUNNY remains useful because it has:

\[
\approx2.3\text{ pp}
\]

gap between text-only and all-modality baseline.

But future experiments must focus on representation formation.

Do NOT continue classifier-level fixes.

---

# 68. Data Limitation

Current packed UR-FUNNY representation does not expose clean:

```text
context boundary
punchline boundary
```

Therefore do NOT yet implement:

```text
context-specific cross-attention
punchline-conditioned routing
context/punchline interaction hierarchy
```

unless a raw-data pipeline is added.

---

# 69. Nonlinear Probe

Keep the capacity-matched nonlinear probe.

But the preferred primary metric remains:

\[
\Delta_k^{linear}.
\]

Why?

Because explicit higher-order representations should ideally linearize interaction structure.

---

# 70. Accessibility Ladder

Evaluate:

\[
Perf(Z_{\le1})
\]

\[
Perf(Z_{\le2})
\]

\[
Perf(Z_{\le3}).
\]

Desired:

\[
Perf(Z_{\le1})
<
Perf(Z_{\le2})
<
Perf(Z_{\le3})
\]

only when the task contains corresponding higher-order structure.

---

# 71. Do Not Force Monotonicity on Every Dataset

Some datasets may contain no true order-3 structure.

Then:

\[
\Delta_3\approx0
\]

is acceptable.

The model must not manufacture artificial third-order dependence.

---

# 72. Health Metrics

For every:

\[
h_S,
\]

report:

```text
variance
effective rank
norm
dimension-wise std
```

These detect collapse.

They do NOT demonstrate utility.

---

# 73. Shortcut Metrics

For:

\[
h_{ij},
\]

report:

\[
R^2(r_i\rightarrow h_{ij})
\]

and:

\[
R^2(r_j\rightarrow h_{ij}).
\]

For:

\[
h_{123},
\]

report predictability from:

```text
r1
r2
r3
each pair
```

individually.

---

# 74. Intervention Metrics

For pair:

\[
h_{12},
\]

shuffle modality 1 and modality 2 separately.

For third-order:

\[
h_{123},
\]

shuffle each of:

\[
r_1,r_2,r_3.
\]

Report resulting accessibility drop.

---

# 75. New Dependency Balance

Pair:

\[
B_{12}
=
\frac{
2\min(D_1,D_2)
}{
D_1+D_2+\epsilon
}.
\]

Third-order:

\[
B_{123}
=
\frac{
3\min(D_1,D_2,D_3)
}{
D_1+D_2+D_3+\epsilon
}.
\]

These are diagnostics only.

---

# 76. No Formal Synergy Claim

Do NOT say:

```text
B12 measures information-theoretic synergy
Δ2 is conditional mutual information
residual target equals PID synergy
```

Use:

```text
empirical order accessibility
interaction dependence
order-specific residual
higher-order representation utility
```

---

# 77. Parameter Matching

Every comparison between:

\[
Z_{\le1}
\]

and:

\[
Z_{\le2}
\]

must control downstream probe capacity.

Every architecture increase must have a parameter-matched generic baseline where practical.

---

# 78. Probe Training Protocol

For all official probes:

```text
same train split
same validation split
same standardization
same optimizer
same epoch budget
same validation selection
same seed
```

Never compare numbers from different probe implementations.

---

# 79. Exploratory Protocol

Use seed 1.

Hyperparameters selected using:

```text
train
validation
```

only.

Test is reported as exploratory but must not drive future hyperparameter decisions.

---

# 80. Confirmatory Protocol

Once configuration is frozen:

```text
seeds 1–5
```

Run all models.

Report:

```text
mean
sample std
paired deltas
CI
paired test
effect size
```

---

# 81. E4 Minimal Config

Suggested:

```yaml
model:
  type: order_decomposed_confu

interaction:
  type: low_rank_multiplicative
  rank: 64
  output_dim: 256
  gate_init: 0.1

first_order_predictor:
  type: linear
  freeze_after_training: true

order2:
  enabled: true
  loss: infonce
  weight: 0.1

task_loss:
  enabled: false

shortcut_adversary:
  enabled: false

residual_classifier:
  enabled: false

cross_attention:
  enabled: false

r123:
  enabled: false
```

These are starting values, not final paper hyperparameters.

---

# 82. E4 Required Outputs

Store:

```text
predictor metrics
residual target health
h12/h13/h23 health
Δ12
Δ13
Δ23
Δ2
single-modality predictability
shuffle dependence
linear probe
nonlinear probe
```

---

# 83. Result JSON

Minimum:

```json
{
  "seed": 1,

  "order1_accuracy": 0.0,
  "order2_accuracy": 0.0,
  "delta2": 0.0,

  "pair_gain_12": 0.0,
  "pair_gain_13": 0.0,
  "pair_gain_23": 0.0,

  "single_r2_12_from_1": 0.0,
  "single_r2_12_from_2": 0.0,

  "shuffle_drop_12_1": 0.0,
  "shuffle_drop_12_2": 0.0,

  "h12_variance": 0.0,
  "h12_effective_rank": 0.0,

  "residual3_variance": 0.0
}
```

---

# 84. Experiment Log Template

Every experiment must include:

```text
Research Question
Hypothesis
Scientific Bottleneck
Architecture Change
Objective Change
What Is Frozen
What Is Trainable
Validation Protocol
Test Protocol
Parameter Count
Order-1 Accuracy
Order-2 Accuracy
Δ2
Pair Gains
Shortcut Diagnostics
Representation Health
Decision
Next Step
```

---

# 85. Abort Conditions

Abort E4 if:

```text
residual targets collapse
interaction embeddings collapse
validation Δ2 strongly negative
training diverges
predictor leakage is detected
```

Do not compensate by adding more architecture.

---

# 86. Predictor Leakage Test

Train a joint predictor:

\[
q([r_1,r_2])\rightarrow r_3
\]

only as a diagnostic.

Do NOT use it to create the residual target.

Compare:

\[
Perf(q_1(r_1)+q_2(r_2))
\]

with:

\[
Perf(q_{joint}(r_1,r_2)).
\]

The gap gives an empirical indication of nonlinear joint predictability.

---

# 87. Important New Diagnostic

Define:

\[
J_{12\rightarrow3}
=
Perf(q_{joint}(r_1,r_2))
-
Perf(q_1(r_1)+q_2(r_2)).
\]

This measures:

\[
\boxed{
\text{joint prediction advantage}
}
\]

under controlled predictor classes.

This is highly relevant to higher-order structure.

---

# 88. Joint Prediction Advantage Is Not Formal Synergy

Use terminology:

```text
joint prediction advantage
non-additive predictive gain
```

Do NOT call it exact synergy.

---

# 89. Dataset Headroom Audit v3

Before heavy training, for each dataset evaluate:

```text
best single predictor
additive multimodal predictor
joint nonlinear predictor
```

If:

\[
Perf_{joint}
>
Perf_{additive},
\]

the dataset contains empirical nonlinear joint headroom under those hypothesis classes.

This is a much better gate for higher-order embedding development.

---

# 90. New Benchmark Selection Rule

A good higher-order benchmark should show:

\[
\boxed{
Perf_{joint}
-
Perf_{additive}
>0.
}
\]

This explicitly demonstrates non-additive headroom.

Use this before investing in full ConFu++ training.

---

# 91. UR-FUNNY Re-Audit

Run:

```text
additive modality predictor
joint nonlinear modality predictor
```

on frozen original representations.

Measure:

\[
J_{VAT}.
\]

If:

\[
J_{VAT}
\approx0,
\]

UR-FUNNY has limited higher-order representational headroom.

If:

\[
J_{VAT}>0,
\]

there is stronger justification for v3.

---

# 92. Synthetic Benchmark Is Mandatory If UR-FUNNY Headroom Is Weak

Do not force UR-FUNNY to demonstrate higher-order structure if the task does not contain enough.

Use synthetic order-controlled data to validate mechanism first.

---

# 93. Third-Order Development Order

Follow:

```text
pair decomposition
↓
pair residual alignment
↓
positive Δ2
↓
pair dependence verification
↓
synthetic pair success
↓
construct order-3 residual
↓
train h123
↓
measure Δ3
```

---

# 94. Do Not Skip Pair Validation

If:

\[
\Delta_2\le0,
\]

do NOT add:

\[
h_{123}.
\]

Third-order modeling cannot be used to escape an invalid second-order formulation.

---

# 95. Long-Term Scalability

For:

\[
M
\]

modalities, all interactions scale as:

\[
2^M-1.
\]

Future work may require:

```text
subset pruning
utility-based interaction selection
sparse order decomposition
learned subset routing
```

This is not part of v3 core.

---

# 96. Main Paper Contribution Candidate

If v3 succeeds, the paper contribution becomes:

1. show that higher-order alignment can produce active but poorly accessible interaction features;

2. identify that joint determinism makes naive conditional-information formulations inappropriate;

3. introduce explicit order decomposition;

4. introduce residual higher-order contrastive targets;

5. evaluate order accessibility through:

\[
\Delta_2,\Delta_3;
\]

6. validate on synthetic and real multimodal benchmarks.

---

# 97. Potential Paper Title

Preferred:

**Beyond Higher-Order Alignment: Order-Decomposed Multimodal Representation Learning**

Alternative:

**Fusion Is Not Interaction: Learning Explicit Higher-Order Multimodal Embeddings**

Alternative:

**Making Multimodal Interaction Explicit: Order-Decomposed Contrastive Learning**

---

# 98. Strong Claim Target

Future justified claim:

> Explicit order decomposition improves the accessibility of multimodal interaction structure beyond standard higher-order alignment.

---

# 99. Strong Third-Order Claim

Only after validation:

> Explicit third-order representations improve downstream accessibility beyond all first- and second-order representations.

This requires:

\[
\boxed{
\Delta_3>0.
}
\]

---

# 100. Immediate Next Experiment

The next experiment is:

\[
\boxed{
\text{E4 — Order-Decomposed Residual Alignment}
}
\]

not:

```text
more residual classifiers
more gates
more S1
cross-attention
r123
```

---

# 101. Immediate Implementation Order

Implement exactly:

```text
1. Freeze canonical Original ConFu checkpoints.

2. Build first-order cross-predictor module.

3. Train q_i→j predictors.

4. Freeze predictors.

5. Build additive first-order predictions.

6. Construct residual targets e1^(2), e2^(2), e3^(2).

7. Audit residual variance and rank.

8. Implement h12/h13/h23 low-rank multiplicative branches.

9. Train only h_ij with residual alignment.

10. Extract Z_≤1 and Z_≤2.

11. Run matched linear probe.

12. Run matched nonlinear probe.

13. Compute Δ2.

14. Compute pair-specific Δ12/Δ13/Δ23.

15. Run modality-shuffle dependence tests.

16. Run single-modality predictability tests.

17. If validation passes:
    run seed-1 exploratory test.

18. If seed-1 is promising:
    freeze configuration.

19. Run five-seed confirmatory E4.

20. Only after positive Δ2:
    consider joint ConFu + order2 training.
```

---

# 102. Do Not Optimize Test Results

No parameter may be selected from:

```text
test Δ2
test accuracy
test pair gain
```

Use validation only.

---

# 103. Core Decision Rule

If E4 fails:

\[
\Delta_2\le0,
\]

do NOT make the architecture bigger.

Instead determine whether failure is due to:

```text
no joint headroom
bad residual target
predictor capacity
wrong dataset
```

---

# 104. Benchmark Switch Trigger

Switch benchmark if:

```text
joint-vs-additive headroom ≈ 0
across multiple seeds
```

because a higher-order model cannot reliably demonstrate value when the task itself lacks measurable higher-order headroom.

---

# 105. Preferred Next Benchmark If UR-FUNNY Fails Again

Priority:

```text
Controlled Synthetic Higher-Order Benchmark
↓
MOSEI Emotion
↓
MUStARD
```

Keep:

```text
AV-MNIST
MOSI
UR-FUNNY
```

as controls and negative analyses.

---

# 106. Final Scientific Principle

The project must distinguish:

\[
\boxed{
\text{information existence}
}
\]

from:

\[
\boxed{
\text{representation accessibility}
}
\]

and:

\[
\boxed{
\text{model capacity}.
}
\]

A higher-order embedding is valuable when it turns joint structure that is difficult for lower-order models to exploit into an explicit representation that simpler downstream models can use.

---

# 107. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ learns explicit order-specific multimodal embeddings by aligning interaction branches to residual structure not captured by restricted lower-order predictors.}
}
\]

---

# 108. Final Rule

Whenever choosing between:

```text
a larger fusion architecture
```

and:

```text
a cleaner test of order-specific accessibility
```

choose:

\[
\boxed{
\text{the cleaner test}.
}
\]

The primary quantities are:

\[
\boxed{
\Delta_2
}
\]

and later:

\[
\boxed{
\Delta_3.
}
\]

Everything else is supporting evidence.