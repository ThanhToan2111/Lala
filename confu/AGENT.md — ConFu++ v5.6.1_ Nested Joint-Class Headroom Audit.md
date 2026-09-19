# AGENT.md

# ConFu++ v5.6.1

## Nested Joint-Class Headroom Audit

---

# 0. Mission

This repository develops **ConFu++**, a framework for diagnosing and discovering higher-order multimodal interaction.

The target remains a **CVPR main-track paper**.

The project has already established several separations:

\[
\text{multimodality}
\not\Rightarrow
\text{cross-modal identifiability}
\]

\[
\text{cross-modal identifiability}
\not\Rightarrow
\text{task joint headroom}
\]

\[
\text{interaction fidelity}
\not\Rightarrow
\text{downstream accessibility}.
\]

v5.6 attempted to determine whether natural G1 failures could simply be caused by an overly restrictive Product-Joint hypothesis class.

That audit was:

\[
\boxed{
\textbf{HYPOTHESIS\_CLASS\_INCONCLUSIVE}
}
\]

because the Generic Joint MLP did not recover the known positive MOSEI interaction signal.

v5.6.1 therefore performs exactly one final architecture-sufficiency audit using a **nested residual joint class**.

After v5.6.1:

\[
\boxed{
\textbf{architecture exploration is hard-frozen.}
}
\]

---

# 1. Current Project State

```text
Synthetic Oracle:
PASS

NIC:
PASS

IPIB:
PASS

Canonical JAD:
FROZEN

MOSEI canonical G1:
PASS

MOSEI G2:
INCONCLUSIVE

MOSEI D2 fidelity:
PASS

MOSEI downstream task gain:
FAIL

MOSEI S0 shortcut audit:
MIXED

MOSEI P1 purification:
FAIL

MUStARD G1:
FAIL

MUStARD G2:
FAIL / WEAK-UNSTABLE

MUStARD:
FROZEN

MELD G1:
FAIL

MELD G2:
FAIL / WEAK-UNSTABLE

MELD:
FROZEN

v5.6 Generic MLP audit:
HYPOTHESIS_CLASS_INCONCLUSIVE

v5.6.1 Nested audit:
NEXT

New dataset search:
BLOCKED

JAD modification:
BLOCKED

Task-aware JAD:
BLOCKED

Third-order:
BLOCKED
```

---

# 2. v5.6 Result

v5.6 compared:

\[
q_A
\]

\[
q_P
\]

\[
q_M.
\]

Where:

- \(q_A\): additive predictor;
- \(q_P\): canonical Product Joint;
- \(q_M\): one-hidden-layer Generic Joint MLP.

---

# 3. MOSEI Positive Control

Frozen mapping:

\[
\boxed{
VA\rightarrow T
}
\]

Validation:

\[
J_P
=
+0.0198\pm0.0019.
\]

Generic MLP:

\[
J_M
=
-0.0041\pm0.0027.
\]

Thus:

\[
\boxed{
q_M
\text{ failed to recover the known positive-control signal}.
}
\]

---

# 4. MOSEI Seed-Level H0 Result

Generic MLP:

\[
-0.0064,
\quad
-0.0049,
\quad
-0.0010.
\]

Product:

\[
+0.0220,
\quad
+0.0190,
\quad
+0.0185.
\]

Therefore the difference is stable across the three seeds.

---

# 5. MOSEI Training Health

Validation \(R^2\):

\[
q_P:
0.0304
\]

\[
q_M:
0.0065.
\]

Average best epoch:

\[
q_P:
22.3
\]

\[
q_M:
4.7.
\]

All runs are numerically finite.

The MLP does not exhibit a numerical bug, but it fails to learn the known Product-accessible signal.

---

# 6. MELD H0 Result

Validation:

### VA→T

\[
J_P
=
+0.0047
\]

\[
J_M
=
+0.0044.
\]

### VT→A

\[
J_P
=
-0.0041
\]

\[
J_M
=
+0.0018.
\]

### AT→V

\[
J_P
\approx0
\]

\[
J_M
\approx0.
\]

No MELD mapping satisfies:

\[
J_M>0.01
\]

for:

\[
3/3
\]

seeds.

---

# 7. v5.6 Official Decision

\[
\boxed{
\textbf{HYPOTHESIS\_CLASS\_INCONCLUSIVE}
}
\]

Do NOT reinterpret this as:

```text
SUFFICIENCY_SUPPORTED
```

or:

```text
PRODUCT_CLASS_INSUFFICIENT
```

Neither conclusion is justified.

---

# 8. Why Parameter Matching Was Not Enough

v5.6 matched parameter counts approximately.

However:

\[
\boxed{
|\theta_P|
\approx
|\theta_M|
}
\]

does not imply:

\[
\boxed{
\mathcal H_P
\approx
\mathcal H_M.
}
\]

Nor does it imply equivalent optimization difficulty.

---

# 9. Main Methodological Lesson

A Generic MLP can theoretically represent multiplicative behavior but may not efficiently learn the same structure under:

```text
finite data
finite optimization
early stopping
fixed capacity
```

Therefore:

\[
\boxed{
\text{equal parameter count}
\neq
\text{equal functional accessibility}.
}
\]

---

# 10. New Audit Philosophy

Instead of asking a new architecture to rediscover the Product solution from scratch, v5.6.1 constructs a **nested hypothesis class**.

The new class contains the frozen Product predictor by construction.

---

# 11. Canonical Product Predictor

Use previously trained:

\[
\boxed{
q_P(r_i,r_j).
}
\]

Do not retrain or modify its architecture.

---

# 12. Nested Joint Predictor

Define:

\[
\boxed{
q_N(r_i,r_j)
=
q_P(r_i,r_j)
+
g([r_i,r_j]).
}
\]

Where:

\[
g
\]

is a small nonlinear residual predictor.

---

# 13. Critical Nesting Property

If:

\[
g=0,
\]

then:

\[
q_N=q_P.
\]

Therefore:

\[
\boxed{
\mathcal H_P
\subset
\mathcal H_N.
}
\]

This removes the main problem of v5.6.

---

# 14. Purpose of v5.6.1

The experiment asks:

> After preserving everything Product Joint already knows, is there stable residual nonlinear predictive structure left?

This is a cleaner sufficiency test than comparing two independent models.

---

# 15. Residual Predictor

Use exactly:

\[
x=[r_i,r_j].
\]

Then:

\[
h
=
GELU(W_1x+b_1)
\]

\[
\boxed{
g(x)=W_2h+b_2.
}
\]

Exactly one hidden layer.

---

# 16. No Product Feature Inside Residual

Do NOT manually add:

```text
r_i ⊙ r_j
outer product
absolute difference
bilinear features
cross-attention
```

to \(g\).

The residual receives concatenated representations only.

---

# 17. Freeze Product Predictor

This is mandatory.

During nested training:

\[
\boxed{
\nabla q_P=0.
}
\]

Only parameters of:

\[
g
\]

are updated.

---

# 18. Residual Initialization

Initialize the final residual output layer so that:

\[
\boxed{
g(x)\approx0
}
\]

at initialization.

Preferred:

```text
zero final-layer weights
zero final-layer bias
```

if compatible with implementation.

This makes the initial model approximately equal to the frozen Product predictor.

---

# 19. Epoch-Zero Baseline

The frozen Product checkpoint must be a valid model-selection candidate.

Validation selection must compare:

\[
q_P
\]

against every trained:

\[
q_P+g_t.
\]

Thus a harmful residual cannot replace the Product baseline merely because training occurred.

---

# 20. Primary Incremental Metric

Define:

\[
\boxed{
\Delta_N
=
R^2(q_N,r_k)
-
R^2(q_P,r_k).
}
\]

This is the primary v5.6.1 statistic.

---

# 21. Total Nested Advantage

Also report:

\[
\boxed{
J_N
=
R^2(q_N,r_k)
-
R^2(q_A,r_k).
}
\]

Thus:

\[
J_N
=
J_P+\Delta_N
\]

up to evaluation/numerical consistency.

---

# 22. Why \(\Delta_N\) Is Better Than \(J_M\)

v5.6 asked:

\[
q_M
\text{ vs }
q_A.
\]

v5.6.1 asks:

\[
q_P+g
\text{ vs }
q_P.
\]

Therefore the residual learner only needs to discover:

\[
\boxed{
\text{structure missing from Product Joint}.
}
\]

It does not need to relearn already-known multiplicative structure.

---

# 23. Dataset Scope

Use exactly:

\[
\boxed{
\text{MOSEI}
}
\]

and:

\[
\boxed{
\text{MELD}.
}
\]

No additional datasets.

---

# 24. MOSEI Role

MOSEI remains the positive structural control.

Mapping:

\[
\boxed{
VA\rightarrow T.
}
\]

Primary question:

> Does a nonlinear residual expose additional predictable structure beyond the already-positive Product predictor?

---

# 25. MELD Role

MELD is the critical negative setting.

Audit:

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V.
\]

Primary question:

> Does a nested nonlinear extension expose meaningful headroom that Product Joint missed?

---

# 26. Frozen Representation — MOSEI

Use exact existing canonical cache:

```text
V = 713
A = 74
T = 300
```

Same:

```text
splits
normalization
pooling
alignment
```

as previous canonical experiments.

---

# 27. Frozen Representation — MELD

Use exact v5.5 cache:

```text
V = 2048
A = 32
T = 300
```

Same:

```text
masked mean
utterance-only
train normalization
official split
```

No dialogue context.

---

# 28. No Representation Changes

Forbidden:

```text
new encoder
sequence Transformer
context encoder
new pooling
new text embedding
feature fine-tuning
```

The only new component is:

\[
g.
\]

---

# 29. Residual Capacity Philosophy

Unlike v5.6, do NOT force:

\[
Params(q_N)
\approx
Params(q_P).
\]

A nested model necessarily adds capacity.

Instead report:

\[
\boxed{
\rho
=
\frac{
Params(g)
}{
Params(q_P)
}.
}
\]

---

# 30. Residual Must Remain Small

Choose one modest hidden width before full experiments.

Goal:

\[
\boxed{
\text{small residual extension}
}
\]

not a new large multimodal model.

---

# 31. Hidden Width Selection

Use one deterministic rule.

Example:

Choose the largest integer \(h\) satisfying:

\[
Params(g)
\le
0.25\,
Params(q_P).
\]

or another fixed preregistered budget.

Once chosen:

\[
\boxed{
\text{do not sweep }h.
}
\]

---

# 32. Preferred Residual Budget

Recommended initial maximum:

\[
\boxed{
Params(g)
\le25\%
\text{ of }Params(q_P).
}
\]

This is a complexity-control convention, not a scientific interaction threshold.

Report exact ratio.

---

# 33. No Capacity Sweep

Do NOT test:

```text
5%
10%
25%
50%
100%
```

residual budgets.

Exactly one.

---

# 34. Optimization

Use the established regression optimizer family.

Keep:

```text
learning rate
weight decay
batch size
max epochs
patience
```

fixed across datasets where practical.

---

# 35. Training Target

The nested model predicts exactly the original target modality representation:

\[
r_k.
\]

Loss:

\[
\boxed{
MSE(q_N,r_k).
}
\]

Do not train directly on:

\[
\Delta_N
\]

or:

\[
J_N.
\]

---

# 36. Early Stopping

Select by:

\[
\boxed{
validation MSE.
}
\]

Not by:

\[
\Delta_N.
\]

---

# 37. Test Use

Test remains:

\[
\boxed{
\text{reporting only}.
}
\]

No decision may be reversed using test results.

---

# 38. Seeds

Use:

\[
\boxed{
3\text{ seeds}.
}
\]

No automatic five-seed confirmation.

---

# 39. Finality Rule

v5.6.1 is the final architecture audit regardless of outcome.

Do NOT create:

```text
v5.6.2
v5.6.3
```

for more architecture tests.

---

# 40. Primary Validation Table

Produce:

| Dataset | Mapping | \(J_P\) | \(J_N\) | \(\Delta_N\) |
|---|---|---:|---:|---:|
| MOSEI | VA→T | | | |
| MELD | VA→T | | | |
| MELD | VT→A | | | |
| MELD | AT→V | | | |

Use:

\[
mean\pm sample\ std.
\]

---

# 41. Seed-Level Table

Also report:

| Dataset | Mapping | Seed | \(J_P\) | \(J_N\) | \(\Delta_N\) |
|---|---|---:|---:|---:|---:|

---

# 42. Training Health

For every residual run save:

```text
initial validation MSE
best validation MSE
best epoch
train R²
validation R²
test R²
residual output norm
finite check
```

---

# 43. Residual Norm Diagnostic

Report:

\[
\boxed{
E\|g(x)\|_2.
}
\]

If residual remains near zero:

interpret as no learned additional structure.

Do not treat it as collapse requiring tuning.

---

# 44. Product Preservation Check

At initialization verify:

\[
q_N\approx q_P.
\]

Add unit test:

\[
\max
\left|
q_N-q_P
\right|
<
\epsilon
\]

before residual training.

---

# 45. Numerical Tolerance

Use a reasonable implementation tolerance such as:

\[
10^{-6}
\]

or corresponding floating-point-safe threshold.

---

# 46. Outcome A — Nested Headroom Absent

If no MELD mapping satisfies:

\[
\boxed{
\Delta_{N,val}>0.01
}
\]

for:

\[
3/3
\]

seeds:

classify:

\[
\boxed{
\textbf{NESTED\_HEADROOM\_ABSENT}.
}
\]

---

# 47. Meaning of Outcome A

Correct interpretation:

> A small nonlinear residual extension on top of the frozen Product Joint does not expose substantial additional cross-modal predictive headroom on MELD.

This strengthens the negative natural G1 evidence.

---

# 48. Do Not Overclaim Outcome A

Never write:

> MELD contains no interaction.

Instead:

\[
\boxed{
\text{no additional headroom was detected under the tested nested extension}.
}
\]

---

# 49. Outcome B — Nested Headroom Present

If at least one MELD mapping satisfies:

\[
\boxed{
\Delta_{N,val}>0.01
}
\]

for:

\[
3/3
\]

seeds:

classify:

\[
\boxed{
\textbf{NESTED\_HEADROOM\_PRESENT}.
}
\]

---

# 50. Meaning of Outcome B

Then Product Joint leaves measurable nonlinear predictive structure unexplained.

This establishes:

\[
\boxed{
\text{joint-class limitation exists empirically}.
}
\]

---

# 51. Outcome B Does Not Automatically Open Architecture Search

Even if Outcome B occurs:

do NOT sweep architectures.

Freeze result.

Move to theory and paper analysis first.

Any method revision must be independently justified later.

---

# 52. Outcome C — Nested Audit Inconclusive

Use:

\[
\boxed{
\textbf{NESTED\_AUDIT\_INCONCLUSIVE}
}
\]

if:

```text
large seed instability
optimization failure
non-finite behavior
positive/negative threshold crossing without consistency
```

---

# 53. No Tuning After Outcome C

If inconclusive:

\[
\boxed{
\text{report limitation and stop}.
}
\]

Do not tune the residual model.

---

# 54. Existing 0.01 Threshold

Reuse:

\[
\boxed{
0.01
}
\]

as the operational threshold for meaningful residual headroom.

Do not invent a new threshold after seeing results.

---

# 55. Important Caveat

\[
\Delta_N>0.01
\]

does NOT mean the residual learned:

```text
PID synergy
causal interaction
true interaction
```

Only:

\[
\boxed{
\text{additional predictable structure beyond Product Joint}.
}
\]

---

# 56. MOSEI Interpretation

MOSEI does not require:

\[
\Delta_N>0.
\]

A near-zero MOSEI result is acceptable and could indicate:

\[
q_P
\]

already captures most measurable cross-modal headroom in that mapping.

---

# 57. MOSEI Unexpected Large Residual

If:

\[
\Delta_N
\]

is large on MOSEI:

this indicates Product is incomplete even on the known positive setting.

Report it.

Do not tune JAD immediately.

---

# 58. Architecture Work Hard Stop

After v5.6.1:

\[
\boxed{
\textbf{NO MORE JOINT-ARCHITECTURE EXPERIMENTS}.
}
\]

This rule holds regardless of result.

---

# 59. Next Research Phase

After v5.6.1 move to:

\[
\boxed{
\textbf{v6.0 — Paper Consolidation and Formalization}
}
\]

not another architecture version.

---

# 60. v6.0 Mandatory Workstreams

The next phase must focus on:

```text
formal theory
gate calibration
Naturalized IPIB
closest-method comparison
paper figures
paper narrative
limitations
```

---

# 61. Workstream T1 — Formal Projection Theory

Formalize the population meaning of additive-vs-joint advantage.

Let:

\[
m(x_i,x_j)
=
E[r_k\mid x_i,x_j].
\]

Define additive function space:

\[
\mathcal H_A.
\]

Define joint function space:

\[
\mathcal H_J.
\]

Under squared loss, characterize:

\[
q_A^*
\]

and:

\[
q_J^*
\]

as population projections.

---

# 62. Desired Proposition

Under suitable nested Hilbert-space assumptions:

\[
\boxed{
R^2(q_J^*,r_k)
-
R^2(q_A^*,r_k)
=
\frac{
\|
\Pi_{\mathcal H_J}m
-
\Pi_{\mathcal H_A}m
\|_2^2
}{
Var(r_k)
}
}
\]

or the correct vector-output analogue.

Do not publish this formula until assumptions and derivation are verified.

---

# 63. Vector-Output Care

Targets are multimodal representations:

\[
r_k\in\mathbb R^d.
\]

Therefore theory must define exactly:

```text
vector R²
loss normalization
expectation over dimensions
projection inner product
```

Do not silently reuse scalar formulas.

---

# 64. Workstream T2 — Gate Calibration

The historical gate is:

\[
J>0.01.
\]

v6.0 must audit sensitivity without changing historical experiment decisions.

---

# 65. Null Calibration

Construct a null distribution using a preregistered procedure such as:

```text
source permutation
target permutation
conditional-safe shuffle if justified
```

Primary goal:

estimate typical finite-sample joint-minus-additive fluctuations under broken cross-modal alignment.

---

# 66. Threshold Sensitivity

Report decisions under:

\[
0.005,
\quad
0.01,
\quad
0.02
\]

as robustness analysis.

Do NOT reselect methods/datasets using alternate thresholds.

Historical gate remains:

\[
0.01.
\]

---

# 67. Workstream T3 — Naturalized IPIB

Create a bridge benchmark between synthetic IPIB and natural representations.

Goal:

\[
\boxed{
\text{realistic source geometry}
+
\text{known interaction ground truth}.
}
\]

---

# 68. Naturalized IPIB Principle

Use source representations sampled from a real dataset.

Construct a controlled target containing:

\[
\boxed{
\text{additive component}
}
\]

+

\[
\boxed{
\text{known interaction component}
}
\]

+

\[
\boxed{
\text{private/noise component}.
}
\]

---

# 69. Why Naturalized IPIB

Current IPIB controls interaction perfectly but uses synthetic latent geometry.

Naturalized IPIB asks whether interaction discovery survives:

```text
natural correlations
anisotropy
real representation geometry
support structure
```

while preserving known ground truth.

---

# 70. No Naturalized Benchmark Tuning

Define one construction family.

Do not tune it until JAD looks good.

Its purpose is controlled bridging, not performance optimization.

---

# 71. Workstream T4 — Closest-Method Comparison

At minimum position against:

```text
Original ConFu
standard additive prediction
joint nonlinear prediction
DMIL-style decomposition where reproducible
relevant synergy/decomposition baseline
```

Exact baseline set must be determined through literature audit.

---

# 72. Comparison Goal

Do NOT only compare downstream accuracy.

Compare where possible:

```text
interaction recovery
false positives
identifiability
task accessibility
representation health
```

because ConFu++ is increasingly a diagnostic framework.

---

# 73. Novelty Positioning

Do NOT claim:

> first multimodal interaction decomposition.

Do NOT claim:

> first multimodal synergy method.

Preferred novelty:

\[
\boxed{
\text{a framework separating whether interaction is identifiable, task-relevant, faithfully represented, and accessible}.
}
\]

---

# 74. Main Paper Hierarchy

Current candidate hierarchy:

\[
\boxed{
\text{Multimodal Inputs}
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
\text{Interaction Fidelity}
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

# 75. Existing Evidence Mapping

### Multimodality does not imply identifiability

Evidence:

```text
MUStARD
MELD
```

under tested representations/classes.

### Identifiability does not imply demonstrated task headroom

Evidence:

```text
MOSEI
```

### Fidelity does not imply task accessibility

Evidence:

```text
IPIB
MOSEI
```

---

# 76. H0 Methodological Finding

v5.6 contributes another limitation:

\[
\boxed{
\text{parameter-count matching does not guarantee equivalent interaction accessibility}.
}
\]

This should be reported as a methodological observation, not a main theoretical claim.

---

# 77. Current Paper Position

Preferred:

\[
\boxed{
\textbf{scientific/diagnostic framework paper}
}
\]

with JAD as a concrete interaction-discovery mechanism.

Not:

\[
\boxed{
\text{SOTA fusion architecture paper}.
}
\]

---

# 78. Type-IV Status

No natural Type-IV setting is currently established.

Do not hide this.

Do not resume broad dataset search after v5.6.1.

---

# 79. Positive Natural Utility Is Desirable, Not Mandatory

If later a theoretically motivated Type-IV setting emerges naturally from paper analysis, it may be tested.

But no dataset fishing is allowed.

---

# 80. Current Forbidden Work

During v5.6.1:

```text
NO dataset search
NO Generic MLP tuning
NO deeper residual MLP
NO residual width sweep
NO optimizer sweep
NO pooling change
NO context
NO new encoders
NO task-aware JAD
NO purification
NO rank sweep
NO loss sweep
NO third-order
```

---

# 81. Current Permitted Work

```text
nested residual implementation
MOSEI nested audit
MELD nested audit
capacity reporting
training-health reporting
theory preparation
paper outline preparation
```

---

# 82. Recommended Code

Create:

```text
src/experiments/hypothesis_class/
    nested_joint.py
    mosei_nested_h0.py
    meld_nested_h0.py
    nested_h0_summary.py
```

Reuse:

```text
frozen Product checkpoints
existing dataset caches
existing metric utilities
```

---

# 83. Result Directory

```text
results/hypothesis_class/v561/
    mosei/
    meld/
    NESTED_JOINT_CLASS_DECISION.md
    nested_h0_summary.json
```

---

# 84. MOSEI Artifact

```text
MOSEI_NESTED_HEADROOM_REPORT.md
```

---

# 85. MELD Artifact

```text
MELD_NESTED_HEADROOM_REPORT.md
```

---

# 86. Combined Decision Artifact

```text
NESTED_JOINT_CLASS_DECISION.md
```

must end with exactly one:

```text
NESTED_HEADROOM_ABSENT

NESTED_HEADROOM_PRESENT

NESTED_AUDIT_INCONCLUSIVE
```

---

# 87. Machine Summary Fields

Save:

```text
dataset
mapping
seed
product_checkpoint
product_params
residual_params
residual_ratio
initial_val_mse
best_val_mse
best_epoch
product_train_r2
nested_train_r2
product_val_r2
nested_val_r2
product_test_r2
nested_test_r2
J_product
J_nested
delta_nested
residual_norm
finite
```

---

# 88. Unit Tests

Required:

```text
Product parameters frozen

Nested output equals Product output at initialization

Residual receives concatenated source representations

Residual target dimension correct

Residual parameter budget obeyed

Exact previous data split reused

Exact previous normalization reused

Validation-only selection

Product baseline included as epoch-zero candidate

Test never affects decision

Deterministic seeds
```

---

# 89. Smoke Test

Run:

```text
MOSEI VA→T seed 1
MELD VA→T seed 1
```

Only verify implementation.

Check:

```text
nested == product at init
finite loss
residual gradients nonzero
product gradients zero
metrics correct
checkpointing correct
```

---

# 90. Full Execution

Then run:

### MOSEI

\[
VA\rightarrow T
\]

seeds:

\[
1,2,3.
\]

### MELD

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V
\]

seeds:

\[
1,2,3.
\]

---

# 91. Primary Decision

Compute:

\[
\Delta_N
=
R^2_N-R^2_P.
\]

Then classify using validation only.

---

# 92. Decision Rule — Absent

If no MELD mapping has:

\[
\Delta_N>0.01
\]

in all 3 seeds:

\[
\boxed{
\textbf{NESTED\_HEADROOM\_ABSENT}.
}
\]

---

# 93. Decision Rule — Present

If any MELD mapping has:

\[
\Delta_N>0.01
\]

in all 3 seeds:

\[
\boxed{
\textbf{NESTED\_HEADROOM\_PRESENT}.
}
\]

---

# 94. Decision Rule — Inconclusive

If optimization or seed instability prevents interpretation:

\[
\boxed{
\textbf{NESTED\_AUDIT\_INCONCLUSIVE}.
}
\]

---

# 95. No Five-Seed Requirement

Unlike earlier screening, v5.6.1 is not opening a method branch.

Three seeds are sufficient for this final audit.

Do not automatically run five.

---

# 96. No Downstream Tasks

Do NOT run:

```text
sentiment
emotion
sarcasm
```

in v5.6.1.

This is a G1 hypothesis-class audit only.

---

# 97. No JAD Target From Nested Model

Do NOT construct:

\[
d_N=q_N-q_A
\]

for JAD training.

That would be a method revision.

Not permitted.

---

# 98. No Synthetic Validation Yet

Do not reopen IPIB from v5.6.1.

The nested model is only an audit instrument.

---

# 99. Hard Architecture Freeze

After results are written:

\[
\boxed{
\textbf{freeze all architecture-level experimentation.}
}
\]

---

# 100. Next Version

After v5.6.1:

\[
\boxed{
\textbf{v6.0 — Formalization and Paper Consolidation}.
}
\]

---

# 101. v6.0 Priority Order

```text
1. Formal projection theory

2. Gate / null calibration

3. Naturalized IPIB

4. Closest-method literature audit

5. Baseline comparison

6. Main figures

7. Natural-regime table

8. Claims matrix

9. Limitations

10. Paper writing
```

---

# 102. Claims Matrix

Create:

```text
paper/claims.md
```

with:

| Claim | Evidence | Dataset | Limitation | Status |
|---|---|---|---|---|

No paper claim should exist without a linked experiment/theorem.

---

# 103. Reviewer Question File

Update:

```text
paper/reviewer_questions.md
```

with at least:

```text
Why does G1 depend on the joint predictor class?

Why is J > 0.01 meaningful?

Why isn't J formal PID synergy?

Why should cross-modal interaction help the task?

Why does JAD not improve MOSEI accuracy?

Why are MUStARD/MELD negative?

Could the negative result be caused by representation choice?

Could it be caused by optimization?

Why is JAD useful without SOTA accuracy?
```

---

# 104. Core Scientific Language

Preferred:

```text
cross-modal predictive interaction
joint predictive advantage
non-additive predictive headroom
representation-conditional identifiability
hypothesis-class-relative identifiability
task joint headroom
interaction fidelity
accessibility
```

---

# 105. Forbidden Language

Avoid:

```text
true synergy
PID synergy
new information created by fusion
causal interaction
pure interaction
interaction does not exist
```

unless formally proven.

---

# 106. Final v5.6.1 Scientific Question

\[
\boxed{
\textbf{
After preserving the full canonical Product-Joint predictor, does a modest nonlinear residual reveal substantial additional cross-modal predictive headroom?
}
}
\]

---

# 107. v5.6.1 Success Definition

The experiment succeeds if it resolves this question without architecture search.

It does NOT require a positive result.

---

# 108. Final Research Rule

\[
\boxed{
\textbf{
One nested audit.
Then stop architecture work.
Move to theory, calibration, controlled bridging, and paper consolidation.
}
}
\]

---

# 109. Immediate Execution Order

```text
1. Freeze existing MOSEI Product checkpoints.

2. Freeze existing MELD Product checkpoints.

3. Implement NestedJointPredictor.

4. Freeze Product parameters.

5. Add one-hidden-layer residual MLP.

6. Derive one fixed residual hidden width from preregistered budget.

7. Zero-initialize residual output.

8. Verify nested == product at initialization.

9. Include Product baseline as epoch-zero validation candidate.

10. Run MOSEI VA→T seed-1 smoke test.

11. Run MELD VA→T seed-1 smoke test.

12. Verify:
    Product grad = 0
    Residual grad != 0
    loss finite
    metrics finite.

13. Run MOSEI VA→T seeds 1..3.

14. Run MELD:
    VA→T
    VT→A
    AT→V
    seeds 1..3.

15. Compute:
    J_Product
    J_Nested
    Delta_N.

16. Produce seed-level and mean±std tables.

17. Write NESTED_JOINT_CLASS_DECISION.md.

18. Classify exactly one:
    NESTED_HEADROOM_ABSENT
    NESTED_HEADROOM_PRESENT
    NESTED_AUDIT_INCONCLUSIVE.

19. Freeze architecture experimentation.

20. Begin v6.0 paper consolidation.
```

---

# 110. Final v5.6.1 Thesis

\[
\boxed{
\textbf{
Hypothesis-class sufficiency should be audited with nested function classes so that additional expressivity is tested without forcing a new model to rediscover already-known interaction structure.
}
}
\]

---

# 111. Next Action

\[
\boxed{
\textbf{
Run the final Nested Joint-Class Headroom Audit, then hard-freeze architecture development.
}
}
\]