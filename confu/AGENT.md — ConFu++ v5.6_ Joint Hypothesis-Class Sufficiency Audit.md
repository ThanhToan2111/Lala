# AGENT.md

# ConFu++ v5.6

## Joint Hypothesis-Class Sufficiency Audit

---

# 0. Mission

This repository develops **ConFu++**, a framework for explicit multimodal interaction discovery and diagnosis.

The target remains a **CVPR main-track paper**.

The project has now established that:

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

and:

\[
\text{interaction fidelity}
\not\Rightarrow
\text{downstream accessibility}.
\]

However, one major unresolved confound remains:

\[
\boxed{
\textbf{Is the current joint predictor class expressive enough to detect natural non-additive structure?}
}
\]

v5.6 exists only to answer this question.

No new JAD method is proposed in v5.6.

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

Synthetic fidelity:
PASS

Synthetic false-positive control:
PASS

Fidelity–Accessibility gap:
ESTABLISHED

MOSEI canonical G1:
PASS

MOSEI G2:
INCONCLUSIVE

MOSEI JAD:
COMPLETE

MOSEI task gain:
FAIL

MOSEI shortcut audit:
COMPLETE

MOSEI purification:
FAIL

MUStARD G1:
FAIL ALL PAIRS

MUStARD G2:
FAIL / WEAK_UNSTABLE

MUStARD:
FROZEN

MELD G1:
FAIL ALL PAIRS

MELD G2:
FAIL / WEAK_UNSTABLE

MELD:
FROZEN

Natural Type-IV setting:
NOT FOUND

Hypothesis-class audit:
NEXT

New dataset search:
BLOCKED

Task-aware JAD:
BLOCKED

Third-order discovery:
BLOCKED
```

---

# 2. Why v5.6 Is Necessary

The current G1 metric is:

\[
\boxed{
J_{\text{cross}}
=
R^2(q_{\text{joint}},r_k)
-
R^2(q_{\text{add}},r_k).
}
\]

But this quantity depends on the chosen model classes:

\[
\boxed{
J
=
J(
\mathcal D,
\phi,
\mathcal H_{\text{add}},
\mathcal H_{\text{joint}}
).
}
\]

Therefore:

\[
J\approx0
\]

does NOT uniquely imply:

> no natural interaction exists.

It may instead imply:

\[
\boxed{
\mathcal H_{\text{joint}}
\text{ is too restrictive}.
}
\]

This is the central unresolved methodological issue.

---

# 3. Current Canonical Joint Predictor

The current G1 joint predictor uses a low-rank multiplicative interaction.

Conceptually:

\[
u_i=P_ir_i
\]

\[
u_j=P_jr_j
\]

and interaction:

\[
u_i\odot u_j.
\]

The final predictor contains explicit multiplicative access.

This family will now be called:

\[
\boxed{
\mathcal H_P
}
\]

where \(P\) denotes:

\[
\text{Product Joint}.
\]

---

# 4. Current Additive Predictor

Canonical additive family:

\[
\boxed{
\mathcal H_A
}
\]

with:

\[
q_{\text{add}}
=
q_i(r_i)
+
q_j(r_j).
\]

The two source branches are independent until output addition.

---

# 5. New Hypothesis-Class Control

v5.6 introduces exactly one additional joint family:

\[
\boxed{
\mathcal H_M
}
\]

where \(M\) denotes:

\[
\text{Generic Joint MLP}.
\]

Define:

\[
\boxed{
q_{\text{MLP}}
=
MLP([r_i,r_j]).
}
\]

This model receives both modalities jointly through concatenation and may represent generic nonlinear interactions.

---

# 6. Purpose of Generic Joint MLP

The generic MLP is NOT a new ConFu++ method.

It is a diagnostic control.

It asks:

> Is there cross-modal nonlinear predictive headroom that the explicit product predictor fails to capture?

It must not be optimized for SOTA performance.

---

# 7. v5.6 Core Comparisons

For every audited mapping:

\[
(i,j)\rightarrow k,
\]

train:

\[
q_A
\]

\[
q_P
\]

\[
q_M.
\]

Then compute:

\[
\boxed{
J_P
=
R^2(q_P,r_k)
-
R^2(q_A,r_k)
}
\]

and:

\[
\boxed{
J_M
=
R^2(q_M,r_k)
-
R^2(q_A,r_k).
}
\]

---

# 8. Primary Research Question

\[
\boxed{
\textbf{
When canonical Product-Joint shows no advantage, does a generic nonlinear joint predictor reveal hidden cross-modal headroom?
}
}
\]

---

# 9. v5.6 Datasets

Use exactly:

\[
\boxed{
\textbf{MOSEI}
}
\]

and:

\[
\boxed{
\textbf{MELD}.
}
\]

Do NOT run MUStARD in v5.6.

---

# 10. Why MOSEI

MOSEI is the natural positive control.

Canonical:

\[
VA\rightarrow T
\]

already satisfies:

\[
J_P>0.
\]

Therefore MOSEI tests whether:

\[
q_M
\]

can recover a known positive natural interaction signal.

---

# 11. Why MELD

MELD is the critical natural negative.

All three mappings satisfy approximately:

\[
J_P\approx0.
\]

Therefore MELD tests whether G1 failure persists under a more generic nonlinear joint family.

---

# 12. Why Not MUStARD

MUStARD is small and already shows substantial variance/overfitting.

Including it would add ambiguity without materially improving the core hypothesis-class question.

Freeze MUStARD.

---

# 13. Frozen Representations

Do not change dataset representations.

### MOSEI

Use:

```text
canonical raw pooled features
V = 713
A = 74
T = 300
```

with exact canonical split and alignment.

### MELD

Use:

```text
utterance-level pooled features
V = 2048
A = 32
T = 300
```

with the same official labels/features and masked-mean pipeline used in v5.5.

---

# 14. No Representation Rescue

Do NOT introduce:

```text
temporal encoder
dialogue context
cross-attention
new embeddings
new feature extractor
alternative pooling
```

in v5.6.

The independent variable is only:

\[
\boxed{
\mathcal H_{\text{joint}}.
}
\]

---

# 15. MOSEI Mapping

Primary MOSEI mapping:

\[
\boxed{
VA\rightarrow T.
}
\]

Do not rerun all MOSEI directions.

Reason:

this is the known positive-control mapping.

---

# 16. MELD Mappings

Audit:

\[
\boxed{
VA\rightarrow T
}
\]

\[
\boxed{
VT\rightarrow A
}
\]

\[
\boxed{
AT\rightarrow V.
}
\]

These are the previously failed G1 mappings.

---

# 17. Capacity Matching

Require:

\[
Params(q_A)
\approx
Params(q_P)
\approx
Params(q_M).
\]

Preferred mismatch:

\[
<1\%.
\]

Maximum:

\[
<5\%.
\]

Report exact parameter counts.

---

# 18. Capacity Matching Philosophy

Do not create a large MLP and then shrink the other models to match it.

Instead:

1. preserve established additive/product scale;
2. solve for a compact MLP hidden width producing similar parameter count;
3. freeze it before full experiments.

---

# 19. Generic MLP Architecture

Preferred initial form:

\[
x=[r_i,r_j]
\]

\[
h=\sigma(W_1x+b_1)
\]

\[
q_M=W_2h+b_2.
\]

Exactly:

\[
\boxed{
\text{one hidden layer}.
}
\]

Do not use deeper MLPs initially.

---

# 20. Activation

Use one fixed activation:

\[
\boxed{
GELU
}
\]

or the same activation family already common in existing predictors.

Do not compare activations.

---

# 21. Normalization

If current predictor families use normalization, preserve comparable normalization.

Do not give the MLP an additional normalization advantage.

---

# 22. Residual Connections

Do NOT add residual blocks.

The generic MLP must remain a simple hypothesis-class diagnostic.

---

# 23. Dropout

Prefer:

\[
\boxed{
\text{no dropout}
}
\]

unless existing G1 predictors already use it.

Do not use dropout tuning to improve MLP results.

---

# 24. Optimization Fairness

Use the same:

```text
optimizer family
learning rate
weight decay
batch size
maximum epochs
early stopping
seed protocol
```

across:

\[
q_A,q_P,q_M
\]

where practical.

---

# 25. Early Stopping

Use validation MSE.

Do not early stop on:

\[
J_P
\]

or:

\[
J_M.
\]

This preserves independence of training and interaction measurement.

---

# 26. Seeds

Initial audit:

\[
\boxed{
3\text{ seeds}.
}
\]

This is sufficient for H0 screening.

---

# 27. Five-Seed Confirmation

Only if MELD generic MLP shows a clear positive candidate:

\[
J_M>0.01
\]

for:

\[
3/3
\]

seeds,

run:

\[
5\text{ seeds total}
\]

with architecture frozen.

---

# 28. Primary Metrics

For every model report:

\[
R^2_{\text{train}}
\]

\[
R^2_{\text{val}}
\]

\[
R^2_{\text{test}}.
\]

For joint models:

\[
J_{P,\text{train/val/test}}
\]

and:

\[
J_{M,\text{train/val/test}}.
\]

---

# 29. Primary Decision Uses Validation

All scientific decisions use:

\[
\boxed{
\text{validation}.
}
\]

Test remains descriptive.

Do not use test to rescue hypothesis-class failure.

---

# 30. Required Main Table

Produce:

| Dataset | Mapping | Additive R² | Product R² | MLP R² | \(J_P\) | \(J_M\) |
|---|---|---:|---:|---:|---:|---:|
| MOSEI | VA→T | | | | | |
| MELD | VA→T | | | | | |
| MELD | VT→A | | | | | |
| MELD | AT→V | | | | | |

Validation mean ± sample std.

---

# 31. Required Seed-Level Table

Also report:

```text
dataset
mapping
seed
J_P
J_M
```

No mean-only interpretation.

---

# 32. Positive-Control Requirement

MOSEI should ideally satisfy:

\[
J_M>0.
\]

If:

\[
J_P>0
\]

but:

\[
J_M\le0,
\]

do NOT immediately conclude generic MLP is weaker.

Audit:

```text
capacity
optimization
early stopping
training R²
```

first.

---

# 33. H0 Outcome A — Both Joint Classes Fail on MELD

If:

\[
J_P\approx0
\]

and:

\[
J_M\approx0
\]

for all MELD mappings:

\[
\boxed{
\textbf{HYPOTHESIS-CLASS SUFFICIENCY AUDIT SUPPORTS THE NEGATIVE G1 RESULT}.
}
\]

Interpretation:

> The absence of measurable joint predictive advantage is not specific to the canonical multiplicative predictor under the tested capacity-controlled classes.

---

# 34. Outcome A Scientific Meaning

This strengthens:

\[
\boxed{
\text{representation-conditional non-identifiability}.
}
\]

Correct wording:

> No stable non-additive predictive headroom was detected under either explicit-product or generic nonlinear joint predictor classes.

Do NOT say:

> MELD has no interaction.

---

# 35. Outcome A Action

If Outcome A holds:

\[
\boxed{
\textbf{FREEZE METHOD DEVELOPMENT}.
}
\]

Do NOT:

```text
add Transformer
add cross-attention
increase width
change pooling
switch dataset
```

automatically.

Move to:

\[
\boxed{
\text{paper consolidation}.
}
\]

---

# 36. H0 Outcome B — Generic MLP Reveals MELD Headroom

If:

\[
J_P\approx0
\]

but:

\[
\boxed{
J_M>0.01
}
\]

for:

\[
3/3
\]

seeds on a MELD mapping:

then:

\[
\boxed{
\textbf{canonical Product-Joint hypothesis class is insufficient}.
}
\]

---

# 37. Outcome B Confirmation

Freeze:

```text
dataset
mapping
MLP architecture
parameter count
optimizer
split
representation
```

and confirm:

\[
5\text{ seeds}.
\]

No changes before confirmation.

---

# 38. Outcome B Action

If confirmed:

open:

\[
\boxed{
\textbf{v5.7 — Joint Hypothesis-Class Revision}.
}
\]

v5.7 may replace or augment the canonical joint predictor with one justified alternative.

Do NOT redesign JAD in v5.6.

---

# 39. H0 Outcome C — MLP Overfits

If:

\[
J_{M,\text{train}}\gg0
\]

but:

\[
J_{M,\text{val}}\le0,
\]

then:

\[
\boxed{
\text{generic nonlinear capacity does not solve identifiability}.
}
\]

Classify as:

```text
GENERIC_JOINT_OVERFIT
```

not hidden interaction.

---

# 40. H0 Outcome D — Mixed MELD Result

Example:

\[
VA\rightarrow T:
J_M>0
\]

but other mappings remain zero.

If validation gate passes for the one mapping:

that mapping becomes a hypothesis-class candidate.

Do not require all MELD mappings to pass.

---

# 41. Hard Positive Criterion

For a new MLP interaction candidate require:

\[
\boxed{
J_{M,val}>0.01
}
\]

for:

\[
3/3
\]

seeds.

Do not weaken the historical G1 threshold.

---

# 42. Why Keep 0.01

The threshold has already been used throughout:

```text
synthetic screening
MOSEI
MUStARD
MELD
```

Changing it now would make cross-dataset interpretation inconsistent.

---

# 43. Product-vs-MLP Difference

Define:

\[
\boxed{
\Delta_{\mathcal H}
=
J_M-J_P.
}
\]

This measures how much extra cross-modal advantage is exposed by generic nonlinear access relative to product interaction.

---

# 44. Interpret \(\Delta_{\mathcal H}\) Carefully

If:

\[
\Delta_{\mathcal H}>0,
\]

it does NOT automatically mean:

> MLP discovers truer interaction.

It only means:

\[
\boxed{
\mathcal H_M
\text{ exposes more predictive headroom than }
\mathcal H_P.
}
\]

---

# 45. Hypothesis-Class Relative Identifiability

After v5.6 the formal framing should be:

\[
\boxed{
\mathcal I
=
\mathcal I(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J
).
}
\]

A G1 result must always specify:

```text
data
representation
additive class
joint class
```

---

# 46. Representation Dependence

The MOSEI processed-vs-canonical result already shows:

\[
\phi
\]

can change G1.

v5.6 now tests whether:

\[
\mathcal H_J
\]

also changes G1.

Together this gives two experimentally grounded axes:

\[
\boxed{
\text{representation dependence}
}
\]

and:

\[
\boxed{
\text{hypothesis-class dependence}.
}
\]

---

# 47. Potential Strong Paper Finding

If MELD remains negative under both product and MLP:

> Interaction identifiability is not rescued merely by replacing structured multiplicative prediction with generic nonlinear prediction.

This is a strong response to reviewer criticism.

---

# 48. Alternative Strong Finding

If generic MLP succeeds where product fails:

> Natural interaction identifiability is strongly hypothesis-class dependent.

This is also valuable.

Either outcome is scientifically useful.

---

# 49. No Task G2 in v5.6

Do NOT rerun task-headroom models.

v5.6 concerns:

\[
\boxed{
G_1\text{ only}.
}
\]

G2 is already sufficiently characterized for the current datasets.

---

# 50. No JAD in v5.6

Even if:

\[
J_M>0,
\]

do NOT immediately train JAD.

First:

1. confirm hypothesis-class result;
2. decide whether JAD target definition must change;
3. open v5.7 explicitly.

---

# 51. Why JAD Cannot Directly Use \(q_M\) Yet

Canonical JAD currently assumes:

\[
d=q_P-q_A.
\]

Replacing:

\[
q_P
\]

with:

\[
q_M
\]

changes the meaning of the interaction target.

That is a method revision.

It requires a separate version and synthetic validation.

---

# 52. No Synthetic Reopening Yet

Do not rerun IPIB during H0.

Synthetic reopening is permitted only if:

\[
\boxed{
\text{Outcome B is confirmed}.
}
\]

Then synthetic can test whether a revised joint class preserves known ground-truth interaction behavior.

---

# 53. No New Dataset Search

Do NOT run:

```text
UR-FUNNY
IEMOCAP
MOSI
CMU-MOSEI variants
AV-MNIST variants
```

during v5.6.

Current natural evidence is enough for the hypothesis-class question.

---

# 54. Why Dataset Search Is Blocked

Continuing to search for a positive Type-IV setting before ruling out model-class limitations would confound:

\[
\text{dataset effect}
\]

with:

\[
\text{predictor-family effect}.
\]

v5.6 isolates the latter.

---

# 55. Main H0 Experiment Name

Use:

\[
\boxed{
\textbf{H0 — Joint Hypothesis-Class Sufficiency Audit}
}
\]

---

# 56. Recommended Code Structure

Create:

```text
src/experiments/hypothesis_class/
    common.py
    joint_mlp.py
    mosei_h0.py
    meld_h0.py
    h0_summary.py
```

Reuse existing additive/product predictors.

---

# 57. Generic MLP Utility

Implement shared:

```text
GenericJointMLP
```

not dataset-specific versions.

---

# 58. Generic MLP Input

Input:

\[
[r_i,r_j].
\]

Do not manually add:

```text
product
difference
absolute difference
outer product
```

because that would create another interaction family.

---

# 59. Output

Output dimension exactly equals:

\[
dim(r_k).
\]

Use the same regression target as:

\[
q_A,q_P.
\]

---

# 60. Standardization

Use the same train-only normalized representations as prior G1 experiments.

Do not recompute a new preprocessing pipeline.

---

# 61. Data Split

Use exact previous splits.

No resplitting.

---

# 62. MOSEI Artifact Reuse

Reuse canonical pooled cache where possible.

Do not regenerate unless required for reproducibility.

---

# 63. MELD Artifact Reuse

Reuse frozen utterance-level MELD cache from v5.5.

No context features.

---

# 64. Required MOSEI Artifact

Create:

```text
results/hypothesis_class/v56/mosei/
    MOSEI_H0_HYPOTHESIS_CLASS_REPORT.md
```

and:

```text
mosei_h0.json
```

---

# 65. Required MELD Artifact

Create:

```text
results/hypothesis_class/v56/meld/
    MELD_H0_HYPOTHESIS_CLASS_REPORT.md
```

and:

```text
meld_h0.json
```

---

# 66. Required Combined Artifact

Create:

```text
results/hypothesis_class/v56/
    H0_JOINT_HYPOTHESIS_CLASS_DECISION.md
```

and:

```text
h0_summary.json
```

---

# 67. Required JSON Fields

For every:

```text
dataset × mapping × seed × predictor
```

save:

```text
dataset
mapping
seed
predictor_type
parameter_count
best_epoch
train_mse
val_mse
test_mse
train_r2
val_r2
test_r2
```

Also save:

```text
J_product
J_mlp
delta_H
```

for each seed.

---

# 68. Required Parameter Audit Table

| Dataset | Mapping | Add Params | Product Params | MLP Params |
|---|---|---:|---:|---:|

Include percentage mismatch.

---

# 69. Required Main Result Table

| Dataset | Mapping | \(J_P\) val | \(J_M\) val | \(\Delta_H\) | Decision |
|---|---|---:|---:|---:|---|

Use:

\[
mean\pm sample\ std.
\]

---

# 70. Test Reporting

Also report test:

\[
J_P
\]

and:

\[
J_M.
\]

But do not use them for H0 decision.

---

# 71. MOSEI Expected Role

MOSEI is positive control, not a pass/fail target.

Check whether generic MLP can expose non-additive predictive structure in the known positive setting.

---

# 72. MELD Expected Role

MELD is the critical sufficiency test.

The main decision depends on whether generic nonlinear access changes the previous G1 conclusion.

---

# 73. H0 PASS for Sufficiency

Define:

\[
\boxed{
\text{SUFFICIENCY_SUPPORTED}
}
\]

if MELD remains below the gate under generic MLP across all mappings.

Meaning:

> Product-specific failure is not sufficient to explain MELD G1 failure.

---

# 74. H0 FAIL for Sufficiency

Define:

\[
\boxed{
\text{PRODUCT_CLASS_INSUFFICIENT}
}
\]

if at least one MELD mapping satisfies:

\[
J_{M,val}>0.01
\]

for:

\[
3/3
\]

seeds while canonical \(J_P\) failed.

---

# 75. H0 Inconclusive

Define:

\[
\boxed{
\text{HYPOTHESIS_CLASS_INCONCLUSIVE}
}
\]

if generic MLP results are strongly seed-sensitive or near-threshold.

Do not tune architecture to resolve it.

---

# 76. No Architecture Sweep

Do NOT try:

```text
1 layer
2 layers
3 layers
ReLU
GELU
SiLU
wide MLP
deep MLP
residual MLP
```

to find a positive result.

Exactly one generic MLP family.

---

# 77. No Width Sweep

Hidden width is derived once from capacity matching.

Do not sweep:

\[
h.
\]

---

# 78. No Optimizer Sweep

One optimizer setup.

If optimization catastrophically fails, fix implementation bugs only.

Do not optimize for better \(J_M\).

---

# 79. Training Health Audit

For each predictor report:

```text
train R²
validation R²
best epoch
gradient/loss finite
```

This distinguishes:

\[
\text{model incapacity}
\]

from:

\[
\text{optimization failure}.
\]

---

# 80. Overfit Diagnostic

Define:

\[
Gap_M
=
J_{M,\text{train}}
-
J_{M,\text{val}}.
\]

Large positive:

\[
Gap_M
\]

suggests generic nonlinear overfitting.

---

# 81. Positive MLP but Negative Generalization

If:

\[
J_{M,\text{train}}>0
\]

and:

\[
J_{M,\text{val}}\le0,
\]

the conclusion remains:

\[
\boxed{
\text{no demonstrated natural identifiability}.
}
\]

---

# 82. Do Not Use Training Advantage as Evidence

This rule is strict.

Natural interaction claim requires validation generalization.

---

# 83. Statistical Reporting

For three seeds report:

```text
mean
sample std
per-seed values
95% CI if desired
```

Do not over-interpret p-values with \(n=3\).

---

# 84. Main Reviewer Objection Addressed

Reviewer:

> Your G1 may fail simply because the multiplicative joint predictor is too restrictive.

v5.6 response:

> We compare it against a parameter-controlled generic nonlinear joint predictor while preserving representations, splits, optimization, and additive baseline.

---

# 85. If Both Joint Classes Agree

This supports the interpretation that G1 result is not an artifact of one specific joint architecture.

Do not claim universality across all function classes.

---

# 86. If They Disagree

Then identifiability is demonstrably hypothesis-class dependent.

This becomes part of the main framework.

---

# 87. Updated Theory

Define operational interaction advantage for class:

\[
\mathcal H_J:
\]

\[
\boxed{
J_{\mathcal H_J}
=
R^2(
q^*_{\mathcal H_J},
r_k
)
-
R^2(
q^*_{\mathcal H_A},
r_k
).
}
\]

---

# 88. Multiple Joint Classes

For:

\[
\mathcal H_P
\]

and:

\[
\mathcal H_M,
\]

compare:

\[
J_P
\]

and:

\[
J_M.
\]

This makes hypothesis-class dependence explicit.

---

# 89. Identifiability Is Operational

Do NOT claim:

\[
J_M=0
\Rightarrow
\text{no interaction exists}.
\]

Correct:

> No interaction advantage is identifiable under the tested representation and predictor classes.

---

# 90. Representation-Hypothesis Pair

The experimental object is:

\[
\boxed{
(\phi,\mathcal H_J).
}
\]

Not just the dataset.

---

# 91. Main Framework After v5.6

The paper may characterize interaction learning as depending on:

\[
\boxed{
\mathcal D
}
\]

\[
\boxed{
\phi
}
\]

\[
\boxed{
\mathcal H_A
}
\]

\[
\boxed{
\mathcal H_J
}
\]

\[
\boxed{
Y
}
\]

and downstream access.

---

# 92. Natural Findings So Far

MOSEI:

\[
\text{representation can change G1}.
\]

MELD/MUStARD:

\[
\text{natural G1 can be absent}.
\]

v5.6 tests:

\[
\text{whether G1 also changes with }\mathcal H_J.
\]

---

# 93. Paper Consolidation Trigger

If H0 returns:

\[
\boxed{
\text{SUFFICIENCY_SUPPORTED},
}
\]

immediately freeze experimental method development.

Focus on:

```text
paper structure
theory
figures
ablation summary
natural-regime table
limitations
supplement
```

---

# 94. Paper Direction Under Sufficiency Supported

Primary positioning:

\[
\boxed{
\textbf{diagnostic framework for multimodal interaction learning}
}
\]

with JAD as one explicit discovery mechanism.

Do not position primarily as:

> new SOTA multimodal classifier.

---

# 95. Paper Core Story

Potential narrative:

1. Standard higher-order fusion assumes interaction.
2. Interaction may not be identifiable under a given representation.
3. Identifiability is hypothesis-class relative.
4. Even identifiable interaction may not be task relevant.
5. Even faithful interaction representation may not be accessible downstream.
6. JAD exposes these distinctions in controlled and natural settings.

---

# 96. Paper Main Figure

Potential hierarchy:

```text
Multimodal inputs
      |
      v
Representation phi
      |
      v
Can H_joint beat H_add?
      |
      v
Cross-modal Identifiability
      |
      v
Does task joint model beat task additive?
      |
      v
Task Relevance
      |
      v
Can interaction target be represented?
      |
      v
Fidelity
      |
      v
Does it improve the task?
      |
      v
Accessibility
```

---

# 97. Paper Natural Table

Maintain:

| Dataset | Representation | Pair | Joint class | G1 | G2 | JAD result |
|---|---|---|---|---|---|---|
| MOSEI | canonical raw | VA | Product | Pass | Inc. | no task gain |
| MUStARD | pooled | VA/VT/AT | Product | Fail | fail/unstable | not run |
| MELD | utterance pooled | VA/VT/AT | Product | Fail | fail/unstable | not run |
| MELD | utterance pooled | ... | Generic MLP | H0 | — | — |

---

# 98. If Outcome B Happens

Do not discard the framework.

Instead update it:

\[
\boxed{
\text{interaction identifiability is strongly joint-class dependent}.
}
\]

This is potentially a stronger scientific result.

---

# 99. v5.7 Trigger

Only:

\[
\boxed{
\text{PRODUCT_CLASS_INSUFFICIENT}
}
\]

permits v5.7.

No other outcome opens method revision.

---

# 100. v5.7 Scope

If opened, v5.7 must answer:

> What is the minimal justified joint class that preserves controlled interaction recovery while improving natural identifiability?

Do not start broad architecture exploration.

---

# 101. Third-Order Status

Still:

\[
\boxed{
\textbf{BLOCKED}.
}
\]

No \(h_{123}\).

---

# 102. Task-Aware JAD Status

Still:

\[
\boxed{
\textbf{BLOCKED}.
}
\]

No task labels inside canonical JAD.

---

# 103. Purification Status

Closed.

Do not reopen additive purification.

---

# 104. Reliability Weighting Status

Closed.

Do not reopen C1/C2.

---

# 105. Loss Calibration Status

Closed.

Do not reopen B-branch loss tuning.

---

# 106. Dataset Search Status

Blocked until H0 concludes.

Do not search for another positive benchmark.

---

# 107. Current Permitted Work

Only:

```text
H0 predictor implementation
MOSEI H0
MELD H0
capacity audit
optimization health audit
paper theory
paper tables
paper figures
```

---

# 108. Current Forbidden Work

```text
NO new dataset
NO JAD retraining
NO new JAD target
NO task-aware JAD
NO purifier
NO rank sweep
NO context
NO temporal encoder
NO cross-attention
NO deep MLP sweep
NO third-order
```

---

# 109. Required Unit Tests

Add tests for:

```text
GenericJointMLP input concatenation
correct target dimension
parameter count
capacity mismatch
same split as prior G1
same normalization
validation-only checkpointing
deterministic seed
no task labels
no test-based selection
```

---

# 110. Reproducibility

Seed:

```text
Python
NumPy
PyTorch
CUDA
DataLoader
initialization
```

where applicable.

---

# 111. Smoke Test

First run:

```text
MOSEI VA->T seed 1
MELD VA->T seed 1
```

only for implementation validation.

Check:

```text
finite loss
R² computation
parameter counts
checkpointing
determinism
```

Do not interpret smoke performance.

---

# 112. Full H0 Execution

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

# 113. Decision Table

At completion classify:

```text
SUFFICIENCY_SUPPORTED

PRODUCT_CLASS_INSUFFICIENT

HYPOTHESIS_CLASS_INCONCLUSIVE
```

Exactly one overall decision.

---

# 114. Decision Rule

### SUFFICIENCY_SUPPORTED

No MELD mapping satisfies:

\[
J_{M,val}>0.01
\]

in all 3 seeds.

Generic MLP shows no reliable rescue.

---

# 115. PRODUCT_CLASS_INSUFFICIENT

At least one MELD mapping satisfies:

\[
J_{M,val}>0.01
\]

for:

\[
3/3
\]

seeds,

while canonical product failed.

Requires frozen five-seed confirmation before method revision.

---

# 116. HYPOTHESIS_CLASS_INCONCLUSIVE

Use when:

```text
MLP results cross threshold inconsistently
high seed variance
optimization instability
```

and cannot be resolved without tuning.

Do not tune.

---

# 117. Main v5.6 Milestone

\[
\boxed{
\textbf{
Determine whether natural G1 failure is robust to a generic nonlinear joint hypothesis class.
}
}
\]

---

# 118. Strongest Outcome A

If generic MLP also fails on MELD:

\[
\boxed{
\text{the natural negative result is not specific to product interaction}.
}
\]

This supports framework consolidation.

---

# 119. Strongest Outcome B

If generic MLP succeeds:

\[
\boxed{
\text{interaction identifiability is strongly hypothesis-class dependent}.
}
\]

This justifies v5.7.

---

# 120. Scientific Success Definition

v5.6 succeeds if it resolves the hypothesis-class objection.

It does NOT require:

```text
higher accuracy
Type IV dataset
new JAD method
```

---

# 121. Updated Main Thesis

\[
\boxed{
\textbf{
Multimodal interaction identifiability is conditional on the data distribution, representation, and predictor hypothesis class; it must be established before downstream interaction learning is justified.
}
}
\]

---

# 122. Operational Principle

\[
\boxed{
\textbf{
Do not interpret a failed interaction screen until representation and joint-class confounds have been audited.
}
}
\]

---

# 123. Immediate Execution Order

```text
1. Freeze existing MOSEI and MELD caches.

2. Implement GenericJointMLP.

3. Solve hidden width for capacity parity.

4. Verify Additive/Product/MLP parameter counts.

5. Run MOSEI VA->T seed-1 smoke.

6. Run MELD VA->T seed-1 smoke.

7. Verify optimization health.

8. Run MOSEI VA->T seeds 1..3.

9. Run MELD:
   VA->T
   VT->A
   AT->V
   seeds 1..3.

10. Compute:
    J_Product
    J_MLP
    Delta_H.

11. Produce per-seed tables.

12. Produce combined H0 report.

13. Classify:
    SUFFICIENCY_SUPPORTED
    PRODUCT_CLASS_INSUFFICIENT
    or HYPOTHESIS_CLASS_INCONCLUSIVE.

14. Do not modify JAD before this decision.

15. If SUFFICIENCY_SUPPORTED:
    freeze method experiments;
    move to paper consolidation.

16. If PRODUCT_CLASS_INSUFFICIENT:
    confirm 5 seeds;
    then open v5.7.

17. If INCONCLUSIVE:
    do not tune;
    report limitation and consolidate.
```

---

# 124. Final v5.6 Rule

\[
\boxed{
\textbf{
One hypothesis-class audit.
No architecture fishing.
No dataset fishing.
Let the result determine whether method development continues.
}
}
\]

---

# 125. Next Action

\[
\boxed{
\textbf{
Run H0 — Joint Hypothesis-Class Sufficiency Audit on MOSEI and MELD.
}
}
\]

No further ConFu++ method development is permitted before H0 is complete.