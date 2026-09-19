# AGENT.md

# ConFu++ v5.2

## Conditional Selectivity, Shortcut Localization, and Purified Predictive Interaction

---

# 0. Mission

This repository develops **ConFu++**, an extension of higher-order multimodal alignment toward explicit multimodal interaction discovery.

The project is targeting a **CVPR main-track paper**.

The synthetic pair-discovery phase is frozen.

CMU-MOSEI has now provided the first natural setting that passes the pre-registered identifiability screen:

\[
\boxed{
V+A\rightarrow T
}
\]

under the exact canonical raw representation.

However:

\[
\boxed{
\text{identifiability pass}
}
\]

did not produce:

\[
\boxed{
\text{downstream sentiment accessibility gain}.
}
\]

The new research question is:

\[
\boxed{
\textbf{Does Joint-Advantage Distillation still contain lower-order modality shortcuts after the additive-vs-joint subtraction?}
}
\]

v5.2 exists to answer this question before any new architecture is proposed.

---

# 1. Current Project Status

```text
Synthetic architecture capability:
PASS

Oracle pair interaction:
PASS

Oracle third-order interaction:
PASS

NIC non-identifiable control:
PASS

IPIB identifiability:
PASS

Canonical D2 / JAD:
FROZEN

Fidelity–Accessibility tradeoff:
ESTABLISHED

CMU-MOSEI processed pooled R1:
FAIL / LIMITED

CMU-MOSEI canonical raw R1:
PASS

Canonical natural mapping:
VA -> T

CMU-MOSEI R2:
COMPLETE

D2 target fit:
PASS

D2 source dependence:
PASS EXPLORATORY

D2 > lower-order accuracy:
FAIL

D2 modality shortcut:
FAIL

Shortcut localization:
NEXT

Conditional purification:
CONDITIONAL

Third-order discovery:
BLOCKED

UR-FUNNY tuning:
BLOCKED
```

---

# 2. Canonical Natural Setting

The frozen MOSEI setting is:

```text
dataset:
CMU-MOSEI canonical raw

representation:
masked pooled raw features

vision:
713 dimensions

audio:
74 dimensions

text:
300 dimensions

mapping:
vision + audio -> text

interaction rank:
64

screening seeds:
1..3

R2 seeds:
1..5

JAD loss:
cosine regression
```

Do not change this setting during shortcut localization.

---

# 3. Why the Old Processed MOSEI Result Is Limited

The old processed file:

```text
mosei_senti_data.pkl
```

contains timestamp-based identifiers.

The canonical raw file:

```text
mosei_raw.pkl
```

contains canonical:

```text
video[index]
```

IDs.

Literal ID intersection is:

\[
0.
\]

Therefore the old processed emotion-conditioned experiment cannot be described as exactly ID-aligned.

---

# 4. Canonical-Source Sensitivity

The canonical representation differs from the processed representation.

Old vision dimension:

\[
35.
\]

Canonical raw vision dimension:

\[
713.
\]

Therefore the shift:

\[
J_{VA\to T}:
-0.0111
\rightarrow
+0.0198
\]

must be described as:

\[
\boxed{
\text{source / representation sensitivity}
}
\]

rather than a pure alignment-only effect.

---

# 5. Canonical R1 Result

Validation joint predictive advantage:

\[
J_{VA\rightarrow T}
=
0.0198\pm0.0019.
\]

Individual seeds:

\[
0.02196,
\quad
0.01905,
\quad
0.01847.
\]

Thus:

\[
\boxed{
J_{val}>0.01
}
\]

for:

\[
3/3
\]

screening seeds.

The frozen mapping is therefore:

\[
\boxed{
VA\rightarrow T.
}
\]

---

# 6. Meaning of R1 PASS

R1 PASS means:

> under the canonical raw MOSEI representation and the tested capacity-matched predictor classes, a joint predictor explains more text-representation variance than an additive predictor.

It does NOT mean:

```text
the discovered structure is pure interaction
the representation is task useful
the interaction is causal
the interaction is PID synergy
```

---

# 7. Frozen JAD Definition

For:

\[
r_V,r_A,r_T,
\]

define additive predictor:

\[
q_A
=
q_V(r_V)
+
q_A^{audio}(r_A).
\]

To avoid symbol ambiguity in code, use explicit names such as:

```text
q_add
q_joint
```

rather than overloading \(q_A\).

---

# 8. Joint Predictor

Train:

\[
q_J(r_V,r_A)
\]

with capacity approximately matched to:

\[
q_{add}.
\]

The natural R1 capacity mismatch remains below 1%.

---

# 9. Joint Predictive Advantage

Define:

\[
\boxed{
J
=
R^2(q_J,r_T)
-
R^2(q_{add},r_T).
}
\]

---

# 10. Canonical D2 Target

Define:

\[
\boxed{
d
=
q_J(r_V,r_A)
-
q_{add}(r_V,r_A).
}
\]

This remains the canonical:

\[
\boxed{
\text{Joint-Advantage Distillation target}.
}
\]

---

# 11. Canonical D2 Interaction Representation

Use:

\[
u_V=LN(U_Vr_V)
\]

\[
u_A=LN(U_Ar_A)
\]

\[
h_{VA}
=
W
\left(
\frac{
u_V\odot u_A
}{
\sqrt{64}
}
\right).
\]

Train:

\[
h_{VA}\leftrightarrow d
\]

using canonical cosine regression.

---

# 12. MOSEI R2 Downstream Result

Lower-order representation:

\[
Z_{low}=[r_V,r_A].
\]

Test accuracy:

\[
64.397\pm0.231.
\]

D0:

\[
63.930\pm0.425.
\]

D1:

\[
63.475\pm0.472.
\]

D2:

\[
64.363\pm0.288.
\]

Thus:

\[
\boxed{
\Delta_{D2}^{acc}
=
-0.034\pm0.099\text{ pp}.
}
\]

---

# 13. Macro F1 Result

Lower-order:

\[
64.370\pm0.232.
\]

D2:

\[
64.318\pm0.280.
\]

Again:

\[
\boxed{
\text{no measurable downstream gain}.
}
\]

---

# 14. AUC Result

Lower-order AUC:

\[
67.204.
\]

D2 AUC:

\[
68.348.
\]

This suggests the D2 representation may affect ranking quality.

However:

\[
\boxed{
\text{AUC improvement does not imply fixed-threshold classification improvement}.
}
\]

Do not claim downstream superiority from AUC alone.

---

# 15. D2 Target Fidelity

D2 target cosine:

\[
\boxed{
0.848\pm0.038.
}
\]

This is much higher than D0/D1 target fit.

Thus:

\[
\boxed{
\text{D2 successfully distills its own JAD target}.
}
\]

---

# 16. D2 Representation Health

Effective rank:

\[
25.66\pm1.36.
\]

The representation is not collapsed.

Therefore:

\[
\boxed{
\text{collapse is not the current bottleneck}.
}
\]

---

# 17. Source-Shuffle Dependence

D2:

\[
VisionShuffleDrop
=
+3.132\pm1.636\text{ pp}
\]

\[
AudioShuffleDrop
=
+3.591\pm0.558\text{ pp}.
\]

Both source modalities affect the learned representation.

This establishes:

\[
\boxed{
\text{source dependence}
}
\]

but not:

\[
\boxed{
\text{pure conditional interaction}.
}
\]

---

# 18. Shortcut Audit

Single-modality reconstruction of:

\[
h_{D2}
\]

gives:

\[
R^2(V\rightarrow h_{D2})
=
0.193\pm0.030
\]

and:

\[
\boxed{
R^2(A\rightarrow h_{D2})
=
0.302\pm0.048.
}
\]

This is the current central bottleneck.

---

# 19. New Scientific Distinction

The project must now distinguish:

\[
\boxed{
\text{joint dependence}
}
\]

from:

\[
\boxed{
\text{conditional interaction selectivity}.
}
\]

A representation may depend on both modalities while still being substantially reconstructible from one modality alone.

---

# 20. Updated Four-Axis Framework

From v5.2 onward evaluate:

\[
\boxed{
\text{Identifiability}
}
\]

\[
\boxed{
\text{Conditional Selectivity}
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

# 21. Identifiability

Operational question:

> Does a capacity-matched joint predictor outperform an additive predictor?

Measured by:

\[
J.
\]

MOSEI canonical VA→T:

\[
\boxed{\text{PASS}}.
\]

---

# 22. Conditional Selectivity

Operational question:

> Is the purported interaction representation difficult to reconstruct from either source modality independently or from an additive combination of source-specific functions?

This is now missing.

---

# 23. Fidelity

Operational question:

> Does the learned representation match the chosen interaction target?

D2 target cosine:

\[
0.848.
\]

Thus:

\[
\boxed{\text{PASS}}.
\]

---

# 24. Accessibility

Operational question:

> Does adding the interaction representation improve downstream task performance over lower-order evidence?

MOSEI sentiment:

\[
\boxed{\text{FAIL}}.
\]

---

# 25. Current Failure Chain

Current MOSEI evidence is:

```text
Identifiable:
YES

Target fidelity:
YES

Both-source dependence:
YES

Single-modality leakage:
YES

Task accessibility gain:
NO
```

Therefore:

\[
\boxed{
\text{joint predictive advantage alone is insufficient}.
}
\]

---

# 26. Immediate Research Question

The next experiment must determine:

\[
\boxed{
\textbf{where does the single-modality shortcut appear?}
}
\]

Possibilities:

### H1

Shortcut already exists in:

\[
d=q_J-q_{add}.
\]

### H2

Target \(d\) is relatively selective, but shortcut emerges during:

\[
d\rightarrow h.
\]

These hypotheses require different solutions.

---

# 27. v5.2-S0 — Shortcut Localization Audit

This is the mandatory next experiment.

Do NOT train a new D2 variant first.

Use existing frozen five-seed MOSEI models.

---

# 28. S0 Representations

Audit:

\[
q_{add}
\]

\[
q_J
\]

\[
d=q_J-q_{add}
\]

\[
h_{D2}.
\]

Optionally also include:

\[
r_T
\]

as a reference.

---

# 29. Source-Specific Reconstruction

For every representation:

\[
z,
\]

train simple frozen probes:

\[
p_V(r_V)\rightarrow z
\]

and:

\[
p_A(r_A)\rightarrow z.
\]

Measure:

\[
\boxed{
R^2_V(z)
}
\]

and:

\[
\boxed{
R^2_A(z).
}
\]

---

# 30. Additive Reconstruction

Train independent source predictors:

\[
f_V(r_V)
\]

and:

\[
f_A(r_A).
\]

Define:

\[
\hat z_{add}
=
f_V(r_V)+f_A(r_A).
\]

Measure:

\[
\boxed{
R^2_{add}(z).
}
\]

This is a critical new diagnostic.

---

# 31. Joint Reconstruction Reference

Optionally train a capacity-matched joint reconstruction probe:

\[
f_J(r_V,r_A)\rightarrow z.
\]

Measure:

\[
R^2_J(z).
\]

Then define diagnostic:

\[
\boxed{
J_z
=
R^2_J(z)
-
R^2_{add}(z).
}
\]

This asks whether the learned representation itself contains non-additive source structure.

---

# 32. Required S0 Table

Produce:

| Representation | V-only \(R^2\) | A-only \(R^2\) | Additive V+A \(R^2\) | Joint V+A \(R^2\) | \(J_z\) |
|---|---:|---:|---:|---:|---:|
| \(q_{add}\) | | | | | |
| \(q_J\) | | | | | |
| \(d\) | | | | | |
| \(h_{D2}\) | | | | | |

Report five-seed mean ± sample std.

---

# 33. Probe Capacity

Single-modality and additive reconstruction probes must be simple and capacity controlled.

Do not use an extremely powerful network that can memorize the dataset.

Preferred:

```text
linear Ridge
small fixed MLP
```

Primary metric should use a simple probe.

---

# 34. Validation-Only Probe Selection

Any regularization or early stopping must be selected on validation only.

Test remains reporting only.

---

# 35. S0 Decision Case A

If:

\[
R^2_A(d)
\]

is already large and close to:

\[
R^2_A(h),
\]

then:

\[
\boxed{
\text{shortcut exists at target-construction level}.
}
\]

The main issue is:

\[
\boxed{
q_J-q_{add}
\neq
\text{conditionally pure interaction}.
}
\]

---

# 36. S0 Decision Case B

If:

\[
R^2_A(d)
\]

is small but:

\[
R^2_A(h)
\]

is large:

\[
\boxed{
\text{shortcut is introduced or amplified during distillation}.
}
\]

Then do NOT purify the target.

Investigate representation learning geometry instead.

---

# 37. S0 Decision Case C

If both:

\[
R^2_A(d)
\]

and:

\[
R^2_A(h)
\]

are high, but:

\[
J_d>0,
\]

then the target contains both:

```text
joint structure
+
lower-order leakage
```

This is likely the most interesting case.

---

# 38. Do Not Assume Audio Is the Only Shortcut

Although current audit shows stronger audio reconstruction, S0 must treat vision and audio symmetrically.

Measure both.

---

# 39. Shortcut Terminology

Preferred:

```text
single-modality leakage
lower-order leakage
additive reconstructability
conditional selectivity
```

Avoid:

```text
causal shortcut
spurious causal feature
```

unless causality is established.

---

# 40. S0 Required Artifact

Create:

```text
results/mosei/v52/
    S0_SHORTCUT_LOCALIZATION.md
```

and:

```text
s0_shortcut_localization.json
```

---

# 41. S0 Hard Gate

Do NOT implement purification until S0 establishes that:

\[
d
\]

itself contains substantial lower-order reconstructability.

This is mandatory.

---

# 42. Conditional Method Branch

If S0 confirms target-level leakage, open:

\[
\boxed{
\text{v5.2-P1 — Additive Purification}
}
\]

This is the only permitted new method-level branch.

---

# 43. Motivation for P1

Current JAD assumes:

\[
q_J-q_{add}
\]

isolates joint structure.

But in finite function classes:

\[
q_J
\]

and:

\[
q_{add}
\]

may represent lower-order effects differently.

Therefore their subtraction may retain:

\[
\boxed{
\text{main-effect-like structure}.
}
\]

---

# 44. Additive Purification

Let canonical JAD target be:

\[
d.
\]

Fit source-specific functions:

\[
f_V(r_V)
\]

and:

\[
f_A(r_A)
\]

to predict:

\[
d.
\]

Define:

\[
\hat d_{add}
=
f_V(r_V)+f_A(r_A).
\]

Then define purified target:

\[
\boxed{
d^\perp
=
d-\hat d_{add}.
}
\]

---

# 45. Interpretation of \(d^\perp\)

\(d^\perp\) is intended to represent:

\[
\boxed{
\text{the part of JAD that is not reconstructible by the chosen additive source model}.
}
\]

Do NOT call it exact:

```text
ANOVA interaction
PID synergy
causal interaction
```

without proof.

Preferred term:

\[
\boxed{
\text{Additively Purified Joint Advantage}
}
\]

or:

\[
\boxed{
\text{Conditionally Selective JAD}.
}
\]

---

# 46. P1 Must Be Label-Free

Purification models:

\[
f_V,f_A
\]

may use:

```text
r_V
r_A
d
```

only.

They must not use:

```text
sentiment
emotion labels
task score
test accuracy
```

---

# 47. Train-Only Purification

Fit:

\[
f_V,f_A
\]

on training data only.

Use validation for model selection.

Freeze before test.

---

# 48. No Test-Based Purification

Never choose purification:

```text
strength
architecture
regularization
```

using test shortcut or test task accuracy.

---

# 49. P1 Target Audit Before Distillation

Before training:

\[
h^\perp,
\]

measure:

\[
R^2_V(d^\perp)
\]

\[
R^2_A(d^\perp)
\]

\[
R^2_{add}(d^\perp).
\]

Compare against canonical:

\[
d.
\]

---

# 50. P1 Required Selectivity Effect

Desired:

\[
\boxed{
R^2_A(d^\perp)
<
R^2_A(d)
}
\]

and:

\[
\boxed{
R^2_V(d^\perp)
<
R^2_V(d).
}
\]

More importantly:

\[
\boxed{
R^2_{add}(d^\perp)
\ll
R^2_{add}(d).
}
\]

---

# 51. P1 Health Gate

Purification must not collapse the target.

Measure:

```text
variance
mean norm
effective rank
fraction near-zero dimensions
```

Reject if:

\[
d^\perp
\]

is trivially zero.

---

# 52. P1 Joint-Structure Gate

Also measure joint reconstruction:

\[
R^2_J(d^\perp).
\]

Desired:

\[
R^2_J(d^\perp)
>
R^2_{add}(d^\perp).
\]

Define:

\[
\boxed{
J_{d^\perp}
=
R^2_J(d^\perp)
-
R^2_{add}(d^\perp).
}
\]

This should remain positive.

---

# 53. P1 Is Not Automatically Better

A lower:

\[
R^2_{add}
\]

does not guarantee useful interaction.

It may simply remove information.

Therefore purification must be evaluated along:

```text
selectivity
health
joint reconstructability
synthetic fidelity
downstream accessibility
```

---

# 54. Synthetic Reopening Rule

The synthetic pair phase was frozen.

However natural MOSEI has exposed a new failure mechanism:

\[
\boxed{
\text{lower-order leakage inside JAD}.
}
\]

Therefore synthetic experiments may be reopened exactly once to test this new hypothesis.

This is scientifically allowed.

---

# 55. Synthetic P1 Validation

Return to IPIB only after target-level MOSEI P1 audit passes.

Compare:

\[
D2
\]

vs:

\[
D2^\perp.
\]

Do NOT reopen D0/D1 tuning.

---

# 56. Synthetic P1 Questions

Test:

1. Does purification reduce single-modality reconstructability?
2. Does it preserve interaction recovery?
3. Does it preserve I0 false-positive control?
4. Does it preserve linear accessibility?
5. Does it help weak regimes?

---

# 57. Synthetic I0 Hard Gate

Require:

\[
\boxed{
\Delta_{I0}=0.
}
\]

Purification must not break the strongest property of canonical D2.

---

# 58. Synthetic Fidelity Gate

Require:

\[
R^2(h^\perp\rightarrow g_{12})
\]

to remain close to canonical D2.

Do not accept selectivity improvement by destroying ground-truth interaction recovery.

---

# 59. Synthetic Shortcut Audit

Measure:

\[
R^2(z_1\rightarrow d)
\]

\[
R^2(z_2\rightarrow d)
\]

and:

\[
R^2(z_1\rightarrow d^\perp)
\]

\[
R^2(z_2\rightarrow d^\perp).
\]

---

# 60. Synthetic Accessibility

Compare:

\[
\Delta^{linear}_{D2}
\]

and:

\[
\Delta^{linear}_{D2^\perp}.
\]

Do not require purification to exceed D1 automatically.

---

# 61. Synthetic P1 Seed Protocol

Exploratory:

\[
seed=1.
\]

If mechanism passes:

\[
5\text{ seeds}.
\]

No parameter sweep.

---

# 62. P1 Purification Model Capacity

Use a small predefined additive purifier.

Preferred:

\[
f_V+f_A
\]

with matched branch capacity.

Do NOT sweep large purifier architectures.

---

# 63. No Purification Strength Hyperparameter Initially

Initial target must be:

\[
\boxed{
d^\perp=d-\hat d_{add}.
}
\]

Do not use:

\[
d-\lambda\hat d_{add}
\]

with a broad \(\lambda\) sweep.

---

# 64. MOSEI P2 Trigger

Only if P1 passes synthetic mechanism validation:

return to canonical MOSEI and train:

\[
h^\perp
\leftrightarrow d^\perp.
\]

This becomes:

\[
\boxed{
P2\text{ — Purified JAD Distillation}.
}
\]

---

# 65. MOSEI P2 Frozen Setting

Keep:

```text
canonical raw MOSEI
VA -> T
rank 64
same optimizer
same training epochs
same five seeds
same probe
```

Only target changes:

\[
d
\rightarrow
d^\perp.
\]

---

# 66. MOSEI P2 Main Comparisons

Compare:

```text
Lower-order
D0
D1
D2
D2-purified
```

---

# 67. P2 Selectivity Metric

Primary mechanism metric:

\[
R^2_A(h^\perp)
\]

and:

\[
R^2_V(h^\perp).
\]

Desired:

\[
\boxed{
R^2_A(h^\perp)
<
0.302
}
\]

substantially.

---

# 68. P2 Additive Leakage Metric

Also measure:

\[
R^2_{add}(h^\perp).
\]

This is more important than single-modality probes alone.

---

# 69. P2 Source Dependence

Retain shuffle diagnostics.

Both:

```text
vision shuffle
audio shuffle
```

should still affect:

\[
h^\perp.
\]

If purification removes one modality entirely, reject.

---

# 70. P2 Representation Health

Report:

```text
variance
effective rank
mean norm
dimension std
```

to rule out collapse.

---

# 71. P2 Accessibility

Primary downstream comparison:

\[
\Delta Accuracy
\]

vs lower-order.

Also report:

```text
macro F1
AUC
```

---

# 72. P2 Strong Success

Ideal:

\[
\boxed{
\text{lower additive leakage}
}
\]

and:

\[
\boxed{
\Delta Accuracy>0
}
\]

with:

\[
5\text{-seed consistency}.
\]

---

# 73. P2 Partial Success

Still scientifically meaningful:

\[
\boxed{
\text{large selectivity improvement}
}
\]

with:

\[
\text{accuracy approximately unchanged}
\]

and improved robustness/AUC.

This would establish conditional purification even without classification gain.

---

# 74. P2 Failure

If purification reduces shortcut but does not improve accessibility:

conclude:

\[
\boxed{
\text{conditional selectivity is still insufficient for task accessibility}.
}
\]

This is an important negative result.

Do NOT then start architecture search automatically.

---

# 75. Distillation Branch if S0 Case B

If S0 shows:

\[
d
\]

is selective but:

\[
h
\]

introduces leakage:

do NOT run P1 purification.

Instead classify the problem as:

\[
\boxed{
\text{distillation-induced shortcut}.
}
\]

---

# 76. Distillation-Induced Shortcut Response

First audit:

```text
factor projections
output projection
normalization
source asymmetry
gradient magnitude
```

Do not change architecture immediately.

---

# 77. Factor-Level Diagnostics

For:

\[
u_V=LN(U_Vr_V)
\]

\[
u_A=LN(U_Ar_A),
\]

measure:

\[
R^2(V\rightarrow u_V\odot u_A)
\]

and:

\[
R^2(A\rightarrow u_V\odot u_A).
\]

This can localize where asymmetry appears.

---

# 78. Gradient Audit

Measure average gradient norm through:

```text
vision factor
audio factor
output projection
```

during D2 training.

This is diagnostic only.

---

# 79. Do Not Reopen B2

Even if distillation is implicated:

do NOT immediately return to:

```text
standardized MSE
cosine+MSE
InfoNCE
```

Those branches have already been investigated.

A new loss requires a new mechanistic justification.

---

# 80. Important CVPR Theory Update

The project originally used:

\[
q_J-q_A
\]

as joint predictive interaction.

MOSEI suggests:

\[
\boxed{
q_J-q_A
}
\]

may still retain lower-order structure because:

\[
q_J
\]

and:

\[
q_A
\]

are finite learned functions.

---

# 81. Finite Predictor Decomposition

Suppose ideal:

\[
q_A^*=A
\]

and:

\[
q_J^*=A+J.
\]

Then:

\[
q_J^*-q_A^*=J.
\]

But finite predictors satisfy:

\[
q_A=A+\epsilon_A
\]

\[
q_J=A+J+\epsilon_J.
\]

Therefore:

\[
\boxed{
d
=
J+
(\epsilon_J-\epsilon_A).
}
\]

---

# 82. New Leakage Interpretation

The error difference:

\[
\epsilon_J-\epsilon_A
\]

can contain structure predictable from:

\[
V
\]

or:

\[
A.
\]

Thus:

\[
\boxed{
\text{joint predictive advantage is not guaranteed to be conditionally pure}.
}
\]

---

# 83. Conditional Purification Interpretation

Purification attempts to project:

\[
d
\]

away from the selected additive function space:

\[
\mathcal H_V+\mathcal H_A.
\]

Conceptually:

\[
\boxed{
d^\perp
=
d-
\Pi_{\mathcal H_V+\mathcal H_A}(d).
}
\]

This is the clean theoretical form.

---

# 84. Orthogonality Goal

Ideally:

\[
E[
d^\perp
f_V(V)
]
\approx0
\]

and:

\[
E[
d^\perp
f_A(A)
]
\approx0
\]

for functions in the chosen purifier classes.

Do not claim universal orthogonality outside those classes.

---

# 85. Method Is Hypothesis-Class Relative

Both:

\[
J
\]

and:

\[
d^\perp
\]

depend on:

```text
representation
additive predictor class
joint predictor class
purifier class
```

This must be stated explicitly.

---

# 86. Updated Identifiability Form

Use:

\[
\boxed{
J
=
J(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J
).
}
\]

---

# 87. Updated Selectivity Form

Conditional selectivity additionally depends on:

\[
\mathcal H_P,
\]

the lower-order purifier/probe class.

Therefore:

\[
\boxed{
S
=
S(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J,
\mathcal H_P
).
}
\]

---

# 88. Updated CVPR Thesis

The emerging paper thesis becomes:

\[
\boxed{
\textbf{
Multimodal interaction must be identifiable, conditionally selective, faithfully represented, and downstream-accessible.
}
}
\]

---

# 89. Updated Failure Taxonomy

### Failure I — Non-identifiability

\[
J\approx0.
\]

### Failure II — Low conditional selectivity

Interaction target remains reconstructible from lower-order source functions.

### Failure III — Low fidelity

Representation fails to fit interaction target.

### Failure IV — Low accessibility

Representation is valid but adds no downstream utility.

---

# 90. MOSEI Current Classification

MOSEI canonical VA→T currently shows:

```text
Failure I:
NO

Failure II:
LIKELY

Failure III:
NO

Failure IV:
YES
```

v5.2 determines whether Failure II truly occurs at the target level.

---

# 91. Paper Figure Candidate

Updated conceptual pipeline:

```text
V ---------> q_V ----\
                      \
                       q_add --------\
A ---------> q_A ----/                \
                                       subtract --> d
V,A -------> q_joint ----------------/              |
                                                     |
                                      lower-order audit
                                                     |
                                         if leakage present
                                                     |
                                                     v
                                            additive purifier
                                                     |
                                                     v
                                                  d_perp
                                                     |
                                                     v
                                            interaction h
                                                     |
                                                     v
                                             downstream task
```

---

# 92. Paper Claim If Purification Works

Potential:

> Joint predictive advantage may still contain lower-order leakage in finite models; projecting it away from additive source functions improves conditional selectivity while preserving identifiable interaction structure.

Only make this claim if P1/P2 support it.

---

# 93. Paper Claim If Purification Fails

Still valid:

> Identifiability and source dependence do not guarantee conditionally selective or task-useful interaction representations.

This itself strengthens the framework.

---

# 94. Relation to Functional Decomposition

Discuss functional ANOVA and interaction purification literature.

But position ConFu++ as:

\[
\boxed{
\text{predictive representation learning}
}
\]

not as exact functional ANOVA estimation.

---

# 95. Avoid Overclaiming

Do NOT say:

```text
D2 recovers causal interaction
D2 recovers PID synergy
purified D2 is exact pure interaction
```

Preferred:

```text
predictive interaction
additive leakage
conditional selectivity
hypothesis-class-relative interaction
```

---

# 96. Current Forbidden Work

Until S0 is complete:

```text
NO new target
NO purification training
NO architecture modification
NO rank sweep
NO task loss
NO InfoNCE
NO cross-attention
NO h123
NO UR-FUNNY tuning
NO MUStARD switch
```

---

# 97. Current Permitted Work

Only:

```text
S0 shortcut localization
paper theory update
related-work audit
figure drafting
```

---

# 98. S0 Code Task

Implement:

```text
src/experiments/multibench/mosei_shortcut_localization.py
```

or equivalent.

Reuse frozen MOSEI checkpoints.

---

# 99. Reconstruction Utilities

Add shared:

```text
fit_single_source_probe()
fit_additive_reconstruction_probe()
fit_joint_reconstruction_probe()
```

with consistent train/validation/test protocol.

---

# 100. S0 Reproducibility

Save:

```text
representation name
seed
probe type
parameter count
train R2
val R2
test R2
```

for every reconstruction probe.

---

# 101. S0 Main Artifact

Produce:

```text
results/mosei/v52/S0_SHORTCUT_LOCALIZATION.md
```

and:

```text
results/mosei/v52/s0_shortcut_localization.json
```

---

# 102. S0 Five-Seed Requirement

S0 uses all existing:

\[
5
\]

R2 seeds.

No new D2 retraining should be necessary.

---

# 103. S0 Primary Decision

The report must explicitly choose one:

```text
TARGET_LEVEL_LEAKAGE
DISTILLATION_LEVEL_LEAKAGE
MIXED
INCONCLUSIVE
```

---

# 104. If TARGET_LEVEL_LEAKAGE

Proceed to:

\[
\boxed{
P1\text{ Additive Purification}.
}
\]

---

# 105. If DISTILLATION_LEVEL_LEAKAGE

Do not purify target.

Open a diagnostic-only:

```text
D-branch
```

focused on factor/output geometry.

---

# 106. If MIXED

Prioritize target purification first because leakage already exists before distillation.

Then reassess residual leakage after purified distillation.

---

# 107. If INCONCLUSIVE

Do not invent a method.

Improve diagnostics first.

---

# 108. Hard Stop Against Hyperparameter Search

Any new method must correspond to a localized failure mechanism.

No method is allowed merely because:

```text
accuracy is low
AUC could be higher
another module may help
```

---

# 109. Real-World Dataset Expansion

Do NOT move to MUStARD/UR-FUNNY before S0 is complete.

Reason:

the MOSEI setting now provides a rare natural mapping where:

\[
J>0.
\]

It should be fully exploited mechanistically before searching another dataset.

---

# 110. When Dataset Expansion Resumes

After one of:

```text
P2 completed
or
S0 proves target purification is unnecessary
```

then resume:

```text
MUStARD screening
UR-FUNNY screening
```

---

# 111. Third-Order Status

Still:

\[
\boxed{
\text{BLOCKED}.
}
\]

Do not open order-3 while pair-level conditional selectivity remains unresolved.

---

# 112. CVPR Reviewer Question

Add to:

```text
paper/reviewer_questions.md
```

> Why should q_joint - q_add be considered interaction if it can still be predicted from one modality alone?

v5.2 must answer this.

---

# 113. Another Reviewer Question

> Does using both modalities during training guarantee the resulting representation contains irreducible joint information?

Answer:

\[
\boxed{\text{No.}}
\]

Shuffle dependence is insufficient.

This is now empirically demonstrated.

---

# 114. Important New Insight

The project can now distinguish:

\[
\boxed{
\text{joint computation}
}
\]

\[
\boxed{
\text{joint dependence}
}
\]

\[
\boxed{
\text{joint predictive advantage}
}
\]

\[
\boxed{
\text{conditional selectivity}.
}
\]

These are increasingly strict concepts.

---

# 115. Hierarchy

Conceptually:

```text
joint computation
      ↓
joint dependence
      ↓
joint predictive advantage
      ↓
conditional selectivity
      ↓
downstream accessibility
```

No implication should be assumed automatically.

---

# 116. Historical Evidence Mapping

AV-MNIST showed:

\[
\boxed{
\text{joint computation}
\not\Rightarrow
\text{joint dependence}.
}
\]

MOSEI now suggests:

\[
\boxed{
\text{joint predictive advantage}
\not\Rightarrow
\text{conditional selectivity}.
}
\]

Synthetic/MOSEI jointly show:

\[
\boxed{
\text{fidelity}
\not\Rightarrow
\text{accessibility}.
}
\]

This creates a coherent paper story.

---

# 117. Updated Contribution Candidate

Potential contribution list:

1. A hierarchy distinguishing computation, dependence, identifiability, conditional selectivity, fidelity, and accessibility.

2. IPIB for controlled interaction discovery.

3. JAD for predictable non-additive multimodal structure.

4. Real-world evidence that representation/source choice controls identifiability.

5. Evidence that JAD may retain lower-order leakage despite joint predictive advantage.

6. Conditional purification if v5.2 P1 succeeds.

---

# 118. Do Not Add Contribution 6 Yet

Conditional purification is not yet a contribution.

It is only a hypothesis.

Promote it only after mechanistic validation.

---

# 119. Paper Positioning

The paper should increasingly be positioned around:

\[
\boxed{
\text{When is a multimodal interaction representation actually interaction-like?}
}
\]

rather than:

\[
\text{How do we obtain the highest multimodal accuracy?}
\]

---

# 120. Current Immediate Execution Order

```text
1. Load frozen five-seed MOSEI R2 models.

2. Extract q_add.

3. Extract q_joint.

4. Compute d = q_joint - q_add.

5. Extract h_D2.

6. Fit V-only reconstruction probes.

7. Fit A-only reconstruction probes.

8. Fit additive V+A reconstruction probes.

9. Optionally fit capacity-matched joint reconstruction probes.

10. Compute R² for q_add, q_joint, d, h.

11. Produce S0 shortcut table.

12. Classify leakage location.

13. If target-level or mixed:
       open P1 purification.

14. If distillation-only:
       open factor-level diagnostic branch.

15. Do not modify model before decision.
```

---

# 121. Immediate Milestone

The next milestone is:

\[
\boxed{
\textbf{determine whether audio leakage originates in the JAD target or in the interaction distillation step.}
}
\]

---

# 122. Hard Decision Rule

No new method is allowed until this question has a clear empirical answer:

\[
\boxed{
R^2(A\rightarrow d)
\quad
\text{vs}
\quad
R^2(A\rightarrow h_{D2}).
}
\]

---

# 123. v5.2 Success

v5.2 succeeds scientifically even if no new method improves classification, provided it clearly determines:

```text
where lower-order leakage appears
whether JAD target is conditionally selective
whether purification preserves interaction structure
```

---

# 124. Current Main Thesis

\[
\boxed{
\textbf{
Higher-order multimodal interaction learning requires more than joint computation or joint predictive advantage: useful interaction representations must also be conditionally selective, faithfully represented, and downstream-accessible.
}
}
\]

---

# 125. Current Required Experiment

\[
\boxed{
\textbf{S0 — MOSEI JAD Shortcut Localization}
}
\]

with:

\[
q_{add},
\quad
q_J,
\quad
d=q_J-q_{add},
\quad
h_{D2}
\]

evaluated under:

\[
V\text{-only},
\quad
A\text{-only},
\quad
additive(V,A)
\]

reconstruction.

Only after S0 may the project decide whether:

\[
\boxed{
\textbf{Additively Purified JAD}
}
\]

is scientifically justified.