# AGENT.md

# ConFu++ v5.3

## Task-Relevance Gating, Joint Task Headroom, and CVPR Method Decision

---

# 0. Mission

This repository develops **ConFu++**, an extension of higher-order multimodal alignment toward explicit multimodal interaction discovery.

The project targets a **CVPR main-track paper**.

The controlled synthetic phase has established that explicit interaction structure can be represented and discovered under identifiable conditions.

The first canonical real-world experiment on CMU-MOSEI established:

\[
\boxed{
\text{cross-modal joint predictive identifiability}
}
\]

for:

\[
V+A\rightarrow T,
\]

but did not produce positive downstream sentiment accuracy gain.

v5.2 subsequently localized lower-order leakage and tested additive purification.

The purification hypothesis failed.

The immediate question is now:

\[
\boxed{
\textbf{Is the identifiable cross-modal interaction actually relevant to the downstream task?}
}
\]

This is the only question v5.3 is permitted to answer before another method is introduced.

---

# 1. Current Project State

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

Canonical JAD:
FROZEN

Fidelity–Accessibility tradeoff:
ESTABLISHED

Canonical MOSEI exact-source R1:
PASS

Natural identifiable mapping:
VA -> T

MOSEI R2:
COMPLETE

D2 target fidelity:
PASS

D2 source dependence:
PASS

D2 downstream sentiment gain:
FAIL

S0 shortcut localization:
MIXED

P1 additive purification:
FAIL

P2 purified distillation:
NOT ALLOWED

Synthetic purification validation:
NOT ALLOWED

Task-relevance audit:
NEXT

Task-aware JAD:
CONDITIONAL

Third-order discovery:
BLOCKED

UR-FUNNY tuning:
BLOCKED
```

---

# 2. Frozen Canonical Natural Setting

Use:

```text
dataset:
CMU-MOSEI canonical raw

source:
mosei_raw.pkl

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

R1 gate:
J_val > 0.01 in 3/3 seeds

interaction rank:
64

R2 seeds:
1..5
```

Do not alter this configuration in v5.3.

---

# 3. Canonical Cross-Modal Identifiability

For source modalities:

\[
r_V,r_A
\]

and target:

\[
r_T,
\]

define:

\[
q_{\text{add}}
=
q_V(r_V)
+
q_A(r_A)
\]

and capacity-matched:

\[
q_{\text{joint}}(r_V,r_A).
\]

Joint Predictive Advantage:

\[
\boxed{
J_{\text{cross}}
=
R^2(q_{\text{joint}},r_T)
-
R^2(q_{\text{add}},r_T).
}
\]

Canonical result:

\[
J_{\text{cross}}
\approx0.0198.
\]

All three screening seeds satisfy:

\[
J_{\text{cross}}>0.01.
\]

Thus:

\[
\boxed{
\text{cross-modal interaction is operationally identifiable}.
}
\]

---

# 4. Canonical JAD Target

Define:

\[
\boxed{
d
=
q_{\text{joint}}
-
q_{\text{add}}.
}
\]

This is the canonical:

\[
\boxed{
\text{Joint-Advantage Distillation target}.
}
\]

---

# 5. Canonical Interaction Representation

Interaction architecture:

\[
u_V=LN(U_Vr_V)
\]

\[
u_A=LN(U_Ar_A)
\]

\[
h_{D2}
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

Training target:

\[
h_{D2}\leftrightarrow d
\]

with canonical cosine regression.

---

# 6. R2 Downstream Finding

Lower-order sentiment representation:

\[
Z_{\text{low}}
=
[r_V,r_A].
\]

Accuracy:

\[
64.397\pm0.231.
\]

D2:

\[
64.363\pm0.288.
\]

Gain:

\[
\boxed{
-0.034\pm0.099\text{ pp}.
}
\]

Therefore:

\[
\boxed{
\text{JAD does not improve fixed-threshold sentiment accuracy}.
}
\]

---

# 7. D2 Fidelity

D2 target cosine:

\[
0.848\pm0.038.
\]

Effective rank:

\[
25.66\pm1.36.
\]

Therefore:

\[
\boxed{
\text{D2 successfully represents its selected interaction target}.
}
\]

Representation collapse is not the explanation for downstream failure.

---

# 8. Source Dependence

Vision shuffle drop:

\[
+3.132\pm1.636\text{ pp}.
\]

Audio shuffle drop:

\[
+3.591\pm0.558\text{ pp}.
\]

Therefore both source modalities influence the representation.

However:

\[
\boxed{
\text{source dependence}
\neq
\text{irreducible interaction}.
}
\]

---

# 9. v5.2 S0 Result

S0 audited:

\[
q_{\text{add}}
\]

\[
q_{\text{joint}}
\]

\[
d=q_{\text{joint}}-q_{\text{add}}
\]

\[
h_{D2}.
\]

The result is:

\[
\boxed{
\text{MIXED LEAKAGE}.
}
\]

---

# 10. Target-Level Leakage

Validation reconstruction of:

\[
d
\]

from vision alone:

\[
\boxed{
R^2(V\rightarrow d)
=
0.296\pm0.045.
}
\]

Audio alone:

\[
R^2(A\rightarrow d)
=
0.140\pm0.045.
\]

Thus lower-order leakage exists before interaction distillation.

---

# 11. Distillation-Level Leakage

For:

\[
h_{D2},
\]

validation audio-only reconstruction becomes:

\[
\boxed{
R^2(A\rightarrow h_{D2})
=
0.301\pm0.045.
}
\]

Therefore audio reconstructability increases substantially during:

\[
d\rightarrow h.
\]

Thus:

\[
\boxed{
\text{both target construction and distillation contribute to leakage}.
}
\]

---

# 12. Leakage Does Not Mean No Joint Structure

S0 also found nontrivial joint reconstruction advantage.

Therefore:

\[
\boxed{
\text{lower-order leakage}
}
\]

and:

\[
\boxed{
\text{non-additive joint structure}
}
\]

can coexist.

This is a critical finding.

---

# 13. v5.2 P1 Hypothesis

P1 attempted additive purification.

Fit:

\[
\hat d_{\text{add}}
=
f_V(r_V)+f_A(r_A).
\]

Define:

\[
\boxed{
d^\perp
=
d-
\hat d_{\text{add}}.
}
\]

Goal:

\[
\text{remove lower-order reconstructability}.
\]

---

# 14. P1 Selectivity Result

P1 reduced:

\[
R^2_V:
0.296
\rightarrow
0.212
\]

and:

\[
R^2_A:
0.140
\rightarrow
0.059.
\]

Therefore purification did reduce single-modality leakage.

---

# 15. P1 Joint-Structure Failure

However:

\[
\boxed{
J_{d^\perp}
=
-0.021\pm0.133
}
\]

on the primary validation audit.

Only:

\[
2/5
\]

seeds have positive joint advantage.

Thus:

\[
\boxed{
\text{purification destroyed the stable joint predictive structure}.
}
\]

---

# 16. P1 Rank Reduction

Effective rank changed from approximately:

\[
7.45
\]

to:

\[
4.69.
\]

Therefore P1 also reduced target dimensional diversity.

---

# 17. P1 Decision

Official decision:

\[
\boxed{
\textbf{P1 FAIL}.
}
\]

Do NOT run:

```text
P2 purified MOSEI distillation
synthetic D2_perp validation
lambda purification sweep
alternative purifier architectures
```

---

# 18. Scientific Lesson From P1

Single-modality reconstructability cannot automatically be classified as removable nuisance.

Some lower-order predictable structure may be geometrically entangled with the non-additive structure.

Therefore:

\[
\boxed{
\text{conditional selectivity is a diagnostic, not an absolute optimization objective}.
}
\]

---

# 19. Previous Four-Axis Framework Is Insufficient

Previous framework:

\[
\text{Identifiability}
\rightarrow
\text{Conditional Selectivity}
\rightarrow
\text{Fidelity}
\rightarrow
\text{Accessibility}.
\]

P1 shows that enforcing selectivity too aggressively can destroy identifiable joint structure.

Therefore v5.3 revises the framework.

---

# 20. Updated Framework

The preferred hierarchy is now:

\[
\boxed{
\text{Cross-Modal Identifiability}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Non-Additive Structure}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Task Relevance}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Representation Fidelity}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Accessibility}.
}
\]

Conditional selectivity remains a diagnostic property.

---

# 21. Critical Missing Quantity

Current R1 establishes:

\[
J_{\text{cross}}>0.
\]

But downstream task is not:

\[
VA\rightarrow T.
\]

The downstream task is:

\[
VA\rightarrow Y_{\text{sentiment}}.
\]

Therefore:

\[
\boxed{
J_{\text{cross}}>0
}
\]

does not imply:

\[
\boxed{
\text{task-relevant interaction exists}.
}
\]

---

# 22. Main v5.3 Question

The central question becomes:

\[
\boxed{
\textbf{Does sentiment classification itself contain measurable non-additive V+A headroom?}
}
\]

This must be answered before modifying JAD.

---

# 23. T0 — Task-Relevance / Joint-Task-Headroom Audit

The next mandatory experiment is:

\[
\boxed{
\textbf{T0 — MOSEI Sentiment Joint-Task-Headroom Audit}.
}
\]

No JAD training occurs in T0.

---

# 24. T0 Task

Use the frozen sentiment label:

\[
\boxed{
Y=\mathbf1[\text{sentiment}>0].
}
\]

Use the same:

\[
r_V,r_A
\]

representations as canonical R2.

---

# 25. T0 Models

T0 compares four controlled models.

### M0 — Linear Lower-Order

\[
c_L([r_V,r_A]).
\]

Purpose:

standard linear baseline.

---

# 26. M1 — Additive Task Predictor

Define:

\[
\boxed{
c_{\text{add}}
=
c_V(r_V)+c_A(r_A).
}
\]

The two source branches are independent before score addition.

No multiplicative or cross-modal interaction is allowed.

---

# 27. M2 — Explicit Joint Task Predictor

Use capacity-matched joint model:

\[
u_V=P_Vr_V
\]

\[
u_A=P_Ar_A
\]

and:

\[
\boxed{
c_{\text{joint}}
=
W[
u_V;
u_A;
u_V\odot u_A
].
}
\]

The important distinction is explicit access to:

\[
u_V\odot u_A.
\]

---

# 28. M3 — Generic Concat MLP

Use:

\[
\boxed{
c_{\text{MLP}}
=
MLP([r_V,r_A]).
}
\]

This model has implicit nonlinear joint access.

It is a control for the possibility that the multiplicative product architecture is too restrictive.

---

# 29. Capacity Matching

Require:

\[
Params(c_{\text{joint}})
\approx
Params(c_{\text{add}})
\]

and preferably:

\[
Params(c_{\text{MLP}})
\]

in the same approximate range.

Preferred mismatch:

\[
<1\%.
\]

Maximum acceptable:

\[
<5\%.
\]

Report exact parameter counts.

---

# 30. Do Not Over-Match Capacity

Do not perform a large hidden-dimension search solely to obtain identical parameter counts.

Use deterministic architecture calculation.

One configuration only.

---

# 31. T0 Training

Train all models using the same:

```text
train split
validation split
test split
optimizer family
maximum epochs
early stopping rule
batch size
seed protocol
```

Only model structure differs.

---

# 32. T0 Seeds

Use:

\[
\boxed{
5\text{ seeds}.
}
\]

Seed:

```text
Python
NumPy
PyTorch
CUDA
DataLoader
model initialization
```

where applicable.

---

# 33. T0 Validation-Only Selection

Use validation only for:

```text
checkpoint selection
early stopping
regularization selection if fixed small candidate set exists
```

Prefer no hyperparameter search.

Test is reporting only.

---

# 34. T0 Metrics

Report:

\[
Accuracy
\]

\[
MacroF1
\]

\[
AUC
\]

and:

\[
\boxed{
BCE / LogLoss.
}
\]

Log-loss is important because it avoids relying only on a fixed classification threshold.

---

# 35. Task Joint Headroom

Define:

\[
\boxed{
H_{\text{task}}^{Acc}
=
Acc_{\text{joint}}
-
Acc_{\text{add}}.
}
\]

Similarly:

\[
H_{\text{task}}^{F1}
=
F1_{\text{joint}}
-
F1_{\text{add}}
\]

and:

\[
H_{\text{task}}^{AUC}
=
AUC_{\text{joint}}
-
AUC_{\text{add}}.
\]

For loss:

\[
\boxed{
H_{\text{task}}^{NLL}
=
NLL_{\text{add}}
-
NLL_{\text{joint}}.
}
\]

Positive means the joint model is better.

---

# 36. Generic Nonlinear Headroom

Also define:

\[
\boxed{
H_{\text{MLP}}
=
Perf(c_{\text{MLP}})
-
Perf(c_{\text{add}}).
}
\]

This determines whether task non-additivity exists beyond the specific multiplicative architecture.

---

# 37. Required T0 Main Table

Produce:

| Model | Params | Accuracy | Macro F1 | AUC | BCE |
|---|---:|---:|---:|---:|---:|
| Linear VA | | | | | |
| Additive | | | | | |
| Joint Product | | | | | |
| Generic MLP | | | | | |

All values:

\[
mean\pm sample\ std
\]

over five seeds.

---

# 38. Required Paired Table

Also report:

| Comparison | Acc Δ | F1 Δ | AUC Δ | BCE improvement |
|---|---:|---:|---:|---:|
| Joint − Additive | | | | |
| MLP − Additive | | | | |

Include:

```text
per-seed differences
95% confidence interval
```

if appropriate.

Do not over-focus on \(p<0.05\).

---

# 39. T0 Primary Question

The primary question is not:

> Which classifier has the highest accuracy?

It is:

\[
\boxed{
\textbf{Is there measurable task-relevant non-additive headroom beyond additive modality contributions?}
}
\]

---

# 40. T0 Decision — Case A

If:

\[
c_{\text{joint}}
\approx
c_{\text{add}}
\]

and:

\[
c_{\text{MLP}}
\approx
c_{\text{add}},
\]

then:

\[
\boxed{
H_{\text{task}}\approx0.
}
\]

Interpretation:

> sentiment has little measurable non-additive V+A headroom under the tested representation and model classes.

---

# 41. Case A Consequence

If Case A holds:

\[
\boxed{
\textbf{FREEZE MOSEI SENTIMENT}.
}
\]

Do not modify JAD for sentiment.

The observed result is then expected:

\[
J_{\text{cross}}>0
\]

but:

\[
H_{\text{task}}\approx0.
\]

Thus the cross-modal interaction is identifiable but not measurably task relevant.

---

# 42. Important Case A Scientific Result

This would establish:

\[
\boxed{
\text{cross-modal identifiability}
\not\Rightarrow
\text{task relevance}.
}
\]

This is stronger and cleaner than claiming JAD failed.

---

# 43. T0 Decision — Case B

If:

\[
c_{\text{joint}}>c_{\text{add}}
\]

consistently,

then:

\[
\boxed{
H_{\text{task}}>0.
}
\]

Task-relevant non-additive structure exists.

---

# 44. Case B Consequence

If:

\[
H_{\text{task}}>0
\]

but canonical JAD still gives:

\[
\Delta_{D2}\approx0,
\]

then:

\[
\boxed{
\text{JAD identifies cross-modal interaction directions that are not sufficiently aligned with task-relevant interaction directions}.
}
\]

This is a meaningful localized failure.

---

# 45. Case B Opens a New Method Hypothesis

Only Case B allows consideration of:

\[
\boxed{
\text{Task-Aware Joint-Advantage Distillation}.
}
\]

Do NOT implement it before T0.

---

# 46. T0 Decision — Case C

Possible:

\[
c_{\text{joint}}\approx c_{\text{add}}
\]

but:

\[
c_{\text{MLP}}>c_{\text{add}}.
\]

Interpretation:

\[
\boxed{
\text{task non-additivity may exist, but the multiplicative hypothesis class is insufficient}.
}
\]

This is a hypothesis-class limitation.

---

# 47. Case C Consequence

Do NOT immediately modify JAD.

First determine whether the issue belongs to:

\[
\mathcal H_J
\]

rather than representation discovery.

This may justify one controlled alternative joint hypothesis class later.

Not now.

---

# 48. T0 Decision — Case D

If:

\[
c_{\text{joint}}>c_{\text{add}}
\]

but:

\[
c_{\text{MLP}}\le c_{\text{add}},
\]

inspect training stability and architecture parity.

Do not claim joint headroom immediately.

This pattern is suspicious.

---

# 49. Practical Task-Headroom Gate

Do not promote Case B based on a tiny statistically noisy difference.

Prefer:

```text
consistent sign across seeds
nontrivial validation improvement
same qualitative direction in more than one metric
```

Do NOT define a universal arbitrary threshold yet.

---

# 50. Validation Is Primary for Gate

Task-headroom decision must use:

\[
\boxed{
\text{validation performance}.
}
\]

Test supports final reporting only.

Do not decide Task-Aware JAD from test results.

---

# 51. Why AUC Matters

Canonical D2 increased AUC while not improving accuracy.

Therefore T0 must include AUC.

If:

\[
AUC_{\text{joint}}>AUC_{\text{add}}
\]

consistently but threshold metrics do not change, task interaction may primarily affect ranking/calibration.

---

# 52. Why BCE Matters

BCE directly measures predictive probability quality.

It can reveal improvements hidden by thresholded accuracy.

Thus:

\[
\boxed{
BCE
}
\]

should be a core T0 metric.

---

# 53. Calibration Diagnostic

Optionally report:

```text
ECE
Brier score
```

only if already easy to implement.

Do not delay T0 for calibration metrics.

---

# 54. Do Not Use JAD Embeddings in T0

T0 operates directly on:

\[
r_V,r_A.
\]

Do NOT include:

\[
h_{D2}
\]

as classifier input.

Otherwise the task-headroom measurement becomes contaminated by JAD quality.

---

# 55. T0 Is a Property Audit

T0 estimates whether the task itself contains exploitable non-additivity under the frozen representations.

It is not a new ConFu++ benchmark.

---

# 56. Required T0 Artifact

Create:

```text
results/mosei/v53/
    T0_TASK_HEADROOM_AUDIT.md
```

and:

```text
results/mosei/v53/
    t0_task_headroom.json
```

---

# 57. T0 JSON Requirements

For every:

```text
model × seed
```

save:

```text
parameter_count
best_validation_epoch
train_loss
val_loss
test_loss
val_accuracy
test_accuracy
val_f1
test_f1
val_auc
test_auc
```

---

# 58. T0 Unit Tests

Add tests for:

```text
additive branches do not exchange features
joint model receives both modalities
product term uses correct dimensions
generic MLP uses concatenated V+A
parameter mismatch within tolerance
same splits across models
test metrics unused for selection
seed controls initialization
```

---

# 59. No Task Label Leakage Into Existing Predictors

T0 classifiers use sentiment labels because T0 explicitly measures task headroom.

But:

\[
q_{\text{add}}
\]

and:

\[
q_{\text{joint}}
\]

from cross-modal JAD remain label-free.

Do not retrain them with sentiment.

---

# 60. Separation of Two Advantages

From v5.3 onward define clearly:

### Cross-modal joint advantage

\[
\boxed{
J_{\text{cross}}
}
\]

measuring:

\[
VA\rightarrow T.
\]

### Task joint headroom

\[
\boxed{
H_{\text{task}}
}
\]

measuring:

\[
VA\rightarrow Y.
\]

These must never be conflated.

---

# 61. Fundamental New Distinction

The paper should explicitly separate:

\[
\boxed{
\text{predictive interaction about another modality}
}
\]

from:

\[
\boxed{
\text{predictive interaction about the downstream task}.
}
\]

They may be different subspaces.

---

# 62. Potential Subspace Interpretation

Let:

\[
\mathcal J_{\text{cross}}
\]

be cross-modal joint structure useful for predicting \(T\).

Let:

\[
\mathcal J_{\text{task}}
\]

be joint structure useful for predicting \(Y\).

There is no guarantee:

\[
\boxed{
\mathcal J_{\text{cross}}
=
\mathcal J_{\text{task}}.
}
\]

Their overlap may be small.

---

# 63. Current JAD Objective

Canonical JAD optimizes:

\[
\boxed{
\mathcal J_{\text{cross}}.
}
\]

It does not explicitly optimize:

\[
\mathcal J_{\text{task}}.
\]

Therefore no task gain should be assumed automatically.

---

# 64. Potential Explanation of MOSEI

Current evidence may eventually support:

\[
\boxed{
\mathcal J_{\text{cross}}\neq\varnothing
}
\]

but either:

\[
\boxed{
\mathcal J_{\text{task}}\approx\varnothing
}
\]

or:

\[
\boxed{
\mathcal J_{\text{cross}}
\cap
\mathcal J_{\text{task}}
\text{ is small}.
}
\]

T0 distinguishes these possibilities partially.

---

# 65. Updated Paper Framework

The emerging pipeline is:

\[
\boxed{
\text{Cross-Modal Identifiability}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Task Joint Headroom}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Interaction Discovery}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Fidelity}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Accessibility}.
}
\]

---

# 66. Conditional Selectivity Position

Conditional selectivity remains:

\[
\boxed{
\text{a representation diagnostic}.
}
\]

Do not make zero single-modality reconstructability a mandatory definition of useful interaction.

P1 demonstrated why.

---

# 67. New Dataset Selection Principle

Future real-world datasets/tasks should ideally satisfy two gates:

\[
\boxed{
J_{\text{cross}}>0
}
\]

and:

\[
\boxed{
H_{\text{task}}>0.
}
\]

Only then is explicit interaction distillation strongly motivated for downstream performance.

---

# 68. Screen-First Becomes Two-Stage

Old rule:

\[
\boxed{
\text{screen first, train second}.
}
\]

Updated rule:

\[
\boxed{
\text{cross-modal screen}
\rightarrow
\text{task-headroom screen}
\rightarrow
\text{interaction training}.
}
\]

---

# 69. Implication for Future Datasets

For MUStARD, UR-FUNNY, or other datasets:

first evaluate:

\[
J_{\text{cross}}.
\]

Then evaluate:

\[
H_{\text{task}}.
\]

Only settings passing both should become positive JAD development settings.

---

# 70. No Dataset Mining Yet

Do NOT immediately screen new datasets before T0 finishes.

MOSEI already contains a valid:

\[
J_{\text{cross}}>0
\]

setting.

Use it to understand task relevance first.

---

# 71. If T0 Case A

If:

\[
H_{\text{task}}\approx0,
\]

MOSEI sentiment becomes a valuable negative control.

Then move to:

\[
\boxed{
\text{MUStARD}
}
\]

with the new two-gate protocol.

---

# 72. MUStARD Hypothesis

Sarcasm may require interaction between:

```text
text content
prosody
visual expression
```

making it a plausible task with:

\[
H_{\text{task}}>0.
\]

This is a hypothesis only.

It must be screened.

---

# 73. UR-FUNNY After MUStARD

UR-FUNNY remains useful because historical results suggest strong text dominance.

It may provide:

\[
\boxed{
H_{\text{task}}\approx0
}
\]

or low cross-modal identifiability.

Either outcome is useful for the framework.

---

# 74. Task-Aware Branch Trigger

The Task-Aware branch is permitted only if:

\[
\boxed{
H_{\text{task}}>0
}
\]

on validation across stable seeds.

---

# 75. Potential Task-Aware Objective

Conceptually only, not yet approved:

\[
q_{\text{task-add}}
\]

vs:

\[
q_{\text{task-joint}}.
\]

Possible task interaction target:

\[
\boxed{
d_{\text{task}}
=
q_{\text{task-joint}}
-
q_{\text{task-add}}.
}
\]

This would be label-supervised.

---

# 76. Task-Aware JAD Is a Different Method Family

If opened, explicitly distinguish:

### Canonical JAD

Self-/cross-supervised predictive interaction discovery.

### Task-Aware JAD

Supervised task-relevant interaction discovery.

Do not silently merge the two.

---

# 77. CVPR Risk of Task-Aware JAD

A task-aware target may improve accuracy but weaken the conceptual novelty of label-free interaction discovery.

Therefore only pursue it if T0 demonstrates a real mismatch between:

\[
\mathcal J_{\text{cross}}
\]

and:

\[
\mathcal J_{\text{task}}.
\]

---

# 78. Prefer Understanding Over Accuracy Chase

Do not implement Task-Aware JAD solely because canonical D2 accuracy is flat.

The method must follow from T0 evidence.

---

# 79. If T0 Case B

Then immediate research question becomes:

\[
\boxed{
\textbf{why does canonical JAD fail to capture the task-relevant joint subspace?}
}
\]

Only then create v5.4.

---

# 80. If T0 Case C

If generic MLP shows headroom but multiplicative joint predictor does not:

the next question is:

\[
\boxed{
\text{Is the current interaction hypothesis class too restrictive?}
}
\]

This could justify one carefully controlled alternative \(\mathcal H_J\).

No broad architecture search.

---

# 81. If T0 Case A

Do not interpret this as:

> multimodal sentiment contains no interaction.

Correct:

> No stable task-relevant non-additive headroom was detected under the tested canonical representation and capacity-controlled predictor families.

---

# 82. Hypothesis-Class Dependence Remains Central

Task headroom is:

\[
\boxed{
H_{\text{task}}
=
H(
\mathcal D,
\phi,
\mathcal H_A^{task},
\mathcal H_J^{task}
).
}
\]

It is not an absolute property of the dataset.

---

# 83. Representation Dependence Remains Central

Canonical raw representation changed R1 outcome.

Therefore:

\[
\phi
\]

must always be specified when discussing:

```text
identifiability
task headroom
interaction
```

---

# 84. Updated Failure Taxonomy

### Failure I — Cross-Modal Non-Identifiability

\[
J_{\text{cross}}\approx0.
\]

### Failure II — No Task Joint Headroom

\[
H_{\text{task}}\approx0.
\]

### Failure III — Discovery Mismatch

\[
J_{\text{cross}}>0,
\quad
H_{\text{task}}>0,
\]

but learned interaction does not capture task-relevant structure.

### Failure IV — Representation Fidelity

Interaction target exists but \(h\) fails to represent it.

### Failure V — Accessibility

Task-relevant interaction is represented but restricted downstream probes cannot exploit it.

---

# 85. MOSEI Current Status Before T0

```text
Failure I:
NO

Failure II:
UNKNOWN

Failure III:
UNKNOWN

Failure IV:
NO

Failure V:
OBSERVED RELATIVE TO CANONICAL JAD
```

T0 resolves Failure II.

---

# 86. P1 Should Move to Supplementary

P1 is no longer a candidate method.

It belongs in:

```text
Supplementary:
Shortcut purification attempt
```

Purpose:

show that naive additive subtraction removes useful joint structure.

---

# 87. S0 Main-Paper Value

S0 may deserve a compact main-paper diagnostic because it establishes:

\[
\boxed{
\text{joint dependence can coexist with substantial lower-order reconstructability}.
}
\]

Detailed tables can move to supplementary.

---

# 88. P1 Main-Paper Value

Main text may state briefly:

> Explicitly projecting away additive reconstructability reduced leakage but also removed measurable joint structure.

Detailed numbers go to supplementary unless space permits.

---

# 89. Important Paper Message

Do not equate:

\[
\boxed{
\text{purity}
}
\]

with:

\[
\boxed{
\text{usefulness}.
}
\]

P1 demonstrates the danger of over-purification.

---

# 90. CVPR Story Evolution

The paper is no longer merely:

> We propose a better fusion representation.

It is increasingly:

> We develop a diagnostic framework for determining when higher-order multimodal interaction is identifiable, task relevant, representable, and useful.

JAD is the method inside that framework.

---

# 91. Potential Main Thesis

\[
\boxed{
\textbf{
Multimodal interaction should not be assumed from modality count: useful higher-order learning requires cross-modal identifiability, task-relevant non-additive headroom, faithful representation, and downstream accessibility.
}
}
\]

---

# 92. Potential Contribution 1

Introduce a hierarchy separating:

```text
cross-modal identifiability
task relevance
interaction fidelity
accessibility
```

for multimodal interaction learning.

---

# 93. Potential Contribution 2

Introduce:

\[
\boxed{
IPIB
}
\]

for controlled identifiable interaction analysis.

---

# 94. Potential Contribution 3

Introduce:

\[
\boxed{
Joint-Advantage Distillation
}
\]

to distill predictable non-additive cross-modal structure.

---

# 95. Potential Contribution 4

Show:

\[
\boxed{
\text{cross-modal interaction fidelity}
\not\Rightarrow
\text{task relevance or accessibility}.
}
\]

---

# 96. Potential Contribution 5

Show that canonical source/representation choice can change whether natural interaction is operationally identifiable.

---

# 97. Do Not Claim P1 as Contribution

P1 failed.

It is evidence supporting the diagnostic framework, not a method contribution.

---

# 98. Current Forbidden Work

Until T0 completes:

```text
NO P2
NO new purifier
NO purification lambda
NO cross-attention
NO rank sweep
NO JAD loss change
NO task loss inside canonical JAD
NO third-order discovery
NO MUStARD screening
NO UR-FUNNY tuning
NO new D2 architecture
```

---

# 99. Current Permitted Work

Only:

```text
T0 task-headroom audit
paper theory update
paper framework writing
related-work analysis
main-figure drafting
```

---

# 100. T0 Code Task

Implement:

```text
src/experiments/multibench/mosei_task_headroom.py
```

or equivalent.

Do not modify canonical R1/R2 runners.

---

# 101. Shared Task Models

Recommended module:

```text
src/experiments/multibench/task_headroom_models.py
```

containing:

```text
LinearTaskProbe
AdditiveTaskPredictor
JointTaskPredictor
ConcatMLPTaskPredictor
```

---

# 102. Fair Architecture Rule

The additive model must not receive concatenated features before branch-level prediction.

Correct:

\[
c_V(V)+c_A(A).
\]

Incorrect additive baseline:

\[
MLP([V,A]).
\]

The latter can learn interactions.

---

# 103. Joint Architecture Rule

The explicit joint model must have access to a defined cross-modal interaction term.

Preferred:

\[
u_V\odot u_A.
\]

Do not hide interaction inside arbitrary deep concatenation only.

---

# 104. Generic MLP Role

Generic MLP is a hypothesis-class control.

It is not the primary task joint estimator.

Its purpose is:

> detect nonlinear headroom missed by the explicit product architecture.

---

# 105. T0 No Representation Fine-Tuning

Use frozen:

\[
r_V,r_A.
\]

Do not fine-tune modality encoders.

Otherwise representation change becomes another confound.

---

# 106. T0 Primary Decision Artifact

The report must end with exactly one:

```text
NO_TASK_HEADROOM

EXPLICIT_JOINT_HEADROOM

GENERIC_NONLINEAR_HEADROOM_ONLY

INCONCLUSIVE
```

---

# 107. NO_TASK_HEADROOM

Choose when additive and both joint models are practically equivalent.

Action:

\[
\boxed{
\text{freeze MOSEI sentiment}.
}
\]

---

# 108. EXPLICIT_JOINT_HEADROOM

Choose when capacity-matched explicit joint predictor reliably improves over additive.

Action:

\[
\boxed{
\text{open task-relevance mismatch investigation}.
}
\]

Potential later Task-Aware JAD.

---

# 109. GENERIC_NONLINEAR_HEADROOM_ONLY

Choose when concat MLP improves but explicit product model does not.

Action:

\[
\boxed{
\text{investigate hypothesis-class limitation}.
}
\]

Do not immediately redesign JAD.

---

# 110. INCONCLUSIVE

Choose when differences are unstable or seed-sensitive.

Action:

\[
\boxed{
\text{do not introduce a new method}.
}
\]

Freeze current interpretation and consider another dataset.

---

# 111. Hard Stop

T0 result must determine the next branch.

Do not override it because one test metric looks favorable.

---

# 112. If MOSEI Is Frozen

Preserve MOSEI as a paper result.

It demonstrates:

```text
canonical representation matters for identifiability

cross-modal prediction contains joint structure

JAD can faithfully learn this structure

joint structure can contain lower-order leakage

purifying leakage can remove joint structure

cross-modal interaction need not improve sentiment
```

This is scientifically useful.

---

# 113. Next Dataset Protocol After MOSEI Freeze

Use:

\[
\boxed{
\text{Two-Gate Real-World Screening}.
}
\]

Gate 1:

\[
J_{\text{cross}}>0.
\]

Gate 2:

\[
H_{\text{task}}>0.
\]

Only then run full JAD.

---

# 114. Preferred Next Dataset

If MOSEI freezes:

\[
\boxed{
\text{MUStARD}
}
\]

is the preferred next screening dataset.

Reason:

task semantics plausibly require cross-modal incongruity.

This remains a hypothesis.

---

# 115. UR-FUNNY

Run after MUStARD.

Do not tune accuracy first.

Evaluate:

\[
J_{\text{cross}}
\]

and:

\[
H_{\text{task}}
\]

before interaction training.

---

# 116. Third-Order Status

Still:

\[
\boxed{
\textbf{BLOCKED}.
}
\]

Pair-level task relevance must be understood before order-3 discovery.

---

# 117. Theory Update

Let target modality:

\[
T.
\]

Cross-modal decomposition:

\[
T
=
A_T(V,A)
+
J_T(V,A)
+
\epsilon.
\]

JAD attempts to estimate:

\[
J_T.
\]

---

# 118. Task Decomposition

For downstream label:

\[
Y,
\]

conceptually:

\[
Y
=
A_Y(V,A)
+
J_Y(V,A)
+
\epsilon_Y.
\]

There is no reason to assume:

\[
\boxed{
J_T=J_Y.
}
\]

This is central to v5.3.

---

# 119. Cross-Task Overlap

Potentially:

\[
J_T
=
J_{\text{shared}}
+
J_{T-only}
\]

and:

\[
J_Y
=
J_{\text{shared}}
+
J_{Y-only}.
\]

Canonical JAD is useful downstream only to the extent that it captures:

\[
J_{\text{shared}}.
\]

This is conceptual language, not an identified decomposition yet.

---

# 120. Do Not Claim Exact Subspaces Yet

The above is a conceptual interpretation only.

Do NOT claim empirical orthogonal decomposition into:

```text
shared
target-only
task-only
```

without a specific experiment.

---

# 121. Strong Potential CVPR Insight

If T0 shows task headroom:

\[
H_{\text{task}}>0
\]

while canonical JAD remains inaccessible, then:

\[
\boxed{
\text{cross-modal predictive interaction and task-relevant interaction are distinct learning targets}.
}
\]

This could become a central paper result.

---

# 122. Alternative Strong Insight

If T0 shows:

\[
H_{\text{task}}\approx0,
\]

then:

\[
\boxed{
\text{identifiable interaction need not be downstream relevant}.
}
\]

This is also a strong result.

---

# 123. Either Outcome Is Informative

Do not design T0 as an experiment that must pass.

The point is to determine which scientific story is true.

---

# 124. Immediate Execution Order

```text
1. Freeze canonical MOSEI V/A features.

2. Freeze sentiment > 0 labels.

3. Implement Linear VA classifier.

4. Implement additive classifier.

5. Implement capacity-matched product-joint classifier.

6. Implement capacity-matched generic concat MLP.

7. Verify parameter counts.

8. Verify no cross-branch access in additive model.

9. Run seed-1 smoke test.

10. Verify validation-only checkpoint selection.

11. Run seeds 1..5.

12. Report Accuracy/F1/AUC/BCE.

13. Compute Joint - Additive paired differences.

14. Compute MLP - Additive paired differences.

15. Classify T0 outcome.

16. Do not implement a new method before classification.
```

---

# 125. Immediate Milestone

The next milestone is:

\[
\boxed{
\textbf{determine whether canonical MOSEI sentiment contains measurable non-additive V+A task headroom.}
}
\]

---

# 126. Main Decision Equation

The key comparison is:

\[
\boxed{
Perf(c_{\text{joint}})
-
Perf(c_{\text{add}})
}
\]

with generic MLP as a secondary hypothesis-class control.

---

# 127. Hard Research Rule

Do not confuse:

\[
\boxed{
J_{\text{cross}}
}
\]

with:

\[
\boxed{
H_{\text{task}}.
}
\]

A dataset may have one without the other.

---

# 128. Updated Screen-First Principle

The new operational rule is:

\[
\boxed{
\textbf{
Cross-modal screen
\rightarrow
Task-headroom screen
\rightarrow
Interaction discovery
\rightarrow
Accessibility evaluation.
}
}
\]

---

# 129. v5.3 Scientific Success Condition

v5.3 succeeds when it determines whether:

\[
\boxed{
\text{MOSEI sentiment has task-relevant non-additive headroom}
}
\]

under the frozen canonical representation.

It does NOT require a new method or improved accuracy.

---

# 130. Final v5.3 Thesis

\[
\boxed{
\textbf{
An interaction may be identifiable for predicting another modality yet irrelevant to a downstream task; therefore cross-modal identifiability and task-relevant joint headroom must be screened separately.
}
}
\]

---

# 131. Current Required Experiment

\[
\boxed{
\textbf{
T0 — MOSEI Sentiment Joint-Task-Headroom Audit
}
}
\]

using:

\[
\boxed{
Linear,\ Additive,\ Explicit\ Joint,\ Generic\ MLP
}
\]

with:

\[
\boxed{
5\text{ seeds}
}
\]

and validation-only model selection.

No new ConFu++ method is permitted until T0 is complete.