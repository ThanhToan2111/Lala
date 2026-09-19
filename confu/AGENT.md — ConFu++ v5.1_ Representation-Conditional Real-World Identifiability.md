# AGENT.md

# ConFu++ v5.1

## Representation-Conditional Real-World Identifiability for CVPR

---

# 0. Mission

This repository develops **ConFu++**, an extension of higher-order multimodal alignment toward explicit multimodal interaction discovery.

The project is targeting a **CVPR main-track submission**.

The synthetic pair-discovery phase is frozen.

The current research question is now:

\[
\boxed{
\textbf{When is multimodal interaction identifiable in natural data representations?}
}
\]

The first real-world screen on CMU-MOSEI using pooled representations failed the pre-registered identifiability gate.

This result is valid and must not be bypassed.

However, two unresolved confounds remain:

1. approximate alignment between processed MultiBench samples and official emotion labels;
2. possible destruction of temporal interaction through early pooling.

v5.1 exists only to resolve these two confounds.

---

# 1. Current Research Status

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

D0/D1/D2 synthetic comparison:
COMPLETE

Canonical D2 / JAD:
FROZEN

Fidelity–Accessibility tradeoff:
ESTABLISHED

Reliability calibration:
CLOSED

CMU-MOSEI pooled R1:
FAIL

CMU-MOSEI R2:
NOT ALLOWED

Exact MOSEI alignment audit:
NEXT

Sequence-aware identifiability:
CONDITIONAL NEXT

UR-FUNNY tuning:
BLOCKED

Third-order discovery:
BLOCKED
```

---

# 2. Frozen Final Pair Method

The final pair-discovery method remains:

\[
\boxed{
\textbf{Joint-Advantage Distillation (JAD)}
}
\]

with:

\[
q_A=q_i(r_i)+q_j(r_j)
\]

and capacity-matched:

\[
q_J(r_i,r_j).
\]

Define:

\[
\boxed{
J_{ij\rightarrow k}
=
R^2(q_J,r_k)
-
R^2(q_A,r_k)
}
\]

and interaction target:

\[
\boxed{
d_{ij\rightarrow k}
=
q_J-q_A.
}
\]

Canonical interaction representation:

\[
h_{ij}
=
W
\left(
\frac{
LN(U_ir_i)\odot LN(U_jr_j)
}{
\sqrt R
}
\right)
\]

with:

\[
R=64.
\]

Canonical JAD loss remains cosine regression.

No method changes are allowed during v5.1.

---

# 3. Core Scientific Framework

The paper must distinguish:

\[
\boxed{\text{Identifiability}}
\]

\[
\boxed{\text{Fidelity}}
\]

\[
\boxed{\text{Accessibility}}.
\]

But v5.1 introduces an important refinement.

Identifiability is not simply a property of the dataset.

It depends on:

\[
\boxed{
\mathcal I
=
\mathcal I(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J
)
}
\]

where:

- \(\mathcal D\): data distribution;
- \(\phi\): representation;
- \(\mathcal H_A\): additive predictor hypothesis class;
- \(\mathcal H_J\): joint predictor hypothesis class.

---

# 4. Operational Identifiability

For representation:

\[
\phi,
\]

define:

\[
r_i=\phi_i(X_i).
\]

Then:

\[
\boxed{
J_\phi
=
R^2(
q_J(r_i,r_j),
r_k
)
-
R^2(
q_A(r_i,r_j),
r_k
).
}
\]

Therefore a failed pooled screen only allows the claim:

> No stable joint predictive advantage was detected under the pooled representation and tested predictor classes.

It does NOT justify:

> The dataset contains no multimodal interaction.

---

# 5. CMU-MOSEI R1 Completed Result

Current pooled representations:

\[
r_V\in\mathbb R^{35}
\]

\[
r_A\in\mathbb R^{74}
\]

\[
r_T\in\mathbb R^{300}.
\]

Mappings:

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V.
\]

---

# 6. Global Pooled Result

Mean three-seed validation joint advantage:

\[
J_{VA\rightarrow T}
=
-0.0111\pm0.0006
\]

\[
J_{VT\rightarrow A}
=
+0.0016\pm0.0062
\]

\[
J_{AT\rightarrow V}
=
+0.0010\pm0.0023.
\]

Therefore:

\[
\boxed{
\text{no global pooled mapping passes}.
}
\]

---

# 7. Emotion-Conditioned Result

Best observed candidate:

\[
\boxed{
fear,\quad VT\rightarrow A
}
\]

with:

\[
J_{val}
=
0.0142\pm0.0152.
\]

But only:

\[
2/3
\]

seeds satisfy:

\[
J_{val}>0.01.
\]

Therefore:

\[
\boxed{
\text{fear / VT→A FAILS the pre-registered gate}.
}
\]

---

# 8. Test Result Must Not Rescue Validation

Fear / VT→A test values are positive.

However:

\[
\boxed{
\text{test results cannot override a failed validation gate}.
}
\]

Do NOT select this setting using test behavior.

This remains a candidate requiring independent confirmation only.

---

# 9. Current R1 Decision

Current official decision:

```text
R1 pooled identifiability:
FAIL

R2:
NOT RUN

D0/D1/D2 real-world comparison:
NOT VALID YET
```

This decision is frozen.

---

# 10. Why MOSEI Is Not Yet Closed

Two confounds remain scientifically reasonable.

### Confound A

Emotion labels are not exactly aligned through canonical sample identifiers.

Current diagnostic agreement is approximately:

\[
97.85\%.
\]

### Confound B

Interaction may be temporal and could be destroyed by early pooling.

Therefore v5.1 permits exactly two additional MOSEI audits:

\[
\boxed{
R1.1\text{ — Exact Alignment Audit}
}
\]

and:

\[
\boxed{
R1.2\text{ — Representation Granularity Audit}.
}
\]

No other MOSEI rescue experiment is currently allowed.

---

# 11. R1.1 — Exact Alignment Audit

Goal:

\[
\boxed{
\textbf{establish deterministic one-to-one sample identity between features and labels}.
}
\]

Do not rely on approximate feature similarity if canonical IDs can be recovered.

---

# 12. Required Sample Identity

Preferred canonical cache format:

```text
segment_id
video_id
split
vision
audio
text
sentiment
happiness
sadness
anger
surprise
disgust
fear
```

Every example must retain a stable identifier.

---

# 13. Preferred Data Source Strategy

Whenever possible, construct:

```text
features + labels + split + IDs
```

from a single canonical dataset source.

Avoid:

```text
processed feature file A
+
labels reconstructed from unrelated file B
+
feature-nearest matching
```

for final paper experiments.

---

# 14. Exact Alignment Requirement

Preferred:

\[
\boxed{
100\%\text{ deterministic ID-based alignment}.
}
\]

If some samples are legitimately absent:

- document their IDs;
- document why they were removed;
- document which split is affected;
- never silently approximate-match them.

---

# 15. Alignment Audit Outputs

Create:

```text
results/mosei/alignment/
```

with:

```text
MOSEI_ALIGNMENT_AUDIT.md
mosei_alignment.json
```

Required fields:

```text
processed_count
official_count
matched_count
unmatched_processed_ids
unmatched_official_ids
duplicate_ids
split_mismatches
label_mismatches
```

---

# 16. Alignment Pass

R1.1 passes if:

\[
\boxed{
\text{all retained samples have deterministic canonical alignment}.
}
\]

The exact number of retained samples may differ from old processed cache.

That is acceptable if fully documented.

---

# 17. Alignment Fail

If exact IDs cannot be reconstructed:

do NOT pretend approximate matching is canonical.

Two valid options:

1. regenerate aligned features from a canonical source;
2. treat current emotion-conditioned MOSEI experiment as limited and do not use it as a core CVPR result.

---

# 18. Re-run Pooled R1 After Alignment

After exact alignment:

\[
\boxed{
\text{re-run the exact same pooled R1 protocol}.
}
\]

Do not change:

```text
predictor architecture
capacity
J threshold
optimizer
number of seeds
mapping definition
```

Only the data alignment changes.

---

# 19. Why Re-run Pooled R1

This isolates:

\[
\boxed{
\text{effect of alignment}
}
\]

from:

\[
\boxed{
\text{effect of representation}.
}
\]

Do not modify both at once.

---

# 20. R1.1 Comparison Table

Required:

| Setting | Old J | Exact-aligned J | Difference |
|---|---:|---:|---:|
| VA→T | | | |
| VT→A | | | |
| AT→V | | | |
| fear VT→A | | | |

Use validation metrics.

---

# 21. Possible R1.1 Outcome A

If exact alignment creates a mapping satisfying:

\[
J_{val}>0.01
\]

for:

\[
3/3
\]

seeds:

\[
\boxed{
\text{R1 PASS}.
}
\]

Freeze that setting.

Proceed to R2.

Do NOT run sequence-aware rescue first.

---

# 22. Possible R1.1 Outcome B

If exact alignment changes little and pooled R1 still fails:

proceed to:

\[
\boxed{
R1.2\text{ — Representation Granularity Audit}.
}
\]

---

# 23. R1.2 Motivation

Current pooled representation effectively asks:

\[
q(
pool(X_i),
pool(X_j)
)
\rightarrow
pool(X_k).
\]

But multimodal interaction may depend on:

\[
(X_i^t,X_j^t)
\]

or:

\[
(X_i^t,X_j^{t+\Delta}).
\]

Early pooling may remove this information.

Therefore:

\[
\boxed{
\text{pooling may destroy identifiability before predictor comparison begins}.
}
\]

---

# 24. Important Reviewer Objection

A likely reviewer could ask:

> How can you conclude that MOSEI lacks jointly predictive structure when temporal multimodal features were pooled before interaction screening?

v5.1 must answer this directly.

---

# 25. R1.2 Goal

Compare:

\[
\boxed{
\text{pooled identifiability}
}
\]

with:

\[
\boxed{
\text{sequence-aware identifiability}.
}
\]

No interaction representation is trained yet.

---

# 26. Sequence Inputs

Use original sequence-level features where available:

\[
V\in\mathbb R^{T\times d_V}
\]

\[
A\in\mathbb R^{T\times d_A}
\]

\[
T\in\mathbb R^{T\times d_T}.
\]

Respect masks and true sequence lengths.

Do not treat padding as valid observations.

---

# 27. Unimodal Temporal Encoders

For each modality:

\[
r_V=E_V(V)
\]

\[
r_A=E_A(A)
\]

\[
r_T=E_T(T).
\]

Important:

\[
\boxed{
E_V,E_A,E_T\text{ must be unimodal}.
}
\]

No encoder may access another modality during R1.2.

---

# 28. Why Unimodal Encoders

If:

\[
E_A(A,T)
\]

already sees text, then:

\[
r_A
\]

contains cross-modal interaction before:

\[
q_A/q_J
\]

comparison.

That invalidates the screening interpretation.

---

# 29. Recommended Temporal Encoder

Prefer a small architecture close to existing ConFu/MultiBench infrastructure.

Possible:

```text
small Transformer encoder
GRU/LSTM
temporal pooling network
```

But use one architecture consistently.

Do not benchmark many temporal encoders.

---

# 30. Recommended Default

Preferred first choice:

\[
\boxed{
\text{small unimodal Transformer encoder}
}
\]

followed by masked temporal pooling.

Reason:

- compatible with existing ConFu design;
- handles temporal structure;
- avoids introducing a large new model family.

---

# 31. Encoder Output Dimension

Use a common dimension:

\[
D
\]

for:

\[
r_V,r_A,r_T.
\]

Recommended:

\[
D=256
\]

if consistent with current ConFu implementation.

Do not sweep \(D\).

---

# 32. Encoder Training Question

R1.2 must avoid label supervision.

Temporal encoders may be:

### Option A

trained jointly with modality prediction;

or:

### Option B

frozen pretrained unimodal encoders.

Preferred first implementation:

\[
\boxed{
\text{train each predictor pipeline end-to-end using only modality reconstruction/prediction loss}.
}
\]

No emotion labels.

---

# 33. Predictor Comparison Must Remain Fair

Both:

\[
q_A
\]

and:

\[
q_J
\]

must use equivalent unimodal encoder families.

Do not give the joint branch a more expressive temporal encoder.

---

# 34. Strongest Fair Design

Use shared encoder outputs for each predictor comparison:

\[
r_i=E_i(X_i)
\]

\[
r_j=E_j(X_j).
\]

Then compare only:

\[
q_A(r_i,r_j)
\]

against:

\[
q_J(r_i,r_j).
\]

This isolates additive-vs-joint prediction.

---

# 35. Recommended Two-Stage Protocol

Preferred:

### Stage 1

learn/freeze unimodal representations using label-free modality objectives.

### Stage 2

train matched:

\[
q_A,q_J
\]

on the frozen representations.

This keeps:

\[
J
\]

easy to interpret.

---

# 36. No Joint Encoder During Screening

Forbidden:

```text
cross-attention
multimodal Transformer
co-attention
early concatenation before E_i
shared cross-modal latent encoder
```

during R1.2 representation creation.

---

# 37. R1.2 Mapping Directions

Still:

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V.
\]

Do not introduce additional mappings yet.

---

# 38. R1.2 Global First

Run global mapping screening first.

Do not immediately condition on emotion.

Required order:

```text
1. global screening
2. inspect validation J
3. only then emotion-conditioned diagnostics
```

---

# 39. Emotion-Conditioned R1.2

If exact emotion alignment is available, compute:

```text
happiness
sadness
anger
surprise
disgust
fear
```

using:

\[
score>0.
\]

These remain diagnostics/gating subsets.

---

# 40. Screening Gate Remains Frozen

Do NOT reduce:

\[
0.01
\]

because pooled R1 failed.

Required:

\[
\boxed{
J_{val}>0.01
}
\]

for:

\[
3/3
\]

seeds.

---

# 41. Why Gate Is Frozen

Changing the threshold after seeing results creates:

\[
\boxed{
\text{post-hoc selection bias}.
}
\]

A weaker threshold can be studied only in a future separately pre-registered experiment.

Not now.

---

# 42. Seeds

R1.2 screening:

\[
\boxed{
3\text{ seeds}
}
\]

with deterministic:

```text
dataset order
encoder initialization
predictor initialization
optimization order
```

---

# 43. Capacity Matching

Require:

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

Encoder parameters shared/equivalent between compared conditions should not create hidden joint-model capacity.

---

# 44. R1.2 Required Metrics

For every mapping:

```text
Add train R²
Add val R²
Add test R²

Joint train R²
Joint val R²
Joint test R²

J train
J val
J test
```

---

# 45. Predictor Overfit Audit

Inspect:

\[
J_{train}
\]

vs:

\[
J_{val}.
\]

If:

\[
J_{train}\gg J_{val},
\]

joint predictor may simply overfit.

Do not interpret this as interaction evidence.

---

# 46. Representation Health

For:

\[
r_V,r_A,r_T
\]

log:

```text
variance
effective rank
mean norm
dimension std
```

to detect collapsed temporal encoders.

---

# 47. Temporal Encoder Health

Also report:

```text
average valid sequence length
padding fraction
mask correctness
number of zero-length samples
```

where applicable.

---

# 48. Representation Granularity Comparison

Required table:

| Representation | Mapping | Add R² | Joint R² | J |
|---|---|---:|---:|---:|
| pooled | VA→T | | | |
| sequence | VA→T | | | |
| pooled | VT→A | | | |
| sequence | VT→A | | | |
| pooled | AT→V | | | |
| sequence | AT→V | | | |

Use validation means.

---

# 49. Key New Quantity

Define representation gain:

\[
\boxed{
\Delta J_{\phi}
=
J_{sequence}
-
J_{pooled}.
}
\]

This is diagnostic.

---

# 50. Important Possible Finding

If:

\[
J_{pooled}\approx0
\]

but:

\[
J_{sequence}>0.01
\]

consistently:

then:

\[
\boxed{
\text{interaction identifiability depends strongly on representation granularity}.
}
\]

This would be a valuable paper result.

---

# 51. Stronger Conceptual Result

Then identifiability should be explicitly described as:

\[
\boxed{
\text{representation-conditional interaction identifiability}.
}
\]

This strengthens the theory beyond dataset-level claims.

---

# 52. Possible Outcome A — Sequence PASS

If one sequence-aware mapping passes:

\[
J_{val}>0.01
\]

for all three seeds:

freeze:

```text
representation
mapping
task/subset
predictors
gate
```

and proceed to:

\[
\boxed{R2}.
\]

---

# 53. R2 After Sequence PASS

Only then compare:

\[
D0
\]

\[
D1
\]

\[
D2/JAD.
\]

Use the same frozen unimodal representations that passed screening.

Do not redesign encoders during R2.

---

# 54. Possible Outcome B — Sequence FAIL

If exact alignment and sequence-aware representations both fail:

\[
\boxed{
\textbf{FREEZE MOSEI AS A NEGATIVE REAL-WORLD IDENTIFIABILITY RESULT}.
}
\]

No further MOSEI rescue work.

---

# 55. MOSEI Freeze Wording

Safe claim:

> Under both pooled and sequence-aware unimodal representations and capacity-matched additive/joint predictor classes, CMU-MOSEI did not exhibit a stable joint predictive advantage satisfying our pre-registered criterion.

Do NOT say:

> MOSEI contains no interaction.

---

# 56. Why a Negative MOSEI Result Is Useful

It supports the paper's core thesis:

\[
\boxed{
\text{multimodal data does not automatically imply jointly identifiable multimodal structure}.
}
\]

This distinguishes ConFu++ from methods that always instantiate higher-order fusion without testing whether the target is identifiable.

---

# 57. Dataset Transition After MOSEI Freeze

If MOSEI fails R1.1 and R1.2:

next screen:

\[
\boxed{
\text{MUStARD}
}
\]

before UR-FUNNY.

---

# 58. Why MUStARD Next

Sarcasm may rely on incongruity among:

```text
text
prosody/audio
facial/visual cues
```

making it a plausible natural interaction benchmark.

This is a hypothesis only.

Screen first.

---

# 59. MUStARD Protocol

Use the same philosophy:

\[
q_A
\]

vs:

\[
q_J.
\]

Mappings:

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V.
\]

No D0/D1/D2 unless a mapping passes.

---

# 60. Dataset Order

Current real-world priority:

\[
\boxed{
\text{MOSEI exact audit}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{MOSEI sequence-aware audit}
}
\]

\[
\downarrow
\]

if fail:

\[
\boxed{
\text{MUStARD screening}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{UR-FUNNY screening}
}
\]

---

# 61. UR-FUNNY Status

UR-FUNNY remains:

```text
transfer benchmark
historical negative-control dataset
text-dominated setting
```

Do not resume tuning.

Only run identifiability screening first.

---

# 62. Why UR-FUNNY Is Later

Historical experiments already suggest:

- small multimodal headroom;
- text dominance;
- weak accessibility;
- several failed ConFu++ variants.

Therefore it is better used as a test of the identifiability framework than as the primary development dataset.

---

# 63. No New Method During Dataset Search

Do NOT modify JAD because a natural dataset fails screening.

Failure to pass:

\[
J>0
\]

is evidence about:

\[
\mathcal D,\phi,\mathcal H
\]

not automatically evidence that JAD needs more capacity.

---

# 64. Important Theory Update

The paper should no longer define identifiability only as:

> Is target modality predictable from source modalities?

Instead:

> Is there additional predictive structure available to a joint hypothesis class beyond an additive hypothesis class under a specified representation?

Formally:

\[
\boxed{
J_\phi
=
R^2_{\mathcal H_J}
-
R^2_{\mathcal H_A}.
}
\]

---

# 65. Hypothesis-Class Dependence

JAD interaction is:

\[
\boxed{
\text{hypothesis-class-relative}.
}
\]

This is a feature of the definition, not something to hide.

Different:

\[
\mathcal H_A,\mathcal H_J
\]

may identify different predictive interaction structure.

---

# 66. Representation Dependence

Similarly:

\[
\phi
\]

changes what is accessible to both predictor classes.

Therefore:

\[
\boxed{
J=J(\phi,\mathcal H_A,\mathcal H_J).
}
\]

This should become explicit in Method/Theory.

---

# 67. Dataset Dependence

The data distribution:

\[
\mathcal D
\]

also determines the available predictive interaction.

Therefore final form:

\[
\boxed{
J=
J(
\mathcal D,
\phi,
\mathcal H_A,
\mathcal H_J
).
}
\]

---

# 68. Implication for CVPR Story

The paper narrative becomes stronger:

> Higher-order multimodal interaction is not guaranteed merely because multiple modalities coexist. Interaction discovery depends on whether non-additive predictive structure is identifiable under the representation and hypothesis classes being used.

---

# 69. Updated Research Questions

## RQ1

When is multimodal interaction identifiable?

## RQ2

How does representation choice change identifiability?

## RQ3

Can JAD recover identifiable interaction structure?

## RQ4

Does higher fidelity imply greater accessibility?

## RQ5

Do these effects transfer to natural multimodal datasets?

---

# 70. Updated Contribution Candidate

Potential new contribution:

> We show that multimodal interaction identifiability is representation-conditional: temporal aggregation can eliminate measurable joint predictive advantage.

Only make this claim if R1.2 provides supporting evidence.

---

# 71. Do Not Claim Before Evidence

Until R1.2:

do NOT state that pooling destroys interaction.

Correct current wording:

\[
\boxed{
\text{pooling is a plausible unresolved confound}.
}
\]

---

# 72. CVPR Reviewer Question

Maintain in:

```text
paper/reviewer_questions.md
```

new objection:

> Is your identifiability result an artifact of mean pooling?

R1.2 must answer it.

---

# 73. Another Reviewer Question

> Is the MOSEI emotion result affected by mismatched samples between processed features and official labels?

R1.1 must answer it.

---

# 74. Hard Data Rule

No real-world core result should depend on approximate sample matching when deterministic IDs are available.

---

# 75. Hard Selection Rule

No setting is selected because:

```text
test J is positive
one seed is strong
fear appears promising
test classification improves
```

Selection remains validation-only.

---

# 76. Hard Gate Rule

Current activation gate remains:

\[
\boxed{
J_{val}>0.01
}
\]

in:

\[
3/3
\]

screening seeds.

---

# 77. Gate Is Operational, Not Universal

Do not claim:

\[
0.01
\]

is a universal statistical threshold for interaction.

It is a pre-registered operational criterion for this project.

---

# 78. Statistical Robustness

Once a real-world setting passes screening:

run:

\[
5\text{ seeds}
\]

for final D0/D1/D2.

Screening remains:

\[
3\text{ seeds}.
\]

---

# 79. No Multiple-Comparison Fishing

Do not enumerate dozens of:

```text
emotions
subgroups
time windows
thresholds
mappings
```

until something passes.

The existing six emotions are allowed because they are canonical labels.

Do not invent arbitrary subgroups post hoc.

---

# 80. Sequence Length Ablation

Do NOT immediately test many:

```text
window sizes
temporal offsets
pooling kernels
attention spans
```

R1.2 should first use one canonical sequence representation.

---

# 81. Temporal Delay Hypothesis

Cross-modal delay:

\[
t+\Delta
\]

is scientifically plausible.

But it is not part of initial R1.2.

Only investigate temporal offset if sequence-aware R1 produces specific evidence motivating it.

---

# 82. R1.2 Must Stay Minimal

The goal is not to build a better sequence model.

The goal is to answer:

\[
\boxed{
\text{does temporal representation restore measurable joint identifiability?}
}
\]

---

# 83. Suggested Code

Add:

```text
src/experiments/multibench/mosei_alignment_audit.py
```

and:

```text
src/experiments/multibench/mosei_sequence_identifiability.py
```

Avoid rewriting the existing pooled runner.

---

# 84. Reuse Existing Predictor Code

The additive/joint predictor definitions should be shared between:

```text
mosei_identifiability.py
mosei_sequence_identifiability.py
```

to prevent protocol drift.

---

# 85. Shared Utility Recommendation

Create:

```text
src/experiments/multibench/identifiability_utils.py
```

containing:

```text
AdditivePredictor
JointPredictor
compute_r2
compute_joint_advantage
capacity_audit
seed_everything
mapping definitions
```

---

# 86. Unit Tests for R1.1

Required:

```text
canonical IDs unique
split assignments consistent
no duplicate segment IDs
labels aligned to expected IDs
no feature-similarity matching in final path
```

---

# 87. Unit Tests for R1.2

Required:

```text
padding masks correct
unimodal encoder sees one modality only
additive branches independent
joint product uses correct sources
capacity difference < 1% preferred
seed deterministic
test set unused for selection
```

---

# 88. Leakage Test

Explicitly assert that R1.2 training never sees:

```text
emotion labels
sentiment labels
test metrics
```

as optimization targets.

---

# 89. Emotion Labels Are Evaluation/Gating Metadata Only

Emotion labels may define canonical positive subsets after representations/predictors are trained.

They must not influence the predictor loss.

---

# 90. Alignment Artifact

Required final artifact:

```text
results/mosei/alignment/MOSEI_ALIGNMENT_AUDIT.md
```

---

# 91. Exact-Aligned Pooled Artifact

Required:

```text
results/mosei/identifiability_exact/
MOSEI_IDENTIFIABILITY_EXACT_REPORT.md
```

---

# 92. Sequence Artifact

Required:

```text
results/mosei/sequence_identifiability/
MOSEI_SEQUENCE_IDENTIFIABILITY_REPORT.md
```

---

# 93. Machine-Readable Result

For every setting save:

```json
{
  "representation": "pooled_or_sequence",
  "task": "global_or_emotion",
  "mapping": "VT_to_A",
  "seed": 1,
  "params_add": 0,
  "params_joint": 0,
  "add_train_r2": 0.0,
  "add_val_r2": 0.0,
  "add_test_r2": 0.0,
  "joint_train_r2": 0.0,
  "joint_val_r2": 0.0,
  "joint_test_r2": 0.0,
  "J_train": 0.0,
  "J_val": 0.0,
  "J_test": 0.0
}
```

---

# 94. Required Comparison Figure

If R1.2 is completed:

create a figure:

X-axis:

```text
VA→T
VT→A
AT→V
```

Y-axis:

\[
J_{val}.
\]

Show:

```text
pooled
sequence-aware
```

with seed variability.

---

# 95. Interpretation Case 1

If:

\[
J_{sequence}>J_{pooled}
\]

but both fail:

conclusion:

> Sequence representations expose more joint structure, but not enough to satisfy the pre-registered identifiability criterion.

Still freeze MOSEI.

---

# 96. Interpretation Case 2

If:

\[
J_{sequence}
\]

passes:

conclusion:

> Joint interaction is identifiable only after preserving temporal structure.

This is potentially a strong CVPR result.

---

# 97. Interpretation Case 3

If:

\[
J_{sequence}\le J_{pooled},
\]

conclusion:

> Temporal encoding does not explain the failed pooled screen under the tested architecture.

Freeze MOSEI.

---

# 98. R2 Trigger Remains Strict

R2 begins only if:

\[
\boxed{
J_{val}>0.01
}
\]

in:

\[
\boxed{3/3\text{ seeds}}
\]

for a frozen natural setting.

---

# 99. If R2 Trigger Fires

Immediately freeze:

```text
dataset version
alignment
representation
mapping
emotion subset
predictor architecture
J threshold
```

Then run:

```text
D0
D1
D2
```

with no method redesign.

---

# 100. Real-World D2 Question

The purpose of R2 is not:

> Can D2 beat every method?

It is:

> Does JAD behave on naturally identifiable multimodal structure as predicted by controlled synthetic experiments?

---

# 101. Real-World Success Modes

Valid results include:

### A

\[
D2>D0
\]

in linear accessibility.

### B

D2 matches D1 utility with stronger selectivity diagnostics.

### C

D1 remains more accessible, reproducing the synthetic fidelity–accessibility tradeoff.

### D

J magnitude predicts when D2 helps.

---

# 102. High-Value Result

Especially valuable:

\[
\boxed{
J\uparrow
\Rightarrow
\text{D2 utility/fidelity evidence strengthens}.
}
\]

This would connect screening with downstream behavior.

---

# 103. Paper-Level Correlation

Eventually across natural settings, test whether:

\[
J
\]

correlates with:

\[
\Delta_{D2-D0}.
\]

Do this only after several pre-defined settings exist.

Do not create arbitrary datasets/subsets solely for correlation.

---

# 104. MOSEI Negative Result Can Enter Paper

If MOSEI ultimately fails, include it.

Do not hide it.

Possible role:

\[
\boxed{
\text{natural non-identifiable / weakly identifiable case}.
}
\]

This is valuable for the framework.

---

# 105. Main Paper Table Could Include Screening

Example:

| Dataset | Representation | Mapping | \(J\) | Pass |
|---|---|---|---:|---|
| MOSEI | pooled | VT→A | | |
| MOSEI | sequence | VT→A | | |
| MUStARD | ... | ... | | |
| UR-FUNNY | ... | ... | | |

This supports the identifiability thesis directly.

---

# 106. CVPR Story Update

Potential final story:

1. Standard higher-order fusion assumes interaction is worth modeling.
2. We show interaction must first be identifiable.
3. Identifiability depends on data, representation, and hypothesis class.
4. IPIB provides controlled ground truth.
5. JAD isolates predictable non-additive structure.
6. Fidelity and accessibility differ.
7. Natural datasets vary strongly in identifiability.
8. JAD is useful only where the interaction is identifiable.

---

# 107. Updated Paper Thesis

\[
\boxed{
\textbf{
Multimodal interaction learning should be conditional on identifiable non-additive predictive structure rather than automatically instantiated whenever multiple modalities are available.
}
}
\]

---

# 108. Current Forbidden Work

Until R1.1/R1.2 finish:

```text
NO D0/D1/D2 MOSEI R2
NO lower J threshold
NO test-based setting selection
NO cross-attention
NO multimodal encoder in screening
NO rank sweep
NO task loss
NO third-order discovery
NO UR-FUNNY tuning
NO C1/C2 reliability revival
NO new D2 loss
```

---

# 109. Current Permitted Work

Only:

```text
exact data alignment
canonical cache regeneration
pooled R1 reproduction
sequence-aware unimodal screening
paper theory writing
paper related work
figure preparation
```

---

# 110. Theory Work Can Continue in Parallel

Formalize:

\[
q_A^*=A
\]

\[
q_J^*=A+J
\]

thus:

\[
q_J^*-q_A^*=J.
\]

Finite predictor:

\[
q_J-q_A
=
J+
(\epsilon_J-\epsilon_A).
\]

Then extend notation:

\[
J=J(\mathcal D,\phi,\mathcal H_A,\mathcal H_J).
\]

---

# 111. Clarify Terminology

Do NOT use:

```text
true synergy
PID synergy
causal interaction
new information
```

unless formally justified.

Use:

```text
joint predictive advantage
non-additive predictive structure
predictive interaction
representation-conditional identifiability
```

---

# 112. Avoid “Causal” Unless Proven

D2 selectivity does not automatically imply:

\[
\text{causal interaction}.
\]

Use:

\[
\boxed{
\text{predictive interaction}
}
\]

not causal interaction.

---

# 113. Paper Claim After v5.1

If MOSEI fails:

safe claim:

> Joint predictive interaction can be weak or absent under standard multimodal representations, motivating interaction screening before higher-order fusion.

If sequence passes:

safe claim:

> Interaction identifiability can depend strongly on representation granularity.

---

# 114. Real-World Dataset Search Rule

After MOSEI, dataset selection must be motivated by expected multimodal interaction structure.

Do not search dozens of datasets until one gives a positive result.

---

# 115. Next Dataset Candidate

Primary after MOSEI:

\[
\boxed{
\text{MUStARD}
}
\]

because multimodal incongruity is central to sarcasm.

---

# 116. Second Candidate

Then:

\[
\boxed{
\text{UR-FUNNY}.
}
\]

Use existing historical evidence as context, not as tuning target.

---

# 117. Optional External Candidate

Only if MultiBench settings all fail:

consider:

```text
IEMOCAP
MELD
```

but only with a clear interaction hypothesis and time budget.

---

# 118. CVPR Time Discipline

Every experiment must answer one of:

```text
Does exact alignment change identifiability?
Does temporal representation change identifiability?
Does JAD work on a setting that passes screening?
Does J predict downstream usefulness?
Does the fidelity–accessibility tradeoff transfer?
```

Otherwise:

\[
\boxed{\text{DO NOT RUN IT}.}
\]

---

# 119. Immediate Execution Order

```text
1. Audit canonical MOSEI IDs.

2. Build deterministic feature-label alignment.

3. Produce MOSEI_ALIGNMENT_AUDIT.md.

4. Regenerate exact-aligned cache if required.

5. Re-run pooled R1 unchanged.

6. Check 3/3 gate.

7. If pass:
       freeze setting
       go to R2.

8. If fail:
       implement sequence-aware unimodal representation.

9. Run sequence global R1.

10. Run canonical emotion-conditioned diagnostics.

11. Check 3/3 gate.

12. If pass:
        freeze setting
        go to R2.

13. If fail:
        freeze MOSEI negative result.

14. Screen MUStARD.

15. Then screen UR-FUNNY.
```

---

# 120. Immediate Milestone

The next milestone is NOT:

\[
\text{improve D2 accuracy}.
\]

It is:

\[
\boxed{
\textbf{determine whether MOSEI's failed identifiability is caused by data alignment or representation granularity.}
}
\]

---

# 121. Hard MOSEI Stop

After:

\[
R1.1
\]

and:

\[
R1.2,
\]

if no setting satisfies the frozen gate:

\[
\boxed{
\textbf{STOP MOSEI DEVELOPMENT}.
}
\]

Do not create R1.3.

---

# 122. Why No R1.3

Otherwise the project risks:

```text
dataset mining
threshold tuning
architecture fishing
post-hoc subgroup selection
```

which weakens a CVPR submission.

---

# 123. Pair Method Remains Frozen

Regardless of MOSEI outcome:

\[
\boxed{
\textbf{canonical JAD remains the final pair method}.
}
\]

Natural-data failure does not automatically reopen synthetic method search.

---

# 124. Main Scientific Principle

\[
\boxed{
\textbf{
Interaction should be discovered only when non-additive predictive structure is identifiable under the representation and hypothesis classes being used.
}
}
\]

---

# 125. Final v5.1 Thesis

\[
\boxed{
\textbf{
Multimodal interaction identifiability is representation-conditional; before learning explicit higher-order representations, one must establish that a joint hypothesis class predicts cross-modal targets beyond a capacity-matched additive class.
}
}
\]

---

# 126. Current Required Experiment

\[
\boxed{
\textbf{R1.1 — CMU-MOSEI Exact Alignment Audit}
}
\]

followed, only if pooled screening still fails, by:

\[
\boxed{
\textbf{R1.2 — Sequence-Aware Identifiability Audit}.
}
\]

No R2 is permitted until one natural setting satisfies:

\[
\boxed{
J_{val}>0.01\quad\text{for all 3 screening seeds}.
}
\]