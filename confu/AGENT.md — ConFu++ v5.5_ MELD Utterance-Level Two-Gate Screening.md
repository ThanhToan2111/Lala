# AGENT.md

# ConFu++ v5.5

## MELD Utterance-Level Two-Gate Natural Interaction Screening

---

# 0. Mission

This repository develops **ConFu++**, a framework for understanding and discovering explicit higher-order multimodal interaction.

The target is a **CVPR main-track paper**.

The project no longer assumes that multimodal interaction should be trained simply because multiple modalities are available.

The current central principle is:

\[
\boxed{
\text{Cross-modal interaction must first be identifiable}
}
\]

and independently:

\[
\boxed{
\text{the downstream task must exhibit measurable joint headroom}.
}
\]

Only after both properties are demonstrated is interaction representation learning scientifically justified.

v5.5 applies this principle to:

\[
\boxed{
\textbf{MELD}
}
\]

using utterance-level multimodal representations.

---

# 1. Current Project State

```text
Synthetic Oracle:
PASS

NIC non-identifiable control:
PASS

IPIB:
PASS

Canonical JAD:
FROZEN

Synthetic fidelity:
PASS

Synthetic false-positive control:
PASS

Synthetic Fidelity–Accessibility gap:
ESTABLISHED

MOSEI exact-source G1:
PASS

MOSEI sentiment G2:
INCONCLUSIVE

MOSEI JAD:
COMPLETE

MOSEI downstream gain:
FAIL

MOSEI shortcut audit:
COMPLETE

MOSEI purification:
FAIL

MOSEI sentiment:
FROZEN

MUStARD G1:
FAIL ALL PAIRS

MUStARD G2:
FAIL / WEAK-UNSTABLE

MUStARD Type IV:
NONE

MUStARD JAD:
NOT ALLOWED

MUStARD:
FROZEN

Next:
MELD utterance-level two-gate screening

Task-aware JAD:
BLOCKED

Third-order discovery:
BLOCKED
```

---

# 2. Frozen Pair Method

The canonical pair method remains:

\[
\boxed{
\textbf{Joint-Advantage Distillation — JAD}
}
\]

For source modalities:

\[
r_i,r_j
\]

and target modality:

\[
r_k,
\]

train:

\[
q_{\text{add}}
=
q_i(r_i)+q_j(r_j)
\]

and a capacity-matched:

\[
q_{\text{joint}}(r_i,r_j).
\]

Define:

\[
\boxed{
d_{ij\rightarrow k}
=
q_{\text{joint}}
-
q_{\text{add}}.
}
\]

Interaction representation:

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
W
\left(
\frac{
u_i\odot u_j
}{
\sqrt{R}
}
\right)
}
\]

with canonical:

\[
\boxed{R=64}.
\]

Distillation loss:

\[
\boxed{
L_{JAD}
=
1-\cos(h_{ij},d_{ij\rightarrow k}).
}
\]

Do not modify JAD during v5.5 screening.

---

# 3. What Has Been Learned So Far

Three different failures have now been observed.

### Failure A

Multimodal data may not contain stable non-additive cross-modal predictability.

MUStARD demonstrates this regime.

### Failure B

Cross-modal interaction can be identifiable without demonstrated downstream task interaction.

MOSEI approximately demonstrates this regime.

### Failure C

A faithfully learned interaction representation may not improve downstream accessibility.

Synthetic and MOSEI experiments demonstrate this distinction.

Therefore:

\[
\boxed{
\text{multimodality}
\neq
\text{identifiability}
\neq
\text{task relevance}
\neq
\text{fidelity}
\neq
\text{accessibility}.
}
\]

---

# 4. Current Interaction Hierarchy

The preferred framework is:

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
\text{Representation Fidelity}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Downstream Accessibility}.
}
\]

Each level must be evaluated independently.

---

# 5. Two-Gate Protocol

A natural multimodal setting must pass:

\[
\boxed{
G_1:\text{ Cross-Modal Identifiability}
}
\]

and:

\[
\boxed{
G_2:\text{ Task Joint Headroom}.
}
\]

Only then may canonical JAD be trained.

---

# 6. Gate G1

Given source modalities:

\[
(i,j)
\]

and target modality:

\[
k,
\]

define:

\[
\boxed{
J_{\text{cross}}
=
R^2(q_{\text{joint}},r_k)
-
R^2(q_{\text{add}},r_k).
}
\]

G1 asks:

> Does joint access to two modalities predict the third modality better than capacity-matched additive access?

---

# 7. G1 Frozen Operational Criterion

Initial screening:

\[
3\text{ seeds}.
\]

Preferred gate:

\[
\boxed{
J_{\text{cross,val}}>0.01
}
\]

for:

\[
\boxed{3/3\text{ seeds}}.
\]

Test results are diagnostic only.

---

# 8. Gate G2

For downstream target:

\[
Y,
\]

train:

\[
c_{\text{add}}
\]

and capacity-matched:

\[
c_{\text{joint}}.
\]

Define task headroom:

\[
\boxed{
H_{\text{task}}
=
Perf(c_{\text{joint}})
-
Perf(c_{\text{add}}).
}
\]

G2 asks:

> Does explicit joint access improve prediction of the downstream task beyond additive modality contributions?

---

# 9. Generic Nonlinear Control

Also train:

\[
c_{\text{MLP}}([r_i,r_j]).
\]

Purpose:

\[
\boxed{
\text{test whether nonlinear headroom exists outside the explicit product class}.
}
\]

Do not confuse the generic MLP with canonical JAD.

---

# 10. Natural Setting Types

Classify each source pair.

### Type I

\[
G_1=FAIL,
\quad
G_2=FAIL.
\]

No demonstrated reason for explicit interaction learning.

### Type II

\[
G_1=PASS,
\quad
G_2=FAIL/INCONCLUSIVE.
\]

Cross-modal interaction exists but task relevance is not demonstrated.

### Type III

\[
G_1=FAIL,
\quad
G_2=PASS.
\]

Task interaction exists but canonical cross-modal JAD target is not supported.

### Type IV

\[
\boxed{
G_1=PASS,
\quad
G_2=PASS.
}
\]

This is the desired canonical JAD setting.

---

# 11. Current Natural Dataset Map

Conceptually:

```text
MOSEI:
G1 PASS
G2 INCONCLUSIVE
≈ Type-II-like

MUStARD VA:
G1 FAIL
G2 FAIL
Type I

MUStARD VT:
G1 FAIL
G2 WEAK/UNSTABLE
INCONCLUSIVE

MUStARD AT:
G1 FAIL
G2 WEAK/UNSTABLE
INCONCLUSIVE
```

No natural Type-IV setting has yet been established.

---

# 12. MUStARD Freeze

MUStARD is now frozen.

Do NOT:

```text
run JAD
run 5-seed confirmation
reduce J threshold
change representation
use sequence rescue
increase rank
change interaction architecture
change loss
use task labels in G1
```

under v5.5.

---

# 13. Why MUStARD Is Frozen

MUStARD G1 failed for all three mappings.

Therefore canonical JAD is not justified under the tested pooled representation.

G2 weak accuracy changes do not override failed G1.

---

# 14. v5.5 Dataset

The next dataset is:

\[
\boxed{
\textbf{MELD}
}
\]

Primary downstream task:

\[
\boxed{
\textbf{multiclass emotion recognition}.
}
\]

---

# 15. Why Emotion First

Do not initially use MELD sentiment.

Emotion is preferred because visual and acoustic cues plausibly contain task-relevant information beyond text.

This is a hypothesis.

The experiment must test it.

---

# 16. Representation Level

v5.5 uses:

\[
\boxed{
\textbf{utterance-level representations only}.
}
\]

Do not introduce dialogue context during the initial screening.

---

# 17. Why No Dialogue Context Yet

Dialogue context introduces another source of predictive information:

\[
C_{<t}.
\]

Then interaction could arise from:

```text
modality interaction
dialogue history
speaker history
context leakage
```

and interpretation becomes difficult.

Therefore initial MELD screening must isolate:

\[
V_t,A_t,T_t.
\]

---

# 18. Context-Aware MELD

Context-aware experiments are:

\[
\boxed{\text{BLOCKED}}
\]

until utterance-level screening is complete.

Only open context later if scientifically motivated.

---

# 19. MELD Data Audit

Before any model training, produce a canonical data audit.

Required:

```text
dataset source
dataset version
sample/utterance ID
dialogue ID
speaker ID if available
train count
validation count
test count
emotion class counts
vision shape
audio shape
text shape
missing modalities
zero vectors
NaNs
Infs
sequence lengths
pooling rule
```

---

# 20. Canonical Identity Requirement

Each utterance should retain:

```text
dialogue_id
utterance_id
split
emotion_label
vision
audio
text
```

Do not approximate-match labels/features from unrelated files if canonical IDs are available.

---

# 21. MELD Primary Labels

Use the canonical emotion labels supplied by the selected dataset pipeline.

Do not remap or merge emotion classes during the primary screening.

---

# 22. Class Imbalance Audit

Report per-class counts for:

```text
train
validation
test
```

before G2.

This matters because:

\[
Accuracy
\]

alone may be misleading.

---

# 23. MELD Modalities

Use:

```text
V = vision
A = audio
T = text
```

with utterance-level representations.

---

# 24. Sequence Handling

If inputs are sequence tensors:

\[
X_m\in\mathbb R^{L\times d_m},
\]

initial v5.5 representation should use a single deterministic pooling procedure.

Preferred:

\[
\boxed{
\text{masked mean pooling over valid frames/tokens}
}
\]

if compatible with the available MELD features.

---

# 25. No Pooling Sweep

Do NOT compare:

```text
mean pooling
max pooling
attention pooling
Transformer pooling
last token
```

during screening.

One representation only.

---

# 26. Representation Standardization

Standardize each modality using:

\[
\boxed{
\text{train statistics only}.
}
\]

Apply frozen statistics to validation/test.

---

# 27. G1 MELD Directions

Run all:

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

---

# 28. G1 Predictors

Reuse the established capacity-controlled predictor families.

Additive:

\[
q_{\text{add}}
=
q_i(r_i)+q_j(r_j).
\]

Joint:

\[
q_{\text{joint}}
\]

with explicit multiplicative interaction.

Do not write MELD-specific high-capacity predictors.

---

# 29. G1 Capacity Matching

Require:

\[
\frac{
|P_J-P_A|
}{
P_A
}
<5\%.
\]

Preferred:

\[
\boxed{<1\%}.
\]

Always report exact parameter counts.

---

# 30. G1 Optimization

Use the same optimization family across all mappings where practical.

Record:

```text
optimizer
learning rate
weight decay
batch size
max epochs
early stopping patience
```

---

# 31. G1 Early Stopping

Use validation target-prediction loss.

Do not early stop using:

\[
J_{\text{cross}}
\]

directly if existing protocol uses prediction MSE.

Avoid modifying training based on the desired gate.

---

# 32. G1 Metrics

For every seed report:

\[
R^2_{add,train}
\]

\[
R^2_{joint,train}
\]

\[
J_{train}
\]

\[
R^2_{add,val}
\]

\[
R^2_{joint,val}
\]

\[
J_{val}
\]

\[
R^2_{add,test}
\]

\[
R^2_{joint,test}
\]

\[
J_{test}.
\]

---

# 33. G1 Pass

Preferred:

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

# 34. G1 Weak Candidate

If only:

\[
2/3
\]

seeds pass:

classify:

```text
WEAK_UNSTABLE
```

Do not promote.

---

# 35. G1 Overfit Pattern

If:

\[
J_{train}>0
\]

but:

\[
J_{val}\le0,
\]

classify this as possible joint-model overfitting.

Do not treat training advantage as interaction evidence.

---

# 36. G2 Task

Primary G2 target:

\[
\boxed{
Y_{\text{emotion}}
}
\]

with multiclass classification.

---

# 37. G2 Source Pairs

Evaluate:

\[
VA\rightarrow Y
\]

\[
VT\rightarrow Y
\]

\[
AT\rightarrow Y.
\]

Each corresponds directly to its G1 source pair.

---

# 38. G2 Models

For each source pair train:

```text
M0 Linear
M1 Additive
M2 Joint Product
M3 Generic Concat MLP
```

---

# 39. M0 Linear

Use:

\[
c_L([r_i,r_j]).
\]

This is a simple reference baseline.

It is not parameter-matched to M1/M2/M3.

---

# 40. M1 Additive

Use independent branches:

\[
z_i=f_i(r_i)
\]

\[
z_j=f_j(r_j)
\]

and logits:

\[
\boxed{
\ell_{\text{add}}
=
g_i(z_i)+g_j(z_j).
}
\]

No cross-modal multiplication or concatenation before branch prediction.

---

# 41. M2 Joint Product

Use:

\[
u_i=P_ir_i
\]

\[
u_j=P_jr_j
\]

and:

\[
\boxed{
\ell_{\text{joint}}
=
W[
u_i;
u_j;
u_i\odot u_j
].
}
\]

This is the explicit task interaction model.

---

# 42. M3 Generic MLP

Use:

\[
\boxed{
\ell_{\text{MLP}}
=
MLP([r_i,r_j]).
}
\]

Purpose:

detect generic nonlinear task headroom outside the product class.

---

# 43. G2 Capacity Matching

M1/M2/M3 should be approximately capacity matched.

Preferred:

\[
<1\%
\]

parameter difference.

Maximum:

\[
<5\%.
\]

Do not distort architectures solely to achieve exact equality.

---

# 44. Multiclass Loss

Use:

\[
\boxed{
CrossEntropyLoss
}
\]

with logits over MELD emotion classes.

---

# 45. Class Weighting

Do not automatically add class weights.

Use the canonical existing protocol first.

If weighting is required by the established MELD benchmark pipeline, document it and apply it equally to all G2 models.

---

# 46. Primary G2 Metrics

Report:

\[
\boxed{
MacroF1
}
\]

\[
\boxed{
WeightedF1
}
\]

\[
Accuracy
\]

and:

\[
\boxed{
CrossEntropy / NLL.
}
\]

---

# 47. Why Macro F1 Matters

MELD emotion classes may be imbalanced.

Macro F1 gives equal importance to classes.

Therefore it should be one of the main task-headroom metrics.

---

# 48. Weighted F1

Weighted F1 can be reported for comparison with common MELD evaluation conventions.

Do not use Weighted F1 alone to declare G2 PASS.

---

# 49. Task Headroom Metrics

Define:

\[
H_{\text{task}}^{MacroF1}
=
MacroF1_{\text{joint}}
-
MacroF1_{\text{add}}.
\]

Similarly:

\[
H_{\text{task}}^{WeightedF1}
\]

\[
H_{\text{task}}^{Acc}.
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

Positive values favor joint modeling.

---

# 50. Generic Headroom

Define:

\[
H_{\text{MLP}}
=
Perf(c_{\text{MLP}})
-
Perf(c_{\text{add}}).
\]

This is a hypothesis-class diagnostic.

---

# 51. G2 Screening Seeds

Initial:

\[
\boxed{
3\text{ seeds}.
}
\]

Do not start with 5 seeds for every pair.

---

# 52. G2 Pass Philosophy

A promising explicit G2 result should show:

```text
positive mean Joint−Additive
consistent sign across seeds
support from Macro F1
support from NLL or another non-threshold metric
```

Do not declare G2 PASS from accuracy alone.

---

# 53. G2 Strong Candidate

Preferred three-seed candidate:

\[
Joint>Additive
\]

for Macro F1 in:

\[
3/3
\]

seeds,

with positive mean NLL improvement.

This may be promoted to confirmatory evaluation.

---

# 54. G2 Confirmation

For a promising frozen setting:

run:

\[
\boxed{
5\text{ seeds total}
}
\]

without changing:

```text
architecture
split
optimizer
representation
early stopping
```

---

# 55. Confirmed G2 Pass

Strong evidence if:

\[
Joint>Additive
\]

in at least:

\[
4/5
\]

seeds on primary validation metric,

with supporting metrics pointing in the same direction.

---

# 56. Generic-Only G2

If:

\[
MLP>Additive
\]

but:

\[
JointProduct\le Additive,
\]

classify:

\[
\boxed{
\text{GENERIC_NONLINEAR_HEADROOM_ONLY}.
}
\]

Do not train canonical JAD automatically.

---

# 57. G1/G2 Pair Matching

A Type-IV candidate requires the **same source pair**.

Example:

\[
VA\rightarrow T
\]

for G1 and:

\[
VA\rightarrow Y_{\text{emotion}}
\]

for G2.

Do not combine:

\[
VA\rightarrow T
\]

with:

\[
AT\rightarrow Y.
\]

---

# 58. MELD Two-Gate Matrix

Produce:

| Pair | Cross-modal mapping | \(J_{cross}\) | G1 | Task pair | \(H_{task}\) | G2 | Type |
|---|---|---:|---|---|---:|---|---|
| VA | VA→T | | | VA→Emotion | | | |
| VT | VT→A | | | VT→Emotion | | | |
| AT | AT→V | | | AT→Emotion | | | |

---

# 59. Type-IV Trigger

Only:

\[
\boxed{
G_1=PASS
}
\]

and:

\[
\boxed{
G_2=PASS
}
\]

for the same source pair automatically permits:

\[
\boxed{
\text{MELD canonical JAD R2}.
}
\]

---

# 60. If Multiple Type-IV Pairs Exist

Choose a primary development pair using validation evidence only.

Criteria:

```text
G1 stability
G2 stability
generalization gap
sample support
capacity fairness
```

Do not choose by test performance.

---

# 61. If No Type IV Exists

Do not modify JAD.

First classify why.

Possible:

```text
all G1 fail
G1 pass but G2 fail
G2 pass but G1 fail
generic nonlinear only
high variance / inconclusive
```

The reason determines the paper interpretation.

---

# 62. MELD R2 Triggered Method Set

If Type IV exists, compare exactly:

\[
\boxed{
LowerOrder
}
\]

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
D2/JAD.
}
\]

---

# 63. R2 D0

Target:

\[
\boxed{
t_{D0}=r_k.
}
\]

---

# 64. R2 D1

Target:

\[
\boxed{
t_{D1}
=
r_k-q_{\text{add}}.
}
\]

---

# 65. R2 D2

Target:

\[
\boxed{
t_{D2}
=
q_{\text{joint}}-q_{\text{add}}.
}
\]

---

# 66. R2 Fairness

D0/D1/D2 must share:

```text
same modality representations
same interaction rank
same architecture
same optimizer
same epoch budget
same seeds
same downstream probe
```

Only target differs.

---

# 67. R2 Seeds

Final positive natural benchmark:

\[
\boxed{
5\text{ seeds}.
}
\]

---

# 68. R2 Lower-Order Representation

For selected pair:

\[
Z_{low}
=
[r_i,r_j].
\]

---

# 69. Interaction-Augmented Representation

\[
Z_{int}
=
[r_i,r_j,h_{ij}].
\]

Define:

\[
\boxed{
\Delta_{\text{task}}
=
Perf(Z_{int})
-
Perf(Z_{low}).
}
\]

---

# 70. R2 Main Metrics

For MELD emotion report:

```text
Macro F1
Weighted F1
Accuracy
NLL
```

---

# 71. R2 Interaction Diagnostics

Retain:

```text
target cosine
effective rank
variance
source shuffle
single-source reconstruction
additive reconstruction
joint reconstruction
```

These diagnose mechanism.

---

# 72. Single-Modality Leakage

Do not automatically classify single-modality reconstructability as failure.

P1 already showed that aggressive removal can destroy joint structure.

Use leakage as diagnostic only.

---

# 73. R2 Desired Outcome

Ideal:

\[
\boxed{
D2>D0
}
\]

on Macro F1 or another primary task metric,

with healthy interaction fidelity.

---

# 74. Stronger Desired Outcome

Preferred:

\[
D2>D0
\]

and:

\[
D2\ge D1.
\]

This is not mandatory.

---

# 75. Critical Type-IV Failure

If MELD is genuinely Type IV but:

\[
D2\le D0
\]

with no stable accessibility gain,

then canonical JAD is substantially challenged.

Do not invent a new method immediately.

First diagnose whether:

```text
cross-modal interaction
task-relevant interaction
```

occupy different directions.

---

# 76. Task-Aware JAD Status

Still:

\[
\boxed{
\textbf{BLOCKED}.
}
\]

It becomes scientifically discussable only if:

\[
G_1=PASS,
\]

\[
G_2=PASS,
\]

but canonical JAD fails.

---

# 77. No Task Supervision Inside Canonical JAD

Even on MELD Type IV:

canonical JAD remains label-free during interaction-target construction.

Emotion labels are used only:

```text
G2
downstream probing
evaluation
```

not for:

\[
q_{\text{add}},
q_{\text{joint}},
d.
\]

---

# 78. Context-Aware Extension

Context-aware MELD is allowed only after utterance-level experiment reaches a scientific conclusion.

Possible reason:

> utterance-level G2 is weak but established MELD performance strongly depends on conversational context.

Still requires a separately specified experiment.

---

# 79. Do Not Rescue a Failed MELD Screen

If utterance-level G1 fails:

do NOT immediately add dialogue context simply to obtain a positive \(J\).

A context experiment must have independent motivation.

---

# 80. Dataset Aggregate Table

Maintain:

```text
paper/tables/two_gate_natural.csv
```

with columns:

```text
dataset
representation
pair
target_modality
J_cross_val
G1
task
H_task_val
G2
type
```

---

# 81. Current Aggregate Table

Initial entries:

```text
MOSEI
MUStARD
MELD
```

Add MELD only after v5.5 completes.

---

# 82. Paper 2×2 Figure

Maintain a natural-regime plot:

```text
                     Task headroom -
                     Task headroom +

Cross-modal G1 -
    Type I
    Type III

Cross-modal G1 +
    Type II
    Type IV
```

Place dataset-pair settings as points only after evidence is finalized.

---

# 83. MOSEI Position

Use conservative label:

```text
G1 PASS
G2 INCONCLUSIVE
Type-II-like
```

Do not force it into definitive Type II.

---

# 84. MUStARD Position

VA:

\[
\boxed{
\text{Type I}.
}
\]

VT/AT:

\[
\boxed{
\text{Inconclusive}
}
\]

because G2 was weak/unstable although G1 failed.

---

# 85. MELD Goal

The scientific goal is not to make MELD Type IV.

The goal is to measure which regime it belongs to.

Type IV is desirable, not mandatory.

---

# 86. If MELD Is Type IV

Freeze the pair immediately.

Do not continue searching for a larger \(J\).

Proceed to canonical R2.

---

# 87. If MELD Is Type II

This provides a second natural example that:

\[
\boxed{
\text{cross-modal identifiability}
\not\Rightarrow
\text{task headroom}.
}
\]

This strengthens diagnostics but weakens the positive method story.

---

# 88. If MELD Is Type III

This would be scientifically interesting:

\[
\boxed{
\text{task interaction exists but canonical cross-modal prediction does not expose it}.
}
\]

This would motivate future task-aware interaction research.

Do not open that method during screening.

---

# 89. If MELD Is Type I

Then three natural datasets will have failed to provide a clean Type-IV setting.

At that point do NOT blindly continue dataset search.

Reassess the paper direction.

---

# 90. Type-I/II-Only Stop Rule

If MELD also produces no Type IV:

pause natural dataset expansion.

Evaluate whether the main paper should become:

\[
\boxed{
\text{a diagnostic / scientific framework paper}
}
\]

rather than a performance-improvement paper.

Do not immediately add five more datasets.

---

# 91. Possible Next Dataset After MELD

UR-FUNNY remains available as a transfer/control benchmark.

But it should not become another positive-result fishing attempt.

Use only if needed to complete the paper's regime analysis.

---

# 92. Third-Order Status

Still:

\[
\boxed{
\textbf{BLOCKED}.
}
\]

No \(h_{123}\) until at least one natural Type-IV pair has been fully understood.

---

# 93. Synthetic Status

IPIB remains frozen.

Do not reopen synthetic experiments during MELD screening.

---

# 94. Current Theoretical View

For modality target:

\[
T_k,
\]

conceptually:

\[
T_k
=
A_k(X_i,X_j)
+
J_k(X_i,X_j)
+
\epsilon_k.
\]

G1 asks whether:

\[
J_k
\]

is operationally detectable relative to:

\[
\mathcal H_{add},
\mathcal H_{joint}.
\]

---

# 95. Task View

For downstream target:

\[
Y,
\]

conceptually:

\[
Y
=
A_Y(X_i,X_j)
+
J_Y(X_i,X_j)
+
\epsilon_Y.
\]

G2 asks whether:

\[
J_Y
\]

produces measurable predictive headroom.

---

# 96. Key Non-Equivalence

There is no guarantee:

\[
\boxed{
J_k=J_Y.
}
\]

This is central to the paper.

---

# 97. JAD Applicability

Canonical JAD learns from:

\[
J_k.
\]

Downstream benefit requires useful overlap with:

\[
J_Y.
\]

This overlap is currently not explicitly identified.

Do not claim otherwise.

---

# 98. Updated Main Research Question

\[
\boxed{
\textbf{
When does cross-modal predictable interaction correspond to task-relevant multimodal interaction?
}
}
\]

---

# 99. Current Main Claim Candidate

> Higher-order multimodal interaction should not be assumed from modality count. Cross-modal identifiability and task joint headroom are distinct and should be independently established before explicit interaction learning.

---

# 100. Paper Contribution Candidate A

A framework separating:

```text
cross-modal identifiability
task joint headroom
interaction fidelity
downstream accessibility
```

---

# 101. Paper Contribution Candidate B

IPIB:

\[
\boxed{
\text{controlled identifiable interaction benchmark}.
}
\]

---

# 102. Paper Contribution Candidate C

JAD:

\[
\boxed{
\text{label-free distillation of predictable non-additive cross-modal structure}.
}
\]

---

# 103. Paper Contribution Candidate D

Empirical evidence that:

\[
\boxed{
\text{cross-modal identifiability}
\not\Rightarrow
\text{task headroom}.
}
\]

---

# 104. Paper Contribution Candidate E

Empirical evidence that:

\[
\boxed{
\text{interaction fidelity}
\not\Rightarrow
\text{accessibility}.
}
\]

---

# 105. Desired Positive Contribution

If MELD Type IV and JAD succeeds:

\[
\boxed{
\text{two-gate screening predicts a setting where explicit interaction learning becomes useful}.
}
\]

This would substantially strengthen the paper.

---

# 106. MELD Code Structure

Recommended:

```text
src/experiments/multibench/meld/
    data.py
    data_audit.py
    g1_cross_modal.py
    g2_task_headroom.py
    two_gate_screening.py
```

Reuse common predictors from existing screening code.

---

# 107. Shared Predictor Requirement

Do not duplicate slightly different implementations across:

```text
MOSEI
MUStARD
MELD
```

where possible.

Reuse:

```text
AdditivePredictor
JointPredictor
TaskAdditive
TaskJointProduct
ConcatMLP
```

---

# 108. Result Structure

Recommended:

```text
results/meld/v55/
    data_audit/
    g1_cross_modal/
    g2_task_headroom/
    two_gate_summary/
```

---

# 109. Data Audit Artifact

Produce:

```text
MELD_DATA_AUDIT.md
```

and machine-readable:

```text
meld_data_audit.json
```

---

# 110. G1 Artifact

Produce:

```text
MELD_G1_CROSS_MODAL_REPORT.md
```

---

# 111. G2 Artifact

Produce:

```text
MELD_G2_TASK_HEADROOM_REPORT.md
```

---

# 112. Final Decision Artifact

Produce:

```text
MELD_TWO_GATE_DECISION.md
```

and:

```text
meld_two_gate.json
```

---

# 113. Machine-Readable Fields

For G1:

```text
mapping
seed
sources
target
parameter counts
train R2
validation R2
test R2
J train
J validation
J test
```

For G2:

```text
pair
seed
model
parameter count
best epoch
train loss
validation loss
test loss
accuracy
macro F1
weighted F1
```

---

# 114. Reproducibility

Seed:

```text
Python
NumPy
PyTorch
CUDA
DataLoader
model initialization
```

where relevant.

---

# 115. Required Unit Tests

At minimum test:

```text
canonical split identity
sample IDs unique
label alignment
modality shapes
masked pooling
train-only normalization
mapping direction
capacity matching
additive branch independence
joint product correctness
multiclass output dimension
validation-only selection
test not used for gates
seed determinism
```

---

# 116. Smoke-Test Rule

Run seed 1 first only to verify:

```text
pipeline executes
loss decreases
shapes correct
no NaNs
checkpoints save
metrics compute
```

Do not interpret smoke-test performance scientifically.

---

# 117. Full Screening

After smoke tests pass:

run:

\[
3\text{ seeds}
\]

for:

```text
G1: all mappings
G2: all source pairs
```

---

# 118. Decision First

Before running any JAD:

generate the complete:

\[
\boxed{
\text{Two-Gate Matrix}.
}
\]

---

# 119. No Partial JAD

Do not run D2 on a promising mapping while waiting for G2.

Both gate decisions must be finalized first.

---

# 120. Test Leakage Rule

Do not use test results for:

```text
pair selection
G1 pass
G2 pass
hidden dimension
class weighting
architecture
early stopping policy
```

---

# 121. Multiple-Comparison Rule

Do not search:

```text
emotion subsets
speaker subsets
dialogue subsets
rare classes
binary reformulations
```

until the preregistered multiclass screening is complete.

---

# 122. No Binary Emotion Rescue

If multiclass G2 fails:

do not immediately transform MELD into:

```text
positive vs negative
happy vs rest
anger vs rest
```

to seek headroom.

That requires a separate hypothesis.

---

# 123. No Sequence Rescue During v5.5

Initial representation is fixed.

Do not introduce temporal Transformer encoding solely because pooled G1 fails.

Representation changes require a new version and scientific rationale.

---

# 124. No Context Rescue During v5.5

Same rule for dialogue context.

No post-hoc context addition to make Type IV appear.

---

# 125. Current Forbidden Work

During MELD screening:

```text
NO JAD modification
NO task-aware JAD
NO purification
NO rank sweep
NO loss sweep
NO cross-attention
NO dialogue-context tuning
NO third-order
NO binary-label rescue
NO test-based selection
NO dataset-specific hyperparameter fishing
```

---

# 126. Current Permitted Work

Only:

```text
MELD canonical data audit
MELD G1
MELD G2
two-gate decision
paper theory
paper tables
paper figures
related work
```

---

# 127. Paper Table Maintenance

Maintain one aggregate natural screening table.

Do not wait until submission to reconstruct experimental history.

---

# 128. Claims File

Update:

```text
paper/claims.md
```

with:

```text
Claim
Evidence
Experiment
Dataset
Limitations
Status
```

---

# 129. Reviewer Question

Add:

> Why should cross-modal predictability imply downstream utility?

Answer:

\[
\boxed{
\text{It should not.}
}
\]

The two-gate framework explicitly tests them separately.

---

# 130. Reviewer Question

Add:

> Why not simply use task supervision to learn interactions?

Answer:

Canonical JAD investigates label-free multimodal predictive structure.

Task supervision addresses a different problem.

---

# 131. Reviewer Question

Add:

> Why are some multimodal benchmarks not improved by interaction learning?

Potential evidence:

```text
G1 absent
G2 absent
or cross-modal/task interaction mismatch
```

rather than merely model weakness.

---

# 132. Immediate Execution Order

```text
1. Locate canonical MELD source used by the repository.

2. Audit IDs, splits, modality shapes, emotion labels.

3. Freeze utterance-level representation.

4. Save MELD_DATA_AUDIT.md.

5. Reuse/implement G1 predictors.

6. Verify capacity matching.

7. Run G1 seed-1 smoke test.

8. Run G1 seeds 1..3:
   VA -> T
   VT -> A
   AT -> V

9. Implement/reuse multiclass G2 models.

10. Verify Additive / Joint / MLP capacity.

11. Run G2 seed-1 smoke tests.

12. Run G2 seeds 1..3:
    VA -> Emotion
    VT -> Emotion
    AT -> Emotion

13. Produce paired validation metrics.

14. Build MELD two-gate matrix.

15. Classify each pair:
    TYPE_I
    TYPE_II
    TYPE_III
    TYPE_IV
    INCONCLUSIVE

16. If Type IV:
    freeze pair.

17. If Type IV is only preliminary:
    confirm to 5 seeds.

18. If confirmed Type IV:
    run canonical D0/D1/D2 R2.

19. If no Type IV:
    stop before modifying JAD.

20. Reassess overall paper direction.
```

---

# 133. Primary v5.5 Milestone

\[
\boxed{
\textbf{
Determine whether MELD contains a source pair that simultaneously exhibits stable cross-modal identifiability and task-relevant joint headroom.
}
}
\]

---

# 134. Desired Result

The strongest outcome is:

\[
\boxed{
G_1=PASS
}
\]

and:

\[
\boxed{
G_2=PASS
}
\]

for the same pair.

This creates the first natural:

\[
\boxed{
\textbf{Type-IV setting}.
}
\]

---

# 135. If Type IV Is Found

Do not celebrate or redesign.

Freeze immediately.

Then ask:

\[
\boxed{
\text{Does canonical JAD exploit the setting that the screening framework predicted should support interaction learning?}
}
\]

This is the decisive experiment.

---

# 136. If Type IV Is Not Found

Do not continue dataset hunting automatically.

The project must then decide between:

### Direction A

Diagnostic/framework CVPR paper.

### Direction B

One final theoretically motivated dataset/task.

### Direction C

Reconsider the relationship between modality prediction and task interaction.

No automatic method tuning.

---

# 137. Hard Research Rule

Every experiment from this point must answer a paper question.

Do not run experiments because:

```text
performance might increase
another model is popular
interaction architecture could be stronger
```

---

# 138. Main v5.5 Thesis

\[
\boxed{
\textbf{
A natural multimodal setting justifies explicit interaction learning only when the same modality pair exhibits both non-additive cross-modal predictability and non-additive task headroom.
}
}
\]

---

# 139. Operational Formula

\[
\boxed{
G_1(i,j\rightarrow k)
\land
G_2(i,j\rightarrow Y)
}
\]

is the prerequisite for canonical JAD development.

---

# 140. Final Rule

\[
\boxed{
\textbf{
Screen first.
Require both gates.
Freeze the setting.
Then test JAD.
}
}
\]

Do not reverse this order.

---

# 141. Next Action

\[
\boxed{
\textbf{
Run MELD utterance-level G1 + G2 Two-Gate Screening.
}
}
\]

No further interaction-method development is permitted until the MELD two-gate decision is complete.