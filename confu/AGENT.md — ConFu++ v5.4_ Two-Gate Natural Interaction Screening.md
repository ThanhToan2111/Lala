# AGENT.md

# ConFu++ v5.4

## Two-Gate Natural Interaction Screening for CVPR

---

# 0. Mission

This repository develops **ConFu++**, a framework for explicit multimodal interaction discovery beyond standard higher-order fusion/alignment.

The project targets a **CVPR main-track submission**.

The project has established that multimodal interaction learning cannot be justified solely because:

- multiple modalities exist;
- a joint architecture can be constructed;
- another modality is jointly predictable;
- an interaction embedding fits its target.

A useful interaction representation must pass several distinct conditions.

The current research objective is:

\[
\boxed{
\textbf{find natural multimodal settings where interaction is both cross-modally identifiable and task relevant.}
}
\]

v5.4 introduces a strict:

\[
\boxed{
\textbf{Two-Gate Natural Interaction Screening Protocol}
}
\]

before JAD is allowed to train.

---

# 1. Current Project State

```text
Synthetic Oracle:
PASS

NIC non-identifiable control:
PASS

IPIB identifiable interaction benchmark:
PASS

Canonical D2 / JAD:
FROZEN

Synthetic interaction fidelity:
PASS

Synthetic false-positive control:
PASS

Fidelity–Accessibility tradeoff:
ESTABLISHED

MOSEI exact-source cross-modal R1:
PASS

MOSEI natural mapping:
VA -> T

MOSEI JAD R2:
COMPLETE

MOSEI D2 target fidelity:
PASS

MOSEI D2 downstream gain:
FAIL

MOSEI S0 shortcut localization:
MIXED

MOSEI P1 purification:
FAIL

MOSEI T0 task-headroom:
INCONCLUSIVE

MOSEI sentiment:
FROZEN

Task-Aware JAD:
BLOCKED

Third-order discovery:
BLOCKED

Next dataset:
MUStARD
```

---

# 2. Frozen Canonical Pair Method

The final current pair-discovery method remains:

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

train additive predictor:

\[
q_{\text{add}}(r_i,r_j)
=
q_i(r_i)+q_j(r_j)
\]

and capacity-matched joint predictor:

\[
q_{\text{joint}}(r_i,r_j).
\]

Define:

\[
\boxed{
d_{ij\to k}
=
q_{\text{joint}}
-
q_{\text{add}}.
}
\]

JAD distills \(d\) into an explicit interaction representation:

\[
h_{ij}.
\]

---

# 3. Canonical Interaction Architecture

Use:

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
\sqrt R
}
\right)
}
\]

with canonical:

\[
\boxed{R=64}.
\]

Canonical loss:

\[
\boxed{
L_{JAD}
=
1-\cos(h_{ij},d_{ij\to k}).
}
\]

Do not change this method during v5.4 screening.

---

# 4. Core Conceptual Separation

The project now distinguishes:

\[
\boxed{
\text{Cross-Modal Identifiability}
}
\]

from:

\[
\boxed{
\text{Task Joint Headroom}
}
\]

from:

\[
\boxed{
\text{Interaction Discovery}
}
\]

from:

\[
\boxed{
\text{Representation Fidelity}
}
\]

from:

\[
\boxed{
\text{Downstream Accessibility}.
}
\]

These are not interchangeable.

---

# 5. Key Lesson From MOSEI

Canonical MOSEI established:

\[
J_{\text{cross}}>0
\]

for:

\[
V+A\rightarrow T.
\]

However JAD did not improve sentiment accuracy.

Therefore:

\[
\boxed{
\text{cross-modal interaction identifiability}
\not\Rightarrow
\text{task usefulness}.
}
\]

---

# 6. MOSEI T0 Result

For:

\[
V+A\rightarrow Y_{\text{sentiment}},
\]

capacity-matched task models did not show reliable non-additive headroom.

Validation behavior:

```text
Joint Product < Additive

Generic MLP ≈ / below Additive

seed behavior:
unstable for generic nonlinear model
```

Official decision:

\[
\boxed{
\text{INCONCLUSIVE}.
}
\]

---

# 7. Meaning of MOSEI T0

Do NOT claim:

\[
H_{\text{task}}=0.
\]

Correct wording:

> Stable task-relevant non-additive headroom was not demonstrated under the tested canonical representation and capacity-controlled task models.

---

# 8. MOSEI Status

MOSEI sentiment is now:

\[
\boxed{
\textbf{FROZEN}.
}
\]

Do NOT continue:

```text
Task-Aware JAD
new JAD loss
new purifier
cross-attention
rank sweep
threshold tuning
representation tuning
```

on MOSEI sentiment.

---

# 9. Why Freeze MOSEI

MOSEI has already answered several important questions:

1. Representation/source choice can change interaction identifiability.
2. Cross-modal non-additive predictive structure can exist naturally.
3. JAD can successfully fit that structure.
4. Lower-order leakage can coexist with joint structure.
5. Naive purification can destroy joint structure.
6. Cross-modal interaction does not guarantee downstream task gain.
7. Task joint headroom must be measured independently.

This is sufficient.

---

# 10. Updated Screen-First Principle

Old rule:

\[
\text{screen}
\rightarrow
\text{train interaction}.
\]

New rule:

\[
\boxed{
\text{Cross-Modal Gate}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Task-Headroom Gate}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Interaction Training}.
}
\]

---

# 11. Gate G1 — Cross-Modal Identifiability

For source pair:

\[
(i,j)
\]

and modality target:

\[
k,
\]

compute:

\[
\boxed{
J_{\text{cross}}
=
R^2(q_{\text{joint}},r_k)
-
R^2(q_{\text{add}},r_k).
}
\]

This measures whether a joint predictor contains predictive structure unavailable to the additive predictor.

---

# 12. G1 Operational Gate

Preferred frozen criterion:

\[
\boxed{
J_{\text{cross,val}}>0.01
}
\]

with consistent sign across all screening seeds.

Initial screening:

\[
\boxed{3\text{ seeds}}.
\]

---

# 13. G1 Does Not Mean Synergy

Do NOT describe:

\[
J_{\text{cross}}
\]

as:

```text
PID synergy
causal interaction
new information
true interaction information
```

Preferred terminology:

```text
joint predictive advantage
non-additive predictive structure
cross-modal predictive interaction
```

---

# 14. Gate G2 — Task Joint Headroom

For downstream target:

\[
Y,
\]

train capacity-controlled:

\[
c_{\text{add}}
\]

and:

\[
c_{\text{joint}}.
\]

Define:

\[
\boxed{
H_{\text{task}}
=
Perf(c_{\text{joint}})
-
Perf(c_{\text{add}}).
}
\]

---

# 15. G2 Interpretation

G2 asks:

> Does the downstream task itself benefit from non-additive access to the source modalities?

This is fundamentally different from G1.

---

# 16. Generic Nonlinear Control

Also train:

\[
c_{\text{MLP}}([r_i,r_j]).
\]

Purpose:

> determine whether nonlinear task headroom exists but is missed by the explicit multiplicative hypothesis class.

---

# 17. Three Possible Task Conditions

### Condition A

\[
c_{\text{joint}}
\approx
c_{\text{add}}
\]

and:

\[
c_{\text{MLP}}
\approx
c_{\text{add}}.
\]

Interpretation:

\[
\boxed{
\text{no demonstrated task headroom}.
}
\]

---

# 18. Condition B

\[
c_{\text{joint}}>c_{\text{add}}.
\]

Interpretation:

\[
\boxed{
\text{explicit joint task headroom}.
}
\]

This is the preferred positive setting for JAD.

---

# 19. Condition C

\[
c_{\text{MLP}}>c_{\text{add}}
\]

but:

\[
c_{\text{joint}}\approx c_{\text{add}}.
\]

Interpretation:

\[
\boxed{
\text{generic nonlinear headroom but explicit-product hypothesis limitation}.
}
\]

Do not train canonical JAD automatically.

---

# 20. Required Positive Development Setting

The preferred condition for a natural positive ConFu++ experiment is:

\[
\boxed{
G_1=PASS
}
\]

and:

\[
\boxed{
G_2=PASS.
}
\]

Operationally:

\[
\boxed{
J_{\text{cross}}>0
\quad\land\quad
H_{\text{task}}>0.
}
\]

---

# 21. Why Both Gates Matter

If:

\[
G_1=PASS
\]

but:

\[
G_2=FAIL,
\]

then interaction may exist for cross-modal prediction but be irrelevant to the downstream task.

MOSEI approximately demonstrates this regime.

---

# 22. Opposite Case

If:

\[
G_1=FAIL
\]

but:

\[
G_2=PASS,
\]

the downstream task contains nonlinear multimodal structure but it is not captured through the tested cross-modal prediction target.

Canonical JAD is not well motivated.

---

# 23. Best JAD Setting

Strongest setting:

\[
\boxed{
G_1=PASS
\land
G_2=PASS.
}
\]

Then:

- cross-modal interaction exists;
- downstream interaction headroom exists;
- JAD has a plausible task-relevant interaction to discover.

---

# 24. v5.4 Dataset

Primary:

\[
\boxed{
\textbf{MUStARD}
}
\]

Role:

\[
\boxed{
\text{first Two-Gate natural screening benchmark}.
}
\]

---

# 25. MUStARD Scientific Hypothesis

Sarcasm may depend on incongruity among:

```text
language
prosody/audio
facial/visual cues
```

Therefore it is a plausible setting for:

\[
H_{\text{task}}>0.
\]

This is a hypothesis only.

Do not assume it passes.

---

# 26. MUStARD Task

Downstream target:

\[
\boxed{
Y_{\text{sarcasm}}.
}
\]

Do not introduce additional auxiliary tasks initially.

---

# 27. MUStARD Modalities

Use:

```text
V = vision
A = audio
T = text
```

with frozen modality representations.

Document exact feature dimensions.

---

# 28. MUStARD Representation Audit

Before screening record:

```text
dataset size
train count
validation count
test count
class balance
vision dimension
audio dimension
text dimension
sequence handling
missing modalities
zero vectors
NaNs/Infs
```

---

# 29. Dataset Identity

If multiple MUStARD processed versions exist:

choose one canonical source before experiments.

Do not combine labels/features from incompatible versions.

---

# 30. Split Rule

Prefer the established repository/dataset split if available.

If a split must be constructed:

- create once;
- seed it deterministically;
- freeze it;
- save sample IDs;
- never resplit based on results.

---

# 31. Small-Data Warning

MUStARD is expected to be substantially smaller than MOSEI.

Therefore:

\[
\boxed{
\text{variance control is especially important}.
}
\]

Avoid large networks.

---

# 32. G1 Directions

Screen all three:

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

# 33. G1 Predictor Families

Reuse the established:

\[
q_{\text{add}}
\]

and:

\[
q_{\text{joint}}
\]

families.

Do not invent a MUStARD-specific predictor.

---

# 34. Capacity Matching

Require:

\[
\frac{
|P_J-P_A|
}{
P_A
}
<0.05.
\]

Preferred:

\[
<0.01.
\]

Report exact counts.

---

# 35. G1 Seeds

Use:

\[
\boxed{3\text{ screening seeds}}.
\]

If a setting later becomes a final positive experiment:

\[
\boxed{5\text{ seeds}}.
\]

---

# 36. G1 Metrics

For every:

```text
mapping × seed
```

report:

\[
R^2_{\text{add,train}}
\]

\[
R^2_{\text{add,val}}
\]

\[
R^2_{\text{add,test}}
\]

\[
R^2_{\text{joint,train}}
\]

\[
R^2_{\text{joint,val}}
\]

\[
R^2_{\text{joint,test}}
\]

and:

\[
J_{\text{train}},
J_{\text{val}},
J_{\text{test}}.
\]

---

# 37. G1 Overfit Audit

If:

\[
J_{\text{train}}\gg J_{\text{val}},
\]

do not count that as interaction evidence.

Possible interpretation:

\[
\boxed{
\text{joint model overfit}.
}
\]

---

# 38. G1 Selection

Use:

\[
\boxed{
\text{validation only}.
}
\]

Test values are diagnostic.

Do not use positive test \(J\) to rescue failed validation.

---

# 39. G1 Pass

A preferred passing mapping has:

\[
J_{\text{val}}>0.01
\]

for:

\[
3/3
\]

seeds.

Also inspect whether:

\[
J_{\text{test}}>0
\]

after selection.

---

# 40. G1 Weak Candidate

If:

\[
2/3
\]

seeds pass:

do NOT promote.

Classify as:

```text
WEAK / UNSTABLE
```

and continue to other mappings.

---

# 41. G2 Input

G2 uses the source modalities relevant to each candidate.

For pair \(i,j\):

\[
[r_i,r_j]\rightarrow Y_{\text{sarcasm}}.
\]

---

# 42. G2 Models

Train:

```text
Linear
Additive
Joint Product
Generic MLP
```

as in MOSEI T0.

---

# 43. G2 Linear Baseline

\[
c_L([r_i,r_j]).
\]

Purpose:

simple lower-order benchmark.

---

# 44. G2 Additive Predictor

\[
\boxed{
c_{\text{add}}
=
c_i(r_i)+c_j(r_j).
}
\]

The branches must remain independent until logit addition.

---

# 45. G2 Explicit Joint Predictor

Preferred:

\[
u_i=P_ir_i
\]

\[
u_j=P_jr_j
\]

\[
\boxed{
c_{\text{joint}}
=
W[
u_i;
u_j;
u_i\odot u_j
].
}
\]

---

# 46. G2 Generic MLP

\[
\boxed{
c_{\text{MLP}}
=
MLP([r_i,r_j]).
}
\]

Capacity should remain near the additive/joint baselines.

---

# 47. G2 Metrics

Report:

```text
Accuracy
Macro F1
AUC
BCE
```

For class imbalance, Macro F1 and AUC are especially useful.

---

# 48. G2 Validation Gate

Do not define success from one metric only.

A preferred positive candidate should show:

```text
positive mean difference
consistent sign across seeds
support from multiple metrics
```

---

# 49. G2 Seed Rule

Initial:

\[
3\text{ seeds}.
\]

If promising but noisy because of small data:

allow:

\[
5\text{ confirmatory seeds}
\]

only after the 3-seed configuration is frozen.

No architecture changes between them.

---

# 50. G2 Strong Pass

Strong evidence:

\[
Joint>Additive
\]

in at least:

\[
4/5
\]

confirmatory seeds,

with positive average:

```text
Accuracy or Macro F1
AUC
BCE improvement
```

Do not require all metrics to be statistically significant.

---

# 51. G2 Generic-Only Case

If:

\[
MLP>Additive
\]

but:

\[
JointProduct\le Additive,
\]

mark:

\[
\boxed{
\text{GENERIC NONLINEAR HEADROOM ONLY}.
}
\]

Canonical JAD should not train on this setting yet.

---

# 52. MUStARD Two-Gate Matrix

Required table:

| Pair / Mapping | \(J_{cross}\) | G1 | \(H_{task}\) | G2 | Decision |
|---|---:|---|---:|---|---|
| VA→T / VA→Y | | | | | |
| VT→A / VT→Y | | | | | |
| AT→V / AT→Y | | | | | |

---

# 53. Important Pair Consistency

For JAD experiment:

cross-modal target and task source pair must correspond.

Example:

If:

\[
VA\rightarrow T
\]

passes G1,

then evaluate:

\[
VA\rightarrow Y.
\]

Do not combine:

\[
VA\rightarrow T
\]

G1 with:

\[
VT\rightarrow Y
\]

G2.

---

# 54. JAD Trigger

Only if the same source pair passes both:

\[
\boxed{G_1}
\]

and:

\[
\boxed{G_2}
\]

may full JAD training begin.

---

# 55. If Multiple Pairs Pass

Do not run all pairs immediately.

Select the strongest validation-supported candidate according to:

```text
stable G1
stable G2
low overfit
reasonable sample support
```

Freeze one primary development mapping.

---

# 56. No Test-Based Pair Ranking

Do not choose the primary pair because test accuracy is highest.

Use validation evidence only.

---

# 57. v5.4-S1

First experiment:

\[
\boxed{
\textbf{S1 — MUStARD Cross-Modal Identifiability}
}
\]

This is G1.

No interaction embedding training.

---

# 58. v5.4-S2

Second experiment:

\[
\boxed{
\textbf{S2 — MUStARD Task Joint-Headroom}
}
\]

This is G2.

No JAD embedding training.

---

# 59. S1 and S2 Ordering

Recommended:

```text
1. run S1 all mappings
2. run S2 same source pairs
3. construct two-gate matrix
4. decide whether JAD is allowed
```

---

# 60. Why Run G2 Even If G1 Fails?

For scientific analysis it can be useful to know whether a task is jointly nonlinear even when cross-modal prediction is not identifiable.

But do not run full JAD if G1 fails.

---

# 61. Four Natural Regimes

Every natural setting can be classified:

### Type I

\[
G_1=0,\quad G_2=0.
\]

No demonstrated interaction motivation.

### Type II

\[
G_1=1,\quad G_2=0.
\]

Cross-modal interaction exists but not demonstrated task relevance.

MOSEI approximates this regime.

### Type III

\[
G_1=0,\quad G_2=1.
\]

Task interaction exists but canonical cross-modal JAD target is unsuitable.

### Type IV

\[
\boxed{
G_1=1,\quad G_2=1.
}
\]

Ideal JAD development setting.

---

# 62. Paper-Level Importance

This four-regime taxonomy may become a central figure.

It directly answers:

> When should higher-order multimodal interaction training be activated?

---

# 63. Type II Interpretation

If:

\[
G_1=PASS
\]

but:

\[
G_2=FAIL,
\]

do not call interaction useless.

Correct:

> It is useful for predicting another modality but not demonstrated as useful for the target task.

---

# 64. Type III Interpretation

If task headroom exists but cross-modal interaction does not:

canonical JAD is mismatched to the task.

This may later motivate task-aware methods.

Not during v5.4.

---

# 65. Type IV Interpretation

If both pass:

canonical JAD has a scientifically justified development target.

Then proceed.

---

# 66. R2 After Two-Gate Pass

If a MUStARD setting reaches Type IV:

run:

\[
\boxed{
\text{R2-MUStARD JAD Benchmark}.
}
\]

---

# 67. R2 Methods

Compare:

```text
Lower-order
D0
D1
D2 / JAD
```

using identical:

```text
representations
rank
optimizer
training budget
probes
seeds
```

---

# 68. D0

\[
t_{D0}=r_k.
\]

---

# 69. D1

\[
\boxed{
t_{D1}
=
r_k-q_{\text{add}}.
}
\]

---

# 70. D2

\[
\boxed{
t_{D2}
=
q_{\text{joint}}-q_{\text{add}}.
}
\]

---

# 71. Downstream Accessibility

Lower-order:

\[
Z_{\le1}=[r_i,r_j].
\]

Interaction representation:

\[
Z_{\le2}
=
[r_i,r_j,h_{ij}].
\]

Compute:

\[
\boxed{
\Delta
=
Perf(Z_{\le2})
-
Perf(Z_{\le1}).
}
\]

---

# 72. R2 Primary Metrics

For sarcasm:

```text
Accuracy
Macro F1
AUC
BCE
```

Use the dataset's established primary metric prominently if applicable.

---

# 73. R2 Interaction Diagnostics

Report:

```text
target cosine
effective rank
single-source reconstructability
additive reconstructability
joint reconstructability
source shuffles
```

---

# 74. Do Not Require Perfect Purity

P1 demonstrated that aggressively removing all additive reconstructability can destroy useful joint structure.

Therefore single-modality leakage is a diagnostic, not an automatic failure.

---

# 75. Main JAD Success Condition

Most important:

\[
\boxed{
D2>D0
}
\]

on a setting where:

\[
G_1=PASS
\]

and:

\[
G_2=PASS.
\]

This would support the core method claim.

---

# 76. Stronger Success

Ideal:

\[
D2>D0
\]

and:

\[
D2\ge D1
\]

with healthy interaction diagnostics.

Not required.

---

# 77. Important Negative Outcome

If Type IV setting exists but:

\[
D2\le D0
\]

and no accessibility gain appears:

then the JAD hypothesis is substantially weakened.

Do not hide this.

---

# 78. CVPR Scientific Standard

A strong paper needs not only positive results but correct conditions of applicability.

The two-gate framework should predict when JAD should and should not help.

---

# 79. Next Dataset if MUStARD Is Inconclusive

If MUStARD is too unstable:

\[
\boxed{
\text{MELD}
}
\]

becomes the preferred next dataset.

Reason:

larger natural multimodal task setting.

---

# 80. MELD Protocol

Use the same:

\[
G_1+G_2
\]

protocol.

Do not redesign screening.

---

# 81. MELD Candidate Tasks

Prefer one clearly defined target first:

```text
emotion recognition
```

Do not simultaneously tune on many label variants.

---

# 82. UR-FUNNY Role

After a positive Type-IV setting is found:

use:

\[
\boxed{
\text{UR-FUNNY}
}
\]

as a transfer / difficult text-dominant benchmark.

---

# 83. No UR-FUNNY Accuracy Tuning

First measure:

\[
G_1
\]

and:

\[
G_2.
\]

Only train JAD if justified.

---

# 84. Dataset Search Stop Rule

Do not keep testing datasets until one gives favorable results.

Current permitted order:

```text
1. MUStARD
2. MELD if MUStARD inconclusive
3. UR-FUNNY as transfer/control
```

Additional datasets require a clear scientific reason.

---

# 85. Main Paper Framework

Current candidate pipeline:

```text
Multimodal data
      |
      v
Cross-modal identifiability gate
      |
      | J_cross > threshold?
      v
Task-headroom gate
      |
      | H_task > 0?
      v
Interaction discovery
      |
      v
Representation fidelity
      |
      v
Downstream accessibility
```

---

# 86. Paper Figure Candidate

A 2×2 matrix:

| | Task headroom − | Task headroom + |
|---|---|---|
| Cross-modal J − | Type I | Type III |
| Cross-modal J + | Type II | **Type IV** |

Use natural datasets/settings as points after experiments complete.

---

# 87. MOSEI Placement

Current MOSEI should be conservatively positioned near:

\[
\boxed{
\text{Type II candidate}
}
\]

because:

\[
G_1=PASS
\]

while G2 produced no stable positive evidence.

Because T0 is officially INCONCLUSIVE, avoid calling G2 definitively negative.

---

# 88. Synthetic IPIB Placement

IPIB positive regimes serve as controlled settings where interaction is known by construction.

They are not directly equivalent to real-world G2.

---

# 89. Main Theoretical Distinction

Let target modality satisfy conceptually:

\[
T
=
A_T(X_i,X_j)
+
J_T(X_i,X_j)
+
\epsilon_T.
\]

JAD seeks:

\[
J_T.
\]

---

# 90. Downstream Task

Let downstream target:

\[
Y
=
A_Y(X_i,X_j)
+
J_Y(X_i,X_j)
+
\epsilon_Y.
\]

There is no guarantee:

\[
\boxed{
J_T=J_Y.
}
\]

---

# 91. Task-Relevant Overlap

JAD helps the downstream task only if there is useful overlap between:

\[
J_T
\]

and:

\[
J_Y.
\]

Do not claim this overlap is directly identified yet.

---

# 92. New Core Research Question

\[
\boxed{
\textbf{
When does cross-modal predictable interaction overlap with task-relevant multimodal interaction?
}
}
\]

This may become the central CVPR question.

---

# 93. Contribution Candidate 1

A diagnostic hierarchy distinguishing:

\[
\boxed{
\text{cross-modal identifiability}
}
\]

from:

\[
\boxed{
\text{task joint headroom}.
}
\]

---

# 94. Contribution Candidate 2

IPIB:

\[
\boxed{
\text{Identifiable Predictive-Interaction Benchmark}.
}
\]

---

# 95. Contribution Candidate 3

JAD:

\[
\boxed{
q_{\text{joint}}-q_{\text{add}}.
}
\]

---

# 96. Contribution Candidate 4

Show empirically:

\[
\boxed{
\text{cross-modal interaction fidelity}
\not\Rightarrow
\text{task accessibility}.
}
\]

---

# 97. Contribution Candidate 5

Demonstrate on natural datasets that:

\[
\boxed{
\text{cross-modal identifiability and task headroom are distinct}.
}
\]

This requires v5.4 real-world screening.

---

# 98. Potential Strongest Contribution

If a Type-IV dataset is found and JAD helps:

> Interaction screening predicts when explicit higher-order interaction learning is useful.

This would substantially strengthen the CVPR story.

---

# 99. Current Forbidden Work

During MUStARD S1/S2:

```text
NO JAD modification
NO task-aware JAD
NO purifier
NO rank sweep
NO loss sweep
NO cross-attention
NO third-order discovery
NO test-driven pair selection
NO dataset-specific architecture tuning
```

---

# 100. Current Permitted Work

```text
MUStARD data audit
MUStARD G1
MUStARD G2
paper theory
paper figures
related work
```

---

# 101. Suggested Code Structure

Create:

```text
src/experiments/multibench/two_gate/
```

with:

```text
common.py
cross_modal_gate.py
task_headroom_gate.py
mustard_screening.py
```

Reuse established predictor classes where possible.

---

# 102. Avoid Protocol Forks

Do not maintain separate definitions of:

```text
AdditivePredictor
JointPredictor
TaskAdditive
TaskJoint
```

for every dataset.

Create reusable implementations.

---

# 103. Result Directory

Recommended:

```text
results/mustard/v54/
    data_audit/
    g1_cross_modal/
    g2_task_headroom/
    two_gate_summary/
```

---

# 104. Required G1 Artifact

```text
MUSTARD_G1_CROSS_MODAL_REPORT.md
```

---

# 105. Required G2 Artifact

```text
MUSTARD_G2_TASK_HEADROOM_REPORT.md
```

---

# 106. Required Two-Gate Artifact

```text
MUSTARD_TWO_GATE_DECISION.md
```

---

# 107. Machine Summary

Save:

```text
mustard_two_gate.json
```

with:

```text
dataset
representation
mapping
seed
J_train
J_val
J_test
task_model
task_val_metrics
task_test_metrics
G1_status
G2_status
final_type
```

---

# 108. Unit Tests

Required:

```text
mapping direction correctness
source/target modality correctness
capacity matching
additive branch isolation
joint product correctness
generic MLP input correctness
same samples across G1/G2
deterministic seed
validation-only selection
test not used for gate
```

---

# 109. Small Dataset Stability

For MUStARD explicitly report:

```text
per-seed values
sample std
confidence interval if appropriate
class balance
```

Avoid conclusions from means alone.

---

# 110. No Arbitrary G2 Threshold

Unlike:

\[
J_{\text{cross}}>0.01,
\]

do not introduce an arbitrary:

```text
accuracy > 1 pp
```

gate unless preregistered before seeing MUStARD results.

Use consistency + multiple metrics.

---

# 111. G2 Primary Validation Metrics

Preferred priority:

```text
Macro F1
AUC
BCE
Accuracy
```

especially if class balance is imperfect.

---

# 112. G1 Primary Metric

Always:

\[
\boxed{
R^2
}
\]

on target-modality prediction.

---

# 113. Task Model Training Objective

For binary task:

\[
\boxed{
BCEWithLogits
}
\]

or existing equivalent.

Use the same objective across additive/joint/MLP.

---

# 114. Early Stopping

Use validation BCE.

Do not early stop one model on F1 and another on BCE.

---

# 115. Model Selection

One fixed architecture per model family.

No broad search.

---

# 116. Seed-1 Smoke Test

Before full screening verify:

```text
loss decreases
no NaNs
correct shapes
parameter counts
checkpoint saves
determinism
```

Smoke test is not evidence.

---

# 117. Screening Run

Then run:

\[
3\text{ seeds}
\]

for:

```text
G1 all three mappings
G2 corresponding source pairs
```

---

# 118. v5.4 Decision

At the end classify each source pair as:

```text
TYPE_I
TYPE_II
TYPE_III
TYPE_IV
INCONCLUSIVE
```

---

# 119. Type IV Trigger

Only:

\[
\boxed{
\text{TYPE IV}
}
\]

automatically permits JAD R2.

---

# 120. Type II

Cross-modal interaction only.

Keep as natural evidence for the diagnostic framework.

No JAD task claim.

---

# 121. Type III

Task interaction exists but canonical JAD cross-modal target is not supported.

Potential future task-aware research.

Do not open it yet.

---

# 122. Type I

No demonstrated reason for explicit interaction training under current representations.

Stop.

---

# 123. Inconclusive

Do not rescue with:

```text
more hidden dimensions
different loss
test-based selection
different random split
```

Decide whether five fixed confirmatory seeds are justified by genuine near-threshold evidence.

---

# 124. Confirmation Rule

Five-seed confirmation is allowed only if:

- architecture is frozen;
- split is frozen;
- preliminary result is scientifically ambiguous rather than simply negative;
- no hyperparameter changes are introduced.

---

# 125. After MUStARD

If Type IV:

\[
\boxed{
\text{run canonical JAD}.
}
\]

If no Type IV and high variance:

\[
\boxed{
\text{move to MELD}.
}
\]

If clear Type I/II/III:

preserve the result and move according to paper needs.

---

# 126. No Synthetic Reopening

Do not reopen IPIB during screening.

Synthetic phase is already sufficient.

Only reopen if a genuinely new mechanism is discovered from a Type-IV real-world failure.

---

# 127. Third-Order Gate

Order-3 discovery remains blocked until:

\[
\boxed{
\text{at least one real-world Type-IV pair setting}
}
\]

has been studied with canonical JAD.

---

# 128. Why Third-Order Is Blocked

Without task-relevant pair-level interaction evidence, order-3 expansion would add complexity without resolving the central scientific question.

---

# 129. CVPR Submission Goal

The ideal final experimental story contains:

```text
IPIB synthetic positive benchmark

MOSEI:
cross-modal interaction without demonstrated task headroom

MUStARD/MELD:
task-relevant positive setting

UR-FUNNY:
transfer/control setting
```

This provides contrasting regimes rather than only cherry-picked positive datasets.

---

# 130. Reviewer Defense

Likely question:

> Why not simply train fusion everywhere?

Answer should be supported experimentally:

> Because measurable non-additive structure varies by representation, modality mapping, and task; higher-order training is only justified when the corresponding interaction is identifiable and task relevant.

---

# 131. Another Reviewer Question

> Why does JAD predict another modality instead of the label?

Answer:

> Canonical JAD is label-independent interaction discovery. The task-headroom gate independently determines whether the discovered class of interaction is likely to matter downstream.

---

# 132. Another Reviewer Question

> Why not directly use a task-aware interaction objective?

Because it answers a different question.

Canonical JAD studies:

\[
\boxed{
\text{cross-modal predictive interaction}.
}
\]

Task-aware learning studies:

\[
\boxed{
\text{supervised task interaction}.
}
\]

Do not conflate them.

---

# 133. Current Main Thesis

\[
\boxed{
\textbf{
Multimodal interaction should not be learned merely because multiple modalities are available; useful higher-order learning requires both identifiable cross-modal structure and task-relevant non-additive headroom.
}
}
\]

---

# 134. Current Operational Rule

\[
\boxed{
\textbf{
G1: Cross-modal identifiability
\quad+\quad
G2: Task joint headroom
\quad\Rightarrow\quad
interaction training.
}
}
\]

---

# 135. Current Required Experiment

\[
\boxed{
\textbf{
MUStARD Two-Gate Screening
}
}
\]

with:

### G1

\[
VA\rightarrow T
\]

\[
VT\rightarrow A
\]

\[
AT\rightarrow V
\]

and:

### G2

\[
VA\rightarrow Y_{\text{sarcasm}}
\]

\[
VT\rightarrow Y_{\text{sarcasm}}
\]

\[
AT\rightarrow Y_{\text{sarcasm}}.
\]

---

# 136. Immediate Execution Order

```text
1. Audit MUStARD files and sample IDs.

2. Freeze canonical representation and split.

3. Verify class balance and missing data.

4. Implement/reuse G1 predictors.

5. Verify capacity matching.

6. Run seed-1 G1 smoke test.

7. Run G1 seeds 1..3 for all three mappings.

8. Implement/reuse G2 task models.

9. Verify task-model parameter matching.

10. Run seed-1 G2 smoke test.

11. Run G2 seeds 1..3 for all source pairs.

12. Build Two-Gate matrix.

13. Classify each pair:
    Type I / II / III / IV / Inconclusive.

14. Do not train JAD before classification.

15. If Type IV:
    freeze that pair and launch five-seed JAD R2.

16. If no Type IV and MUStARD is unstable:
    move to MELD.
```

---

# 137. Hard Stop

No new ConFu++ method may be proposed during v5.4 screening.

The goal is not:

\[
\text{make JAD win}.
\]

The goal is:

\[
\boxed{
\textbf{identify the conditions under which JAD is scientifically justified.}
}
\]

---

# 138. v5.4 Success Condition

v5.4 succeeds if it produces a reliable classification of natural multimodal settings into:

\[
\boxed{
\text{cross-modal interaction present/absent}
}
\]

and:

\[
\boxed{
\text{task interaction headroom present/absent}.
}
\]

Finding a Type-IV positive setting is highly desirable but is not required for the screening experiment itself to be scientifically valid.

---

# 139. Strongest Desired Outcome

The strongest next result would be:

\[
\boxed{
J_{\text{cross}}>0
}
\]

and:

\[
\boxed{
H_{\text{task}}>0
}
\]

for the same source pair,

followed by:

\[
\boxed{
D2>D0
}
\]

after canonical JAD training.

That would directly connect:

\[
\text{screening}
\rightarrow
\text{interaction discovery}
\rightarrow
\text{downstream utility}.
\]

---

# 140. Final v5.4 Thesis

\[
\boxed{
\textbf{
Cross-modal predictability and downstream task interaction are distinct properties; explicit multimodal interaction learning should be activated only after both are independently established.
}
}
\]

---

# 141. Next Action

\[
\boxed{
\textbf{
Run MUStARD G1 + G2 Two-Gate Screening before any further JAD training.
}
}
\]