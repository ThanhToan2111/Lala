# AGENT.md

# ConFu++ v4.1

## Discovery of Explicit Higher-Order Multimodal Representations

---

# 0. Mission

This repository develops **ConFu++**, a research extension of:

**ConFu — Contrastive Fusion for Higher-Order Multimodal Alignment**

The current research question is no longer whether the architecture can represent pairwise or third-order interaction.

The Oracle synthetic experiment has already established:

\[
\boxed{
\text{Architecture Capability = PASS}
}
\]

The primary open question is now:

\[
\boxed{
\text{Can higher-order interactions be discovered without ground-truth interaction supervision?}
}
\]

Therefore the project moves from:

```text
Q1 — Architecture Capability
```

to:

```text
Q2 — Discovery Objective Capability
```

Real-world improvement remains blocked until Q2 is validated.

---

# 1. Current Project State

Current status:

```text
Q1 — Architecture capability
PASS

Q2 — Discovery objective capability
NEXT

Q3 — Real-world improvement
BLOCKED
```

Do NOT return to heavy UR-FUNNY optimization before Q2 passes.

---

# 2. Scientific Thesis

A deterministic interaction representation:

\[
h_{12}=F(r_1,r_2)
\]

does not create new Shannon information.

Its purpose is instead to:

\[
\boxed{
\text{reparameterize implicit nonlinear joint structure}
}
\]

into:

\[
\boxed{
\text{explicit and downstream-accessible interaction features}.
}
\]

The central goal is therefore:

\[
\boxed{
\text{order-specific accessibility}
}
\]

rather than information creation.

---

# 3. Historical Findings

The current project has established several negative controls.

## AV-MNIST

Observed:

\[
h_{12}\approx f(r_{image}).
\]

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

# 4. MOSI

Observed:

- multimodal dependence;
- strong text dominance;
- low or negative conditional utility.

Interpretation:

\[
\boxed{
\text{joint dependence does not guarantee accessible interaction utility}.
}
\]

Role:

```text
dependence-without-accessibility control
```

---

# 5. UR-FUNNY

Original ConFu:

\[
64.462\pm0.713\%.
\]

Multiple ConFu++ variants were tested:

```text
S1 adversarial de-shortcutting
v2 residual correction
v3 residual alignment
v3.1 nonlinear-additive residual alignment
```

None produced stable positive linear accessibility.

Conclusion:

\[
\boxed{
\text{real-world failure cannot currently be attributed to architecture capacity alone}.
}
\]

---

# 6. Synthetic Oracle Result

Controlled synthetic benchmark now contains:

```text
S0 — first-order only
S1 — single pair interaction
S2 — all pair interactions
S3 — pure third-order
S4 — mixed order
```

Oracle interaction supervision directly trained:

\[
h_{12}\rightarrow z_1\odot z_2
\]

\[
h_{13}\rightarrow z_1\odot z_3
\]

\[
h_{23}\rightarrow z_2\odot z_3
\]

and:

\[
h_{123}
\rightarrow
z_1\odot z_2\odot z_3.
\]

---

# 7. Oracle Pair Capability

Observed approximately:

\[
S1:
\Delta_2\approx+29\text{ pp}
\]

and:

\[
S2:
\Delta_2\approx+29.5\text{ pp}.
\]

Correct pair selectivity was recovered.

Therefore:

\[
\boxed{
\text{pair interaction architecture is expressive enough}.
}
\]

---

# 8. Oracle Third-Order Capability

On pure triple S3:

\[
Perf(Z_{\le1})\approx50\%
\]

\[
Perf(Z_{\le2})\approx50\%
\]

\[
Perf(Z_{\le3})\approx93.5\%.
\]

Thus:

\[
\boxed{
\Delta_3\approx+43.4\text{ pp}.
}
\]

This demonstrates:

\[
\boxed{
\text{the explicit }h_{123}\text{ architecture can linearize true order-3 structure}.
}
\]

---

# 9. Mixed-Order Oracle

S4 showed:

\[
Perf(Z_{\le1})
<
Perf(Z_{\le2})
<
Perf(Z_{\le3}).
\]

This pattern held across all five Oracle seeds.

Therefore:

\[
\boxed{
\text{functional order hierarchy is representable by the architecture}.
}
\]

---

# 10. Q1 Decision

Oracle architecture:

\[
\boxed{
PASS
}
\]

Do NOT modify:

```text
low-rank pair interaction
low-rank triple interaction
basic order hierarchy
```

unless future evidence specifically identifies an architecture limitation.

---

# 11. Immediate Research Bottleneck

The current bottleneck is:

\[
\boxed{
\text{interaction discovery}
}
\]

not:

```text
interaction rank
architecture capacity
interaction variance
representation collapse
```

The central problem is:

> How can the model discover the interaction target without seeing the latent ground truth?

---

# 12. Immediate Experimental Pipeline

Follow exactly:

```text
Phase A
Generic MLP sanity

Phase B
Oracle diagnostics completion

Phase C
D0 — Original ConFu discovery

Phase D
D1 — Residual discovery

Phase E
D2 — Joint-Advantage Distillation

Phase F
Five-seed discovery confirmation

Phase G
Third-order discovery

Phase H
Real-world transfer
```

Do not skip stages.

---

# 13. Phase A — Generic MLP Sanity

The current generic MLP baseline underperforms strongly on pure triple S3.

This result is not yet sufficient for publication-level comparison.

The immediate goal is to distinguish:

\[
\text{optimization failure}
\]

from:

\[
\text{representational difficulty}.
\]

---

# 14. MLP Inputs

Evaluate MLP on:

### Raw observation

\[
[x_1,x_2,x_3]
\]

### Latent input

\[
[z_1,z_2,z_3]
\]

### Learned first-order representation

\[
[r_1,r_2,r_3].
\]

---

# 15. MLP Capacity Control

Use fixed parameter budgets.

Recommended comparison:

```text
MLP-small
MLP-medium
MLP-large
```

but limit architecture search.

Do NOT perform unrestricted hyperparameter tuning.

---

# 16. MLP Diagnostics

Always log:

```text
train loss
validation loss
test loss

train accuracy
validation accuracy
test accuracy

best epoch
parameter count
```

If:

\[
Train\approx100\%
\]

but:

\[
Test\approx50\%,
\]

the issue is generalization.

If:

\[
Train\approx50\%,
\]

the issue is optimization/capacity.

---

# 17. MLP Purpose

Do NOT use generic MLP as evidence against explicit interaction until it has been properly optimized.

The key question is:

> How difficult is the same higher-order function for an implicit generic nonlinear model versus an explicit multiplicative representation?

---

# 18. Phase B — Complete Oracle Diagnostics

Before discovery experiments, add:

```text
variance
effective rank
dimension-wise standard deviation
mean L2 norm
target cosine
```

for all:

\[
h_{12},
h_{13},
h_{23},
h_{123}.
\]

---

# 19. Pair Shuffle Diagnostics

For:

\[
h_{12},
\]

evaluate:

\[
h_{12}(\pi(z_1),z_2)
\]

and:

\[
h_{12}(z_1,\pi(z_2)).
\]

Repeat for:

\[
h_{13},
h_{23}.
\]

Both source modalities should matter.

---

# 20. Triple Shuffle Diagnostics

For:

\[
h_{123},
\]

independently shuffle:

\[
z_1,
z_2,
z_3.
\]

Require all three modalities to influence the representation and downstream performance.

---

# 21. All Branches Must Be Enabled

From Discovery experiments onward:

\[
h_{12},
h_{13},
h_{23}
\]

must exist in every regime.

Do NOT hard-zero inactive branches based on synthetic ground truth.

The objective must learn whether a branch should be useful.

---

# 22. Why All Branches Must Be Enabled

S0 must answer:

> Does the discovery objective invent false higher-order utility when no interaction exists?

S1 must answer:

> Does the objective discover the correct pair instead of all possible pairs?

This cannot be tested if inactive branches are manually removed.

---

# 23. Discovery Benchmark Definition

Discovery methods must be evaluated on:

```text
S0
S1
S2
S3
```

before S4.

S4 is a mixed-complexity confirmation regime.

---

# 24. D0 — Original ConFu Alignment

D0 is the primary discovery baseline.

For pair:

\[
h_{12}=F_{12}(r_1,r_2).
\]

Original ConFu-style alignment:

\[
h_{12}\leftrightarrow r_3
\]

\[
h_{13}\leftrightarrow r_2
\]

\[
h_{23}\leftrightarrow r_1.
\]

Use the same interaction architecture as Oracle.

Only the objective changes.

---

# 25. D0 Research Question

D0 answers:

> Can standard higher-order contrastive alignment recover known pair interaction structure without Oracle supervision?

This directly tests the central assumption behind ConFu.

---

# 26. D0 Expected Outcomes

On S1:

\[
\Delta_{12}
\]

should ideally be positive.

Wrong pair gains:

\[
\Delta_{13},
\Delta_{23}
\]

should remain near zero.

On S2:

\[
\Delta_2>0
\]

is desired.

On S0:

\[
\Delta_2\approx0.
\]

---

# 27. D0 Failure Interpretation

If:

\[
\Delta_2^{Oracle}\gg0
\]

but:

\[
\Delta_2^{D0}\approx0,
\]

then:

\[
\boxed{
\text{ConFu alignment fails to recover an interaction that its architecture can represent}.
}
\]

This is a strong research motivation.

---

# 28. D1 — Residual Alignment

D1 reproduces the residual-target approach.

Train independent predictors:

\[
q_i(r_i).
\]

Construct:

\[
q_{add}(r_i,r_j)
=
q_i(r_i)+q_j(r_j).
\]

For target modality:

\[
r_k,
\]

define:

\[
e_{ij\rightarrow k}
=
r_k-q_{add}(r_i,r_j).
\]

---

# 29. D1 Alignment

Train:

\[
h_{ij}
\leftrightarrow
stopgrad(
e_{ij\rightarrow k}
).
\]

Example:

\[
h_{12}\leftrightarrow e_{12\rightarrow3}.
\]

---

# 30. D1 Limitation

The residual:

\[
r_k-q_{add}
\]

contains all structure not explained by the additive predictor.

This may include:

```text
true joint-predictable structure
target noise
unpredictable variation
predictor approximation error
```

Therefore D1 is a baseline, not the preferred method.

---

# 31. D2 — Joint-Advantage Distillation

D2 is the primary ConFu++ v4 discovery hypothesis.

For pair:

\[
(i,j)\rightarrow k,
\]

train:

### Additive predictor

\[
q_{add}(r_i,r_j)
=
q_i(r_i)+q_j(r_j).
\]

### Joint predictor

\[
q_{joint}(r_i,r_j)
=
q_J([r_i,r_j]).
\]

---

# 32. Capacity Matching

Require:

\[
Params(q_{joint})
\approx
Params(q_i)+Params(q_j).
\]

Target:

\[
<5\%
\]

difference.

Prefer:

\[
<1\%.
\]

Report exact parameter counts.

---

# 33. Joint Predictive Advantage

Define:

\[
\boxed{
d_{ij\rightarrow k}
=
q_{joint}(r_i,r_j)
-
q_{add}(r_i,r_j)
}
\]

with predictors frozen.

This quantity represents:

> structure that a joint predictor can model but a matched additive predictor cannot.

---

# 34. D2 Interaction Target

Train:

\[
\boxed{
h_{ij}
\leftrightarrow
stopgrad(
d_{ij\rightarrow k}
)
}
\]

using:

```text
cosine regression
MSE
contrastive alignment
```

as controlled variants.

Start with one simple objective.

Do NOT perform broad loss sweeps initially.

---

# 35. Why D2 Is Different From D1

D1:

\[
r_k-q_{add}.
\]

This contains everything the additive predictor misses.

D2:

\[
q_{joint}-q_{add}.
\]

This retains only what the joint predictor successfully models beyond additive predictors.

Thus:

\[
\boxed{
D2\text{ removes unpredictable target variation from the interaction target}.
}
\]

---

# 36. Predictor Training Order

For D2:

```text
1. Train independent predictors.

2. Freeze independent predictors.

3. Train joint predictor.

4. Freeze joint predictor.

5. Compute d_joint-advantage.

6. Train interaction representation.
```

Do NOT co-adapt predictors with:

\[
h_{ij}.
\]

---

# 37. Predictor Target

All predictors for:

\[
(i,j)\rightarrow k
\]

must predict the same:

\[
r_k.
\]

Do not compare predictors trained against different target representations.

---

# 38. Predictor Validation

Report:

\[
R^2_{add}
\]

and:

\[
R^2_{joint}.
\]

Define:

\[
\boxed{
J_{ij\rightarrow k}
=
R^2_{joint}
-
R^2_{add}.
}
\]

This is the Joint Prediction Advantage.

---

# 39. J Is Not Formal Synergy

Do NOT call:

\[
J
\]

PID synergy.

Use:

```text
joint prediction advantage
non-additive predictive advantage
```

It is hypothesis-class dependent.

---

# 40. Joint-Advantage Target Health

Before training:

\[
h_{ij},
\]

measure for:

\[
d_{ij\rightarrow k}:
\]

```text
variance
effective rank
mean norm
dimension std
```

If:

\[
Var(d)\approx0,
\]

do not force an interaction representation.

---

# 41. D0/D1/D2 Architecture Freeze

All discovery methods must use identical:

```text
pair interaction architecture
rank
output dimension
encoder
probe
optimizer budget where possible
```

Only discovery objective changes.

This is mandatory for a clean comparison.

---

# 42. Pair Architecture

Keep:

\[
u_i=LN(U_ir_i)
\]

\[
u_j=LN(U_jr_j).
\]

Then:

\[
\boxed{
h_{ij}
=
W_{ij}
\left[
\frac{
u_i\odot u_j
}{
\sqrt R
}
\right].
}
\]

Default:

\[
R=64.
\]

---

# 43. No Pair Rank Sweep

Do not sweep rank during D0/D1/D2.

Architecture has already passed Oracle.

The question is objective quality.

---

# 44. Discovery Evaluation Layer 1 — Ground-Truth Recovery

Synthetic benchmark provides true:

\[
g_{12}=z_1\odot z_2.
\]

Measure interaction recovery.

Possible metrics:

\[
Cosine(h_{12},P(g_{12}))
\]

and:

\[
R^2(h_{12}\rightarrow g_{12}).
\]

Repeat for each pair.

---

# 45. Recovery Mapping

If dimensions differ, train a simple validation-controlled linear probe:

\[
h_{ij}\rightarrow g_{ij}.
\]

Do not use a large MLP to claim recovery.

---

# 46. Discovery Evaluation Layer 2 — Accessibility

Primary metric:

\[
\boxed{
\Delta_2^{linear}
=
Perf_{linear}(Z_{\le2})
-
Perf_{linear}(Z_{\le1}).
}
\]

This remains the primary downstream metric.

---

# 47. Discovery Evaluation Layer 3 — Pair Selectivity

In S1 where only pair 12 is task relevant:

\[
\Delta_{12}>0.
\]

Require:

\[
\Delta_{12}
>
\Delta_{13}
\]

and:

\[
\Delta_{12}
>
\Delta_{23}.
\]

---

# 48. Selectivity Score

Define:

\[
\boxed{
Sel_{12}
=
\Delta_{12}
-
\frac{
\Delta_{13}+\Delta_{23}
}{2}.
}
\]

Positive:

\[
Sel_{12}>0
\]

supports pair-specific discovery.

---

# 49. Discovery Evaluation Layer 4 — False Positives

On S0:

\[
\Delta_{12},
\Delta_{13},
\Delta_{23}
\approx0.
\]

On S3:

\[
\Delta_2\approx0.
\]

Pair discovery must not substitute for a true order-3 representation.

---

# 50. Oracle Recovery Ratio

Define:

\[
\boxed{
Recovery_2
=
\frac{
\Delta_2^{method}
}{
\Delta_2^{oracle}+\epsilon
}
}
\]

when:

\[
\Delta_2^{oracle}
\]

is meaningfully positive.

---

# 51. Interpretation of Recovery₂

Example:

\[
\Delta_2^{oracle}=30
\]

and:

\[
\Delta_2^{D2}=15.
\]

Then:

\[
Recovery_2=0.5.
\]

Interpretation:

> D2 recovers approximately half of the accessibility gain available under Oracle supervision.

---

# 52. Recovery Ratio Warning

Do NOT report Recovery when:

\[
|\Delta_2^{oracle}|
\]

is near zero.

This applies to S0 and S3.

---

# 53. Main Discovery Table

Produce:

| Regime | Method | Δ12 | Δ13 | Δ23 | Δ2 | Oracle Recovery₂ |
|---|---|---:|---:|---:|---:|---:|
| S0 | D0 | | | | | N/A |
| S0 | D1 | | | | | N/A |
| S0 | D2 | | | | | N/A |
| S1 | D0 | | | | | |
| S1 | D1 | | | | | |
| S1 | D2 | | | | | |
| S2 | D0 | | | | | |
| S2 | D1 | | | | | |
| S2 | D2 | | | | | |
| S3 | D0 | | | | | N/A |
| S3 | D1 | | | | | N/A |
| S3 | D2 | | | | | N/A |

---

# 54. Ground-Truth Recovery Table

| Regime | Method | h12→g12 R² | h13→g13 R² | h23→g23 R² |
|---|---|---:|---:|---:|
| S1 | D0 | | | |
| S1 | D1 | | | |
| S1 | D2 | | | |
| S2 | D0 | | | |
| S2 | D1 | | | |
| S2 | D2 | | | |

---

# 55. Joint Advantage Table

| Pair→Target | Additive R² | Joint R² | J |
|---|---:|---:|---:|
| 12→3 | | | |
| 13→2 | | | |
| 23→1 | | | |

---

# 56. Linearization Evaluation

Continue to report:

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

Desired:

\[
LG_2<LG_1.
\]

---

# 57. D0 Success Definition

D0 is successful if it recovers meaningful pair accessibility in synthetic interaction regimes.

D0 does not need to match Oracle.

It serves as the original-ConFu discovery baseline.

---

# 58. D1 Success Definition

D1 succeeds if it consistently improves over D0 in:

```text
interaction recovery
Δ2
pair selectivity
```

without increasing false positive interactions.

---

# 59. D2 Success Definition

D2 pair mechanism passes if all of the following hold.

### S0

\[
\Delta_2\approx0.
\]

### S1

\[
\Delta_{correct-pair}>0.
\]

and:

\[
Sel_{correct}>0.
\]

### S2

\[
\boxed{
\Delta_2>0.
}
\]

### S3

\[
\Delta_2\approx0.
\]

### Relative Performance

Prefer:

\[
D2>D0
\]

and:

\[
D2>D1.
\]

---

# 60. D2 Does Not Need Oracle-Level Performance

D2 can be scientifically successful even if:

\[
Recovery_2<1.
\]

The goal is to demonstrate a meaningful fraction of Oracle accessibility without ground-truth interaction supervision.

---

# 61. Exploratory Protocol

First run:

```text
seed = 1
```

on:

```text
S0
S1
S2
S3
```

for:

```text
D0
D1
D2
```

Do not run S4 immediately.

---

# 62. Confirmatory Trigger

Proceed to five seeds only if D2 seed-1 shows:

```text
positive S1 correct-pair gain
positive S2 Delta2
no major false S0 gain
no major S3 pair gain
```

and implementation diagnostics pass.

---

# 63. Confirmatory Protocol

Freeze:

```text
architecture
rank
learning rate
predictor capacity
loss
temperature
training schedule
```

then run:

```text
seeds 1–5
```

---

# 64. Statistical Reporting

For:

\[
\Delta_2,
\]

report:

```text
mean
sample standard deviation
per-seed results
paired differences
95% CI
effect size
```

when relevant.

---

# 65. D2 Failure Interpretation

If:

\[
Oracle\gg0
\]

but:

\[
D0\approx D1\approx D2\approx0,
\]

conclude:

\[
\boxed{
\text{architecture is capable but current discovery targets cannot identify the interaction}.
}
\]

Do NOT change architecture immediately.

Research new discovery objectives.

---

# 66. If D0 Already Performs Well

If D0 recovers most Oracle gain:

\[
Recovery_2^{D0}
\]

is already high.

Then ConFu may already discover controlled pair interaction successfully.

ConFu++ must demonstrate benefit in:

```text
selectivity
noise robustness
sample efficiency
real-world transfer
```

rather than claiming ConFu cannot discover interaction.

---

# 67. If D1 > D0

This supports:

\[
\text{target decomposition}
\]

as useful.

But still compare against D2 to determine whether removing unpredictable target variation matters.

---

# 68. If D2 > D1

This supports the central hypothesis:

\[
\boxed{
q_{joint}-q_{add}
}
\]

is a cleaner target than:

\[
\boxed{
r_k-q_{add}.
}
\]

This would become the main ConFu++ method.

---

# 69. All-Branch False-Positive Test

Do not deactivate:

\[
h_{13},
h_{23}
\]

on S1.

All branches train simultaneously.

A good discovery method should learn that only the correct interaction is downstream useful.

---

# 70. False-Positive Gain

Define:

\[
FP_{S1}
=
\frac{
|\Delta_{wrong1}|+
|\Delta_{wrong2}|
}{2}.
\]

Desired:

\[
FP_{S1}
\ll
\Delta_{correct}.
\]

---

# 71. Selectivity Ratio

Optional:

\[
SR_{12}
=
\frac{
\Delta_{12}
}{
|\Delta_{13}|+
|\Delta_{23}|+\epsilon
}.
\]

Diagnostic only.

Do not overinterpret unstable ratios.

---

# 72. Interaction Health

For D0/D1/D2, log:

```text
variance
effective rank
dimension std
mean norm
```

for each:

\[
h_{ij}.
\]

---

# 73. Shortcut Diagnostics

Measure:

\[
R^2(r_i\rightarrow h_{ij})
\]

and:

\[
R^2(r_j\rightarrow h_{ij}).
\]

No arbitrary threshold.

Use as diagnostic for modality domination.

---

# 74. Modality Intervention

For each:

\[
h_{ij},
\]

shuffle modality \(i\) and modality \(j\) independently.

Measure downstream drop.

Both should matter when the correct interaction is present.

---

# 75. Discovery Target Diagnostics

For D2:

\[
d_{ij\rightarrow k},
\]

report:

```text
variance
effective rank
norm
cosine to Oracle interaction target
linear recoverability of g_ij
```

Synthetic data makes this possible.

---

# 76. Direct Target-to-Oracle Comparison

Measure:

\[
R^2(
d_{ij\rightarrow k}
\rightarrow
g_{ij}
).
\]

This tests whether Joint-Advantage target itself contains the true interaction before:

\[
h_{ij}
\]

is trained.

---

# 77. D2 Failure Localization

If:

\[
d\rightarrow g
\]

has high recovery but:

\[
h\rightarrow g
\]

is poor:

\[
\boxed{
\text{distillation/alignment failure}.
}
\]

If:

\[
d\rightarrow g
\]

is already poor:

\[
\boxed{
\text{target construction failure}.
}
\]

This distinction is mandatory.

---

# 78. Loss Choice for D2

Start with a simple normalized regression:

\[
L_{D2}
=
1-
cos(
P_h(h_{ij}),
P_d(d_{ij\rightarrow k})
).
\]

Optionally combine with:

\[
MSE.
\]

Do not start with multiple complex objectives.

---

# 79. Contrastive D2 Ablation

InfoNCE may be evaluated later as:

```text
D2-contrastive
```

only after regression D2 is characterized.

Do not confound initial mechanism experiment.

---

# 80. Third-Order Discovery Is Blocked

Even though Oracle:

\[
h_{123}
\]

works, discovery:

\[
h_{123}
\]

must NOT be implemented yet.

Reason:

order-3 discovery must distinguish:

\[
\text{order 3}
\]

from both:

\[
\text{order 1}
\]

and:

\[
\text{order 2}.
\]

---

# 81. Third-Order Discovery Trigger

Require:

```text
D2 pair discovery passes
pair selectivity passes
false pair gains controlled
five-seed pair result stable
```

before designing order-3 discovery.

---

# 82. Future Order-3 Concept

Eventually compare:

### Lower-order predictor

\[
q_{\le2}
\]

against:

### Fully joint three-modal predictor

\[
q_{123}.
\]

Potential target:

\[
d_{123}
=
q_{123}(r_1,r_2,r_3)
-
q_{\le2}(r_1,r_2,r_3).
\]

This is only a future hypothesis.

Do NOT implement until pair discovery is validated.

---

# 83. Important Third-Order Requirement

The lower-order predictor for order 3 must include:

```text
all order-1 effects
all order-2 effects
```

otherwise:

\[
d_{123}
\]

will be contaminated by pairwise interactions.

---

# 84. Third-Order Oracle Remains Reference

Oracle:

\[
\Delta_3^{oracle}
\]

provides the upper-bound-style accessibility reference for future third-order discovery.

---

# 85. Generic MLP Sanity Must Be Finished Before Strong Claims

Do not claim:

> explicit interaction representation is more powerful than generic MLP

until MLP optimization/capacity controls are complete.

The valid current claim is:

> explicit Oracle interaction representations make the synthetic higher-order structure linearly accessible.

---

# 86. Noise Stress Test

Do NOT run noise sweeps before D2 passes clean synthetic.

After success, use:

\[
\sigma
\in
\{0,0.1,0.25,0.5,1.0\}.
\]

Measure:

\[
\Delta_2
\]

and:

\[
Recovery_2.
\]

---

# 87. Interaction Strength Sweep

After D2 passes:

\[
\beta
\in
\{0,0.25,0.5,1,2\}.
\]

Desired:

\[
\Delta_2
\]

should generally increase with pair signal strength.

---

# 88. Data Sample-Efficiency Sweep

Later evaluate:

```text
1k
5k
10k
20k
```

training samples.

Compare:

```text
generic MLP
ConFu
D2
Oracle
```

This can reveal whether explicit interaction inductive bias improves sample efficiency.

---

# 89. Real-World Return Is Blocked

Do not return to:

```text
UR-FUNNY tuning
MOSI tuning
MOSEI architecture development
```

until pair discovery passes controlled synthetic.

---

# 90. Real-World Dataset Screening

After D2 synthetic passes, screen natural tasks using:

\[
P0=\text{linear additive}
\]

\[
P1=\text{nonlinear additive}
\]

\[
P2=\text{capacity-matched joint}.
\]

---

# 91. Preferred Real-World Next Dataset

First candidate:

\[
\boxed{
CMU\text{-}MOSEI\ Emotion
}
\]

not sentiment.

Reason:

emotion may contain more acoustic/visual headroom than strongly text-dominated sentiment.

This is a hypothesis to test, not an assumption.

---

# 92. MOSEI Screening

For each emotion/task:

\[
J=P2-P1.
\]

Rank tasks by validated joint advantage only for benchmark selection.

Do not claim one task is intrinsically more synergistic from this metric alone.

---

# 93. UR-FUNNY Role

UR-FUNNY remains valuable as:

```text
negative mechanism result
real-world transfer benchmark
comparison with Original ConFu
```

Return to it after discovery method is validated elsewhere.

---

# 94. Real-World Metrics

When returning to real data, evaluate:

```text
classification accuracy
F1 where appropriate
linear order accessibility
nonlinear probe
retrieval
modality interventions
```

---

# 95. Preserve Original ConFu Capability

ConFu++ should not improve classification by destroying ConFu's multimodal alignment behavior.

Where available retain:

```text
1→1 retrieval
2→1 retrieval
```

evaluation.

---

# 96. Accuracy Is Important but Not Sufficient

A stronger supervised fusion head can increase accuracy without discovering explicit interactions.

Therefore the final method must demonstrate:

\[
\boxed{
\text{mechanism gain}
+
\text{task gain}
}
\]

not task gain alone.

---

# 97. Capacity Baseline on Real Data

Any ConFu++ parameter increase requires:

\[
\text{parameter-matched generic baseline}.
\]

This separates:

\[
\text{better objective}
\]

from:

\[
\text{more capacity}.
\]

---

# 98. No Cross-Attention Yet

Do NOT add cross-attention until:

```text
simple multiplicative architecture works with D2
```

or Oracle experiments show a specific capacity limitation.

---

# 99. No Rank Sweep Yet

Architecture already passes Oracle.

Rank sweep is not the current bottleneck.

---

# 100. No Dynamic Routing Yet

Routing is irrelevant until interaction discovery itself works.

---

# 101. No Task Loss Yet

Do not inject labels into D0/D1/D2.

First characterize representation discovery without direct task supervision.

A supervised task-aware extension may be considered later.

---

# 102. No UR-FUNNY Accuracy Chase

Do not optimize specifically for:

\[
64.462\rightarrow X
\]

until the discovery mechanism is validated.

Otherwise accuracy improvement may be uninterpretable.

---

# 103. Mapping Safety

Use explicit names:

```text
12_to_3
13_to_2
23_to_1
```

Everywhere.

Never infer mapping from array position.

---

# 104. Required Mapping

Canonical mapping:

\[
h_{12}\rightarrow target_3
\]

\[
h_{13}\rightarrow target_2
\]

\[
h_{23}\rightarrow target_1.
\]

Unit-test it.

---

# 105. Synthetic Ground-Truth Mapping

Ground-truth interaction:

```text
h12 ↔ g12
h13 ↔ g13
h23 ↔ g23
h123 ↔ g123
```

This is different from cross-modal discovery target mapping.

Do not conflate the two.

---

# 106. Unit Tests

Required:

```text
correct S0–S4 regime
correct pair mapping
correct cross-target mapping
all branches enabled
no split leakage
predictors frozen before distillation
capacity matching calculation
seed determinism
```

---

# 107. Reproducibility

Each run saves:

```text
seed
git commit
config
regime
method
parameter counts
best epoch
checkpoint
target health
interaction health
probe metrics
ground-truth recovery
```

---

# 108. Recommended Repository Structure

Use:

```text
src/experiments/synthetic_order/
    dataset.py
    models.py
    predictors.py
    oracle.py
    d0_confu.py
    d1_residual.py
    d2_joint_advantage.py
    probes.py
    diagnostics.py
    metrics.py
    runner.py
```

---

# 109. D0/D1/D2 Common API

All discovery methods should return:

```text
h12
h13
h23

training losses

target diagnostics

interaction diagnostics
```

with a shared interface.

---

# 110. Suggested Experiment IDs

```text
O — Oracle

D0 — ConFu Alignment

D1 — Residual Alignment

D2 — Joint-Advantage Distillation
```

Examples:

```text
S1_D0_seed1
S1_D1_seed1
S1_D2_seed1
```

---

# 111. D2 Result JSON

Minimum:

```json
{
  "regime": "S1",
  "seed": 1,

  "additive_r2_12_to_3": 0.0,
  "joint_r2_12_to_3": 0.0,
  "joint_advantage_12_to_3": 0.0,

  "target_variance_12": 0.0,
  "target_effective_rank_12": 0.0,

  "h12_oracle_r2": 0.0,
  "h13_oracle_r2": 0.0,
  "h23_oracle_r2": 0.0,

  "z1_linear_accuracy": 0.0,
  "z2_linear_accuracy": 0.0,

  "delta12": 0.0,
  "delta13": 0.0,
  "delta23": 0.0,
  "delta2": 0.0,

  "selectivity12": 0.0,
  "oracle_recovery2": 0.0
}
```

---

# 112. Experiment Report Template

Every D0/D1/D2 report must contain:

```text
Research Question

Hypothesis

Regime

Architecture

Discovery target

Predictor parameter counts

Target quality

Interaction recovery

Order accessibility

Pair selectivity

False-positive interaction

Health metrics

Modality interventions

Oracle recovery ratio

Decision

Next action
```

---

# 113. Decision Rule for D0

D0 tells us whether Original ConFu alignment can discover controlled pair structure.

Do not alter D0 to make it stronger.

It is a baseline.

---

# 114. Decision Rule for D1

D1 tells us whether subtracting additive prediction improves discovery.

Keep the formulation faithful to the residual hypothesis.

---

# 115. Decision Rule for D2

D2 is successful only if it improves interaction discovery while preserving order specificity.

A higher:

\[
\Delta_2
\]

with large false-positive pair gains is NOT sufficient.

---

# 116. D2 Preferred Pattern

S1 ideal:

```text
Delta12: strongly positive
Delta13: approximately zero
Delta23: approximately zero
```

S2:

```text
Delta2: positive
```

S0:

```text
Delta2: approximately zero
```

S3:

```text
Delta2: approximately zero
```

---

# 117. If D2 Passes Seed 1

Do NOT immediately modify architecture.

Freeze configuration.

Run:

\[
5\text{ seeds}.
\]

---

# 118. If D2 Passes Five Seeds

Next:

```text
noise sweep
interaction-strength sweep
sample-efficiency sweep
```

Then design order-3 discovery.

---

# 119. If D2 Fails

Do not abandon the architecture.

Oracle proves architecture capability.

The next research target becomes:

\[
\boxed{
\text{better interaction target discovery}.
}
\]

---

# 120. Possible Future Objective Families

Only after D2 characterization consider:

```text
teacher disagreement
conditional predictive coding
counterfactual prediction
cross-modal error decomposition
functional ANOVA distillation
```

Do not implement all at once.

---

# 121. Third-Order Discovery Research Question

Future:

> What can a fully joint three-modal model predict beyond all validated order-1 and order-2 predictors?

This is the correct order-3 discovery framing.

---

# 122. Future Order-3 Target

Conceptual:

\[
d_{123}
=
q_{joint123}(r_1,r_2,r_3)
-
q_{\le2}(r_1,r_2,r_3).
\]

Where:

\[
q_{\le2}
\]

contains all validated lower-order predictor components.

---

# 123. Do Not Use Simple Triple Residual

Do NOT use:

\[
r_k-
q_{order1}
\]

as an order-3 target.

It mixes pair and triple effects.

---

# 124. Main Paper Story If D2 Works

The scientific story becomes:

1. ConFu performs higher-order alignment but does not explicitly guarantee interaction recovery.

2. Real-world diagnostics reveal shortcutting, redundancy, and low accessibility.

3. Controlled Oracle experiments show that multiplicative order-specific architecture is fully capable.

4. Standard alignment and residual objectives are evaluated under known ground truth.

5. Joint-Advantage Distillation isolates predictable non-additive structure.

6. Explicit interaction embeddings recover higher-order accessibility.

7. The mechanism transfers to real multimodal tasks.

---

# 125. Main Contribution Candidate

If successful:

\[
\boxed{
\textbf{Joint-Advantage Distillation}
}
\]

for explicit higher-order multimodal representation learning.

---

# 126. Preferred Terminology

Use:

```text
interaction representation
order-specific embedding
non-additive predictive advantage
joint prediction advantage
interaction accessibility
order recovery
```

---

# 127. Avoid

Do not claim:

```text
exact synergy
PID decomposition
new Shannon information
pure complementary information
```

without formal analysis.

---

# 128. Paper Accuracy Positioning

ConFu classification accuracy can be improved later.

But the current contribution should first explain:

\[
\boxed{
\text{why higher-order alignment does not necessarily yield task-accessible higher-order representations}.
}
\]

This is a stronger scientific foundation than accuracy chasing.

---

# 129. Next Immediate Coding Tasks

Implement exactly:

```text
1. Finish Generic MLP sanity.

2. Add effective-rank/std diagnostics.

3. Add pair and triple shuffle diagnostics.

4. Ensure all interaction branches stay enabled.

5. Implement D0 Original ConFu objective.

6. Implement D1 residual objective.

7. Implement D2 additive/joint predictors.

8. Implement D2 target:
   joint prediction - additive prediction.

9. Implement ground-truth interaction recovery probes.

10. Implement Delta12/13/23 and Delta2.

11. Implement pair selectivity.

12. Implement Oracle Recovery2.

13. Run S0/S1/S2/S3 seed 1.

14. Compare D0/D1/D2.

15. Freeze D2 if it passes.

16. Run five seeds.
```

---

# 130. Current Forbidden Work

Until D2 five-seed confirmation, do NOT:

```text
implement discovered h123
return to UR-FUNNY optimization
fine-tune large encoders
add cross-attention
sweep interaction rank
add dynamic routing
add task-supervised interaction loss
```

---

# 131. Current Milestone

The next milestone is:

\[
\boxed{
\text{recover pairwise Oracle structure without Oracle labels}.
}
\]

Not:

\[
\boxed{
\text{beat UR-FUNNY accuracy baseline}.
}
\]

---

# 132. Strong D2 Milestone

The strongest immediate result would be:

\[
\boxed{
D2>D1>D0
}
\]

on:

\[
\Delta_2
\]

and interaction ground-truth recovery,

while maintaining:

\[
\boxed{
\text{low false-positive order gain}.
}
\]

---

# 133. Project Decision Tree

```text
Oracle architecture
       │
       └── PASS
            │
            ▼
       D0 ConFu
            │
            ▼
       D1 Residual
            │
            ▼
       D2 Joint Advantage
            │
       ┌────┴─────┐
       │          │
     FAIL        PASS
       │          │
       ▼          ▼
 new discovery   five seeds
 objective         │
                   ▼
              robustness
                   │
                   ▼
              order-3
                   │
                   ▼
              real-world
```

---

# 134. Final Scientific Principle

Whenever an experiment fails, ask:

```text
Is the architecture incapable?

Is the target wrong?

Is the objective unable to discover the target?

Is the downstream probe unable to access it?

Is the dataset missing the interaction?
```

Do not solve all five problems by making the neural network larger.

---

# 135. Current Answer

Architecture capability:

\[
\boxed{PASS}
\]

Discovery capability:

\[
\boxed{UNKNOWN}
\]

Real-world superiority:

\[
\boxed{UNKNOWN}
\]

Therefore the only high-priority research problem now is:

\[
\boxed{
\textbf{Can ConFu++ discover pair interaction structure without Oracle supervision?}
}
\]

The immediate experiment is:

\[
\boxed{
\textbf{D0 vs D1 vs D2 on controlled synthetic S0–S3.}
}
\]

---

# 136. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ aims to distill the predictive advantage of joint multimodal models over capacity-matched additive models into explicit, order-specific interaction embeddings.}
}
\]