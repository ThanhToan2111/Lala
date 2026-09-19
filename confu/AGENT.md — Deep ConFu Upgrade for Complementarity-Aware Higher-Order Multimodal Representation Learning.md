# AGENT.md

# Project: ConFu++

## Complementarity-Aware Global–Local Higher-Order Multimodal Representation Learning

---

# 0. Scope

This repository is a direct research extension of:

**The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment — CVPR 2026**

The project must remain focused on:

- multimodal representation learning,
- contrastive fusion,
- pairwise alignment,
- higher-order alignment,
- multimodal complementarity,
- interaction representations,
- global/local multimodal representations,
- conditional utility,
- redundancy,
- higher-order synergy.

Do NOT redirect this project toward:

- egocentric video,
- IMU,
- robotics,
- pose estimation,
- VLA,
- SLAM,
- domain-specific applications unrelated to the original paper.

The immediate scientific goal is to improve ConFu itself.

---

# 1. Original ConFu Baseline

ConFu starts from modality-specific encoders:

\[
h_i = E_i(x_i)
\]

and projection heads:

\[
z_i=P_i(h_i).
\]

For three modalities:

\[
X_1,X_2,X_3,
\]

pairwise contrastive learning aligns:

\[
z_1\leftrightarrow z_2
\]

\[
z_1\leftrightarrow z_3
\]

\[
z_2\leftrightarrow z_3.
\]

ConFu additionally creates fused representations:

\[
z_{12}=F_{12}(h_1,h_2)
\]

\[
z_{13}=F_{13}(h_1,h_3)
\]

\[
z_{23}=F_{23}(h_2,h_3)
\]

and aligns:

\[
z_{12}\leftrightarrow z_3
\]

\[
z_{13}\leftrightarrow z_2
\]

\[
z_{23}\leftrightarrow z_1.
\]

Conceptually:

```text
PAIRWISE

X1 <--------> X2
X1 <--------> X3
X2 <--------> X3


HIGHER-ORDER

(X1 + X2) <--------> X3
(X1 + X3) <--------> X2
(X2 + X3) <--------> X1
```

This is the baseline that this project extends.

---

# 2. Core Limitation We Are Investigating

The original ConFu fusion representation:

\[
z_{12}=F(z_1,z_2)
\]

is not explicitly required to contain information that is complementary to:

\[
z_1
\]

and:

\[
z_2.
\]

Therefore it is possible that:

\[
z_{12}\approx f(z_1)
\]

or:

\[
z_{12}\approx g(z_2),
\]

while still achieving a valid contrastive objective.

This means:

\[
I(Y;z_{12})>0
\]

does NOT imply:

\[
I(Y;z_{12}\mid z_1,z_2)>0.
\]

This distinction is the central research problem.

---

# 3. Current Experimental Finding

The current clean AV-MNIST benchmark has already shown:

```text
interaction collapse            substantially fixed

interaction variance            healthy

permutation sensitivity         present

interaction predictive power    strong

stable complementary gain       absent
```

Observed behavior:

```text
probe(r12)                         high

probe([r1,r2])                     high

probe([r1,r2,r12])                 approximately unchanged
```

Therefore:

\[
r_{12}
\]

contains task information, but that information appears to be mostly redundant with the information already available in:

\[
(r_1,r_2).
\]

The project must now target:

\[
\boxed{
\text{conditional complementarity}
}
\]

rather than merely:

\[
\boxed{
\text{interaction activity}
}
\]

---

# 4. New Main Research Question

The primary research question is:

> Can fused multimodal representations be explicitly optimized to capture information that is useful only after conditioning on the individual modality representations?

The target quantity is conceptually:

\[
\boxed{
I(Y;r_{12}\mid r_1,r_2)
}
\]

rather than:

\[
I(Y;r_{12}).
\]

For three modalities, the eventual target becomes:

\[
I(Y;r_{123}\mid
r_1,r_2,r_3,
r_{12},r_{13},r_{23}).
\]

---

# 5. Terminology

Use terminology carefully.

## Interaction representation

Any representation computed jointly from multiple modalities.

Example:

\[
r_{12}=F(r_1,r_2).
\]

## Predictive interaction

An interaction representation that predicts the target.

## Interaction dependency

An interaction representation that genuinely depends on more than one modality.

## Interaction utility

An interaction representation that improves downstream performance.

## Complementary interaction

An interaction representation that improves performance beyond what individual modalities already provide.

## Synergy

Reserve the term **synergy** for interaction information with demonstrated conditional utility.

Do NOT call every fused representation synergistic.

---

# 6. Research Hypothesis

The new hypothesis is:

> Existing higher-order contrastive learning encourages fused representations to be predictive and aligned, but does not guarantee that these representations contain information beyond lower-order modality representations.

The proposed extension will attempt to learn:

\[
\boxed{
\text{predictive}
+
\text{non-collapsed}
+
\text{multi-modal dependent}
+
\text{conditionally useful}
}
\]

interaction representations.

---

# 7. Research Direction

The project will extend ConFu along three axes:

```text
Axis 1
Conditional Utility

Axis 2
Global–Local Interaction

Axis 3
Higher-Order Residual Representation
```

These must be introduced incrementally.

Do NOT implement all three simultaneously.

---

# 8. Stage A — Diagnose Existing Interaction

Before changing the architecture further, complete the following diagnostics on the current model.

---

# 9. Single-Modality Shuffle Test

Given:

\[
r_{12}=F(r_1,r_2),
\]

evaluate:

\[
F(r_1,r_2)
\]

\[
F(r_1,\pi(r_2))
\]

\[
F(\pi(r_1),r_2)
\]

where:

\[
\pi
\]

is a random batch permutation.

Record:

```text
normal

shuffle modality 1

shuffle modality 2

shuffle both modalities
```

Define:

\[
D_1=A_{normal}-A_{shuffle1}
\]

\[
D_2=A_{normal}-A_{shuffle2}.
\]

---

# 10. Shuffle Interpretation

If:

\[
D_1\gg D_2,
\]

the interaction is modality-1 dominant.

If:

\[
D_2\gg D_1,
\]

the interaction is modality-2 dominant.

If both are significant:

\[
D_1>0,\quad D_2>0,
\]

the interaction depends on both modalities.

This still does NOT prove complementarity.

---

# 11. Interaction Zero Test

Evaluate:

\[
r_{12}=0.
\]

Compare:

\[
A_{normal}
\]

and:

\[
A_{zero}.
\]

Define:

\[
ZeroDrop=A_{normal}-A_{zero}.
\]

Interpretation:

```text
ShuffleDrop > 0
does not prove utility.

ZeroDrop > 0
provides stronger evidence that the interaction is required.
```

---

# 12. Interaction Shuffle Test

Shuffle:

\[
r_{12}
\]

between samples while leaving:

\[
r_1,r_2
\]

unchanged.

Define:

\[
ShuffleDrop
=
A_{normal}-A_{shuffle-r12}.
\]

Do not interpret ShuffleDrop alone as synergy.

A mismatched interaction can simply act as an adversarial perturbation.

---

# 13. Gate Sweep

For the same checkpoint evaluate:

\[
g\in
\{
0,
0.25,
0.5,
0.75,
1
\}.
\]

For:

\[
c_{12}
=
LN(
LN(r_1)
+
LN(r_2)
+
gLN(r_{12})
).
\]

Store:

```text
accuracy

macro-F1

loss

correction count

regression count
```

for every value.

---

# 14. Correction–Regression Analysis

Compare predictions from:

\[
c_{base}=Compose(r_1,r_2)
\]

against:

\[
c_{full}=Compose(r_1,r_2,r_{12}).
\]

Count:

```text
base wrong -> full correct

base correct -> full wrong

both correct

both wrong
```

Define:

\[
Correction =
N_{\text{base wrong/full correct}}
\]

\[
Regression =
N_{\text{base correct/full wrong}}
\]

and:

\[
NetCorrection=Correction-Regression.
\]

Report per class.

---

# 15. Conditional Linear Probe

Freeze encoder representations.

Train:

\[
Probe([r_1,r_2])
\]

and:

\[
Probe([r_1,r_2,r_{12}]).
\]

Define:

\[
Gain^{linear}_{12}
=
Perf([r_1,r_2,r_{12}])
-
Perf([r_1,r_2]).
\]

---

# 16. Conditional Nonlinear Probe

Repeat with a small parameter-matched MLP.

Use identical:

```text
hidden dimensions

optimizer

epochs

weight decay

seed
```

for:

\[
MLP([r_1,r_2])
\]

and:

\[
MLP([r_1,r_2,r_{12}]).
\]

Define:

\[
Gain^{nonlinear}_{12}.
\]

If both:

\[
Gain^{linear}_{12}\approx0
\]

and:

\[
Gain^{nonlinear}_{12}\approx0,
\]

there is strong empirical evidence that:

\[
r_{12}
\]

is largely redundant.

---

# 17. Stage B — Conditional Utility Objective

The first architectural upgrade must be minimal.

Do NOT add cross-attention yet.

Add a direct utility objective.

---

# 18. Base Representation

Define additive/global baseline:

\[
c_{base}
=
LN(
LN(r_1)
+
LN(r_2)
).
\]

Interaction representation:

\[
r_{12}=F(r_1,r_2).
\]

Full representation:

\[
c_{full}
=
LN(
c_{base}
+
gLN(r_{12})
).
\]

---

# 19. Base Loss

Compute:

\[
L_{base}
=
CE(c_{base},y).
\]

---

# 20. Full Loss

Compute:

\[
L_{full}
=
CE(c_{full},y).
\]

---

# 21. Utility Margin

Introduce:

\[
\boxed{
L_{utility}
=
\max(
0,
L_{full}
-
stopgrad(L_{base})
+
m
)
}
\]

where:

\[
m\ge0.
\]

Initial:

```yaml
utility:
  weight: 0.1
  margin: 0.0
```

This explicitly rewards the interaction only when it helps relative to lower-order evidence.

---

# 22. Utility Objective Interpretation

The previous objective asks:

> Can the fused representation predict the correct class?

The new objective asks:

> Can the fused representation predict better than the lower-order representation?

This distinction is fundamental.

---

# 23. Utility Weight Sweep

Only after a seed-1 sanity run.

Test:

```text
lambda_utility

0.0
0.05
0.1
0.2
0.5
```

Do not sweep:

```text
interaction rank
model width
encoder size
```

at the same time.

---

# 24. Utility Margin Sweep

Only after selecting a reasonable:

\[
\lambda_{utility}.
\]

Test:

```text
m = 0
m = 0.01
m = 0.05
```

Do not use aggressive positive margins initially.

---

# 25. Interaction Dependence Objective

If single-modality shuffle shows one modality dominates, introduce an auxiliary dependence objective.

The fused representation should be sensitive to both modalities.

Construct positives:

\[
r_{12}^{+}=F(r_1,r_2)
\]

and corrupted interactions:

\[
r_{12}^{(1-)}
=
F(\pi(r_1),r_2)
\]

\[
r_{12}^{(2-)}
=
F(r_1,\pi(r_2)).
\]

Require:

\[
Score(r_{12}^{+})
>
Score(r_{12}^{(1-)})
\]

and:

\[
Score(r_{12}^{+})
>
Score(r_{12}^{(2-)}).
\]

---

# 26. Dependence Loss

Possible implementation:

\[
L_{dep}
=
\max(
0,
m_d
-
s(r_{12}^{+})
+
s(r_{12}^{(1-)})
)
\]

\[
+
\max(
0,
m_d
-
s(r_{12}^{+})
+
s(r_{12}^{(2-)})
).
\]

Do NOT add this unless modality-dominance diagnostics justify it.

---

# 27. Anti-Redundancy Revisited

Current covariance regularization handles approximately linear redundancy.

It does NOT guarantee:

\[
r_{12}\neq f(r_1)
\]

for nonlinear:

\[
f.
\]

Therefore nonlinear redundancy must be investigated.

---

# 28. Predictability Test

Train predictors:

\[
q_1(r_1)\rightarrow r_{12}
\]

\[
q_2(r_2)\rightarrow r_{12}
\]

and:

\[
q_{12}(r_1,r_2)\rightarrow r_{12}.
\]

Measure:

```text
MSE

cosine similarity

R²
```

If:

\[
q_1(r_1)
\]

predicts nearly all of:

\[
r_{12},
\]

the interaction is modality-1 redundant.

---

# 29. Conditional Residual Representation

Introduce only after redundancy is confirmed.

Predict the interaction from lower-order evidence:

\[
\hat r_{12}
=
q(r_1,r_2).
\]

Define:

\[
\boxed{
r_{12}^{R}
=
r_{12}
-
stopgrad(\hat r_{12})
}
\]

as an operational residual representation.

This is NOT claimed to be exact information decomposition.

---

# 30. Residual Prediction Loss

Train:

\[
q(r_1,r_2)
\]

with:

\[
L_{pred}
=
\|
q(r_1,r_2)
-
stopgrad(r_{12})
\|_2^2.
\]

The residual branch remains task-trained.

---

# 31. Residual Utility

Compare:

\[
[r_1,r_2]
\]

against:

\[
[r_1,r_2,r_{12}]
\]

and:

\[
[r_1,r_2,r_{12}^{R}].
\]

The residual representation is useful only if it produces:

\[
Gain^{R}
>
Gain^{raw}
\]

or reduces regressions while preserving corrections.

---

# 32. Stage C — Global–Local ConFu

Only after conditional-utility experiments are complete, investigate representation scale.

Original ConFu operates primarily on global embeddings.

The new question is:

> Does global pooling remove local cross-modal evidence that is necessary for complementarity?

---

# 33. Global Representation

For modality:

\[
X_i,
\]

retain token or spatial representations:

\[
H_i
\in
\mathbb R^{B\times T_i\times D}.
\]

Global representation:

\[
r_i^G
=
Pool(H_i).
\]

---

# 34. Global Interaction

Keep the validated global interaction:

\[
r_{12}^{G}
=
F_G(r_1^G,r_2^G).
\]

This must remain a baseline.

---

# 35. Local Representations

Instead of using every encoder token immediately, compress:

\[
H_i
\]

into:

\[
K
\]

learned local slots.

Initial:

\[
K=4.
\]

Output:

\[
P_i
\in
\mathbb R^{B\times K\times D}.
\]

---

# 36. Local Slot Pooling

Implement learned-query pooling:

\[
P_i
=
AttnPool_K(H_i).
\]

Do NOT use a deep cross-modal Transformer in the first local experiment.

The goal is to preserve sub-global information with minimal additional capacity.

---

# 37. Local Multiplicative Interaction

For local slots:

\[
p_{1,k},
p_{2,k},
\]

compute:

\[
u_{1,k}
=
LN(W_1p_{1,k})
\]

\[
u_{2,k}
=
LN(W_2p_{2,k}).
\]

Interaction:

\[
s_k
=
u_{1,k}
\odot
u_{2,k}.
\]

Pool:

\[
r_{12}^{L}
=
AttnPool(
s_1,\ldots,s_K
).
\]

---

# 38. Local All-to-All Interaction

Second variant:

\[
s_{ij}
=
LN(W_1p_{1,i})
\odot
LN(W_2p_{2,j}).
\]

Compatibility:

\[
a_{ij}
=
w^Ts_{ij}.
\]

Weights:

\[
\alpha_{ij}
=
softmax(a_{ij}).
\]

Representation:

\[
r_{12}^{L}
=
\sum_{ij}
\alpha_{ij}s_{ij}.
\]

Call this:

**Sparse Local ConFu**.

---

# 39. Do Not Assume Local Correspondence

For AV-MNIST, local image regions and local audio regions are not physically aligned.

Therefore local interaction must be interpreted as:

```text
sub-global semantic interaction
```

not:

```text
physical token correspondence
```

Do not make stronger claims.

---

# 40. Global-Conditioned Local Interaction

The local branch should eventually answer:

> What local information remains important after observing the global representation?

Context:

\[
q_G
=
MLP(c_G).
\]

Interaction weighting:

\[
a_{ij}
=
q_G^TW_s s_{ij}.
\]

Then:

\[
\alpha_{ij}=softmax(a_{ij})
\]

and:

\[
r_{12}^{L}
=
\sum_{ij}
\alpha_{ij}s_{ij}.
\]

---

# 41. Global Prediction

Construct:

\[
c_G
=
Compose(
r_1^G,
r_2^G,
r_{12}^G
).
\]

Predict:

\[
l_G.
\]

---

# 42. Local Correction

Instead of concatenating:

\[
c_G
\]

and:

\[
r_{12}^{L},
\]

predict a correction:

\[
\Delta l_L
=
W_Lr_{12}^{L}.
\]

Final:

\[
\boxed{
l_F
=
l_G
+
g_L\Delta l_L
}
\]

This forces local interaction to behave as a correction mechanism.

---

# 43. Why Residual Logit Correction

The goal is not:

> Can local features independently classify the sample?

The goal is:

> Can local multimodal information correct what global fusion misses?

Therefore logit residual is preferred over naive concatenation for the first experiment.

---

# 44. Global–Local Utility Loss

Define:

\[
L_G
=
CE(l_G,y)
\]

and:

\[
L_F
=
CE(l_F,y).
\]

Then:

\[
L_{GL-utility}
=
\max(
0,
L_F-stopgrad(L_G)+m
).
\]

---

# 45. Global–Local Residual Representation

Later predict:

\[
\hat r_{12}^{L}
=
q(c_G).
\]

Define:

\[
r_{12}^{LR}
=
r_{12}^{L}
-
stopgrad(\hat r_{12}^{L}).
\]

Then:

\[
\Delta l
=
W_Rr_{12}^{LR}.
\]

This is the candidate:

\[
\boxed{
\text{Global–Local Residual Synergy}
}
\]

representation.

---

# 46. Proposed Representation Hierarchy

For a pair of modalities:

\[
\boxed{
\mathcal R_{12}
=
\{
r_1^G,
r_2^G,
r_{12}^G,
r_{12}^L,
r_{12}^{LR}
\}
}
\]

Interpretation:

```text
r1G
global modality-1 information

r2G
global modality-2 information

r12G
global interaction

r12L
local/sub-global interaction

r12LR
local interaction unexplained by global evidence
```

---

# 47. Stage D — Return to True Higher-Order ConFu

AV-MNIST is useful for pair interaction analysis.

It is NOT the final higher-order benchmark.

After pairwise complementarity is validated, return to genuine three-modality datasets used by or closely related to ConFu.

Candidate benchmark families:

```text
MOSI

MUStARD

UR-FUNNY

Bird-MML

other genuine tri-modal ConFu benchmarks
```

Use exact original splits and protocols where possible.

---

# 48. Three-Modality Representation

For:

\[
X_1,X_2,X_3,
\]

learn:

\[
r_1,r_2,r_3
\]

\[
r_{12},r_{13},r_{23}
\]

and:

\[
r_{123}.
\]

Do NOT call:

\[
r_{123}
\]

synergistic unless it provides conditional utility beyond all lower-order representations.

---

# 49. Third-Order Conditional Utility

Lower-order evidence:

\[
R_{lower}
=
[
r_1,r_2,r_3,
r_{12},r_{13},r_{23}
].
\]

Third-order:

\[
r_{123}.
\]

Evaluate:

\[
Probe(R_{lower})
\]

versus:

\[
Probe(R_{lower},r_{123}).
\]

Define:

\[
\boxed{
Gain_{123}
=
Perf(R_{lower},r_{123})
-
Perf(R_{lower})
}
\]

This is more meaningful than simply evaluating:

\[
r_{123}
\]

alone.

---

# 50. Third-Order Utility Objective

Define lower-order prediction:

\[
l_{lower}.
\]

Full prediction:

\[
l_{full}.
\]

Then:

\[
L_{utility}^{123}
=
\max(
0,
CE(l_{full},y)
-
stopgrad(CE(l_{lower},y))
+
m
).
\]

This directly extends pairwise conditional utility to third-order interaction.

---

# 51. Third-Order Dependency Test

Evaluate:

\[
r_{123}(r_1,r_2,r_3)
\]

against:

\[
r_{123}(\pi(r_1),r_2,r_3)
\]

\[
r_{123}(r_1,\pi(r_2),r_3)
\]

\[
r_{123}(r_1,r_2,\pi(r_3)).
\]

A true three-modality interaction should respond meaningfully when any required modality is corrupted.

---

# 52. Third-Order Zero and Shuffle Tests

Evaluate:

```text
normal r123

zero r123

shuffle r123
```

and compare:

```text
Gain123

ZeroDrop123

ShuffleDrop123
```

Again:

```text
ShuffleDrop alone is insufficient evidence.
```

---

# 53. Higher-Order Residual

Eventually model:

\[
r_{123}^{R}
\]

as third-order information unexplained by lower-order representations.

Predict:

\[
\hat r_{123}
=
q(
r_1,r_2,r_3,
r_{12},r_{13},r_{23}
).
\]

Residual:

\[
r_{123}^{R}
=
r_{123}
-
stopgrad(\hat r_{123}).
\]

Evaluate conditional utility.

---

# 54. ConFu++ Conceptual Hierarchy

The long-term representation becomes:

```text
ORDER 1

r1
r2
r3


ORDER 2

r12
r13
r23


ORDER 3

r123
```

combined with representation scale:

```text
GLOBAL

LOCAL

RESIDUAL
```

Conceptually:

\[
\boxed{
Representation
=
InteractionOrder
\times
RepresentationScale
}
\]

---

# 55. Long-Term Structured Representation

Potential future representation:

\[
\mathcal Z
=
\{
r_S^G,
r_S^L,
r_S^R
\}
\]

for modality subsets:

\[
S.
\]

Examples:

\[
r_{12}^{G}
\]

\[
r_{12}^{L}
\]

\[
r_{12}^{R}
\]

\[
r_{123}^{G}
\]

\[
r_{123}^{L}
\]

\[
r_{123}^{R}.
\]

Do NOT implement the full combinatorial version initially.

---

# 56. Combinatorial Scalability

Original higher-order alignment scales poorly when considering all modality subsets.

The project must explicitly track:

```text
number of modalities

number of subset interactions

number of contrastive terms

number of fusion modules

parameter count

training memory

training time
```

---

# 57. Selective Higher-Order Fusion

Future scalability direction:

Instead of computing every possible:

\[
r_S,
\]

learn whether subset:

\[
S
\]

provides useful information.

Possible gate:

\[
g_S
=
\sigma(
G(r_S)
).
\]

But do NOT build the subset router before pairwise utility is solved.

---

# 58. Utility-Based Subset Selection

A stronger future formulation:

\[
\Delta_S
=
Perf(R_{\leq |S|})
-
Perf(R_{<|S|}).
\]

Train router to prioritize subsets with:

\[
\Delta_S>0.
\]

This may reduce the combinatorial cost of generalized ConFu.

---

# 59. Loss Hierarchy

Potential full objective:

\[
L
=
L_{task}
+
\lambda_{align}L_{align}
+
\lambda_{utility}L_{utility}
+
\lambda_{var}L_{var}
+
\lambda_{cov}L_{cov}
+
\lambda_{dep}L_{dependency}
+
\lambda_{pred}L_{residual}.
\]

Do NOT enable every term simultaneously.

---

# 60. Experimental Progression

Use the following order.

```text
P0
Current interaction diagnostics

P1
Conditional utility objective

P2
Modality-dependence diagnostics/objective

P3
Nonlinear redundancy analysis

P4
Residual interaction

P5
Global–local interaction

P6
Global-conditioned local correction

P7
Global–local residual representation

P8
True tri-modal conditional utility

P9
Third-order residual synergy

P10
Selective higher-order scalability
```

This order is mandatory unless experimental evidence justifies changing it.

---

# 61. Current AV-MNIST Baselines

Maintain the clean fair protocol for:

```text
Additive

Concat + MLP

ConFu-style

current low-rank SynergyFormer
```

Add sequentially:

```text
+ utility loss

+ dependence objective

+ residual interaction

+ global/local

+ global/local residual
```

---

# 62. Do Not Change Encoders During Core Ablations

During interaction-objective experiments, keep:

```text
same image encoder

same audio encoder

same embedding dimension

same optimizer

same batch size

same split

same seeds

same early stopping

same augmentation policy
```

Only change the component under investigation.

---

# 63. Five-Seed Protocol

Official experiments:

```text
5 seeds minimum
```

Report:

\[
mean\pm sample\ std.
\]

For every proposed model report paired differences against baseline.

---

# 64. Statistical Reporting

For paired comparisons report:

```text
per-seed difference

mean paired difference

sample std

paired t-test

effect size if appropriate
```

Do not claim superiority from:

```text
p > 0.05
```

or from one seed.

---

# 65. Main Performance Metrics

Report:

```text
Accuracy

Macro F1

Weighted F1
```

for classification.

For retrieval benchmarks report:

```text
Recall@1

Recall@5

Recall@10

Median Rank

Mean Rank
```

---

# 66. Mandatory Representation Metrics

Every interaction representation must log:

```text
norm

variance

effective rank

cosine with each lower-order representation

cross-covariance

predictability from each modality

predictability from lower-order representation
```

---

# 67. Conditional Utility Metrics

Mandatory:

\[
Gain_{12}
\]

\[
ZeroDrop_{12}
\]

\[
ShuffleDrop_{12}
\]

\[
Gain_{12}^{linear}
\]

\[
Gain_{12}^{nonlinear}
\]

\[
NetCorrection_{12}.
\]

For third order:

\[
Gain_{123}
\]

\[
ZeroDrop_{123}
\]

\[
ShuffleDrop_{123}
\]

\[
NetCorrection_{123}.
\]

---

# 68. Complementarity Acceptance Gate

An interaction representation should not be called complementary unless most of the following hold:

```text
positive conditional probe gain

positive zero-drop

positive net correction

multi-modality shuffle sensitivity

non-collapsed variance

healthy effective rank

positive mean gain across seeds
```

---

# 69. Strong Synergy Acceptance Gate

A stronger claim requires:

\[
Gain>0
\]

across most seeds and:

\[
ZeroDrop>0
\]

plus:

\[
NetCorrection>0
\]

and:

\[
ConditionalProbeGain>0.
\]

Permutation sensitivity alone is insufficient.

---

# 70. Synthetic Benchmark Upgrade

Keep XOR but stop treating simple XOR accuracy as sufficient evidence.

Generic MLP can also solve XOR.

Create synthetic decomposition tasks where ground-truth interaction order is explicitly known.

---

# 71. Synthetic Variables

Generate:

\[
a,b,c
\sim
Bernoulli(0.5).
\]

Define:

\[
S_1=a
\]

\[
S_2=b
\]

\[
S_3=c
\]

\[
S_{12}=a\oplus b
\]

\[
S_{13}=a\oplus c
\]

\[
S_{23}=b\oplus c
\]

\[
S_{123}=a\oplus b\oplus c.
\]

---

# 72. Representation Selectivity

Test:

\[
r_{12}\rightarrow S_{12}
\]

and compare against:

\[
r_1\rightarrow S_{12}
\]

\[
r_2\rightarrow S_{12}.
\]

Define:

\[
Selectivity_{12}
=
Acc(r_{12}\rightarrow S_{12})
-
\max(
Acc(r_1\rightarrow S_{12}),
Acc(r_2\rightarrow S_{12})
).
\]

---

# 73. Third-Order Selectivity

Define:

\[
Selectivity_{123}
=
Acc(r_{123}\rightarrow S_{123})
-
Acc(R_{lower}\rightarrow S_{123}).
\]

This is more informative than simple XOR classification.

---

# 74. Global–Local Synthetic Benchmark

Create data containing:

```text
global signal

local signal

redundant signal

synergistic local signal
```

Example structure:

\[
Y
=
f(G_1,G_2,L_1,L_2)
\]

where only:

\[
L_1\oplus L_2
\]

contains the additional correction signal.

This directly tests the Global–Local hypothesis.

---

# 75. Architecture Comparison

Required interaction operators:

```text
Concat MLP

ConFu MLP fusion

Low-rank multiplicative fusion

Bilinear interaction

Global–local multiplicative interaction
```

Cross-attention becomes a later baseline, not the immediate solution.

---

# 76. Cross-Attention Rule

Do NOT add cross-attention until:

```text
conditional utility objective has been evaluated

modality dependence has been measured

nonlinear redundancy has been measured

local representations have shown potential
```

Then compare:

```text
MLP fusion

multiplicative fusion

cross-attention fusion

cross-attention + multiplicative interaction
```

---

# 77. Why Cross-Attention Is Not the Current Fix

Cross-attention increases interaction capacity.

The current bottleneck is not clearly insufficient capacity.

The current bottleneck is:

\[
\boxed{
\text{useful conditional information}
}
\]

Therefore more capacity may simply create a stronger redundant predictor.

---

# 78. Effective Rank Rule

Do NOT use effective rank alone as success.

A representation can have:

```text
high variance

reasonable rank

strong task accuracy
```

and still be redundant.

Rank is a health diagnostic, not a complementarity metric.

---

# 79. Probe Rule

Do NOT use:

\[
Probe(r_{12})
\]

alone as evidence of synergy.

A high:

\[
Probe(r_{12})
\]

only proves:

\[
I(Y;r_{12})>0.
\]

The important comparison is:

\[
Probe([r_1,r_2,r_{12}])
\]

versus:

\[
Probe([r_1,r_2]).
\]

---

# 80. Shuffle Rule

Do NOT interpret:

\[
ShuffleDrop>0
\]

alone as interaction utility.

Always compare with:

\[
ZeroDrop.
\]

Desired behavior:

\[
ShuffleDrop>0
\]

and:

\[
ZeroDrop>0.
\]

---

# 81. Gate Rule

A scalar gate near:

\[
0.5
\]

is not automatically healthy or unhealthy.

Use gate sweep to determine whether the learned value corresponds to an actual performance optimum.

---

# 82. Future Sample-Dependent Gate

Only after global interaction has demonstrated conditional utility.

Possible:

\[
g_i
=
\sigma(
MLP(
r_1,r_2,r_{12}
)
).
\]

Purpose:

```text
increase interaction when useful

decrease interaction when harmful
```

Do not add this before establishing useful interaction signal.

---

# 83. Uncertainty-Conditioned Fusion

Future experiment:

\[
u
=
H(p_{base})
\]

where:

\[
H
\]

is predictive entropy.

Use:

\[
g
=
\sigma(
f(r_{12},u)
).
\]

Hypothesis:

> Interaction may be most useful when lower-order evidence is uncertain.

This should be tested after basic utility succeeds.

---

# 84. Hard-Sample Analysis

Partition samples by baseline confidence:

```text
high confidence

medium confidence

low confidence
```

Measure:

\[
Gain_{12}
\]

for each group.

This can reveal whether interaction helps only difficult samples.

---

# 85. Per-Class Complementarity

Measure:

\[
Correction
\]

and:

\[
Regression
\]

per class.

Some modalities may only be complementary for specific classes.

Do not average away this behavior.

---

# 86. Repository Strategy

Do not perform a large repository rewrite.

Extend the existing ConFu repository incrementally.

Suggested modules:

```text
src/modules/interactions/
    low_rank.py
    local_interaction.py
    residual_interaction.py

src/modules/losses/
    conditional_utility.py
    dependence.py
    residual_prediction.py

src/modules/pooling/
    local_slots.py

src/metrics/
    complementarity.py
    correction_regression.py
    conditional_probe.py

src/experiments/
    diagnostics/
```

---

# 87. Experiment Configuration

Every experiment must be configuration-driven.

Example:

```yaml
interaction:
  type: low_rank
  rank: 64
  normalize_factors: true
  output_bias: false

utility:
  enabled: true
  weight: 0.1
  margin: 0.0

dependency:
  enabled: false

residual:
  enabled: false

local:
  enabled: false
  num_slots: 4

diagnostics:
  modality_shuffle: true
  gate_sweep: true
  correction_regression: true
  nonlinear_probe: true
```

---

# 88. Experiment Naming

Use descriptive names.

Examples:

```text
avmnist_base

avmnist_utility

avmnist_dependency

avmnist_residual

avmnist_global_local

avmnist_global_local_residual

mosi_confu_base

mosi_confu_utility

mosi_global_local

mosi_higher_order_residual
```

---

# 89. Experiment Metadata

Every result must save:

```text
seed

git commit

working tree status

config

dataset

split

parameter count

best epoch

validation metric

test metrics

training duration
```

---

# 90. Result JSON

Minimum:

```json
{
  "seed": 1,
  "accuracy": 0.0,
  "macro_f1": 0.0,

  "gain12": 0.0,
  "zero_drop12": 0.0,
  "shuffle_drop12": 0.0,

  "shuffle_modality1_drop": 0.0,
  "shuffle_modality2_drop": 0.0,

  "correction": 0,
  "regression": 0,
  "net_correction": 0,

  "linear_probe_gain": 0.0,
  "nonlinear_probe_gain": 0.0,

  "r12_variance": 0.0,
  "r12_effective_rank": 0.0
}
```

---

# 91. Global–Local Result JSON

Additionally:

```json
{
  "global_accuracy": 0.0,
  "global_local_accuracy": 0.0,

  "local_gain": 0.0,
  "local_zero_drop": 0.0,
  "local_shuffle_drop": 0.0,

  "local_variance": 0.0,
  "local_effective_rank": 0.0,

  "residual_gain": 0.0,
  "residual_predictability_r2": 0.0
}
```

---

# 92. Unit Tests

Every new module must include:

```text
shape test

finite forward test

finite gradient test

batch-size-one test

gradient-flow test

shuffle test

zero test
```

Utility loss must test:

```text
loss is zero when full is sufficiently better

loss is positive when full is worse

stop-gradient is correct
```

---

# 93. Residual Test

Verify:

```text
predictor receives gradient

target interaction is detached for predictor target

interaction still receives task gradient

residual is finite

residual is not accidentally detached
```

---

# 94. Local Slot Tests

Verify:

\[
[B,T,D]
\rightarrow
[B,K,D].
\]

Test:

```text
variable T

K=1

K=4

batch size 1
```

---

# 95. Reproduction Before Modification

Before implementing a new research stage:

```text
reproduce the previous official result

verify metric calculation

verify checkpoints

verify seed
```

Do not build new conclusions on an unreproducible baseline.

---

# 96. Research Logging

Maintain:

```text
EXPERIMENTS.md
```

with:

```text
hypothesis

change

expected result

actual result

interpretation

next decision
```

Every experiment must answer a research question.

---

# 97. Decision Template

After each experiment write:

```text
Hypothesis:

Evidence for:

Evidence against:

Representation diagnostics:

Conditional utility:

Failure mode:

Decision:
```

Avoid:

```text
accuracy increased,
therefore method works.
```

---

# 98. Immediate Next Experiment

Do NOT implement Global–Local first.

Run the existing model with:

```text
shuffle modality 1 before interaction

shuffle modality 2 before interaction

gate sweep

correction/regression

nonlinear conditional probe
```

This determines whether:

\[
r_{12}
\]

actually depends on both modalities.

---

# 99. Immediate Objective Experiment

Then add:

\[
L_{utility}.
\]

Use:

```yaml
utility:
  weight: 0.1
  margin: 0.0
```

Run:

```text
seed 1
```

first.

---

# 100. Seed-1 Utility Acceptance

Require:

```text
Gain12 > 0

ZeroDrop > 0

NetCorrection > 0

interaction variance remains healthy

effective rank does not collapse
```

Also inspect:

```text
modality-1 shuffle drop

modality-2 shuffle drop
```

---

# 101. Five-Seed Utility Experiment

Only after seed 1 behaves sensibly.

Run:

```text
seeds 1–5
```

Report:

```text
mean Gain12

median Gain12

Gain12 std

ZeroDrop

NetCorrection

linear probe gain

nonlinear probe gain
```

---

# 102. Residual Experiment Trigger

Move to residual interaction only if:

```text
r12 remains predictive

r12 depends on both modalities

conditional gain remains weak

nonlinear redundancy remains likely
```

This is exactly the scenario residual representation is designed to address.

---

# 103. Global–Local Trigger

Move to Global–Local only if:

```text
global interaction is non-collapsed

objective is valid

conditional utility analysis is complete

global embedding may plausibly be discarding useful information
```

Do not use Global–Local to avoid understanding the current model.

---

# 104. Cross-Attention Trigger

Move to cross-attention only if local multiplicative interaction demonstrates positive conditional utility.

Then ask:

> Can more expressive local correspondence improve the existing useful local signal?

Not:

> Can more model capacity fix an unclear objective?

---

# 105. Tri-Modal Trigger

Return to genuine three-modality ConFu experiments after the pairwise complementarity mechanism is understood.

Then apply the same framework:

```text
dependency

conditional utility

zero/drop

shuffle

probe gain

residual decomposition
```

to:

\[
r_{12},r_{13},r_{23},r_{123}.
\]

---

# 106. Scientific Contributions Under Investigation

Potential contribution A:

**Conditional Utility Contrastive Fusion**

Interaction representations are explicitly optimized to improve upon lower-order representations.

Potential contribution B:

**Global–Local Contrastive Fusion**

Higher-order interaction is modeled at both global and sub-global scales.

Potential contribution C:

**Residual Synergy Representation**

The model explicitly represents cross-modal interaction unexplained by lower-order evidence.

Potential contribution D:

**Complementarity Diagnostics**

A systematic framework distinguishes:

```text
predictiveness

dependency

utility

complementarity

synergy
```

Potential contribution E:

**Selective Higher-Order Fusion**

Only useful modality subsets are retained as modality count increases.

Do NOT claim these are novel until a systematic literature review confirms novelty.

---

# 107. Main Paper-Level Claim Target

A valid future claim would look like:

> Standard higher-order alignment can learn predictive fused representations that remain redundant with lower-order modalities. Explicitly optimizing conditional utility produces interaction representations with measurable complementary information.

This claim requires strong experimental evidence.

---

# 108. Stronger Claim Target

Later:

> Global multimodal embeddings capture dominant shared semantics, while complementary multimodal information can reside in sub-global interactions. Global-conditioned residual interaction modeling improves higher-order multimodal representations.

This requires Global–Local experiments across more than one dataset.

---

# 109. What Not To Claim

Do NOT claim:

```text
r12 is synergy because its probe accuracy is high

r12 is synergy because shuffle hurts

r12 is independent because cosine is near zero

r12 is good because effective rank is high

higher-order interaction works because XOR reaches 100%
```

Each of these is insufficient alone.

---

# 110. Current Scientific State

Current status:

```text
Interaction collapse
SOLVED substantially

Interaction activity
PASS

Predictive interaction
PASS

Permutation sensitivity
PASS

Multi-modality dependence
NOT YET FULLY VERIFIED

Conditional linear complementarity
FAIL

Conditional nonlinear complementarity
NOT YET VERIFIED

Stable interaction utility
FAIL

Global/local complementarity
NOT YET TESTED

Third-order conditional synergy
NOT YET TESTED
```

---

# 111. Next Scientific State

The immediate target is:

```text
Multi-modality dependence
PASS

Conditional nonlinear complementarity
MEASURED

Utility objective
VALIDATED

Gain12
POSITIVE AND STABLE

ZeroDrop
POSITIVE

NetCorrection
POSITIVE
```

Only then advance to Global–Local.

---

# 112. Project Philosophy

The project is not trying to create the largest fusion architecture.

The project is trying to answer:

\[
\boxed{
\text{What information actually emerges from multimodal interaction?}
}
\]

and:

\[
\boxed{
\text{How can that information be separated from redundant modality evidence?}
}
\]

---

# 113. Final Representation Vision

Original multimodal fusion:

```text
X1
 \
  ---> Fusion ---> z
 /
X2
```

ConFu:

```text
X1 ---> z1
          \
           ---> z12 <----> z3
          /
X2 ---> z2
```

ConFu++ target:

```text
X1 ---> global representation
   \
    \---- local representations
             \
              ---> interaction
                     |
                     +--> predictive component
                     |
                     +--> redundant component
                     |
                     +--> complementary residual
                                |
                                v
                           correction / synergy
```

For three modalities:

```text
Order 1
individual evidence

Order 2
pairwise interaction

Order 3
higher-order interaction

Within each order:

global
local
residual complementary information
```

---

# 114. One-Sentence Research Thesis

> ConFu++ learns not only whether multimodal representations can be fused, but which information is uniquely useful because the modalities were fused.

---

# 115. Immediate Agent Instruction

The next agent execution must perform, in order:

```text
1. Reproduce current 5-seed benchmark.

2. Implement single-modality interaction shuffle diagnostics.

3. Implement gate sweep.

4. Implement correction/regression analysis.

5. Implement parameter-matched nonlinear conditional probes.

6. Analyze whether r12 depends on both modalities.

7. Add conditional utility loss.

8. Run seed 1.

9. Inspect Gain12, ZeroDrop, NetCorrection,
   modality-shuffle dependence,
   variance and effective rank.

10. If healthy, run five seeds.

11. Only after this decision point,
    evaluate residual interaction.

12. Only after residual experiments,
    begin Global–Local ConFu.

13. Only after pairwise complementarity is understood,
    return to genuine tri-modal higher-order ConFu.
```

Do not skip directly to a larger architecture.

---

# 116. Final Rule

Whenever the agent must choose between:

```text
increasing model complexity
```

and:

```text
obtaining stronger evidence that the learned interaction
contains conditional complementary information
```

choose the second.

The central objective of this project is:

\[
\boxed{
\text{Predictive interaction}
\neq
\text{Synergy}
}
\]

and the entire research program must determine how to transform the former into the latter.