# AGENT.md

# ConFu++

## Complementarity-Aware Higher-Order Multimodal Representation Learning

---

# 1. Project Scope

This repository directly extends the research direction introduced by:

**The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment (ConFu).**

The project investigates a fundamental limitation of higher-order multimodal alignment:

> A representation can be computed jointly from multiple modalities and satisfy a higher-order alignment objective without actually depending on all modalities or containing complementary information.

The project must remain focused on:

- multimodal representation learning,
- multimodal contrastive learning,
- higher-order multimodal alignment,
- interaction representations,
- modality dependence,
- modality complementarity,
- synergy,
- conditional utility,
- redundancy,
- higher-order information,
- representation diagnostics,
- multimodal benchmark design.

Do NOT redirect the project toward robotics, egocentric vision, IMU, pose estimation, SLAM, VLA, or unrelated application domains.

---

# 2. Central Scientific Problem

Given modalities:

\[
X_1,X_2,\ldots,X_M,
\]

ConFu learns unimodal representations:

\[
r_i = E_i(X_i)
\]

and fused representations such as:

\[
r_{12}=F(r_1,r_2).
\]

The original higher-order objective encourages:

\[
r_{12}\leftrightarrow r_3.
\]

However, nothing fundamentally guarantees that:

\[
r_{12}
\]

actually requires both:

\[
r_1
\]

and:

\[
r_2.
\]

A valid shortcut may be:

\[
r_{12}\approx f(r_1).
\]

This representation can still:

- have high variance,
- have non-trivial effective rank,
- predict the target,
- satisfy contrastive alignment,
- affect the classifier,

while containing almost no unique information from modality 2.

Therefore:

\[
\boxed{
\text{joint computation}
\neq
\text{joint information}
}
\]

and:

\[
\boxed{
\text{higher-order alignment}
\neq
\text{higher-order synergy}.
}
\]

This distinction is the central motivation of ConFu++.

---

# 3. Main Research Hypothesis

ConFu++ investigates the hypothesis:

> Higher-order multimodal representation learning should explicitly optimize not only alignment and predictiveness, but also multi-modality dependence and conditional complementarity.

A valid interaction representation should ideally satisfy:

\[
\boxed{
\text{Activity}
+
\text{Joint Dependence}
+
\text{Conditional Utility}
}
\]

before being described as synergistic.

---

# 4. Terminology

Use the following terminology consistently.

## 4.1 Fused representation

Any representation computed from multiple modality representations:

\[
r_{12}=F(r_1,r_2).
\]

This term makes no claim about complementarity.

---

## 4.2 Interaction representation

A fused representation intended to model relationships between modalities.

---

## 4.3 Active interaction

A representation with non-trivial variation:

\[
Var(r_{12})>0.
\]

---

## 4.4 Predictive interaction

An interaction representation for which:

\[
I(Y;r_{12})>0
\]

operationally approximated by successful downstream probing.

---

## 4.5 Multi-modality dependence

The interaction representation genuinely depends on every required modality.

Operationally:

\[
F(r_1,r_2)
\]

must change meaningfully when either:

\[
r_1
\]

or:

\[
r_2
\]

is corrupted.

---

## 4.6 Conditional utility

The interaction representation improves predictions after accounting for lower-order information:

\[
I(Y;r_{12}\mid r_1,r_2)>0.
\]

Operational approximation:

\[
Perf(r_1,r_2,r_{12})
>
Perf(r_1,r_2).
\]

---

## 4.7 Complementary interaction

An interaction satisfying both:

```text
multi-modality dependence
+
conditional utility
```

---

## 4.8 Synergy

Reserve **synergy** for interaction information with convincing evidence of complementary utility.

Do NOT call every fused representation synergistic.

---

# 5. Current Experimental State

The current AV-MNIST experiments have already established an important negative result.

The interaction representation:

\[
r_{12}
\]

is:

```text
non-collapsed
predictive
permutation-sensitive
conditionally useful to a small degree
```

but is almost entirely predictable from the image representation.

Measured nonlinear predictability:

```text
image r1 -> r12:
R² ≈ 0.988
cosine ≈ 0.996

audio r2 -> r12:
R² ≈ 0.031
cosine ≈ 0.203

(r1,r2) -> r12:
R² ≈ 0.996
cosine ≈ 0.999
```

Therefore approximately:

\[
\boxed{
r_{12}\approx f(r_1)
}
\]

despite being produced by a multimodal interaction module.

This is the most important empirical finding so far.

---

# 6. Current AV-MNIST Baselines

Official clean five-seed protocol:

```text
Dataset:
MultiBench AV-MNIST

Seeds:
5

Encoders:
ResNet18 image
ResNet18 audio

Representation:
D = 256

Interaction rank:
R = 64

Optimizer:
AdamW

Learning rate:
1e-4

Batch size:
512

Early stopping:
validation accuracy
patience 4

Augmentation:
none
```

Results:

```text
Additive:
70.448 ± 0.522

Concat + MLP:
70.518 ± 0.453

ConFu-style:
70.886 ± 0.578

Previous SynergyFormer:
70.762 ± 0.338

P1 Conditional Utility:
70.296 ± 0.545
```

Do NOT claim superiority on AV-MNIST.

---

# 7. AV-MNIST P1 Conditional Utility Result

Utility objective:

\[
c_{base}
=
LN(
LN(r_1)+LN(r_2)
)
\]

\[
c_{full}
=
LN(
c_{base}
+
\sigma(a) LN(r_{12})
)
\]

and:

\[
L_{utility}
=
\operatorname{ReLU}
\left(
CE_{full}
-
stopgrad(CE_{base})
+
m
\right).
\]

Canonical exploratory configuration:

```yaml
lambda_utility: 0.2
utility_margin: 0.0
```

Five-seed result:

```text
Gain12:
+0.234 ± 0.263 pp
5/5 positive

Zero drop:
+0.262 ± 0.309 pp
4/5 positive

Interaction shuffle drop:
+1.346 ± 0.785 pp
5/5 positive

Image shuffle drop:
+1.482 ± 0.847 pp

Audio shuffle drop:
+0.004 ± 0.027 pp

Net correction:
+23.4 ± 26.3
5/5 positive

Linear conditional gain:
+0.126 ± 0.186 pp

Nonlinear conditional gain:
+0.294 ± 0.162 pp
5/5 positive
```

Interpretation:

```text
conditional utility:
measurable

genuine audio dependence:
absent

balanced bimodal complementarity:
not demonstrated
```

---

# 8. AV-MNIST P2 Dependence Result

A ranking objective attempted to force the true paired interaction to outperform interactions with one shuffled modality.

Result:

```text
lambda_dependency 0.1:
audio-shuffle drop ≈ -0.02 pp

lambda_dependency 0.5:
audio-shuffle drop ≈ +0.11 pp

lambda_dependency 1.0:
audio-shuffle drop ≈ +0.11 pp
```

The stronger objective reduced useful interaction performance without creating robust modality dependence.

Decision:

```text
P2 failed.
```

Do NOT expand this objective to five seeds unless redesigned.

---

# 9. Core Interpretation of AV-MNIST

AV-MNIST should now be treated primarily as:

\[
\boxed{
\text{a modality-shortcut / negative-control benchmark}
}
\]

rather than the primary benchmark for proving multimodal synergy.

The current experiment demonstrates:

> A fused representation can be active, predictive, high-rank, and permutation-sensitive while still being almost entirely derived from one modality.

This result is scientifically useful.

Do NOT attempt to erase it through excessive architecture tuning.

---

# 10. Central Research Question Going Forward

The project now asks:

> Under what data and objective conditions does higher-order multimodal alignment actually produce higher-order dependence and complementary information?

Secondary question:

> How can multimodal interaction representations be trained so that each required modality contributes information that cannot be replaced by the others?

---

# 11. Three Required Properties

For pair interaction:

\[
r_{12}=F(r_1,r_2),
\]

evaluate three properties separately.

---

## 11.1 Activity

Require:

\[
Var(r_{12})>0.
\]

Diagnostics:

```text
variance
norm
effective rank
dimension-wise standard deviation
```

Activity alone is weak evidence.

---

## 11.2 Multi-Modality Dependence

Define:

\[
A=A_{normal}.
\]

Shuffle modality 1 before interaction:

\[
A_{-1}.
\]

Shuffle modality 2:

\[
A_{-2}.
\]

Define:

\[
D_1=A-A_{-1}
\]

\[
D_2=A-A_{-2}.
\]

A genuine pair interaction should have:

\[
D_1>0
\]

and:

\[
D_2>0.
\]

---

## 11.3 Conditional Utility

Lower-order representation:

\[
R_{low}=[r_1,r_2].
\]

Full representation:

\[
R_{full}=[r_1,r_2,r_{12}].
\]

Require:

\[
Perf(R_{full})
>
Perf(R_{low}).
\]

Use:

```text
task accuracy

linear probe

nonlinear probe

zero interaction test

correction/regression
```

---

# 12. Interaction Health Is Not Synergy

Never infer synergy from:

```text
high variance

high rank

low cosine similarity

high standalone probe accuracy

interaction shuffle drop
```

These are diagnostics only.

---

# 13. Mandatory Dependency Diagnostic

For every learned interaction:

\[
r_S,
\]

where:

\[
S
\]

is a modality subset, evaluate leave-one-modality corruption.

For:

\[
r_{123},
\]

run:

\[
r_{123}(\pi(r_1),r_2,r_3)
\]

\[
r_{123}(r_1,\pi(r_2),r_3)
\]

\[
r_{123}(r_1,r_2,\pi(r_3)).
\]

Every required modality should create measurable degradation.

---

# 14. Dependency Balance Metric

For pair interaction define:

\[
D_1
\]

and:

\[
D_2.
\]

Define balanced dependence:

\[
B_{12}
=
\frac{
2\min(D_1,D_2)
}{
D_1+D_2+\epsilon
}.
\]

Properties:

\[
0\le B_{12}\le1.
\]

Interpretation:

```text
B ≈ 0
one modality dominates

B ≈ 1
both modalities contribute similarly
```

Do NOT interpret this as formal information-theoretic synergy.

It is an empirical diagnostic.

---

# 15. Conditional Utility Metric

Define:

\[
U_{12}
=
Perf(R_{full})
-
Perf(R_{low}).
\]

Possible forms:

```text
task Gain12

linear probe gain

nonlinear probe gain
```

---

# 16. Empirical Complementarity Score

Optional diagnostic:

\[
C_{12}
=
B_{12}
\cdot
\max(0,U_{12}).
\]

Purpose:

```text
penalize high utility
that comes from one-modality shortcutting.
```

This is an exploratory metric.

Do NOT present it as an information-theoretic quantity.

---

# 17. Zero vs Shuffle Distinction

Always report both.

## Zero interaction

\[
r_{12}=0.
\]

Measures:

> Is the interaction necessary?

---

## Shuffled interaction

\[
r_{12}^{(i)}
\rightarrow
r_{12}^{(\pi(i))}.
\]

Measures:

> Is the classifier sensitive to mismatched interaction signals?

A positive shuffle drop without a positive zero drop is insufficient evidence of utility.

---

# 18. Correction–Regression Analysis

For lower-order prediction:

\[
\hat y_{low}
\]

and full prediction:

\[
\hat y_{full},
\]

count:

```text
lower wrong -> full correct
= correction

lower correct -> full wrong
= regression

both correct

both wrong
```

Define:

\[
NetCorrection
=
Correction-Regression.
\]

Report:

```text
overall
per-class
per-seed
```

---

# 19. Conditional Linear Probe

Freeze representations.

Train equal-protocol linear probes:

\[
Probe(R_{low})
\]

and:

\[
Probe(R_{full}).
\]

Report:

\[
Gain^{linear}.
\]

---

# 20. Conditional Nonlinear Probe

Train identical parameter-matched MLP probes:

\[
MLP(R_{low})
\]

versus:

\[
MLP(R_{full}).
\]

Report:

\[
Gain^{nonlinear}.
\]

This is mandatory because interaction redundancy may be nonlinear.

---

# 21. Predictability Probe

For each interaction:

\[
r_{12},
\]

train:

\[
q_1(r_1)\rightarrow r_{12}
\]

\[
q_2(r_2)\rightarrow r_{12}
\]

\[
q_{12}(r_1,r_2)\rightarrow r_{12}.
\]

Report:

```text
R²
MSE
cosine
```

This diagnostic is mandatory.

---

# 22. Shortcut Criterion

If:

\[
R^2(q_1)\gg R^2(q_2)
\]

and:

\[
R^2(q_1)\approx R^2(q_{12}),
\]

then:

\[
r_{12}
\]

is effectively modality-1 dominated.

Do NOT proceed to residual interaction merely because:

\[
r_{12}
\]

is predictive.

---

# 23. Next Main Experimental Stage

Do NOT continue architecture expansion on AV-MNIST.

The next main experiment must use a benchmark where multiple modalities have genuine potential to provide complementary evidence.

Priority benchmark family:

```text
CMU-MOSI

MUStARD

UR-FUNNY

Bird-MML
```

Prefer datasets used directly or closely related to the original ConFu evaluation.

---

# 24. Benchmark Selection Criteria

A dataset should preferably have:

```text
multiple real observed modalities

no modality generated directly from the target label

no artificial label leakage

identical train/val/test protocol available

compatible ConFu baseline

meaningful multi-modal complementarity
```

Before training, document known modality imbalance.

---

# 25. Dataset Audit Before Experiment

For each candidate dataset, first train unimodal baselines.

For three modalities:

\[
X_1,X_2,X_3,
\]

measure:

```text
Perf(X1)

Perf(X2)

Perf(X3)

Perf(X1,X2)

Perf(X1,X3)

Perf(X2,X3)

Perf(X1,X2,X3)
```

This establishes whether the dataset can actually test multimodal complementarity.

---

# 26. Modality Necessity Audit

For a trained full model evaluate:

```text
full modalities

drop modality 1

drop modality 2

drop modality 3
```

Do this before introducing ConFu++ objectives.

If full performance is unchanged after dropping a modality, that modality may not be useful for the current task/protocol.

---

# 27. ConFu Reproduction Stage

For every new benchmark:

1. reproduce the original ConFu-style baseline,
2. reproduce unimodal baselines,
3. reproduce pairwise fusion,
4. reproduce higher-order fusion,
5. verify results,
6. only then add ConFu++ objectives.

Do not optimize ConFu++ before the baseline is reproducible.

---

# 28. Stage C — Conditional Utility on Genuine Multimodal Data

For pair interaction:

\[
r_{ij},
\]

define lower-order representation:

\[
R^{ij}_{low}=[r_i,r_j].
\]

Full:

\[
R^{ij}_{full}
=
[r_i,r_j,r_{ij}].
\]

Compute:

\[
L^{ij}_{utility}
=
\operatorname{ReLU}
(
CE_{full}
-
stopgrad(CE_{low})
+
m
).
\]

---

# 29. Utility Objective for Retrieval

For retrieval tasks where CE is not appropriate, use task loss:

\[
L_{task}^{low}
\]

and:

\[
L_{task}^{full}.
\]

Then:

\[
L_{utility}
=
\operatorname{ReLU}
(
L_{task}^{full}
-
stopgrad(L_{task}^{low})
+
m
).
\]

Keep implementation task-agnostic where practical.

---

# 30. Higher-Order Utility

For three modalities define:

\[
R_{lower}
=
[
r_1,r_2,r_3,
r_{12},r_{13},r_{23}
].
\]

Full:

\[
R_{full}
=
[
R_{lower},
r_{123}
].
\]

Third-order utility:

\[
L_{utility}^{123}
=
\operatorname{ReLU}
(
L_{full}
-
stopgrad(L_{lower})
+
m
).
\]

This directly tests whether:

\[
r_{123}
\]

adds information beyond all lower-order terms.

---

# 31. Stage D — Modality Necessity Objective

If the dataset contains genuine multimodal information but learned interactions still shortcut, investigate explicit modality-necessity objectives.

The objective should encourage:

```text
full multimodal prediction
to require all relevant modalities
```

without intentionally destroying useful unimodal representations.

---

# 32. Do Not Use Naive Anti-Unimodal Loss

Avoid directly maximizing unimodal CE such as:

\[
-CE(p(y|X_i),y).
\]

This may intentionally degrade useful unimodal representations.

The goal is not:

```text
make unimodal features bad
```

but:

```text
make fused interaction use information unavailable from only one modality.
```

---

# 33. Conditional Necessity Formulation

Possible strategy:

Compare full prediction:

\[
p_F
\]

with leave-one-modality-out predictions:

\[
p_{-i}.
\]

For samples where modality \(i\) contains useful evidence, encourage:

\[
p_F(y)
>
p_{-i}(y).
\]

Use controlled margins.

Do NOT assume every modality must be necessary for every sample.

---

# 34. Sample-Selective Necessity

Eventually define:

\[
n_i(x)
\]

representing estimated necessity of modality \(i\) for sample \(x\).

This may depend on:

```text
confidence change

entropy change

representation disagreement

validation evidence
```

Do not implement this before fixed necessity objectives are understood.

---

# 35. Stage E — Residual Interaction

Residual interaction is NOT currently justified on AV-MNIST because pair interaction does not depend meaningfully on audio.

Residual interaction becomes valid when:

```text
interaction depends on all required modalities

interaction remains redundant with lower-order evidence

conditional utility is weak
```

Then predict:

\[
\hat r_{12}
=
q(r_1,r_2).
\]

Residual:

\[
r_{12}^R
=
r_{12}
-
stopgrad(\hat r_{12}).
\]

---

# 36. Residual Interpretation

Do NOT claim:

\[
r_{12}^R
\]

is exact unique information.

It is an operational residual representation.

The research question is:

> Is the unpredictable part of the interaction more complementary than the raw interaction?

---

# 37. Residual Evaluation

Compare:

\[
R_{low}
\]

\[
[R_{low},r_{12}]
\]

\[
[R_{low},r_{12}^R].
\]

Report:

```text
raw gain

residual gain

zero drop

shuffle drop

conditional probe gain

net correction
```

---

# 38. Stage F — Global–Local Fusion

Only investigate Global–Local representations after:

```text
genuine multimodal dependence exists

conditional utility framework works

shortcut behavior is understood
```

Question:

> Does global pooling discard complementary cross-modal information?

---

# 39. Global Representation

For encoder tokens:

\[
H_i\in\mathbb R^{B\times T_i\times D},
\]

global representation:

\[
r_i^G=Pool(H_i).
\]

Global interaction:

\[
r_{ij}^{G}
=
F_G(r_i^G,r_j^G).
\]

---

# 40. Local Representation

Keep token-level or sub-global representations:

\[
H_i.
\]

Compress to:

\[
K
\]

local slots.

Start:

\[
K=4.
\]

Do not immediately use large token-token cross-attention.

---

# 41. Local Multiplicative Interaction

For local slots:

\[
p_{i,k}
\]

and:

\[
p_{j,k},
\]

compute:

\[
s_k
=
LN(W_ip_{i,k})
\odot
LN(W_jp_{j,k}).
\]

Then:

\[
r_{ij}^{L}
=
Pool(s_1,\ldots,s_K).
\]

---

# 42. Global–Local Comparison

Compare:

```text
global only

global interaction

global + local interaction

global + local residual
```

Do not combine all into a Transformer immediately.

---

# 43. Local Correction Objective

Preferred first local formulation:

\[
l_F
=
l_G
+
g_L\Delta l_L.
\]

Where:

\[
\Delta l_L
=
W_Lr_{ij}^{L}.
\]

This tests:

> Can local interaction correct global fusion errors?

---

# 44. Global–Local Utility

Define:

\[
L_G
\]

from global prediction.

Define:

\[
L_F
\]

after local correction.

Then:

\[
L_{GL}
=
\operatorname{ReLU}
(
L_F
-
stopgrad(L_G)
+
m
).
\]

---

# 45. Do Not Add Cross-Attention First

Cross-attention increases interaction capacity.

It does not guarantee complementarity.

Only test cross-attention after simple local interactions produce measurable conditional utility.

---

# 46. Stage G — Third-Order Synergy

Once pairwise ConFu++ works, return to:

\[
r_{123}.
\]

Third-order interaction must satisfy:

```text
non-collapse

dependence on X1

dependence on X2

dependence on X3

conditional utility beyond all lower-order representations
```

---

# 47. Third-Order Dependency Balance

Let:

\[
D_1,D_2,D_3
\]

be performance drops after shuffling each modality before computing:

\[
r_{123}.
\]

Possible balance diagnostic:

\[
B_{123}
=
\frac{
3\min(D_1,D_2,D_3)
}{
D_1+D_2+D_3+\epsilon
}.
\]

Use only as an empirical diagnostic.

---

# 48. Third-Order Conditional Gain

Define:

\[
Gain_{123}
=
Perf(
r_1,r_2,r_3,
r_{12},r_{13},r_{23},
r_{123}
)
-
Perf(
r_1,r_2,r_3,
r_{12},r_{13},r_{23}
).
\]

This is the central third-order utility metric.

---

# 49. Third-Order Probe

Train:

\[
Probe(R_{lower})
\]

and:

\[
Probe(R_{lower},r_{123}).
\]

Run:

```text
linear probe

parameter-matched nonlinear probe
```

---

# 50. Synthetic Benchmark

Keep XOR as a sanity check only.

Simple XOR does NOT prove superiority because generic MLPs can solve it.

Develop controlled synthetic datasets with known information structure.

---

# 51. Pairwise Synthetic Synergy

Generate:

\[
a,b\sim Bernoulli(0.5).
\]

Modalities contain individual signals.

Target includes:

\[
a\oplus b.
\]

Measure whether:

\[
r_{12}
\]

selectively represents the interaction.

---

# 52. Third-Order Synthetic Synergy

Generate:

\[
a,b,c\sim Bernoulli(0.5).
\]

Define:

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

# 53. Representation Selectivity

For pair:

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

For third order:

\[
Selectivity_{123}
=
Acc(r_{123}\rightarrow S_{123})
-
Acc(R_{lower}\rightarrow S_{123}).
\]

---

# 54. Synthetic Shortcut Benchmark

Create datasets where one modality strongly predicts the label but the second modality provides rare corrections.

Example:

```text
90% samples:
modality 1 sufficient

10% samples:
modality 2 required
```

Evaluate whether ConFu learns:

```text
dominant shortcut
```

or:

```text
conditional correction.
```

This benchmark directly tests the observed AV-MNIST failure mode.

---

# 55. Complementarity-Controlled Dataset

Build synthetic tasks with tunable:

```text
redundancy level

unique information

synergy level

modality imbalance

noise level
```

This allows causal testing of the method.

---

# 56. Core Baselines

Maintain:

```text
unimodal

additive

concat MLP

ConFu MLP

low-rank multiplicative

ConFu + utility

ConFu + necessity

ConFu + utility + necessity
```

Later:

```text
residual

global-local

cross-attention
```

---

# 57. Capacity-Matched Comparison

Any new module that adds parameters must include an approximately parameter-matched baseline.

Do not attribute gains to interaction design when they may come from additional capacity.

---

# 58. Encoders Must Be Frozen Across Core Ablations

During objective studies keep:

```text
same encoder type

same initialization strategy

same representation dimension

same optimizer

same scheduler

same splits

same random seeds

same augmentation
```

Change only the intended component.

---

# 59. Statistical Protocol

Official benchmark results require at least:

```text
5 seeds
```

Report:

\[
mean\pm sample\ standard\ deviation.
\]

---

# 60. Paired Statistical Comparison

For proposed vs baseline compute:

\[
\Delta_s
=
Metric_{proposed,s}
-
Metric_{baseline,s}.
\]

Report:

```text
mean paired difference

sample std

paired t-test

individual seed results
```

Do not overstate conclusions from five seeds.

---

# 61. Hyperparameter Selection Rule

Do NOT select hyperparameters using test performance for confirmatory experiments.

The previous AV-MNIST P1 sweep used seed-1 test diagnostics and must remain labeled exploratory.

Going forward use:

```text
training set

validation set
```

for hyperparameter selection.

Test set is evaluated only after configuration is fixed.

---

# 62. Experiment Tiers

Use:

## Exploratory

Hyperparameters may still be changing.

Results cannot support final claims.

## Confirmatory

Configuration frozen before test evaluation.

Used for paper tables and statistical claims.

---

# 63. Interaction Diagnostics

Every interaction representation must log:

```text
norm

variance

effective rank

dimension-wise std

cosine with lower-order representations

cross-covariance

R² predictability from each modality

R² predictability from all lower-order representations
```

---

# 64. Effective Rank

Effective rank is a collapse diagnostic only.

Never claim:

```text
higher rank = better synergy
```

The AV-MNIST results already show that increased rank does not necessarily improve conditional utility.

---

# 65. Gradient Diagnostics

Log:

\[
\left\|
\frac{\partial L}{\partial r_i}
\right\|
\]

for individual modalities and interaction representations.

Purpose:

```text
detect gradient starvation

detect modality domination

detect interaction domination
```

---

# 66. Contribution Diagnostics

Measure:

```text
interaction zero drop

interaction shuffle drop

modality-specific shuffle drop

conditional probe gain

net correction
```

These are more important than raw interaction norm.

---

# 67. Hard-Sample Analysis

Bin samples by lower-order prediction confidence:

```text
high confidence

medium confidence

low confidence
```

Measure:

\[
Gain
\]

for each bin.

Hypothesis:

> complementary interaction may primarily help uncertain lower-order predictions.

---

# 68. Per-Class Analysis

Report:

```text
corrections

regressions

Gain

modality dependence
```

per class when applicable.

Interaction utility may be highly class-specific.

---

# 69. Missing-Modality Analysis

On genuine multimodal datasets test:

```text
all modalities

missing modality 1

missing modality 2

missing modality 3
```

Do not infer multimodal necessity from full-input accuracy alone.

---

# 70. Noise Analysis

Later evaluate controlled corruption:

```text
modality-specific Gaussian noise

feature masking

token dropout

temporal masking

semantic corruption where appropriate
```

Measure whether interaction appropriately shifts dependence.

---

# 71. Representation API

Models should expose representations explicitly.

Example:

```python
{
    "r1": ...,
    "r2": ...,
    "r3": ...,

    "r12": ...,
    "r13": ...,
    "r23": ...,

    "r123": ...,

    "lower_order": ...,
    "full": ...,

    "logits_lower": ...,
    "logits_full": ...,
}
```

Do not hide research representations inside opaque modules.

---

# 72. Diagnostic API

Provide utilities for:

```text
shuffle modality before interaction

shuffle interaction representation

zero interaction

force interaction gate

drop modality

extract representations

run probes
```

These are first-class research features.

---

# 73. Pair Interaction API

```python
class InteractionModule(nn.Module):

    def forward(
        self,
        r1: torch.Tensor,
        r2: torch.Tensor,
    ) -> torch.Tensor:
        """
        r1: [B, D]
        r2: [B, D]

        returns:
            r12: [B, D]
        """
```

---

# 74. Low-Rank Multiplicative Baseline

Keep current validated implementation:

\[
u_1=LN(U_1r_1)
\]

\[
u_2=LN(U_2r_2)
\]

\[
h=
\frac{
u_1\odot u_2
}{
\sqrt R
}
\]

\[
r_{12}=W_oh.
\]

Use:

```text
factor bias false
output bias false
```

unless an ablation explicitly changes them.

---

# 75. Anti-Collapse Losses

Maintain:

```text
variance floor

cross-covariance

off-diagonal self-covariance
```

as needed.

Do not assume these produce complementarity.

Their role is representation health.

---

# 76. Current Canonical Anti-Collapse Setup

Current AV-MNIST formulation:

\[
L
=
L_{task}
+
0.2L_{unimodal}
+
1.0L_{variance}
+
0.01L_{crosscov}
+
0.5L_{selfcov}.
\]

When transferring to another dataset, revalidate scales before reusing exact coefficients.

---

# 77. Loss Logging

Always log raw and weighted components separately:

```text
task_raw

utility_raw

dependency_raw

variance_raw

cross_cov_raw

self_cov_raw

weighted_task

weighted_utility

weighted_dependency

...
```

Never allow a regularizer to become effectively inactive without detecting it.

---

# 78. Gradient Alignment

Optional diagnostic:

Compute cosine similarity between gradients of:

```text
main task loss

utility loss

dependency loss

anti-collapse losses
```

This may reveal objective conflict.

Do not use as the main evaluation metric.

---

# 79. Global–Local Trigger

Do not implement global/local interaction merely because it is architecturally interesting.

Trigger only if:

```text
dataset has genuine multimodal dependence

global interaction uses both modalities

conditional utility is measurable

remaining errors plausibly require finer-grained features
```

---

# 80. Residual Trigger

Use residual representation only if:

```text
interaction depends on required modalities

interaction is predictive

interaction is still highly predictable from lower-order representations

conditional utility remains limited
```

---

# 81. Cross-Attention Trigger

Use cross-attention only if:

```text
simpler interactions already demonstrate genuine complementarity
```

Then ask:

> Does more expressive interaction improve complementary information?

Not:

> Can a larger network hide the shortcut?

---

# 82. Third-Order Trigger

Do not proceed to third-order claims before pairwise methodology is understood.

When ready, evaluate:

```text
r12
r13
r23
```

individually before:

\[
r_{123}.
\]

---

# 83. Higher-Order Scalability

For:

\[
M
\]

modalities, the number of subsets grows combinatorially.

Track:

```text
number of interaction modules

number of active subset losses

parameters

GPU memory

training time
```

---

# 84. Selective Higher-Order Fusion

Future direction:

Instead of computing all subset interactions, estimate utility:

\[
U_S
\]

for subset:

\[
S.
\]

Only activate interactions with significant incremental value.

Do NOT implement until pair/tri-modal studies are complete.

---

# 85. Subset Utility

Conceptually:

\[
U_S
=
Perf(R_{\le |S|})
-
Perf(R_{<|S|}).
\]

This may eventually guide subset selection.

---

# 86. Research Questions for Genuine Tri-Modal Benchmarks

For each dataset answer:

1. Which modality is strongest?
2. Which pair provides the highest improvement?
3. Does ConFu fused representation depend on both modalities?
4. Which fused representations are shortcut-dominated?
5. Does \(r_{123}\) depend on all three modalities?
6. Does \(r_{123}\) provide conditional utility?
7. Does utility loss improve complementarity?
8. Does modality necessity improve dependence?
9. Are gains stable across seeds?
10. Does improved complementarity improve final task performance?

---

# 87. AV-MNIST Role Going Forward

Do NOT abandon AV-MNIST.

Keep it as:

```text
negative control

shortcut benchmark

regression test
```

A good future ConFu++ objective should not falsely claim audio dependence when the dataset does not provide meaningful conditional need.

---

# 88. Benchmark Diversity

Final publication-quality evidence should include at least:

```text
one shortcut-prone benchmark

one genuine tri-modal affective benchmark

one benchmark with strong higher-order complementarity

one synthetic controlled benchmark
```

Do not rely on only one dataset family.

---

# 89. Research Log

Maintain:

```text
EXPERIMENTS.md
```

For every experiment record:

```text
Hypothesis

Motivation

Change

Unchanged controls

Expected result

Actual result

Representation diagnostics

Dependency result

Conditional utility result

Failure mode

Decision
```

---

# 90. Negative Results

Never hide negative results.

Preserve:

```text
AV-MNIST modality dominance

failed P2 dependence objective

rank increases without utility gains

non-significant accuracy comparisons
```

These findings shape the scientific story.

---

# 91. Decision Discipline

Do not say:

```text
accuracy increased,
therefore hypothesis confirmed.
```

Instead ask:

```text
Did dependence improve?

Did conditional utility improve?

Did representation remain healthy?

Did final task performance improve?

Was the result stable?

Was capacity controlled?
```

---

# 92. Main Failure Modes

Track explicitly:

```text
representation collapse

rank collapse

one-modality shortcut

linear redundancy

nonlinear redundancy

interaction ignored

interaction harmful

objective conflict

overfitting

dataset modality bias

label leakage

capacity confound
```

---

# 93. Main Scientific Claim Target

The desired future paper-level claim is:

> Higher-order multimodal alignment can learn predictive fused representations without learning genuine higher-order dependence. ConFu++ explicitly measures and optimizes modality dependence and conditional utility, producing representations with stronger empirical complementarity.

Do not claim this until genuine multimodal benchmarks support it.

---

# 94. Stronger Claim Target

If Global–Local eventually works:

> Complementary multimodal information may exist at finer representational scales than global fusion captures. Conditional global-local interaction modeling improves higher-order multimodal complementarity.

Requires multiple datasets.

---

# 95. Potential Contributions

Potential contribution 1:

**Higher-order alignment vs higher-order dependence analysis**

Potential contribution 2:

**Conditional Utility Objective**

Potential contribution 3:

**Modality Necessity Objective**

Potential contribution 4:

**Empirical Complementarity Diagnostics**

Potential contribution 5:

**Residual Interaction Representation**

Potential contribution 6:

**Global–Local Higher-Order Fusion**

Potential contribution 7:

**Selective Higher-Order Fusion**

These are hypotheses, not established contributions.

---

# 96. What Not To Claim

Do NOT claim:

```text
first multimodal synergy method

first higher-order fusion

first tensor multimodal representation

first conditional multimodal method

first interaction decomposition
```

without systematic literature verification.

---

# 97. Publication Language

Prefer:

```text
we observe

we empirically find

we measure

we encourage

we approximate

we operationalize
```

Avoid unjustified wording such as:

```text
we guarantee unique information

we recover true synergy

we perfectly disentangle modalities
```

---

# 98. Reproducibility

Every official run saves:

```text
dataset version

split

seed

config

parameter count

git commit

git working-tree state

best epoch

checkpoint

validation metric

test metrics

runtime

hardware
```

---

# 99. Working Tree Rule

Confirmatory runs should ideally use a clean git working tree.

If dirty:

```text
save git diff
```

with experiment metadata.

---

# 100. Unit Tests

Mandatory tests for every new loss/module:

```text
correct tensor shapes

finite outputs

finite gradients

batch size 1

gradient propagation

no accidental detach

shuffle behavior

zero behavior
```

---

# 101. Utility Loss Unit Test

Verify:

```text
full better than lower
-> low/zero penalty

full worse than lower
-> positive penalty
```

Verify:

```text
lower baseline is stop-gradient
```

when intended.

---

# 102. Dependency Loss Test

Verify corrupted interactions:

```text
modality 1 shuffled

modality 2 shuffled
```

are actually recomputed.

Do not shuffle only already-generated:

\[
r_{12}.
\]

---

# 103. Predictability Probe Protocol

Use:

```text
train representation extractor

freeze extractor

split probe train/validation/test

train predictor only on probe train

select on probe validation

evaluate probe test
```

Avoid leakage.

---

# 104. Probe Capacity Matching

Whenever comparing:

\[
Probe(R_{low})
\]

and:

\[
Probe(R_{full}),
\]

match parameter capacity as closely as possible.

A larger input should not automatically receive a much larger hidden network.

---

# 105. Hyperparameter Selection

Use validation only.

Procedure:

```text
exploratory sweep

select config from validation

freeze config

run official seeds

evaluate test once per seed
```

---

# 106. Immediate Next Task

Stop modifying AV-MNIST architecture.

The next agent task is:

```text
select and reproduce one genuine tri-modal ConFu benchmark
```

Priority:

```text
MOSI
```

unless repository support makes another original ConFu benchmark materially easier to reproduce.

---

# 107. Immediate Benchmark Audit

Before ConFu++ training:

1. inspect existing dataset loader,
2. confirm official split,
3. confirm modality shapes,
4. reproduce unimodal baselines,
5. reproduce pair baselines,
6. reproduce ConFu-style baseline,
7. log modality dominance.

---

# 108. Immediate MOSI Diagnostics

For:

```text
text
audio
vision
```

measure:

```text
text only

audio only

vision only

text + audio

text + vision

audio + vision

text + audio + vision
```

Then determine:

```text
dominant modality

weak modality

useful modality pairs
```

---

# 109. Immediate Pair Interaction Diagnostics

For each:

\[
r_{TA}
\]

\[
r_{TV}
\]

\[
r_{AV},
\]

run:

```text
variance

effective rank

single-modality shuffle

zero interaction

shuffle interaction

conditional linear probe

conditional nonlinear probe

predictability probe
```

---

# 110. Immediate Higher-Order Diagnostics

For original ConFu-style:

\[
r_{TAV},
\]

or corresponding fused higher-order representation if implemented:

run:

```text
shuffle text

shuffle audio

shuffle vision

zero r123

shuffle r123

lower-order vs full probe
```

---

# 111. MOSI Decision Gate

If ConFu interaction already demonstrates:

```text
multi-modality dependence
+
conditional utility
```

then proceed to compare ConFu++ utility objective.

If it does not:

```text
document shortcut pattern
```

before modifying architecture.

---

# 112. ConFu++ P1 on Genuine Multimodal Data

Add only:

\[
L_{utility}.
\]

No new architecture.

Compare:

```text
ConFu

ConFu + utility
```

on identical encoders and seeds.

---

# 113. ConFu++ P2

If utility improves but dependence remains unbalanced, test a redesigned modality-necessity objective.

Do NOT reuse failed AV-MNIST ranking objective blindly.

Use leave-one-modality-out behavior as the main design target.

---

# 114. ConFu++ P3

Only if:

```text
all required modalities influence interaction
```

but:

```text
interaction remains redundant
```

test residual interaction.

---

# 115. ConFu++ P4

Only if:

```text
global fusion shows real complementarity
```

test Global–Local interaction.

---

# 116. ConFu++ P5

Only after pairwise methodology is validated:

```text
optimize third-order conditional utility
```

and evaluate:

\[
r_{123}.
\]

---

# 117. ConFu++ P6

Only after third-order utility works:

```text
study combinatorial scalability

selective subset fusion

dynamic higher-order routing
```

---

# 118. Development Order

The agent must follow:

```text
1. Preserve AV-MNIST as negative control.

2. Reproduce genuine tri-modal ConFu benchmark.

3. Audit unimodal and pairwise strength.

4. Measure ConFu interaction dependence.

5. Measure ConFu conditional utility.

6. Apply utility objective only.

7. Evaluate five seeds.

8. If needed, design modality-necessity objective.

9. Re-evaluate dependence.

10. If dependence passes but redundancy remains:
    test residual interaction.

11. If pairwise complementarity is established:
    test third-order complementarity.

12. Only then:
    Global–Local.

13. Only then:
    cross-attention / larger fusion.

14. Finally:
    selective higher-order scalability.
```

Do not reorder without written experimental justification.

---

# 119. Agent Behavior Before Coding

Before changing code:

```text
read current experiment log

inspect latest result JSONs

inspect configs

inspect interaction implementation

inspect losses

inspect tests

reproduce current baseline if required
```

Do not infer implementation state from AGENT.md alone.

---

# 120. Agent Report Format

For every experiment report:

```text
Research question

Hypothesis

Root cause addressed

Files changed

Mathematical change

Parameter-count change

Protocol

Validation result

Test result

Dependency diagnostics

Conditional utility diagnostics

Representation diagnostics

Statistical result

Interpretation

Failure mode

Decision

Next experiment
```

---

# 121. Architecture Change Rule

Every architecture addition must answer a specific unresolved question.

Bad reason:

```text
Cross-attention may improve accuracy.
```

Valid reason:

```text
Global interaction shows genuine complementarity,
but analysis suggests global pooling discards
task-relevant sub-global evidence.
```

---

# 122. Objective Change Rule

Every loss term must have:

```text
explicit failure mode

expected effect

diagnostic measuring that effect
```

Example:

```text
Utility loss

Failure:
interaction predicts target but adds no conditional value.

Expected:
increase Gain12 and ZeroDrop.

Diagnostic:
conditional probes + correction/regression.
```

---

# 123. Research Philosophy

The project must distinguish:

\[
\boxed{
\text{Fusion}
}
\]

from:

\[
\boxed{
\text{Dependence}
}
\]

from:

\[
\boxed{
\text{Complementarity}
}
\]

from:

\[
\boxed{
\text{Synergy}.
}
\]

These terms are not interchangeable.

---

# 124. Key AV-MNIST Lesson

The strongest current evidence is:

\[
\boxed{
r_{12}=F(r_1,r_2)
}
\]

but:

\[
\boxed{
r_{12}\approx f(r_1).
}
\]

This demonstrates:

\[
\boxed{
\textbf{Fusion is not dependence.}
}
\]

Keep this result visible throughout the project.

---

# 125. Key ConFu++ Thesis

The new scientific thesis is:

\[
\boxed{
\textbf{Higher-order alignment is not sufficient evidence of higher-order information.}
}
\]

ConFu++ aims to determine when interaction representations contain genuinely complementary information and how to optimize for it.

---

# 126. Final Target Representation

For three modalities:

```text
Individual evidence

r1
r2
r3

Pair interactions

r12
r13
r23

Third-order interaction

r123
```

Each interaction should be evaluated along:

```text
Activity

Dependence

Conditional Utility
```

and eventually, when justified:

```text
Global scale

Local scale

Residual complementarity
```

---

# 127. Final Scientific Goal

The project is successful only if it can demonstrate:

\[
\boxed{
\text{interaction representations that require multiple modalities}
}
\]

and:

\[
\boxed{
\text{provide information beyond lower-order representations}
}
\]

under controlled, reproducible experiments.

Raw accuracy improvement alone is insufficient.

---

# 128. One-Sentence Paper Thesis

> **ConFu++ distinguishes multimodal fusion from genuine multimodal synergy by explicitly measuring and optimizing higher-order dependence and conditional complementarity.**

---

# 129. Immediate Agent Execution

The next execution must:

```text
1. Leave the current AV-MNIST result unchanged.

2. Record it as a negative-control benchmark.

3. Inspect repository support for MOSI,
   MUStARD, UR-FUNNY, and Bird-MML.

4. Select the easiest genuine tri-modal
   benchmark that reproduces original ConFu conditions.

5. Reproduce its original ConFu baseline.

6. Train unimodal and pairwise baselines.

7. Run dependency and conditional-utility diagnostics.

8. Produce a benchmark audit report.

9. Do NOT add residual, Global–Local,
   cross-attention, rank expansion,
   or new large architecture yet.

10. Use the audit result to decide
    the next valid ConFu++ experiment.
```

---

# 130. Final Rule

Whenever choosing between:

```text
making the fusion network stronger
```

and:

```text
obtaining stronger evidence that the representation
actually contains multimodal complementary information
```

choose the second.

The project exists to answer:

\[
\boxed{
\text{What does multimodal interaction know}
}
\]

that:

\[
\boxed{
\text{the individual modalities do not already know?}
}
\]

Everything else is secondary.