# AGENT.md

# ConFu++ v2

## Utility-First Higher-Order Multimodal Representation Learning

---

# 0. Mission

This repository develops a direct research extension of **ConFu: Contrastive Fusion for Higher-Order Multimodal Alignment**.

The current research objective is no longer:

> make fused representations less redundant at any cost.

The new objective is:

> learn multimodal interaction representations that provide measurable conditional utility beyond strong lower-order representations.

The central principle is:

\[
\boxed{
\text{de-redundancy}
\neq
\text{complementarity}
}
\]

and:

\[
\boxed{
\text{fusion}
\neq
\text{dependence}
\neq
\text{conditional utility}
\neq
\text{synergy}.
}
\]

ConFu++ v2 must prioritize:

\[
\boxed{
\text{useful interaction}
}
\]

over:

\[
\boxed{
\text{interaction that merely looks different from unimodal features}.
}
\]

---

# 1. Current Scientific State

The project has completed controlled evaluation on:

```text
AV-MNIST
CMU-MOSI
UR-FUNNY
```

The experiments reveal three distinct failure patterns.

---

# 2. AV-MNIST Finding

AV-MNIST demonstrated:

```text
interaction non-collapse
+
interaction predictiveness
+
small conditional utility
```

but approximately:

\[
r_{12}\approx f(r_{image}).
\]

The interaction is almost entirely image-derived.

Scientific interpretation:

\[
\boxed{
\text{joint computation does not imply joint information}
}
\]

AV-MNIST is therefore retained as a:

```text
negative control
modality-shortcut benchmark
```

Do not optimize architecture aggressively on AV-MNIST.

---

# 3. MOSI Finding

CMU-MOSI demonstrated a different failure.

Text-related pair interactions:

\[
r_{VT},
r_{AT}
\]

respond meaningfully to corruption of both modalities.

However:

\[
R^2(R_{low}\rightarrow r_{ij})
\approx0.92-0.95
\]

and average conditional utility is negative.

Scientific interpretation:

\[
\boxed{
\text{multi-modality dependence does not imply complementarity}
}
\]

MOSI is retained as a:

```text
redundancy control
text-dominance benchmark
```

---

# 4. UR-FUNNY Finding

UR-FUNNY is the current primary development benchmark.

Original ConFu:

\[
64.462\pm0.713\%
\]

ConFu++ S1 + task head:

\[
63.549\pm0.855\%.
\]

Difference:

\[
-0.913\pm1.012\text{ pp}.
\]

The current ConFu++ configuration therefore fails to improve the baseline.

---

# 5. UR-FUNNY Modality Baseline

Approximate frozen-probe performance:

```text
Vision:
51.25%

Audio:
57.88%

Text:
62.18%

All lower-order modalities:
64.46%
```

Thus:

\[
64.46-62.18
\approx
2.28\text{ pp}
\]

multimodal improvement exists beyond text alone.

This is important.

UR-FUNNY has empirical multimodal headroom.

The goal is to exploit this headroom without destroying the strong lower-order representations.

---

# 6. UR-FUNNY Pair Interaction State

Original ConFu interactions are active.

Typical seed-1 diagnostics:

```text
r12 interaction shuffle drop:
~7.5 pp

r13:
~8.0 pp

r23:
~9.2 pp
```

Zeroing interactions also causes large degradation in pair-level probes.

Therefore:

```text
interaction activity = PASS
interaction ignored = FALSE
representation collapse = FALSE
```

---

# 7. UR-FUNNY Redundancy State

Interaction predictability remains extremely high:

\[
R^2(
[r_1,r_2]
\rightarrow
r_{12}
)
\approx0.963
\]

\[
R^2(
[r_1,r_3]
\rightarrow
r_{13}
)
\approx0.970
\]

\[
R^2(
[r_2,r_3]
\rightarrow
r_{23}
)
\approx0.970.
\]

However, ConFu++ v2 must NOT treat these high values as a standalone failure.

The new question is:

\[
\boxed{
\text{Does the interaction provide useful residual information?}
}
\]

not:

\[
\boxed{
\text{Can we force }R^2<0.5?
}
\]

---

# 8. Why S1 Is Rejected

Previous S1 attempted adversarial de-shortcutting.

For each pair:

\[
q_i(r_i)\rightarrow r_{ij}
\]

and:

\[
q_j(r_j)\rightarrow r_{ij}.
\]

The generator was penalized when the adversaries predicted the interaction too accurately.

This objective is no longer part of the default ConFu++ method.

---

# 9. S1 Failure

S1 optimized:

\[
\text{lower interaction predictability}
\]

without explicitly optimizing:

\[
\text{higher conditional task utility}.
\]

Therefore:

\[
\boxed{
\text{less predictable representation}
}
\]

did not become:

\[
\boxed{
\text{more useful representation}.
}
\]

S1 also reduced all three pair probe accuracies on average.

Therefore:

```yaml
shortcut_adversary:
  enabled: false
```

is the new default.

---

# 10. Do Not Continue S1 Hyperparameter Sweeps

Do NOT spend compute sweeping:

```text
lambda_shortcut
shortcut_tau
adversary hidden size
adversary learning rate
number of adversary steps
```

unless future evidence specifically reopens the adversarial-dependence hypothesis.

The current bottleneck is utility.

---

# 11. Major Training–Evaluation Mismatch

Original ConFu learns:

\[
r_1,r_2,r_3
\]

and:

\[
r_{12},r_{13},r_{23}.
\]

However the current primary:

```text
AllModalities
```

evaluation uses:

\[
[r_1;r_2;r_3]
\]

only.

It does NOT use:

\[
r_{12},r_{13},r_{23}.
\]

This creates a fundamental mismatch.

We currently modify pair interaction representations during training while the final main probe discards them.

This must be corrected before introducing larger architectures.

---

# 12. First Mandatory Experiment — E0

Before any new training, use the original ConFu checkpoints.

Compare:

\[
Z_{low}
=
[r_1;r_2;r_3]
\]

against:

\[
Z_{pairs}
=
[r_1;r_2;r_3;r_{12};r_{13};r_{23}].
\]

Run:

```text
linear probe

capacity-matched nonlinear probe
```

on both representations.

This is evaluation-only.

Do not retrain encoders.

---

# 13. E0 Research Question

E0 answers:

> Does original ConFu already contain useful interaction information that the repository's standard AllModalities evaluation ignores?

Define:

\[
Gain_{pairs}
=
Perf(Z_{pairs})
-
Perf(Z_{low}).
\]

If:

\[
Gain_{pairs}>0,
\]

the interaction representations already contain downstream headroom.

This would strongly justify the new downstream architecture.

---

# 14. E0 Probe Capacity Rule

The nonlinear probe comparison must be capacity matched.

Do not simply give:

\[
Z_{pairs}
\]

a larger MLP because its input dimension is larger.

Use approximately equal total trainable parameters.

Report:

```text
parameter count
optimizer
learning rate
epochs
validation metric
test metric
```

---

# 15. ConFu++ v2 Core Idea

The new architecture is:

\[
\boxed{
\text{strong lower-order prediction}
+
\text{conservative interaction correction}
}
\]

rather than:

\[
\boxed{
\text{interaction as another full predictor}.
}
\]

Interaction representations should correct errors that remain after lower-order multimodal reasoning.

---

# 16. Lower-Order Representation

Define:

\[
z_L
=
[r_1;r_2;r_3].
\]

Optionally project:

\[
h_L=P_L(z_L).
\]

Lower-order logits:

\[
\ell_L
=
C_L(h_L).
\]

This branch represents the strongest current non-interaction baseline.

---

# 17. Pair Interaction Representation

Pair interactions remain:

\[
r_{12},
r_{13},
r_{23}.
\]

Normalize:

\[
\hat r_{ij}
=
LN(r_{ij}).
\]

Do not force them to become independently predictive.

---

# 18. Pair Residual Correction

Each pair produces a correction:

\[
\Delta\ell_{12}
=
C_{12}(\hat r_{12})
\]

\[
\Delta\ell_{13}
=
C_{13}(\hat r_{13})
\]

\[
\Delta\ell_{23}
=
C_{23}(\hat r_{23}).
\]

The final prediction is:

\[
\boxed{
\ell_F
=
\ell_L
+
g_{12}\Delta\ell_{12}
+
g_{13}\Delta\ell_{13}
+
g_{23}\Delta\ell_{23}
}
\]

This is the main ConFu++ v2 architecture.

---

# 19. Why Logit Residual Correction

The purpose of an interaction is no longer:

> reproduce the whole label decision.

Its purpose is:

> correct what lower-order evidence gets wrong.

Therefore:

\[
\Delta\ell_{ij}
\]

is preferred over another independent classifier.

This makes interaction utility directly measurable.

---

# 20. Conservative Pair Gates

Each pair has:

\[
g_{ij}
=
\sigma(a_{ij}).
\]

Do NOT initialize:

\[
g_{ij}=0.5.
\]

Initialize conservatively:

\[
g_{ij}\approx0.1.
\]

Therefore:

\[
a_{ij}
=
\operatorname{logit}(0.1)
\approx-2.197.
\]

The interaction must learn to earn influence over the lower-order prediction.

---

# 21. Why Small Gate Initialization

The lower-order representation already achieves strong performance.

Large random correction may damage correct predictions.

Small initialization ensures:

\[
\ell_F
\approx
\ell_L
\]

at the start of training.

This improves training stability and interpretability.

---

# 22. Stage-Wise Training

Do NOT train everything end-to-end immediately.

Use:

```text
Stage A
freeze original ConFu backbone

Stage B
train lower-order classifier if needed

Stage C
freeze lower branch

Stage D
train interaction correction heads + gates

Stage E
optional low-learning-rate joint fine-tuning
```

---

# 23. Stage A — Original ConFu Backbone

Start from reproduced original ConFu checkpoint.

Freeze:

```text
modality encoders
unimodal projections
initial interaction modules
```

during the first residual-correction experiment.

The objective is to test:

> Can existing ConFu interactions provide useful corrections?

without changing the representation itself.

---

# 24. Stage B — Lower-Order Head

Train:

\[
C_L([r_1,r_2,r_3]).
\]

Select checkpoint using validation downstream metric.

Possible metric:

```text
validation binary cross entropy
```

or:

```text
validation accuracy / macro F1
```

depending on protocol.

The exact inference graph must be used for checkpoint selection.

---

# 25. Stage C — Freeze Lower Predictor

Once:

\[
\ell_L
\]

is stable, freeze:

```text
encoders
lower-order classifier
```

and train only:

```text
interaction correction heads
pair gates
```

This isolates incremental utility.

---

# 26. Stage D — Optional Joint Fine-Tuning

Only after residual correction demonstrates positive validation utility.

Use:

\[
lr_{encoder}
\ll
lr_{correction}.
\]

Recommended starting ratio:

\[
lr_{encoder}
=
0.1
\times
lr_{correction}.
\]

Potentially use:

\[
0.01
\]

if gradients destabilize contrastive geometry.

---

# 27. Main Final Loss

For final logits:

\[
\ell_F,
\]

define:

\[
L_{final}
=
CE(\ell_F,y).
\]

This is the primary task loss.

---

# 28. Lower-Order Loss

Lower prediction:

\[
\ell_L.
\]

Per-sample lower loss:

\[
c_i^L
=
CE(\ell_{L,i},y_i).
\]

Per-sample final loss:

\[
c_i^F
=
CE(\ell_{F,i},y_i).
\]

---

# 29. Conditional Utility Loss

Define:

\[
\boxed{
L_{utility}
=
\frac1B
\sum_i
\max(
0,
c_i^F
-
stopgrad(c_i^L)
+
m
)
}
\]

Initial:

\[
m=0.
\]

Interpretation:

> interaction correction should not produce worse task loss than the frozen lower-order predictor.

---

# 30. Utility Loss Starting Weight

Start with:

```yaml
utility:
  weight: 0.1
  margin: 0.0
```

Do NOT perform large test-set sweeps.

Select using validation.

Candidate exploratory weights:

```text
0.05
0.10
0.20
```

only if required.

---

# 31. Preserve Loss

Interactions should not destroy strong high-confidence correct lower-order predictions.

Let:

\[
p_L=softmax(\ell_L).
\]

For samples where:

\[
\hat y_L=y
\]

and:

\[
p_L(y)>\tau,
\]

define:

\[
L_{preserve}
=
KL(
stopgrad(p_L)
\Vert
p_F
).
\]

---

# 32. Preserve Threshold

Starting point:

\[
\tau=0.8.
\]

This is a validation-selected hyperparameter.

Do NOT use ground-truth correctness during inference.

Ground truth is allowed only for the training loss mask.

---

# 33. Preserve Loss Purpose

The model should behave approximately as:

```text
easy / confident sample
        ↓
do little

uncertain / difficult sample
        ↓
allow interaction correction
```

This directly targets:

\[
Correction>Regression.
\]

---

# 34. Hard-Sample Weighting

Define lower-order uncertainty.

Possible:

\[
u_i
=
1-p_L(y_i)
\]

during training.

Or label-free entropy:

\[
u_i
=
H(p_L).
\]

Task loss may be weighted:

\[
L_{hard}
=
\frac1B
\sum_i
(1+\alpha u_i)
CE(\ell_{F,i},y_i).
\]

This is optional.

Do not enable in E1.

---

# 35. Development Rule

Introduce one mechanism at a time.

Experiment order:

```text
E0
evaluation-only pair headroom

E1
residual correction

E2
+ utility loss

E3
+ preserve loss

E4
+ scalar pair gates

E5
+ sample-dependent gates

E6
+ modality necessity

E7
optional low-LR joint fine-tuning

E8
explicit r123
```

Do not jump ahead.

---

# 36. E1 — Residual Correction Only

Architecture:

\[
\ell_F
=
\ell_L
+
\Delta\ell.
\]

No utility regularizer.

No preserve loss.

No adversary.

No dynamic gates.

Use fixed or learnable conservative scalar pair gates.

Purpose:

> determine whether existing ConFu interaction contains incremental downstream information.

---

# 37. E1 Success Criteria

Validation:

\[
Gain_{pairs}>0.
\]

Test confirmatory target:

\[
Perf_{E1}
>
Perf_{original\ ConFu}.
\]

Also require:

\[
NetCorrection>0.
\]

---

# 38. E2 — Add Utility Loss

Only if E1 has:

```text
interaction signal
but unstable corrections
```

Add:

\[
L_{utility}.
\]

Evaluate whether:

```text
Gain increases
NetCorrection increases
Regression decreases
```

---

# 39. E3 — Add Preserve Loss

Only if:

```text
correction count positive
but regressions remain high
```

Add:

\[
L_{preserve}.
\]

Evaluate:

\[
Correction
\]

and:

\[
Regression
\]

separately.

---

# 40. E4 — Pair-Specific Scalar Gates

Different pair interactions should not be treated equally.

Use:

\[
g_{12},
g_{13},
g_{23}.
\]

Do not force:

\[
g_{12}=g_{13}=g_{23}.
\]

UR-FUNNY pair utility is heterogeneous.

---

# 41. Gate Diagnostics

Log:

```text
g12
g13
g23
```

every epoch.

At evaluation, run forced gate ablation:

```text
gate = 0
gate = learned
gate = 1
```

for each pair.

---

# 42. Pair Zero Test

For each pair:

\[
\Delta\ell_{ij}=0.
\]

Measure:

\[
ZeroDrop_{ij}.
\]

A useful pair should have:

\[
ZeroDrop_{ij}>0.
\]

---

# 43. Pair Shuffle Test

Shuffle:

\[
r_{ij}
\]

across samples while keeping all lower representations fixed.

Measure:

\[
ShuffleDrop_{ij}.
\]

Interpret with ZeroDrop.

Do not use ShuffleDrop alone as proof of utility.

---

# 44. Sample-Specific Gating

Only after scalar gates demonstrate useful pair heterogeneity.

Define:

\[
g_{ij}(x)
=
\sigma(
G_{ij}(
r_i,
r_j,
r_{ij},
u_L
)
).
\]

Use a small network.

Recommended starting hidden dimension:

```text
64
```

Do NOT use a deep router.

---

# 45. Gate Sparsity

Optional:

\[
L_{gate}
=
\lambda_g
\sum_{ij}
g_{ij}.
\]

Starting:

```yaml
gate_regularization:
  weight: 0.001
```

Purpose:

> interaction should activate only when needed.

---

# 46. Gate Statistics

Log:

```text
mean gate
std gate

gate on lower-correct samples
gate on lower-wrong samples

gate on high-confidence samples
gate on low-confidence samples
```

Desired trend:

\[
g_{hard}>g_{easy}.
\]

---

# 47. Interaction Utility Metrics

Every experiment must report:

\[
Gain
=
Perf_{full}-Perf_{lower}.
\]

Also report:

```text
Correction
Regression
NetCorrection
ZeroDrop
ShuffleDrop
```

---

# 48. Correction–Regression Metric

Define:

```text
lower wrong -> final correct
= Correction

lower correct -> final wrong
= Regression
```

Then:

\[
NetCorrection
=
Correction-Regression.
\]

ConFu++ v2 should prioritize:

\[
NetCorrection>0.
\]

---

# 49. Conditional Probe Metrics

Continue to evaluate:

\[
Probe(R_{low})
\]

versus:

\[
Probe(R_{low},R_{interaction}).
\]

Use:

```text
linear probe
capacity-matched nonlinear probe
```

The nonlinear probe must no longer be skipped for official evaluation.

If CPU is too slow, implement GPU MLP probe.

---

# 50. GPU Nonlinear Probe Requirement

Implement a small PyTorch MLP probe.

Recommended:

```text
hidden = 128
activation = GELU or ReLU
early stopping = validation
```

Use matched total parameter budget between lower and full representations.

---

# 51. Predictability Is Diagnostic Only

Continue logging:

\[
R^2(R_{low}\rightarrow r_{ij}).
\]

But remove:

```text
R² < 0.5
```

as a hard acceptance gate.

High predictability is allowed if:

\[
conditional\ utility>0.
\]

---

# 52. New Redundancy Interpretation

A representation may contain:

\[
r_{ij}
=
r_{shared}
+
r_{complementary}.
\]

Removing:

\[
r_{shared}
\]

may damage semantic stability.

Therefore ConFu++ v2 should optimize:

\[
\boxed{
task-relevant residual utility
}
\]

instead of:

\[
\boxed{
minimal statistical predictability
}
\]

alone.

---

# 53. Modality Necessity — Later Stage

Only after interaction correction works.

For full input:

\[
p_F
=
p(y|V,A,T).
\]

Evaluate leave-one-modality-out predictions:

\[
p_{-V},
p_{-A},
p_{-T}.
\]

The goal is to identify when each modality contributes useful conditional evidence.

---

# 54. Do Not Force Every Modality on Every Sample

UR-FUNNY is heterogeneous.

Some samples may be solvable from text.

Others may require:

```text
audio prosody
visual behavior
context
```

Therefore:

\[
\boxed{
\text{modality necessity must eventually be sample-selective}.
}
\]

Do not optimize global balanced modality dependence blindly.

---

# 55. Selective Necessity Objective

Possible future formulation.

For modality:

\[
m,
\]

compute:

\[
\Delta c_m
=
CE_{-m}
-
CE_F.
\]

Define necessity weight:

\[
w_m
=
stopgrad(
\max(0,\Delta c_m)
).
\]

Then apply necessity regularization proportional to:

\[
w_m.
\]

This encourages reliance only when removing the modality actually hurts.

Do not implement before E1-E5.

---

# 56. Gradient Conflict

Previous ConFu++ optimized:

\[
L_{contrastive}
+
L_{task}
+
L_{shortcut}.
\]

These objectives can compete.

ConFu++ v2 must explicitly monitor gradient conflict.

---

# 57. Gradient Diagnostics

For shared encoder parameters, periodically compute gradient cosine between:

\[
\nabla L_{contrastive}
\]

and:

\[
\nabla L_{final},
\]

between:

\[
\nabla L_{utility}
\]

and:

\[
\nabla L_{final}.
\]

Log:

```text
gradient norm
gradient cosine
```

Do not modify optimization based on these metrics initially.

Use them diagnostically.

---

# 58. Stop-Gradient Strategy

During residual-correction training:

\[
z_L
=
stopgrad([r_1,r_2,r_3]).
\]

Lower-order logits may also be detached.

Interaction correction learns relative to a stable teacher.

This isolates the correction problem.

---

# 59. No Full End-to-End Training Initially

Do NOT backpropagate:

\[
L_{utility}
\]

into all modality encoders during E1/E2.

This may distort the lower-order representation to artificially make the interaction appear useful.

Only open encoder gradients after correction utility is established.

---

# 60. Checkpoint Selection

This is mandatory.

Select checkpoints according to the exact downstream inference graph.

If inference uses:

\[
\ell_F,
\]

checkpoint selection must use validation performance of:

\[
\ell_F.
\]

Do NOT select based on:

```text
contrastive loss alone
combined auxiliary training loss
adversarial loss
```

unless that quantity is the actual inference objective.

---

# 61. Early Stopping

Previous task-only results suggest long training may degrade downstream geometry.

Therefore use early stopping.

Starting:

```yaml
max_epochs: 40
patience: 5
```

The maximum is not a requirement to train all 40 epochs.

Select best validation checkpoint.

---

# 62. No Test-Driven Epoch Selection

Do NOT select:

```text
30 epochs
```

because a previous test result happened to be good.

Use validation curves.

Test set is evaluated only after configuration is frozen.

---

# 63. Context Audit — UR-FUNNY

Before major architecture work, inspect the UR-FUNNY loader.

Determine whether the current:

```text
[B,50,D]
```

sequences represent:

```text
punchline only
context only
context + punchline
truncated merged sequence
```

This must be documented.

---

# 64. Context/Punchline Representation

If the current loader discards or poorly merges context, create separate:

\[
H_i^C
\]

for context and:

\[
H_i^P
\]

for punchline.

Do NOT automatically implement this.

First audit current data path.

---

# 65. Context-Conditioned Representation

If justified:

\[
r_i^C
=
E_i(H_i^C)
\]

\[
r_i^P
=
E_i(H_i^P)
\]

then:

\[
r_i
=
F_i(r_i^C,r_i^P).
\]

Only after proving context information is currently lost.

---

# 66. Why Context Matters

UR-FUNNY humor classification may require:

```text
what was said before
+
how the punchline was delivered
```

Therefore data representation can become a bottleneck independently of fusion architecture.

Always audit data before adding model complexity.

---

# 67. Original ConFu Baseline Must Remain Frozen

Do not modify the stored 5-seed original ConFu results.

Use them as canonical comparison.

Primary baseline:

\[
64.462\pm0.713\%.
\]

All future claims must compare against the exact same probe/evaluation implementation.

---

# 68. Probe Solver Consistency

Always use the same evaluation implementation for baseline and proposed model.

Do not compare metrics produced by different probe solvers.

Store:

```text
solver
regularization
standardization
random seed
validation selection
```

in result metadata.

---

# 69. Five-Seed Confirmatory Protocol

Exploratory experiments:

```text
seed 1
```

or a small validation-only sweep.

Confirmatory experiment:

```text
seeds 1–5
```

with configuration frozen beforehand.

---

# 70. Baseline Target

Current target:

\[
64.462\%.
\]

Minimum useful confirmatory result:

\[
mean\ accuracy
>
64.462\%.
\]

Prefer:

\[
65.0\%-65.5\%
\]

with stable seed behavior before claiming a practical improvement.

Do not encode this number as a training target.

---

# 71. Statistical Reporting

For each final model report:

```text
mean
sample std
per-seed scores
paired difference
paired difference std
95% CI
paired t-test
effect size
```

Do not claim superiority solely from mean difference.

---

# 72. Pair-Level Performance

Also report:

```text
r12
r13
r23
```

and:

```text
lower + r12
lower + r13
lower + r23
```

to determine which pair actually contributes to global performance.

---

# 73. Pair Utility Matrix

Create table:

| Pair | Gain | ZeroDrop | ShuffleDrop | Gate | Correction | Regression |
|---|---:|---:|---:|---:|---:|---:|
| V-A | | | | | | |
| V-T | | | | | | |
| A-T | | | | | | |

This is mandatory.

---

# 74. Sample Difficulty Analysis

Partition validation/test samples by lower-order confidence.

Example:

```text
high confidence
medium confidence
low confidence
```

Measure:

\[
Gain_{pairs}
\]

within each group.

Hypothesis:

\[
Gain_{low-confidence}
>
Gain_{high-confidence}.
\]

---

# 75. Per-Class Analysis

For binary humor:

```text
humorous
non-humorous
```

report:

```text
correction
regression
pair gates
missing-modality sensitivity
```

Interaction behavior may differ substantially by class.

---

# 76. Full ConFu++ v2 Initial Loss

For E3:

\[
\boxed{
L
=
L_{final}
+
\lambda_U L_{utility}
+
\lambda_P L_{preserve}
}
\]

Optional gate sparsity later:

\[
+
\lambda_G L_{gate}.
\]

No S1 adversarial loss.

---

# 77. Initial Configuration

Suggested starting point:

```yaml
model:
  mode: residual_correction

backbone:
  source: original_confu
  freeze: true

correction:
  hidden_dim: 128
  gate_type: scalar
  initial_gate: 0.10

utility:
  enabled: true
  weight: 0.10
  margin: 0.0

preserve:
  enabled: true
  weight: 0.10
  confidence_threshold: 0.80

gate_regularization:
  enabled: false
  weight: 0.001

shortcut_adversary:
  enabled: false

training:
  max_epochs: 40
  early_stopping_patience: 5
  checkpoint_metric: validation_final_loss
```

Treat these values as starting hypotheses.

Select final values on validation only.

---

# 78. Required Ablations

Run:

```text
A0
Original ConFu lower-only

A1
Original ConFu lower+pairs probe

A2
Residual correction only

A3
Residual + utility

A4
Residual + utility + preserve

A5
A4 + pair scalar gates

A6
A5 + sample-specific gates

A7
A6 + modality necessity

A8
A7 + low-LR joint fine-tuning
```

Do NOT skip directly to A8.

---

# 79. Capacity-Matched Baseline

If correction architecture adds:

\[
N
\]

parameters, add a lower-only MLP baseline with approximately:

\[
N
\]

additional parameters.

This controls for generic capacity gains.

---

# 80. Interaction Correction Parameter Budget

Keep correction modules small.

Preferred total trainable inference overhead:

\[
<5\%
\]

relative to original ConFu.

Report:

```text
backbone params
correction params
gate params
total inference params
training-only params
```

---

# 81. Third-Order r123 — Not Yet

Do NOT implement:

\[
r_{123}
\]

until pair interactions demonstrate stable positive conditional utility.

The next scientific bottleneck is pair utility, not third-order capacity.

---

# 82. Trigger for r123

Implement explicit:

\[
r_{123}
\]

only when:

```text
Gain_pairs > 0
NetCorrection > 0
at least one pair has positive ZeroDrop
5-seed pair-augmented model is competitive with ConFu
```

---

# 83. Explicit Third-Order Representation

When triggered, use a simple initial operator.

Given:

\[
r_1,r_2,r_3,
\]

project:

\[
u_1=LN(U_1r_1)
\]

\[
u_2=LN(U_2r_2)
\]

\[
u_3=LN(U_3r_3).
\]

Compute:

\[
h_{123}
=
\frac{
u_1\odot u_2\odot u_3
}{
\sqrt R
}.
\]

Then:

\[
r_{123}
=
W_oh_{123}.
\]

Use bias-free output initially.

---

# 84. r123 Is a Correction, Not a Full Predictor

Third-order representation produces:

\[
\Delta\ell_{123}
=
C_{123}(r_{123}).
\]

Final:

\[
\ell_F
=
\ell_L
+
\Delta\ell_{pairs}
+
g_{123}\Delta\ell_{123}.
\]

Initialize:

\[
g_{123}\approx0.05
\]

or:

\[
0.1.
\]

---

# 85. Third-Order Lower Baseline

Define:

\[
R_{\le2}
=
[
r_1,r_2,r_3,
r_{12},r_{13},r_{23}
].
\]

Third-order full:

\[
R_{\le3}
=
[
R_{\le2},
r_{123}
].
\]

The central metric is:

\[
\boxed{
Gain_{123}
=
Perf(R_{\le3})
-
Perf(R_{\le2})
}
\]

not standalone:

\[
Perf(r_{123}).
\]

---

# 86. Third-Order Dependency Test

Evaluate:

\[
r_{123}(V,A,T)
\]

against:

\[
r_{123}(\pi(V),A,T)
\]

\[
r_{123}(V,\pi(A),T)
\]

\[
r_{123}(V,A,\pi(T)).
\]

Define:

\[
D_V,D_A,D_T.
\]

Desired:

\[
D_V>0,
D_A>0,
D_T>0.
\]

---

# 87. Third-Order Acceptance

Do NOT claim higher-order synergy unless:

```text
Gain123 > 0
ZeroDrop123 > 0
NetCorrection123 > 0
D_V > 0
D_A > 0
D_T > 0
```

preferably across multiple seeds.

---

# 88. Do Not Add Cross-Attention Yet

Cross-attention is prohibited before:

```text
residual correction works
pair utility is established
```

More interaction capacity does not solve an objective mismatch.

---

# 89. Do Not Sweep Rank Yet

Do NOT sweep:

```text
64
128
256
```

interaction dimensions to solve current performance.

The current failure is not collapse or insufficient rank.

---

# 90. Do Not Add Global–Local Yet

Global–Local interaction remains a future direction.

Trigger only if:

```text
global pair correction works
but residual error analysis indicates
missing fine-grained interaction evidence
```

---

# 91. Synthetic Benchmark

Maintain a controlled synthetic benchmark.

It should contain:

```text
redundant information
unique information
pairwise synergy
third-order synergy
```

This is the ground-truth mechanism benchmark.

---

# 92. Synthetic Pair Utility

For known:

\[
S_{12},
\]

evaluate:

\[
r_{12}\rightarrow S_{12}.
\]

Compare with:

\[
[r_1,r_2]\rightarrow S_{12}.
\]

---

# 93. Synthetic Third-Order Utility

For known:

\[
S_{123},
\]

compare:

\[
R_{\le2}\rightarrow S_{123}
\]

against:

\[
[R_{\le2},r_{123}]
\rightarrow S_{123}.
\]

This is the cleanest proof of order-specific utility.

---

# 94. Dataset Roles

Maintain:

```text
AV-MNIST
one-modality shortcut control

MOSI
multimodal dependence but redundancy control

UR-FUNNY
main real-world development benchmark

Synthetic
ground-truth higher-order mechanism benchmark
```

Later:

```text
MOSEI emotion
scale/generalization benchmark

MUStARD
small high-complementarity stress test
```

---

# 95. Current Scientific Thesis

The project now studies:

\[
\boxed{
\text{how to turn active but redundant interactions}
}
\]

into:

\[
\boxed{
\text{conditionally useful corrections}.
}
\]

This is more precise than simply:

> reduce redundancy.

---

# 96. New ConFu++ Thesis

The target paper thesis is:

> Higher-order multimodal representations should be evaluated by the incremental information they contribute beyond strong lower-order evidence. ConFu++ learns conservative interaction corrections that are activated only when multimodal interaction improves the downstream decision.

---

# 97. Potential Paper Story

Current empirical progression:

```text
AV-MNIST:
joint computation without genuine joint dependence

MOSI:
joint dependence without conditional utility

UR-FUNNY:
active multimodal interaction but strong redundancy

ConFu++ S1:
de-redundancy without utility

ConFu++ v2:
utility-first residual interaction
```

This is the current scientific narrative.

---

# 98. Main Failure Modes To Monitor

Always check:

```text
interaction ignored
interaction collapse
interaction redundancy
pair overfitting
lower-order degradation
correction regression
gate saturation
text domination
gradient conflict
checkpoint mismatch
evaluation mismatch
probe capacity mismatch
```

---

# 99. New Acceptance Hierarchy

## Health Gate

Require:

```text
finite representation
non-zero variance
reasonable effective rank
stable training
```

## Utility Gate

Require:

\[
Gain>0.
\]

## Correction Gate

Require:

\[
NetCorrection>0.
\]

## Necessity Gate

When applicable:

```text
useful modality removal hurts
```

## Statistical Gate

Prefer:

```text
positive paired mean
majority of seeds improve
confidence interval compatible with practical improvement
```

---

# 100. Removed Acceptance Rule

Remove:

```text
R² interaction predictability < 0.5
```

from hard acceptance criteria.

R² is now diagnostic only.

---

# 101. Experiment Logging

Every run must save:

```text
seed
git commit
working tree
dataset
split
model config
parameter counts
best epoch
validation metric
test metric
pair gates
correction
regression
ZeroDrop
ShuffleDrop
conditional probe gain
predictability R²
gradient diagnostics if enabled
```

---

# 102. Result JSON

Minimum example:

```json
{
  "seed": 1,

  "lower_accuracy": 0.0,
  "final_accuracy": 0.0,
  "gain": 0.0,

  "correction": 0,
  "regression": 0,
  "net_correction": 0,

  "g12": 0.0,
  "g13": 0.0,
  "g23": 0.0,

  "zero_drop_12": 0.0,
  "zero_drop_13": 0.0,
  "zero_drop_23": 0.0,

  "shuffle_drop_12": 0.0,
  "shuffle_drop_13": 0.0,
  "shuffle_drop_23": 0.0,

  "linear_conditional_gain": 0.0,
  "nonlinear_conditional_gain": 0.0
}
```

---

# 103. Experiment Report Format

After every experiment write:

```text
Research question

Hypothesis

Root cause addressed

Change

What remained fixed

Parameter-count difference

Validation result

Test result

Gain

Correction / regression

Pair contribution

Representation health

Failure mode

Decision

Next experiment
```

---

# 104. Decision Rule

Never say:

```text
accuracy improved,
therefore interaction improved.
```

Instead evaluate:

```text
Did lower-order performance remain intact?

Did pair interaction provide incremental utility?

Did correction exceed regression?

Did model rely on useful interaction?

Did improvement generalize across seeds?

Was capacity controlled?
```

---

# 105. Immediate Coding Task

The next coding task is NOT a new interaction backbone.

Implement:

```text
E0 pair-augmented probes

E1 frozen-backbone residual correction
```

first.

---

# 106. Immediate E0 Steps

1. Load original ConFu UR-FUNNY checkpoints.

2. Extract:

```text
r1
r2
r3
r12
r13
r23
```

3. Build:

\[
Z_{low}
\]

and:

\[
Z_{pairs}.
\]

4. Run standardized linear probe.

5. Run GPU parameter-matched MLP probe.

6. Report:

\[
Gain_{pairs}.
\]

7. No training modification.

---

# 107. Immediate E1 Steps

If E0 shows any useful signal:

1. Freeze ConFu backbone.

2. Train lower-order downstream classifier.

3. Freeze lower-order classifier.

4. Add pair correction heads.

5. Initialize pair gates to 0.1.

6. Train correction only.

7. Select checkpoint on validation final loss.

8. Evaluate test only after selection.

9. Report 1 seed exploratory result.

10. If sensible, freeze configuration.

11. Run five seeds.

---

# 108. E1 Abort Condition

If E0 shows:

\[
Gain_{pairs}\ll0
\]

under both linear and nonlinear probes,

do NOT immediately train residual correction.

First perform:

```text
pair-level hard-sample analysis

context/punchline loader audit

pair-specific conditional utility audit
```

to determine whether interaction information exists at all.

---

# 109. E2 Trigger

Add utility loss only if E1 shows:

```text
some corrections
but unstable or small positive gain
```

Do not add utility loss if the interaction carries no useful signal.

---

# 110. E3 Trigger

Add preserve loss only if:

\[
Regression
\]

is a meaningful source of lost accuracy.

---

# 111. E5 Trigger

Add dynamic gates only if:

```text
pair utility differs across samples
```

as demonstrated by confidence-stratified or per-sample analysis.

---

# 112. Joint Fine-Tuning Trigger

Open encoder gradients only if frozen correction already improves validation performance.

Otherwise joint fine-tuning risks hiding a weak correction mechanism behind representation drift.

---

# 113. r123 Trigger

Explicit higher-order embedding begins only after pair residual utility is established.

Do not use r123 as an escape from failed pair utility.

---

# 114. Efficiency Rule

Target:

\[
<5\%
\]

additional inference parameters for ConFu++ v2 core.

Training-only components may exceed this modestly, but must be reported separately.

---

# 115. Reproducibility Rule

Official runs require:

```text
fixed dataset version
fixed split
fixed seeds
fixed probe implementation
clean checkpoint naming
saved config
saved git commit
saved result JSON
```

No manual metric copying.

---

# 116. Test Leakage Rule

Do not select:

```text
loss weight
gate initialization
epoch count
hidden dimension
```

from test performance.

Use validation.

Previous exploratory runs that used test diagnostics must remain labeled exploratory.

---

# 117. Publication Claim Rule

Do NOT claim:

```text
true information-theoretic synergy
exact PID decomposition
unique information recovery
```

from residual correction alone.

Preferred language:

```text
conditional utility
incremental information
empirical complementarity
interaction correction
higher-order dependence
```

---

# 118. Strong Final Claim Target

A justified future claim would be:

> Original ConFu learns active higher-order interaction representations, but these representations can remain highly redundant with lower-order evidence. ConFu++ improves their practical value by learning conservative residual corrections that contribute only when interaction information improves the downstream decision.

---

# 119. Stronger Third-Order Claim Target

Only after r123 passes its gates:

> Explicit third-order interaction contributes task-relevant information beyond all unimodal and pairwise representations.

This requires:

\[
Gain_{123}>0.
\]

---

# 120. Current Project Status

```text
AV-MNIST shortcut diagnosis
DONE

MOSI redundancy diagnosis
DONE

UR-FUNNY baseline reproduction
DONE

UR-FUNNY S1
REJECTED

Task-only auxiliary head
REJECTED

Evaluation mismatch
IDENTIFIED

Pair-augmented probe
NEXT

Residual correction
NEXT

Conditional utility
PENDING

Preserve loss
PENDING

Dynamic routing
PENDING

Selective modality necessity
PENDING

Explicit r123
BLOCKED

Global-Local
BLOCKED

Cross-attention
BLOCKED
```

---

# 121. Main Priority

The next objective is NOT:

\[
\text{make }r_{ij}\text{ less predictable}.
\]

It is:

\[
\boxed{
\text{make }r_{ij}\text{ improve the prediction}
}
\]

without harming strong lower-order evidence.

---

# 122. Final Development Order

Follow exactly:

```text
1. Freeze all previous official results.

2. Implement E0 lower vs lower+pairs evaluation.

3. Add GPU nonlinear matched probe.

4. Run original ConFu E0 across five seeds.

5. Audit context/punchline data path.

6. If pair headroom exists:
   implement frozen residual correction.

7. Run seed 1 validation-driven E1.

8. If useful:
   run five seeds.

9. Add utility loss only if required.

10. Add preserve loss if regressions remain.

11. Add pair scalar gates.

12. Analyze gate behavior.

13. Add dynamic gates only if justified.

14. Add sample-selective modality necessity only if justified.

15. Allow low-LR joint fine-tuning only after frozen-stage success.

16. Only after pair utility passes:
    implement explicit r123.

17. Evaluate Gain123 and three-modality dependence.

18. Only after r123 success:
    consider Global-Local interaction.

19. Only after Global-Local evidence:
    consider cross-attention.

20. Never increase architecture complexity
    merely because the current method fails.
```

---

# 123. Core Scientific Principle

Whenever choosing between:

```text
more representation complexity
```

and:

```text
a cleaner test of incremental multimodal utility
```

choose the second.

---

# 124. One-Sentence Thesis

\[
\boxed{
\textbf{ConFu++ learns multimodal interactions as conservative residual corrections to strong lower-order evidence, rather than forcing fused representations to be different for the sake of being different.}
}
\]

---

# 125. Final Rule

The project is successful only when interaction representations answer:

> What can the multimodal interaction correct that the lower-order modalities could not already solve?

The primary quantity is therefore:

\[
\boxed{
\text{Conditional Utility}
}
\]

not:

\[
\boxed{
\text{Low Redundancy}.
}
\]

Everything else is secondary.