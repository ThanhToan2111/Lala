# AGENT.md

# ConFu++ / Representation-Level Complementarity

## Solving the Fusion–Dependence Gap in Higher-Order Multimodal Representation Learning

---

# 0. Status of This Document

This document is the **current primary research-agent specification**.

It builds on, and where indicated supersedes, the previous specifications:

```text
AGENT.md — Deep ConFu Upgrade ...
AGENT.md — ConFu++ Complementarity-Aware ...
```

Superseded items:

```text
- The P2 dependence-ranking objective is retired.
  It failed on AV-MNIST and must not be reused.

- The rule "change the dataset before changing the objective"
  is relaxed: the representation-level objectives defined here
  are tested on synthetic ground truth and on AV-MNIST
  (as negative control) BEFORE new benchmarks.

- Residual interaction is no longer gated behind
  "dependence must exist first"; it is integrated as
  a first-class training-time mechanism (Stage 3).
```

Everything related to diagnostics discipline, statistical protocol,
terminology, negative results, and benchmark auditing in the previous
ConFu++ specification remains in force unless explicitly replaced here.

---

# 1. The Problem Being Solved

ConFu learns:

\[
r_i = E_i(X_i), \qquad r_{12} = F(r_1, r_2),
\]

and aligns fused and remaining-modality representations contrastively:

\[
r_{12} \leftrightarrow r_3.
\]

Empirically established on AV-MNIST (five seeds, clean protocol):

```text
r12 is non-collapsed          (variance healthy, effective rank ~9)
r12 is predictive             (high standalone probe accuracy)
r12 is permutation-sensitive  (interaction shuffle drop > 0)
r12 is conditionally useful   (small but positive nonlinear gain)

BUT:

r12 ≈ f(r1)                   R²(r1 -> r12) = 0.988
                              R²(r2 -> r12) = 0.031
audio shuffle drop            +0.004 ± 0.027 pp  (absent)
```

The central fact:

\[
\boxed{\text{joint computation} \neq \text{joint information}}
\]

No term in the ConFu objective — nor in the ConFu++ P1 utility loss,
nor in the failed P2 ranking loss — penalizes the mechanism

\[
F(r_1, r_2) \;\longleftarrow\; f(r_1)
\]

at the representation level. Every objective tried so far acts on the
**task loss**, which a unimodal shortcut satisfies completely.

This specification defines the representation-level solution program.

---

# 2. Root-Cause Analysis

For the low-rank multiplicative interaction actually implemented:

\[
u_1 = LN(U_1 r_1), \qquad u_2 = LN(U_2 r_2),
\]

\[
h = \frac{u_1 \odot u_2}{\sqrt{R}}, \qquad r_{12} = W_o h,
\]

the optimizer has (at least) three escape routes that satisfy every
existing loss while producing a shortcut:

```text
Escape 1 — Strong-factor dominance
  u1 carries all label information; u2 contributes a near-constant
  or label-independent modulation. The product h ≈ u1 ⊙ c still has
  high variance, high rank, and full predictive power.

Escape 2 — Nonlinear copy
  r12 is a nonlinear re-encoding of r1 alone. Linear cross-covariance
  regularization does not prevent nonlinear redundancy. Measured:
  R²(q1) = 0.988 with a nonlinear probe.

Escape 3 — Classifier-side bypass
  Even if r12 were informative, the gated composition
  c_full = LN(c_base + g·LN(r12)) allows the classifier to rely on
  c_base; the gate is free to carry any residual gradient signal.
```

Why previous fixes failed:

```text
P1 utility loss (ReLU(CE_full - sg(CE_base))):
  Penalizes the OUTCOME of the shortcut, not the mechanism.
  The shortcut already satisfies CE_full ≈ CE_base;
  the loss is near zero and provides almost no gradient
  toward dual-modality dependence.

P2 ranking loss (true vs modality-shuffled interaction):
  Requires the corrupted interaction F(r1, π(r2)) to be separable
  from the positive F(r1, r2). Under shortcut,
  F(r1, π(r2)) ≈ F(r1, r2), so the ranking signal is ~zero
  at exactly the point where it is needed. The objective cannot
  bootstrap dependence from a shortcut solution.
```

Design conclusion:

\[
\boxed{
\text{Dependence cannot be recovered by task-level pressure alone.}
}
\]

The mechanism must be constrained where it is formed:

\[
\boxed{
\text{penalize the predictability of } r_{12}
\text{ from each single modality, during training.}
}
\]

---

# 3. Solution Overview

Three new training-time mechanisms, introduced in fixed order:

```text
S1  Adversarial de-shortcutting (representation level)
    r12 must be UNPREDICTABLE from r1 alone and from r2 alone,
    while remaining a deterministic function of (r1, r2) jointly
    and remaining task-useful.

S2  Factor health constraints
    Per-factor variance floors and capacity limits on u1, u2
    prevent Escape 1 (constant-factor collapse).

S3  Residual interaction as a training-time object
    r12^R = r12 - stopgrad(q(r1, r2)) is used as an explicit
    representation with its own head and utility objective,
    not only as an evaluation diagnostic.
```

Plus the methodology change that makes the program falsifiable:

```text
M1  Synthetic ground-truth validation FIRST.
    Objectives are validated on data with KNOWN synergy structure
    before any natural benchmark is touched.

M2  AV-MNIST is kept as negative control.
    A correct objective must NOT fabricate audio dependence there.
    Honest failure on AV-MNIST is a pass condition, not a bug.
```

---

# 4. Formal Objective Components

All losses below extend the canonical composition:

\[
c_{base} = LN(LN(r_1) + LN(r_2)), \qquad
c_{full} = LN(c_{base} + \sigma(a)\, LN(r_{12})).
\]

Classification uses the existing prototype classifier.

---

## 4.1 Task and unimodal losses (unchanged)

\[
L_{task} = CE(c_{full}, y), \qquad
L_{uni} = CE(r_1, y) + CE(r_2, y).
\]

---

## 4.2 Utility loss (retained, P1 configuration)

\[
L_{utility} = \operatorname{ReLU}\big(CE_{full} - sg(CE_{base}) + m\big),
\qquad \lambda_{utility} = 0.2,\; m = 0.
\]

---

## 4.3 Anti-collapse losses (retained)

```text
variance floor on r12
pair cross-covariance (r12, r1), (r12, r2)
off-diagonal self-covariance on r12
```

Canonical weights:

\[
L = L_{task} + 0.2 L_{uni} + 1.0 L_{var} + 0.01 L_{xcov} + 0.5 L_{selfcov}
+ \ldots
\]

---

## 4.4 S1 — Adversarial de-shortcutting loss (NEW, core mechanism)

Let:

\[
\mu, \sigma = \text{EMA batch statistics of } r_{12} \text{ (stop-grad)},
\qquad
t = \frac{r_{12} - sg(\mu)}{sg(\sigma)}.
\]

Two adversarial predictors, each an MLP capacity-matched to the
predictability probe:

\[
q_1 : r_1 \mapsto \hat t, \qquad q_2 : r_2 \mapsto \hat t.
\]

Crucially there is NO adversary from (r1, r2) jointly: r12 SHOULD be
predictable from the pair; it must NOT be predictable from either
modality alone.

### Adversary step (every iteration, gradients to q only)

\[
L_{adv} =
\big\| q_1(sg\, r_1) - sg\, t \big\|_2^2
+
\big\| q_2(sg\, r_2) - sg\, t \big\|_2^2.
\]

### Generator step (hinged penalty, gradients to encoders + interaction only)

\[
L_{shortcut}
=
\operatorname{ReLU}\big(\tau - \| sg[q_1](r_1) - t \|_2^2 / D\big)
+
\operatorname{ReLU}\big(\tau - \| sg[q_2](r_2) - t \|_2^2 / D\big),
\]

with gradient flowing through `t` (hence into `r12`, the interaction,
and the encoders) and NOT into `q1`, `q2`.

Design rationale:

```text
- Because t is standardized with stop-grad statistics, the
  normalized MSE equals 1 - R² of the adversary. Penalizing
  ReLU(tau - MSE) pushes single-modality R² BELOW (1 - tau)
  without demanding unpredictability beyond tau.

- Hinge (not raw MSE maximization) prevents the degenerate
  solution "r12 = noise": once the adversary fails by margin tau,
  the penalty is exactly zero and the task/utility losses dominate.

- tau is a bandwidth of tolerated redundancy. Initial: tau = 0.5
  (i.e., single-modality R² up to 0.5 is tolerated, above is taxed).
```

The minimax is trained by **alternating updates**, not gradient
reversal, to keep optimization explicit and debuggable:

```python
# per training step
for _ in range(k_adv):                      # k_adv = 1 initially
    loss_adv = mse(q1(sg(r1)), sg(t)) + mse(q2(sg(r2)), sg(t))
    opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

mse1 = mse(sg(q1)(r1), t) / D               # t carries grad to r12
mse2 = mse(sg(q2)(r2), t) / D
loss_shortcut = relu(tau - mse1) + relu(tau - mse2)
loss = loss_task + ... + lambda_shortcut * loss_shortcut
opt_gen.zero_grad(); loss.backward(); opt_gen.step()
```

### Mandatory logging

```text
adv/mse1, adv/mse2              (adversary fit quality)
adv/r2_1, adv/r2_2              (= 1 - mse, live shortcut meter)
loss_shortcut_raw, weighted
grad cosine(L_task, L_shortcut) (conflict diagnostic)
```

If `adv/r2_1` stays high while `loss_shortcut` stays at the hinge,
the penalty is saturated and must be reported as such — never tune
`lambda_shortcut` silently until the number looks good.

---

## 4.5 S2 — Factor health constraints (NEW)

To close Escape 1 (constant-factor collapse):

```text
per-factor variance floor:
    L_factorvar = VarFloor(u1) + VarFloor(u2)     (VICReg-style, gamma=1)

factor capacity limit:
    keep rank R = 64; do not raise it during S1/S2 studies.
```

Do NOT normalize away scale entirely: the product `u1 ⊙ u2` must keep
meaningful per-sample magnitude variation.

---

## 4.6 S3 — Residual interaction as a training-time representation

Predictor (stop-grad target):

\[
\hat r_{12} = q(r_1, r_2), \qquad
L_{pred} = \big\| q(r_1, r_2) - sg(r_{12}) \big\|_2^2,
\]

\[
r_{12}^{R} = r_{12} - sg(\hat r_{12}).
\]

The residual is used in a **second gated composition head**:

\[
c_{res} = LN(c_{base} + \sigma(a_r)\, LN(r_{12}^{R})),
\]

\[
L_{utility}^{R} =
\operatorname{ReLU}\big(CE(c_{res}) - sg(CE_{base}) + m\big).
\]

Interpretation discipline:

```text
r12^R is an OPERATIONAL residual, not exact unique information.
Claim: "the part of the interaction not linearly+nonlinearly
predictable from lower-order evidence", nothing stronger.
```

---

## 4.7 Full loss (all stages enabled; never start with all enabled)

\[
\begin{aligned}
L =\;& L_{task} + \lambda_{uni} L_{uni} + \lambda_{var} L_{var}
      + \lambda_{xcov} L_{xcov} + \lambda_{selfcov} L_{selfcov} \\
    & + \lambda_{utility} L_{utility}
      + \lambda_{shortcut} L_{shortcut}
      + \lambda_{factorvar} L_{factorvar}
      + \lambda_{pred} L_{pred}
      + \lambda_{res} L_{utility}^{R}.
\end{aligned}
\]

Mandatory rule:

```text
Enable exactly ONE new term per experiment stage.
Log raw and weighted values of every term separately.
```

---

# 5. M1 — Synthetic Ground-Truth Validation (Stage 0, MANDATORY FIRST)

No natural benchmark can prove the objective works, because natural
data has unknown synergy structure. Synthetic data has ground truth.

## 5.1 Pairwise synergy generator

Latent factors:

\[
a, b \sim \mathrm{Bernoulli}(0.5), \qquad
y = a \oplus b \quad \text{(pure synergy)}.
\]

Observations (two modalities, tunable noise σ):

```text
X1 = G1(a) + noise          (sees only a)
X2 = G2(b) + noise          (sees only b)
```

where G1, G2 are fixed random nonlinear maps to high-dimensional
vectors (e.g., 64-dim embeddings of the binary patterns + Gaussian
noise), so both modalities are nontrivial to encode.

Control knobs:

```text
synergy level      y = a⊕b  vs  y = a  vs  mixtures with prob rho
redundancy         leak y into X1 with probability rho
noise level        sigma on each modality
```

## 5.2 What must be measured

Representation selectivity against KNOWN latent targets:

\[
\text{Sel}_{12} =
Acc(r_{12} \rightarrow a \oplus b)
-
\max\big(Acc(r_1 \rightarrow a \oplus b),\, Acc(r_2 \rightarrow a \oplus b)\big).
\]

## 5.3 Pass conditions for S1 (adversarial de-shortcutting)

```text
Config A (pure synergy, y = a⊕b):
    baseline (no S1):     Sel12 ≈ 0, r12 collapses to one factor
    with S1:              Sel12 > 0 and significant over 5 seeds,
                          r12 predictability from r1 alone ≈ chance,
                          task accuracy near ceiling.

Config B (no synergy, y = a, both modalities see their factor):
    with S1:              the model must NOT fabricate dependence:
                          gate -> low, r12 utility ≈ 0,
                          no hallucinated r2 dependence.
                          (honesty / negative control)

Config C (mixture, rho controls redundancy):
    Sel12 and dependence metrics degrade gracefully with rho.
```

If S1 fails Config A, the mechanism is unsound: fix or abandon
before any natural-benchmark run. If S1 fabricates dependence in
Config B, the mechanism is dishonest: same consequence.

## 5.4 Third-order synthetic extension (later)

\[
a,b,c \sim \mathrm{Bernoulli}(0.5),\quad
y = a \oplus b \oplus c,
\]

with pairwise XORs observable. Used only after Stage 0 passes.

---

# 6. Stage Plan (MANDATORY ORDER)

```text
Stage 0   Synthetic ground truth (M1)
          Implement generator, baselines, S1 objective.
          PASS/FAIL gate: §5.3.

Stage 1   AV-MNIST negative control with S1 (+S2)
          Same frozen protocol as the canonical five-seed setup.
          Expected honest outcome:
            - image shortcut predictability R²(q1) decreases
              toward the hinge band OR utility collapses to zero;
            - NO fabricated audio dependence
              (audio shuffle drop must remain ≈ 0 if audio truly
               carries no conditional label information);
            - accuracy must not degrade beyond ~1 pp vs P1 config.
          Interpretation rule:
            if accuracy drops AND no dependence appears, the correct
            conclusion is "AV-MNIST offers nothing to align" — this
            is a PASS for honesty.

Stage 2   Residual interaction (S3) on synthetic Config A and AV-MNIST
          Compare raw r12 vs residual r12^R utility.

Stage 3   MOSI (or easiest genuine tri-modal ConFu benchmark)
          Follow the audit protocol from the previous specification
          (unimodal/pair/ConFu baselines, modality-dominance log)
          THEN apply S1 to the strongest pair (expected text–audio
          and text–vision) and to r123 only after pair gates pass.

Stage 4   Tri-modal utility + dependence on genuine data.

Stage 5   Global–Local / cross-attention / selective fusion
          (unchanged from previous specification; still gated).
```

Do not reorder. Every stage transition requires the written
experiment-log entry specified in §10.

---

# 7. Decision Gates (quantitative)

For each interaction under study, with metrics defined exactly as in
the existing diagnostics (`gain12`, `zero_drop12`, `shuffle_drop12`,
`shuffle_modality{1,2}_drop`, `net_correction`, probe gains,
predictability R²):

```text
GATE D (dependence):
    D1 > 0 AND D2 > 0 with consistent sign across seeds,
    AND adversarial meter adv/r2_i < 1 - tau for both i.

GATE U (utility):
    gain12 > 0 (paired across seeds), zero_drop12 > 0,
    net_correction > 0, nonlinear conditional gain > 0.

GATE H (health):
    variance floor satisfied, effective rank within ±50% of the
    no-S1 baseline, no loss term silently zero for > 3 epochs.

GATE E (efficiency):
    parameter increase over the matched baseline < 5%
    (adversaries and residual predictors are TRAINING-ONLY;
     they do not count toward inference capacity, but this must
     be stated explicitly in every report).

Proceed to the next stage only if the relevant gates pass.
Failure of a gate is a reportable result, not a bug to be
tuned away silently.
```

---

# 8. Training Protocol

```text
Alternating minimax:
    k_adv = 1 adversary step per generator step initially.
    If adv/r2 saturates at the hinge from below (adversary too weak),
    increase k_adv to 2–3 before increasing lambda_shortcut.

Warmup:
    lambda_shortcut ramps linearly from 0 to target over the first
    20% of training epochs. Never start at full strength.

Adversary capacity:
    IDENTICAL architecture to the predictability probe used in
    evaluation (hidden 256, GELU, 1 hidden layer). The mechanism and
    the measurement must see the same function class.

EMA statistics:
    mu, sigma of r12 updated with momentum 0.99, stop-grad.

Hyperparameter selection:
    validation set only; exploratory sweeps at seed 1;
    confirmatory runs across 5 seeds with frozen config;
    paired statistics against the matched no-S1 baseline.

Everything else (optimizer, lr 1e-4, batch 512, early stopping,
prototype classifier, temperatures) stays identical to the
canonical AV-MNIST protocol unless the experiment explicitly varies it.
```

---

# 9. Known Failure Modes and Required Mitigations

```text
F1  Adversarial collapse (r12 -> noise to fool q1, q2)
    Detection: variance floor triggers; effective rank crashes;
    utility loss rises.
    Mitigation: hinge tau (not raw maximization); variance floor;
    warmup; reduce lambda_shortcut. NEVER remove anti-collapse terms
    while S1 is active.

F2  Oscillating minimax
    Detection: adv/mse oscillates with period of a few epochs.
    Mitigation: lower adversary lr (0.5x generator lr); k_adv = 1.

F3  Gradient conflict between L_task and L_shortcut
    Detection: persistent negative gradient cosine.
    Mitigation: report the conflict; do NOT resolve by silently
    upweighting either side; consider per-sample hinge.

F4  Adversary weaker than the evaluation probe
    Detection: training-time adv/r2 low but evaluation probe R² high.
    Mitigation: capacity matching rule (§8); retrain probe each eval.

F5  Honest-zero misread as failure
    On data without conditional complementarity, the correct output
    IS "no dependence". Report gate behavior and utility ≈ 0 as the
    method working, not failing.

F6  Capacity confound
    Adversaries/predictors are training-only. Confirm: parameter
    count at inference identical to baseline; state it in the report.
```

---

# 10. Experiment Log Discipline

Every experiment appends to `EXPERIMENTS.md` with:

```text
Research question
Hypothesis
Root cause addressed          (which escape route from §2)
Stage and gate targeted
Files changed
Mathematical change           (exact loss, exact weights)
Parameter-count change        (train vs inference, separately)
Protocol                      (seeds, splits, selection rule)
Training-time meters          (adv/r2 trajectories, hinge saturation)
Validation result
Test result
Dependence diagnostics
Conditional utility diagnostics
Representation diagnostics    (variance, rank, cosines, R² probes)
Statistical result            (paired, 5 seeds for confirmatory)
Interpretation
Failure mode                  (from §9 if applicable)
Decision
Next experiment
```

Negative results are mandatory content. Never remove failed
objectives from the log (the retired P2 ranking loss stays
documented forever).

---

# 11. Terminology (unchanged, enforced)

```text
fused representation     any F(r1, r2); no complementarity claim
active interaction       Var(r12) > 0
predictive interaction   I(Y; r12) > 0 approximately
dependent interaction    D1 > 0 and D2 > 0 (both modalities)
useful interaction       Perf(full) > Perf(low)
complementary            dependent AND useful
synergy                  complementary with convincing evidence
```

Never infer synergy from variance, rank, cosine, standalone probes,
or shuffle drops alone. Effective rank is a collapse diagnostic only.

---

# 12. What Not To Claim

```text
- Do not claim "we recover true synergy" or "we guarantee unique
  information". Claim: "we reduce single-modality predictability
  while preserving conditional utility".

- Do not claim superiority on AV-MNIST. It is a negative-control /
  shortcut benchmark.

- Do not claim the residual representation equals PID-style unique
  information. It is an operational residual.

- Do not claim any "first" without systematic literature
  verification (synergy, unique information, adversarial
  de-redundancy all have adjacent literature).

- Do not present adv/r2 improvements as dependence improvements
  without the behavioral metrics (D1, D2, gain12) agreeing.
```

Preferred language: "we observe", "we empirically find",
"we operationalize", "we encourage".

---

# 13. Implementation Map (this repository)

Extend, do not rewrite:

```text
src/modules/models/synergyformer.py
    - add AdversarialPredictor (MLP, capacity = eval probe)
    - add shortcut_hinge_loss(t, q1(r1), q2(r2), tau)
    - add factor_variance_loss(u1, u2)
    - add ResidualPredictor + residual composition head
    - keep: conditional_utility_loss, variance_loss,
      covariance_loss, pair_cross_covariance_loss
    - RETIRED: dependence_ranking_loss (keep code + comment
      "failed on AV-MNIST, do not reuse", for reproducibility)

src/experiments/synthetic/                 (NEW)
    - synergy_datamodule.py   (§5.1 generator, configs A/B/C)
    - synergy_experiment.py   (same diagnostics + selectivity)

src/experiments/av_mnist/synergyformer.py
    - add cfg.lambda_shortcut, cfg.shortcut_tau, cfg.k_adv,
      cfg.adv_warmup_fraction, cfg.lambda_factorvar,
      cfg.lambda_pred, cfg.lambda_residual
    - alternate optimizer steps (manual optimization or two
      Lightning optimizers; adversary params in a separate group)

configs/av_mnist.yaml
    - new keys with defaults OFF (lambda_shortcut: 0.0)

configs/synthetic.yaml                     (NEW)

tests/
    - test_complementarity.py (NEW, see §14)
    - extend test_synergyformer.py
```

---

# 14. Mandatory Unit Tests

For every new loss/module:

```text
- correct shapes, finite outputs, finite gradients
- batch size 1 works
- gradient flows INTO r12/encoders from L_shortcut
- NO gradient flows into q1, q2 from the generator step
- NO gradient flows into r12 from the adversary step
- hinge behavior: penalty = 0 when adversary MSE >= tau
- stop-grad statistics: mu, sigma never receive gradients
- utility loss: full better -> zero penalty; worse -> positive
- residual: r12^R == r12 - sg(q(r1,r2)) numerically
- adversary overfit sanity: on synthetic Config A with a frozen
  shortcut model, q1 reaches R² ≈ 1 (validates the meter)
- synthetic Config B: model converges to utility ≈ 0 behavior
```

---

# 15. Reproducibility (unchanged, enforced)

Every official run saves: dataset version, split, seed, config,
train/inference parameter counts, git commit, git working-tree state
(save diff if dirty), best epoch, checkpoint, validation metric,
test metrics, runtime, hardware, and the full training-time meter
trajectories (`adv/r2_1`, `adv/r2_2`, hinge saturation).

Confirmatory runs: 5 seeds, mean ± sample std, paired comparisons
against the matched baseline, individual seed values reported.
No confirmatory claim from exploratory sweeps.

---

# 16. Success Criteria for the Whole Program

The program succeeds when it can demonstrate, on synthetic data AND
on at least one genuine multimodal benchmark, that:

```text
1. With synergy present in the data, the S1 objective produces
   interaction representations that pass GATE D + GATE U + GATE H,
   where the same architecture without S1 fails GATE D.

2. With synergy absent (negative controls), the method reports
   absence honestly instead of fabricating dependence.

3. Inference-time capacity is unchanged relative to the baseline.
```

The paper-level claim, if and only if the above holds:

> Higher-order alignment objectives alone permit unimodal shortcuts.
> A representation-level constraint — explicit penalization of
> single-modality predictability, validated against ground-truth
> synergy — produces interaction representations that empirically
> depend on, and add information beyond, their constituent modalities.

---

# 17. Final Rule

When choosing between:

```text
a higher score on a benchmark
```

and

```text
stronger evidence that the representation knows something
the individual modalities do not already know
```

choose the second, always. And when the data contains nothing more
than the modalities already know, the correct scientific output of
this project is to **measure and report exactly that** — with the
same rigor as a positive result.
