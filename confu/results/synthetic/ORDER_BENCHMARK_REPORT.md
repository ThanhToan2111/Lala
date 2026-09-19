# Controlled Synthetic Order Benchmark — Final Report

Date: 2026-09-17. Code: `src/experiments/synthetic/order3_experiment.py`,
`src/experiments/synthetic/synergy_experiment.py`, `src/modules/models/complementarity.py`.
This is the controlled benchmark mandated by the UR-FUNNY v3.1 decision and the ConFu++
Representation-Level Complementarity specification.

## Design

Three modalities observe independent latent bits `a, b, c` via fixed random prototypes + Gaussian
noise (σ=1.0), 64-dim observations, 50k train / 4k test, shared latent world across seeds.

- **Order-3 task**: `y = a ⊕ b ⊕ c` — y is independent of every single modality and of every pair
  (information-theoretic chance = 50% for any order < 3 function).
- **Order-2 control**: `y = a ⊕ b` with c a pure distractor — honesty control: nothing may
  fabricate dependence on c.

Probes are trained on train-split representations and scored on the test split (no same-set
inflation). All cells: 5 seeds unless noted.

## Headline results (order-3)

| Model | Orders used | Test accuracy |
|---|---|---|
| Additive (order-1 sum) | 1 | 0.4957 ± 0.0061 (chance) |
| Pairs-only (order-1+2, sum readout) | 1+2 | 0.5010 ± 0.0102 (chance) |
| Concat-MLP (raw concat, 2×256 hidden, 60 and 200 ep) | joint | 1.0000 ± 0.0000 |
| **SynergyFormer full (order-1+2+3)** | **1+2+3** | **1.0000 ± 0.0000** |
| full + S1 (λ=1.0, τ=0.5) | 1+2+3 | 1.0000 ± 0.0000 |

Margin over every order-restricted baseline: **+49.9 pp** — the largest possible.

## Representation diagnostics (full model, order-3)

| Metric | no S1 | +S1 |
|---|---|---|
| probe(r123 → y), held-out | 1.000 | 1.000 |
| probe(r1/r2/r3 → y), held-out | ≤ 0.51 (chance) | ≤ 0.51 (chance) |
| probe(each pair → y), held-out | ≈ 0.50 (chance) | ≈ 0.50 (chance) |
| var(r123) | 0.441 ± 0.006 | 0.443 ± 0.007 |
| R²(r1 → r123) | 0.155 ± 0.075 | **0.124 ± 0.029** |
| shuffle drop per modality | +0.505 / +0.506 / +0.505 | identical |

**Third-order dependence is perfectly balanced**: corrupting any single modality costs 0.5
accuracy (chance) — maximum observable dependence on all three modalities, exactly matching the
ground-truth structure.

## Order-2 control (honesty)

| Model | accuracy | shuffle_c drop |
|---|---|---|
| pairs_only | 1.0000 | +0.0000 |
| full | 1.0000 | **+0.0000** |
| full + S1 | 1.0000 | +0.0000 |

The third-order term stays silent when a distractor modality is present: no fabricated dependence.

## Interpretation

1. **The decisive positive result**: on genuine higher-order synergy, order-restricted models hit
   their information/readout caps (chance), while the explicit third-order multiplicative
   interaction solves the task perfectly with maximally balanced modality dependence. This is the
   "the more, the merrier" claim, demonstrated with ground truth.
2. Concat-MLP also solves it (universal approximation + capacity), so the claim is NOT "only
   SynergyFormer can" — the claim is that **order-matched explicit interaction is necessary among
   structured decompositions**, and that the failure of order-2 aggregation is a *readout* failure
   (pairs contain the bits — pair-concat probe 0.97 — yet the pair-sum + linear readout is at
   chance).
3. **S1 is honest where no shortcut exists**: identical accuracy, mildly reduced single-modality
   predictability of r123, zero collateral damage on both the order-3 task and the order-2 control.

## Addendum — M-modality ladder and two mechanistic findings

**M-ladder** (y = XOR of M bits, 5 seeds, LN-free linear readout):

| M | additive | pairs_only | full (order-M) | concat-MLP |
|---|---|---|---|---|
| 3 | 0.496 | 0.503 | **1.0000** | 1.0000 |
| 4 | 0.499 | 0.499 | **1.0000** | 1.0000 |
| 5 | 0.500 | 0.497 | **1.0000** | 1.0000 |
| 6 | 0.500 | 0.495 | **1.0000** | 0.9975 |

Order-restricted decompositions are at chance at every M; the order-matched term solves every M
with balanced maximal dependence (min single-modality shuffle drop +0.49..0.50); concat-MLP keeps
up but starts to crack at M=6. Params of the full model grow mildly (26k→160k across M=3..6).

**Finding: post-aggregation LayerNorm is a hidden higher-order term.** The first ladder version
(invalid) had additive=0.874/pairs=0.9998 on 3-parity because a final LayerNorm divides by a
cross-modal per-sample statistic, leaking higher-order signal (converged test: additive+LN = 0.75,
additive = 0.50 exactly). Any "low-order ablation" ending in LayerNorm is not low-order.

**Finding: S1's residual shortcut is an adversary gap.** On AV-MNIST long training (30 epochs),
the online adversary equilibrates at R² ≈ 0.3-0.4 (below the hinge band), while the converged
offline probe reads 0.84; validation accuracy decays after epoch 0, so early stopping is optimal,
not an artifact. Closing the gap trades directly against task information on a benchmark where the
dominant modality IS the label.

## Relation to the natural-benchmark arc

AV-MNIST (image shortcut), MOSI (text dominance), UR-FUNNY (redundant pairs): all negative-natural
results show shortcuts arise from information absence/dominance. This synthetic suite completes the
causal chain: **when higher-order information is present and observable, the interaction
architecture extracts it completely, and S1 neither breaks nor fabricates anything.**
