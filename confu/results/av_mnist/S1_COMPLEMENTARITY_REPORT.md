# S1 Adversarial De-Shortcutting — Benchmark Report

Date: 2026-09-17. Method and gates: `AGENT.md — ConFu++ Representation-Level Complementarity.md`.
Code: `src/modules/models/complementarity.py`, `src/experiments/av_mnist/complementarity.py`,
`src/experiments/synthetic/synergy_experiment.py`. Tests: `tests/test_complementarity.py` (9/9 pass).

## What S1 is

During training, two MLP adversaries (capacity-matched to the evaluation probe) try to predict the
EMA-standardized interaction `t = std(r12)` from `r1` alone and from `r2` alone (alternating minimax,
k_adv=1, adversary lr 0.5×). The generator pays a hinged penalty `ReLU(τ − MSE(ŝingle))`, τ = 0.5,
with 10% warmup, plus a per-factor variance floor (S2). Adversaries are training-only
(+263k params at train time, 0 at inference). Joint predictability from `(r1, r2)` is never penalized.

## Stage 0 — synthetic ground truth (5 seeds, fixed latent world)

| Config | Baseline (S1 off) | S1 (λ=1.0) |
|---|---|---|
| A: y = a⊕b (pure synergy) | acc .999, gain12 +.499, both shuffle drops +.5, sel +.432 | acc 1.000, gain12 +.501, R²(r2→r12) .407→.202 |
| B: y = a (no synergy) | acc 1.000, drops ≈ 0, sel .000 | identical — no fabricated dependence |
| C: 50% y=a / 50% y=a⊕b | acc .748 (a-only floor), gain12 −.000 | acc .750, sel +.010→+.038, XOR fraction NOT recovered |

Gates: soundness (A) PASS, honesty (B) PASS, shortcut-stress (C) PARTIAL — S1 alone at this strength
does not overcome easy-signal dominance in a mixture task.

NOTE: an earlier data-generator bug (train/test prototype mismatch) invalidated the first synthetic
runs; it was found via a below-chance test anomaly and fixed. All numbers above are post-fix.

## Stage 1 — AV-MNIST negative control (5 seeds, paired vs same-codepath λ=0 ablation)

| Metric | Ablation λ=0 | S1 λ=0.1 | Paired Δ |
|---|---|---|---|
| accuracy (%) | 69.962 ± 0.594 | 70.046 ± 0.476 | +0.084 ± 0.219 (ns) |
| R²(r1 → r12) | **0.989 ± 0.002** | **0.682 ± 0.155** | **−0.307 ± 0.155 (t=−4.43, p≈.011)** |
| R²(r2 → r12) | 0.029 ± 0.006 | 0.157 ± 0.096 | +0.128 |
| audio shuffle drop (pp) | +0.084 ± 0.069 | +0.236 ± 0.147 | +0.152 (t=+2.62, p≈.059) |
| image shuffle drop (pp) | +1.244 ± 0.604 | +0.974 ± 0.637 | ns |
| gain12 (pp) | +0.188 ± 0.216 | +0.174 ± 0.357 | ns |
| zero drop (pp) | +0.188 ± 0.256 | +0.220 ± 0.349 | ns |
| net correction | +18.8 ± 21.6 | +17.4 ± 35.7 | ns |
| linear / nonlinear probe gain (pp) | −0.02 / +0.23 | −0.79 / −0.46 | lower |

Canonical baselines for accuracy reference (5 seeds): Additive 70.448 ± 0.522,
Concat+MLP 70.518 ± 0.453, ConFu-style 70.886 ± 0.578, previous SynergyFormer 70.762 ± 0.338,
P1 utility 70.296 ± 0.545.

## Interpretation

1. **S1 is the first objective in this project that significantly reduces the single-modality
   shortcut** (image predictability 0.99 → 0.68, 5/5 seeds negative), at zero accuracy cost.
   Neither P1 (utility) nor P2 (ranking, retired) ever moved R²(r1 → r12).
2. Audio dependence doubles (borderline significant) — but stays small in absolute terms, exactly
   as expected on a benchmark where audio carries almost no conditional label information.
3. No superiority claim on AV-MNIST (per protocol it is a negative-control benchmark). The accuracy
   ranking across methods on this benchmark is dominated by the fusion-implementation family, and
   early stopping selects an epoch ~0–5 checkpoint, which bounds how much any training-time
   objective can act.
4. The reduced conditional probe gains indicate the hinge removes redundant content that parameter-
   matched probes could exploit; whether this is pure redundancy or useful-but-redundant content
   must be answered on a benchmark with genuine headroom.

## Decision and next step

Accept S1 as a validated, honest representation-level mechanism. Per the MOSI audit decision and
Stage-0 Config-C result, the next positive demonstration requires a benchmark with validated
lower-order headroom; Config-C-style synthetic mixture with stronger dependence objectives
(e.g., necessity-aware margins or curriculum on the hard fraction) is the cheapest next test.

## Addendum — S4 hard-fraction emphasis, Config-C redesign, noise ladder

**Config-C bug (important)**: the original mixture config made the rule label unobservable
(Bayes-optimal 0.75); all earlier "shortcut floor" numbers were memorization artifacts and have been
discarded. The redesigned Config C carries an observable routing flag in modality 2
(flag → y=a, else y=a⊕b; Bayes = 1.0).

| Setting | baseline | S1 | S4 | S1+S4 |
|---|---|---|---|---|
| C rho=0.5, 60 ep | 0.976 | 0.976 | 0.976 | 0.975 |
| C rho=0.9, 60 ep | 0.987 | 0.988 | 0.986 | 0.988 |
| C rho=0.9, 6 ep | **0.990** | 0.990 | 0.983 | 0.982 |
| — xor-fraction only | 0.928 | 0.933 | **0.977** | 0.973 |
| AV-MNIST seed 1 | 69.34 (λ=0) | 69.54 | 68.98 | 69.63 (R²r1 ↓ 0.61) |

**Noise ladder** (Config C rho=0.7, X2 noise 1.5→3.0): the baseline interaction degrades gracefully
(xor-fraction 0.958→0.903) but never abandons the synergy cue; S1+S4 shifts the operating point
toward the hard fraction (xor +1.5 pp at noise 2.0) at a small easy-fraction cost.

## Final conclusion of this research arc

1. **S1 is validated**: the only objective that significantly reduces single-modality shortcut
   predictability (AV-MNIST R²(r1→r12): 0.989→0.682, paired p≈0.011, zero accuracy cost;
   audio dependence doubled, p≈0.059). Sound on ground truth (Config A), honest on negative
   controls (Config B and AV-MNIST).
2. **The ConFu shortcut is a data property, not an architecture/optimization failure**: whenever
   the synergy cue exists and is observable (synthetic suite), the plain low-rank multiplicative
   interaction learns it without any auxiliary objective; whenever the cue is absent or dominated
   (AV-MNIST, MOSI, UR-FUNNY), no objective can create it. Demonstrated on three natural
   benchmarks and a controlled synthetic suite.
3. **S4 is exploratory-negative**: it redistributes capacity toward hard samples and nets negative
   unless hard samples are both under-served and learnable.
4. "Beating the baseline" on shortcut benchmarks is the wrong target (spec §17). S1 delivers the
   honest Pareto improvement: equal accuracy, strictly less shortcut, +263k training-only params,
   zero inference cost.

Next justified step (unchanged): a genuine multimodal benchmark with validated lower-order
headroom (Bird-MML), with S1 applied from the start.
