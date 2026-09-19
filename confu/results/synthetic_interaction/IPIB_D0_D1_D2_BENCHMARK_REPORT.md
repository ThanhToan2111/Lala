# ConFu++ v4.2 IPIB D0/D1/D2 benchmark

Run date: 2026-09-18. Benchmark: Identifiable Predictive-Interaction Benchmark (IPIB). Direction: `12 -> 3`. Protocol: five seeds (`1..5`), train/validation/test sizes `20,000/5,000/5,000`, latent dimension 32, source and target dimension 64, rank 64, AdamW, validation early stopping, and frozen linear downstream probes.

## Executive conclusion

The v4.2 experiment answers the identifiability question positively but does not yet establish D2 as the best discovery target.

- The old IID synthetic benchmark is correctly retained as a non-identifiable control (NIC), not used as evidence for discovery.
- IPIB is identifiable: the target contains a controlled `z1 ⊙ z2` component, and the matched joint predictor has positive validation advantage in I1–I4.
- D2 is substantially more selective than D0: it recovers the known joint component and beats D0 in every positive regime.
- D2 passes the positive/negative-control gates: it is inactive in I0 and has positive target recovery, interaction recovery, and linear utility in I1–I3.
- D2 does not consistently beat D1. D1 is clearly better in weak and medium interaction regimes; D2 is only marginally better in strong/private regimes.
- The remaining bottleneck is not identifiability or rank. It is downstream linear accessibility and predictor-subtraction quality: the D2 target is selective, but the learned D2 embedding does not convert all of that target quality into more useful linear label information than the direct residual target.

Therefore v4.2 is a valid discovery benchmark and a useful methodological result, but it is not a complete pass for returning to UR-FUNNY or claiming ConFu++ superiority.

## 1. What changed from v4.1

The v4.1 IID generator used independent `z1`, `z2`, and `z3` while asking whether a source pair predicts an unrelated target latent. Ridge recovery was approximately zero for all directions. That was a benchmark identifiability failure, not a model failure.

v4.2 creates the identifiable one-direction benchmark:

```text
z1, z2, u3 ~ iid Rademacher
x1 = P1 z1
x2 = P2 z2
x3 = A31 z1 + A32 z2 + beta B3 (z1 ⊙ z2) + lambda_p P3 u3
```

The private projection uses the same target subspace as the joint projection. This prevents a recovery probe from succeeding merely because private and joint information occupy different linear output subspaces. Labels are generated from `wᵀ(z1 ⊙ z2)` and are never used while training predictors or interaction embeddings.

The five regimes are:

| Regime | `beta` | Private strength | Purpose |
|---|---:|---:|---|
| I0 | 0 | 1 | Negative identifiability control |
| I1 | 0.25 | 1 | Weak interaction |
| I2 | 0.5 | 1 | Medium interaction |
| I3 | 1.0 | 1 | Strong interaction |
| I4 | 1.0 | 2 | Private-target robustness |

## 2. Predictor audit

The additive predictor is two independent MLP branches. The joint predictor has separate source projections, a product feature, and a linear output. Their parameter counts are 10,190 and 10,240, a 0.491% difference.

The joint advantage is:

```text
J = R²(q_joint, x3) - R²(q_add, x3)
```

| Regime | Additive test R² | Joint test R² | `J` | Validation `J` |
|---|---:|---:|---:|---:|
| I0 | 0.495 ± 0.002 | 0.498 ± 0.002 | 0.0033 ± 0.0004 | 0.0033 ± 0.0004 |
| I1 | 0.479 ± 0.002 | 0.501 ± 0.003 | 0.0214 ± 0.0009 | 0.0215 ± 0.0007 |
| I2 | 0.439 ± 0.002 | 0.541 ± 0.001 | 0.1018 ± 0.0017 | 0.1016 ± 0.0026 |
| I3 | 0.328 ± 0.001 | 0.662 ± 0.004 | 0.3348 ± 0.0043 | 0.3348 ± 0.0050 |
| I4 | 0.161 ± 0.002 | 0.326 ± 0.003 | 0.1655 ± 0.0020 | 0.1651 ± 0.0034 |

The validation gate `J > 0.01` correctly disables D2 in I0 and enables it in I1–I4. This avoids the false-positive behavior seen when D2 is trained against a nearly zero target.

## 3. Main five-seed result

The values below are test-set linear classification gains over the clean lower-order baseline `[x1,x2]`, in percentage points. The Oracle target is the known projected interaction `B3(z1 ⊙ z2)` and provides the achievable reference.

| Regime | Oracle | D0 original | D1 residual | D2 joint advantage |
|---|---:|---:|---:|---:|
| I0 | 0.00 ± 0.00 | 4.12 ± 1.02 | 6.12 ± 0.98 | **0.00 ± 0.00** |
| I1 | 49.99 ± 0.28 | 13.52 ± 3.56 | **39.43 ± 1.19** | 25.19 ± 3.91 |
| I2 | 49.99 ± 0.28 | 31.88 ± 1.46 | **44.83 ± 0.70** | 38.78 ± 2.34 |
| I3 | 49.99 ± 0.28 | 46.40 ± 0.88 | 47.00 ± 0.45 | **47.12 ± 2.22** |
| I4 | 49.99 ± 0.28 | 44.06 ± 2.39 | 44.66 ± 0.78 | **44.78 ± 2.34** |

Relative to D0, D2 improves by `+11.66`, `+6.90`, `+0.72`, and `+0.72` points in I1–I4. Relative to D1, D2 changes by `-14.24`, `-6.05`, `+0.12`, and `+0.13` points. D2 beats D1 in 4/5 seeds in both I3 and I4, but not in the weak/medium regimes.

## 4. Target and interaction recovery

`TargetRecovery` is the linear test R² from the discovery target to the known projected interaction. `InteractionRecovery` is the linear test R² from the learned embedding to `g12 = z1 ⊙ z2`.

| Regime | Method | Target recovery | Interaction recovery |
|---|---|---:|---:|
| I1 | D0 / D1 / D2 | 0.049 / 0.058 / **0.573** | 0.023 / **0.845** / 0.562 |
| I2 | D0 / D1 / D2 | 0.174 / 0.199 / **0.870** | 0.360 / **0.959** / 0.841 |
| I3 | D0 / D1 / D2 | 0.455 / 0.498 / **0.984** | 0.855 / **0.983** / 0.960 |
| I4 | D0 / D1 / D2 | 0.191 / 0.199 / **0.956** | 0.829 / **0.959** / 0.934 |

This is the central result. D2 is more selective with respect to the known interaction than D1, especially in I1–I2, but D1 can still produce a more linearly label-accessible embedding. Thus “target selectivity” and “downstream utility” are not the same metric.

## 5. Representation health and component sanity checks

The generated components have approximately unit variance and near-zero cross-correlation across all regimes:

- additive variance: `1.0018` mean;
- joint variance: `0.9994` mean;
- private variance: `0.9997` mean;
- absolute cross-correlations: at most about `0.003` on average;
- D2 effective rank: about `22.7–25.3` out of 64 in I1–I4.

There is no rank-collapse explanation for the D2 gap. In I0, the D2 target is set to exact zero after the validation identifiability gate, so D2 variance, rank, and label gain are all zero.

## 6. Gate decision

| Gate | Result | Evidence |
|---|---|---|
| I0 `J ≈ 0` | Pass | `0.0033 ± 0.0004` |
| I0 D2 target variance near zero | Pass | Exact zero target after validation gate |
| I0 D2 false-positive label gain | Pass | `0.00 ± 0.00` pp |
| I1–I3 `J > 0` | Pass | Positive in every regime and seed |
| I1–I3 target recovery | Pass | D2 R² `0.573/0.870/0.984` |
| I1–I3 interaction recovery | Pass | D2 R² `0.562/0.841/0.960` |
| I1–I3 positive linear utility | Pass | D2 gains `25.19/38.78/47.12` pp |
| D2 > D0 | Pass | Positive mean margin in every positive regime |
| D2 > D1 | **Partial** | Only I3/I4; D1 wins I1/I2 |
| I4 degrades less than D1 | **Not established** | D2 and D1 both drop about 2.34 pp from I3 to I4 |
| Five-seed confirmation | Pass | All I0–I4, seeds 1–5 |

## 7. Bottleneck diagnosis

The bottleneck has moved compared with v4.1:

1. **Not benchmark identifiability.** IPIB has a known predictive direction and the matched joint predictor has positive validation advantage.
2. **Not predictor capacity matching.** The parameter difference is below 1%, and train/validation/test R² are close enough to rule out a simple capacity overfit.
3. **Not representation collapse.** Component statistics and D2 effective rank are healthy.
4. **Not target selectivity.** D2 target recovery is much higher than D0/D1 in every positive regime.
5. **Primary bottleneck: downstream accessibility after subtraction.** In I1/I2, D2 isolates the predictable interaction but qjoint−qadd is a weaker linearly label-accessible carrier than the contaminated direct residual x3−qadd. D1 benefits from preserving residual signal that is not strictly identifiable joint structure.
6. **Secondary bottleneck: predictor error geometry.** The D2 target is not the ground-truth interaction; it is the difference between two learned predictors. Its target recovery is high but not perfect, and the remaining error changes the geometry seen by the fixed low-rank interaction learner.

## 8. Implementation and reproducibility correction

The first exploratory run exposed that dataset shuffling was seeded but model initialization was not. The runner now seeds predictor and interaction initialization at their construction sites. A unit test verifies repeated model initialization under the same seed. The final report uses only `results/synthetic_interaction/confirm_seeded/`.

The Ridge warnings emitted by the unregularized-looking small-alpha candidates are numerical condition warnings during probe selection; they do not indicate failed training. The final result files are valid and contain the selected validation/test metrics.

## 9. Decision and next step

Accept IPIB as the v4.2 identifiability benchmark and preserve NIC as the negative control. Do not claim that D2 is a universal replacement for D1: the current evidence supports “D2 is more selective,” not “D2 always produces the best downstream utility.”

Do not return to UR-FUNNY, add third-order discovery, cross-attention, rank sweeps, dynamic routing, or task loss yet. The smallest scientifically justified next experiment is a D2 accessibility ablation on IPIB—keeping the benchmark, rank, predictor capacity, and test protocol fixed—using only a validation-selected regression loss variant or a calibrated target projection. Any such change must first beat D1 in I1/I2 without losing I0 false-positive control or I4 robustness.

## Reproduction artifacts

Code:

- `src/datasets/identifiable_interaction.py`
- `src/experiments/synthetic_interaction/ipib.py`
- `tests/test_identifiable_interaction.py`

Final results:

- `results/synthetic_interaction/confirm_seeded/i0_no_joint_seed_{1..5}.json`
- `results/synthetic_interaction/confirm_seeded/i1_weak_seed_{1..5}.json`
- `results/synthetic_interaction/confirm_seeded/i2_medium_seed_{1..5}.json`
- `results/synthetic_interaction/confirm_seeded/i3_strong_seed_{1..5}.json`
- `results/synthetic_interaction/confirm_seeded/i4_private_noise_seed_{1..5}.json`

Commands:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_identifiable_interaction -v
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python -m src.experiments.synthetic_interaction.ipib \
  --seeds 5 --out results/synthetic_interaction/confirm_seeded
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python -m src.experiments.synthetic_interaction.ipib \
  --regime i1_weak --seeds 5 --out results/synthetic_interaction/confirm_seeded
```

