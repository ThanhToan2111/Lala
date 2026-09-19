# ConFu++ v4.4 pair-discovery freeze report

Run date: 2026-09-18. Benchmark: IPIB, direction `12 -> 3`, five seeds for the canonical D0/D1/D2 and C0 experiments. C1 was intentionally stopped after seed 1 because its required practical gate failed.

## Executive decision

The synthetic pair-discovery phase is frozen.

The final method is **Joint-Advantage Distillation (canonical D2)**. It is the best scientific choice because it identifies the interaction through a capacity-matched joint-predictor advantage, preserves the I0 negative control, and is substantially more selective than D0/D1. It is not the best downstream-accessibility method in weak and medium regimes: D1 remains stronger there.

The v4.4 reliability mechanism is rejected as a promotion candidate. C0 passed for the intended weak/medium regimes, but C1 did not produce a meaningful seed-1 validation improvement. Per the v4.4 decision rule, no C1 five-seed confirmation, alpha sweep, C2, or further synthetic pair hyperparameter search is justified.

## Architecture findings

The IPIB architecture is adequate for the question being tested:

- source and target dimensions are 64, interaction rank is 64, and the predictor parameter difference is only 0.491%;
- the joint predictor has positive validation advantage in every identifiable regime;
- D2 effective rank is approximately 22.7–25.3/64 in the canonical five-seed run;
- I0 is exactly disabled after the validation identifiability gate, preventing a false-positive interaction representation.

Therefore the remaining gap is not a missing cross-attention block, rank capacity, or representation collapse. It is the tradeoff between selective target construction and downstream linear accessibility.

## Identifiability findings

The old IID benchmark is retained only as the non-identifiable control (NIC). Its unrelated target cannot support interaction discovery. IPIB fixes this by generating:

```text
x3 = A31 z1 + A32 z2 + beta B3 (z1 ⊙ z2) + lambda_p P3 u3
```

The validation gate uses:

```text
J = R²(q_joint, x3) - R²(q_add, x3)
```

Labels and the ground-truth interaction are not used to construct predictors or discovery targets.

## Canonical D0/D1/D2 result

Values are test linear-classification gains over `[x1,x2]`, in percentage points, mean ± sample standard deviation over five seeds.

| Regime | Oracle | D0 original | D1 residual | D2 joint advantage |
|---|---:|---:|---:|---:|
| I0 no joint | 0.00 ± 0.00 | 4.12 ± 1.02 | 6.12 ± 0.98 | **0.00 ± 0.00** |
| I1 weak | 49.99 ± 0.28 | 13.52 ± 3.56 | **39.43 ± 1.19** | 25.19 ± 3.91 |
| I2 medium | 49.99 ± 0.28 | 31.88 ± 1.46 | **44.83 ± 0.70** | 38.78 ± 2.34 |
| I3 strong | 49.99 ± 0.28 | 46.40 ± 0.88 | 47.00 ± 0.45 | **47.12 ± 2.22** |
| I4 private target | 49.99 ± 0.28 | 44.06 ± 2.39 | 44.66 ± 0.78 | **44.78 ± 2.34** |

D2 improves over D0 by `+11.66`, `+6.90`, `+0.72`, and `+0.72` points in I1–I4, but trails D1 by `14.24` and `6.05` points in I1/I2. This establishes D2 as a selective discovery objective, not a universal replacement for D1.

## A0 accessibility diagnosis

The v4.3 A0 audit showed that D2's target is more task-accessible than D1's target in I1/I2, and that D2 retains most of its own task-score signal after distillation. The gap is therefore not explained by a missing task signal alone.

The better explanation is inductive bias: D1's contaminated residual gives its nonlinear interaction learner a task-friendly representation, while D2's cleaner predictor difference is more selective but harder to convert into the same linearly accessible geometry. B2 standardized MSE improved target fidelity, but not the downstream accessibility gap.

## B2 result

B2 was selected without labels using validation target recovery. Five-seed test gains were:

| Regime | Canonical D2 | B2 standardized MSE | B2 − D2 |
|---|---:|---:|---:|
| I0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.000 |
| I1 | 25.19 ± 3.91 | **25.44 ± 3.85** | +0.252 |
| I2 | 38.78 ± 2.34 | **38.86 ± 2.49** | +0.076 |
| I3 | **47.12 ± 2.22** | 47.07 ± 2.29 | -0.048 |
| I4 | **44.78 ± 2.34** | 44.69 ± 2.25 | -0.092 |

B2 is a useful fidelity calibration, not a reliable accessibility solution. It is not part of the frozen method.

## C0 reliability audit

C0 is diagnostic only. Reliability is computed from validation predictor errors:

```text
E_add,k   = mean((q_add,k - x3,k)^2)
E_joint,k = mean((q_joint,k - x3,k)^2)
R_k       = max(0, E_add,k - E_joint,k) / (E_add,k + epsilon)
```

No labels, task scores, oracle interaction, or test data enter `R`.

| Regime | Pearson R–F | Spearman R–F | Positive Spearman seeds | Top-bottom fidelity gap |
|---|---:|---:|---:|---:|
| I1 weak | 0.289 ± 0.109 | **0.286 ± 0.094** | **5/5** | **+0.097 ± 0.047** |
| I2 medium | 0.753 ± 0.106 | **0.691 ± 0.132** | **5/5** | **+0.131 ± 0.039** |
| I3 strong | 0.146 ± 0.404 | 0.176 ± 0.331 | 3/5 | +0.011 ± 0.026 |
| I4 private target | 0.198 ± 0.331 | 0.142 ± 0.261 | 4/5 | +0.019 ± 0.019 |

C0 passes its intended I1/I2 reliability criterion. Reliability is non-degenerate, with zero-reliability fraction equal to zero in all regimes. This pass only justified one C1 seed-1 test; it did not justify a search.

## C1 seed-1 decision

C1 used exactly `alpha=1`, the validation-normalized reliability-weighted target, canonical D2 cosine loss, and no B2 combination.

| Regime | D2 validation gain | C1 validation gain | C1 − D2 |
|---|---:|---:|---:|
| I0 | 0.00 pp | 0.00 pp | 0.00 pp |
| I1 | 24.00 pp | 23.98 pp | -0.02 pp |
| I2 | 37.82 pp | 37.84 pp | +0.02 pp |
| I3 | 46.16 pp | 46.16 pp | 0.00 pp |
| I4 | 39.52 pp | 39.52 pp | 0.00 pp |

C1 preserves I0 control and does not visibly damage I3/I4, but it fails the practical improvement gate: neither I1 nor I2 exceeds the required improvement of more than 1 point. The observed differences are at noise level and do not warrant five-seed confirmation.

## Final frozen formulation

The frozen pair method is **Joint-Advantage Distillation / canonical D2**:

1. Fit additive predictor `q_add` and capacity-matched joint predictor `q_joint`.
2. Compute validation joint advantage `J`.
3. Set `d = q_joint - q_add` only when validation `J > 0.01`; otherwise use an exact zero target.
4. Train the rank-64 interaction representation against `d` with the canonical cosine loss.
5. Evaluate with I0 control, target recovery, interaction recovery, linear accessibility, and health diagnostics.

Frozen: predictor capacity, validation gate, target definition, cosine loss, rank, optimizer, training budget, and five-seed protocol.

Not frozen/promoted: reliability weighting, B2 standardized MSE, task-loss weighting, cross-attention, rank sweeps, third-order discovery, or UR-FUNNY tuning.

## Bottleneck and limitations

The bottleneck is now specifically **fidelity–accessibility tradeoff after predictor subtraction**:

- D2 is selective and identifiable, but D1 is more linearly accessible in weak/medium regimes;
- `q_joint - q_add` contains predictor error geometry, so it is not a clean oracle interaction;
- B2 improves target matching without recovering D1-level accessibility;
- C0 is reliable mainly in I1/I2 and weak/inconsistent in I3/I4, where fidelity ranking has little headroom or private noise interferes;
- the result is synthetic and does not establish transfer to UR-FUNNY or another natural multimodal task;
- five seeds support the direction of the result but not broad universal claims.

The correct scientific claim is therefore: **D2 is a selectivity-oriented interaction discovery objective; D1 is a less selective but more accessibility-oriented residual objective.**

## Real-world entry criteria

The next phase is screening, not immediate ConFu++ training. First run a label-free identifiability audit on CMU-MOSEI Emotion for mappings `12→3`, `13→2`, and `23→1`, separately by emotion where possible. For every mapping, report validation/test additive R², joint R², and `J`, and retain only directions with positive, stable validation joint advantage.

Only after screening should the selected mappings compare Original ConFu, D1, and frozen canonical D2 under matched encoders and protocol. UR-FUNNY tuning and third-order expansion remain blocked until that screening produces a stable jointly identifiable direction.

## Reproduction artifacts

- Canonical benchmark: [`IPIB_D0_D1_D2_BENCHMARK_REPORT.md`](../IPIB_D0_D1_D2_BENCHMARK_REPORT.md)
- v4.3 diagnosis/calibration: [`V43_ACCESSIBILITY_CALIBRATION_REPORT.md`](../v43/V43_ACCESSIBILITY_CALIBRATION_REPORT.md)
- C0 report: [`C0_RELIABILITY_AUDIT.md`](C0_RELIABILITY_AUDIT.md)
- C0 machine artifact: `c0_reliability_audit.json`
- C1 seed-1 machine artifact: `c1_seed1/seed_1.json`
- Reliability audit: `src/experiments/synthetic_interaction/reliability_audit.py`
- C1 runner: `src/experiments/synthetic_interaction/reliability_weighted.py`
- Tests: `tests/test_identifiable_interaction.py`

The required pair-phase decision is complete: **freeze synthetic pair discovery and move to real-world identifiability screening.**
