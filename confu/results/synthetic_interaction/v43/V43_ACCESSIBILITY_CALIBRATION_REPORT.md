# ConFu++ v4.3 accessibility diagnosis and calibration

Run date: 2026-09-18. Base revision: 7bdf392094434bf9194c6a9a8615c17a2e4d11cd. Benchmark: IPIB, direction 12-to-3, five seeds, canonical v4.2 protocol unchanged except the explicitly tested D2 distillation loss.

## Executive conclusion

v4.3 localized the D2 gap and tested the first permitted calibration:

1. A0 target-versus-embedding audit;
2. seed-1 selection of B2 standardized MSE versus canonical cosine and two small B3 variants;
3. five-seed confirmation of the selected B2 candidate.

The result is a partial improvement, not a new winner.

- A0 shows D2 target and D2 embedding have similar task-score accessibility; D2 retention is approximately one.
- The large I1/I2 gap to D1 is caused by distillation geometry/inductive bias, not a missing D2 task signal.
- B2 improves D2 by 0.252 pp in I1 with paired p=0.0283.
- B2 improves I2 by only 0.076 pp, and slightly decreases I3/I4.
- D1 remains far ahead in I1/I2.
- I0 false-positive control remains exact.

The current scientific conclusion is:

    canonical D2 is selective;
    B2 makes the embedding more target-faithful;
    neither canonical D2 nor B2 recovers D1-level weak/medium accessibility.

Do not add task loss, InfoNCE, architecture changes, rank sweeps, or third-order discovery yet.

## A0 diagnosis

The A0 artifact is:

    results/synthetic_interaction/v43/a0/a0_accessibility_audit.json

The detailed A0 report is:

    results/synthetic_interaction/v43/A0_ACCESSIBILITY_AUDIT.md

The decisive five-seed task-score R2 values are:

| Regime | D1 target | D2 target | D1 embedding | D2 embedding |
|---|---:|---:|---:|---:|
| I1 | 0.054 ± 0.005 | 0.470 ± 0.105 | 0.854 ± 0.023 | 0.498 ± 0.103 |
| I2 | 0.194 ± 0.008 | 0.858 ± 0.053 | 0.963 ± 0.001 | 0.849 ± 0.050 |
| I3 | 0.493 ± 0.009 | 0.980 ± 0.013 | 0.984 ± 0.001 | 0.960 ± 0.011 |
| I4 | 0.194 ± 0.009 | 0.955 ± 0.023 | 0.962 ± 0.001 | 0.942 ± 0.020 |

D2 target accessibility is higher than D1 target accessibility, and D2 embedding retains its own task-score recovery. D1's nonlinear interaction learner extracts much more task-relevant geometry from a residual target whose linear target score is weak. This identifies the gap as a distillation-geometry problem relative to D1, while also showing that canonical D2 itself is not suffering catastrophic target-to-embedding loss.

## B-branch selection

The D2 target remains exactly:

    d = q_joint - q_add

Only the interaction loss changes.

| Variant | Loss |
|---|---|
| B1 / canonical D2 | cosine regression |
| B2 | MSE after batch standardization of prediction and train-standardized target |
| B3-0.01 | cosine plus 0.01 standardized MSE |
| B3-0.1 | cosine plus 0.1 standardized MSE |

Seed-1 validation target recovery selected B2 in every positive regime:

| Regime | B1 validation R2 | B2 validation R2 | B3-0.01 | B3-0.1 |
|---|---:|---:|---:|---:|
| I1 | 0.8480 | **0.8691** | 0.8486 | 0.8512 |
| I2 | 0.9490 | **0.9690** | 0.9503 | 0.9540 |
| I3 | 0.9638 | **0.9814** | 0.9672 | 0.9747 |
| I4 | 0.9566 | **0.9778** | 0.9589 | 0.9649 |

Selection used validation target recovery only. Labels and task score were not used to select B2.

## Five-seed B2 confirmation

Values are test linear gains over lower-order evidence, in percentage points.

| Regime | Canonical D2 | B2 standardized MSE | D1 | B2 minus D2 |
|---|---:|---:|---:|---:|
| I0 | 0.00 ± 0.00 | **0.00 ± 0.00** | 6.12 ± 0.98 | 0.00 |
| I1 | 25.19 ± 3.91 | **25.44 ± 3.85** | **39.43 ± 1.19** | +0.252 ± 0.168 |
| I2 | 38.78 ± 2.34 | **38.86 ± 2.49** | **44.83 ± 0.70** | +0.076 ± 0.215 |
| I3 | **47.12 ± 2.22** | 47.07 ± 2.29 | 47.00 ± 0.45 | -0.048 ± 0.091 |
| I4 | **44.78 ± 2.34** | 44.69 ± 2.25 | 44.66 ± 0.78 | -0.092 ± 0.187 |

Paired B2 minus canonical D2 confidence results:

| Regime | Mean difference | 95% CI | Paired p |
|---|---:|---:|---:|
| I1 | +0.252 pp | [+0.044, +0.460] | 0.0283 |
| I2 | +0.076 pp | [-0.191, +0.343] | 0.4737 |
| I3 | -0.048 pp | [-0.161, +0.065] | 0.3046 |
| I4 | -0.092 pp | [-0.324, +0.140] | 0.3327 |

The I1 gain is statistically detectable in this exploratory five-seed comparison, but it is far smaller than the 14.0-point gap to D1. B2 therefore does not satisfy the strong v4.3 gate.

## Fidelity and health

Because B2 leaves the target construction unchanged, target recovery is unchanged. B2 improves embedding recovery:

| Regime | B2 target recovery | B2 embedding recovery | B2 task-score R2 |
|---|---:|---:|---:|
| I1 | 0.573 ± 0.032 | 0.570 ± 0.030 | 0.504 ± 0.107 |
| I2 | 0.870 ± 0.014 | 0.863 ± 0.014 | 0.872 ± 0.053 |
| I3 | 0.984 ± 0.010 | 0.978 ± 0.014 | 0.978 ± 0.014 |
| I4 | 0.956 ± 0.011 | 0.952 ± 0.016 | 0.959 ± 0.019 |

B2 effective rank is 18.1, 19.7, 23.4, and 22.9 for I1–I4. This is not exact rank collapse, but it is lower than canonical D2 and indicates that standardized MSE changes representation scale/rank geometry. The modest I1/I2 utility improvement is therefore not a clean accessibility win.

## Gate decision

| Gate | Status |
|---|---|
| A0 diagnosis complete | Pass |
| D2 target contains task-relevant signal | Pass |
| D2 target-to-embedding retention measured | Pass |
| B2 selected without labels | Pass |
| I0 false-positive control | Pass |
| B2 improves I1 over canonical D2 | Pass, +0.252 pp |
| B2 reaches D1 in I1/I2 | **Fail** |
| B2 improves all positive regimes | Fail |
| D2 selectivity preserved | Pass; target unchanged, embedding recovery remains high |
| v4.3 complete accessibility solution | **Not yet** |

## Bottleneck after v4.3

The evidence now separates three effects:

1. Predictor subtraction has a real weak-regime signal-to-error problem: SER is 1.584 in I1, 5.823 in I2, 37.340 in I3, and 14.753 in I4.
2. Canonical D2 does not lose most of its own task-score utility during distillation; its retention is approximately one.
3. D1 receives a favorable nonlinear inductive bias from its contaminated residual target, which B2 does not reproduce.

Thus B2 is a useful geometry diagnostic, not the final solution. Standardized MSE improves target fidelity more consistently than downstream task accessibility. The remaining gap is a fidelity-accessibility tradeoff under the current unsupervised target and low-rank interaction learner.

## Files and commands

Code:

- src/experiments/synthetic_interaction/accessibility_audit.py
- src/experiments/synthetic_interaction/distillation_calibration.py
- src/experiments/synthetic_interaction/ipib.py
- src/datasets/identifiable_interaction.py
- tests/test_identifiable_interaction.py

Results:

- results/synthetic_interaction/v43/a0/
- results/synthetic_interaction/v43/calibration_seed1/
- results/synthetic_interaction/v43/calibration_b2_final/

Checks:

    PYTHONPATH=. .venv/bin/python -m unittest tests.test_identifiable_interaction tests.test_synthetic_order tests.test_synergyformer -v

The next permitted experiment is a minimal label-free target-side diagnostic or a freeze decision for the selectivity/accessibility tradeoff. Do not return to UR-FUNNY until the v4.3 pair formulation is either improved without losing I0 control or explicitly frozen with this tradeoff documented.
