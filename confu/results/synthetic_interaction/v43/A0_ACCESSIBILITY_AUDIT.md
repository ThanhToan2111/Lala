# ConFu++ v4.3 A0 accessibility audit

Run date: 2026-09-18. This audit uses the canonical v4.2 IPIB generator, predictor architectures, optimizer, interaction architecture, seeds, and frozen downstream probe protocol. It evaluates the existing D0/D1/D2 target and embedding geometry for the 12-to-3 direction; labels and the synthetic task score are used only after representation training.

## Decision

The v4.2 D2 target is not missing the task direction in the primary weak/medium regimes. The D2 target itself has higher linear task-score recovery and label accessibility than the D1 target, and the D2 embedding retains approximately all of its own target-level task-score recovery.

The gap to D1 appears because D1 distillation produces a more task-accessible geometry than its target-level linear probe suggests. In I1 and I2:

- D1 target task-score R2 is only 0.054 ± 0.005 and 0.194 ± 0.008;
- D1 embedding task-score R2 rises to 0.854 ± 0.023 and 0.963 ± 0.001;
- D2 target task-score R2 is 0.470 ± 0.105 and 0.858 ± 0.053;
- D2 embedding remains 0.498 ± 0.103 and 0.849 ± 0.050.

Therefore the next branch is distillation calibration (B2/B3), not target calibration (A1/A2/A3). This does not mean D2 loses all information during distillation: D2 retention is close to one. It means canonical D2 cosine regression does not create the same favorable task-direction geometry that D1 obtains from its residual target.

## A0 protocol

For every IPIB sample the audit retains:

    g       = z1 elementwise-multiplied-by z2
    g_tilde = B3 g
    s_y     = w transposed-times g

The audited features are:

    g, g_tilde,
    D0 target, D1 target, D2 target,
    D0 embedding, D1 embedding, D2 embedding

For every feature the audit reports:

- Ridge R2(feature to g);
- Ridge R2(feature to g_tilde);
- Ridge R2(feature to s_y);
- validation-selected linear label accuracy;
- lower-order augmented accuracy and delta;
- feature health;
- D1/D2 target-to-embedding retention.

All probes select hyperparameters on validation and use test only for final reporting. The A0 artifact is:

    results/synthetic_interaction/v43/a0/a0_accessibility_audit.json

## Full interaction recovery versus task-score recovery

Values are five-seed mean ± sample standard deviation. The table reports R2(feature to g12) / R2(feature to s_y).

| Regime | D1 target | D2 target | D1 embedding | D2 embedding |
|---|---:|---:|---:|---:|
| I1 | 0.057 ± 0.001 / 0.054 ± 0.005 | 0.524 ± 0.029 / 0.470 ± 0.105 | 0.845 ± 0.015 / 0.854 ± 0.023 | 0.562 ± 0.026 / 0.498 ± 0.103 |
| I2 | 0.198 ± 0.002 / 0.194 ± 0.008 | 0.845 ± 0.013 / 0.858 ± 0.053 | 0.959 ± 0.001 / 0.963 ± 0.001 | 0.840 ± 0.014 / 0.849 ± 0.050 |
| I3 | 0.496 ± 0.002 / 0.493 ± 0.009 | 0.979 ± 0.013 / 0.980 ± 0.013 | 0.983 ± 0.001 / 0.984 ± 0.001 | 0.960 ± 0.012 / 0.960 ± 0.011 |
| I4 | 0.198 ± 0.002 / 0.194 ± 0.009 | 0.945 ± 0.016 / 0.955 ± 0.023 | 0.959 ± 0.001 / 0.962 ± 0.001 | 0.934 ± 0.015 / 0.942 ± 0.020 |

D2 is substantially more faithful to the complete interaction than D1. In I1/I2, however, D1's learned embedding is more task-direction accessible despite its weaker target-level linear recovery.

## Label accessibility

The lower-order [x1,x2] probe is approximately chance (about 50%). Values below are feature-only label accuracy; the final column is the accuracy gain after concatenating the feature to [x1,x2].

| Regime | D1 target | D2 target | D1 embedding | D2 embedding |
|---|---:|---:|---:|---:|
| I1 | 57.6% / +7.4 pp | 73.9% / +24.7 pp | 89.1% / +39.4 pp | 74.9% / +25.2 pp |
| I2 | 64.7% / +14.7 pp | 87.7% / +38.5 pp | 94.4% / +44.8 pp | 88.4% / +38.8 pp |
| I3 | 74.8% / +25.0 pp | 95.6% / +46.5 pp | 96.6% / +47.0 pp | 96.5% / +47.1 pp |
| I4 | 64.6% / +14.9 pp | 93.2% / +43.8 pp | 94.3% / +44.7 pp | 94.3% / +44.8 pp |

This explains the v4.2 result: D2 target and embedding are internally consistent, but D1's distillation path yields a better downstream representation in I1/I2.

## Distillation retention

Retention is computed as:

    (Perf(embedding) - Perf(lower)) /
    (Perf(target) - Perf(lower))

using task-score R2 as the primary diagnostic and label accuracy as a secondary diagnostic.

| Regime | D1 task-score retention | D2 task-score retention | D1 label retention | D2 label retention |
|---|---:|---:|---:|---:|
| I1 | 15.18 ± 1.88 | 1.06 ± 0.03 | 4.996 ± 0.20 | 1.04 ± 0.02 |
| I2 | 4.91 ± 0.22 | 0.99 ± 0.01 | 2.99 ± 0.06 | 1.02 ± 0.01 |
| I3 | 1.99 ± 0.04 | 0.98 ± 0.00 | 1.87 ± 0.06 | 1.02 ± 0.00 |
| I4 | 4.91 ± 0.24 | 0.99 ± 0.00 | 2.98 ± 0.07 | 1.02 ± 0.00 |

D1 retention above one is not a claim that D1 preserves a target direction linearly. It indicates that the nonlinear low-rank interaction learner extracts task-relevant structure from the residual target that the linear target probe does not expose. D2 retention near one means canonical D2 does not catastrophically destroy its own target utility.

## Predictor error geometry

The D2 target is q_joint minus q_add. The signal-to-error ratio is:

    Var(beta times joint) / Var(D2 target - beta times joint)

| Regime | D2 signal/error ratio | q_joint to joint R2 | D2 target to joint R2 |
|---|---:|---:|---:|
| I1 | 1.584 ± 0.108 | 0.300 ± 0.017 | 0.573 ± 0.032 |
| I2 | 5.823 ± 0.489 | 0.608 ± 0.008 | 0.870 ± 0.014 |
| I3 | 37.340 ± 9.246 | 0.843 ± 0.008 | 0.984 ± 0.010 |
| I4 | 14.753 ± 2.611 | 0.822 ± 0.007 | 0.956 ± 0.011 |

The expected I1 < I2 < I3 pattern holds. Weak interaction is genuinely harder because predictor-error geometry is comparable to the joint signal. This explains why D2 target fidelity is lower in I1, but does not by itself explain why D1's interaction learner produces much higher task-score recovery. That remaining gap is the B-branch distillation geometry question.

## Decision tree result

| Question | Result |
|---|---|
| Does D2 target contain task-relevant structure? | Yes; target task-score R2 is positive and above D1 in I1-I4 |
| Does D2 embedding retain its target task utility? | Yes; retention is approximately 1 |
| Does D1 embedding outperform D2 embedding in I1/I2? | Yes, strongly |
| Is target calibration the first intervention? | No |
| Is distillation calibration justified? | Yes; run B2/B3 |
| Is label-aware training allowed? | No |

## Next experiment

Keep D2 target construction fixed. Compare:

1. B1 canonical cosine regression (already the v4.2 control);
2. B2 standardized MSE;
3. B3 cosine plus one small fixed MSE coefficient selected on seed-1 validation.

Do not add InfoNCE, task loss, whitening, reliability weights, architecture changes, rank sweeps, or third-order discovery before this comparison.
