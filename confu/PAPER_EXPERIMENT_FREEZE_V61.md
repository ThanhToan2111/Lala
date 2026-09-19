# ConFu++ v6.1 paper experiment freeze

Frozen on 2026-09-19 after the v6.1-C1 calibration-only confirmation.

## Previous freeze

The v6.0 freeze in `PAPER_EXPERIMENT_FREEZE.md` remains binding. No architecture, representation, optimizer, pooling, rank, loss, task-aware JAD, or dataset search was reopened.

## C1 protocol

- Dataset/mapping: canonical MOSEI `VA_to_T`.
- Null: independently shuffle the right/source modality within train, validation, and test splits.
- Predictors: unchanged capacity-controlled Additive and Product models.
- Training: batch 512, 80 epochs, patience 8, AdamW `1e-3`, weight decay `1e-4`.
- Replicates: `B=50`; validation-only selection.
- Observed statistic: mean validation `J` across the three frozen MOSEI records, `0.01982565`.
- Full preregistration: `results/calibration/v61/C1_PROTOCOL.md`.

## C1 result

| Statistic | Value |
|---|---:|
| Null mean | 0.01242 |
| Null std | 0.00522 |
| Null median | 0.01160 |
| Null q90 | 0.01974 |
| Null q95 | 0.02031 |
| Null q99 | 0.02202 |
| Null maximum | 0.02299 |
| `p_hat(J >= 0.01)` | 0.64706 |
| `p_hat(J >= observed_J)` | 0.09804 |

## Updated wording

MOSEI remains operationally positive under the frozen predictor comparison, but is not separated from the alignment-breaking null in this C1 confirmation. The historical `J=0.01` threshold remains unchanged for comparability and must not be described as a calibrated discovery threshold.

This is a calibration confirmation, not a PASS/FAIL method result. It strengthens the paper's uncertainty statement and does not invalidate the framework or the Naturalized IPIB construct-validity result.
