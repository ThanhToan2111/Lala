# v6.1-C1 MOSEI calibration confirmation

This is a calibration-only extension. No model, representation, historical threshold, or prior label was changed.

Observed frozen validation J mean: `0.01982565` from `[0.021956249140203, 0.019046952947974205, 0.01847374252974987]`.
Null: independently shuffled source modality within each split; B=50.

## Null summary

| Mean | Std | Median | q90 | q95 | q99 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01242 | 0.00522 | 0.01160 | 0.01974 | 0.02031 | 0.02202 | 0.02299 |

## Empirical exceedance probabilities

- `p_hat(J >= 0.01) = 0.64706` with +1 correction.
- `p_hat(J >= observed_J) = 0.09804` with +1 correction.

## Decision-independent interpretation

MOSEI is **operational positive under the frozen predictor comparison but not separated from this alignment-breaking null**.
The historical `tau=0.01` remains an operational historical threshold, not a universal discovery threshold. This null preserves the v6.0 alignment-breaking procedure and does not establish PID, causality, or universal statistical significance.
