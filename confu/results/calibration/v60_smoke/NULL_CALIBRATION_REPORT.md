# v6.0 null and threshold calibration

Alignment-breaking null: independently shuffle the right/source modality within each train/validation/test split; B=1 replicates per setting.

## Null distributions

| Setting | Mean | Std | Median | q95 | q99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| `IPIB_I0` | 0.02379 | 0.00000 | 0.02379 | 0.02379 | 0.02379 | 0.02379 |
| `MOSEI_VA_to_T` | 0.01779 | 0.00000 | 0.01779 | 0.01779 | 0.01779 | 0.01779 |
| `MELD_VA_to_T` | 0.00887 | 0.00000 | 0.00887 | 0.00887 | 0.00887 | 0.00887 |

## Interpretation

The historical gate τ=0.01 is preserved. Threshold sensitivity at 0.005 and 0.02 is descriptive and does not change historical labels. The shuffle is an operational alignment-breaking null; it does not preserve all lower-order dependencies.

See `threshold_sensitivity.json` for MOSEI, MELD, and MUStARD historical values.
