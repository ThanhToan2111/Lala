# v6.0 null and threshold calibration

Alignment-breaking null: independently shuffle the right/source modality within each train/validation/test split; B=10 replicates per setting.

The historical gate is τ=0.01. B=10 is a compute-limited calibration replicate count; it is reported as a finite-sample audit, not as a replacement for a larger preregistered null.

## Null distributions

| Setting | Mean | Std | Median | q95 | q99 | Max | q99 > τ |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `IPIB_I0` | -0.02034 | 0.00104 | -0.02057 | -0.01887 | -0.01846 | -0.01836 | no |
| `MOSEI_VA_to_T` | 0.01227 | 0.00563 | 0.01113 | 0.02118 | 0.02263 | 0.02299 | yes |
| `MELD_VA_to_T` | 0.00374 | 0.00154 | 0.00351 | 0.00557 | 0.00571 | 0.00574 | no |

## Interpretation

The historical gate τ=0.01 is preserved for comparability; it is not recalibrated from these nulls. Since MOSEI q99 exceeds τ, the gate is not a 1% false-positive control under this setting and should be treated as descriptive until a larger preregistered null is run. Threshold sensitivity at 0.005 and 0.02 is descriptive and does not change historical labels. The shuffle is an operational alignment-breaking null; it does not preserve all lower-order dependencies.

See `threshold_sensitivity.json` for MOSEI, MELD, and MUStARD historical values.
