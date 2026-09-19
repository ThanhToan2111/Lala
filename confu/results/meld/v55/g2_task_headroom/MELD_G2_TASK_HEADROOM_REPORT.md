# MELD G2 task headroom

Frozen utterance-level features; seven-class emotion CE; validation NLL early stopping; three screening seeds.

| Pair | Joint−Add Macro F1 val | NLL improvement val | Acc val | Weighted F1 val | G2 |
|---|---:|---:|---:|---:|---|
| `VA` | 5.985 ± 0.723 pp | -0.24498 ± 0.05255 | -12.395 ± 3.520 pp | 0.830 ± 1.431 pp | **WEAK_UNSTABLE** |
| `VT` | -1.282 ± 0.261 pp | -0.12832 ± 0.02069 | -2.677 ± 0.407 pp | -1.964 ± 0.819 pp | **FAIL** |
| `AT` | 0.579 ± 1.097 pp | -0.04244 ± 0.01185 | -1.143 ± 0.544 pp | -0.438 ± 0.857 pp | **WEAK_UNSTABLE** |

Strong G2 requires joint-vs-additive validation Macro F1 positive in all three seeds and positive mean NLL improvement. Test values are reporting-only.

## Test reporting

| Pair | Joint−Add Macro F1 test | NLL improvement test | Acc test | Weighted F1 test |
|---|---:|---:|---:|---:|
| `VA` | 4.723 ± 0.117 pp | -0.29855 ± 0.04930 | -16.028 ± 2.790 pp | -2.677 ± 0.819 pp |
| `VT` | -0.179 ± 1.232 pp | -0.20400 ± 0.04736 | -2.401 ± 1.163 pp | -1.299 ± 1.242 pp |
| `AT` | -0.892 ± 0.886 pp | -0.06931 ± 0.00824 | -2.146 ± 0.369 pp | -1.680 ± 0.840 pp |

## Main test metrics

| Pair | Model | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | NLL |
|---|---|---:|---:|---:|---:|
| `VA` | `linear` | 37.050 ± 1.899 | 17.083 ± 0.521 | 34.266 ± 0.515 | 1.7370 ± 0.0201 |
| `VA` | `additive` | 45.888 ± 1.622 | 12.281 ± 0.533 | 33.577 ± 0.201 | 1.5927 ± 0.0211 |
| `VA` | `joint_product` | 29.860 ± 1.246 | 17.004 ± 0.645 | 30.900 ± 0.690 | 1.8913 ± 0.0292 |
| `VA` | `concat_mlp` | 45.951 ± 0.876 | 11.651 ± 0.790 | 33.071 ± 0.919 | 1.5855 ± 0.0117 |
| `VT` | `linear` | 54.061 ± 0.400 | 28.232 ± 0.155 | 50.366 ± 0.291 | 1.4942 ± 0.0230 |
| `VT` | `additive` | 56.450 ± 0.808 | 26.882 ± 1.227 | 50.341 ± 1.118 | 1.3224 ± 0.0052 |
| `VT` | `joint_product` | 54.049 ± 0.721 | 26.703 ± 0.621 | 49.042 ± 0.564 | 1.5264 ± 0.0459 |
| `VT` | `concat_mlp` | 54.968 ± 0.059 | 24.946 ± 0.547 | 48.071 ± 0.574 | 1.3707 ± 0.0136 |
| `AT` | `linear` | 57.931 ± 0.199 | 28.012 ± 0.128 | 51.476 ± 0.121 | 1.2663 ± 0.0031 |
| `AT` | `additive` | 58.455 ± 0.632 | 28.748 ± 0.759 | 52.512 ± 0.837 | 1.2370 ± 0.0128 |
| `AT` | `joint_product` | 56.309 ± 0.312 | 27.857 ± 0.300 | 50.833 ± 0.112 | 1.3063 ± 0.0046 |
| `AT` | `concat_mlp` | 58.467 ± 0.404 | 28.777 ± 0.357 | 52.476 ± 0.418 | 1.2365 ± 0.0021 |

## Per-seed validation primary metric

| Pair | Seed 1 Δ Macro F1 / ΔNLL | Seed 2 Δ Macro F1 / ΔNLL | Seed 3 Δ Macro F1 / ΔNLL |
|---|---:|---:|---:|
| `VA` | +5.594 / -0.2558 | +5.541 / -0.2913 | +6.819 / -0.1879 |
| `VT` | -1.477 / -0.1187 | -0.986 / -0.1142 | -1.385 / -0.1521 |
| `AT` | +1.575 / -0.0288 | -0.597 / -0.0492 | +0.758 / -0.0494 |