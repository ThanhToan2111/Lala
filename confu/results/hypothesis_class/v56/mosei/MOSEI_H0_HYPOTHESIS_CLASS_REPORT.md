# MOSEI H0 hypothesis-class audit

Frozen canonical pooled representations; additive, capacity-matched Product Joint, and one-hidden-layer Generic Joint MLP.

Dataset metadata: `{'cache': True, 'label_alignment_sentiment_match': 1.0}`.

## Parameter audit

| Mapping | Additive | Product | Generic MLP | MLP mismatch vs additive |
|---|---:|---:|---:|---:|
| `VA_to_T` | 25,602 | 25,635 | 25,324 | 1.086% |

## Main validation result

| Mapping | Additive R² | Product R² | MLP R² | J Product | J MLP | ΔH |
|---|---:|---:|---:|---:|---:|---:|
| `VA_to_T` | 0.0106 ± 0.0019 | 0.0304 ± 0.0018 | 0.0065 ± 0.0045 | 0.0198 ± 0.0019 | -0.0041 ± 0.0027 | -0.0239 ± 0.0044 |

## Seed-level validation result

| Mapping | Seed | J Product | J MLP | ΔH |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | 0.0220 | -0.0064 | -0.0283 |
| `VA_to_T` | 2 | 0.0190 | -0.0049 | -0.0239 |
| `VA_to_T` | 3 | 0.0185 | -0.0010 | -0.0195 |

## Descriptive test result

| Mapping | J Product | J MLP | ΔH |
|---|---:|---:|---:|
| `VA_to_T` | 0.0231 ± 0.0018 | -0.0188 ± 0.0126 | -0.0419 ± 0.0110 |

## Overfit diagnostic

| Mapping | J MLP train | J MLP val | Gap_M |
|---|---:|---:|---:|
| `VA_to_T` | -0.0161 ± 0.0087 | -0.0041 ± 0.0027 | -0.0120 ± 0.0091 |

## Training health

| Mapping | Predictor | Train R² | Val R² | Test R² | Best epoch | Finite |
|---|---|---:|---:|---:|---:|---|
| `VA_to_T` | `additive` | 0.0533 | 0.0106 | 0.0086 | 13.7 | True |
| `VA_to_T` | `product` | 0.0567 | 0.0304 | 0.0317 | 22.3 | True |
| `VA_to_T` | `mlp` | 0.0371 | 0.0065 | -0.0101 | 4.7 | True |

## MOSEI positive-control audit

Status: **MLP_DID_NOT_RECOVER_PRODUCT_SIGNAL**.

Product J validation: `0.0198 ± 0.0019`; Generic MLP J validation: `-0.0041 ± 0.0027`.
MOSEI Product Joint is a positive control; Generic MLP must not be interpreted as weaker without checking training health and optimization.
