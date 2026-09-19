# MOSEI H0 hypothesis-class audit

Frozen canonical pooled representations; additive, capacity-matched Product Joint, and one-hidden-layer Generic Joint MLP.

Dataset metadata: `{'cache': True, 'label_alignment_sentiment_match': 1.0}`.

## Parameter audit

| Mapping | Additive | Product | Generic MLP | MLP mismatch vs additive |
|---|---:|---:|---:|---:|
| `VA_to_T` | 25,602 | 25,635 | 25,324 | 1.086% |

## Main validation result

| Mapping | J Product | J MLP | ΔH |
|---|---:|---:|---:|
| `VA_to_T` | 0.0143 ± 0.0000 | 0.0029 ± 0.0000 | -0.0114 ± 0.0000 |

## Seed-level validation result

| Mapping | Seed | J Product | J MLP | ΔH |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | 0.0143 | 0.0029 | -0.0114 |

## Training health

| Mapping | Predictor | Train R² | Val R² | Test R² | Best epoch |
|---|---|---:|---:|---:|---:|
| `VA_to_T` | `additive` | 0.0197 | -0.0026 | -0.0049 | 3.0 |
| `VA_to_T` | `product` | 0.0201 | 0.0118 | 0.0059 | 3.0 |
| `VA_to_T` | `mlp` | 0.0288 | 0.0004 | -0.0225 | 3.0 |
