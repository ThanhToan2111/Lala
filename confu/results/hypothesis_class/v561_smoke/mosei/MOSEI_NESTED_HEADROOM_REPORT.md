# MOSEI nested joint-class headroom audit

Frozen Product Joint plus one zero-initialized, one-hidden-layer concatenation residual. Product parameters are frozen.

Dataset metadata: `{'cache': True, 'label_alignment_sentiment_match': 1.0}`.

## Capacity and validation result

| Mapping | Product params | Residual params | Residual ratio | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|---:|---:|
| `VA_to_T` | 25,635 | 5,740 | 0.224 | 0.0220 ± 0.0000 | 0.0223 ± 0.0000 | 0.0004 ± 0.0000 |

## Seed-level validation result

| Mapping | Seed | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | 0.0220 | 0.0223 | 0.0004 |

## Training and residual health

| Mapping | Seed | Initial val MSE | Best val MSE | Best epoch | Product R² val | Nested R² val | Residual norm val | Finite | Product grad max | Residual grad first |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `VA_to_T` | 1 | 0.935551 | 0.935180 | 3 | 0.0318 | 0.0321 | 0.3941 | True | 0.00e+00 | 5.13e-03 |

## Test result (descriptive)

| Mapping | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|
| `VA_to_T` | 0.0210 ± 0.0000 | 0.0217 ± 0.0000 | 0.0007 ± 0.0000 |
