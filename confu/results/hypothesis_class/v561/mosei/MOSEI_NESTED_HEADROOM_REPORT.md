# MOSEI nested joint-class headroom audit

Frozen Product Joint plus one zero-initialized, one-hidden-layer concatenation residual. Product parameters are frozen.

Dataset metadata: `{'cache': True, 'label_alignment_sentiment_match': 1.0}`.

## Product reference provenance

- `frozen_existing_checkpoint`

## Capacity and validation result

| Mapping | Product params | Residual params | Residual ratio | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|---:|---:|
| `VA_to_T` | 25,635 | 5,740 | 0.224 | 0.0198 ± 0.0019 | 0.0200 ± 0.0021 | 0.0001 ± 0.0003 |

## Seed-level validation result

| Mapping | Seed | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | 0.0220 | 0.0224 | 0.0004 |
| `VA_to_T` | 2 | 0.0190 | 0.0190 | 0.0000 |
| `VA_to_T` | 3 | 0.0185 | 0.0185 | 0.0000 |

## Training and residual health

| Mapping | Seed | Initial val MSE | Best val MSE | Best epoch | Product R² val | Nested R² val | Residual norm val | Finite | Product grad max | Residual grad first |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `VA_to_T` | 1 | 0.935551 | 0.935129 | 5 | 0.0318 | 0.0322 | 0.5708 | True | 0.00e+00 | 5.13e-03 |
| `VA_to_T` | 2 | 0.938864 | 0.938864 | 0 | 0.0283 | 0.0283 | 0.0000 | True | 0.00e+00 | 4.51e-03 |
| `VA_to_T` | 3 | 0.936074 | 0.936074 | 0 | 0.0312 | 0.0312 | 0.0000 | True | 0.00e+00 | 5.19e-03 |

## Test result (descriptive)

| Mapping | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|
| `VA_to_T` | 0.0231 ± 0.0018 | 0.0233 ± 0.0015 | 0.0002 ± 0.0003 |
