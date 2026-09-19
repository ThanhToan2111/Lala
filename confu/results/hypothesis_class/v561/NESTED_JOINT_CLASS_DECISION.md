# Nested Joint-Class Headroom Decision

## Research question

Does a modest nonlinear residual on top of the frozen Product Joint reveal substantial additional cross-modal predictive headroom?

## Validation result

| Dataset | Mapping | J Product | J Nested | Δ Nested |
|---|---|---:|---:|---:|
| MOSEI | `VA_to_T` | 0.0198 ± 0.0019 | 0.0200 ± 0.0021 | 0.0001 ± 0.0003 |
| MELD | `VA_to_T` | 0.0055 ± 0.0027 | 0.0058 ± 0.0027 | 0.0003 ± 0.0004 |
| MELD | `VT_to_A` | -0.0054 ± 0.0052 | -0.0016 ± 0.0046 | 0.0038 ± 0.0014 |
| MELD | `AT_to_V` | 0.0001 ± 0.0006 | 0.0002 ± 0.0006 | 0.0000 ± 0.0000 |

## Decision

**NESTED_HEADROOM_ABSENT**

No MELD mapping exceeds Δ_nested=0.01 in all three seeds.
MELD candidates: `{'VA_to_T': False, 'VT_to_A': False, 'AT_to_V': False}`.

Product is frozen, the epoch-zero Product output is a validation candidate, and test metrics are descriptive only.

## Scope

No architecture sweep, optimizer sweep, representation change, JAD training, downstream task, synthetic reopen, or new dataset was run.

NESTED_HEADROOM_ABSENT
