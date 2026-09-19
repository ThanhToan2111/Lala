# Nested Joint-Class Headroom Decision

## Research question

Does a modest nonlinear residual on top of the frozen Product Joint reveal substantial additional cross-modal predictive headroom?

## Validation result

| Dataset | Mapping | J Product | J Nested | Δ Nested |
|---|---|---:|---:|---:|
| MOSEI | `VA_to_T` | 0.0220 ± 0.0000 | 0.0223 ± 0.0000 | 0.0004 ± 0.0000 |
| MELD | `VA_to_T` | -0.0041 ± 0.0000 | 0.0005 ± 0.0000 | 0.0046 ± 0.0000 |

## Decision

**NESTED_HEADROOM_ABSENT**

No MELD mapping exceeds Δ_nested=0.01 in all three seeds.
MELD candidates: `{'VA_to_T': False}`.

Product is frozen, the epoch-zero Product output is a validation candidate, and test metrics are descriptive only.

## Scope

No architecture sweep, optimizer sweep, representation change, JAD training, downstream task, synthetic reopen, or new dataset was run.

NESTED_HEADROOM_ABSENT
