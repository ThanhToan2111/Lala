# H0 Joint Hypothesis-Class Sufficiency Decision

## Research question

Does a capacity-controlled generic nonlinear joint predictor reveal cross-modal predictive headroom that the canonical low-rank Product Joint misses?

## Main validation table

| Dataset | Mapping | J Product | J MLP | ΔH | Decision relevance |
|---|---|---:|---:|---:|---|
| MOSEI | `VA_to_T` | 0.0143 ± 0.0000 | 0.0029 ± 0.0000 | -0.0114 ± 0.0000 | positive control |
| MELD | `VA_to_T` | 0.0131 ± 0.0000 | 0.0147 ± 0.0000 | 0.0015 ± 0.0000 | candidate |

## Overall decision

**PRODUCT_CLASS_INSUFFICIENT**

The decision is based on MELD validation J_MLP. Test results are not used for model, pair, or hypothesis-class selection.

## Scope

No JAD, task-aware objective, representation change, pooling change, architecture sweep, or new dataset was run. A PRODUCT_CLASS_INSUFFICIENT result would require frozen five-seed confirmation before v5.7.
