# H0 Joint Hypothesis-Class Sufficiency Decision

## Research question

Does a capacity-controlled generic nonlinear joint predictor reveal cross-modal predictive headroom that the canonical low-rank Product Joint misses?

## Main validation table

| Dataset | Mapping | J Product | J MLP | ΔH | Decision relevance |
|---|---|---:|---:|---:|---|
| MOSEI | `VA_to_T` | 0.0198 ± 0.0019 | -0.0041 ± 0.0027 | -0.0239 ± 0.0044 | positive control |
| MELD | `VA_to_T` | 0.0047 ± 0.0031 | 0.0044 ± 0.0022 | -0.0003 ± 0.0021 | not a 3/3 candidate |
| MELD | `VT_to_A` | -0.0041 ± 0.0045 | 0.0018 ± 0.0014 | 0.0060 ± 0.0032 | not a 3/3 candidate |
| MELD | `AT_to_V` | 0.0000 ± 0.0004 | -0.0002 ± 0.0003 | -0.0002 ± 0.0003 | not a 3/3 candidate |

## Overall decision

**HYPOTHESIS_CLASS_INCONCLUSIVE**

MELD remains below the MLP gate, but the MOSEI positive control was not recovered by the Generic MLP in all seeds; hypothesis-class sufficiency therefore remains unverified.
MOSEI positive-control status: **MLP_DID_NOT_RECOVER_PRODUCT_SIGNAL**. Product J is positive, but Generic MLP J is not positive in every seed; this prevents claiming hypothesis-class sufficiency without further permitted audit.

The decision uses validation only. Test results are descriptive and never create a candidate.

## Scope

No JAD, task-aware objective, representation change, pooling change, architecture sweep, or new dataset was run. A PRODUCT_CLASS_INSUFFICIENT result would require frozen five-seed confirmation before v5.7.
