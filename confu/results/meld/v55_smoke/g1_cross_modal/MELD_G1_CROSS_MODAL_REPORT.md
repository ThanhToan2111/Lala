# MELD G1 cross-modal identifiability

Utterance-level pooled V/A/T features; three screening seeds; validation-only model selection.

| Mapping | Capacity additive/joint | Val J | Test J | G1 |
|---|---:|---:|---:|---|
| `VA→T` | 54,240/53,976 (0.487%) | 0.0097 ± 0.0000 | 0.0125 ± 0.0000 | **FAIL** |
| `VT→A` | 108,694/107,656 (0.955%) | -0.0083 ± 0.0000 | 0.0018 ± 0.0000 | **FAIL** |
| `AT→V` | 79,406/79,784 (0.476%) | 0.0030 ± 0.0000 | 0.0035 ± 0.0000 | **FAIL** |

G1 pass requires joint-vs-additive validation R² advantage J > 0.01 in every screening seed. Test J is reporting-only.
