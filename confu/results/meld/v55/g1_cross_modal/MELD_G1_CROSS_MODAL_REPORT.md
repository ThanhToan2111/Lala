# MELD G1 cross-modal identifiability

Utterance-level pooled V/A/T features; three screening seeds; validation-only model selection.

| Mapping | Capacity additive/joint | Val J | Test J | G1 |
|---|---:|---:|---:|---|
| `VA→T` | 54,240/53,976 (0.487%) | 0.0020 ± 0.0041 | 0.0013 ± 0.0009 | **FAIL** |
| `VT→A` | 108,694/107,656 (0.955%) | -0.0028 ± 0.0057 | 0.0035 ± 0.0118 | **FAIL** |
| `AT→V` | 79,406/79,784 (0.476%) | 0.0000 ± 0.0004 | 0.0002 ± 0.0002 | **FAIL** |

G1 pass requires joint-vs-additive validation R² advantage J > 0.01 in every screening seed. Test J is reporting-only.

## Per-seed validation J

| Mapping | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| `VA→T` | 0.0066 | -0.0013 | 0.0006 |
| `VT→A` | -0.0083 | -0.0032 | 0.0032 |
| `AT→V` | 0.0002 | -0.0004 | 0.0003 |