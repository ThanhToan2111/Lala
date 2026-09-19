# MUStARD G2 task joint-headroom

Frozen pooled V/A/T features; sarcasm label; three seeds; validation BCE early stopping.

| Pair | Joint−Add Acc val | F1 val | AUC val | BCE improvement val | G2 |
|---|---:|---:|---:|---:|---|
| `VA` | 0.000 ± 0.000 pp | 1.375 ± 0.000 pp | 1.358 ± 0.000 pp | -0.0085 ± 0.0000 | **FAIL** |
| `VT` | 1.449 ± 0.000 pp | 0.999 ± 0.000 pp | -1.961 ± 0.000 pp | -0.0293 ± 0.0000 | **FAIL** |
| `AT` | 2.174 ± 0.000 pp | 0.455 ± 0.000 pp | -0.194 ± 0.000 pp | 0.0074 ± 0.0000 | **FAIL** |

G2 is a screening result; no JAD embedding is trained. A promising three-seed result requires confirmation before R2.
