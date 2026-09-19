# MUStARD G2 task joint-headroom

Frozen pooled V/A/T features; sarcasm label; three seeds; validation BCE early stopping.

| Pair | Joint−Add Acc val | F1 val | AUC val | BCE improvement val | G2 |
|---|---:|---:|---:|---:|---|
| `VA` | -1.449 ± 3.834 pp | -1.342 ± 4.344 pp | -0.036 ± 1.562 pp | -0.0099 ± 0.0054 | **FAIL** |
| `VT` | 0.242 ± 1.508 pp | -0.266 ± 1.708 pp | -1.422 ± 1.569 pp | -0.0159 ± 0.0185 | **WEAK_UNSTABLE** |
| `AT` | 1.208 ± 0.418 pp | 0.449 ± 1.077 pp | -0.697 ± 1.288 pp | -0.0026 ± 0.0077 | **WEAK_UNSTABLE** |

G2 is a screening result; no JAD embedding is trained. A promising three-seed result requires confirmation before R2.
