# MUStARD G1 cross-modal identifiability

Processed MultiBench MUStARD, masked-mean pooled features, fixed split, three screening seeds.

| Mapping | Capacity | Val J | Test J | G1 |
|---|---:|---:|---:|---|
| `VA_to_T` | 20,626/20,610 (0.078%) | 0.0170 ± 0.0000 | -0.0106 ± 0.0000 | **PASS** |
| `VT_to_A` | 18,532/18,401 (0.707%) | 0.0416 ± 0.0000 | 0.0123 ± 0.0000 | **PASS** |
| `AT_to_V` | 19,867/19,819 (0.242%) | 0.0144 ± 0.0000 | -0.0584 ± 0.0000 | **PASS** |

G1 pass requires validation J > 0.01 in all three screening seeds. Test J is reporting-only.
