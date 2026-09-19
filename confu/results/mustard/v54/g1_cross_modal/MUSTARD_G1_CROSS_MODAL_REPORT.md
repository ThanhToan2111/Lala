# MUStARD G1 cross-modal identifiability

Processed MultiBench MUStARD, masked-mean pooled features, fixed split, three screening seeds.

| Mapping | Capacity | Val J | Test J | G1 |
|---|---:|---:|---:|---|
| `VA_to_T` | 20,626/20,610 (0.078%) | -0.0057 ± 0.0060 | -0.1496 ± 0.0535 | **FAIL** |
| `VT_to_A` | 18,532/18,401 (0.707%) | -0.0101 ± 0.0069 | -0.0022 ± 0.0064 | **FAIL** |
| `AT_to_V` | 19,867/19,819 (0.242%) | -0.0591 ± 0.0450 | -0.1336 ± 0.1145 | **FAIL** |

G1 pass requires validation J > 0.01 in all three screening seeds. Test J is reporting-only.
