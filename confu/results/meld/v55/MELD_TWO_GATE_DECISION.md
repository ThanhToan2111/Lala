# MELD two-gate decision

## Two-gate matrix

| Pair / mapping | G1 | G2 | Final type |
|---|---|---|---|
| `VA` / `VA_to_T` | FAIL | WEAK_UNSTABLE | **INCONCLUSIVE** |
| `VT` / `VT_to_A` | FAIL | FAIL | **TYPE_I** |
| `AT` / `AT_to_V` | FAIL | WEAK_UNSTABLE | **INCONCLUSIVE** |

## Hard stop

JAD is not trained unless the same pair reaches TYPE_IV after permitted frozen-seed confirmation. This screening does not change pooling, rank, loss, purifier, context, or architecture.

G1 and G2 are validation-gated; test metrics are descriptive only.
