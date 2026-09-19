# MUStARD two-gate decision

## Data audit

Source: `data/multibench/mustard_download/sarcasm.pkl`; split sizes: train=414, valid=138, test=138.
Feature dimensions: {'vision': 371, 'audio': 81, 'text': 300}; sequence handling: `masked_mean_nonzero_frames`; unique IDs: `True`.

## Two-gate matrix

| Pair / mapping | G1 | G2 | Final type |
|---|---|---|---|
| `VA` / `VA_to_T` | PASS | FAIL | **TYPE_II** |
| `VT` / `VT_to_A` | PASS | FAIL | **TYPE_II** |
| `AT` / `AT_to_V` | PASS | FAIL | **TYPE_II** |

## Hard decision

No JAD training is allowed from this screening artifact unless the same pair reaches TYPE_IV after any permitted frozen-seed confirmation. This run does not change JAD, loss, rank, purifier, or architecture.

MUStARD is small; preserve per-seed values and do not rank pairs by test performance.
