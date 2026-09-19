# MELD G2 task headroom

Frozen utterance-level features; seven-class emotion CE; validation NLL early stopping; three screening seeds.

| Pair | Joint−Add Macro F1 val | NLL improvement val | Acc val | Weighted F1 val | G2 |
|---|---:|---:|---:|---:|---|
| `VA` | 5.103 ± 0.000 pp | -0.26233 ± 0.00000 | -13.357 ± 0.000 pp | 0.246 ± 0.000 pp | **WEAK_UNSTABLE** |
| `VT` | -1.477 ± 0.000 pp | -0.11869 ± 0.00000 | -2.256 ± 0.000 pp | -1.736 ± 0.000 pp | **FAIL** |
| `AT` | 1.000 ± 0.000 pp | -0.07572 ± 0.00000 | 0.271 ± 0.000 pp | 1.212 ± 0.000 pp | **WEAK_UNSTABLE** |

Strong G2 requires joint-vs-additive validation Macro F1 positive in all three seeds and positive mean NLL improvement. Test values are reporting-only.
