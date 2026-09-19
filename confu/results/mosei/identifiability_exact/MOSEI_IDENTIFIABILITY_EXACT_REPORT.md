# CMU-MOSEI canonical-source pooled identifiability

Run date: 2026-09-18. Phase: ConFu++ v5.1 R1.1. This is the exact canonical-source audit after the original processed-feature screen failed.

## Data decision

`mosei_raw.pkl` contains canonical IDs (`video[index]`), official seven-column labels, and split membership. Its internal identity/label alignment is exact and unique. The older `mosei_senti_data.pkl` contains timestamp triplets and cannot be joined to those canonical IDs literally; its old emotion result remains limited.

The canonical cache uses the raw source directly, masked-mean pools the first 50 frames with the same `_pool` function, and keeps official labels. It has train/validation/test sizes `16,327/1,871/4,662` and dimensions vision/audio/text `713/74/300`.

This is the permitted canonical-source regeneration path. Because the raw vision representation is 713-dimensional rather than the old processed 35-dimensional representation, the comparison below is a source-sensitivity result, not a label-only ablation.

## Capacity audit

| Direction | Additive parameters | Joint parameters | Difference |
|---|---:|---:|---:|
| VA→T | 25,602 | 25,635 | 0.129% |
| VT→A | 18,756 | 18,629 | 0.677% |
| AT→V | 35,664 | 35,923 | 0.726% |

All comparisons remain below the 1% capacity mismatch preference.

## Old versus canonical pooled R1

Values are validation `J` mean ± sample standard deviation over three seeds. The old row is from the processed-feature R1 report; the canonical row is rerun with the same predictor family, optimizer, threshold, mappings, and seeds.

| Mapping | Old processed J | Canonical raw J | Difference |
|---|---:|---:|---:|
| VA→T | -0.0111 ± 0.0006 | **+0.0198 ± 0.0019** | +0.0309 |
| VT→A | +0.0016 ± 0.0062 | +0.0009 ± 0.0050 | -0.0007 |
| AT→V | +0.0010 ± 0.0023 | +0.0019 ± 0.0029 | +0.0010 |
| fear, VT→A | +0.0142 ± 0.0152 | -0.0170 ± 0.0059 | -0.0312 |

Canonical `VA→T` seed-level validation values are `+0.02196`, `+0.01905`, and `+0.01847`; all three exceed `0.01`. Its test values are `+0.02105`, `+0.02426`, and `+0.02404`.

## R1 decision

**Pass.** Freeze the natural setting:

```text
source: canonical MOSEI raw
representation: pooled raw features
mapping: vision + audio -> text
task: global modality prediction
seeds: 1..3 screening; 1..5 for R2
gate: validation J > 0.01 in every screening seed
```

The previous fear-conditioned candidate is not selected. On the exact canonical source, fear/VT→A is negative in all three validation seeds. The alignment audit therefore changes both the winning mapping and the scientific interpretation of the old candidate.

## R2 trigger

R2 is now allowed on this frozen setting only. It uses canonical JAD: rank 64, bias-free low-rank projections, factor LayerNorm, product scaled by `sqrt(64)`, output LayerNorm, cosine regression, and five fixed seeds. No cross-attention, rank sweep, task loss, or third-order method is introduced.

Artifacts:

- Canonical machine results: `results/mosei/identifiability_exact/mosei_identifiability.json`
- Canonical pooled cache: `results/mosei/identifiability_exact/mosei_pooled.npz`
- Alignment audit: `results/mosei/alignment/MOSEI_ALIGNMENT_AUDIT.md`
- Alignment machine audit: `results/mosei/alignment/mosei_alignment.json`
- Runner: `src/experiments/multibench/mosei_identifiability.py`
- Alignment runner: `src/experiments/multibench/mosei_alignment_audit.py`
