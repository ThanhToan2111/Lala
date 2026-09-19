# CMU-MOSEI exact alignment audit

Run date: 2026-09-18. Phase: ConFu++ v5.1 R1.1.

## Decision

The canonical raw MOSEI source contains deterministic IDs (`video[index]`), official seven-column labels, and split membership. Its internal alignment is exact. The older `mosei_senti_data.pkl` feature file stores timestamp triplets (`video|start|end`) rather than those canonical IDs; literal ID intersection is therefore zero. It must not be described as exactly ID-aligned to the raw label source.

A canonical pooled cache was regenerated from `mosei_raw.pkl` with official labels. This is the allowed canonical-source regeneration path. Because it retains 713-dimensional raw vision features and a different sample count, its R1 comparison is a source-sensitivity result, not a pure label-only ablation of the old 35-dimensional processed representation.

## Required audit fields

- `processed_count`: 22777
- `official_count`: 22860
- `matched_count`: 0
- `duplicate_ids.processed`: 0
- `duplicate_ids.official`: 0
- `split_mismatches`: 0
- `label_mismatches`: 0

Full unmatched ID lists are in `mosei_alignment.json`.

## Canonical cache

- Source: `mosei_raw.pkl`
- Split sizes: `{'train': 16327, 'valid': 1871, 'test': 4662}`
- Pooled dimensions: `{'vision': 713, 'audio': 74, 'text': 300}`
- Label source: `official raw labels, columns 1:7`
- Cache: `results/mosei/identifiability_exact/mosei_pooled.npz`
- Alignment rate: 1.0 by construction from the canonical source.

## Limitation

The timestamp-to-canonical bridge in the old processed file remains unresolved without using a feature/content matching heuristic. The old emotion-conditioned R1 result therefore remains limited. The canonical-source R1 run is the correct next audit, but any difference must be interpreted jointly with the raw-source representation change.

## Canonical-source pooled R1 result

The unchanged predictor protocol was rerun on the exact canonical cache. Values are validation `J` mean ± sample standard deviation over three seeds.

| Mapping | Old processed J | Canonical raw J | Difference |
|---|---:|---:|---:|
| VA→T | -0.0111 ± 0.0006 | **+0.0198 ± 0.0019** | +0.0309 |
| VT→A | +0.0016 ± 0.0062 | +0.0009 ± 0.0050 | -0.0007 |
| AT→V | +0.0010 ± 0.0023 | +0.0019 ± 0.0029 | +0.0010 |
| fear, VT→A | +0.0142 ± 0.0152 | -0.0170 ± 0.0059 | -0.0312 |

Canonical `VA→T` passes the frozen gate in all seeds: `0.02196`, `0.01905`, and `0.01847`. It is the only mapping that passes. The exact-aligned R1 report is in `results/mosei/identifiability_exact/MOSEI_IDENTIFIABILITY_EXACT_REPORT.md`; this triggers R2 on `VA→T` only.
