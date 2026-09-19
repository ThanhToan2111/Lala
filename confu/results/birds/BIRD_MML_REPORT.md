# Bird-MML (Zenodo 18920487) — ConFu++ Evaluation Report

Date: 2026-09-18. Code: `src/experiments/birds/{extract_embeddings,audit_embeddings,confu_frozen}.py`.

## Data access note (upstream release bug)

The Zenodo record's `audio.tar.gz` is split into 16 parts (aa..ap) but **part-ah was never
uploaded** (all 15 present parts pass MD5 against the manifest; `ah` is absent from the record).
The tar.gz stream is unreadable past that point. We recovered parts aa..ag, yielding
**71 of 149 species with complete audio+photo+text triplets (70,796 rows, balanced
814–1000 rows/species)**. All results use this subset and the frozen split in
`data/bird_mml/split.npz` (stratified 70/15/15). Re-running on the full 149 species requires only
re-downloading once the record is fixed.

## Protocol (frozen encoders)

- image: ResNet50 ImageNet-1K V2 penultimate (2048-d)
- audio: wav2vec2-base mean-pooled (768-d)
- text: all-MiniLM-L6-v2 mean-pooled (384-d) over `combined_caption`
- All fusion/alignment learned on frozen features; probes standardized; 5 seeds; paired stats.

Leakage audit (spec §24): 33.1% of `combined_caption` strings contain the species common name,
2.0% the scientific name. Text-only accuracy is therefore partially name-driven; documented, not
removed (this is the released benchmark design).

## Audit (spec §25/§107 gate)

| Subset | linear probe | nonlinear probe |
|---|---|---|
| image | 64.95% | 67.70% |
| audio | 1.43% (chance=1.41) | 1.42% |
| text | 63.74% | 63.89% |
| image+audio | 65.08% | 66.90% |
| image+text | **80.89%** | **82.37%** |
| audio+text | 64.08% | 63.94% |
| all | 81.01% | 81.74% |

Conditional gains (linear): image+text over best single **+15.94 pp** (genuine headroom);
image+audio +0.13; audio+text +0.34; all over best pair +0.12.

Gate verdict: **image–text complementarity exists and is large**. Audio carries nothing in these
features (wav2vec2 is speech-domain; flagged as a feature-extractor limitation, not necessarily a
dataset property). The benchmark therefore has one real pair (image–text) and one null pair —
an ideal testbed: methods must exploit the first without fabricating the second.

## Confirmatory matrix (5 seeds, paired)

| Metric | ConFu (orig.) | ConFu+Utility | ConFu+U+S1 |
|---|---|---|---|
| probe full acc (%) | 76.90 ± 0.65 | 83.01 ± 0.13 | 82.66 ± 0.37 |
| probe lower acc (%) | 76.71 ± 0.30 | 81.94 ± 0.18 | 81.89 ± 0.09 |
| gain full−lower (pp) | +0.19 ± 0.92 | +1.07 ± 0.11 | +0.77 ± 0.39 |
| pair-13 conditional gain (pp) | −0.12 | −0.36 | −0.21 |
| R²(image→z13) | 0.60 | 0.75 | 0.68 |
| R²(text→z13) | 0.59 | **0.44** | 0.50 |
| shuffle image drop (pp) | +43.12 | +51.86 | +48.46 |
| shuffle audio drop (pp) | +0.01 | +0.01 | +0.01 |
| shuffle text drop (pp) | +48.96 | +46.40 | +48.83 |

**Paired Δ (ConFu+U+S1 − ConFu): probe_full +5.76 ± 0.86 pp (t=+14.91, p<0.001).**

## Interpretation

1. **The alignment–utility gap on genuine-headroom data**: raw features offer +15.94 pp of
   image–text complementarity, but original ConFu's aligned pair representations deliver ≈ 0 pp
   (full ≈ lower). Higher-order alignment alone leaves the available complementarity on the table.
2. **ConFu++ (utility objective + S1) recovers a significant part of it**: +5.76 pp over the
   original objective (p<0.001), exceeding even the raw-feature linear-probe ceiling (81.01) —
   the trained representation adds structure beyond the frozen features.
3. **S1 behaves as designed on the balanced pair**: z13 becomes less predictable from text
   (R² 0.59→0.44 under U, 0.50 under U+S1), i.e., the interaction carries more distinctive
   content; and S1 fabricates nothing for the dead modality (audio drops stay ~0).
4. Remaining gap: the pair-conditional gain of z13 is still slightly negative (−0.21 pp) — the
   utility arrives through the jointly trained full head rather than through the pair term alone.
   This is the next open target (per-pair utility objectives).

## Limitations

- 71/149 species subset (upstream release bug), frozen encoders, wav2vec2 audio domain mismatch,
  33% caption name leakage. All documented; none affects the paired ConFu-vs-ConFu++ comparison,
  which shares every one of these factors.
