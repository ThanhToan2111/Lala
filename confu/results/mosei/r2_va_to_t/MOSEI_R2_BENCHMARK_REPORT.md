# CMU-MOSEI canonical R2 benchmark: VA→T

Run date: 2026-09-18. Phase: ConFu++ v5.1 R2. The setting was frozen only after canonical-source R1 passed: vision + audio → text, pooled raw MOSEI features, global modality prediction, five seeds.

## Frozen protocol

The exact canonical cache has split sizes `16,327/1,871/4,662` and dimensions `713/74/300` for vision/audio/text. R2 uses the same standardized representations and the canonical rank-64 JAD interaction for every method:

```text
D0 = r_T
D1 = r_T - q_A(r_V, r_A)
D2 = q_J(r_V, r_A) - q_A(r_V, r_A)
```

The interaction is bias-free low-rank multiplicative fusion with factor LayerNorm, `sqrt(64)` scaling, output LayerNorm, and cosine regression. No task loss or architecture sweep was used. Sentiment-positive probing (`sentiment > 0`) is a downstream accessibility diagnostic, not the R1 selection target.

## Downstream task result

Values are test accuracy, macro F1, and AUC in percent; mean ± sample standard deviation over five seeds. `Gain` is relative to the lower-order `concat(r_V,r_A)` probe.

| Representation | Accuracy | Gain | Macro F1 | AUC |
|---|---:|---:|---:|---:|
| Lower-order `r_V+r_A` | 64.397 ± 0.231 | — | 64.370 ± 0.232 | 67.204 ± 0.230 |
| D0 | 63.930 ± 0.425 | -0.468 ± 0.222 | 63.919 ± 0.428 | 66.693 ± 0.939 |
| D1 | 63.475 ± 0.472 | -0.922 ± 0.638 | 63.436 ± 0.468 | 68.004 ± 0.424 |
| D2 / JAD | 64.363 ± 0.288 | **-0.034 ± 0.099** | 64.318 ± 0.280 | 68.348 ± 1.399 |

D2 is statistically close to the lower-order accuracy baseline but does not improve it. D1 is consistently less accessible under accuracy and macro F1. AUC favors D2, but this does not translate into a positive fixed-threshold accuracy gain.

The capacity-controlled R1 signal therefore does not yield a downstream sentiment improvement after JAD distillation. This is not evidence that D2 is useless; it is evidence that predictive interaction and task accessibility are different quantities.

## Interaction diagnostics

| Method | Target cosine | Effective rank | Variance | Vision shuffle drop | Audio shuffle drop |
|---|---:|---:|---:|---:|---:|
| D0 | 0.163 ± 0.001 | 7.88 ± 0.18 | 0.976 ± 0.005 | +7.353 ± 0.687 pp | +4.680 ± 0.880 pp |
| D1 | 0.063 ± 0.016 | 12.01 ± 1.09 | 0.989 ± 0.006 | +2.239 ± 0.745 pp | +2.518 ± 0.341 pp |
| D2 | **0.848 ± 0.038** | **25.66 ± 1.36** | 0.525 ± 0.266 | +3.132 ± 1.636 pp | +3.591 ± 0.558 pp |

Shuffling either source changes the learned D2 output in all five seeds, so the representation is not completely ignored. However, shuffle sensitivity alone is insufficient: the task probe still has no positive accuracy gain.

## Shortcut audit

Test R² when reconstructing the learned interaction output from one modality alone:

| Method | Vision-only R² | Audio-only R² |
|---|---:|---:|
| D0 | 0.199 ± 0.012 | 0.207 ± 0.010 |
| D1 | -0.082 ± 0.034 | 0.038 ± 0.019 |
| D2 | 0.193 ± 0.030 | **0.302 ± 0.048** |

D2 has a notable audio-only shortcut. This explains why it can have high target cosine and positive shuffle drops without adding stable sentiment information beyond the lower-order representation. The current bottleneck is accessibility/selectivity, not collapse: D2 has nontrivial effective rank but remains partly reconstructible from audio alone.

## Decision

| Gate | Status |
|---|---|
| Canonical exact-source R1 | Pass: VA→T in 3/3 seeds |
| Five-seed R2 completed | Pass |
| D2 target fit | Pass: cosine 0.848 ± 0.038 |
| D2 source dependence | Pass exploratory: both shuffles hurt in 5/5 seeds |
| D2 beats lower-order sentiment accuracy | **Fail: -0.034 ± 0.099 pp** |
| D2 removes modality shortcut | **Fail: audio-only R² 0.302 ± 0.048** |
| Superiority over lower-order baseline | Fail |

R2 demonstrates a valid, naturally identifiable `VA→T` predictive interaction after exact canonical alignment, but the current D2 representation does not provide a measurable sentiment-accessibility gain. The correct scientific result is a fidelity–accessibility gap: D2 matches the lower-order task baseline while preserving a strong fit to the JAD interaction target.

Do not add cross-attention, task loss, rank sweeps, or another R2 loss. The next method-level bottleneck, if pursued under a separately approved experiment, is conditional selectivity against the audio shortcut. The v5.1 protocol itself is complete for this setting.

## Artifacts

- Machine summary: `results/mosei/r2_va_to_t/mosei_r2_va_to_t.json`
- Per-seed records: `mosei_r2_va_to_t_seed_{1..5}.json`
- R2 runner: `src/experiments/multibench/mosei_r2.py`
- Exact R1 report: `results/mosei/identifiability_exact/MOSEI_IDENTIFIABILITY_EXACT_REPORT.md`
- Alignment report: `results/mosei/alignment/MOSEI_ALIGNMENT_AUDIT.md`
