# MOSEI v5.3 T0 task-headroom audit

Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → sentiment, five seeds.

## Protocol

The canonical vision/audio features are frozen. T0 trains only task heads on `Y=1[sentiment>0]`; it does not use `h_D2`, does not retrain cross-modal predictors, and uses validation-only BCE for early stopping. All models use AdamW (`1e-3`, weight decay `1e-4`), batch size 512, maximum 80 epochs, patience 8.

## Capacity audit

| Model | Parameters | Mismatch vs additive |
|---|---:|---:|
| Linear VA | 788 | 93.775% |
| Additive | 12,658 | 0.000% |
| Joint Product | 12,641 | 0.134% |
| Generic MLP | 12,625 | 0.261% |

## Main five-seed test result

| Model | Params | Accuracy (%) | Macro F1 (%) | AUC (%) | BCE |
|---|---:|---:|---:|---:|---:|
| Linear VA | 788 | 62.763 ± 0.000 | 62.751 ± 0.000 | 67.532 ± 0.000 | 0.651 ± 0.000 |
| Additive | 12,658 | 63.578 ± 0.000 | 63.576 ± 0.000 | 68.917 ± 0.000 | 0.637 ± 0.000 |
| Joint Product | 12,641 | 63.814 ± 0.000 | 63.803 ± 0.000 | 67.617 ± 0.000 | 0.649 ± 0.000 |
| Generic MLP | 12,625 | 64.329 ± 0.000 | 64.321 ± 0.000 | 68.807 ± 0.000 | 0.640 ± 0.000 |

## Paired headroom

Positive BCE improvement means the compared model has lower BCE than additive. Values are mean ± sample standard deviation; 95% CIs and per-seed differences are in JSON.

| Comparison | Accuracy Δ (pp) | F1 Δ (pp) | AUC Δ (pp) | BCE improvement |
|---|---:|---:|---:|---:|
| Joint − Additive | 0.236 ± 0.000 | 0.227 ± 0.000 | -1.300 ± 0.000 | -0.012 ± 0.000 |
| MLP − Additive | 0.751 ± 0.000 | 0.745 ± 0.000 | -0.111 ± 0.000 | -0.003 ± 0.000 |

## Validation decision

T0 decision: **INCONCLUSIVE**.

The decision is based only on validation paired differences. A headroom candidate must have positive mean and positive sign in at least 4/5 seeds for accuracy, macro F1, AUC, and BCE improvement. This prevents a single favorable test metric from opening a new method branch.

## Interpretation

T0 separates task-relevant non-additivity (`VA→sentiment`) from canonical JAD identifiability (`VA→T`). It does not claim causality, PID synergy, or exact interaction decomposition.

## Artifacts

- `t0_task_headroom.json` — per-model/per-seed train, validation, and test metrics plus paired differences.
- `mosei_task_headroom.py` — reproducible T0 runner.
