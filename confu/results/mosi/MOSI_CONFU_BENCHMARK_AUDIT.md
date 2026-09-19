# CMU-MOSI original ConFu benchmark audit

Run date: 2026-09-17. Dataset: official MultiBench CMU-MOSI packed split. The task is binary sentiment (`negative < 0`, otherwise positive), with vision/audio/text in that repository order. Split sizes are train/validation/test = 1,284/229/686; inputs are `[B,50,20]`, `[B,50,5]`, and `[B,50,300]` respectively. AV-MNIST was left unchanged as the project's negative control.

## Benchmark choice and protocol

MOSI is the only directly supported, genuine tri-modal original-ConFu benchmark in this checkout. MUStARD and UR-FUNNY have no equally direct loader/configuration, while Bird-MML is not released in the repository. The MultiBench MOSI files were obtained from the source linked by the vendored MultiBench loader. A packed-data loader error was fixed by making optional `torchtext` and robustness imports lazy; the normal packed MOSI path therefore has no unnecessary `torchtext` dependency.

Original ConFu was trained for 100 epochs with the repository's three Transformer encoders, 256-dimensional projections, pair fusion MLP hidden size 512, AdamW `5e-5`, weight decay `1e-4`, batch size 128, no augmentation, and the best validation-loss checkpoint. Five fixed seeds use the same official split. The model has 25.5M trainable parameters.

ConFu's original graph contains only `r12`, `r13`, and `r23`. It has **no `r123`**, classifier head, or trained higher-order fusion; downstream accuracy below comes from frozen linear probes. Thus this is a valid pairwise-ConFu reproduction and tri-modal dataset audit, not evidence for a third-order representation.

## Five-seed frozen-probe baseline

Values are test accuracy (%) mean ± sample standard deviation. `12=vision+audio`, `13=vision+text`, and `23=audio+text`; “concat” is lower-order concatenation.

| Representation | Accuracy |
|---|---:|
| Vision | 52.609 ± 3.272 |
| Audio | 57.817 ± 1.899 |
| Text | **67.681 ± 2.177** |
| `r12` | 57.241 ± 2.730 |
| `r13` | 60.198 ± 1.546 |
| `r23` | 62.379 ± 1.161 |
| Vision + audio concat | 57.633 ± 2.100 |
| Vision + text concat | 65.931 ± 2.754 |
| Audio + text concat | 67.630 ± 2.051 |
| Vision + audio + text concat | 66.170 ± 2.861 |

Per-seed all-modalities concat: 66.848, 67.708, 62.690, 63.885, and 69.718. The task is text-dominant: neither all-modalities nor either text-containing pair consistently improves on text alone. The runner's pair/unimodal rows are frozen-probe dataset baselines from each trained ConFu checkpoint, so they do not claim independent end-to-end supervised encoders.

As a non-interaction sanity reference, a matched seed-1 TriCLIP run (24.2M parameters, identical encoder/training protocol) yielded 66.281% all-modalities, versus 66.848% for ConFu (+0.567 points). This is only one seed and is not a superiority claim; the five-seed ConFu++-versus-ConFu comparison is deliberately deferred until a valid objective passes the audit gate.

## Pair dependency and conditional-utility audit

For each checkpoint, a validation-selected standardized logistic probe compares `R_low=[ri,rj]` with `[ri,rj,rij]`. A fixed-width 128-unit nonlinear probe provides a capacity-matched secondary check. Intervention drops evaluate the linear full probe after replacing only `rij`; single-modality shuffle first recomputes `rij` with one input permuted while keeping lower-order probe evidence clean. Predictability is ridge regression from `R_low` to `rij` on the held-out test split.

| Pair | Linear conditional gain (pp) | Nonlinear gain (pp) | Interaction shuffle drop (pp) | Effective rank | `R_low -> rij` R² |
|---|---:|---:|---:|---:|---:|
| Vision + audio | -0.962 ± 1.248 | +0.437 ± 2.768 | +0.875 ± 3.359 | 80.054 ± 2.616 | 0.937 ± 0.002 |
| Vision + text | -1.429 ± 1.608 | -0.408 ± 6.461 | +9.184 ± 3.012 | 97.269 ± 1.661 | 0.915 ± 0.007 |
| Audio + text | -2.274 ± 0.992 | -4.461 ± 4.147 | +12.128 ± 3.178 | 81.446 ± 2.694 | 0.953 ± 0.004 |

Single-modality interaction shuffle confirms real pair dependence for text pairs: vision/text drops are +7.318 ± 3.966 pp after shuffling vision and +9.942 ± 1.871 after shuffling text; audio/text drops are +8.047 ± 1.396 after audio and +10.671 ± 4.073 after text. Vision/audio is weak and unstable (+0.437 ± 4.548 and +1.254 ± 2.722 pp). Interaction variance is healthy (about `0.0031–0.0032`) and ranks are high, so this is not representation collapse.

## Decision

The bottleneck is **redundancy, not activity or rank**:

- `r13` and `r23` materially depend on both observed inputs under interventions, yet are highly reconstructible from lower-order evidence (`R²≈0.92–0.95`).
- Every pair has negative mean linear conditional utility; audio+text is negative under both linear and nonlinear probes.
- Vision+audio has neither stable input dependence nor stable interaction-use drop.
- Original ConFu has no `r123`, so a higher-order claim cannot be tested or made.

Therefore the ConFu++ utility objective must **not** be added yet. The correct next experiment is to document this as a shortcut/redundancy pattern and choose a task where lower-order evidence leaves measurable headroom, or design a utility target only after validation establishes a positive conditional-gain signal. No residual, Global–Local, cross-attention, rank expansion, or new architecture was added.

## Reproduction artifacts

- Five-seed result rows: `results/mosi_confu/confu-mosi.csv`
- Checkpoints: `results/mosi_confu/checkpoints/mosi/confu/seed_{1..5}/best_model.ckpt`
- Per-seed audits: `results/mosi_confu/mosi_audit_seed_{1..5}.json`
- Audit runner: `src/experiments/multibench/mosi_audit.py`

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.multibench.main \
  scenario=confu dataset=mosi dataset.embedding.common_dim=256 fusion_hidden_dim=512 \
  iteration=1 seed=1 results_on_test=true results_path='${hydra:runtime.cwd}/results/mosi_confu' \
  training.max_epochs=100 training.batch_size=128 training.num_workers=4

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.multibench.mosi_audit \
  --checkpoint results/mosi_confu/checkpoints/mosi/confu/seed_1/best_model.ckpt \
  --output results/mosi_confu/mosi_audit_seed_1.json --batch-size 128 --num-workers 0 --seed 1
```
