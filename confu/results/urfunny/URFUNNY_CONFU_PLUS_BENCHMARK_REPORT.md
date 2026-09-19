# UR-FUNNY ConFu / ConFu++ benchmark

Run date: 2026-09-17. Dataset: official MultiBench UR-FUNNY packed data. The
binary humor task has train/validation/test sizes `8074/1034/1058`; modality
order is vision/audio/text with padded inputs `[B,50,371]`, `[B,50,81]`, and
`[B,50,300]`.

## Protocol

The matched baseline is the repository's original pairwise ConFu: three
Transformer encoders, 256-dimensional projections, 512-unit pair MLPs,
AdamW `5e-5`, weight decay `1e-4`, batch size 128, 100 epochs, deterministic
seeds 1--5, no augmentation, and the best validation-loss checkpoint.

The final metric is a frozen linear probe on the best checkpoint. To remove a
probe implementation difference from the comparison, baseline and upgraded
models were both evaluated with `urfunny_probe.py`, sklearn fast-search, and
the same split/extraction protocol. The original runner CSV is retained as a
reproduction artifact; its probe values can differ slightly from the strict
comparison below.

## Baseline versus ConFu++

ConFu++ here is the implemented S1 adversarial de-shortcutting adapter plus a
task-aware auxiliary head:

```text
lambda_shortcut = 0.1, tau = 0.5, k_adv = 1
adversary learning-rate scale = 0.5, hidden size = 256
warmup fraction = 0.2, lambda_task = 0.1
```

The adversaries and task head are training-only. The inference backbone has
the same 25.7M parameters as ConFu; the temporary training graph reports
26.5M parameters (+3.1%), within the 5% efficiency gate.

All numbers are test accuracy (%) mean ± sample standard deviation.

| Representation | ConFu | ConFu++ S1 + task | Paired difference |
|---|---:|---:|---:|
| Vision `r1` | 51.248 ± 2.791 | 51.822 ± 3.017 | +0.573 ± 0.937 |
| Audio `r2` | 57.879 ± 0.992 | 57.667 ± 0.871 | -0.213 ± 0.253 |
| Text `r3` | 62.183 ± 1.045 | 62.878 ± 0.805 | +0.695 ± 1.266 |
| Vision + audio `r12` | 58.851 ± 0.711 | 58.192 ± 0.695 | -0.659 ± 0.525 |
| Vision + text `r13` | 61.758 ± 0.295 | 61.074 ± 1.016 | -0.685 ± 1.177 |
| Audio + text `r23` | 62.162 ± 0.515 | 61.624 ± 1.121 | -0.538 ± 1.108 |
| All modalities `r1‖r2‖r3` | **64.462 ± 0.713** | 63.549 ± 0.855 | **-0.913 ± 1.012** |

Per-seed all-modality accuracy (%):

| Seed | ConFu | ConFu++ S1 + task | Δ ConFu++ − ConFu |
|---:|---:|---:|---:|
| 1 | 65.469 | 64.266 | -1.203 |
| 2 | 64.508 | 62.562 | -1.946 |
| 3 | 64.639 | 64.082 | -0.557 |
| 4 | 63.510 | 64.165 | +0.655 |
| 5 | 64.183 | 62.670 | -1.513 |

The paired t-test is `t=-2.016, p=0.114` with four degrees of freedom.
ConFu++ is lower in four of five seeds and does not beat the baseline.

## Bottleneck diagnosis

The baseline is not failing because of representation collapse. Seed-1 audit
metrics show:

| Pair | Conditional gain (pp) | Interaction shuffle drop (pp) | Lower → interaction R² | Effective rank |
|---|---:|---:|---:|---:|
| Vision + audio | -0.662 | +7.467 | 0.963 | 50.7 |
| Vision + text | +0.189 | +8.034 | 0.970 | 52.7 |
| Audio + text | -0.756 | +9.168 | 0.970 | 50.3 |

The pair outputs react to permutations, but roughly 96--97% of each
interaction can be reconstructed from its lower-order pair. Activity is real;
conditional information is not. This is the same redundancy bottleneck seen
in the MOSI audit.

The S1+task run does not resolve it. On the seed-1 standardized audit, the
conditional gains are `-0.189`, `+2.268`, and `+0.473` pp for vision/audio,
vision/text, and audio/text, while lower-to-interaction R² remains
`0.969/0.975/0.979`. The interaction remains active, but its representation
is still largely redundant and the all-modality lower-order embedding loses
accuracy.

An additional task-only pilot with `lambda_task=1.0` reached 65.597% after 30
epochs at seed 1, but the matched 100-epoch run selected a different best
checkpoint and reached only 64.188%. This is evidence of early-checkpoint
overfitting, not a reproducible improvement; the remaining task-only seed was
not expanded after this failure.

## Gate decision

| Gate | Decision | Evidence |
|---|---|---|
| Dependence D | Fail | Positive permutation drops, but adversarial shortcut meter did not pass both single-modality thresholds and R² stayed high. |
| Utility U | Fail | All-modality paired gain is `-0.913 ± 1.012` pp; pair conditional gains are not consistently positive. |
| Health H | Pass | Effective ranks remain within the allowed range; no collapse signal. |
| Efficiency E | Pass | Inference backbone is unchanged; training-only overhead is 3.1%. |

## Conclusion

The UR-FUNNY reproduction of original ConFu is complete at `64.462 ± 0.713%`
with the strict five-seed frozen-probe protocol. The current ConFu++ upgrade
is a scientifically useful negative result, but it does not exceed baseline.
The bottleneck is objective/data alignment: contrastive pair fusion learns
active but redundant interaction features, while the auxiliary task pressure
interferes with the representation geometry used by the downstream probe.

Do not add cross-attention, rank expansion, or a larger sweep. The next valid
upgrade requires a training target with measurable conditional headroom, or a
residual/utility mechanism validated first on the mandated synthetic ground
truth and negative controls.

## Artifacts and reproduction commands

- Baseline checkpoint directory: `results/urfunny_confu/checkpoints/humor/confu/seed_{1..5}/`
- ConFu++ checkpoint directory: `results/urfunny_confu_plus_task_final/checkpoints/humor/confu_plus/seed_{1..5}/`
- Strict baseline probes: `results/urfunny_confu/urfunny_probe_seed_{1..5}.json`
- Strict ConFu++ probes: `results/urfunny_confu_plus_task_final/urfunny_probe_seed_{1..5}.json`
- Baseline audit: `results/urfunny_confu/urfunny_audit_seed_1.json`
- ConFu++ audit: `results/urfunny_confu_plus_task_final/urfunny_audit_seed_1.json`
- Original runner table: `results/urfunny_confu/confu-humor.csv`

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.multibench.main \
  scenario=confu dataset=humor dataset.embedding.common_dim=256 fusion_hidden_dim=512 \
  iteration=1 seed=1 results_on_test=true \
  results_path='${hydra:runtime.cwd}/results/urfunny_confu' \
  training.max_epochs=100 training.batch_size=128 training.num_workers=0

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python -m src.experiments.multibench.urfunny_complementarity \
  dataset=humor dataset.embedding.common_dim=256 fusion_hidden_dim=512 \
  iteration=1 seed=1 results_path='${hydra:runtime.cwd}/results/urfunny_confu_plus_task_final' \
  training.max_epochs=100 training.batch_size=128 training.num_workers=0 \
  lambda_shortcut=0.1 shortcut_tau=0.5 k_adv=1 adv_lr_scale=0.5 \
  adv_hidden=256 adv_warmup_fraction=0.2 lambda_task=0.1

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.multibench.urfunny_probe \
  --checkpoint results/urfunny_confu/checkpoints/humor/confu/seed_1/best_model.ckpt \
  --output results/urfunny_confu/urfunny_probe_seed_1.json \
  --model-type confu --seed 1 --batch-size 256
```

Validation checks:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_synergyformer -v
PYTHONPATH=. .venv/bin/python -m py_compile \
  src/experiments/multibench/main.py \
  src/experiments/multibench/mosi_audit.py \
  src/experiments/multibench/urfunny_complementarity.py \
  src/experiments/multibench/urfunny_probe.py
```
