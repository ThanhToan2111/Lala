# AV-MNIST bimodal SynergyFormer benchmark

Run date: 2026-09-17. Dataset: MultiBench AV-MNIST. Protocol: five seeds, best validation-accuracy checkpoint, early stopping patience 4, batch size 512, AdamW at `1e-4`, no augmentation.

## Goal and root cause

The AV-MNIST path was refactored from a tri-modal image/audio/random-text experiment into the requested image-audio experiment. The old objective used instance InfoNCE despite repeated class labels and optimized unused `r13`, `r23`, `r123`, and global paths.

The first clean run fixed the low-variance collapse but exposed a second failure: variance regularization can copy one latent direction into every output dimension. That run had `r12` variance 0.435 but effective rank 1.56, and shuffle/zero interaction did not hurt accuracy. Factor/output decomposition showed that the product representation put 93.1% of its singular-value mass in the first direction while the output matrix remained full-rank. Removing factor bias alone did not fix it.

The final objective therefore adds an off-diagonal VICReg self-covariance term. A scale check at coefficients 0.01, 0.1, and 0.5 increased effective rank from about 2.8 to 4.7 to 8.5 without increasing rank 64. The final coefficient is 0.5.

## Mathematical and tensor changes

For `r1,r2: [B,256]`, factor rank `R=64`:

```text
u1  = LN(U1 r1)                         [B,64]
u2  = LN(U2 r2)                         [B,64]
h12 = (u1 * u2) / sqrt(64)              [B,64]
r12 = Wo h12, factor/output bias false  [B,256]
c12 = LN(LN(r1) + LN(r2) + sigmoid(a) LN(r12))
```

The scalar gate starts at `sigmoid(0)=0.5`. Ten normalized learnable class prototypes replace the random text encoder.

```text
L = CE(c12,y)
  + 0.2 [CE(r1,y) + CE(r2,y)]
  + 1.0 L_variance(r12)
  + 0.01 L_cross_covariance(r12,r1,r2)
  + 0.5 L_off_diagonal_covariance(r12)
```

The corrected cross-covariance loss uses squared Frobenius sum divided by `D`, not an elementwise mean that effectively divides by `D²`.

## Fair five-seed result

Values are test-set mean ± sample standard deviation. All four methods use the same image/audio ResNet18 encoders, prototypes, split, seeds, optimizer, batch size, maximum epochs, early stopping, and no augmentation.

| Model | Parameters | Accuracy | Macro F1 | Weighted F1 | Best val accuracy |
|---|---:|---:|---:|---:|---:|
| Additive | 22.607M | 70.448 ± 0.522 | 69.772 ± 0.708 | 70.239 ± 0.682 | 74.164 ± 0.287 |
| Concat + MLP | 22.805M | 70.518 ± 0.453 | 69.778 ± 0.703 | 70.242 ± 0.690 | 74.256 ± 0.152 |
| ConFu-style | 22.805M | **70.886 ± 0.578** | **70.336 ± 0.632** | **70.789 ± 0.611** | **74.652 ± 0.260** |
| SynergyFormer | 22.659M | 70.762 ± 0.338 | 70.189 ± 0.283 | 70.635 ± 0.278 | 74.100 ± 0.185 |

Per-seed test accuracy (%):

| Seed | Additive | Concat MLP | ConFu-style | SynergyFormer | Synergy − Additive | Gain12 within Synergy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.73 | 70.14 | 71.03 | 70.61 | -0.12 | +0.04 |
| 2 | 70.09 | 69.96 | 70.34 | 70.88 | +0.79 | -0.40 |
| 3 | 70.32 | 70.93 | 71.81 | 70.35 | +0.03 | -0.43 |
| 4 | 71.20 | 70.95 | 70.74 | 70.71 | -0.49 | +0.56 |
| 5 | 69.90 | 70.61 | 70.51 | 71.26 | +1.36 | +0.34 |

Paired SynergyFormer differences are +0.314 ± 0.748 points versus additive, +0.244 ± 0.630 versus concat MLP, and -0.124 ± 0.878 versus ConFu-style. Paired t-tests are not significant (`p=0.401`, `0.435`, and `0.768` respectively); five seeds are too few for a strong superiority claim.

The historical project ConFu run was 71.51% at one seed. It used the old random-text/contrastive protocol, so it is recorded for continuity but is not a fair direct comparator to this supervised prototype benchmark.

## Interaction diagnostics

| Metric | Five-seed result | Interpretation |
|---|---:|---|
| Internal `Gain12` | +0.022 ± 0.440 points | Positive mean, but negative in 2/5 seeds and not significant (`p=0.916`) |
| Shuffle interaction drop | **+1.106 ± 0.349 points** | Positive in 5/5 seeds; significant exploratory test (`p=0.002`) |
| Zero interaction drop | +0.064 ± 0.390 points | Positive in 3/5 seeds; not significant (`p=0.732`) |
| `r12` variance | 0.0598 ± 0.0031 | Stable |
| `r12` / weaker-modality variance | 1.627 ± 0.121 | Passes the required 0.1 threshold in 5/5 seeds |
| `r12` effective rank | 8.522 ± 0.072 | No longer rank-1, but still only 13.3% of rank 64 |
| Interaction gate | 0.4948 ± 0.0018 | Healthy; neither rejected nor saturated |
| cosine(`r12`,`r1`) | 0.0079 ± 0.0104 | Low, interpreted only with variance/rank checks |
| cosine(`r12`,`r2`) | -0.0025 ± 0.0060 | Low, interpreted only with variance/rank checks |

The deterministic seed-1 linear probes are: `r1=63.40%`, `r2=32.95%`, `r12=66.09%`, `concat(r1,r2)=67.20%`, and `concat(r1,r2,r12)=67.15%`. Thus `r12` is predictive by itself, but does not add linearly unique information beyond both modalities (`-0.05` points).

## Acceptance decision

| Gate | Status |
|---|---|
| Clean bimodal graph; no random text/tri-modal branches | Pass |
| Ten learnable prototypes; no instance InfoNCE | Pass |
| Normalized low-rank factors and bias-free interaction output | Pass |
| Correct variance/cross-covariance scaling and scale-aware gate | Pass |
| Variance ratio ≥ 0.1 in every seed | Pass |
| Mean `Gain12 > 0` | Technically pass (+0.022 points), scientifically weak |
| Shuffle interaction hurts | Pass in 5/5 seeds |
| Zero interaction hurts | Weak: mean positive, only 3/5 seeds |
| Healthy effective rank | Improved and stable, still low relative to rank 64 |
| Beats all fair baselines | Fail: trails ConFu-style by 0.124 points |
| Linear probe with `r12` beats unimodal concat | Fail by 0.05 points |

Conclusion: the collapse bottleneck is substantially fixed and the interaction is actively used under permutation, but complementary information and composition utility are not stable enough for a superiority claim. Do not add cross-attention or sweep rank yet. The next experiment should target interaction selectivity/composition, using the current five-seed protocol unchanged.

## Files and reproducibility

Core changes:

- `src/modules/models/synergyformer.py`: normalized/bias-configurable low-rank interaction, gated normalized composition, corrected cross-covariance, variance and covariance losses.
- `src/experiments/av_mnist/synergyformer.py`: fair four-model runner, prototype CE, diagnostics, shuffle/zero evaluation, deterministic probes, metadata.
- `configs/av_mnist.yaml`: final reproducible configuration.
- `tests/test_synergyformer.py`: interaction/gradient, collapse-loss, covariance, missing-modality, finite batch-size-one, checkpoint-name checks.

Commands:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_synergyformer -v
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.av_mnist.synergyformer model_type=synergy seed=1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.av_mnist.synergyformer model_type=additive seed=1 run_linear_probes=false
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.av_mnist.synergyformer model_type=concat_mlp seed=1 run_linear_probes=false
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.av_mnist.synergyformer model_type=confu seed=1 run_linear_probes=false
```

Repeat with `seed=2` through `seed=5`. Per-run JSON is in `results/av_mnist/bimodal/`, checkpoints in `results/av_mnist/checkpoints/`, CSV epoch logs in `outputs/bimodal_avmnist/`, and failed/intermediate diagnostic attempts in `results/av_mnist/diagnostics/`.

No augmentation, rank sweep, cross-attention, token interaction, or tri-modal expansion was added because the current interaction-complementarity gate is not yet robust.
