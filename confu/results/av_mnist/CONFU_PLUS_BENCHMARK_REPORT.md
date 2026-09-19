# ConFu++ AV-MNIST upgrade report

Run date: 2026-09-17. Dataset: MultiBench AV-MNIST. Shared protocol: official split, five seeds, image/audio ResNet18 encoders, 256-dimensional representations, rank 64, AdamW `1e-4`, batch 512, early stopping on validation accuracy, no augmentation. All reported uncertainty is sample standard deviation.

## Outcome

The upgrade fixes neither the central scientific bottleneck nor the baseline accuracy gap: `r12` is non-collapsed and conditionally predictive, but it is almost completely image-derived. Conditional utility is measurable, while genuine bimodal complementarity is not. The project must not claim SynergyFormer or ConFu++ superiority on AV-MNIST.

## Baseline reproduction and Stage A diagnosis

| Model | Test accuracy (%) |
|---|---:|
| Additive | 70.448 ± 0.522 |
| Concat + MLP | 70.518 ± 0.453 |
| ConFu-style | **70.886 ± 0.578** |
| Previous SynergyFormer | 70.762 ± 0.338 |

The baseline interaction is used under interaction permutation, but not as balanced multimodal evidence: image-before-interaction shuffle drops accuracy by `+1.268 ± 0.493 pp`, versus `+0.008 ± 0.079 pp` for audio shuffle. Its nonlinear conditional gain is only `+0.120 ± 0.362 pp`; `Gain12` is `+0.022 ± 0.446 pp`.

## P1: conditional utility loss

The implemented objective is:

```text
c_base = LN(LN(r1) + LN(r2))
c_full = LN(c_base + sigmoid(a) LN(r12))
L_utility = mean(relu(CE_full - stopgrad(CE_base) + margin))
```

Seed-1 weight sweep at margin zero selected `lambda_utility=0.2`.

| Weight | Accuracy (%) | Gain12 (pp) | Zero drop (pp) | Net correction |
|---:|---:|---:|---:|---:|
| 0.00 | 70.17 | -0.15 | -0.08 | -15 |
| 0.05 | 70.35 | +0.17 | +0.06 | +17 |
| 0.10 | 70.45 | -0.05 | +0.04 | -5 |
| 0.20 | 70.47 | **+0.36** | **+0.49** | **+36** |
| 0.50 | 70.62 | +0.28 | +0.39 | +28 |

Margin `.01` and `.05` did not improve the seed-1 choice, so the five-seed run uses weight `.2`, margin zero.

| Metric, five seeds | P1 utility result |
|---|---:|
| Accuracy | 70.296 ± 0.545% |
| Gain12 | +0.234 ± 0.263 pp; 5/5 positive |
| Zero drop | +0.262 ± 0.309 pp; 4/5 positive |
| Interaction shuffle drop | +1.346 ± 0.785 pp; 5/5 positive |
| Image / audio shuffle drop | +1.482 ± 0.847 / +0.004 ± 0.027 pp |
| Net correction | +23.4 ± 26.3; 5/5 positive |
| Linear conditional gain | +0.126 ± 0.186 pp; 4/5 positive |
| Matched nonlinear conditional gain | +0.294 ± 0.162 pp; 5/5 positive |
| `r12` effective rank | 9.182 ± 0.656 |
| `r12` / weaker-modality variance | 1.578 ± 0.107 |

The nonlinear probe has equal-capacity base and full MLPs (199,434 parameters each). Its gain is exploratory significant against zero at p=0.015; `Gain12` is not (p=0.118). The gate sweep peaks at gate `.5` on average (70.290%), consistent with learned gate `0.497 ± 0.003`.

### Comparison with reproduced baselines

| Paired comparison | P1 minus baseline (pp) | Paired t-test p |
|---|---:|---:|
| Additive | -0.152 ± 0.480 | 0.518 |
| Concat + MLP | -0.222 ± 0.663 | 0.496 |
| ConFu-style | -0.590 ± 0.978 | 0.249 |
| Previous SynergyFormer | -0.466 ± 0.513 | 0.112 |

P1 passes the conditional-utility and collapse gates, but not the accuracy/superiority or balanced-dependence gates. Hyperparameter selection used seed-1 test diagnostics, so these results are exploratory rather than confirmatory.

## P2: dependence objective

The implemented task-aware ranking term compares correct-class log-probability of the true interaction with interactions recomputed after shuffling each modality. It adds no parameters.

| `lambda_dependency`, seed 1 | Accuracy (%) | Gain12 (pp) | Audio-shuffle drop (pp) |
|---:|---:|---:|---:|
| 0.1 | 70.79 | +0.22 | -0.02 |
| 0.5 | 70.54 | +0.05 | +0.11 |
| 1.0 | 69.93 | +0.09 | +0.11 |

The small audio drops at stronger weights do not meet a robust dependence criterion and erase utility. P2 therefore fails at seed 1 and was not expanded to five seeds.

## P3: nonlinear redundancy

An MLP is trained to reconstruct `r12` from lower-order representations on the selected P1 checkpoint:

| Predictor | R² | Cosine |
|---|---:|---:|
| `r1 -> r12` | 0.988 | 0.996 |
| `r2 -> r12` | 0.031 | 0.203 |
| `(r1,r2) -> r12` | 0.996 | 0.999 |

At dependency weight `.5`, the values are materially unchanged (`r1` R²=0.989; `r2` R²=0.031). This identifies the bottleneck: the interaction takes non-collapsed image features through a multiplicative module but contributes almost no audio-specific information.

## Decision and next valid experiment

Do not proceed to residual interaction, Global–Local interaction, cross-attention, rank sweeps, or tri-modal expansion. Residual is not triggered because `r12` fails the prerequisite of depending on both modalities.

The next valid experiment is a task/data protocol in which audio is conditionally necessary, followed by the unchanged dependence, utility, shuffle, and probe gates. An instance-matching contrastive target is inappropriate here because AV-MNIST pairs repeated class labels rather than unique semantic instances.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_synergyformer -v
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.av_mnist.synergyformer model_type=synergy seed=1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.av_mnist.diagnose \
  +source_experiment=avmnist_utility_w020_official \
  +diagnostic_experiment=avmnist_utility_predictability \
  seed=1 run_linear_probes=false run_nonlinear_probes=false run_predictability_probes=true
```
