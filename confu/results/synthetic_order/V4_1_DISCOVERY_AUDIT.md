# ConFu++ v4.1 discovery audit

Run date: 2026-09-18. This audit follows `AGENT.md — ConFu++ v4.1_ Discovery of Explicit Higher-Order Multimodal Representations.md` and uses the existing v4 synthetic Oracle artifacts.

## Executive decision

| Question | Status |
|---|---|
| Q1 architecture capability | PASS from v4 Oracle |
| Phase A generic MLP sanity | Incomplete/weak: matched shallow MLP is near chance on dense S3 |
| Phase B Oracle health/intervention | PASS for active pair/triple branches |
| D0 Original ConFu discovery | BLOCKED by target identifiability |
| D1 residual discovery | BLOCKED by target identifiability |
| D2 joint-advantage distillation | BLOCKED by target identifiability |
| Q3 real-world transfer | BLOCKED |

The blocker is in the v4.1 protocol, not in the multiplicative architecture.

## Phase A — generic MLP sanity

The S3 label is a dense 32-term third-order rule. The fixed one-hidden-layer MLP was evaluated on raw observations, exact latent vectors, and the learned first-order Oracle representation. Each size used the same optimizer and early stopping protocol.

| Input | Hidden | Parameters | Train acc. | Validation acc. | Test acc. | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 128 | 24,962 | 55.87% | 50.26% | 49.68% | 1 |
| Raw | 256 | 49,922 | 57.55% | 51.10% | 50.00% | 1 |
| Raw | 512 | 99,842 | 57.80% | 50.28% | 49.24% | 1 |
| Latent | 128 | 12,674 | 53.52% | 50.36% | 49.10% | 0 |
| Latent | 256 | 25,346 | 55.20% | 51.40% | 49.80% | 1 |
| Latent | 512 | 50,690 | 56.40% | 49.74% | 50.36% | 1 |
| Learned first-order | 128 | 12,674 | 53.50% | 49.90% | 49.32% | 0 |
| Learned first-order | 256 | 25,346 | 55.27% | 51.32% | 50.00% | 1 |
| Learned first-order | 512 | 50,690 | 56.53% | 49.86% | 50.08% | 1 |

Train accuracy is also low, so this is not a train/test generalization gap. It is a capacity/optimization limitation for the current shallow generic MLP. The valid current claim remains that Oracle explicit interaction features make S3 linearly accessible; a claim that they are more powerful than a generic MLP is not yet justified.

## Phase B — Oracle diagnostics

The new diagnostics include variance, dimension-wise standard deviation, mean norm, effective rank, target cosine, and modality interventions.

### S1 pair 12

`h12` has variance `0.962`, mean dimension standard deviation `0.981`, mean norm `5.549`, effective rank `30.90`, and target cosine `0.981`.

Shuffling modality 1 reduced the clean linear-probe accuracy from `93.02%` to `56.02%` (`-37.00 pp`). Shuffling modality 2 reduced it to `56.38%` (`-36.64 pp`). Both inputs are therefore used by the active pair branch.

### S3 triple

`h123` has variance `0.895`, mean dimension standard deviation `0.942`, mean norm `5.355`, effective rank `30.54`, and target cosine `0.962`.

Independent modality interventions produced drops of `41.92 pp`, `42.22 pp`, and `42.26 pp` for modalities 1, 2, and 3 respectively. The Oracle triple depends on all three sources.

The inactive Oracle branches are exactly zero by design. That is valid for the Oracle upper-bound experiment, but discovery experiments must not reuse this hard gating.

## Target identifiability audit

The v4.1 discovery target for pair `12_to_3` is the target modality representation `r3`; the other mappings are `13_to_2` and `23_to_1`. The generator, however, samples `z1`, `z2`, and `z3` independently.

Even with exact latent variables supplied to the predictor, Ridge regression gives:

| Regime | `12 → 3` R² | `13 → 2` R² | `23 → 1` R² |
|---|---:|---:|---:|
| S0 | -0.0037 | -0.0033 | -0.0039 |
| S1 | -0.0037 | -0.0033 | -0.0039 |
| S2 | -0.0037 | -0.0033 | -0.0039 |
| S3 | -0.0037 | -0.0033 | -0.0039 |

The values are finite-sample noise around population R² zero. Additive and joint predictors have the same result, with joint-advantage variance only about `5.7e-6` and mean norm about `0.04`.

This is a structural impossibility for the proposed D0/D1/D2 target:

```text
z3 independent of (z1,z2)
=> r3 unpredictable from (r1,r2)
=> q_joint(r1,r2) - q_add(r1,r2) ≈ 0
=> no D2 signal to distill into h12
```

The fact that S1 labels contain a real pair-12 term does not solve this. The pair signal exists in the label-generating function, while D0/D1/D2 are forbidden from using the label and are trained against an independent target modality.

## Why D0/D1/D2 were not run

Running them now would create a misleading negative result. A near-zero D0/D1/D2 gain would only show that the target modality is statistically independent, not that ConFu alignment, residual alignment, or Joint-Advantage Distillation cannot discover interaction.

This is not fixed by:

- increasing rank;
- adding cross-attention;
- changing the optimizer;
- adding task loss while still calling the experiment unsupervised;
- increasing predictor capacity.

The missing quantity is target information.

## Required protocol decision

One of these changes must be selected before implementing D0/D1/D2:

1. Keep iid modalities, but allow task/label-aware discovery. Then D2 can target the joint advantage for predicting the synthetic task score rather than an independent `r_k`.
2. Keep target-modality discovery, but change the synthetic generator so the target modality contains predictable shared/joint structure. This invalidates the current iid assumption and needs a new named benchmark protocol.
3. Keep both the iid generator and no task loss, and explicitly declare D0/D1/D2 as a negative identifiability control rather than a discovery benchmark.

Option 1 best matches the current S0–S4 purpose: the labels already contain order-controlled signal. Option 2 best preserves the original ConFu target-modality formulation. Neither should be applied silently.

## Current bottleneck

The project has moved past architecture capacity. The bottleneck is now:

```text
discovery target construction / identifiability
```

The generic MLP issue is secondary and should be solved as a baseline-calibration experiment, but it does not explain the zero D2 target.

## Reproducibility

Implemented audit tools:

- `src/experiments/synthetic_order/mlp_sanity.py`
- `src/experiments/synthetic_order/diagnostics.py`
- `src/experiments/synthetic_order/identifiability.py`

Artifacts:

- `results/synthetic_order/mlp_sanity_s3_seed_1.json`
- `results/synthetic_order/oracle/diagnostics_s1_seed_1.json`
- `results/synthetic_order/oracle/diagnostics_s3_seed_1.json`
- `results/synthetic_order/discovery/identifiability.json`

Checks:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_synthetic_order tests.test_synergyformer -v
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.synthetic_order.mlp_sanity
PYTHONPATH=. .venv/bin/python -m src.experiments.synthetic_order.identifiability --seeds 1
```
