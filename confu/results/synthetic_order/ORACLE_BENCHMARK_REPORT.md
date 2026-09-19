# ConFu++ v4 synthetic Oracle benchmark

Run date: 2026-09-18. The benchmark follows `AGENT.md — ConFu++ v4_ Synthetic-First Higher-Order Representation Learning.md`.

## Scope and protocol

This is a mechanism validation benchmark, not a real-world superiority claim. It separates architecture capability from discovery-objective capability before changing ConFu or returning to UR-FUNNY.

- Latents: three independent Rademacher vectors, dimension 32.
- Observations: fixed orthogonal projections to dimension 64; clean setting, `noise_std=0`.
- Splits: 20,000 train / 5,000 validation / 5,000 test, one fixed world per run.
- Regimes: S0 first-order, S1 pair 12, S2 all pairs, S3 pure triple, S4 mixed.
- Oracle: normalized low-rank multiplicative pair/triple modules, rank 64, direct MSE targets `g12`, `g13`, `g23`, and `g123`.
- Probe: standardized validation-selected linear probe is primary; parameter-matched MLP is secondary.
- Confirmatory seeds: 1–5 for primary linear metrics. Full MLP probes are retained for exploratory seed 1.

The explicit mappings are `12_to_3`, `13_to_2`, and `23_to_1`; no index-based target mapping is used.

## Five-seed primary result

Numbers are test accuracy (%) mean ± sample standard deviation. Δ values are percentage-point gains.

| Regime | Linear Z1 | Linear Z2 | Linear Z3 | Δ2 | Δ3 |
|---|---:|---:|---:|---:|---:|
| S0 first-order | 98.376 ± 0.161 | 98.376 ± 0.161 | 98.376 ± 0.161 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| S1 pair 12 | 63.944 ± 0.562 | 92.948 ± 0.569 | 92.948 ± 0.569 | 29.004 ± 1.033 | 0.000 ± 0.000 |
| S2 all pairs | 64.476 ± 0.700 | 93.932 ± 2.341 | 93.932 ± 2.341 | 29.456 ± 1.940 | 0.000 ± 0.000 |
| S3 pure triple | 50.144 ± 0.284 | 50.144 ± 0.284 | 93.504 ± 1.395 | 0.000 ± 0.000 | 43.360 ± 1.185 |
| S4 mixed | 60.300 ± 0.875 | 74.812 ± 1.270 | 92.064 ± 3.597 | 14.512 ± 1.356 | 17.252 ± 2.914 |

Raw concatenated observations are a useful non-interaction baseline: linear accuracy is 99.820 ± 0.020 on S0, 64.024 ± 0.596 on S1, 64.420 ± 0.626 on S2, 50.160 ± 0.282 on S3, and 60.248 ± 0.750 on S4. The raw baseline cannot expose the pair/triple structure to a linear probe; the Oracle embeddings do.

## Pair selectivity

Pair-specific linear gains over Z1, in percentage points:

| Regime | Δ12 | Δ13 | Δ23 |
|---|---:|---:|---:|
| S0 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| S1 | **29.004 ± 1.033** | 0.000 ± 0.000 | 0.000 ± 0.000 |
| S2 | 9.096 ± 0.752 | 9.160 ± 0.575 | 8.936 ± 1.270 |
| S3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| S4 | 6.084 ± 0.929 | 6.332 ± 0.291 | 5.356 ± 0.727 |

S1 has the required correct-pair selectivity: Δ12 is positive in all five seeds, while both inactive pairs are exactly zero because their branches are disabled. S2 and S4 show all active pairs contributing.

## Direct Oracle target fit

For active components, five-seed mean test metrics are:

| Regime | Active target | MSE | Cosine |
|---|---|---:|---:|
| S1 | h12 → g12 | 0.0369 | 0.9814 |
| S2 | h12/h13/h23 → g12/g13/g23 | 0.0437–0.0472 | 0.9771–0.9786 |
| S3 | h123 → g123 | 0.0561 | 0.9720 |
| S4 | h12/h13/h23 → pairs | 0.0436–0.0516 | 0.9748–0.9786 |
| S4 | h123 → g123 | 0.0758 | 0.9617 |

Inactive interaction outputs have zero variance and zero mean norm. The direct target fit and accessibility gains agree: this is not a variance-only or rank-only artifact.

## Secondary MLP probe, exploratory seed 1

| Regime | MLP Z1 | MLP Z2 | MLP Z3 | Δ2 | Δ3 | Linearization gap Z1/Z2/Z3 |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 97.92 | 98.00 | 98.04 | 0.08 | 0.04 | -0.38 / -0.30 / -0.26 |
| S1 | 89.18 | 93.40 | 93.62 | 4.22 | 0.22 | 25.46 / 0.38 / 0.60 |
| S2 | 85.88 | 95.44 | 95.72 | 9.56 | 0.28 | 21.62 / 1.52 / 1.44 |
| S3 | 49.32 | 49.62 | 91.34 | 0.30 | 41.72 | -0.74 / -0.44 / -0.32 |
| S4 | 68.48 | 84.72 | 92.48 | 16.24 | 7.76 | 7.72 / 11.28 / 6.30 |

The raw-feature MLP accuracies for seed 1 were S0 98.64%, S1 87.50%, S2 82.76%, S3 50.08%, and S4 67.80%. The matched generic MLP is weak on the dense 32-term S3 third-order rule, whereas the explicit `h123` representation raises linear S3 accuracy to 91.66% and MLP S3 accuracy to 91.34%.

This is a diagnostic limitation, not an Oracle-gate failure: the task is learnable by the explicit order representation and the five-seed primary linear pattern is exact. Before treating a generic MLP as a strong end-to-end baseline, its capacity/optimization protocol should be isolated in a follow-up sanity experiment.

## Acceptance decision

| Gate | Result |
|---|---|
| Seed determinism, label balance, split independence | Pass; 6 synthetic unit tests cover these controls |
| Explicit pair/triple target mapping | Pass; direct tensor tests and named branches |
| S0 no false pair/triple gain | Pass in 5/5 seeds |
| S1 correct-pair selectivity | Pass in 5/5 seeds |
| S2 Δ2 > 0 and Δ3 ≈ 0 | Pass in 5/5 seeds |
| S3 Δ3 > 0 and Δ2 ≈ 0 | Pass in 5/5 seeds |
| S4 Linear Z1 < Z2 < Z3 | Pass in 5/5 seeds |
| Direct Oracle pair/triple reconstruction | Pass; cosine 0.96–0.98 on active targets |
| Generic MLP S3 sanity | Warning; only near chance under current matched probe |

## Bottleneck and next step

The architecture bottleneck is cleared. The controlled evidence shows that normalized low-rank multiplicative pair and triple modules can expose the correct order-specific signal to a linear probe without inventing inactive orders.

The remaining bottleneck is objective validation, not architecture: no unsupervised discovery loss has been tested yet. The generic MLP S3 control also needs a separate capacity/optimization sanity check before being used as a headline baseline. Therefore the next allowed experiment is D0 original ConFu on S0–S4, followed by D1 residual alignment and only then D2 joint-advantage distillation. Do not change rank, add attention, or claim real-world improvement from this Oracle result.

## Reproducibility

Implementation:

- `src/datasets/synthetic_order.py`
- `src/experiments/synthetic_order/oracle.py`
- `tests/test_synthetic_order.py`

Commands:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_synthetic_order tests.test_synergyformer -v
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.synthetic_order.oracle --seeds 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m src.experiments.synthetic_order.oracle --seeds 5 --linear-only --out results/synthetic_order/oracle_linear
```

Full seed-1 JSON/checkpoints are under `results/synthetic_order/oracle/`; five-seed primary JSON/checkpoints are under `results/synthetic_order/oracle_linear/`.
