# ConFu++ v4.4 C0 reliability audit

Run date: 2026-09-18. Benchmark: IPIB, mappings 12-to-3, regimes I1-I4, five seeds. C0 is diagnostic only: reliability uses validation predictor errors; ground-truth interaction is read only after reliability is fixed.

## Reliability definition

For output dimension k:

    E_add,k   = mean((q_add,k - x3,k)^2) on validation
    E_joint,k = mean((q_joint,k - x3,k)^2) on validation
    R_k       = max(0, E_add,k - E_joint,k) / (E_add,k + epsilon)

The reliability score uses only x3, q_add, and q_joint. It does not access labels, task score, g12, projected interaction, or test data.

For the post-hoc audit, the fidelity target is the generated interaction contribution beta times B3(g12). This is scale-equivalent to the unscaled projected interaction B3(g12) within each fixed regime. The machine-readable artifact stores both direct and scale-corrected fidelity fields.

## Aggregate C0 result

| Regime | Pearson R-F | Spearman R-F | Positive Spearman seeds | Top-bottom fidelity gap | Zero reliability |
|---|---:|---:|---:|---:|---:|
| I1 weak | 0.289 ± 0.109 | **0.286 ± 0.094** | **5/5** | **+0.097 ± 0.047** | 0.000 |
| I2 medium | 0.753 ± 0.106 | **0.691 ± 0.132** | **5/5** | **+0.131 ± 0.039** | 0.000 |
| I3 strong | 0.146 ± 0.404 | 0.176 ± 0.331 | 3/5 | +0.011 ± 0.026 | 0.000 |
| I4 private noise | 0.198 ± 0.331 | 0.142 ± 0.261 | 4/5 | +0.019 ± 0.019 | 0.000 |

The primary weak/medium criterion passes: Spearman reliability-fidelity correlation is positive in every I1 and I2 seed, and the top reliability quartile has higher fidelity than the bottom quartile in every I1/I2 seed.

## Per-seed primary audit

| Regime | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
|---|---:|---:|---:|---:|---:|
| I1 Spearman | 0.181 | 0.308 | 0.433 | 0.263 | 0.247 |
| I1 fidelity gap | +0.075 | +0.101 | +0.171 | +0.090 | +0.046 |
| I2 Spearman | 0.698 | 0.701 | 0.774 | 0.472 | 0.811 |
| I2 fidelity gap | +0.093 | +0.132 | +0.150 | +0.095 | +0.187 |
| I3 Spearman | -0.057 | +0.229 | +0.716 | -0.113 | +0.107 |
| I4 Spearman | +0.293 | +0.047 | +0.460 | -0.231 | +0.143 |

The weaker I3/I4 consistency is not a C0 failure: strong interaction already has high fidelity in nearly all dimensions, leaving little ranking headroom; private noise also perturbs dimension-level reliability. The decision is driven by I1/I2 as required by the agent.

## Reliability distribution

| Regime | Mean R | Median R | Std R | Top-10% mass | Top25 fidelity | Middle50 fidelity | Bottom25 fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|
| I1 | 0.041 ± 0.001 | 0.041 ± 0.002 | 0.008 ± 0.001 | 0.128 ± 0.004 | 0.379 | 0.369 | 0.282 |
| I2 | 0.181 ± 0.004 | 0.183 ± 0.004 | 0.017 ± 0.002 | 0.106 ± 0.002 | 0.873 | 0.837 | 0.742 |
| I3 | 0.497 ± 0.007 | 0.499 ± 0.006 | 0.012 ± 0.005 | 0.097 ± 0.001 | 0.974 | 0.972 | 0.963 |
| I4 | 0.196 ± 0.004 | 0.197 ± 0.004 | 0.011 ± 0.001 | 0.102 ± 0.001 | 0.936 | 0.931 | 0.917 |

Reliability is non-degenerate: no regime has an exactly-zero reliability fraction, and the weak/medium regimes show clear top-versus-bottom separation.

## C0 decision

C0 passes for the intended weak/medium mechanism:

- I1 Spearman positive in 5/5 seeds;
- I2 Spearman positive in 5/5 seeds;
- I1 and I2 top-bottom fidelity gaps positive in 5/5 seeds;
- reliability computation is label-free and validation-only.

This justifies one and only one C1 test with alpha equal to 1. C0 does not justify reliability-weighting sweeps or any ground-truth-informed weighting.

Machine-readable results:

    results/synthetic_interaction/v44/c0_reliability_audit.json

