# Naturalized IPIB v6.0

Frozen MOSEI V/A representation geometry with a known additive component, product interaction component, private component, and no post-result tuning.

## Locked construction

- Beta ladder: `[0.0, 0.25, 0.5, 1.0]`; seeds: `[1, 2, 3, 4, 5]`; interaction rank: `32`; private strength: `0.25`.
- The source is the canonical MOSEI V/A cache. Fixed projection and generator matrices are in `fixed_generators.npz`; all split IDs and seeds are in `data_spec.json`.

## Predictor screen

| Beta | J validation | J test | Direct d→ground-truth R² | JAD h→ground-truth R² | Accessibility Δ |
|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.01427 ± 0.00667 | -0.02499 | None | None | 0.09550 |
| 0.25 | 0.00262 ± 0.00172 | -0.01247 | 0.12126167118549347 | 0.12390999048948288 | 0.10914 |
| 0.50 | 0.05787 ± 0.00748 | 0.05002 | 0.2982706785202026 | 0.19088953137397766 | 0.14106 |
| 1.00 | 0.22625 ± 0.01462 | 0.24367 | 0.41479982137680055 | 0.259568852186203 | 0.17898 |

## Interpretation

The benchmark is a controlled bridge, not a natural ground-truth claim. Beta=0 is the false-positive control for the representation-target score J: its target contains no injected interaction. The optional accessibility task is deliberately defined from the interaction score itself, so its beta=0 accessibility delta is not a null for task utility; it demonstrates that target-side absence and task-side interaction relevance are separate axes. Increasing beta is expected to increase predictive headroom and recovery, but monotonicity is evaluated descriptively. Accessibility labels are never used to train the predictor or JAD. Recovery uses one fixed Ridge(alpha=1) and accessibility uses one fixed LogisticRegression(C=1) fit on train+validation; no estimator sweep is performed.
