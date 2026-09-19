# Naturalized IPIB v6.0

Frozen MOSEI V/A representation geometry with a known additive component, product interaction component, private component, and no post-result tuning.

## Locked construction

- Beta ladder: `[0.0, 1.0]`; seeds: `[1]`; interaction rank: `32`; private strength: `0.25`.
- The source is the canonical MOSEI V/A cache. Fixed projection and generator matrices are in `fixed_generators.npz`; all split IDs and seeds are in `data_spec.json`.

## Predictor screen

| Beta | J validation | J test | Direct d→ground-truth R² | JAD h→ground-truth R² | Accessibility Δ |
|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.01176 ± nan | -0.00720 | None | None | 0.12763 |
| 1.00 | 0.05454 ± nan | 0.06417 | 0.1771584153175354 | 0.13037420809268951 | 0.11969 |

## Interpretation

The benchmark is a controlled bridge, not a natural ground-truth claim. Beta=0 is the false-positive control; increasing beta is expected to increase predictive headroom and recovery, but monotonicity is evaluated descriptively. Accessibility labels depend on the injected interaction and are never used to train the predictor or JAD. Recovery uses one fixed Ridge(alpha=1) and accessibility uses one fixed LogisticRegression(C=1) fit on train+validation; no estimator sweep is performed.
