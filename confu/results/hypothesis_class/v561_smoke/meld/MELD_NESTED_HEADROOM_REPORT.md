# MELD nested joint-class headroom audit

Frozen Product Joint plus one zero-initialized, one-hidden-layer concatenation residual. Product parameters are frozen.

Dataset metadata: `{'dataset': 'MELD', 'dataset_version': 'official declare-lab/MELD annotation CSVs plus MM-Align processed MELD pickle release', 'feature_dims': {'vision': 2048, 'audio': 32, 'text': 300}, 'sequence_handling': 'masked_mean over the complete utterance sequence; no dialogue context', 'text_representation': 'provided 300-dimensional embedding.p lookup followed by masked mean', 'standardization': 'train split mean/std per modality; validation and test never affect statistics', 'split_counts': {'train': 9988, 'valid': 1108, 'test': 2610}, 'split_feature_dims': {'train': {'vision': 2048, 'audio': 32, 'text': 300}, 'valid': {'vision': 2048, 'audio': 32, 'text': 300}, 'test': {'vision': 2048, 'audio': 32, 'text': 300}}}`.

## Capacity and validation result

| Mapping | Product params | Residual params | Residual ratio | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|---:|---:|
| `VA_to_T` | 53,976 | 12,205 | 0.226 | -0.0041 ± 0.0000 | 0.0005 ± 0.0000 | 0.0046 ± 0.0000 |

## Seed-level validation result

| Mapping | Seed | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | -0.0041 | 0.0005 | 0.0046 |

## Training and residual health

| Mapping | Seed | Initial val MSE | Best val MSE | Best epoch | Product R² val | Nested R² val | Residual norm val | Finite | Product grad max | Residual grad first |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `VA_to_T` | 1 | 1.003684 | 0.999063 | 3 | -0.0017 | 0.0029 | 0.9938 | True | 0.00e+00 | 6.15e-03 |

## Test result (descriptive)

| Mapping | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|
| `VA_to_T` | -0.0048 ± 0.0000 | -0.0012 ± 0.0000 | 0.0037 ± 0.0000 |
