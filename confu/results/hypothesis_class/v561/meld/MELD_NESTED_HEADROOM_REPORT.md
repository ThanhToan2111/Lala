# MELD nested joint-class headroom audit

Frozen Product Joint plus one zero-initialized, one-hidden-layer concatenation residual. Product parameters are frozen.

Dataset metadata: `{'dataset': 'MELD', 'dataset_version': 'official declare-lab/MELD annotation CSVs plus MM-Align processed MELD pickle release', 'feature_dims': {'vision': 2048, 'audio': 32, 'text': 300}, 'sequence_handling': 'masked_mean over the complete utterance sequence; no dialogue context', 'text_representation': 'provided 300-dimensional embedding.p lookup followed by masked mean', 'standardization': 'train split mean/std per modality; validation and test never affect statistics', 'split_counts': {'train': 9988, 'valid': 1108, 'test': 2610}, 'split_feature_dims': {'train': {'vision': 2048, 'audio': 32, 'text': 300}, 'valid': {'vision': 2048, 'audio': 32, 'text': 300}, 'test': {'vision': 2048, 'audio': 32, 'text': 300}}}`.

## Product reference provenance

- `recreated_v56_product_reference`

## Capacity and validation result

| Mapping | Product params | Residual params | Residual ratio | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|---:|---:|
| `VA_to_T` | 53,976 | 12,205 | 0.226 | 0.0055 ± 0.0027 | 0.0058 ± 0.0027 | 0.0003 ± 0.0004 |
| `VT_to_A` | 107,656 | 26,223 | 0.244 | -0.0054 ± 0.0052 | -0.0016 ± 0.0046 | 0.0038 ± 0.0014 |
| `AT_to_V` | 79,784 | 18,715 | 0.235 | 0.0001 ± 0.0006 | 0.0002 ± 0.0006 | 0.0000 ± 0.0000 |

## Seed-level validation result

| Mapping | Seed | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | 0.0086 | 0.0088 | 0.0003 |
| `VA_to_T` | 2 | 0.0036 | 0.0036 | 0.0000 |
| `VA_to_T` | 3 | 0.0042 | 0.0049 | 0.0007 |
| `VT_to_A` | 1 | -0.0110 | -0.0060 | 0.0050 |
| `VT_to_A` | 2 | -0.0008 | 0.0032 | 0.0041 |
| `VT_to_A` | 3 | -0.0043 | -0.0020 | 0.0023 |
| `AT_to_V` | 1 | -0.0004 | -0.0003 | 0.0001 |
| `AT_to_V` | 2 | 0.0000 | 0.0000 | 0.0000 |
| `AT_to_V` | 3 | 0.0007 | 0.0008 | 0.0001 |

## Training and residual health

| Mapping | Seed | Initial val MSE | Best val MSE | Best epoch | Product R² val | Nested R² val | Residual norm val | Finite | Product grad max | Residual grad first |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `VA_to_T` | 1 | 0.991039 | 0.990786 | 1 | 0.0109 | 0.0112 | 0.3027 | True | 0.00e+00 | 5.19e-03 |
| `VA_to_T` | 2 | 0.989992 | 0.989992 | 0 | 0.0120 | 0.0120 | 0.0000 | True | 0.00e+00 | 5.12e-03 |
| `VA_to_T` | 3 | 0.991693 | 0.990987 | 2 | 0.0103 | 0.0110 | 0.5187 | True | 0.00e+00 | 5.47e-03 |
| `VT_to_A` | 1 | 0.962479 | 0.957693 | 3 | -0.0029 | 0.0021 | 0.4068 | True | 0.00e+00 | 1.83e-02 |
| `VT_to_A` | 2 | 0.957557 | 0.953640 | 4 | 0.0022 | 0.0063 | 0.4690 | True | 0.00e+00 | 1.92e-02 |
| `VT_to_A` | 3 | 0.955171 | 0.953001 | 2 | 0.0047 | 0.0070 | 0.2984 | True | 0.00e+00 | 1.61e-02 |
| `AT_to_V` | 1 | 1.004271 | 1.004214 | 1 | -0.0006 | -0.0006 | 0.3221 | True | 0.00e+00 | 1.85e-03 |
| `AT_to_V` | 2 | 1.003875 | 1.003875 | 0 | -0.0002 | -0.0002 | 0.0000 | True | 0.00e+00 | 1.87e-03 |
| `AT_to_V` | 3 | 1.003737 | 1.003646 | 1 | -0.0001 | -0.0000 | 0.2256 | True | 0.00e+00 | 2.03e-03 |

## Test result (descriptive)

| Mapping | J Product | J Nested | Δ Nested |
|---|---:|---:|---:|
| `VA_to_T` | 0.0035 ± 0.0044 | 0.0040 ± 0.0044 | 0.0005 ± 0.0005 |
| `VT_to_A` | -0.0012 ± 0.0050 | 0.0023 ± 0.0042 | 0.0034 ± 0.0008 |
| `AT_to_V` | 0.0005 ± 0.0012 | 0.0005 ± 0.0012 | 0.0001 ± 0.0001 |
