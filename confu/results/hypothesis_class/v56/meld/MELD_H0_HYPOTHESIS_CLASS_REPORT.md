# MELD H0 hypothesis-class audit

Frozen canonical pooled representations; additive, capacity-matched Product Joint, and one-hidden-layer Generic Joint MLP.

Dataset metadata: `{'dataset': 'MELD', 'dataset_version': 'official declare-lab/MELD annotation CSVs plus MM-Align processed MELD pickle release', 'feature_dims': {'vision': 2048, 'audio': 32, 'text': 300}, 'sequence_handling': 'masked_mean over the complete utterance sequence; no dialogue context', 'text_representation': 'provided 300-dimensional embedding.p lookup followed by masked mean', 'standardization': 'train split mean/std per modality; validation and test never affect statistics', 'split_counts': {'train': 9988, 'valid': 1108, 'test': 2610}, 'split_feature_dims': {'train': {'vision': 2048, 'audio': 32, 'text': 300}, 'valid': {'vision': 2048, 'audio': 32, 'text': 300}, 'test': {'vision': 2048, 'audio': 32, 'text': 300}}}`.

## Parameter audit

| Mapping | Additive | Product | Generic MLP | MLP mismatch vs additive |
|---|---:|---:|---:|---:|
| `VA_to_T` | 54,240 | 53,976 | 55,063 | 1.517% |
| `VT_to_A` | 108,694 | 107,656 | 109,558 | 0.795% |
| `AT_to_V` | 79,406 | 79,784 | 78,240 | 1.468% |

## Main validation result

| Mapping | Additive R² | Product R² | MLP R² | J Product | J MLP | ΔH |
|---|---:|---:|---:|---:|---:|---:|
| `VA_to_T` | 0.0056 ± 0.0030 | 0.0103 ± 0.0006 | 0.0100 ± 0.0016 | 0.0047 ± 0.0031 | 0.0044 ± 0.0022 | -0.0003 ± 0.0021 |
| `VT_to_A` | 0.0067 ± 0.0032 | 0.0026 ± 0.0022 | 0.0085 ± 0.0018 | -0.0041 ± 0.0045 | 0.0018 ± 0.0014 | 0.0060 ± 0.0032 |
| `AT_to_V` | -0.0005 ± 0.0003 | -0.0005 ± 0.0001 | -0.0007 ± 0.0002 | 0.0000 ± 0.0004 | -0.0002 ± 0.0003 | -0.0002 ± 0.0003 |

## Seed-level validation result

| Mapping | Seed | J Product | J MLP | ΔH |
|---|---:|---:|---:|---:|
| `VA_to_T` | 1 | 0.0082 | 0.0059 | -0.0023 |
| `VA_to_T` | 2 | 0.0024 | 0.0020 | -0.0004 |
| `VA_to_T` | 3 | 0.0036 | 0.0054 | 0.0019 |
| `VT_to_A` | 1 | -0.0080 | 0.0010 | 0.0090 |
| `VT_to_A` | 2 | 0.0008 | 0.0035 | 0.0026 |
| `VT_to_A` | 3 | -0.0052 | 0.0010 | 0.0063 |
| `AT_to_V` | 1 | -0.0002 | -0.0006 | -0.0004 |
| `AT_to_V` | 2 | -0.0003 | -0.0002 | 0.0001 |
| `AT_to_V` | 3 | 0.0005 | 0.0001 | -0.0004 |

## Descriptive test result

| Mapping | J Product | J MLP | ΔH |
|---|---:|---:|---:|
| `VA_to_T` | 0.0043 ± 0.0022 | 0.0033 ± 0.0035 | -0.0010 ± 0.0016 |
| `VT_to_A` | -0.0013 ± 0.0063 | 0.0013 ± 0.0053 | 0.0025 ± 0.0046 |
| `AT_to_V` | 0.0005 ± 0.0005 | 0.0007 ± 0.0004 | 0.0002 ± 0.0001 |

## Overfit diagnostic

| Mapping | J MLP train | J MLP val | Gap_M |
|---|---:|---:|---:|
| `VA_to_T` | -0.0048 ± 0.0067 | 0.0044 ± 0.0022 | -0.0092 ± 0.0049 |
| `VT_to_A` | -0.0087 ± 0.0173 | 0.0018 ± 0.0014 | -0.0105 ± 0.0180 |
| `AT_to_V` | 0.0018 ± 0.0009 | -0.0002 ± 0.0003 | 0.0020 ± 0.0008 |

## Training health

| Mapping | Predictor | Train R² | Val R² | Test R² | Best epoch | Finite |
|---|---|---:|---:|---:|---:|---|
| `VA_to_T` | `additive` | 0.0450 | 0.0056 | 0.0090 | 7.7 | True |
| `VA_to_T` | `product` | 0.0399 | 0.0103 | 0.0133 | 5.7 | True |
| `VA_to_T` | `mlp` | 0.0402 | 0.0100 | 0.0123 | 5.7 | True |
| `VT_to_A` | `additive` | 0.0896 | 0.0067 | 0.0255 | 2.7 | True |
| `VT_to_A` | `product` | 0.0801 | 0.0026 | 0.0243 | 2.0 | True |
| `VT_to_A` | `mlp` | 0.0809 | 0.0085 | 0.0268 | 2.0 | True |
| `AT_to_V` | `additive` | 0.0083 | -0.0005 | 0.0005 | 19.0 | True |
| `AT_to_V` | `product` | 0.0088 | -0.0005 | 0.0010 | 15.3 | True |
| `AT_to_V` | `mlp` | 0.0100 | -0.0007 | 0.0012 | 11.0 | True |

## Dataset interpretation

`HYPOTHESIS_CLASS_INCONCLUSIVE` is the dataset-level H0 interpretation. Decisions use validation only; test values are descriptive.
