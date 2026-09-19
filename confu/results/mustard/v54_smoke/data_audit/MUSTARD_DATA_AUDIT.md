# MUStARD data audit

Canonical source: `data/multibench/mustard_download/sarcasm.pkl`.

Feature dimensions: `{'vision': 371, 'audio': 81, 'text': 300}`.
Sequence handling: `masked_mean_nonzero_frames`.
Unique sample IDs across splits: `True`; duplicates: `0`.

| Split | N | Class 0 | Class 1 |
|---|---:|---:|---:|
| train | 414 | 208 | 206 |
| valid | 138 | 58 | 80 |
| test | 138 | 79 | 59 |

- `train` zero pooled vectors: {'vision': 2, 'audio': 10, 'text': 2}
- `valid` zero pooled vectors: {'vision': 1, 'audio': 3, 'text': 1}
- `test` zero pooled vectors: {'vision': 0, 'audio': 0, 'text': 0}