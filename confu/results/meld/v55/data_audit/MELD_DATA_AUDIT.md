# MELD data audit

Official MELD emotion annotations joined to the processed MM-Align utterance features.

Feature source: `data/multibench/meld_download/processed`.
Sequence handling: `masked_mean over the complete utterance sequence; no dialogue context`.
Text: `provided 300-dimensional embedding.p lookup followed by masked mean`.
Seven-class mapping: `{'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}`.
Unique split-qualified sample IDs: `True`; duplicates: `0`.

| Split | Features | Official rows | Dialogues | Speakers | Length min/mean/max |
|---|---:|---:|---:|---:|---:|
| train | 9988 | 9989 | 1038 | 260 | 1/10.02/88 |
| valid | 1108 | 1109 | 114 | 47 | 1/10.04/49 |
| test | 2610 | 2610 | 280 | 100 | 1/10.35/55 |

| Split | Neutral | Surprise | Fear | Sadness | Joy | Disgust | Anger |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 4709 | 1205 | 268 | 683 | 1743 | 271 | 1109 |
| valid | 469 | 150 | 40 | 111 | 163 | 22 | 153 |
| test | 1256 | 281 | 50 | 208 | 402 | 68 | 345 |

The feature package omits any annotation row without a matching feature. The official CSV label is authoritative; feature-pickle label discrepancies are reported, not used for training.

- `train` missing feature IDs: `['125_3']`; feature-label mismatch IDs: `['717_6']`; zero pooled vectors: `{'vision': 0, 'audio': 0, 'text': 0}`.
- `valid` missing feature IDs: `['110_7']`; feature-label mismatch IDs: `[]`; zero pooled vectors: `{'vision': 0, 'audio': 0, 'text': 0}`.
- `test` missing feature IDs: `[]`; feature-label mismatch IDs: `[]`; zero pooled vectors: `{'vision': 0, 'audio': 0, 'text': 0}`.