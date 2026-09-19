# v6.1-C1 MOSEI null calibration confirmation

Status: preregistered before execution on 2026-09-19.

## Question

Does the frozen MOSEI `VA_to_T` empirical joint predictive advantage exceed most replicates of the previously defined alignment-breaking null?

This is a calibration confirmation, not a method experiment. It cannot change the architecture, representation, historical gate, or prior labels.

## Frozen observed statistic

The observed statistic is the mean validation `J` across the three frozen MOSEI `VA_to_T` records in:

`results/mosei/identifiability_exact/mosei_identifiability.json`

The observed value is read before null execution and is expected to be approximately `0.01982565`. Test metrics are not used.

## Null and data

- Dataset: canonical MOSEI pooled representation cache.
- Mapping: vision + audio → text (`VA_to_T`).
- Standardization: existing train-only standardization from the frozen v6.0 pipeline.
- Null: independently permute the right/source modality inside each train, validation, and test split.
- Every replicate receives a deterministic permutation seed and the same capacity-controlled Additive and Product predictors as v6.0.
- No new feature source, pooling, rank, optimizer, or architecture is allowed.

## Replicates and training

- Replicates: `B=50`, the minimum worthwhile confirmation specified by v6.1.
- Predictor training: batch size 512, maximum 80 epochs, patience 8, AdamW learning rate `1e-3`, weight decay `1e-4`.
- Model initialization/training seeds: `1000 + replicate_index`, matching the v6.0 null implementation.
- Selection split: validation only.
- The test split is loaded only because the common runner evaluates all splits; it is not used for selection, thresholding, or the decision rule.

## Prespecified statistics

Report mean, sample standard deviation, median, q90, q95, q99, and maximum of null validation `J`. Also report, with the +1 correction:

```text
p_hat(tau) = (1 + count(J_null >= 0.01)) / (B + 1)
p_hat(observed) = (1 + count(J_null >= observed_J)) / (B + 1)
```

The observed value is interpreted as operationally positive if it remains positive under the frozen predictor comparison. A small `p_hat(observed)` means only that the observed score exceeds most replicates of this operational null; it is not a universal statistical interaction claim.

## Decision-independent interpretation

- If the observed score exceeds most null replicates: report “operational positive and separated from this alignment-breaking null,” without changing the historical gate.
- Otherwise: report “operational positive under the frozen predictor comparison but not separated from the alignment-breaking null.”
- In either case, `tau=0.01` remains the historical operational threshold and is not retroactively recalibrated.
