# v4.1 discovery identifiability audit

## Result

The v4.1 D0/D1/D2 target protocol is not identifiable on the current synthetic generator.

The generator makes `z1`, `z2`, and `z3` independent. However, D0, D1, and D2 define the target for `h12` through the target modality `r3`:

```text
h12 ↔ r3
h13 ↔ r2
h23 ↔ r1
```

For independent latent variables, neither an additive predictor nor a joint predictor from `(z1,z2)` can predict `z3`. The same applies to the other two mappings. Therefore:

```text
R²_joint ≈ R²_additive ≈ 0
joint advantage ≈ 0
```

This remains true on S1, where the classification label contains a real pair-12 signal. The signal is in the label-generating function, not in the target modality representation used by D0/D1/D2.

## Evidence

Ridge audit from the exact latent inputs to the target modality produced approximately:

| Regime | 12→3 | 13→2 | 23→1 |
|---|---:|---:|---:|
| S0 | -0.0037 | -0.0033 | -0.0039 |
| S1 | -0.0037 | -0.0033 | -0.0039 |
| S2 | -0.0037 | -0.0033 | -0.0039 |
| S3 | -0.0037 | -0.0033 | -0.0039 |

The small negative values are finite-sample/generalization noise around the population value zero. Using exact latent variables is already the strongest possible test; learned encoders cannot recover a target dependency that is absent from the data.

## Consequence

Running D0/D1/D2 without changing the protocol would not be a valid discovery comparison. A failure would prove only that the discovery target has no signal, not that ConFu alignment, residual alignment, or Joint-Advantage Distillation is ineffective.

The current v4.1 milestone is therefore blocked at target construction, not architecture capacity.

## Required protocol choice before D0/D1/D2

Choose one scientifically explicit correction:

1. Keep independent modalities and define discovery targets from a task signal. This requires allowing task/label-aware discovery, which v4.1 currently forbids.
2. Keep target-modality discovery and change the generator so the target modality contains predictable shared/joint structure. This changes the current iid synthetic assumption and must be documented.
3. Keep both the iid generator and no task loss, but accept that D0/D1/D2 are a deliberate negative identifiability control rather than a discovery benchmark.

Until this choice is made, do not report a D0/D1/D2 failure as a ConFu++ result and do not move to UR-FUNNY.
