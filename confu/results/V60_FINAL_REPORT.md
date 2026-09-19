# ConFu++ v6.0 final experiment report

Run date: 2026-09-19. This report consolidates T1 formalization, T2 null calibration, T3 Naturalized IPIB, and T4 paper freeze. v6.0 is a diagnostic and paper-consolidation release; it does not introduce a new fusion architecture.

## Executive decision

The project now has a coherent evidence chain:

```text
representation and data
  -> additive versus joint prediction
  -> cross-modal predictive identifiability
  -> downstream task headroom
  -> interaction distillation
  -> fidelity
  -> accessibility
```

The strongest conclusion is conditional, not architectural: a positive predictive interaction score does not guarantee downstream task improvement. The canonical MOSEI VA→T setting has positive cross-modal advantage (`J=0.01983` validation mean), but the distilled interaction D2 is effectively tied with the lower-order sentiment probe (`-0.034 ± 0.099` percentage points). MUStARD and MELD do not provide a consistent positive natural screen under the frozen protocol.

## T1 — Formal projection theory

The operational score is defined in `L²(P_X; R^d)` under squared prediction risk. For closed nested hypothesis spaces `H_A ⊆ H_J`, the population risk gap is:

```text
R(f_A*) - R(f_J*) = || Π_J m - Π_A m ||²,
```

where `m(x)=E[Y|X=x]` and `Π` denotes the corresponding orthogonal projection. The vector-output empirical score uses variance-weighted global `R²`; finite neural fits are only approximations to the population projections.

This formalization explicitly does not identify PID atoms, causal effects, “true synergy,” or information created by fusion. It is a representation- and hypothesis-class-relative predictive diagnostic.

## T2 — Null calibration

The alignment-breaking null independently shuffles the source modality within train, validation, and test splits. The historical gate `τ=0.01` is retained for comparability and is not changed using test results. The run used `B=10` per setting because the larger null was compute-prohibitive; therefore the quantiles are a finite-sample audit, not a final 1%-level calibration.

| Setting | Mean J | Std | Median | q95 | q99 | Max | q99 > 0.01 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| IPIB I0 | -0.02034 | 0.00104 | -0.02057 | -0.01887 | -0.01846 | -0.01836 | No |
| MOSEI VA→T | 0.01227 | 0.00563 | 0.01113 | 0.02118 | 0.02263 | 0.02299 | **Yes** |
| MELD VA→T | 0.00374 | 0.00154 | 0.00351 | 0.00557 | 0.00571 | 0.00574 | No |

The key bottleneck is calibration: on MOSEI, the null q99 is above the historical gate. Consequently, `τ=0.01` should be described as an operational historical threshold, not as a controlled 1% false-positive threshold. Sensitivity at `0.005`, `0.01`, and `0.02` is recorded without changing prior PASS/FAIL labels.

## T3 — Naturalized IPIB

Frozen canonical MOSEI V/A representations were projected into a fixed 64-dimensional source geometry. The target combines a fixed additive component, a known rank-32 product interaction, and private noise. The interaction strength ladder is β=`{0, 0.25, 0.5, 1}` over five seeds.

| β | J validation | J test | direct d→βj R² | JAD h→βj R² | accessibility Δ |
|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.01427 ± 0.00667 | -0.02499 | — | — | 0.09550 ± 0.01541 |
| 0.25 | 0.00262 ± 0.00172 | -0.01247 | 0.12126 | 0.12391 | 0.10914 ± 0.00837 |
| 0.50 | 0.05787 ± 0.00748 | 0.05002 | 0.29827 | 0.19089 | 0.14106 ± 0.01930 |
| 1.00 | 0.22625 ± 0.01462 | 0.24367 | 0.41480 | 0.25957 | 0.17898 ± 0.01555 |

Interpretation:

- The predictor screen behaves as intended: β=0 is a useful J false-positive control, and J increases strongly by β=0.5 and β=1.
- Direct recovery of the known interaction increases with β, confirming that the learned difference contains increasing interaction signal.
- JAD recovers only part of the ground-truth interaction (`R²=0.25957` at β=1), so target fidelity is not perfect.
- The accessibility task is defined from the interaction latent itself. Therefore the positive β=0 accessibility delta is not an accessibility null; it shows that target-side absence of interaction and task-side interaction relevance are separate axes.

## Natural benchmark comparison

| Audit | Result | Meaning |
|---|---:|---|
| MOSEI VA→T G1 | `J=0.01983 ± 0.00187` | Positive representation-conditional predictive interaction |
| MOSEI sentiment after JAD | `-0.034 ± 0.099` pp | No downstream accuracy gain over lower-order baseline |
| MUStARD G1 | all tested mappings ≤ `-0.00568` mean | No positive natural screen |
| MELD G1 | VA→T `0.00197 ± 0.00412` | Below historical gate and unstable |
| MELD v5.6.1 nested headroom | max mean Δ `0.0038` | Modest residual does not reveal substantial extra headroom |

The best available natural result is therefore a fidelity–accessibility separation, not a superiority result for ConFu++.

## Bottleneck diagnosis

1. **Threshold calibration is not transportable.** MOSEI's alignment null overlaps the historical positive region. A larger preregistered null, preferably B≥50/100, is required before interpreting `J>0.01` as a calibrated discovery gate.
2. **Task relevance is the main natural bottleneck.** MOSEI D2 fits the interaction target well (`cosine 0.848 ± 0.038` in the prior R2 audit) and is affected by both modality shuffles, yet does not improve sentiment accuracy. The interaction is measurable but not reliably useful to the task.
3. **Interaction fidelity is incomplete.** Naturalized IPIB direct recovery and JAD recovery rise with β but remain far from perfect, especially at β=1. A learned interaction representation can be predictive without being a faithful copy of the injected target.
4. **Accessibility is probe/task-relative.** The Naturalized IPIB accessibility task is intentionally interaction-defined, so it must not be used to claim a beta-zero task null. Natural real-world Type-IV evidence—positive identifiability, positive task headroom, and positive accessibility—has not been established.
5. **Architecture is not the current remedy.** The v5.6.1 nested audit found no substantial MELD residual headroom beyond Product Joint; generic joint-class evidence was inconclusive because its positive control failed. More architecture fishing would confound the scientific question.

## Final acceptance

| Requirement | Status |
|---|---|
| Formal population definition and projection result | Pass |
| Vector-output and finite-sample caveats | Pass |
| Null distribution and threshold sensitivity artifacts | Pass, compute-limited B=10 |
| Naturalized IPIB with fixed real source geometry | Pass |
| Five-seed β ladder and ground-truth recovery | Pass |
| Original ConFu/natural screening comparison | Pass |
| Nested robustness result included | Pass |
| Claims, limitations, related-work, reviewer matrix | Pass |
| All core figures reproducible from JSON | Pass |
| Evidence that ConFu++ beats the strongest natural baseline | **Fail / not claimed** |

The correct paper claim is:

> Multimodal interaction learning should be evidence-driven: interaction must be measurable beyond additive prediction, relevant to the downstream task, representable with sufficient fidelity, and accessible to the task learner.

## Reproducibility artifacts

- [T1 projection theory](/home/linhkastner/Depth-Anything-3/confu/paper/theory/projection_theory.md)
- [T2 null report](/home/linhkastner/Depth-Anything-3/confu/results/calibration/v60/NULL_CALIBRATION_REPORT.md)
- [T2 null distributions](/home/linhkastner/Depth-Anything-3/confu/results/calibration/v60/null_distributions.json)
- [T3 Naturalized IPIB report](/home/linhkastner/Depth-Anything-3/confu/results/naturalized_ipib/v60/NATURALIZED_IPIB_REPORT.md)
- [v5.6.1 nested decision](/home/linhkastner/Depth-Anything-3/confu/results/hypothesis_class/v561/NESTED_JOINT_CLASS_DECISION.md)
- [paper freeze](/home/linhkastner/Depth-Anything-3/confu/PAPER_EXPERIMENT_FREEZE.md)
- [experiment registry](/home/linhkastner/Depth-Anything-3/confu/results/registry.json)
