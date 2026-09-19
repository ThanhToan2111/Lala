# UR-FUNNY ConFu++ v2 benchmark and bottleneck report

Run date: 2026-09-18. Dataset: official MultiBench UR-FUNNY packed data. The
binary humor task has train/validation/test sizes `8074/1034/1058`; modality
order is vision/audio/text with padded inputs `[B,50,371]`, `[B,50,81]`, and
`[B,50,300]`.

## Objective

ConFu++ v2 follows the utility-first specification. The question is whether
original ConFu pair outputs provide conditional utility beyond the lower-order
representation, and whether a small frozen-backbone residual can exploit that
utility without damaging the baseline.

The v2 path intentionally does not add S1 adversarial de-shortcutting,
dynamic gates, cross-attention, rank expansion, context reconstruction, or
`r123`. The packed `humor.pkl` loader exposes aligned padded modality features,
but no context/punchline boundary, so a context-specific model is not
justified by the available representation.

## Matched baseline

The reference is the reproduced original pairwise ConFu checkpoint, evaluated
with the strict frozen linear probe on the official split and five seeds:

| Model | Test accuracy (%) |
|---|---:|
| Original ConFu | **64.462 ± 0.713** |
| Earlier ConFu++ S1 + task head | 63.549 ± 0.855 |

The original ConFu is the strongest valid baseline. Its seed-1 strict
all-modality score is `65.469%`.

The baseline interaction audit already showed the core problem: pair outputs
react to input permutations, but their conditional utility is not stable.
Seed-1 conditional gains for vision/audio, vision/text, and audio/text were
`-0.662`, `+0.189`, and `-0.756` percentage points, while lower-order to
interaction R² was `0.963`, `0.970`, and `0.970`. The interactions are active,
not collapsed, but mostly redundant with lower-order information.

## E0: pair-augmented frozen probes

E0 reuses the original ConFu checkpoints and changes only the downstream
probe. It compares:

```text
Z_low   = [r1; r2; r3]
Z_pairs = [r1; r2; r3; r12; r13; r23]
```

The linear probe is the existing sklearn fast-search implementation. The
nonlinear probe is a GPU MLP selected by validation loss, with hidden width
matched to approximately the same parameter budget as the lower-only probe.
No test labels select epochs or hyperparameters.

### Five-seed E0 results

| Probe | Lower `Z_low` | Pair-augmented `Z_pairs` | Gain (pp) |
|---|---:|---:|---:|
| Linear | 64.544 ± 0.553 | 64.527 ± 1.122 | **-0.017 ± 1.179** |
| Capacity-matched nonlinear | 60.151 ± 1.166 | 61.342 ± 1.975 | **+1.191 ± 2.579** |

Linear pair augmentation has zero practical headroom: gains by seed were
`-0.635`, `-1.024`, `+1.200`, `+1.328`, and `-0.953` pp. The nonlinear
average is positive but unstable, with gains `-0.851`, `+2.930`, `-2.268`,
`+3.497`, and `+2.647` pp. Therefore E0 supports an exploratory E1, but not a
five-seed claim that pair outputs are useful.

The strongest exploratory nonlinear signal is audio+text (`r23`), with
conditional gains `+1.701 ± 1.469` pp across five seeds. This signal is not
uniform enough to justify architecture changes or a broad hyperparameter
sweep.

## E1: frozen-backbone residual correction

E1 trains only a lower classifier and three pair correction heads. The
original ConFu encoders and pair outputs are frozen:

```text
ell_L = lower([r1; r2; r3])
ell_F = ell_L + g12 Δ12(r12) + g13 Δ13(r13) + g23 Δ23(r23)
```

Each scalar gate is initialized to `0.1`. The lower classifier is selected on
the training/validation split, the correction checkpoint is selected by
validation loss, and only then is the lower classifier refit on train plus
validation for the final test measurement. E1 uses 40 maximum epochs and
patience 5. The correction module has only `1542` pair-head parameters and
three gates; the lower head has `1538` parameters.

### Seed-1 exploratory results

| Stage | Utility loss | Preserve loss | Lower (%) | Final (%) | Δ final−lower (pp) | Net correction |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0 | 0 | 65.217 | 63.989 | **-1.229** | -13 |
| E2 | 0.1 | 0 | 65.217 | 64.367 | **-0.851** | -9 |
| E3 | 0.1 | 0.1 | 65.217 | 64.461 | **-0.756** | -8 |

Against the original ConFu seed-1 baseline (`65.469%`), the final deltas
are `-1.480`, `-1.102`, and `-1.008` pp for E1, E2, and E3. Against the
lower-only classifier, all three residual stages regress. E3 is the least
bad exploratory variant, but it is still below both its lower branch and the
baseline.

The gate values remain close to their conservative initialization:

| Stage | `g12` | `g13` | `g23` |
|---|---:|---:|---:|
| E1 | 0.0968 | 0.0993 | 0.0966 |
| E2 | 0.0990 | 0.0997 | 0.0990 |
| E3 | 0.0990 | 0.0997 | 0.0990 |

This is not evidence that a gate sweep is needed. It means the correction
heads did not learn a reliable validation-selected direction. Pair
interventions were correspondingly weak and inconsistent; for E3, zeroing
`r12` changed accuracy by `+0.284` pp, while zeroing `r13` and `r23` changed
it by `-0.662` and `-0.095` pp. Several shuffle interventions improved
accuracy, which identifies noise/anti-utility rather than useful composition.

## Gate decision

| Gate | Status | Evidence |
|---|---|---|
| E0 linear conditional headroom | Fail | `-0.017 ± 1.179` pp for all pair outputs. |
| E0 nonlinear exploratory headroom | Weak | `+1.191 ± 2.579` pp, mixed signs across seeds. |
| E1 utility | Fail | Final is `-1.229` pp below its lower branch; net correction `-13`. |
| E2 conditional utility loss | Fail | Final remains `-0.851` pp below lower; net correction `-9`. |
| E3 preserve loss | Fail | Final remains `-0.756` pp below lower; net correction `-8`. |
| Representation health | Pass | Backbone is frozen; no collapse or rank change is introduced. |
| Efficiency | Pass | Only a 3-classifier-head residual is added; inference backbone is unchanged. |
| Baseline superiority | Fail | No v2 stage beats the `64.462 ± 0.713%` five-seed ConFu baseline. |

## Bottleneck diagnosis

The bottleneck is now localized more precisely than “the interaction is
redundant”:

1. **The linear downstream task has no pair headroom.** Adding all three pair
   outputs to `[r1;r2;r3]` does not improve the matched linear probe. This
   rules out the simplest explanation that the baseline merely forgot to
   expose `r12/r13/r23` to the classifier.

2. **Any remaining headroom is nonlinear and unstable.** The nonlinear probe
   sees a positive mean, especially for `r23`, but the sign changes across
   seeds. A correction mechanism trained on one fixed seed cannot turn this
   into a confirmatory claim.

3. **Static residual correction creates more regressions than corrections.**
   E1 corrected 10 test examples and regressed 23; E2 corrected 14 and
   regressed 23; E3 corrected 14 and regressed 22. The pair logits therefore
   contain predictive variation that is not calibrated as a safe residual for
   the lower classifier.

4. **The utility loss is applied too late to create a new representation.**
   E2/E3 can only reweight frozen `r12/r13/r23`; they cannot remove the
   lower-order shortcut or recover conditional information absent from those
   outputs. Preserve loss reduces damage slightly but does not create useful
   pair evidence.

5. **The task is text-dominant and has limited headroom.** Text alone is
   about `62.18%`, while the complete original ConFu probe reaches `64.46%`.
   The remaining margin is small, so a noisy pair correction is more likely
   to regress correct lower-order decisions than to add stable new decisions.

## Final decision

ConFu++ v2 is implemented and audited, but it does **not** beat the original
ConFu baseline on UR-FUNNY. E0/E1/E2/E3 provide a complete negative result:
the failure is conditional-utility/calibration, not collapse, capacity, or
inference cost.

Per the v2 specification, no five-seed confirmatory E1 run is justified after
the seed-1 utility gate fails. Do not add cross-attention, `r123`, rank
sweeps, dynamic routing, or another loss sweep. The next meaningful upgrade
requires either a benchmark split with stable conditional headroom or a
validated training target that makes pair evidence useful before the residual
classifier is attached.

## Artifacts and commands

- Implementation: `src/experiments/multibench/urfunny_v2.py`
- E0 five-seed results: `results/urfunny_v2/e0/e0_seed_{1..5}.json`
- E1 result: `results/urfunny_v2/e1/e1_seed_1.json`
- E2 result: `results/urfunny_v2/e2/e2_seed_1.json`
- E3 result: `results/urfunny_v2/e3/e3_seed_1.json`
- E1/E2/E3 checkpoints: matching `.pt` files in those directories
- Frozen features: `results/urfunny_v2/features/features_seed_1.npz`

```bash
PYTHONPATH=. .venv/bin/python -m py_compile \
  src/experiments/multibench/urfunny_v2.py
PYTHONPATH=. .venv/bin/python -m unittest tests.test_synergyformer -v

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
  -m src.experiments.multibench.urfunny_v2 \
  --mode e0 --seed 1 --output-dir results/urfunny_v2/e0

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
  -m src.experiments.multibench.urfunny_v2 \
  --mode e1 --seed 1 --output-dir results/urfunny_v2/e1
```
