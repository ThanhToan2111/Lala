# CMU-MOSEI emotion identifiability screening

Run date: 2026-09-18. Phase: ConFu++ v5.0 R1. Dataset: official MultiBench CMU-MOSEI processed features plus official `All Labels` from `mosei.hdf5`. Protocol: three seeds (`1..3`), label-free modality predictors, validation-only selection. Predictor initialization was fixed before this final run so the seed controls both initialization and optimization order.

## Objective

R1 tests whether natural CMU-MOSEI contains jointly identifiable mappings before training any D0/D1/D2 interaction representation. The directions are:

```text
VA_to_T: vision + audio -> text
VT_to_A: vision + text  -> audio
AT_to_V: audio + text   -> vision
```

For each direction:

```text
q_A = additive predictor
q_J = capacity-matched joint predictor
J   = R²(q_J, target) - R²(q_A, target)
```

The activation gate is `J_val > 0.01` in every one of the three screening seeds. Test metrics are diagnostic only and never select a setting.

## Data and leakage control

The official processed split is train/validation/test = `16,265/1,869/4,643`. Pooled dimensions are vision/audio/text = `35/74/300`. Each modality is standardized using train statistics only.

Predictors receive only the two source modality representations and predict the third modality. Emotion labels are not predictor inputs or regression targets. After fitting, official emotion scores are used to stratify validation and test R². Validation stratification participates in the pre-registered setting gate; test stratification is held out for confirmation only. No valid D0, D1, or D2 result is accepted because no setting passed R1.

The six emotion-positive subsets use score `>0`: happiness, sadness, anger, surprise, disgust, and fear. The processed pickle does not contain these scores, so they were aligned from official HDF5 `All Labels` using all 74 audio features, without using sentiment to choose a candidate. Sentiment agreement is only an alignment diagnostic: `0.9785`. The residual mismatch is a dataset-version/preprocessing limitation.

## Capacity audit

| Direction | Additive parameters | Joint parameters | Difference |
|---|---:|---:|---:|
| VA→T | 13,398 | 13,443 | 0.336% |
| VT→A | 7,908 | 7,900 | 0.101% |
| AT→V | 7,206 | 7,250 | 0.611% |

The additive model has two independent MLP branches. The joint model has two source projections, an elementwise product, and an output projection. Capacity mismatch is below 1% in every direction.

## Global predictor result

Values are mean ± sample standard deviation over three seeds.

| Direction | Add train R² | Joint train R² | J train | Add val R² | Joint val R² | J val | Add test R² | Joint test R² | J test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VA→T | 0.0908 ± 0.0013 | 0.0738 ± 0.0056 | -0.0170 ± 0.0043 | 0.0566 ± 0.0002 | 0.0455 ± 0.0004 | -0.0111 ± 0.0006 | 0.0661 ± 0.0014 | 0.0601 ± 0.0008 | -0.0060 ± 0.0010 |
| VT→A | 0.1555 ± 0.0301 | 0.1333 ± 0.0055 | -0.0223 ± 0.0266 | 0.0916 ± 0.0041 | 0.0932 ± 0.0022 | +0.0016 ± 0.0062 | 0.0925 ± 0.0052 | 0.0957 ± 0.0030 | +0.0033 ± 0.0075 |
| AT→V | 0.0875 ± 0.0077 | 0.0880 ± 0.0111 | +0.0004 ± 0.0143 | 0.0380 ± 0.0008 | 0.0389 ± 0.0016 | +0.0010 ± 0.0023 | 0.0442 ± 0.0021 | 0.0406 ± 0.0014 | -0.0036 ± 0.0006 |

No direction is globally active under the `0.01` validation gate.

## Emotion-conditioned result

Values are validation `J` mean ± sample standard deviation; `positive` counts seeds with `J_val > 0.01`.

| Emotion | VA→T | VT→A | AT→V |
|---|---:|---:|---:|
| happiness | -0.0131 ± 0.0011 (0/3) | +0.0063 ± 0.0067 (1/3) | +0.0025 ± 0.0019 (0/3) |
| sadness | -0.0084 ± 0.0005 (0/3) | +0.0001 ± 0.0097 (0/3) | +0.0006 ± 0.0045 (0/3) |
| anger | -0.0047 ± 0.0024 (0/3) | +0.0011 ± 0.0119 (1/3) | +0.0013 ± 0.0139 (1/3) |
| surprise | -0.0066 ± 0.0038 (0/3) | +0.0050 ± 0.0043 (0/3) | -0.0030 ± 0.0043 (0/3) |
| disgust | -0.0138 ± 0.0015 (0/3) | -0.0042 ± 0.0061 (0/3) | +0.0011 ± 0.0105 (1/3) |
| fear | +0.0021 ± 0.0027 (0/3) | **+0.0142 ± 0.0152 (2/3)** | +0.0008 ± 0.0084 (0/3) |

The apparent best setting is `fear / VT→A`, but it fails the pre-registered consistency gate: only seeds 2 and 3 exceed `0.01`; seed 1 is `-0.0029`. Its held-out test J values are `+0.0121`, `+0.0109`, and `+0.0161`, but these test values cannot repair a failed validation gate.

## R1 decision

**Fail the identifiability gate. No R2 setting is selected.**

The correct conclusion is conditional, not that MOSEI has no multimodal structure. The data show a weak and seed-sensitive signal in fear-conditioned `vision + text -> audio`, but not a reliable three-seed joint advantage under the required capacity-controlled predictor test. Training D0/D1/D2 now would make the downstream comparison post-hoc and would violate v5.0's screen-first protocol.

A preliminary one-seed R2 smoke artifact exists under `results/mosei/r2_smoke/`, but it predates the final seeded-initialization fix and is explicitly excluded from this report. It must not be compared with the final R1 run.

The practical bottleneck is not missing model capacity. It is unstable conditional joint predictability: the interaction term is small, changes sign across seeds, and does not clear the fixed `0.01` validation margin consistently. The alignment diagnostic (`97.85%` sentiment agreement) is also a limitation to resolve in a future data audit, but it is not a license to select using test performance.

## Reproducibility artifacts

- Machine summary: `results/mosei/identifiability/mosei_identifiability.json`
- Per-seed JSON: `va_to_t_seed_{1..3}.json`, `vt_to_a_seed_{1..3}.json`, `at_to_v_seed_{1..3}.json`
- Pooled cache: `results/mosei/identifiability/mosei_pooled.npz`
- Predictor checkpoints: `results/mosei/identifiability/checkpoints/`
- Runner: `src/experiments/multibench/mosei_identifiability.py`
- Tests: `tests/test_mosei_identifiability.py`

Reproduction command:

```bash
PYTHONWARNINGS=ignore PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m src.experiments.multibench.mosei_identifiability \
  --seeds 1 2 3 --out results/mosei/identifiability
```

Do not run an R2 real-world interaction benchmark, tune a third-order model, or return to UR-FUNNY under v5.0 until a future R1 audit either fixes the data alignment issue or establishes a new setting that passes the same three-seed gate.
