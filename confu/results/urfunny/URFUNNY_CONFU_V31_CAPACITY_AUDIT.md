# UR-FUNNY ConFu++ v3.1 capacity-controlled predictor audit

Run date: 2026-09-18. Dataset: official MultiBench UR-FUNNY packed data.
The official split is train/validation/test `8074/1034/1058`, with frozen
Original ConFu representations `r1/r2/r3` of dimension 256.

## Research question

The previous v3 audit compared a linear-additive predictor with a joint MLP.
That confounded two changes at once: additive to joint access and linear to
nonlinear capacity. v3.1 asks the controlled question:

```text
After nonlinear unimodal effects are removed, does a joint predictor still
have measurable advantage over an equally sized additive predictor?
```

The metric is joint prediction advantage, not formal synergy:

```text
J = R2(P2 joint nonlinear) - R2(P1 nonlinear additive)
```

## Predictor definitions

| Predictor | Definition |
|---|---|
| P0 | Two independent linear maps plus bias: `Wi ri + Wk rk + b` |
| P1 | Two independent `256→128→256` LayerNorm/GELU MLPs plus bias, summed |
| P2 | Joint `512→171→256` LayerNorm/GELU MLP on `[ri;rk]` |

P1 and P2 have nearly identical capacity:

| Predictor | Parameters |
|---|---:|
| P0 linear additive | 131,840 |
| P1 nonlinear additive | 133,120 |
| P2 joint nonlinear | 132,779 |

P1 versus P2 differs by only `0.256%`. All predictors use the same
standardization, AdamW learning rate `1e-3`, weight decay `1e-4`, batch size
512, maximum 40 epochs, patience 5, official train/validation/test split,
and validation-only checkpoint selection. Seeds 1–3 were run because the
single-seed capacity conclusion was initially ambiguous.

## Capacity-controlled results

Values are mean ± sample standard deviation across seeds 1–3.

### Validation R²

| Target | P0 linear additive | P1 nonlinear additive | P2 joint nonlinear | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| `r1 ← r2,r3` | 0.5999 ± 0.0112 | 0.6089 ± 0.0094 | 0.6105 ± 0.0103 | +0.0089 | **+0.0016** |
| `r2 ← r1,r3` | 0.9043 ± 0.0029 | 0.9081 ± 0.0033 | 0.9091 ± 0.0029 | +0.0038 | **+0.0011** |
| `r3 ← r1,r2` | 0.8815 ± 0.0083 | 0.8841 ± 0.0076 | 0.8841 ± 0.0078 | +0.0026 | **−0.0000** |

### Test R²

| Target | P0 linear additive | P1 nonlinear additive | P2 joint nonlinear | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| `r1 ← r2,r3` | 0.6373 ± 0.0117 | 0.6428 ± 0.0094 | 0.6426 ± 0.0107 | +0.0054 | **−0.0002** |
| `r2 ← r1,r3` | 0.9113 ± 0.0052 | 0.9163 ± 0.0056 | 0.9164 ± 0.0042 | +0.0049 | **+0.0001** |
| `r3 ← r1,r2` | 0.8932 ± 0.0097 | 0.8958 ± 0.0088 | 0.8949 ± 0.0097 | +0.0026 | **−0.0009** |

The validation advantage is small: about `0.16` R² percentage points for
`r1`, `0.11` for `r2`, and effectively zero for `r3`. Test advantages are
near zero and include negative means. This is a weak exploratory signal, so
E4.2 is allowed only as a single-seed diagnostic, not as a confirmatory claim.

## Directed unimodal nonlinearity audit

The table below reports MLP minus linear R², mean ± sample standard deviation
over seeds 1–3. These are independent one-modality predictors; no predictor
receives the other modality.

| Source → target | Validation nonlinear gain | Test nonlinear gain |
|---|---:|---:|
| Vision → Audio | +0.0029 ± 0.0022 | +0.0012 ± 0.0028 |
| Vision → Text | +0.0049 ± 0.0013 | +0.0008 ± 0.0024 |
| Audio → Vision | +0.0086 ± 0.0011 | +0.0030 ± 0.0018 |
| Audio → Text | +0.0015 ± 0.0005 | +0.0012 ± 0.0006 |
| Text → Vision | +0.0064 ± 0.0004 | +0.0036 ± 0.0007 |
| Text → Audio | +0.0030 ± 0.0003 | +0.0043 ± 0.0002 |

Every directed relationship benefits at least slightly from nonlinear
capacity. This directly supports the v3.1 hypothesis that the old linear
residual contained nonlinear unimodal residue.

## Interpretation of the earlier v3 result

The previous v3 audit reported large apparent joint advantages because it
compared P2 against P0. v3.1 shows:

```text
P1 − P0 > 0 for all targets
P2 − P1 ≈ 0 for all targets
```

Therefore the earlier `P2−P0` signal was primarily the benefit of allowing
nonlinear unimodal functions, not evidence of robust non-additive multimodal
structure under capacity control.

This also explains why v3 E4 residual alignment produced healthy interaction
representations but negative linear `Delta2`: its residual targets preserved
unimodal nonlinear residue and were not clean order-specific targets.

## E4.2 exploratory diagnostic

Because P2−P1 was weakly positive on validation for `r1` and `r2`, one
seed-1 E4.2 run was performed with nonlinear additive P1 residual targets.
The interaction architecture and optimizer were unchanged. The corrected
order mapping was used:

```text
h12 -> e3, h13 -> e2, h23 -> e1
```

| Representation | Validation linear | Test linear | Test nonlinear |
|---|---:|---:|---:|
| `Z_<=1` | 59.475% | 65.204% | 61.531% |
| `Z_<=2` | 56.967% | 63.981% | 64.556% |
| `Delta2` | **−2.508 pp** | **−1.223 pp** | **+3.025 pp** |

Pair-specific test linear gains were `−4.747`, `−2.365`, and `−2.529` pp
for `h12`, `h13`, and `h23`. Nonlinear gains were positive (`+1.040`,
`+2.647`, `+1.985` pp), but the primary validation linear gate still failed.
The branches remained healthy with effective ranks `35.50`, `43.57`, and
`39.04`; all six shuffle drops were positive. Thus nonlinear P1 residuals
improve interaction dependence diagnostics but still do not produce linear
order accessibility.

The previous v3 E4 artifact is superseded: an implementation audit found its
target mapping used `h12→e1` and `h23→e3` instead of the specified mapping.
The v3.1 E4.2 run uses the corrected mapping and is the valid follow-up.

## Gate decision

| Gate | Status | Evidence |
|---|---|---|
| Independent unimodal predictors | Pass | P1 branches never see the other modality. |
| P1/P2 capacity matching | Pass | Difference `0.256%`. |
| Standardization/protocol matching | Pass | Same preprocessing, optimizer, split, epochs, and validation selection. |
| P1 improves over P0 | Pass | Nonlinear gains for all three targets. |
| P2 clearly improves over P1 | Weak / exploratory | Small validation gains for `r1/r2`; test gains are near zero. |
| E4.2 nonlinear residual alignment | **Fail** | Corrected validation linear `Delta2 = -2.508 pp`. |
| Interaction health in E4.2 | Pass | No collapse; all shuffle drops positive. |
| Stable non-additive headroom | Fail | P2−P1 is not robust across targets and test splits. |

## Final decision

E4.1 shows that the prior apparent joint headroom was largely a
linear-versus-nonlinear confound. The weak residual P2−P1 signal justified one
E4.2 diagnostic, but nonlinear additive residuals still produced negative
validation linear `Delta2`.

The complete v3.1 branch therefore fails the accessibility gate. Do not run
five-seed E4.2, task loss, encoder fine-tuning, rank sweeps, cross-attention,
or `h123`. The next step is the controlled synthetic order benchmark, where
pair and triple structure are known by construction. UR-FUNNY remains useful
as a negative/diagnostic benchmark.

## Artifacts and command

- Implementation: `src/experiments/multibench/urfunny_v31.py`
- Seed results: `results/urfunny_v31/e4_1_seed_{1..3}.json`
- E4.2 exploratory result: `results/urfunny_v31/e4_2_seed_1.json`
- Predictor checkpoints: `results/urfunny_v31/checkpoints/seed_{1..3}/`
- Frozen features: `results/urfunny_v2/features/features_seed_{1..3}.npz`

```bash
PYTHONPATH=. .venv/bin/python -m py_compile \
  src/experiments/multibench/urfunny_v31.py

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
  -m src.experiments.multibench.urfunny_v31 \
  --seed 1 --output results/urfunny_v31/e4_1_seed_1.json
```
