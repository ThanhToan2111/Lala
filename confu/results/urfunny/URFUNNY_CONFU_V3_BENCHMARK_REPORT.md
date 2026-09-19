# UR-FUNNY ConFu++ v3 benchmark report

**Superseded note (2026-09-18):** A runner audit found that the original E4
artifact aligned `h12` to `e1` and `h23` to `e3`, instead of the specified
mapping `h12→e3`, `h13→e2`, `h23→e1`. Those E4 numbers are retained for
engineering history but are not valid final evidence. Use the corrected
v3.1 E4.2 result in `URFUNNY_CONFU_V31_CAPACITY_AUDIT.md`.

Run date: 2026-09-18. Dataset: official MultiBench UR-FUNNY packed data.
The binary humor task has train/validation/test sizes `8074/1034/1058`; the
modality order is vision/audio/text.

## Scientific change from v2

v3 no longer treats high joint predictability of an interaction from its
inputs as a failure. A deterministic interaction is an explicit
reparameterization of joint structure, not a source of new Shannon
information. The primary question is whether an order-specific embedding
makes that structure easier for a downstream model to access.

The main metric is:

```text
Delta2 = Perf([r1;r2;r3;h12;h13;h23]) - Perf([r1;r2;r3])
```

Linear accessibility is the primary gate; the capacity-matched nonlinear
probe is secondary. No formal synergy or PID claim is made.

## Implementation

The new runner is `src/experiments/multibench/urfunny_v3.py` and contains:

1. a frozen-representation joint-vs-additive headroom audit;
2. independent first-order linear predictors `q_i->j`;
3. additive residual targets with predictors frozen before interaction training;
4. low-rank multiplicative pair branches `h12`, `h13`, and `h23`;
5. symmetric InfoNCE alignment between each `hij` and the corresponding
   first-order residual target;
6. matched accessibility, health, single-modality R², and shuffle diagnostics.

The original ConFu encoders remain frozen. There is no task loss, adversary,
residual classifier, dynamic gate, cross-attention, rank sweep, or `h123`.

## Baseline context

The strongest valid matched baseline remains original ConFu:

| Model | Test accuracy |
|---|---:|
| Original ConFu, five seeds | **64.462 ± 0.713%** |
| ConFu++ v2 E3, exploratory seed 1 | 64.461% |

The v3 E4 result is a representation-accessibility experiment, not a new
end-to-end accuracy claim. Its frozen lower probe on seed 1 is `65.204%`,
consistent with the original ConFu feature baseline.

## Headroom audit before E4

For each target modality, independent predictors from the other two modalities
were summed additively. A diagnostic joint MLP received the concatenated two
modalities. The metric is representation-target R², not task accuracy and not
formal synergy:

| Target | Additive R² | Joint MLP R² | Joint advantage |
|---|---:|---:|---:|
| `r1` from `r2,r3` | 0.119 | 0.636 | **+0.517** |
| `r2` from `r1,r3` | 0.676 | 0.911 | **+0.235** |
| `r3` from `r1,r2` | 0.658 | 0.886 | **+0.228** |

This seed-1 audit passes the v3 headroom prerequisite: frozen UR-FUNNY
representations contain non-additive joint structure that the additive
predictor does not capture. It does not prove that the structure is label
useful, so it only justifies the exploratory E4 run.

## Residual-target health

The additive first-order residuals did not collapse:

| Residual target | Variance | Effective rank | Mean norm | R² from additive predictor |
|---|---:|---:|---:|---:|
| `e1` | 0.00226 | 11.65 | 0.693 | 0.119 |
| `e2` | 0.00115 | 27.90 | 0.516 | 0.676 |
| `e3` | 0.00120 | 42.57 | 0.529 | 0.658 |

The residuals are non-trivial. Their negative cosine with the original target
(`-0.394`, `-0.309`, `-0.240`) is expected from subtracting a predictable
component and is not used as a utility gate.

## E4 result: Order-Decomposed Residual Alignment

E4 used rank `64`, output dimension `256`, InfoNCE temperature `0.07`, AdamW
`1e-3`, weight decay `1e-4`, maximum 40 epochs, patience 5, and seed 1. The
best validation alignment checkpoint was epoch 39.

### Accessibility

| Representation | Validation linear | Test linear | Test nonlinear |
|---|---:|---:|---:|
| `Z_<=1 = [r1;r2;r3]` | 59.475% | 65.204% | 61.531% |
| `Z_<=1 + h12` | 56.379% | 58.889% | 60.019% |
| `Z_<=1 + h13` | 55.994% | 61.901% | 60.870% |
| `Z_<=1 + h23` | 57.057% | 64.084% | 65.123% |
| `Z_<=2` with all pairs | 57.643% | 62.542% | 63.043% |

The resulting gains are:

| Pair/set | Validation linear gain | Test linear gain | Test nonlinear gain |
|---|---:|---:|---:|
| `h12` | -3.096 pp | -6.315 pp | -1.512 pp |
| `h13` | -3.480 pp | -3.302 pp | -0.662 pp |
| `h23` | -2.417 pp | -1.119 pp | +3.592 pp |
| All pairs `Delta2` | **-1.832 pp** | **-2.662 pp** | **+1.512 pp** |

The primary E4 gate fails because validation `Delta2` is negative. The
nonlinear test gain is exploratory only and cannot override the linear gate.

### Interaction health and dependence

The learned interaction branches are active and non-collapsed:

| Pair | Variance | Effective rank | Single-modality R² | Shuffle drops (pp) |
|---|---:|---:|---|---:|
| `h12` | 0.965 | 29.80 | from `r1`: 0.724; from `r2`: 0.165 | left +0.699; right +4.990 |
| `h13` | 0.930 | 25.93 | from `r1`: 0.708; from `r3`: 0.470 | left +5.330; right +6.944 |
| `h23` | 0.980 | 42.72 | from `r2`: 0.152; from `r3`: 0.782 | left +5.784; right +10.841 |

All six shuffle drops are positive, so each pair branch is used by the
downstream linear probe. However, the single-modality R² values show strong
imbalance: `h12` and `h13` lean toward vision, while `h23` leans toward text.
Dependence therefore passes as an activity diagnostic but not as evidence of
successful order accessibility.

## Bottleneck diagnosis

v3 separates two facts that were previously conflated:

1. **There is joint structure to expose.** The headroom audit gives a sizeable
   joint-vs-additive advantage for all three target modalities.
2. **The current residual-alignment objective does not expose it to the task
   probe.** E4 decreases linear accessibility on validation and test.

The failure is therefore not simply “UR-FUNNY has no higher-order structure.”
The more precise bottlenecks are:

- **Residual target mismatch:** the residual targets contain representation
  content that is non-additive, but not necessarily label-relevant. InfoNCE
  can align sample identity/content without making humor classification
  easier.
- **Objective/probe mismatch:** E4 optimizes pair-to-residual alignment, not
  the accessibility of label-relevant joint structure. The aligned `hij`
  branches can be healthy while adding them makes a linear classifier worse.
- **Order imbalance:** each pair branch remains more predictable from one
  modality than the other. E4 reduces neither the vision shortcut in image
  pairs nor the text shortcut in audio-text.
- **Representation scale/calibration:** LayerNorm makes each `hij` branch
  high-variance and full amplitude. Concatenating these branches changes the
  downstream geometry even though the probe capacity is controlled.
- **No end-to-end adaptation:** the frozen ConFu features constrain what the
  residual target and interaction branch can express. E4 tests the mechanism
  cleanly, but cannot repair the original representation formation.

## Gate decision

| Gate | Status | Evidence |
|---|---|---|
| Additive predictors independent | Pass | Each `q_i->j` sees one modality only. |
| Predictor leakage | Pass | No pair feature enters residual predictors. |
| Residual target health | Pass | Non-zero variance and effective rank for all targets. |
| Interaction health | Pass | Effective ranks 25.9–42.7; no collapse. |
| Both modalities affect `hij` | Pass, weakly | All six shuffle drops are positive, but R² is imbalanced. |
| Validation linear `Delta2 > 0` | **Fail** | `-1.832 pp`. |
| Test linear `Delta2 > 0` | Fail | `-2.662 pp`. |
| Nonlinear exploratory `Delta2` | Weak pass | `+1.512 pp`, but not stable/primary. |
| Five-seed confirmation | Not run | E4 primary gate failed. |
| Third-order `h123` | Not run | v3 explicitly requires positive `Delta2` first. |

## Final decision

E4 is a useful negative mechanism result. It shows that a clean
order-decomposed residual target can be non-trivial and that learned pair
branches can depend on both modalities, yet this is insufficient to improve
linear order accessibility. The representation is active; its task utility is
not established.

Do not run five seeds, E5 task loss, joint encoder fine-tuning, or `h123` from
this E4 configuration. The next minimal diagnostic, if v3 is continued, is a
predictor-capacity ablation (`P0` linear versus one small two-layer MLP) while
keeping the interaction architecture fixed. If `Delta2` remains negative,
move the mechanism test to the mandatory controlled synthetic order benchmark
instead of adding model complexity to UR-FUNNY.

## Artifacts and commands

- Implementation: `src/experiments/multibench/urfunny_v3.py`
- Headroom audit: `results/urfunny_v3/headroom_audit_seed_1.json`
- E4 metrics: `results/urfunny_v3/e4_seed_1.json`
- E4 checkpoint: `results/urfunny_v3/e4_seed_1.pt`
- Frozen input features: `results/urfunny_v2/features/features_seed_1.npz`

```bash
PYTHONPATH=. .venv/bin/python -m py_compile \
  src/experiments/multibench/urfunny_v3.py

PYTHONPATH=. .venv/bin/python -m src.experiments.multibench.urfunny_v3 \
  --mode audit --seed 1 \
  --output results/urfunny_v3/headroom_audit_seed_1.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
  -m src.experiments.multibench.urfunny_v3 \
  --mode e4 --seed 1 --output results/urfunny_v3/e4_seed_1.json
```
