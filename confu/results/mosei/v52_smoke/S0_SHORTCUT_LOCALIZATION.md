# MOSEI v5.2 S0 shortcut localization

Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → text, five R2 seeds.

## Protocol

S0 replays the frozen v5.1 predictor and rank-64 JAD settings to recover `q_add`, `q_joint`, `d=q_joint-q_add`, and `h_D2`. It does not introduce a new target, loss, architecture, task label, or hyperparameter sweep.

V-only and A-only reconstruction use Ridge with alpha selected on validation. Additive and joint V+A references use the existing capacity-matched predictor classes (`AdditivePredictor`/`JointPredictor`, under 1% parameter mismatch), also selected by validation only. Test is reporting-only.

## Required test-set table

| Representation | V-only R² | A-only R² | Additive V+A R² | Joint V+A R² | J_z |
|---|---:|---:|---:|---:|---:|
| `q_add` | 0.461 ± 0.000 | 0.283 ± 0.000 | 0.155 ± 0.000 | 0.314 ± 0.000 | 0.159 ± 0.000 |
| `q_joint` | 0.466 ± 0.000 | 0.422 ± 0.000 | 0.155 ± 0.000 | 0.424 ± 0.000 | 0.269 ± 0.000 |
| `d` | 0.184 ± 0.000 | 0.077 ± 0.000 | -0.468 ± 0.000 | -0.114 ± 0.000 | 0.354 ± 0.000 |
| `h_d2` | 0.140 ± 0.000 | 0.152 ± 0.000 | 0.200 ± 0.000 | 0.381 ± 0.000 | 0.181 ± 0.000 |

Values are mean ± sample standard deviation over seeds 1–5. `J_z` is the joint reconstruction R² minus additive reconstruction R².

## Validation gate and decision

The pre-registered S0 classification is **TARGET_LEVEL_LEAKAGE**. It is based on validation reconstruction, not test performance.

For `d`, the largest validation single-modality R² is 0.253 ± 0.000; for `h_D2`, it is 0.159 ± 0.000. See the machine JSON for per-probe uncertainty.

Interpretation: S0 localizes whether lower-order leakage is already present in the JAD target or is introduced/amplified by distillation. It does not establish causality, PID synergy, or exact functional ANOVA interaction.

## Frozen baseline context

The prior five-seed R2 task baseline was lower-order accuracy `64.397 ± 0.231%`; D2 was `64.363 ± 0.288%` with audio-only shortcut R² `0.302 ± 0.048`. S0 is a mechanism audit, not a new accuracy benchmark.

## Artifacts

- `s0_shortcut_localization.json` — per-seed reconstruction metrics, probe hyperparameters, train/validation/test R².
- `mosei_shortcut_localization.py` — reproducible S0 runner.
