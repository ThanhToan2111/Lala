# MOSEI v5.2 S0 shortcut localization

Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → text, five R2 seeds.

## Protocol

S0 replays the frozen v5.1 predictor and rank-64 JAD settings to recover `q_add`, `q_joint`, `d=q_joint-q_add`, and `h_D2`. It does not introduce a new target, loss, architecture, task label, or hyperparameter sweep.

V-only and A-only reconstruction use Ridge with alpha selected on validation. Additive and joint V+A references use the existing capacity-matched predictor classes (`AdditivePredictor`/`JointPredictor`, under 1% parameter mismatch), also selected by validation only. Test is reporting-only.

## Required test-set table

| Representation | V-only R² | A-only R² | Additive V+A R² | Joint V+A R² | J_z |
|---|---:|---:|---:|---:|---:|
| `q_add` | 0.442 ± 0.084 | 0.248 ± 0.026 | 0.628 ± 0.094 | 0.576 ± 0.064 | -0.053 ± 0.127 |
| `q_joint` | 0.517 ± 0.029 | 0.407 ± 0.018 | 0.688 ± 0.089 | 0.934 ± 0.014 | 0.247 ± 0.086 |
| `d` | 0.189 ± 0.066 | 0.104 ± 0.032 | -0.027 ± 0.316 | 0.267 ± 0.091 | 0.295 ± 0.264 |
| `h_d2` | 0.204 ± 0.028 | 0.303 ± 0.048 | 0.386 ± 0.048 | 0.654 ± 0.061 | 0.268 ± 0.020 |

Values are mean ± sample standard deviation over seeds 1–5. `J_z` is the joint reconstruction R² minus additive reconstruction R².

## Validation gate and decision

The pre-registered S0 classification is **MIXED**. It is based on validation reconstruction, not test performance: leakage is already present in `d`, while distillation amplifies the audio component.

Validation single-modality R²: `d` is V-only `0.296 ± 0.045` and A-only `0.140 ± 0.045`; `h_D2` is V-only `0.280 ± 0.007` and A-only `0.301 ± 0.045`. Thus target-level leakage is clear through vision, and the audio shortcut is amplified during distillation.

Interpretation: S0 localizes whether lower-order leakage is already present in the JAD target or is introduced/amplified by distillation. It does not establish causality, PID synergy, or exact functional ANOVA interaction.

## Frozen baseline context

The prior five-seed R2 task baseline was lower-order accuracy `64.397 ± 0.231%`; D2 was `64.363 ± 0.288%` with audio-only shortcut R² `0.302 ± 0.048`. S0 is a mechanism audit, not a new accuracy benchmark.

## Artifacts

- `s0_shortcut_localization.json` — per-seed reconstruction metrics, probe hyperparameters, train/validation/test R².
- `mosei_shortcut_localization.py` — reproducible S0 runner.
