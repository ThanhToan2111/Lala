# MOSEI v5.2 P1 additive purification target audit

Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → text, five seeds.

## Protocol

The frozen target is `d=q_joint-q_add`. A label-free capacity-matched additive predictor is fit on train only, selected by validation MSE, and subtracted to form `d_perp=d-d_hat_add`. No sentiment, test score, architecture sweep, or purification strength is used.

## Reconstruction audit

| Target | V-only R² | A-only R² | Additive V+A R² | Joint V+A R² | J_d |
|---|---:|---:|---:|---:|---:|
| `d` | 0.189 ± 0.066 | 0.104 ± 0.032 | 0.066 ± 0.333 | 0.278 ± 0.075 | 0.211 ± 0.286 |
| `d_perp` | 0.194 ± 0.033 | 0.035 ± 0.005 | -0.035 ± 0.108 | 0.028 ± 0.136 | 0.064 ± 0.130 |

All values are mean ± sample standard deviation over seeds 1–5. Primary probe selection is validation-only.

## Health

| Target | Variance | Mean norm | Effective rank | Near-zero dimensions |
|---|---:|---:|---:|---:|
| `d` | 0.052 ± 0.005 | 2.461 ± 0.173 | 7.448 ± 1.026 | 0.000 ± 0.000 |
| `d_perp` | 0.052 ± 0.008 | 1.948 ± 0.190 | 4.686 ± 0.430 | 0.000 ± 0.000 |

## P1 gate

- Reduced V/A/additive reconstructability: **PASS**
- Positive joint residual advantage `J_d_perp`: **FAIL**
- Nontrivial target health: **PASS**
- Target-side P1 audit: **FAIL**

This audit establishes only hypothesis-class-relative conditional selectivity; it does not establish causal interaction, PID synergy, or exact ANOVA interaction.

## Decision

If the P1 target audit passes, the next permitted step is the frozen seed-1 synthetic IPIB mechanism check comparing canonical D2 and `D2_perp`. If it fails, do not distill purified MOSEI or add another method.
