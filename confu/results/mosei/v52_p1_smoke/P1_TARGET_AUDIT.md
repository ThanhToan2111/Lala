# MOSEI v5.2 P1 additive purification target audit

Run date: 2026-09-19. Frozen canonical raw MOSEI, pooled vision+audio → text, five seeds.

## Protocol

The frozen target is `d=q_joint-q_add`. A label-free capacity-matched additive predictor is fit on train only, selected by validation MSE, and subtracted to form `d_perp=d-d_hat_add`. No sentiment, test score, architecture sweep, or purification strength is used.

## Reconstruction audit

| Target | V-only R² | A-only R² | Additive V+A R² | Joint V+A R² | J_d |
|---|---:|---:|---:|---:|---:|
| `d` | 0.184 ± 0.000 | 0.077 ± 0.000 | -0.303 ± 0.000 | -0.076 ± 0.000 | 0.227 ± 0.000 |
| `d_perp` | 0.272 ± 0.000 | 0.066 ± 0.000 | -0.519 ± 0.000 | -0.053 ± 0.000 | 0.466 ± 0.000 |

All values are mean ± sample standard deviation over seeds 1–5. Primary probe selection is validation-only.

## Health

| Target | Variance | Mean norm | Effective rank | Near-zero dimensions |
|---|---:|---:|---:|---:|
| `d` | 0.047 ± 0.000 | 2.451 ± 0.000 | 8.343 ± 0.000 | 0.000 ± 0.000 |
| `d_perp` | 0.079 ± 0.000 | 2.665 ± 0.000 | 5.426 ± 0.000 | 0.000 ± 0.000 |

## P1 gate

- Reduced V/A/additive reconstructability: **FAIL**
- Positive joint residual advantage `J_d_perp`: **PASS**
- Nontrivial target health: **PASS**
- Target-side P1 audit: **FAIL**

This audit establishes only hypothesis-class-relative conditional selectivity; it does not establish causal interaction, PID synergy, or exact ANOVA interaction.

## Decision

If the P1 target audit passes, the next permitted step is the frozen seed-1 synthetic IPIB mechanism check comparing canonical D2 and `D2_perp`. If it fails, do not distill purified MOSEI or add another method.
