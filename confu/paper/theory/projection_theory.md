# Projection interpretation of joint predictive advantage

## Definitions

Let `X=(X_i,X_j)` and let `Y in R^d` have finite second moment. Define the conditional mean

```text
m(X) = E[Y | X].
```

Work in the vector-valued Hilbert space

```text
L = L2(P_X; R^d),
<f,g> = E[f(X)^T g(X)],
||f||^2 = E||f(X)||_2^2.
```

For a predictor `f`, use squared population risk

```text
R(f) = E||Y-f(X)||_2^2.
```

The additive space is

```text
H_A = closure({ f_i(X_i)+f_j(X_j) }),
```

where closure is taken in `L`. Let `H_J` be a closed joint space satisfying `H_A subseteq H_J`. Define

```text
f_A* = argmin_{f in H_A} R(f),
f_J* = argmin_{f in H_J} R(f).
```

The closures and nestedness are assumptions of the proposition; they are not asserted automatically for two finite neural networks.

## Proposition

Under the assumptions above,

```text
f_A* = Pi_A m,
f_J* = Pi_J m,
R(f_A*) - R(f_J*) = ||Pi_J m - Pi_A m||^2.
```

Let

```text
V_Y = E||Y-EY||_2^2,
R2(f) = 1 - R(f)/V_Y.
```

When `V_Y>0`, the population joint advantage is therefore

```text
J* = R2(f_J*) - R2(f_A*)
   = ||Pi_J m - Pi_A m||^2 / V_Y.
```

## Proof

For any square-integrable `f(X)`, write `Y-f=(Y-m)+(m-f)`. The cross term is zero because `E[Y-m | X]=0`, so

```text
R(f) = E||Y-m||^2 + ||m-f||^2.
```

Thus minimizing risk over a closed subspace is equivalent to projecting `m` onto that subspace. Let `a=Pi_A m` and `j=Pi_J m`. Because `H_A subseteq H_J`, `j-a` belongs to `H_J`; because `j` is the projection, `m-j` is orthogonal to `H_J`. The Pythagorean identity gives

```text
||m-a||^2 = ||m-j||^2 + ||j-a||^2.
```

Substitution into the risk decomposition proves the risk identity. The `R^2` identity follows by dividing by the common `V_Y`.

## Interaction residual

The population residual is

```text
d*(X) = f_J*(X) - f_A*(X).
```

Under the nested projection assumptions, `d*` lies in `H_J` and is orthogonal to the joint projection residual `m-f_J*`. It is not automatically an exact ANOVA interaction, PID synergy, causal effect, or a source of new Shannon information.

The empirical JAD target

```text
d_hat = q_J - q_A
```

is a finite-data estimate of the difference between two trained predictors. It should not be identified with `d*` without additional consistency assumptions.

## Empirical convention

All v6.0 vector `R^2` values use

```text
R2_hat = 1 - sum_n ||y_n-yhat_n||_2^2 /
             sum_n ||y_n-ybar||_2^2.
```

This is the variance-weighted multi-output convention used by the repository's `r2_score` calls. The denominator is one global sum of squared deviations, not an unreported average of per-dimension scores.

The empirical statistic

```text
J_hat = R2_hat(q_J,Y) - R2_hat(q_A,Y)
```

mixes population structure with finite-sample variation, optimization, regularization, model misspecification, and checkpoint selection. It is consequently called an operational or hypothesis-class-relative interaction score.

## Non-claims

This proposition does not prove PID synergy, causality, information creation, universal interaction sufficiency, or downstream task benefit. For deterministic fusion `r_ij=F(r_i,r_j)`, `H(r_ij|r_i,r_j)=0` and `I(Y;r_ij|r_i,r_j)=0`; the framework concerns predictive organization and accessibility, not creation of Shannon information.
