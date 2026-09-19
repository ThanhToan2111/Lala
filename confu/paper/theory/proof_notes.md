# Proof notes

For any square-integrable predictor `f(X)`, conditional-expectation orthogonality gives

```text
R(f) = E||Y-m||^2 + ||m-f||_L2^2.
```

The first term is independent of `f`. If `H` is a closed linear subspace, the risk minimizer over `H` is the orthogonal projection `Pi_H m`.

With `H_A subseteq H_J`, let `a=Pi_A m` and `j=Pi_J m`. The projection residual `m-j` is orthogonal to `H_J`, while `j-a` belongs to `H_J`. Hence

```text
||m-a||^2 = ||m-j||^2 + ||j-a||^2.
```

Subtracting the two population risks yields

```text
R(a)-R(j) = ||j-a||^2.
```

Dividing by the common target variance `V_Y` gives the vector `R^2` gap. The proof does not identify `j-a` with PID synergy, causality, or newly created information.
