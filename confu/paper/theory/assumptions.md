# T1 assumptions

The projection statement is a population result for squared loss. It assumes:

1. `X=(X_i,X_j)` has a probability law `P_X` and `Y` is a square-integrable vector in `R^d`.
2. The conditional mean `m(X)=E[Y|X]` exists in `L2(P_X;R^d)`.
3. Predictors are identified with elements of the Hilbert space `L2(P_X;R^d)` under the inner product `E[f(X)^T g(X)]`.
4. The additive and joint hypothesis spaces used by the proposition are closed linear subspaces, or are replaced explicitly by their closures.
5. The joint space contains the additive space: `H_A subseteq H_J`.
6. The target variance denominator `V_Y=E||Y-EY||^2` is finite and non-zero.

These are population assumptions. A finite neural predictor, a finite sample, early stopping, regularization, and a capacity-matched Product Joint need not satisfy literal nested-subspace assumptions. Such results are therefore reported as operational, hypothesis-class-relative estimates rather than exact population projections.
