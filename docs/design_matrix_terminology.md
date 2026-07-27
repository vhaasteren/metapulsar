# Design matrix vs residual Jacobian

**Status:** locked vocabulary for MetaPulsar, JUG, and nltiming  
**Canonical contract:** `ref-packages/jug/feature_designmatrix_naming_conventions.md`

This stack uses three linear objects. Classical PTA packages expose only the
first under the name “design matrix.” Nonlinear timing requires the second as a
separate public product.

| Symbol | Public name | Meaning |
|---|---|---|
| \(M\) | **`design_matrix`** | Raw **fitter basis** in the PINT/tempo2 convention. |
| \(J\) | **`residual_jacobian`** | Exact local Jacobian of `residual_delta`. |
| \(W\) | **`waveform_jacobian`** | Delay / waveform tangent. Defined by \(W = -J\). |

\[
M := \frac{\partial d_\mathrm{fit}}{\partial\theta_\mathrm{fit}},
\qquad
J := \left.\frac{\partial\,\Delta r}{\partial\delta\theta}\right|_{\delta\theta=0},
\qquad
W := -J.
\]

Fitter sign for \(M\):

\[
r(\theta+\delta\theta) \approx r(\theta) - M\,\delta\theta.
\]

## Rules

1. **`design_matrix` means only raw \(M\).** Uncentered, unweighted, public fit
   units. MetaPulsar’s combined matrix (`MetaPulsar.Mmat` / Enterprise
   `designmatrix`) is this object.
2. **`residual_jacobian` is never a design matrix.** It is
   `jacfwd(residual_delta)(0)` (or an equivalent cached form) of the engine’s
   residual function.
3. **Do not say bare “Jacobian” when you mean \(J\).** \(M\) is also a Jacobian
   (of the fitter prediction). Prefer **residual Jacobian**.
4. **\(W=-J\) is definitional**, once `delay = -residual_delta` (Discovery /
   Model D waveform convention).
5. **\(J = -C(M)\) is not a definition.** It may hold after a known residual
   transform \(C\) (e.g. mean removal). TZR anchoring or pulse connection can
   break it. Never reconstruct \(M\) as `-jac(residual_delta)` in a public API.

## Why split the names?

tempo2, PINT, Enterprise, and Vela publish a **design matrix** \(M\) for
least-squares fitting. In that setting people often treat \(M\) and
\(-\partial r/\partial\theta\) as the same column up to a global minus.

Nonlinear timing (JUG graph, nltiming engines, Model D) evaluates an exact
`residual_delta`. Its Jacobian \(J\) can differ from \(-M\) after projection or
TZR reference terms. Keeping one public noun for both was the source of sign
and unit bugs. This vocabulary stops that.

## Where to look in code

| Want | Ask for |
|---|---|
| Fitter / GLS / Enterprise columns | `design_matrix` / `Mmat` |
| Tangent of nonlinear residual evaluation | `residual_jacobian` |
| Discovery / Model D delay linearization | `waveform_jacobian` (\(=-J\)) |
