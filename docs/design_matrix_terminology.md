# Design matrix vs residual Jacobian

**Status:** Locked vocabulary for MetaPulsar, JUG, and nltiming  
**Design notes:** `ref-packages/jug/feature_phase_gauge.md`  
**Prior naming doc:** `ref-packages/jug/feature_designmatrix_naming_conventions.md` (Phases A/B; Phase C superseded)

This stack locks **two objects, two construction routes, and one operator**. The
old third noun (`waveform_jacobian` / \(W\)) is deleted: once residual products
are gauge-free, the delay tangent *is* \(M\).

## Object vs route (read this first)

**Objects** (what the array *is* — never how it was computed):

| Symbol | Public name | Meaning |
|---|---|---|
| \(M\) | **`design_matrix`** | Delay tangent in the PINT/tempo2 fitter sign: uncentered, unweighted, public fit units. |
| \(J\) | **`residual_jacobian`** | Residual tangent: \(J=-M\). Never called a design matrix. |
| — | **phase gauge** | Rank-1 affine operator on residual vectors, applied only at reporting/parity boundaries. |

**Routes** (how \(M\) or \(J\) is computed — orthogonal to which object it is):

| Route | Means |
|---|---|
| `"analytic"` | assembled from hand-derived derivative blocks |
| `"autodiff"` | differentiated from the gauge-free residual graph |

\[
M := \frac{\partial d_\mathrm{fit}}{\partial\theta_\mathrm{fit}},
\qquad
J := \left.\frac{\partial\,\Delta r}{\partial\delta\theta}\right|_{\delta\theta=0}
   = -M,
\qquad
G_c(r) := r - c\,\mathbf{1}.
\]

Fitter sign for \(M\):

\[
r(\theta+\delta\theta) \approx r(\theta) - M\,\delta\theta.
\]

With no centering inside the engine, \(J=-M\) holds with no intervening
transform. Analytic and autodiff are two routes to the **same** object \(M\);
they may differ by analytic-block approximation error only.

## Rules

1. **Every object named `design_matrix` is \(M\).** Uncentered, unweighted,
   public fit units, fitter sign. **How it was computed is not part of the
   definition.** MetaPulsar’s combined matrix (`MetaPulsar.Mmat` / Enterprise
   `designmatrix`) is this object whether filled from host analytic columns or
   from `-residual_jacobian()`.
2. **Object and route are orthogonal.** `derivative_method` selects the route,
   never the object. Do not invent parallel public nouns such as
   `analytic_design_matrix` / `autodiff_design_matrix`.
3. **`residual_jacobian` is never a design matrix.** It is
   `jacfwd(residual_delta)(0)` (or an equivalent cached form) of the engine’s
   **gauge-free** residual function.
4. **Export products are gauge-free.** Engines do not subtract a mean from
   `residual_delta` / `residual_jacobian`. The phase gauge is applied only at
   reporting boundaries (`jug.residuals.gauge`).
5. **Do not say bare “Jacobian” when you mean \(J\).** \(M\) is also a Jacobian
   (of the fitter prediction). Prefer **residual Jacobian**.
6. **`waveform_jacobian` / \(W\) is deleted.** The delay tangent is \(M\); ask
   for `design_matrix` by either route.
7. **Run knob:** nltiming / MetaPulsar use
   `derivative_method="analytic"|"autodiff"`. `"analytic"` → `pulsar.Mmat`;
   `"autodiff"` → `-engine.residual_jacobian()`. Same object, same gauge, same
   sign; the knob selects derivative *quality*. JUG’s public
   `compute_designmatrix` currently exposes only the analytic route — that is
   API scope for one helper, not a claim that autodiff \(M\) would stop being a
   design matrix.

## Why split the names?

tempo2, PINT, Enterprise, and Vela publish a **design matrix** \(M\) for
least-squares fitting. Nonlinear timing evaluates an exact `residual_delta`
whose Jacobian is \(J=-M\) once the residual is gauge-free. Keeping one public
noun for both \(M\) and a centered residual Jacobian was the source of sign and
unit bugs. This vocabulary stops that.

## Where to look in code

| Want | Ask for |
|---|---|
| Fitter / GLS / Enterprise columns | `design_matrix` / `Mmat` (analytic or autodiff route) |
| Tangent of nonlinear residual evaluation | `residual_jacobian` (\(=-M\)) |
| Delay tangent | `design_matrix` (same object) |
| Reporting / plot residuals | `jug.residuals.gauge.apply_phase_gauge` |
