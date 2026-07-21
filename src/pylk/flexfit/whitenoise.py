"""Per-backend white-noise (EFAC/EQUAD) estimation for flexfit.

White-noise parameters are (approximately) correlated with the rest of the
model only through the per-backend whitened residual power, so a Gibbs/ECM
alternation converges quickly:

1. **E/EB step** — the flexible-``Phi`` fit (coefficient posterior + group
   variances) at the current white noise ``N``;
2. **M step** — per backend ``b``, maximize the expected complete-data
   log-likelihood over ``(efac_b, equad_b)``:

   .. math::

      (f, q) \\leftarrow \\arg\\max \\sum_{i \\in b}
          -\\tfrac12 \\ln(f^2 \\sigma_i^2 + q^2)
          -\\tfrac12 \\, e_i / (f^2 \\sigma_i^2 + q^2),

   where ``e_i = E[(y - T c)_i^2] = (y - T m)_i^2 + [T \\Sigma T^T]_{ii}`` is
   the posterior-expected squared whitened residual. The correction term uses
   the coefficient covariance the conditional solve already produces, so the
   update never mistakes absorbed signal power for white noise.

The equad fitted here is the **TempoNest convention** (``tnequad``):
``N = efac^2 sigma^2 + equad^2``, i.e. added in quadrature *after* efac —
matching Discovery's ``makenoise_measurement(..., tnequad=True)``. Use
:meth:`WhiteNoiseResult.noisedict` to emit either convention.

Like everything in flexfit this is *quick-look empirical Bayes* (a joint MPE,
not a posterior); it is designed to seed production noise dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .basis import BasisBlock, assemble
from .flexible_phi import FlexiblePhiResult, solve_flexible_phi
from .noise import DiagonalNoise
from .timing import TimingModel


@dataclass(frozen=True)
class WhiteNoiseResult:
    """Immutable per-backend white-noise MPE with the final flexible-Phi fit."""

    efac: Mapping[str, float]
    equad: Mapping[str, float]  # tnequad convention, seconds
    n_toas: Mapping[str, int]
    iterations: int
    converged: bool
    history: tuple[Mapping[str, tuple[float, float]], ...]
    solve: FlexiblePhiResult | None  # None for a pure-WN fit (no basis blocks)
    variance: np.ndarray  # final per-TOA N diagonal (s^2)

    def __post_init__(self) -> None:
        for name in ("efac", "equad", "n_toas"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        var = np.array(self.variance, dtype=float, copy=True)
        var.setflags(write=False)
        object.__setattr__(self, "variance", var)

    def noisedict(
        self, psr_name: str, *, convention: str = "tnequad", equad_floor: float = 1e-10
    ) -> dict[str, float]:
        """Discovery/Enterprise-style noise dictionary.

        ``tnequad``: ``{psr}_{b}_efac`` and ``{psr}_{b}_log10_tnequad`` with
        ``N = efac^2 sigma^2 + tnequad^2`` (the convention fitted here);
        ``t2equad``: ``N = efac^2 (sigma^2 + t2equad^2)``, i.e.
        ``t2equad = tnequad / efac``. ``equad_floor`` keeps ``log10`` finite
        for backends whose equad converged to zero.
        """
        if convention not in ("tnequad", "t2equad"):
            raise ValueError("convention must be 'tnequad' or 't2equad'")
        out: dict[str, float] = {}
        for backend, f in self.efac.items():
            q = self.equad[backend]
            if convention == "t2equad":
                q = q / f
            out[f"{psr_name}_{backend}_efac"] = float(f)
            key = f"{psr_name}_{backend}_log10_{convention}"
            out[key] = float(np.log10(max(q, equad_floor)))
        return out


def expected_squared_residuals(
    y: np.ndarray, matrix: np.ndarray, solve: FlexiblePhiResult
) -> np.ndarray:
    """Posterior-expected squared whitened residuals ``e_i``.

    ``e_i = (y - T m)_i^2 + [T Sigma T^T]_{ii}`` — the EM sufficient statistic
    for the white-noise M step. The covariance term charges the white noise for
    coefficient uncertainty instead of letting absorbed signal power leak out.
    """
    y = np.asarray(y, dtype=float)
    r = y - matrix @ solve.coefficient_mean
    # diag(T Sigma T^T) via the (already dense, k x k) coefficient covariance.
    tsig = matrix @ solve.coefficient_covariance
    corr = np.einsum("ij,ij->i", tsig, matrix)
    return r**2 + np.maximum(corr, 0.0)


def _backend_m_step(
    e: np.ndarray,
    sigma2: np.ndarray,
    *,
    fit_equad: bool,
    efac_bounds: tuple[float, float],
    equad_max: float,
    x0: tuple[float, float],
) -> tuple[float, float]:
    """Maximize ``sum_i [-ln N_i - e_i/N_i]/2`` over ``(efac, equad)``.

    With ``fit_equad=False`` the EM update is closed-form:
    ``efac^2 = mean(e_i / sigma_i^2)``.
    """
    if not fit_equad:
        f2 = float(np.mean(e / sigma2))
        return float(np.clip(np.sqrt(f2), *efac_bounds)), 0.0

    def negloglike(x):
        f, q = x
        n = f * f * sigma2 + q * q
        val = 0.5 * float(np.sum(np.log(n) + e / n))
        w = (n - e) / (n * n)
        grad = np.array([f * float(np.sum(sigma2 * w)), q * float(np.sum(w))])
        return val, grad

    res = minimize(
        negloglike,
        x0=np.asarray(x0, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=[efac_bounds, (0.0, equad_max)],
    )
    return float(res.x[0]), float(res.x[1])


def fit_white_noise(
    *,
    toaerrs: np.ndarray,
    backend_flags: np.ndarray,
    blocks: Sequence[BasisBlock] = (),
    timing: TimingModel | None = None,
    residuals: np.ndarray | None = None,
    fit_equad: bool = True,
    efac_bounds: tuple[float, float] = (0.1, 10.0),
    equad_max: float = 1.0e-4,
    max_iterations: int = 10,
    tolerance: float = 1.0e-3,
    n_sweeps: int = 3,
    sweep_tolerance: float | None = None,
) -> WhiteNoiseResult:
    """Gibbs/ECM white-noise fit: alternate the flexible-``Phi`` EB solve with a
    per-backend ``(efac, equad)`` M step until the noise parameters settle.

    Parameters
    ----------
    toaerrs, backend_flags
        Per-TOA uncertainties (seconds) and backend labels. Each distinct label
        gets its own ``(efac, equad)``; empty labels raise.
    blocks
        GP basis blocks (red/DM/chromatic/ECORR...) whose group variances the EB
        sweeps learn *jointly* with the white noise.
    timing
        Optional flexfit :class:`~pylk.flexfit.timing.TimingModel`; its basis blocks
        are placed first and its anchor supplies the residuals (a single linear
        pass — WN estimation does not relinearize).
    residuals
        Residual vector when there is no timing model.
    fit_equad
        Fit per-backend equad (tnequad convention) as well as efac.
    tolerance
        Stop when every backend's ``(efac, equad)`` fractional change (equad
        measured against the backend's median toaerr) drops below this.
    """
    toaerrs = np.asarray(toaerrs, dtype=float)
    labels = np.asarray(backend_flags)
    if toaerrs.ndim != 1 or labels.shape != toaerrs.shape:
        raise ValueError("toaerrs and backend_flags must be aligned 1-D arrays")
    backends = [str(b) for b in np.unique(labels)]
    if any(b == "" for b in backends):
        raise ValueError("backend_flags contains empty labels")
    masks = {b: np.asarray(labels == b) for b in backends}
    sigma2 = toaerrs**2

    if timing is not None:
        timing_blocks = tuple(timing.blocks())
        y = np.asarray(timing.residuals(), dtype=float)
    elif residuals is not None:
        timing_blocks = ()
        y = np.asarray(residuals, dtype=float)
    else:
        raise ValueError("provide either a timing model or a residuals vector")
    all_blocks = (*timing_blocks, *blocks)
    model = assemble(all_blocks) if all_blocks else None
    if model is not None and model.n_obs != toaerrs.shape[0]:
        raise ValueError("basis n_obs does not match toaerrs")
    if y.shape != toaerrs.shape:
        raise ValueError("residuals must be aligned with toaerrs")

    efac = {b: 1.0 for b in backends}
    equad = {b: 0.0 for b in backends}
    scale = {b: float(np.median(toaerrs[masks[b]])) for b in backends}

    history: list[dict[str, tuple[float, float]]] = []
    solve: FlexiblePhiResult | None = None
    phi_prev: np.ndarray | None = None
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        variance = np.empty_like(sigma2)
        for b in backends:
            m = masks[b]
            variance[m] = efac[b] ** 2 * sigma2[m] + equad[b] ** 2
        if model is not None:
            solve = solve_flexible_phi(
                y,
                model,
                DiagonalNoise(variance),
                n_sweeps=n_sweeps,
                tolerance=sweep_tolerance,
                initial_phi=phi_prev,
            )
            phi_prev = np.asarray(solve.phi_diagonal, dtype=float)
            e = expected_squared_residuals(y, model.matrix, solve)
        else:
            e = y**2

        delta = 0.0
        for b in backends:
            m = masks[b]
            f_new, q_new = _backend_m_step(
                e[m],
                sigma2[m],
                fit_equad=fit_equad,
                efac_bounds=efac_bounds,
                equad_max=equad_max,
                x0=(efac[b], max(equad[b], 0.1 * scale[b])),
            )
            delta = max(
                delta,
                abs(f_new - efac[b]) / max(efac[b], 1e-12),
                abs(q_new - equad[b]) / scale[b],
            )
            efac[b], equad[b] = f_new, q_new
        history.append({b: (efac[b], equad[b]) for b in backends})
        if delta < tolerance:
            converged = True
            break

    # Final EB solve at the converged white noise so the reported fit matches it.
    variance = np.empty_like(sigma2)
    for b in backends:
        m = masks[b]
        variance[m] = efac[b] ** 2 * sigma2[m] + equad[b] ** 2
    solve = (
        solve_flexible_phi(
            y,
            model,
            DiagonalNoise(variance),
            n_sweeps=n_sweeps,
            tolerance=sweep_tolerance,
            initial_phi=phi_prev,
        )
        if model is not None
        else None
    )

    return WhiteNoiseResult(
        efac=efac,
        equad=equad,
        n_toas={b: int(np.sum(masks[b])) for b in backends},
        iterations=iteration,
        converged=converged,
        history=tuple(history),
        solve=solve,
        variance=variance,
    )
