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

   where ``e_i`` is the posterior-expected squared measurement residual. With
   kernel ECORR the statistic excludes the epoch-correlated part (§3.6 of
   ``feature_flexfit_fasttnt.md``).

The equad fitted here is the **TempoNest convention** (``tnequad``):
``N = efac^2 sigma^2 + equad^2``, i.e. added in quadrature *after* efac —
matching Discovery's ``makenoise_measurement(..., tnequad=True)``. Use
:meth:`WhiteNoiseResult.noisedict` to emit either convention.

Like everything in flexfit this is *quick-look empirical Bayes* (a joint MPE,
not a posterior); it is designed to seed production noise dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import scipy.linalg as sl
from scipy.optimize import minimize

from .basis import BasisBlock, assemble
from .flexible_phi import FlexiblePhiResult, solve_flexible_phi
from .noise import DiagonalNoise, EpochKernelNoise, ShermanMorrisonNoise
from .timing import TimingModel

if TYPE_CHECKING:
    from .fasttnt import Factorization
    from .waveform import StageSpec, WaveformAnalysis

CHUNK = 256  # columns of L per pass; ~150 MB at n = 7e4


@dataclass(frozen=True)
class KernelEcorrMoments:
    """Epoch-corrected residual statistic and epoch second moments (§3.6/§3.7)."""

    e: np.ndarray
    a_hat: np.ndarray
    a_second_moment: np.ndarray

    def __post_init__(self) -> None:
        for name in ("e", "a_hat", "a_second_moment"):
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


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
    variance: np.ndarray  # final per-TOA D diagonal (s^2)
    residuals: np.ndarray  # y vector of the final solve (seconds)
    kernel: EpochKernelNoise | None = None

    def __post_init__(self) -> None:
        for name in ("efac", "equad", "n_toas"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("variance", "residuals"):
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    def waveform_analysis(
        self,
        *,
        toas: np.ndarray,
        toa_mjd: np.ndarray,
        block_kinds: Mapping[str, str],
        block_frequencies: Mapping[str, np.ndarray] | None = None,
        freqs_mhz: np.ndarray | None = None,
        stages: Sequence[StageSpec] | None = None,
    ) -> WaveformAnalysis:
        if self.solve is None:
            raise RuntimeError("WhiteNoiseResult.solve is None (pure-WN fit)")
        from .waveform import analyze_waveforms

        return analyze_waveforms(
            self.residuals,
            self.variance,
            self.solve,
            toas=toas,
            toa_mjd=toa_mjd,
            block_kinds=block_kinds,
            block_frequencies=block_frequencies,
            freqs_mhz=freqs_mhz,
            stages=stages,
            noise=self.kernel,
        )

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


def _expand_model(model, v: np.ndarray) -> np.ndarray:
    expand = getattr(model, "expand", None)
    if expand is not None:
        return expand(v)
    matrix = np.asarray(model, dtype=float)
    return matrix @ np.asarray(v, dtype=float)


def _kernel_weights(model, indicator, d: np.ndarray) -> np.ndarray:
    """``W = E^T diag(d) T`` for a LinearModel or a bare matrix."""
    op = getattr(model, "kernel_weights", None)
    if op is not None:
        return op(indicator, d)
    matrix = np.asarray(model, dtype=float)
    return np.asarray(indicator.T @ (np.asarray(d, dtype=float)[:, None] * matrix))


def _epoch_row_dot(model, g: np.ndarray, epoch: np.ndarray) -> np.ndarray:
    """``out_i = T_i . g[epoch_i]`` for a LinearModel or a bare matrix."""
    op = getattr(model, "epoch_row_dot", None)
    if op is not None:
        return op(g, epoch)
    matrix = np.asarray(model, dtype=float)
    epoch = np.asarray(epoch, dtype=np.int64)
    out = np.zeros(matrix.shape[0], dtype=float)
    rows = np.flatnonzero(epoch >= 0)
    step = max(1, int(4_000_000 // max(matrix.shape[1], 1)))
    for i0 in range(0, rows.size, step):
        idx = rows[i0 : i0 + step]
        out[idx] = np.einsum("ij,ij->i", matrix[idx], np.asarray(g)[epoch[idx]])
    return out


def _kernel_ecorr_moments(
    r: np.ndarray,
    corr: np.ndarray,
    model,
    solve: FlexiblePhiResult,
    noise: EpochKernelNoise | ShermanMorrisonNoise,
) -> KernelEcorrMoments:
    """§3.6 residual statistic under kernel ECORR; also returns ``E[a_e^2]``."""
    if isinstance(noise, EpochKernelNoise):
        e_ind = noise.indicator
        dinv = noise._dinv
        s = np.asarray(noise.capacitance_scale, dtype=float)
        epoch = noise.epoch
        valid = noise._valid
    else:
        from .fasttnt import _indicator

        e_ind = _indicator(noise)
        dinv = 1.0 / noise.diagonal
        t = np.asarray(e_ind.multiply(e_ind).T @ dinv).ravel()
        s = 1.0 / (1.0 / noise.jitter + t)
        # Reconstruct a per-TOA epoch index from the CSR indicator.
        epoch = np.full(noise.n_obs, -1, dtype=np.int64)
        coo = e_ind.tocoo()
        epoch[coo.row] = coo.col
        valid = epoch >= 0

    # W = E^T D^-1 T, tier-wise (factored) or one sparse product (dense) — never
    # by rebuilding T column by column, which would cost O(n k^2).
    w = _kernel_weights(model, e_ind, dinv)

    u = np.asarray(e_ind.T @ (dinv * r)).ravel()
    a_hat = s * u
    sigma = np.asarray(solve.coefficient_covariance, dtype=float)
    g = w @ sigma  # (n_ep, k)
    wsw_diag = np.einsum("ij,ij->i", g, w)
    a_second = s**2 * u**2 + s + s**2 * wsw_diag

    # Cross term: -2 s_e (T G^T)_{i, e(i)}, plus the epoch variance terms.
    cross = np.zeros_like(r)
    if valid.any():
        tg = _epoch_row_dot(model, g, epoch)
        s_i = s[epoch.clip(0)]
        cross = valid * (-2.0 * s_i * tg + s_i + s_i**2 * wsw_diag[epoch.clip(0)])

    r_corr = r.copy()
    r_corr[valid] -= a_hat[epoch[valid]]
    e = r_corr**2 + corr + cross
    return KernelEcorrMoments(e=e, a_hat=a_hat, a_second_moment=a_second)


def _covariance_diagonal(model, solve: FlexiblePhiResult) -> np.ndarray:
    """``diag(T Σ T^T)`` via chunked ``expand`` of ``chol(Σ)``."""
    L = sl.cholesky(np.asarray(solve.coefficient_covariance), lower=True)
    n = getattr(model, "n_obs", None)
    if n is None:
        n = np.asarray(model, dtype=float).shape[0]
    corr = np.zeros(int(n), dtype=float)
    for j0 in range(0, L.shape[1], CHUNK):
        tl = _expand_model(model, L[:, j0 : j0 + CHUNK])
        corr += np.einsum("ij,ij->i", tl, tl)
    return corr


def expected_squared_residuals(
    y: np.ndarray,
    model,
    solve: FlexiblePhiResult,
    *,
    noise=None,
) -> np.ndarray:
    """``e_i = E[(y - T b - E a)_i^2]``, wholly in the model's own basis.

    ``model`` is a LinearModel or a bare basis matrix. Every occurrence of ``T``
    is applied through ``expand``, so a factored model uses its substituted
    basis consistently. ``noise`` is required for a kernel-ECORR operator,
    whose epoch term and cross term would otherwise be charged to EFAC.
    """
    y = np.asarray(y, dtype=float)
    r = y - _expand_model(model, solve.coefficient_mean)
    corr = _covariance_diagonal(model, solve)
    if isinstance(noise, (EpochKernelNoise, ShermanMorrisonNoise)):
        return _kernel_ecorr_moments(r, corr, model, solve, noise).e
    return r**2 + corr


def kernel_ecorr_moments(
    y: np.ndarray,
    model,
    solve: FlexiblePhiResult,
    noise: EpochKernelNoise | ShermanMorrisonNoise,
) -> KernelEcorrMoments:
    """Full §3.6/§3.7 moments ``(e, â, E[a_e^2])`` for a kernel-ECORR operator."""
    y = np.asarray(y, dtype=float)
    r = y - _expand_model(model, solve.coefficient_mean)
    corr = _covariance_diagonal(model, solve)
    return _kernel_ecorr_moments(r, corr, model, solve, noise)


def _kernel_ecorr_m_step(
    moments: KernelEcorrMoments,
    noise: EpochKernelNoise,
    *,
    ecorr_min: float,
    ecorr_max: float,
) -> np.ndarray:
    """Per-backend EM update ``λ_b ← mean_{e∈b} E[a_e^2]``, clipped to amplitude bounds.

    Returns a new ``(n_ep,)`` jitter array (s²). Requires ``noise.epoch_backends``.
    """
    if noise.epoch_backends is None:
        raise ValueError(
            "learn_kernel_ecorr requires EpochKernelNoise.epoch_backends "
            "(build the operator with EpochKernelNoise.from_backends)"
        )
    lower = float(ecorr_min) ** 2
    upper = float(ecorr_max) ** 2
    if not (upper >= lower > 0.0):
        raise ValueError("require 0 < ecorr_min <= ecorr_max")
    a2 = np.asarray(moments.a_second_moment, dtype=float)
    new_jitter = np.array(noise.jitter, dtype=float, copy=True)
    backends = noise.epoch_backends
    for backend in sorted(set(backends)):
        idxs = [i for i, b in enumerate(backends) if b == backend]
        lam = float(np.mean(a2[idxs]))
        new_jitter[idxs] = float(np.clip(lam, lower, upper))
    return new_jitter


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
    factorization: Factorization | None = None,
    kernel_ecorr: EpochKernelNoise | None = None,
    learn_kernel_ecorr: bool = False,
    ecorr_min: float = 1.0e-9,
    ecorr_max: float = 1.0e-5,
    initial_efac: Mapping[str, float] | None = None,
    initial_equad: Mapping[str, float] | None = None,
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
    factorization
        Optional :class:`~pylk.flexfit.fasttnt.Factorization` applied after
        ``assemble``.
    kernel_ecorr
        Optional Topology-B ECORR operator. Its ``diagonal`` is replaced each
        iteration with the current EFAC/EQUAD ``D``. With
        ``learn_kernel_ecorr=False`` (default) the jitter is pinned; with
        ``True`` the §3.7 per-backend epoch M-step updates it (mode 2.2).
    learn_kernel_ecorr
        Learn ECORR amplitudes in the kernel (requires ``kernel_ecorr`` built
        by :meth:`~pylk.flexfit.noise.EpochKernelNoise.from_backends`).
    ecorr_min, ecorr_max
        Amplitude bounds (seconds) for the kernel ECORR M-step; match
        ``ecorr_blocks`` defaults.
    initial_efac, initial_equad
        Warm-start maps from a release noisedict (§1.7 mode 2).
    tolerance
        Stop when every backend's ``(efac, equad)`` fractional change (equad
        measured against the backend's median toaerr) — and, when learning
        kernel ECORR, the ECORR amplitude change — drops below this.
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

    if learn_kernel_ecorr and kernel_ecorr is None:
        raise ValueError("learn_kernel_ecorr=True requires kernel_ecorr=...")
    if kernel_ecorr is not None and not blocks and timing is None:
        # e_i would fall back to y^2, which charges the epoch-correlated power
        # to EFAC — the exact bias the kernel statistic exists to prevent.
        raise ValueError(
            "kernel_ecorr needs a basis (blocks= and/or timing=): a pure "
            "white-noise fit cannot subtract the epoch waveform, so ECORR "
            "power would be absorbed into efac/equad"
        )
    if learn_kernel_ecorr and kernel_ecorr.epoch_backends is None:
        raise ValueError(
            "learn_kernel_ecorr requires EpochKernelNoise.epoch_backends "
            "(use EpochKernelNoise.from_backends)"
        )

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
    if model is not None and factorization is not None:
        model = factorization.apply(model)
    if model is not None and model.n_obs != toaerrs.shape[0]:
        raise ValueError("basis n_obs does not match toaerrs")
    if y.shape != toaerrs.shape:
        raise ValueError("residuals must be aligned with toaerrs")

    efac = {b: 1.0 for b in backends}
    equad = {b: 0.0 for b in backends}
    if initial_efac:
        efac.update({str(k): float(v) for k, v in initial_efac.items()})
    if initial_equad:
        equad.update({str(k): float(v) for k, v in initial_equad.items()})
    scale = {b: float(np.median(toaerrs[masks[b]])) for b in backends}

    history: list[dict[str, tuple[float, float]]] = []
    solve: FlexiblePhiResult | None = None
    phi_prev: np.ndarray | None = None
    converged = False
    iteration = 0
    active_kernel: EpochKernelNoise | None = None
    current_jitter: np.ndarray | None = (
        np.array(kernel_ecorr.jitter, dtype=float, copy=True)
        if kernel_ecorr is not None
        else None
    )
    for iteration in range(1, max_iterations + 1):
        variance = np.empty_like(sigma2)
        for b in backends:
            m = masks[b]
            variance[m] = efac[b] ** 2 * sigma2[m] + equad[b] ** 2
        if kernel_ecorr is None:
            noise: DiagonalNoise | EpochKernelNoise = DiagonalNoise(variance)
            active_kernel = None
        else:
            assert current_jitter is not None
            noise = replace(kernel_ecorr, diagonal=variance, jitter=current_jitter)
            active_kernel = noise
        if model is not None:
            solve = solve_flexible_phi(
                y,
                model,
                noise,
                n_sweeps=n_sweeps,
                tolerance=sweep_tolerance,
                initial_phi=phi_prev,
            )
            phi_prev = np.asarray(solve.phi_diagonal, dtype=float)
            if learn_kernel_ecorr:
                assert isinstance(noise, EpochKernelNoise)
                moments = kernel_ecorr_moments(y, model, solve, noise)
                e = moments.e
                new_jitter = _kernel_ecorr_m_step(
                    moments,
                    noise,
                    ecorr_min=ecorr_min,
                    ecorr_max=ecorr_max,
                )
            else:
                e = expected_squared_residuals(y, model, solve, noise=noise)
                new_jitter = None
        else:
            e = y**2
            new_jitter = None

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
        if new_jitter is not None and current_jitter is not None:
            # Fractional change in ECORR amplitude (seconds).
            old_amp = np.sqrt(current_jitter)
            new_amp = np.sqrt(new_jitter)
            delta = max(
                delta,
                float(np.max(np.abs(new_amp - old_amp) / np.maximum(old_amp, 1e-30))),
            )
            current_jitter = new_jitter
            active_kernel = replace(
                kernel_ecorr, diagonal=variance, jitter=current_jitter
            )
        history.append({b: (efac[b], equad[b]) for b in backends})
        if delta < tolerance:
            converged = True
            break

    # Final EB solve at the converged white noise so the reported fit matches it.
    variance = np.empty_like(sigma2)
    for b in backends:
        m = masks[b]
        variance[m] = efac[b] ** 2 * sigma2[m] + equad[b] ** 2
    if kernel_ecorr is None:
        final_noise: DiagonalNoise | EpochKernelNoise = DiagonalNoise(variance)
        active_kernel = None
    else:
        assert current_jitter is not None
        final_noise = replace(kernel_ecorr, diagonal=variance, jitter=current_jitter)
        active_kernel = final_noise
    solve = (
        solve_flexible_phi(
            y,
            model,
            final_noise,
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
        residuals=y,
        kernel=active_kernel,
    )
