"""Build flexfit GP blocks and white noise from Discovery primitives.

This adapter is the single place that knows Discovery conventions: the
``fourierbasis``/``dmfourierbasis`` column layout (sin/cos interleaved,
``f_i = i / T_span`` Hz), the ``powerlaw`` normalization (including the ``df``
factor and ``f_yr`` reference), and the ``makenoise_measurement`` white kernel.
Reusing Discovery's own functions keeps the quick-look fit from drifting away
from the production stacks it is meant to initialize.

Discovery is imported lazily so the numerical core imports without it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..basis import BasisBlock, VarianceGroup, fourier_pair_groups
from ..noise import (
    DiagonalNoise,
    EpochKernelNoise,
    NoiseOperator,
    ecorr_from_kernel as ecorr_from_kernel,
)
from ..projection import SpectrumProjection, project_spectrum
from ..waveform import WaveformAnalysis, analyze_waveforms, frequencies_from_blocks

# Sensible default induced-RMS bounds for a learnable red/DM group: 0.1 ns to 10 us.
# The upper bound guards a free-spectrum EM from absorbing white noise into the
# highest-frequency modes; loosen it for pulsars with large red/DM signals.
DEFAULT_SIGMA_MIN = 1.0e-10
DEFAULT_SIGMA_MAX = 1.0e-5

# Groups with this update_from_sweep are held fixed (never updated) — used to
# reconstruct a waveform at a supplied MLE spectrum.
NEVER_UPDATE = 10**9


def _coefficient_names(prefix: str, n_freq: int) -> tuple[str, ...]:
    names: list[str] = []
    for k in range(n_freq):
        names.append(f"{prefix}_f{k}_sin")
        names.append(f"{prefix}_f{k}_cos")
    return tuple(names)


def _fixed_pair_groups(
    prefix: str, n_freq: int, phi: np.ndarray
) -> tuple[VarianceGroup, ...]:
    """Per-frequency groups pinned to a supplied ``Phi`` diagonal (never updated)."""
    phi = np.asarray(phi, dtype=float)
    groups: list[VarianceGroup] = []
    for k in range(n_freq):
        rho = float(0.5 * (phi[2 * k] + phi[2 * k + 1]))  # sin/cos share the value
        groups.append(
            VarianceGroup(
                name=f"{prefix}_f{k}",
                indices=(2 * k, 2 * k + 1),
                lower=rho,
                upper=rho,
                initial=rho,
                update_from_sweep=NEVER_UPDATE,
            )
        )
    return tuple(groups)


def _fourier_block(
    matrix: np.ndarray,
    freqs: np.ndarray,
    df: np.ndarray,
    *,
    name: str,
    kind: str,
    n_freq: int,
    sigma_min: float,
    sigma_max: float,
    update_from_sweep: int,
    fixed_phi: np.ndarray | None = None,
    extra_metadata: dict | None = None,
) -> BasisBlock:
    if fixed_phi is not None:
        groups = _fixed_pair_groups(name, n_freq, fixed_phi)
    else:
        groups = fourier_pair_groups(
            matrix,
            prefix=name,
            n_freq=n_freq,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            update_from_sweep=update_from_sweep,
            interleaved=True,
        )
    metadata = {
        "frequencies": np.asarray(freqs, dtype=float),
        "df": np.asarray(df, dtype=float),
        "n_freq": int(n_freq),
        "layout": "interleaved-sincos",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return BasisBlock(
        name=name,
        matrix=np.asarray(matrix, dtype=float),
        coefficient_names=_coefficient_names(name, n_freq),
        groups=groups,
        kind=kind,  # type: ignore[arg-type]
        metadata=metadata,
    )


def _powerlaw_phi(freqs, df, log10_A, gamma) -> np.ndarray:
    import discovery as ds

    return np.asarray(ds.powerlaw(freqs, df, float(log10_A), float(gamma)), dtype=float)


def red_noise_block(
    psr,
    *,
    components: int = 30,
    name: str = "red",
    T: float | None = None,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
    update_from_sweep: int = 1,
    log10_A: float | None = None,
    gamma: float | None = None,
) -> BasisBlock:
    """Achromatic red-noise Fourier block (Discovery ``fourierbasis``).

    With ``log10_A``/``gamma`` supplied, the per-frequency ``Phi`` is pinned to
    that power law and never updated — use this to reconstruct a waveform at
    fixed MLE hyperparameters. Otherwise the variances are learned (free
    spectrum) within the induced-RMS bounds.
    """
    import discovery as ds

    freqs, df, fmat = ds.fourierbasis(psr, components, T=T)
    fixed = None
    if log10_A is not None and gamma is not None:
        fixed = _powerlaw_phi(freqs, df, log10_A, gamma)
    return _fourier_block(
        fmat,
        freqs,
        df,
        name=name,
        kind="red",
        n_freq=components,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        update_from_sweep=update_from_sweep,
        fixed_phi=fixed,
    )


def dm_noise_block(
    psr,
    *,
    components: int = 30,
    name: str = "dm",
    fref: float = 1400.0,
    T: float | None = None,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
    update_from_sweep: int = 1,
    log10_A: float | None = None,
    gamma: float | None = None,
) -> BasisBlock:
    """Dispersion-variation Fourier block (Discovery ``dmfourierbasis``, alpha=2).

    Supplying ``log10_A``/``gamma`` pins ``Phi`` to that power law (fixed MLE
    reconstruction); otherwise the free-spectrum variances are learned.
    """
    import discovery as ds

    freqs, df, fmat = ds.dmfourierbasis(psr, components, T=T, fref=fref)
    fixed = None
    if log10_A is not None and gamma is not None:
        fixed = _powerlaw_phi(freqs, df, log10_A, gamma)
    return _fourier_block(
        fmat,
        freqs,
        df,
        name=name,
        kind="dm",
        n_freq=components,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        update_from_sweep=update_from_sweep,
        fixed_phi=fixed,
        extra_metadata={"fref": float(fref), "alpha": 2.0},
    )


def chromatic_noise_block(
    psr,
    *,
    components: int = 30,
    alpha: float = 4.0,
    name: str = "chrom",
    fref: float = 1400.0,
    T: float | None = None,
    tndm: bool = False,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
    update_from_sweep: int = 1,
    log10_A: float | None = None,
    gamma: float | None = None,
) -> BasisBlock:
    """General chromatic Fourier block (Discovery ``make_dmfourierbasis(alpha)``)."""
    import discovery as ds

    basis = ds.make_dmfourierbasis(alpha=alpha, tndm=tndm)
    freqs, df, fmat = basis(psr, components, T=T, fref=fref)
    fixed = None
    if log10_A is not None and gamma is not None:
        fixed = _powerlaw_phi(freqs, df, log10_A, gamma)
    return _fourier_block(
        fmat,
        freqs,
        df,
        name=name,
        kind="chromatic",
        n_freq=components,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        update_from_sweep=update_from_sweep,
        fixed_phi=fixed,
        extra_metadata={"fref": float(fref), "alpha": float(alpha)},
    )


def _selection_labels(psr, selection) -> np.ndarray:
    """Resolve the backend/PTA label array that keys a release noisedict."""
    if selection is None:
        return np.asarray(psr.backend_flags)
    if callable(selection):
        return np.asarray(selection(psr))
    return np.asarray(selection)


def _equad_seconds(noisedict: dict, name: str, backend: str) -> tuple[float, bool]:
    """Return ``(equad_seconds, is_tnequad)`` from a release noisedict entry."""
    tn_key = f"{name}_{backend}_log10_tnequad"
    t2_key = f"{name}_{backend}_log10_t2equad"
    has_tn = tn_key in noisedict
    has_t2 = t2_key in noisedict
    if has_tn and has_t2:
        raise ValueError(
            f"noisedict has both tnequad and t2equad keys for {name}_{backend}"
        )
    if not has_tn and not has_t2:
        raise KeyError(
            f"noisedict missing equad key for {name}_{backend} "
            f"(expected {tn_key!r} or {t2_key!r})"
        )
    if has_tn:
        return float(10.0 ** noisedict[tn_key]), True
    return float(10.0 ** noisedict[t2_key]), False


def white_noise(
    psr,
    noisedict: dict,
    *,
    selection=None,
    ecorr: bool = False,
    dt: float = 1.0,
    ecorr_min: float = 1e-9,
) -> NoiseOperator:
    """Diagonal or kernel-ECORR white noise from a release noisedict.

    ``selection`` is the label array (or a callable returning one) that keys the
    dict — PTA labels for AEI combinations, fine backends for per-backend dicts.
    The same array is used for ``D`` and for the ECORR epochs. The equad
    convention is read from the dict's own key, not from a flag.

    With ``ecorr=True`` returns an :class:`~pylk.flexfit.noise.EpochKernelNoise`.
    """
    labels = _selection_labels(psr, selection)
    sigma = np.asarray(psr.toaerrs, dtype=float)
    name = psr.name
    variance = np.empty_like(sigma)
    backends = sorted({str(x) for x in labels.tolist() if str(x)})
    if not any(f"{name}_{b}_efac" in noisedict for b in backends):
        raise ValueError(
            f"no {name}_<label>_efac key matches the selection labels "
            f"{backends[:5]!r}{' ...' if len(backends) > 5 else ''} — selection "
            "mismatch (pass selection= with the label array that keys this "
            "noisedict, e.g. the -pta flags for a combination dict)"
        )
    for b in backends:
        m = labels == b
        efac = float(noisedict[f"{name}_{b}_efac"])
        q, tn = _equad_seconds(noisedict, name, b)
        # tnequad: N = efac^2 sigma^2 + q^2 ; t2equad: N = efac^2 (sigma^2 + q^2)
        variance[m] = (
            efac**2 * sigma[m] ** 2 + q**2 if tn else efac**2 * (sigma[m] ** 2 + q**2)
        )
    if not ecorr:
        return DiagonalNoise(variance)
    amps = {
        b: float(10.0 ** noisedict[f"{name}_{b}_log10_ecorr"])
        for b in backends
        if f"{name}_{b}_log10_ecorr" in noisedict
    }
    return EpochKernelNoise.from_backends(
        diagonal=variance,
        toas=np.asarray(psr.toas, dtype=float),
        backend_flags=labels,
        ecorr=amps,
        dt=dt,
        ecorr_min=ecorr_min,
    )


def ecorr_blocks(
    psr,
    *,
    selection_labels: np.ndarray,
    ecorr_min: float = 1.0e-9,
    ecorr_max: float = 1.0e-5,
    update_from_sweep: int = 1,
) -> list[BasisBlock]:
    """Per-backend ECORR epoch-averaging blocks (enterprise convention).

    Mirrors Discovery's ``makegp_ecorr(..., enterprise=True)`` basis exactly:
    per backend, TOAs are quantized (``dt = 1 s``) and only epochs with two or
    more TOAs get a column. Each backend becomes one block with a single
    variance group over its epoch columns, so the flexible-``Phi`` EB sweep
    learns ``ecorr_b^2`` directly (in s^2) jointly with EFAC/EQUAD and the GP
    spectra. Backends with no multi-TOA epochs are skipped (no ECORR
    information). Bounds are ecorr amplitudes in seconds.

    Quantization now runs on each backend's own TOAs through
    :func:`pylk.flexfit.fasttnt.quantize` instead of Discovery's
    ``quantize(toas * mask)``. The epoch partition is unchanged (verified
    column-for-column on real data); the change drops a Discovery import and
    makes these epochs byte-identical to
    :meth:`~pylk.flexfit.noise.EpochKernelNoise.from_backends`, which is the
    precondition for Topology-A/B parity.
    """
    from ..fasttnt import quantize

    labels = np.asarray(selection_labels)
    toas = np.asarray(psr.toas, dtype=float)
    blocks: list[BasisBlock] = []
    for backend in [b for b in sorted(set(labels.tolist())) if b != ""]:
        mask = labels == backend
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        # Quantize this backend's TOAs alone (same rule as EpochKernelNoise.from_backends).
        bins_local = quantize(toas[idx])
        uniques, counts = np.unique(bins_local, return_counts=True)
        epoch_masks = []
        for local_id, cnt in zip(uniques.tolist(), counts.tolist()):
            if cnt <= 1:
                continue
            col = np.zeros(toas.shape[0], dtype=float)
            col[idx[bins_local == local_id]] = 1.0
            epoch_masks.append(col)
        if not epoch_masks:
            continue
        U = np.column_stack(epoch_masks)
        n_col = U.shape[1]
        group = VarianceGroup(
            f"ecorr_{backend}",
            tuple(range(n_col)),
            lower=float(ecorr_min) ** 2,
            upper=float(ecorr_max) ** 2,
            update_from_sweep=int(update_from_sweep),
        )
        blocks.append(
            BasisBlock(
                name=f"ecorr_{backend}",
                matrix=U,
                coefficient_names=tuple(f"epoch{i}" for i in range(n_col)),
                groups=(group,),
                kind="ecorr",
                metadata={"backend": str(backend), "n_epochs": n_col},
            )
        )
    return blocks


def ecorr_from_fit(solve) -> dict[str, float]:
    """Per-backend ECORR amplitudes (seconds) from a fit's group variances."""
    out: dict[str, float] = {}
    for name, rho in solve.group_variances.items():
        if name.startswith("ecorr_"):
            out[name[len("ecorr_") :]] = float(np.sqrt(max(rho, 0.0)))
    return out


def white_noise_from_variance(variance: np.ndarray) -> DiagonalNoise:
    """Wrap a precomputed per-TOA variance (e.g. ``build_map_white_variance``)."""
    return DiagonalNoise(np.asarray(variance, dtype=float))


def powerlaw_spectrum(block: BasisBlock) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``theta=[log10_A, gamma] -> phi`` using Discovery's ``powerlaw``.

    The returned callable evaluates the same per-bin variance (including the
    ``df`` factor and ``f_yr`` reference) Discovery uses for its GP prior, so a
    projection result is directly comparable with production power-law noise.
    """
    import discovery as ds

    freqs = np.asarray(block.metadata["frequencies"], dtype=float)
    df = np.asarray(block.metadata["df"], dtype=float)

    def spectrum(theta: np.ndarray) -> np.ndarray:
        log10_A, gamma = float(theta[0]), float(theta[1])
        return np.asarray(ds.powerlaw(freqs, df, log10_A, gamma), dtype=float)

    return spectrum


def project_powerlaw(
    result,
    block: BasisBlock,
    *,
    log10_A0: float = -14.0,
    gamma0: float = 3.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-20.0, -10.0),
        (0.0, 7.0),
    ),
) -> SpectrumProjection:
    """Project a fitted block's second moments onto a Discovery power law.

    ``result`` is a :class:`~pylk.flexfit.flexible_phi.FlexiblePhiResult` or a
    :class:`~pylk.flexfit.fastfit.FastFitResult`; the block's per-frequency second
    moments are read by block name.
    """
    solve = getattr(result, "solve", result)
    second = solve.block_second_moment(block.name)
    return project_spectrum(
        second,
        powerlaw_spectrum(block),
        theta0=(log10_A0, gamma0),
        parameter_names=("log10_A", "gamma"),
        bounds=bounds,
    )


# --------------------------------------------------------------------------- #
# Waveform reconstruction via Discovery's own PulsarLikelihood.conditional
# --------------------------------------------------------------------------- #
def _use_metamath_kernels():
    """Opt into Discovery's graph path. Default is still ``kernels="matrix"``."""
    import discovery as ds

    ds.config(kernels="metamath")


def powerlaw_gp(psr, kind, *, components, log10_A, gamma, name=None, fref=1400.0):
    """Build a fixed power-law Discovery GP and its parameter dict.

    Returns ``(gp, params)`` where ``gp`` is a Discovery ``VariableGP`` (red or
    chromatic Fourier) and ``params`` pins its ``log10_A``/``gamma``. ``kind`` is
    ``"red"`` (achromatic) or ``"dm"``/``"chromatic"`` (chromatic, ν⁻ᵅ).
    """
    import discovery as ds

    _use_metamath_kernels()
    name = name or kind
    if kind == "red":
        gp = ds.makegp_fourier(psr, ds.powerlaw, components=components, name=name)
    elif kind in ("dm", "chromatic"):
        gp = ds.makegp_fourier(
            psr,
            ds.powerlaw,
            components=components,
            fourierbasis=ds.dmfourierbasis,
            name=name,
        )
    else:
        raise ValueError(f"unknown GP kind {kind!r}")
    params = {
        f"{psr.name}_{name}_log10_A": float(log10_A),
        f"{psr.name}_{name}_gamma": float(gamma),
    }
    return gp, params


def map_powerlaw_hypers(
    psr,
    *,
    variance,
    timing,
    specs,
    residuals=None,
    method: str = "Nelder-Mead",
):
    """MAP/MLE of red/DM power-law hyperparameters via Discovery's ``logL``.

    A second, independent way to build ``Phi`` (the opposite of flexfit's
    empirical Bayes): maximize the marginal Discovery likelihood over the GP
    power-law hyperparameters, with the timing model marginalized. This reuses
    Discovery's own ``PulsarLikelihood.logL`` — the same objective the MAP
    noisedict and NUTS sampler use — so the result is consistent with them.

    ``specs`` is an iterable of dicts ``{name, kind, components, fref?, bounds?,
    start?}``. Returns ``{name: {"log10_A", "gamma"}}``.
    """
    from scipy.optimize import minimize

    specs = [dict(s) for s in specs]
    gp_list, keys, x0, bounds = [], [], [], []
    for spec in specs:
        name = spec["name"]
        gp, _ = powerlaw_gp(
            psr,
            spec["kind"],
            components=spec["components"],
            log10_A=spec.get("start", (-14.0, 3.0))[0],
            gamma=spec.get("start", (-14.0, 3.0))[1],
            name=name,
            fref=spec.get("fref", 1400.0),
        )
        gp_list.append(gp)
        b = spec.get("bounds", ((-20.0, -10.0), (0.0, 7.0)))
        start = spec.get("start", (-14.0, 3.0))
        keys += [f"{psr.name}_{name}_log10_A", f"{psr.name}_{name}_gamma"]
        x0 += [start[0], start[1]]
        bounds += [b[0], b[1]]

    like, _ = _reconstruction_likelihood(
        psr, variance, timing, gp_list, residuals=residuals
    )

    def neg_logL(x):
        params = dict(zip(keys, map(float, x)))
        return -float(like.logL(params))

    res = minimize(neg_logL, np.asarray(x0, dtype=float), method=method, bounds=bounds)
    out: dict[str, dict[str, float]] = {}
    for i, spec in enumerate(specs):
        out[spec["name"]] = {
            "log10_A": float(res.x[2 * i]),
            "gamma": float(res.x[2 * i + 1]),
        }
    return out


def _reconstruction_likelihood(psr, variance, timing, gp_list, *, residuals=None):
    import discovery as ds
    from discovery import metamath
    from discovery import utils as ds_utils

    # Graph-path kernels and PulsarLikelihood. Default is still matrix;
    # Transport / metamath.NoiseMatrix are refused unless this is set first.
    _use_metamath_kernels()
    variance = np.asarray(variance, dtype=float)
    y = psr.residuals if residuals is None else np.asarray(residuals, dtype=float)
    timing_signals = _timing_signals(psr, timing)
    noise_kernel = metamath.NoiseMatrix(ds_utils.jnparray(variance))
    like = ds.PulsarLikelihood([y, noise_kernel, *timing_signals, *gp_list])
    if like.delay:
        raise ValueError(
            "conditional reconstruction needs a delay-free (marginalized) timing "
            "model; pass a no-refit ctx's discovery_signals() or a design matrix"
        )
    return like, timing_signals


def _timing_signals(psr, timing):
    if isinstance(timing, np.ndarray):
        from discovery import signals as dsig

        basis = _normalized_design(np.asarray(timing, dtype=float))
        return [dsig.makegp_improper(psr, basis, constant=1.0e40, name="timingmodel")]
    return list(timing)


def _normalized_design(design_matrix: np.ndarray) -> np.ndarray:
    from nltiming.whitening import normalized_basis

    return np.asarray(
        normalized_basis(np.asarray(design_matrix, dtype=float)), dtype=float
    )


def timing_marginalization_block(
    design_matrix, *, name: str = "timingmodel"
) -> BasisBlock:
    """A broad-prior (``1e40``) timing-marginalization block from a design matrix.

    Columns are unit-normalized (span-preserving, float64-safe) and never
    updated, reproducing the Enterprise/Discovery improper timing marginalization
    as a ``flexfit`` block so its conditional-mean waveform is reconstructed
    jointly with the GP blocks.
    """
    basis = _normalized_design(design_matrix)
    n_col = basis.shape[1]
    groups = tuple(
        VarianceGroup(
            f"{name}_{i}",
            (i,),
            1.0e40,
            1.0e40,
            initial=1.0e40,
            update_from_sweep=NEVER_UPDATE,
        )
        for i in range(n_col)
    )
    return BasisBlock(
        name=name,
        matrix=basis,
        coefficient_names=tuple(f"{name}_c{i}" for i in range(n_col)),
        groups=groups,
        kind="timing",
        metadata={"role": "analytically-marginalized", "n_col": n_col},
    )


def reconstruct_waveforms(
    pulsar,
    *,
    variance,
    design_matrix,
    spectra,
    residuals=None,
    n_sweeps: int = 2,
) -> WaveformAnalysis:
    """One joint reconstruction of **all** stochastic components (timing + GPs).

    Solves the joint conditional over ``T = [timing_marg, F_red, F_dm, ...]`` with
    timing marginalized at ``1e40`` and each GP pinned to its ``spectra`` power
    law, using ``pylk.flexfit.solve_flexible_phi`` — bit-identical to Discovery's
    Woodbury but, unlike ``PulsarLikelihood.conditional``, it also returns the
    timing-model coefficients (Discovery folds a constant timing GP into ``N``).

    ``pulsar.toas`` are raw seconds (``MJD × 86400``, the same convention
    MetaPulsar materializes); ``toa_mjd`` is therefore ``toas / 86400.0``.

    Parameters
    ----------
    pulsar
        MetaPulsar pulsar (or Discovery pulsar); ``pulsar.residuals`` used unless
        ``residuals`` is given.
    variance
        Per-TOA white-noise variance (s²).
    design_matrix
        Timing design to marginalize (e.g. ``pulsar.Mmat`` — the linearized model;
        set the sampled/nonlinear params at their MPE in ``residuals`` first).
    spectra
        Mapping ``name -> {"kind": "red"|"dm"|"chromatic", "components": int,
        "log10_A": float, "gamma": float, "fref"?: float}``.
    residuals
        Residual vector to condition on (defaults to ``pulsar.residuals``); pass
        residuals with the sampled timing params set at their MPE.

    Returns
    -------
    WaveformAnalysis
        Dict-like over per-block conditional-mean waveforms at the TOAs
        (``analysis["red"]`` etc.), plus staged residuals and
        :meth:`~pylk.flexfit.waveform.WaveformAnalysis.predict_gp` returning a
        :class:`~pylk.flexfit.waveform.GPBand`.
    """
    from pylk.flexfit import assemble, solve_flexible_phi

    blocks = [timing_marginalization_block(design_matrix)]
    for name, spec in spectra.items():
        kind = spec.get("kind", name)
        builder = red_noise_block if kind == "red" else dm_noise_block
        kw = {
            "components": spec["components"],
            "name": name,
            "log10_A": spec["log10_A"],
            "gamma": spec["gamma"],
        }
        if kind != "red":
            kw["fref"] = spec.get("fref", 1400.0)
        blocks.append(builder(pulsar, **kw))

    y = pulsar.residuals if residuals is None else residuals
    res = solve_flexible_phi(
        np.asarray(y, dtype=float),
        assemble(blocks),
        white_noise_from_variance(variance),
        n_sweeps=n_sweeps,
    )

    toas = np.asarray(pulsar.toas, dtype=float)
    return analyze_waveforms(
        np.asarray(y, dtype=float),
        np.asarray(variance, dtype=float),
        res,
        toas=toas,
        toa_mjd=toas / 86400.0,
        block_kinds={b.name: b.kind for b in blocks},
        block_frequencies=frequencies_from_blocks(blocks),
        freqs_mhz=np.asarray(pulsar.freqs, dtype=float),
    )
