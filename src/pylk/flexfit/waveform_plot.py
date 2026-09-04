"""Optional matplotlib helper for EB waveform panel arrays.

Layout matches ``paper/code/figures/plots.py::plot_waveform_panels`` (the
Kepler / Condor offline figure style): six x-aligned residual/GP panels with a
frequency colorbar and a rotated N(0,1) density histogram for whitened
residuals.

Import only when matplotlib is available; not re-exported from ``pylk.flexfit``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .waveform import WaveformPanelArrays

_CMAP = "viridis"


def _scatter(ax, mjd, y, freq):
    return ax.scatter(mjd, y, s=5, c=freq, cmap=_CMAP, alpha=0.5)


def _rms_us(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(a**2)))


def plot_waveform_panels(
    panels: WaveformPanelArrays,
    *,
    pulsar_name: str | None = None,
    title: str | None = None,
    fref_mhz: float = 1400.0,
    include_dm: bool = True,
    axs: Any = None,
):
    """Draw the stacked (a)–(g) waveform figure used by the paper / Condor jobs.

    Panels (top to bottom): (a) pre-fit residuals, (b) timing subtracted,
    (c) red-noise GP, (d) DM delay @ ``fref_mhz``, (e) everything subtracted,
    (f) normalized residuals. Panel (g) is a 90°-rotated density histogram of
    the normalized residuals under the frequency colorbar.

    ``include_dm=False`` omits panel (d) (DMX data, where DM is already in the
    timing model). Remaining panel letters stay (a)–(c), (e)–(g).

    Returns the Matplotlib ``Figure``. Does not call ``plt.show()``.
    ``axs`` is accepted only for backward compatibility and must be ``None``.
    """
    if axs is not None:
        raise ValueError(
            "plot_waveform_panels no longer accepts a caller-supplied axs grid; "
            "it builds the paper/Condor stacked layout itself"
        )

    import matplotlib.pyplot as plt
    from scipy import stats

    mjd = np.asarray(panels.mjd, dtype=float)
    freq = np.asarray(panels.freq_mhz, dtype=float)
    if not np.any(np.isfinite(freq)):
        freq = np.zeros_like(mjd)

    stage_rms = dict(panels.stage_rms_us or {})
    rms_a = stage_rms.get("raw", _rms_us(panels.resid_us))
    rms_b = stage_rms.get("after_timing", _rms_us(panels.after_timing_us))
    rms_e = stage_rms.get("after_all", _rms_us(panels.after_all_us))
    z = np.asarray(panels.z, dtype=float)
    z_finite = z[np.isfinite(z)]
    z_mean = float(np.mean(z_finite)) if z_finite.size else 0.0
    z_std = float(np.std(z_finite)) if z_finite.size else 1.0

    n_rows = 6 if include_dm else 5
    fig = plt.figure(figsize=(12, 15 if include_dm else 13))
    gs = fig.add_gridspec(
        n_rows,
        2,
        width_ratios=[26, 1.5],
        hspace=0.42,
        wspace=0.03,
        top=0.965,
        bottom=0.045,
        left=0.08,
        right=0.99,
    )
    ax = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
    for a in ax[1:]:
        a.sharex(ax[0])
    cax = fig.add_subplot(gs[0 : n_rows - 1, 1])
    hax = fig.add_subplot(gs[n_rows - 1, 1], sharey=ax[-1])

    sc = _scatter(ax[0], mjd, panels.resid_us, freq)
    ax[0].set_title(
        f"(a) original par-file pre-fit residuals   (RMS {rms_a:.2f} µs)",
        fontsize=9,
    )
    ax[0].set_ylabel("residual (µs)")

    _scatter(ax[1], mjd, panels.after_timing_us, freq)
    ax[1].set_title(
        f"(b) timing-model subtracted   (RMS {rms_b:.2f} µs)",
        fontsize=9,
    )
    ax[1].set_ylabel("residual (µs)")

    gm = np.asarray(panels.grid_mjd, dtype=float)
    red_mean = np.asarray(panels.red_mean_us, dtype=float)
    red_std = np.asarray(panels.red_std_us, dtype=float)
    if gm.size and red_mean.size:
        ax[2].fill_between(
            gm,
            red_mean - red_std,
            red_mean + red_std,
            color="C3",
            alpha=0.25,
            label="±1σ",
        )
        ax[2].plot(gm, red_mean, "C3-", lw=1.2, label="RN mean")
        ax[2].legend(loc="upper right", fontsize=7)
    ax[2].set_title("(c) red-noise GP prediction (mean ± 1σ)", fontsize=9)
    ax[2].set_ylabel("RN (µs)")

    row = 3
    if include_dm:
        dm_mean = np.asarray(panels.dm_mean_us, dtype=float)
        dm_std = np.asarray(panels.dm_std_us, dtype=float)
        if gm.size and dm_mean.size:
            ax[row].fill_between(
                gm,
                dm_mean - dm_std,
                dm_mean + dm_std,
                color="C0",
                alpha=0.25,
                label="±1σ",
            )
            ax[row].plot(
                gm,
                dm_mean,
                "C0-",
                lw=1.2,
                label=f"DM @ {fref_mhz:g} MHz",
            )
            ax[row].legend(loc="upper right", fontsize=7)
        ax[row].set_title(
            f"(d) DM-variation delay time series @ {fref_mhz:g} MHz (mean ± 1σ)",
            fontsize=9,
        )
        ax[row].set_ylabel("DM delay (µs)")
        row += 1

    _scatter(ax[row], mjd, panels.after_all_us, freq)
    ax[row].set_title(
        f"(e) everything subtracted (timing + GPs)   (RMS {rms_e:.2f} µs)",
        fontsize=9,
    )
    ax[row].set_ylabel("residual (µs)")
    row += 1

    _scatter(ax[row], mjd, z, freq)
    ax[row].set_title(
        f"(f) normalized residuals   (mean {z_mean:.2f}, std {z_std:.2f})",
        fontsize=9,
    )
    ax[row].set_ylabel(r"$r / \sigma_{\rm white}$")
    ax[row].set_xlabel("MJD")

    for a in ax:
        a.axhline(0.0, color="0.7", lw=0.6, zorder=0)

    fig.colorbar(sc, cax=cax, label="observing frequency (MHz)")

    if z_finite.size:
        z_lo = float(np.min(z_finite))
        z_hi = float(np.max(z_finite))
        if z_hi <= z_lo:
            z_hi = z_lo + 1.0
        bins = np.linspace(z_lo, z_hi, 40)
        hax.hist(
            z_finite,
            bins=bins,
            density=True,
            orientation="horizontal",
            color="steelblue",
            alpha=0.6,
        )
        zg = np.linspace(z_lo, z_hi, 200)
        hax.plot(stats.norm.pdf(zg), zg, "r-", lw=1.2)
    hax.tick_params(labelleft=False, labelbottom=False)
    hax.set_xlabel("density", fontsize=8)
    hax.set_title("(g)", fontsize=9, pad=4)

    if title is None:
        psr = pulsar_name or ""
        gp = "timing + RN + DM" if include_dm else "timing + RN"
        title = f"{psr}: multi-PTA [{gp}] reconstruction".strip()
        if title.startswith(":"):
            title = title[1:].lstrip()
    fig.suptitle(title, y=0.995, fontsize=11)
    return fig
