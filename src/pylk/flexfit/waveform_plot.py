"""Optional matplotlib helper for EB waveform panel arrays.

Import only when matplotlib is available; not re-exported from ``pylk.flexfit``.
"""

from __future__ import annotations

import numpy as np

from .waveform import WaveformPanelArrays


def plot_waveform_panels(
    panels: WaveformPanelArrays,
    *,
    axs: np.ndarray | None = None,
    show_dm: bool = True,
    title: str | None = None,
) -> np.ndarray:
    """Draw panels (a)–(g) into a 4×2 or 3×2 Axes grid.

    Layout (locked):
      [a raw]     [b after timing]
      [c red GP]  [d dm GP]          # d hidden if show_dm is False and dm arrays empty
      [e after all] [f whitened z]
      [g Q-Q of z spanning both columns]

    Returns the Axes array. Does not call ``plt.show()``.
    Requires matplotlib; use ``pytest.importorskip("matplotlib")`` in tests.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import probplot

    dm_present = panels.dm_mean_us.size > 0
    draw_dm = bool(show_dm and dm_present)

    if axs is None:
        nrows = 4 if draw_dm else 3
        fig, axs = plt.subplots(nrows, 2, figsize=(8.0, 2.4 * nrows), squeeze=False)
        if title:
            fig.suptitle(title)
    else:
        axs = np.atleast_2d(axs)

    mjd = np.asarray(panels.mjd, dtype=float)
    sigma = np.asarray(panels.sigma_us, dtype=float)

    ax_a, ax_b = axs[0, 0], axs[0, 1]
    ax_a.errorbar(
        mjd,
        panels.resid_us,
        yerr=sigma,
        fmt=".",
        ms=2,
        elinewidth=0.4,
        alpha=0.7,
    )
    ax_a.set_title("(a) raw")
    ax_a.set_ylabel(r"residual (µs)")

    ax_b.errorbar(
        mjd,
        panels.after_timing_us,
        yerr=sigma,
        fmt=".",
        ms=2,
        elinewidth=0.4,
        alpha=0.7,
    )
    ax_b.set_title("(b) after timing")

    if draw_dm:
        ax_c, ax_d = axs[1, 0], axs[1, 1]
        row_ef = 2
        row_qq = 3
    else:
        ax_c = axs[1, 0]
        ax_d = axs[1, 1]
        ax_d.set_visible(False)
        row_ef = 1 if axs.shape[0] == 3 else 2
        row_qq = 2 if axs.shape[0] == 3 else 3
        if axs.shape[0] >= 4:
            # Caller supplied a 4×2 grid but DM is hidden: reuse row 1 for red only.
            row_ef = 2
            row_qq = 3

    if panels.red_mean_us.size:
        g = np.asarray(panels.grid_mjd, dtype=float)
        ax_c.plot(g, panels.red_mean_us, color="C0")
        ax_c.fill_between(
            g,
            panels.red_mean_us - panels.red_std_us,
            panels.red_mean_us + panels.red_std_us,
            color="C0",
            alpha=0.25,
        )
    ax_c.set_title("(c) red GP")
    ax_c.set_ylabel(r"delay (µs)")

    if draw_dm:
        g = np.asarray(panels.grid_mjd, dtype=float)
        ax_d.plot(g, panels.dm_mean_us, color="C1")
        ax_d.fill_between(
            g,
            panels.dm_mean_us - panels.dm_std_us,
            panels.dm_mean_us + panels.dm_std_us,
            color="C1",
            alpha=0.25,
        )
        ax_d.set_title("(d) DM / chromatic GP")

    ax_e, ax_f = axs[row_ef, 0], axs[row_ef, 1]
    ax_e.errorbar(
        mjd,
        panels.after_all_us,
        yerr=sigma,
        fmt=".",
        ms=2,
        elinewidth=0.4,
        alpha=0.7,
    )
    ax_e.set_title("(e) after all")
    ax_e.set_ylabel(r"residual (µs)")

    ax_f.plot(mjd, panels.z, ".", ms=2, alpha=0.7)
    ax_f.set_title("(f) whitened z")
    ax_f.set_ylabel("z")

    # Q–Q spans both columns of the last row.
    ax_g = axs[row_qq, 0]
    if axs.shape[1] > 1:
        axs[row_qq, 1].set_visible(False)
        # Merge visually: draw Q–Q in left; hide right.
    probplot(np.asarray(panels.z, dtype=float), dist="norm", plot=ax_g)
    ax_g.set_title("(g) Q–Q of z")

    for ax in (ax_a, ax_b, ax_c, ax_e, ax_f):
        ax.set_xlabel("MJD")
    if draw_dm:
        ax_d.set_xlabel("MJD")

    return axs
