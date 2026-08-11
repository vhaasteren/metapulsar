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

    With the DM panel drawn (``show_dm=True`` and non-empty DM arrays), 4×2:

      [a raw]        [b after timing]
      [c red GP]     [d dm GP]
      [e after all]  [f whitened z]
      [g Q-Q of z]   [hidden]

    Without it (``show_dm=False`` or empty DM arrays), 3×2:

      [a raw]        [b after timing]
      [c red GP]     [e after all]
      [f whitened z] [g Q-Q of z]

    A caller-supplied 4×2 grid is accepted in both cases (the unused Axes are
    hidden); a 3×2 grid only when the DM panel is not drawn.

    Returns the Axes array. Does not call ``plt.show()``.
    Requires matplotlib; use ``pytest.importorskip("matplotlib")`` in tests.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import probplot

    draw_dm = bool(show_dm and panels.dm_mean_us.size > 0)
    nrows = 4 if draw_dm else 3

    if axs is None:
        fig, axs = plt.subplots(nrows, 2, figsize=(8.0, 2.4 * nrows), squeeze=False)
        if title:
            fig.suptitle(title)
    else:
        axs = np.atleast_2d(axs)
        if axs.shape[0] < nrows or axs.shape[1] < 2:
            raise ValueError(
                f"axs must be at least {nrows}x2 for this panel set; got {axs.shape}"
            )

    ax_a, ax_b = axs[0, 0], axs[0, 1]
    ax_d = None
    hidden: list = []
    if axs.shape[0] >= 4:
        ax_c, ax_e, ax_f = axs[1, 0], axs[2, 0], axs[2, 1]
        ax_g, unused = axs[3, 0], axs[3, 1]
        hidden.append(unused)
        if draw_dm:
            ax_d = axs[1, 1]
        else:
            hidden.append(axs[1, 1])
    else:
        ax_c, ax_e = axs[1, 0], axs[1, 1]
        ax_f, ax_g = axs[2, 0], axs[2, 1]
    for ax in hidden:
        ax.set_visible(False)

    mjd = np.asarray(panels.mjd, dtype=float)
    sigma = np.asarray(panels.sigma_us, dtype=float)

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

    if ax_d is not None:
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

    probplot(np.asarray(panels.z, dtype=float), dist="norm", plot=ax_g)
    ax_g.set_title("(g) Q–Q of z")

    labeled = [ax_a, ax_b, ax_c, ax_e, ax_f]
    if ax_d is not None:
        labeled.append(ax_d)
    for ax in labeled:
        ax.set_xlabel("MJD")

    return axs
