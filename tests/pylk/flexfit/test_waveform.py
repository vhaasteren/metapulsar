"""Unit + integration tests for ``pylk.flexfit.waveform``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pylk.flexfit import (
    BasisBlock,
    DiagonalNoise,
    VarianceGroup,
    assemble,
    fastfit,
    fit_white_noise,
    fourier_pair_groups,
    solve_flexible_phi,
)
from pylk.flexfit.waveform import (
    STANDARD_PTA_STAGES,
    GPBand,
    StageSpec,
    WaveformAnalysis,
    aggregate_bands,
    analyze_waveforms,
    frequencies_from_blocks,
    load_waveform_figdata,
    predict_fourier_gp,
    write_waveform_figdata,
)


def _fourier_matrix(t: np.ndarray, n_freq: int) -> np.ndarray:
    tspan = float(t.max() - t.min()) or 1.0
    cols = []
    freqs = []
    for k in range(1, n_freq + 1):
        f = k / tspan
        cols.append(np.sin(2 * np.pi * f * t))
        cols.append(np.cos(2 * np.pi * f * t))
        freqs.extend([f, f])
    return np.column_stack(cols), np.asarray(freqs, dtype=float)


def _fourier_block(
    t: np.ndarray,
    n_freq: int,
    name: str,
    *,
    kind: str = "red",
    sigma_min: float = 1e-12,
    sigma_max: float = 1e-2,
) -> BasisBlock:
    matrix, freqs = _fourier_matrix(t, n_freq)
    groups = fourier_pair_groups(
        matrix,
        prefix=name,
        n_freq=n_freq,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    return BasisBlock(
        name=name,
        matrix=matrix,
        coefficient_names=tuple(f"{name}_c{i}" for i in range(matrix.shape[1])),
        groups=groups,
        kind=kind,  # type: ignore[arg-type]
        metadata={"frequencies": freqs, "n_freq": n_freq},
    )


def _timing_block(t: np.ndarray, name: str = "timing") -> BasisBlock:
    matrix = np.column_stack([np.ones(t.size), (t - t.mean()) / (t.std() or 1.0)])
    groups = tuple(
        VarianceGroup(
            f"{name}_{i}",
            (i,),
            lower=1e40,
            upper=1e40,
            initial=1e40,
            update_from_sweep=10**9,
        )
        for i in range(matrix.shape[1])
    )
    return BasisBlock(
        name=name,
        matrix=matrix,
        coefficient_names=("offset", "slope"),
        groups=groups,
        kind="timing",
    )


def _ecorr_block(n: int, name: str = "ecorr_be1") -> BasisBlock:
    # Two epoch columns covering disjoint halves of the TOAs.
    u = np.zeros((n, 2), dtype=float)
    u[: n // 2, 0] = 1.0
    u[n // 2 :, 1] = 1.0
    group = VarianceGroup(
        name,
        (0, 1),
        lower=1e-18,
        upper=1e-6,
        update_from_sweep=1,
    )
    return BasisBlock(
        name=name,
        matrix=u,
        coefficient_names=("epoch0", "epoch1"),
        groups=(group,),
        kind="ecorr",
        metadata={"backend": "be1", "n_epochs": 2},
    )


def _solve_red(n=200, n_freq=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 365.25 * 86400.0, n))
    block = _fourier_block(t, n_freq, "red")
    phi = np.full(2 * n_freq, 1.0e-12)
    coeffs = np.sqrt(phi) * rng.standard_normal(2 * n_freq)
    variance = np.full(n, (1.0e-7) ** 2)
    y = block.matrix @ coeffs + np.sqrt(variance) * rng.standard_normal(n)
    solve = solve_flexible_phi(
        y, assemble([block]), DiagonalNoise(variance), n_sweeps=4
    )
    return t, y, variance, block, solve


@pytest.mark.unit
def test_predict_fourier_gp_matches_toa_waveform():
    t, y, variance, block, solve = _solve_red()
    span = solve.block_spans["red"]
    mean, std = predict_fourier_gp(
        frequencies=block.metadata["frequencies"],
        coefficient_mean=solve.coefficient_mean[span],
        coefficient_covariance=solve.coefficient_covariance[span, span],
        t_grid=t,
    )
    np.testing.assert_allclose(mean, solve.waveform("red"), rtol=1e-10, atol=1e-18)
    assert np.all(std >= 0.0)


@pytest.mark.unit
def test_stage_resolution_by_kind_and_name():
    rng = np.random.default_rng(1)
    n = 80
    t = np.linspace(0.0, 1.0e8, n)
    timing = _timing_block(t)
    red = _fourier_block(t, 4, "red")
    variance = np.full(n, 1e-14)
    y = 1e-6 + 0.1e-6 * rng.standard_normal(n)
    solve = solve_flexible_phi(
        y, assemble([timing, red]), DiagonalNoise(variance), n_sweeps=2
    )
    kinds = {"timing": "timing", "red": "red"}
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds=kinds,
        block_frequencies=frequencies_from_blocks([red]),
        stages=(
            StageSpec("by_kind", subtract_kinds=("timing",)),
            StageSpec("by_name", subtract_names=("red",)),
        ),
    )
    assert analysis.stage("by_kind").subtracted == ("timing",)
    assert analysis.stage("by_name").subtracted == ("red",)
    with pytest.raises(ValueError, match="not both"):
        StageSpec("bad", subtract_names=("red",), subtract_kinds=("timing",))


@pytest.mark.unit
def test_standard_stages_rms_decreases():
    rng = np.random.default_rng(2)
    n = 300
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    timing = _timing_block(t)
    red = _fourier_block(t, 8, "red")
    timing_wave = timing.matrix @ np.array([3e-6, -1e-6])
    red_wave = red.matrix @ (3e-7 * rng.standard_normal(red.matrix.shape[1]))
    variance = np.full(n, (5e-8) ** 2)
    y = timing_wave + red_wave + np.sqrt(variance) * rng.standard_normal(n)
    solve = solve_flexible_phi(
        y, assemble([timing, red]), DiagonalNoise(variance), n_sweeps=5
    )
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"timing": "timing", "red": "red"},
        block_frequencies=frequencies_from_blocks([red]),
    )
    rms_raw = analysis.stage("raw").rms
    rms_timing = analysis.stage("after_timing").rms
    rms_all = analysis.stage("after_all").rms
    assert rms_raw > rms_timing > rms_all


@pytest.mark.unit
def test_whitened_stage_unit_variance():
    rng = np.random.default_rng(3)
    n = 5000
    t = np.linspace(0.0, 1.0e8, n)
    variance = np.full(n, (1.0e-6) ** 2)
    y = np.sqrt(variance) * rng.standard_normal(n)
    # Empty GP: timing-only improper block still allowed; use pure empty assemble.
    # solve_flexible_phi needs at least one column — use a fixed zero-ish custom
    # block with tiny prior so the whitened stage is essentially y/sigma.
    col = np.ones((n, 1)) * 1e-30
    block = BasisBlock(
        "tiny",
        col,
        ("c0",),
        (
            VarianceGroup(
                "tiny", (0,), 1e-40, 1e-40, initial=1e-40, update_from_sweep=10**9
            ),
        ),
        kind="custom",
    )
    solve = solve_flexible_phi(
        y, assemble([block]), DiagonalNoise(variance), n_sweeps=2
    )
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"tiny": "custom"},
    )
    assert abs(analysis.stage("whitened").rms - 1.0) < 0.05


@pytest.mark.unit
def test_panel_arrays_shapes_and_units():
    t, y, variance, block, solve = _solve_red(n=120, n_freq=5, seed=4)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
        freqs_mhz=None,
    )
    panels = analysis.panel_arrays(n_grid=50)
    assert panels.mjd.shape == (120,)
    assert panels.resid_us.shape == (120,)
    np.testing.assert_allclose(panels.resid_us, y * 1e6)
    assert panels.grid_mjd.shape == (50,)
    assert panels.red_mean_us.shape == (50,)
    assert panels.dm_mean_us.shape == (0,)
    assert panels.dm_std_us.shape == (0,)
    assert np.all(np.isnan(panels.freq_mhz))
    assert panels.label == "quick-look empirical Bayes"


@pytest.mark.unit
def test_gp_band_aggregation_two_red_blocks():
    rng = np.random.default_rng(5)
    n = 160
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    a = _fourier_block(t, 4, "red_a")
    b = _fourier_block(t, 4, "red_b")
    variance = np.full(n, 1e-14)
    y = (
        a.matrix @ (1e-7 * rng.standard_normal(a.matrix.shape[1]))
        + b.matrix @ (1e-7 * rng.standard_normal(b.matrix.shape[1]))
        + np.sqrt(variance) * rng.standard_normal(n)
    )
    solve = solve_flexible_phi(y, assemble([a, b]), DiagonalNoise(variance), n_sweeps=3)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red_a": "red", "red_b": "red"},
        block_frequencies=frequencies_from_blocks([a, b]),
    )
    grid = np.linspace(t.min(), t.max(), 40)
    ba = analysis.predict_gp("red_a", grid)
    bb = analysis.predict_gp("red_b", grid)
    agg = aggregate_bands([ba, bb], name="red", kind="red")
    np.testing.assert_allclose(agg.mean, ba.mean + bb.mean)
    np.testing.assert_allclose(agg.std, np.sqrt(ba.std**2 + bb.std**2))


@pytest.mark.unit
def test_write_roundtrip_figdata(tmp_path: Path):
    t, y, variance, block, solve = _solve_red(n=90, n_freq=4, seed=6)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
        freqs_mhz=np.full(t.size, 1400.0),
    )
    path = tmp_path / "psr_waveform.feather"
    write_waveform_figdata(analysis, path, pulsar_name="J0000+0000", n_grid=33)
    loaded = load_waveform_figdata(path)
    panels = analysis.panel_arrays(n_grid=33)
    for name in (
        "mjd",
        "freq_mhz",
        "resid_us",
        "after_timing_us",
        "after_all_us",
        "z",
        "sigma_us",
        "grid_mjd",
        "red_mean_us",
        "red_std_us",
        "dm_mean_us",
        "dm_std_us",
    ):
        np.testing.assert_allclose(getattr(loaded, name), getattr(panels, name))
    assert loaded.label == panels.label
    assert dict(loaded.stage_rms_us) == dict(panels.stage_rms_us)


@pytest.mark.unit
def test_fastfit_waveform_analysis_method():
    rng = np.random.default_rng(7)
    n = 100
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    red = _fourier_block(t, 5, "red")
    variance = np.full(n, 1e-14)
    y = red.matrix @ (2e-7 * rng.standard_normal(red.matrix.shape[1]))
    y = y + np.sqrt(variance) * rng.standard_normal(n)
    fit = fastfit(
        noise=DiagonalNoise(variance),
        blocks=[red],
        residuals=y,
        n_sweeps=3,
    )
    analysis = fit.waveform_analysis(
        variance=variance,
        toas=t,
        toa_mjd=t / 86400.0,
        block_frequencies=frequencies_from_blocks([red]),
    )
    assert isinstance(analysis, WaveformAnalysis)
    assert "red" in analysis
    np.testing.assert_allclose(analysis["red"], fit.waveform("red"))


@pytest.mark.unit
def test_discovery_reconstruct_waveforms_returns_analysis():
    ds = pytest.importorskip("discovery")
    from discovery.pulsar import Pulsar

    from pylk.flexfit.adapters import discovery as dx

    rng = np.random.default_rng(8)
    n = 400
    psr = Pulsar()
    psr.name = "J0000+0000"
    span = 10.0 * 365.25 * 86400.0
    psr.toas = np.sort(rng.uniform(0.0, span, n))
    psr.stoas = psr.toas.copy()
    psr.toaerrs = np.full(n, 1.0e-7)
    psr.freqs = np.full(n, 1400.0)
    psr.backend_flags = np.array(["be1"] * n)
    tt = psr.toas - psr.toas.mean()
    psr.Mmat = np.column_stack([np.ones(n), tt / tt.std()])
    psr.fitpars = ["Offset", "F0"]
    psr.setpars = []
    psr.mintoa = float(psr.toas.min())
    psr.maxtoa = float(psr.toas.max())
    psr.flags = {"be": psr.backend_flags}
    psr.pos = np.array([1.0, 0.0, 0.0])
    psr.phi, psr.theta = 0.0, np.pi / 2
    psr.dm = 10.0

    red_gp = ds.makegp_fourier(psr, ds.powerlaw, components=12, name="red")
    A, g = -13.0, 3.0
    p = {f"{psr.name}_red_log10_A": A, f"{psr.name}_red_gamma": g}
    variance = psr.toaerrs**2
    psr.residuals = np.asarray(red_gp.F) @ (
        np.sqrt(red_gp.Phi.getN(p)) * rng.standard_normal(24)
    ) + np.sqrt(variance) * rng.standard_normal(n)

    analysis = dx.reconstruct_waveforms(
        psr,
        variance=variance,
        design_matrix=psr.Mmat,
        spectra={"red": {"kind": "red", "components": 12, "log10_A": A, "gamma": g}},
    )
    assert isinstance(analysis, WaveformAnalysis)
    assert "red" in analysis
    assert set(analysis) >= {"red", "timingmodel"}
    band = analysis.predict_gp("red", psr.toas)
    assert isinstance(band, GPBand)
    np.testing.assert_allclose(band.mean, analysis["red"], rtol=1e-8, atol=1e-14)


@pytest.mark.unit
def test_plot_waveform_panels_smoke():
    plt = pytest.importorskip("matplotlib.pyplot")
    from pylk.flexfit.waveform_plot import plot_waveform_panels

    t, y, variance, block, solve = _solve_red(n=60, n_freq=3, seed=9)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
    )
    panels = analysis.panel_arrays(n_grid=20)
    fig, axs = plt.subplots(4, 2)
    out = plot_waveform_panels(panels, axs=axs)
    assert out.shape == (4, 2)
    plt.close(fig)


@pytest.mark.unit
def test_plot_waveform_panels_auto_layout_no_dm():
    plt = pytest.importorskip("matplotlib.pyplot")
    from pylk.flexfit.waveform_plot import plot_waveform_panels

    t, y, variance, block, solve = _solve_red(n=60, n_freq=3, seed=9)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
    )
    panels = analysis.panel_arrays(n_grid=20)  # red only -> no DM band
    axs = plot_waveform_panels(panels)
    assert axs.shape == (3, 2)
    assert all(ax.get_visible() for ax in axs.ravel())
    assert {ax.get_title() for ax in axs.ravel()} == {
        "(a) raw",
        "(b) after timing",
        "(c) red GP",
        "(e) after all",
        "(f) whitened z",
        "(g) Q–Q of z",
    }
    plt.close(axs[0, 0].figure)


@pytest.mark.unit
def test_plot_waveform_panels_auto_layout_with_dm():
    plt = pytest.importorskip("matplotlib.pyplot")
    from pylk.flexfit.waveform_plot import plot_waveform_panels

    rng = np.random.default_rng(18)
    n = 80
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    red = _fourier_block(t, 3, "red")
    dm = _fourier_block(t, 3, "dm", kind="dm")
    variance = np.full(n, 1e-14)
    y = (
        red.matrix @ (1e-7 * rng.standard_normal(red.matrix.shape[1]))
        + dm.matrix @ (1e-7 * rng.standard_normal(dm.matrix.shape[1]))
        + np.sqrt(variance) * rng.standard_normal(n)
    )
    solve = solve_flexible_phi(
        y, assemble([red, dm]), DiagonalNoise(variance), n_sweeps=2
    )
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red", "dm": "dm"},
        block_frequencies=frequencies_from_blocks([red, dm]),
    )
    panels = analysis.panel_arrays(n_grid=20)
    axs = plot_waveform_panels(panels)
    assert axs.shape == (4, 2)
    assert axs[1, 1].get_title() == "(d) DM / chromatic GP"
    assert axs[1, 1].get_visible()
    assert not axs[3, 1].get_visible()  # Q-Q occupies the left slot only
    with pytest.raises(ValueError, match="at least 4x2"):
        fig, small = plt.subplots(3, 2)
        plot_waveform_panels(panels, axs=small)
    plt.close("all")


@pytest.mark.unit
def test_standard_whitened_stage_differs_from_whitened_residuals():
    rng = np.random.default_rng(10)
    n = 150
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    timing = _timing_block(t)
    red = _fourier_block(t, 5, "red")
    custom = BasisBlock(
        "custom",
        np.ones((n, 1)),
        ("c0",),
        (VarianceGroup("custom", (0,), 1e-18, 1e-6),),
        kind="custom",
    )
    variance = np.full(n, 1e-14)
    y = (
        timing.matrix @ np.array([1e-6, 2e-7])
        + red.matrix @ (1e-7 * rng.standard_normal(red.matrix.shape[1]))
        + 1e-7 * custom.matrix[:, 0]
        + np.sqrt(variance) * rng.standard_normal(n)
    )
    fit = fastfit(
        noise=DiagonalNoise(variance),
        blocks=[timing, red, custom],
        residuals=y,
        n_sweeps=3,
    )
    analysis = fit.waveform_analysis(
        variance=variance,
        toas=t,
        toa_mjd=t / 86400.0,
        block_frequencies=frequencies_from_blocks([red]),
    )
    assert not np.allclose(
        analysis.stage("whitened").residuals, fit.whitened_residuals()
    )
    np.testing.assert_allclose(
        analysis.residuals_after_excluding_kinds("timing"),
        fit.whitened_residuals(),
    )
    assert set(analysis.block_names_excluding("timing")) == {"red", "custom"}


@pytest.mark.unit
def test_whitenoise_residuals_and_waveform_analysis():
    rng = np.random.default_rng(11)
    n = 400
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    red = _fourier_block(t, 6, "red")
    toaerrs = np.full(n, 1.0e-7)
    backends = np.array(["A", "B"])[np.arange(n) % 2]
    y = red.matrix @ (1e-7 * rng.standard_normal(red.matrix.shape[1]))
    y = y + toaerrs * rng.standard_normal(n)
    result = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[red],
        residuals=y,
        max_iterations=4,
        n_sweeps=2,
    )
    np.testing.assert_allclose(result.residuals, y)
    analysis = result.waveform_analysis(
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([red]),
    )
    np.testing.assert_allclose(analysis.y, result.residuals)
    np.testing.assert_allclose(analysis.variance, result.variance)

    pure = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[],
        residuals=toaerrs * rng.standard_normal(n),
        max_iterations=3,
        n_sweeps=2,
    )
    assert pure.solve is None
    with pytest.raises(RuntimeError, match="pure-WN"):
        pure.waveform_analysis(
            toas=t,
            toa_mjd=t / 86400.0,
            block_kinds={},
        )


@pytest.mark.unit
def test_stage_missing_name_and_empty_kind():
    t, y, variance, block, solve = _solve_red(n=50, n_freq=3, seed=12)
    with pytest.raises(KeyError, match=r"stage 'bad'.*unknown block 'nope'"):
        analyze_waveforms(
            y,
            variance,
            solve,
            toas=t,
            toa_mjd=t / 86400.0,
            block_kinds={"red": "red"},
            stages=(StageSpec("bad", subtract_names=("nope",)),),
        )
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
        stages=(StageSpec("no_dm", subtract_kinds=("dm",)),),
    )
    np.testing.assert_allclose(analysis.stage("no_dm").residuals, y)


@pytest.mark.unit
def test_custom_t_grid_interpolates_grid_mjd():
    t, y, variance, block, solve = _solve_red(n=70, n_freq=3, seed=13)
    # Shuffle TOAs so sorting for interp matters.
    order = np.arange(t.size)
    rng = np.random.default_rng(13)
    rng.shuffle(order)
    t = t[order]
    y = y[order]
    variance = variance[order]
    # Re-solve on shuffled rows with matching block matrix.
    block = _fourier_block(t, 3, "red")
    solve = solve_flexible_phi(
        y, assemble([block]), DiagonalNoise(variance), n_sweeps=2
    )
    toa_mjd = t / 86400.0
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=toa_mjd,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
    )
    t_grid = np.linspace(t.min(), t.max(), 25)
    panels = analysis.panel_arrays(t_grid=t_grid)
    order_s = np.argsort(t)
    expected = np.interp(t_grid, t[order_s], toa_mjd[order_s])
    np.testing.assert_allclose(panels.grid_mjd, expected)
    assert panels.red_mean_us.shape == (25,)
    assert panels.grid_mjd.shape == (25,)


@pytest.mark.unit
def test_ecorr_in_stages_but_not_in_gp_bands():
    rng = np.random.default_rng(14)
    n = 100
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    ecorr = _ecorr_block(n)
    variance = np.full(n, 1e-14)
    y = 5e-7 * ecorr.matrix @ np.array([1.0, -1.0])
    y = y + np.sqrt(variance) * rng.standard_normal(n)
    solve = solve_flexible_phi(
        y, assemble([ecorr]), DiagonalNoise(variance), n_sweeps=3
    )
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={ecorr.name: "ecorr"},
    )
    after = analysis.residuals_after_kinds("ecorr")
    assert np.sqrt(np.mean(after**2)) < np.sqrt(np.mean(y**2))
    assert analysis.gp_bands() == ()
    with pytest.raises(KeyError, match="Fourier frequencies"):
        analysis.predict_gp(ecorr.name, t)


@pytest.mark.unit
def test_chromatic_band_populates_dm_panel():
    rng = np.random.default_rng(15)
    n = 120
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    chrom = _fourier_block(t, 5, "chrom", kind="chromatic")
    variance = np.full(n, 1e-14)
    y = chrom.matrix @ (2e-7 * rng.standard_normal(chrom.matrix.shape[1]))
    y = y + np.sqrt(variance) * rng.standard_normal(n)
    solve = solve_flexible_phi(
        y, assemble([chrom]), DiagonalNoise(variance), n_sweeps=3
    )
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"chrom": "chromatic"},
        block_frequencies=frequencies_from_blocks([chrom]),
    )
    panels = analysis.panel_arrays(n_grid=40)
    assert panels.dm_mean_us.shape == (40,)
    assert panels.dm_std_us.shape == (40,)
    assert panels.red_mean_us.shape == (0,)
    assert np.any(np.abs(panels.dm_mean_us) > 0.0)


@pytest.mark.unit
def test_aggregate_bands_ignores_cross_covariance():
    rng = np.random.default_rng(16)
    n = 180
    t = np.sort(rng.uniform(0.0, 1.0e8, n))
    # Shared design directions → non-zero cross-block posterior covariance.
    base, freqs = _fourier_matrix(t, 4)
    a = BasisBlock(
        "red_a",
        base,
        tuple(f"a{i}" for i in range(base.shape[1])),
        fourier_pair_groups(
            base, prefix="red_a", n_freq=4, sigma_min=1e-12, sigma_max=1e-2
        ),
        kind="red",
        metadata={"frequencies": freqs},
    )
    b = BasisBlock(
        "red_b",
        base + 0.1 * rng.standard_normal(base.shape),
        tuple(f"b{i}" for i in range(base.shape[1])),
        fourier_pair_groups(
            base, prefix="red_b", n_freq=4, sigma_min=1e-12, sigma_max=1e-2
        ),
        kind="red",
        metadata={"frequencies": freqs},
    )
    variance = np.full(n, 1e-14)
    y = (a.matrix + b.matrix) @ (1e-7 * rng.standard_normal(a.matrix.shape[1]))
    y = y + np.sqrt(variance) * rng.standard_normal(n)
    solve = solve_flexible_phi(y, assemble([a, b]), DiagonalNoise(variance), n_sweeps=3)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red_a": "red", "red_b": "red"},
        block_frequencies=frequencies_from_blocks([a, b]),
    )
    grid = np.linspace(t.min(), t.max(), 30)
    ba = analysis.predict_gp("red_a", grid)
    bb = analysis.predict_gp("red_b", grid)
    agg = aggregate_bands([ba, bb], name="red", kind="red")
    indep = np.sqrt(ba.std**2 + bb.std**2)
    np.testing.assert_allclose(agg.std, indep)

    # True joint std from the full coefficient covariance on the concatenated basis.
    fa = np.asarray(a.metadata["frequencies"], dtype=float)
    fb = np.asarray(b.metadata["frequencies"], dtype=float)
    phase_a = 2.0 * np.pi * np.outer(grid, fa)
    phase_b = 2.0 * np.pi * np.outer(grid, fb)
    basis_a = np.empty_like(phase_a)
    basis_b = np.empty_like(phase_b)
    basis_a[:, 0::2] = np.sin(phase_a[:, 0::2])
    basis_a[:, 1::2] = np.cos(phase_a[:, 1::2])
    basis_b[:, 0::2] = np.sin(phase_b[:, 0::2])
    basis_b[:, 1::2] = np.cos(phase_b[:, 1::2])
    basis = np.concatenate([basis_a, basis_b], axis=1)
    sa = solve.block_spans["red_a"]
    sb = solve.block_spans["red_b"]
    # Dense joint block covering both spans (contiguous after assemble).
    cov = np.asarray(
        solve.coefficient_covariance[sa.start : sb.stop, sa.start : sb.stop]
    )
    var_joint = np.einsum("gi,ij,gj->g", basis, cov, basis)
    std_joint = np.sqrt(np.clip(var_joint, 0.0, None))
    assert not np.allclose(agg.std, std_joint, rtol=1e-3, atol=0.0)


@pytest.mark.unit
def test_write_figdata_json_sidecar(tmp_path: Path):
    t, y, variance, block, solve = _solve_red(n=40, n_freq=3, seed=17)
    analysis = analyze_waveforms(
        y,
        variance,
        solve,
        toas=t,
        toa_mjd=t / 86400.0,
        block_kinds={"red": "red"},
        block_frequencies=frequencies_from_blocks([block]),
    )
    path = tmp_path / "J0000_waveform.feather"
    write_waveform_figdata(
        analysis, path, pulsar_name="J0000+0000", n_grid=10, json_sidecar=True
    )
    sidecar = path.with_name(f"{path.stem}_waveform.json")
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["format"] == "pylk.flexfit.waveform_figdata/v1"
    assert payload["pulsar"] == "J0000+0000"
    assert payload["label"] == "quick-look empirical Bayes"
    assert "summary" in payload and "stage_rms_us" in payload
    assert "resid_us" not in payload
    assert "grid" not in payload

    path2 = tmp_path / "noside.feather"
    write_waveform_figdata(analysis, path2, pulsar_name="J0000+0000", n_grid=10)
    assert not path2.with_name(f"{path2.stem}_waveform.json").exists()


@pytest.mark.unit
def test_standard_pta_stages_constant():
    assert tuple(s.name for s in STANDARD_PTA_STAGES) == (
        "raw",
        "after_timing",
        "after_red",
        "after_all",
        "whitened",
    )
