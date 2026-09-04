"""Integration tests: flexfit vs Discovery's own Woodbury conditional.

These build a synthetic ``discovery.Pulsar`` from arrays and prove that the
flexfit conditional solve reproduces Discovery's GP coefficient posterior mean
for the *same* Fourier basis and ``powerlaw`` ``Phi`` — i.e. the adapter uses
Discovery conventions correctly and the numerical core matches the production
Woodbury algebra. Skips cleanly when Discovery is unavailable (pulsar runs).

Run in the devcontainer: ``pytest tests/pylk/flexfit/test_integration_discovery.py``.
"""

from __future__ import annotations


import numpy as np
import pytest

ds = pytest.importorskip("discovery")

from pylk.flexfit import DiagonalNoise, conditional_moments, fastfit  # noqa: E402
from pylk.flexfit.adapters import discovery as dx  # noqa: E402


def _synthetic_pulsar(n=300, seed=0, span_years=12.0):
    from discovery.pulsar import Pulsar

    rng = np.random.default_rng(seed)
    psr = Pulsar()
    psr.name = "J0000+0000"
    span = span_years * 365.25 * 86400.0
    psr.toas = np.sort(rng.uniform(0.0, span, n))
    psr.stoas = psr.toas.copy()
    psr.toaerrs = np.full(n, 1.0e-7)
    psr.freqs = rng.choice([1400.0, 820.0], size=n).astype(float)
    psr.backend_flags = np.array(["be1"] * n)
    t = psr.toas - psr.toas.mean()
    psr.Mmat = np.column_stack([np.ones(n), t / t.std()])
    psr.fitpars = ["Offset", "F0"]
    psr.setpars = []
    psr.residuals = np.zeros(n)  # set by caller
    psr.mintoa = float(psr.toas.min())
    psr.maxtoa = float(psr.toas.max())
    psr.flags = {"be": psr.backend_flags}
    psr.pos = np.array([1.0, 0.0, 0.0])
    psr.phi, psr.theta = 0.0, np.pi / 2
    psr.dm = 10.0
    return psr, rng


def test_red_block_matches_discovery_basis():
    psr, _ = _synthetic_pulsar()
    gp = ds.makegp_fourier(psr, ds.powerlaw, components=12, name="red")
    block = dx.red_noise_block(psr, components=12, name="red")
    np.testing.assert_allclose(block.matrix, np.asarray(gp.F, dtype=float))
    assert block.matrix.shape == (len(psr.residuals), 24)
    # metadata frequencies match the powerlaw's expected length (2C).
    assert block.metadata["frequencies"].shape == (24,)


def test_chromatic_block_matches_discovery_alpha4_basis():
    psr, _ = _synthetic_pulsar()
    basis = ds.make_dmfourierbasis(alpha=4.0)
    freqs, df, fmat = basis(psr, 12, fref=1400.0)
    block = dx.chromatic_noise_block(
        psr, components=12, alpha=4.0, name="chrom", fref=1400.0
    )
    np.testing.assert_allclose(block.matrix, np.asarray(fmat, dtype=float))
    assert block.kind == "chromatic"
    assert block.metadata["alpha"] == 4.0
    assert block.metadata["frequencies"].shape == freqs.shape


def test_conditional_mean_matches_woodbury_covariance_form():
    # Cross-check the flexfit precision-form solve against the algebraically
    # distinct Woodbury covariance form mu = Phi F^T (N + F Phi F^T)^-1 y, using
    # Discovery's own Fourier basis F and powerlaw Phi.
    psr, rng = _synthetic_pulsar()
    components = 15
    gp = ds.makegp_fourier(psr, ds.powerlaw, components=components, name="red")
    F = np.asarray(gp.F, dtype=float)

    params = {"J0000+0000_red_log10_A": -13.5, "J0000+0000_red_gamma": 3.5}
    phi = np.asarray(gp.Phi.getN(params), dtype=float)  # length 2C

    coeffs = np.sqrt(phi) * rng.standard_normal(F.shape[1])
    variance = psr.toaerrs**2
    psr.residuals = F @ coeffs + np.sqrt(variance) * rng.standard_normal(len(variance))

    sigma = np.diag(variance) + F @ np.diag(phi) @ F.T
    mu_woodbury = phi * (F.T @ np.linalg.solve(sigma, psr.residuals))

    mean, _, _ = conditional_moments(psr.residuals, F, phi, DiagonalNoise(variance))
    np.testing.assert_allclose(mean, mu_woodbury, rtol=1e-6, atol=1e-9)


def test_white_noise_adapter_from_variance_matches_diagonal():
    psr, _ = _synthetic_pulsar(n=50)
    variance = (2.0 * psr.toaerrs) ** 2
    noise = dx.white_noise_from_variance(variance)
    v = np.arange(50, dtype=float)
    np.testing.assert_allclose(noise.solve(v), v / variance)


def test_fastfit_recovers_red_and_dm_waveforms():
    psr, rng = _synthetic_pulsar(n=400, seed=3)
    red_gp = ds.makegp_fourier(psr, ds.powerlaw, components=20, name="red")
    dm_gp = ds.makegp_fourier(
        psr, ds.powerlaw, components=20, fourierbasis=ds.dmfourierbasis, name="dm"
    )
    Fr, Fd = np.asarray(red_gp.F), np.asarray(dm_gp.F)
    p = {
        "J0000+0000_red_log10_A": -13.0,
        "J0000+0000_red_gamma": 3.0,
        "J0000+0000_dm_log10_A": -13.5,
        "J0000+0000_dm_gamma": 2.5,
    }
    red_wave = Fr @ (np.sqrt(red_gp.Phi.getN(p)) * rng.standard_normal(Fr.shape[1]))
    dm_wave = Fd @ (np.sqrt(dm_gp.Phi.getN(p)) * rng.standard_normal(Fd.shape[1]))
    variance = psr.toaerrs**2
    psr.residuals = (
        red_wave + dm_wave + np.sqrt(variance) * rng.standard_normal(len(variance))
    )

    blocks = [
        dx.red_noise_block(psr, components=20, name="red"),
        dx.dm_noise_block(psr, components=20, name="dm"),
    ]
    noise = dx.white_noise_from_variance(variance)
    fit = fastfit(noise=noise, blocks=blocks, residuals=psr.residuals, n_sweeps=5)

    assert np.corrcoef(fit.waveform("red"), red_wave)[0, 1] > 0.85
    assert np.corrcoef(fit.waveform("dm"), dm_wave)[0, 1] > 0.85
    # Whitened residuals are much smaller than the raw residuals.
    raw_rms = np.sqrt(np.mean(psr.residuals**2))
    white_rms = np.sqrt(np.mean(fit.whitened_residuals() ** 2))
    assert white_rms < 0.5 * raw_rms


def test_dm_gp_captures_variations_with_dm_quadratic_marginalized():
    """Mirror the consistent-MetaPulsar regime: no DMX, only a deterministic DM
    quadratic {DM, DM1, DM2} in the (marginalized) timing model, with stochastic
    DM left to a DM GP. The DM GP must recover the injected DM and collapse the
    chromatic (nu^-2) scatter, while red noise stays separable."""
    from pylk.flexfit import BasisBlock, VarianceGroup, assemble, solve_flexible_phi

    psr, rng = _synthetic_pulsar(n=800, seed=9)
    chroma = (1400.0 / psr.freqs) ** 2  # (fref / nu)^2 chromatic weight per TOA
    t = (psr.toas - psr.toas.mean()) / (psr.toas.max() - psr.toas.min())

    # Timing model: spin (offset, F0, F1) + deterministic DM quadratic
    # (DM, DM1, DM2 columns are chromatic: scaled by (fref/nu)^2).
    mmat = np.column_stack(
        [np.ones(len(t)), t, t**2, chroma, chroma * t, chroma * t**2]
    )
    tgroups = tuple(
        VarianceGroup(
            f"tm_{i}", (i,), 1e40, 1e40, initial=1e40, update_from_sweep=10**9
        )
        for i in range(mmat.shape[1])
    )
    tblock = BasisBlock(
        "timing_marg",
        mmat,
        tuple(f"m{i}" for i in range(mmat.shape[1])),
        tgroups,
        kind="timing",
    )

    red_gp = ds.makegp_fourier(psr, ds.powerlaw, components=20, name="red")
    dm_gp = ds.makegp_fourier(
        psr, ds.powerlaw, components=20, fourierbasis=ds.dmfourierbasis, name="dm"
    )
    Fr, Fd = np.asarray(red_gp.F), np.asarray(dm_gp.F)
    p = {
        "J0000+0000_red_log10_A": -12.7,
        "J0000+0000_red_gamma": 3.0,
        "J0000+0000_dm_log10_A": -12.6,
        "J0000+0000_dm_gamma": 2.2,
    }
    red_wave = Fr @ (np.sqrt(red_gp.Phi.getN(p)) * rng.standard_normal(Fr.shape[1]))
    dm_wave = Fd @ (np.sqrt(dm_gp.Phi.getN(p)) * rng.standard_normal(Fd.shape[1]))
    variance = psr.toaerrs**2
    psr.residuals = (
        red_wave + dm_wave + np.sqrt(variance) * rng.standard_normal(len(variance))
    )

    red = dx.red_noise_block(psr, components=20, name="red")
    dm = dx.dm_noise_block(psr, components=20, name="dm")
    noise = dx.white_noise_from_variance(variance)
    res = solve_flexible_phi(
        psr.residuals, assemble([tblock, red, dm]), noise, n_sweeps=4
    )

    # DM GP recovers the injected DM and red stays separable (the two share
    # low-frequency covariance, so red is recovered more loosely than DM).
    assert np.corrcoef(res.waveform("dm"), dm_wave)[0, 1] > 0.8
    assert np.corrcoef(res.waveform("red"), red_wave)[0, 1] > 0.65

    # The chromatic (nu^-2) signature collapses after DM subtraction: the residual
    # correlation with the per-TOA chromatic weight drops sharply.
    raw = np.asarray(psr.residuals)
    dm_subtracted = raw - res.waveform("dm")
    corr_raw = abs(np.corrcoef(raw, chroma)[0, 1])
    corr_sub = abs(np.corrcoef(dm_subtracted, chroma)[0, 1])
    assert corr_sub < corr_raw


def test_reconstruct_waveforms_whitens_timing_plus_rn_plus_dm():
    """The corrected procedure: inject *large* timing-parameter deviations + RN +
    DM, reconstruct all three JOINTLY, and subtract together. Subtracting only
    RN+DM (the earlier bug) leaves the timing deviation; subtracting the timing
    waveform too collapses to the white-noise floor, and the normalized whitened
    residuals follow N(0, 1)."""
    psr, rng = _synthetic_pulsar(n=1200, seed=17)
    chroma = (1400.0 / psr.freqs) ** 2
    t = (psr.toas - psr.toas.mean()) / (psr.toas.max() - psr.toas.min())
    mmat = np.column_stack([np.ones(len(t)), t, t**2, chroma, chroma * t])

    red_gp = ds.makegp_fourier(psr, ds.powerlaw, components=20, name="red")
    dm_gp = ds.makegp_fourier(
        psr, ds.powerlaw, components=20, fourierbasis=ds.dmfourierbasis, name="dm"
    )
    A_red, g_red, A_dm, g_dm = -13.0, 3.0, -12.8, 2.0
    p = {
        "J0000+0000_red_log10_A": A_red,
        "J0000+0000_red_gamma": g_red,
        "J0000+0000_dm_log10_A": A_dm,
        "J0000+0000_dm_gamma": g_dm,
    }
    timing_dev = mmat @ np.array([3e-6, 2e-5, -1.5e-5, 8e-7, 4e-6])
    red_wave = np.asarray(red_gp.F) @ (
        np.sqrt(red_gp.Phi.getN(p)) * rng.standard_normal(40)
    )
    dm_wave = np.asarray(dm_gp.F) @ (
        np.sqrt(dm_gp.Phi.getN(p)) * rng.standard_normal(40)
    )
    variance = psr.toaerrs**2
    white = np.sqrt(variance) * rng.standard_normal(len(variance))
    psr.residuals = timing_dev + red_wave + dm_wave + white

    spectra = {
        "red": {"kind": "red", "components": 20, "log10_A": A_red, "gamma": g_red},
        "dm": {"kind": "dm", "components": 20, "log10_A": A_dm, "gamma": g_dm},
    }
    waves = dx.reconstruct_waveforms(
        psr, variance=variance, design_matrix=mmat, spectra=spectra
    )
    assert set(waves) == {"timingmodel", "red", "dm"}
    assert np.corrcoef(waves["dm"], dm_wave)[0, 1] > 0.8

    whitened = psr.residuals - waves["timingmodel"] - waves["red"] - waves["dm"]
    raw_rms = np.sqrt(np.mean(psr.residuals**2))
    white_rms = np.sqrt(np.mean(whitened**2))
    floor = np.sqrt(np.mean(white**2))
    only_gp_rms = np.sqrt(np.mean((psr.residuals - waves["red"] - waves["dm"]) ** 2))
    assert white_rms < 0.3 * raw_rms
    assert white_rms < 0.5 * only_gp_rms  # subtracting timing matters
    assert white_rms < 2.0 * floor

    z = whitened / np.sqrt(variance)
    assert abs(np.std(z) - 1.0) < 0.2


@pytest.mark.slow
def test_map_powerlaw_hypers_recovers_injected_and_agrees_with_flexfit():
    """The Discovery-logL MAP of the power-law hypers (second Phi source) should
    recover the injected DM/red spectrum, independently of flexfit's EB MLE."""
    psr, rng = _synthetic_pulsar(n=900, seed=13)
    chroma = (1400.0 / psr.freqs) ** 2
    t = (psr.toas - psr.toas.mean()) / (psr.toas.max() - psr.toas.min())
    mmat = np.column_stack(
        [np.ones(len(t)), t, t**2, chroma, chroma * t, chroma * t**2]
    )

    red_gp = ds.makegp_fourier(psr, ds.powerlaw, components=20, name="red")
    dm_gp = ds.makegp_fourier(
        psr, ds.powerlaw, components=20, fourierbasis=ds.dmfourierbasis, name="dm"
    )
    p = {
        "J0000+0000_red_log10_A": -13.0,
        "J0000+0000_red_gamma": 3.0,
        "J0000+0000_dm_log10_A": -12.7,
        "J0000+0000_dm_gamma": 2.0,
    }
    rw = np.asarray(red_gp.F) @ (np.sqrt(red_gp.Phi.getN(p)) * rng.standard_normal(40))
    dw = np.asarray(dm_gp.F) @ (np.sqrt(dm_gp.Phi.getN(p)) * rng.standard_normal(40))
    variance = psr.toaerrs**2
    psr.residuals = rw + dw + np.sqrt(variance) * rng.standard_normal(len(variance))

    hypers = dx.map_powerlaw_hypers(
        psr,
        variance=variance,
        timing=mmat,
        specs=[
            {"name": "red", "kind": "red", "components": 20},
            {"name": "dm", "kind": "dm", "components": 20},
        ],
    )
    assert hypers["dm"]["log10_A"] == pytest.approx(-12.7, abs=0.6)
    assert hypers["red"]["log10_A"] == pytest.approx(-13.0, abs=0.6)

    # The MAP hypers feed the same reconstruction and remove real DM power.
    waves = dx.reconstruct_waveforms(
        psr,
        variance=variance,
        design_matrix=mmat,
        spectra={
            "red": {"kind": "red", "components": 20, **hypers["red"]},
            "dm": {"kind": "dm", "components": 20, **hypers["dm"]},
        },
    )
    assert np.corrcoef(waves["dm"], dw)[0, 1] > 0.8


def test_predict_gp_on_grid():
    """GP grid prediction: at the TOA times it reproduces the red waveform; on a
    regular grid it returns a finite mean and a non-negative std band; timing has
    no Fourier basis and is rejected."""
    psr, rng = _synthetic_pulsar(n=700, seed=21)
    t = (psr.toas - psr.toas.mean()) / (psr.toas.max() - psr.toas.min())
    mmat = np.column_stack([np.ones(len(t)), t])
    red_gp = ds.makegp_fourier(psr, ds.powerlaw, components=20, name="red")
    A, g = -12.9, 3.0
    p = {"J0000+0000_red_log10_A": A, "J0000+0000_red_gamma": g}
    variance = psr.toaerrs**2
    psr.residuals = np.asarray(red_gp.F) @ (
        np.sqrt(red_gp.Phi.getN(p)) * rng.standard_normal(40)
    ) + np.sqrt(variance) * rng.standard_normal(len(variance))

    recon = dx.reconstruct_waveforms(
        psr,
        variance=variance,
        design_matrix=mmat,
        spectra={"red": {"kind": "red", "components": 20, "log10_A": A, "gamma": g}},
    )
    # Grid prediction evaluated at the TOAs equals the TOA-domain waveform.
    band_toa = recon.predict_gp("red", psr.toas)
    np.testing.assert_allclose(band_toa.mean, recon["red"], rtol=1e-8, atol=1e-14)
    assert np.all(band_toa.std >= 0.0)

    # A finer regular grid works (interpolation/extrapolation) with a valid band.
    grid = np.linspace(psr.toas.min(), psr.toas.max(), 300)
    band = recon.predict_gp("red", grid)
    assert band.mean.shape == (300,) and band.std.shape == (300,)
    assert np.all(np.isfinite(band.mean)) and np.all(band.std >= 0.0)

    with pytest.raises(KeyError):
        recon.predict_gp("timingmodel", grid)


def test_project_powerlaw_recovers_injected_spectrum():
    psr, rng = _synthetic_pulsar(n=600, seed=5)
    components = 25
    gp = ds.makegp_fourier(psr, ds.powerlaw, components=components, name="red")
    F = np.asarray(gp.F)
    log10_A, gamma = -13.0, 4.0
    p = {"J0000+0000_red_log10_A": log10_A, "J0000+0000_red_gamma": gamma}
    phi = np.asarray(gp.Phi.getN(p))
    variance = psr.toaerrs**2
    psr.residuals = F @ (np.sqrt(phi) * rng.standard_normal(F.shape[1])) + np.sqrt(
        variance
    ) * rng.standard_normal(len(variance))

    block = dx.red_noise_block(psr, components=components, name="red")
    fit = fastfit(
        noise=dx.white_noise_from_variance(variance),
        blocks=[block],
        residuals=psr.residuals,
        n_sweeps=6,
    )
    proj = dx.project_powerlaw(fit, block)
    assert proj.success
    # Loose recovery: this is a quick-look estimate on one noise realization.
    assert proj.values["gamma"] == pytest.approx(gamma, abs=1.5)
    assert proj.values["log10_A"] == pytest.approx(log10_A, abs=1.0)
