"""Frontend-neutral tests for the flexfit numerical core.

These exercise the conditional solve, the staged empirical-Bayes updates, the
variance-group machinery, phase-origin invariance, and spectrum projection with
synthetic data only (NumPy/SciPy), so they run without Discovery or nltiming.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylk.flexfit import (  # noqa: E402
    BasisBlock,
    DiagonalNoise,
    VarianceGroup,
    assemble,
    bounded_variance_update,
    conditional_moments,
    fastfit,
    fourier_pair_groups,
    project_spectrum,
    rho_bounds_from_rms,
    solve_flexible_phi,
)


def _dense_gls_mean(y, T, phi, variance):
    ninv = np.diag(1.0 / variance)
    precision = T.T @ ninv @ T + np.diag(1.0 / phi)
    cov = np.linalg.inv(precision)
    mean = cov @ T.T @ ninv @ y
    return mean, cov


def _random_block(
    rng, n_obs, n_col, name, *, sigma_min=1e-9, sigma_max=1e-2, kind="red"
):
    matrix = rng.standard_normal((n_obs, n_col))
    n_freq = n_col // 2
    groups = fourier_pair_groups(
        matrix, prefix=name, n_freq=n_freq, sigma_min=sigma_min, sigma_max=sigma_max
    )
    names = tuple(f"c{i}" for i in range(n_col))
    return BasisBlock(
        name=name, matrix=matrix, coefficient_names=names, groups=groups, kind=kind
    )


# --------------------------------------------------------------------------- #
# conditional solve
# --------------------------------------------------------------------------- #
def test_conditional_moments_matches_dense_gls():
    rng = np.random.default_rng(0)
    n_obs, n_col = 60, 8
    T = rng.standard_normal((n_obs, n_col))
    y = rng.standard_normal(n_obs)
    variance = 0.5 + rng.random(n_obs)
    phi = 10.0 ** rng.uniform(-3, 1, n_col)

    mean, cov, second = conditional_moments(y, T, phi, DiagonalNoise(variance))
    ref_mean, ref_cov = _dense_gls_mean(y, T, phi, variance)

    np.testing.assert_allclose(mean, ref_mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(second, mean**2 + np.diag(ref_cov), rtol=1e-7)


def test_broad_timing_prior_marginalizes_like_least_squares():
    # A column with a 1e40 prior variance behaves like an unconstrained fit.
    rng = np.random.default_rng(1)
    n_obs = 40
    trend = np.linspace(-1, 1, n_obs)
    T = np.column_stack([np.ones(n_obs), trend])  # timing-like design
    y = 3.0 + 2.0 * trend + 0.01 * rng.standard_normal(n_obs)
    variance = np.full(n_obs, 0.01**2)
    phi = np.array([1e40, 1e40])

    mean, _, _ = conditional_moments(y, T, phi, DiagonalNoise(variance))
    ols, *_ = np.linalg.lstsq(T, y, rcond=None)
    np.testing.assert_allclose(mean, ols, rtol=1e-5)


# --------------------------------------------------------------------------- #
# bounded variance update
# --------------------------------------------------------------------------- #
def test_bounded_variance_update_em_and_map():
    second = np.array([4.0, 2.0, 6.0])
    em = VarianceGroup("g", (0, 1, 2), lower=1e-6, upper=1e6)
    rho, hit = bounded_variance_update(second, em)
    assert rho == pytest.approx(np.mean(second))
    assert not hit

    clipped = VarianceGroup("g", (0, 1, 2), lower=1e-6, upper=1.0)
    rho, hit = bounded_variance_update(second, clipped)
    assert rho == pytest.approx(1.0)
    assert hit

    prior = VarianceGroup("g", (0, 1, 2), lower=1e-6, upper=1e6, alpha=2.0, beta=5.0)
    rho, _ = bounded_variance_update(second, prior)
    assert rho == pytest.approx((second.sum() + 2 * 5.0) / (3 + 2 * 2.0 + 2))


# --------------------------------------------------------------------------- #
# staged solve recovers injected variance
# --------------------------------------------------------------------------- #
def test_solve_recovers_injected_group_variance():
    rng = np.random.default_rng(2)
    n_obs, n_freq = 400, 10
    n_col = 2 * n_freq
    F = rng.standard_normal((n_obs, n_col))
    rho_true = 0.3
    coeffs = np.sqrt(rho_true) * rng.standard_normal(n_col)
    noise_sigma = 0.05
    y = F @ coeffs + noise_sigma * rng.standard_normal(n_obs)

    # Single group tying all columns to one variance, wide bounds.
    group = VarianceGroup("all", tuple(range(n_col)), lower=1e-6, upper=1e2)
    block = BasisBlock(
        "red", F, tuple(f"c{i}" for i in range(n_col)), (group,), kind="red"
    )
    model = assemble([block])
    noise = DiagonalNoise(np.full(n_obs, noise_sigma**2))

    result = solve_flexible_phi(y, model, noise, n_sweeps=5)
    assert result.group_variances["all"] == pytest.approx(rho_true, rel=0.35)
    # Waveform should explain most of the (noise-free) signal.
    truth = F @ coeffs
    wave = result.waveform("red")
    assert np.corrcoef(truth, wave)[0, 1] > 0.9


def test_staged_timing_marginalized_on_first_sweep():
    rng = np.random.default_rng(3)
    n_obs = 200
    t = np.linspace(0, 1, n_obs)
    timing = np.column_stack([np.ones(n_obs), t])
    F = rng.standard_normal((n_obs, 6))
    y = (
        5.0
        - 3.0 * t
        + F @ (0.2 * rng.standard_normal(6))
        + 0.02 * rng.standard_normal(n_obs)
    )

    tblock = BasisBlock(
        "timing",
        timing,
        ("offset", "slope"),
        (
            VarianceGroup(
                "offset", (0,), 1e40, 1e40, initial=1e40, update_from_sweep=10**9
            ),
            VarianceGroup(
                "slope", (1,), 1e40, 1e40, initial=1e40, update_from_sweep=10**9
            ),
        ),
        kind="timing",
    )
    gblock = BasisBlock(
        "red",
        F,
        tuple(f"c{i}" for i in range(6)),
        (VarianceGroup("all", tuple(range(6)), 1e-6, 1e2),),
        kind="red",
    )
    model = assemble([tblock, gblock])
    noise = DiagonalNoise(np.full(n_obs, 0.02**2))
    result = solve_flexible_phi(y, model, noise, n_sweeps=3)
    # Timing prior stays fixed at 1e40 throughout.
    np.testing.assert_allclose(result.phi_diagonal[:2], [1e40, 1e40])
    # The linear timing trend is absorbed: whitened residual has small trend.
    whitened = y - result.waveform("timing") - result.waveform("red")
    assert np.sqrt(np.mean(whitened**2)) < 0.05


# --------------------------------------------------------------------------- #
# phase-origin invariance
# --------------------------------------------------------------------------- #
def test_phase_origin_invariance_of_tied_pairs():
    rng = np.random.default_rng(4)
    n_obs, n_freq = 120, 5
    n_col = 2 * n_freq
    F = rng.standard_normal((n_obs, n_col))
    y = rng.standard_normal(n_obs)
    variance = np.full(n_obs, 0.1)

    def build(matrix):
        groups = fourier_pair_groups(
            matrix, prefix="r", n_freq=n_freq, sigma_min=1e-6, sigma_max=1e2
        )
        block = BasisBlock(
            "red", matrix, tuple(f"c{i}" for i in range(n_col)), groups, kind="red"
        )
        return solve_flexible_phi(
            y, assemble([block]), DiagonalNoise(variance), n_sweeps=4
        )

    # Rotate each sin/cos pair by a per-pair angle (orthogonal within the pair).
    rotated = F.copy()
    for k in range(n_freq):
        theta = rng.uniform(0, 2 * np.pi)
        s, c = F[:, 2 * k], F[:, 2 * k + 1]
        rotated[:, 2 * k] = np.cos(theta) * s + np.sin(theta) * c
        rotated[:, 2 * k + 1] = -np.sin(theta) * s + np.cos(theta) * c

    wave_a = build(F).waveform("red")
    wave_b = build(rotated).waveform("red")
    np.testing.assert_allclose(wave_a, wave_b, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# RMS-based bounds and projection
# --------------------------------------------------------------------------- #
def test_rho_bounds_from_rms_roundtrip():
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((100, 4))
    lower, upper = rho_bounds_from_rms(matrix, (0, 1), sigma_min=1e-8, sigma_max=1e-5)
    q = np.sum(matrix[:, :2] ** 2) / 100
    assert lower == pytest.approx(1e-8**2 / q)
    assert upper == pytest.approx(1e-5**2 / q)


def test_project_spectrum_recovers_flat_model():
    # second moments generated from phi = 10^theta; recover theta.
    rng = np.random.default_rng(6)
    n = 12
    theta_true = -2.0
    phi_true = 10.0**theta_true
    second = phi_true * rng.chisquare(df=1, size=n)  # E[s] = phi_true

    def spectrum(theta):
        return np.full(n, 10.0 ** float(theta[0]))

    proj = project_spectrum(
        second, spectrum, theta0=[0.0], parameter_names=("log10_phi",)
    )
    assert proj.success
    assert proj.values["log10_phi"] == pytest.approx(np.log10(second.mean()), abs=1e-3)


# --------------------------------------------------------------------------- #
# fastfit orchestration (no timing model: explicit residuals)
# --------------------------------------------------------------------------- #
def test_fastfit_without_timing_uses_residuals():
    rng = np.random.default_rng(7)
    n_obs, n_freq = 150, 8
    F = rng.standard_normal((n_obs, 2 * n_freq))
    y = F @ (0.3 * rng.standard_normal(2 * n_freq)) + 0.05 * rng.standard_normal(n_obs)
    groups = fourier_pair_groups(
        F, prefix="red", n_freq=n_freq, sigma_min=1e-6, sigma_max=1e2
    )
    block = BasisBlock(
        "red", F, tuple(f"c{i}" for i in range(2 * n_freq)), groups, kind="red"
    )
    noise = DiagonalNoise(np.full(n_obs, 0.05**2))

    fit = fastfit(noise=noise, blocks=[block], residuals=y, n_sweeps=3)
    assert fit.outer_iterations == 1
    assert "red" in fit.block_names
    np.testing.assert_allclose(fit.whitened_residuals(), y - fit.waveform("red"))
    assert fit.provenance["label"] == "quick-look empirical Bayes"


def test_basisblock_requires_full_partition():
    F = np.ones((10, 3))
    with pytest.raises(ValueError, match="partition"):
        BasisBlock(
            "b",
            F,
            ("a", "b", "c"),
            (VarianceGroup("g", (0, 1), 1e-3, 1e3),),
            kind="custom",
        )
