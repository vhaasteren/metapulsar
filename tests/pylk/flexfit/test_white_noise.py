"""Gibbs/ECM per-backend white-noise estimation (whitenoise.py).

Synthetic data with known per-backend EFAC/EQUAD plus an injected smooth
(red-noise-like) signal: the alternation must recover the white noise without
letting the GP absorb it (or vice versa). NumPy/SciPy only.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylk.flexfit import (  # noqa: E402
    BasisBlock,
    fit_white_noise,
    fourier_pair_groups,
)
from pylk.flexfit.whitenoise import expected_squared_residuals  # noqa: E402


def _fourier_matrix(t, n_freq):
    tspan = t.max() - t.min()
    cols = []
    for k in range(1, n_freq + 1):
        cols.append(np.sin(2 * np.pi * k * t / tspan))
        cols.append(np.cos(2 * np.pi * k * t / tspan))
    return np.column_stack(cols)


def _make_data(rng, *, n=1200, n_freq=8):
    t = np.sort(rng.uniform(0.0, 1.0, n))
    F = _fourier_matrix(t, n_freq)
    # red-ish injected signal: variance falling as k^-3, few-microsecond scale
    k = np.repeat(np.arange(1, n_freq + 1), 2)
    coeff_sigma = 3.0e-6 * k**-1.5
    coeffs = rng.standard_normal(2 * n_freq) * coeff_sigma
    signal = F @ coeffs

    toaerrs = rng.uniform(0.5e-6, 1.5e-6, n)
    backends = np.array(["A", "B", "C"])[np.arange(n) % 3]
    efac_true = {"A": 1.4, "B": 0.8, "C": 2.0}
    equad_true = {"A": 0.0, "B": 8.0e-7, "C": 1.5e-6}
    variance = np.empty(n)
    for b in ("A", "B", "C"):
        m = backends == b
        variance[m] = efac_true[b] ** 2 * toaerrs[m] ** 2 + equad_true[b] ** 2
    y = signal + rng.standard_normal(n) * np.sqrt(variance)

    groups = fourier_pair_groups(
        F, prefix="red", n_freq=n_freq, sigma_min=1e-9, sigma_max=1e-4
    )
    block = BasisBlock(
        name="red",
        matrix=F,
        coefficient_names=tuple(f"c{i}" for i in range(2 * n_freq)),
        groups=groups,
        kind="red",
    )
    return y, toaerrs, backends, block, efac_true, equad_true


def test_recovers_per_backend_efac_equad():
    rng = np.random.default_rng(42)
    y, toaerrs, backends, block, efac_true, equad_true = _make_data(rng)

    result = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        max_iterations=15,
    )
    assert result.converged
    for b, f_true in efac_true.items():
        assert result.efac[b] == pytest.approx(f_true, rel=0.12), b
    # equad in quadrature against the backend's toaerr scale
    assert result.equad["A"] < 4.0e-7  # true 0
    assert result.equad["B"] == pytest.approx(8.0e-7, rel=0.35)
    assert result.equad["C"] == pytest.approx(1.5e-6, rel=0.25)
    # per-TOA variance is consistent with the reported (efac, equad)
    m = backends == "C"
    expected = result.efac["C"] ** 2 * toaerrs[m] ** 2 + result.equad["C"] ** 2
    np.testing.assert_allclose(result.variance[m], expected, rtol=1e-12)


def test_efac_only_closed_form():
    rng = np.random.default_rng(3)
    n = 900
    toaerrs = rng.uniform(0.8e-6, 1.2e-6, n)
    backends = np.array(["X", "Y"])[np.arange(n) % 2]
    efac_true = {"X": 1.5, "Y": 0.7}
    y = np.concatenate(
        [rng.standard_normal(np.sum(backends == b)) * efac_true[b] for b in ("X", "Y")]
    )
    # order y to match the backends layout
    out = np.empty(n)
    out[backends == "X"] = y[: np.sum(backends == "X")] * toaerrs[backends == "X"]
    out[backends == "Y"] = y[np.sum(backends == "X") :] * toaerrs[backends == "Y"]

    result = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        residuals=out,
        fit_equad=False,
    )
    assert result.converged
    assert result.equad["X"] == 0.0 and result.equad["Y"] == 0.0
    for b, f_true in efac_true.items():
        assert result.efac[b] == pytest.approx(f_true, rel=0.08), b


def test_covariance_correction_prevents_signal_absorption():
    """Without the diag(T Sigma T^T) term, absorbed-signal uncertainty would be
    double-counted; check e_i >= (y - T m)_i^2 and that the correction is
    nonzero where the basis has support."""
    rng = np.random.default_rng(7)
    y, toaerrs, backends, block, *_ = _make_data(rng, n=600, n_freq=6)
    result = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        max_iterations=3,
    )
    from pylk.flexfit.basis import assemble

    model = assemble((block,))
    e = expected_squared_residuals(y, model, result.solve)
    r2 = (y - model.expand(result.solve.coefficient_mean)) ** 2
    assert np.all(e >= r2 - 1e-30)
    assert np.mean(e - r2) > 0.0


def test_ecorr_block_recovery_with_white_noise():
    """Epoch-correlated (ECORR) noise on one backend is recovered by an ECORR
    variance-group block fitted jointly with per-backend EFAC via the Gibbs
    alternation, without inflating that backend's EFAC."""
    from pylk.flexfit import BasisBlock, VarianceGroup

    rng = np.random.default_rng(19)
    n_epoch, per_epoch = 120, 4
    n = n_epoch * per_epoch
    epoch = np.repeat(np.arange(n_epoch), per_epoch)
    backends = np.where(epoch % 2 == 0, "NG", "OTH")  # alternate whole epochs
    toaerrs = rng.uniform(0.8e-6, 1.2e-6, n)
    efac_true = {"NG": 1.2, "OTH": 0.9}
    ecorr_true = 2.0e-6  # on NG epochs only

    y = rng.standard_normal(n) * toaerrs
    for b in ("NG", "OTH"):
        m = backends == b
        y[m] *= efac_true[b]
    ng_epochs = np.unique(epoch[backends == "NG"])
    shift = {e: rng.standard_normal() * ecorr_true for e in ng_epochs}
    for e in ng_epochs:
        y[epoch == e] += shift[e]

    U = np.vstack([(epoch == e) for e in ng_epochs]).T.astype(float)
    k = U.shape[1]
    block = BasisBlock(
        name="ecorr_NG",
        matrix=U,
        coefficient_names=tuple(f"e{i}" for i in range(k)),
        groups=(VarianceGroup("ecorr_NG", tuple(range(k)), lower=1e-18, upper=1e-10),),
        kind="ecorr",
    )
    result = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        fit_equad=False,
        max_iterations=10,
    )
    ecorr_fit = np.sqrt(result.solve.group_variances["ecorr_NG"])
    assert ecorr_fit == pytest.approx(ecorr_true, rel=0.25)
    for b, f_true in efac_true.items():
        assert result.efac[b] == pytest.approx(f_true, rel=0.12), b


def test_noisedict_conventions():
    rng = np.random.default_rng(11)
    y, toaerrs, backends, block, *_ = _make_data(rng, n=600, n_freq=4)
    result = fit_white_noise(
        toaerrs=toaerrs,
        backend_flags=backends,
        blocks=[block],
        residuals=y,
        max_iterations=4,
    )
    nd_tn = result.noisedict("J0000+0000", convention="tnequad")
    nd_t2 = result.noisedict("J0000+0000", convention="t2equad")
    for b in ("A", "B", "C"):
        assert nd_tn[f"J0000+0000_{b}_efac"] == pytest.approx(result.efac[b])
        q_tn = 10.0 ** nd_tn[f"J0000+0000_{b}_log10_tnequad"]
        q_t2 = 10.0 ** nd_t2[f"J0000+0000_{b}_log10_t2equad"]
        # t2equad = tnequad / efac (equal N by construction)
        if result.equad[b] > 1e-9:
            assert q_t2 == pytest.approx(q_tn / result.efac[b], rel=1e-6)
