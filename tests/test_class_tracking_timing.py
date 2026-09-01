"""Class-tracked white noise in the nltiming joint transport on real IPTA DR2 data.

Geometry gate for `discovery.transport.class_tracking` through the real seam
(`nltiming.sampling.numpyro.build_joint_transport`), on the two pulsars whose
bake points are committed under ``tests/fixtures/class_tracking_mpe_<psr>.json``
(MPE of the marginalized likelihood plus ridge-regularized Laplace widths;
``components=15`` below is 30 Fourier columns per GP, the same ``k = 88 / 103``
basis the probe measured):
at seeded Laplace-box draws around the bake point and at four stress points,
the tracked chart's whitened metric stays within ``cond <= 1.15`` of the exact
conditional precision (measured 1.02 / 1.11 for J1918-0642 / J1600-3053) and
the conditional-mode offset stays below 0.5 whitened units, while the chart
frozen at the same bake point exceeds ``cond = 10`` on most draws.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from metapulsar.file_discovery import PTA_DATA_RELEASES

DATA = Path(__file__).parent.parent / "data" / "ipta-dr2"
FIXTURES = Path(__file__).parent / "fixtures"

_RELEASE = {"epta": "epta_dr1_v2_2", "ppta": "ppta_dr2", "ng9": "nanograv_9y"}
_BOUNDS = {
    "_efac": (0.3, 10.0),
    "_log10_t2equad": (-9.0, -5.0),
    "_log10_ecorr": (-9.0, -5.0),
}
_BOX_SCALES = (1.0, 2.0, 3.0)
_DRAWS_PER_SCALE = 4


def _file_data(pulsar: str) -> dict:
    base = {
        p: PTA_DATA_RELEASES[k]["base_dir"].rstrip("/") for p, k in _RELEASE.items()
    }
    pkg = {p: PTA_DATA_RELEASES[k]["timing_package"] for p, k in _RELEASE.items()}
    paths = {
        "epta": (
            DATA / base["epta"] / pulsar / f"{pulsar}.par",
            DATA / base["epta"] / pulsar / f"{pulsar}_all.tim",
        ),
        "ppta": (
            DATA / base["ppta"] / "par" / f"{pulsar}_dr1dr2.par",
            DATA / base["ppta"] / "tim" / f"{pulsar}_dr1dr2.tim",
        ),
        "ng9": (
            DATA / base["ng9"] / "par" / f"{pulsar}_NANOGrav_9yv1.gls.par",
            DATA / base["ng9"] / "tim" / f"{pulsar}_NANOGrav_9yv1.tim",
        ),
    }
    out = {}
    for pta, (par, tim) in paths.items():
        if par.is_file() and tim.is_file():
            out[pta] = [{"par": par, "tim": tim, "timing_package": pkg[pta]}]
    return out


def _clock_dir() -> Path:
    path = Path(os.environ.get("TEMPO2", "/opt/software/tempo2/T2runtime")) / "clock"
    if not path.is_dir():
        pytest.skip(f"tempo2 clock directory not found at {path}")
    return path


def _bounds_for(name: str):
    for suffix, b in _BOUNDS.items():
        if name.endswith(suffix):
            return b
    raise KeyError(name)


def _build(pulsar_name: str):
    """(ctx, likelihood, mpe, sigma): the probe's construction through public APIs."""
    import discovery as ds
    from enterprise.signals import selections as es

    from metapulsar import create_metapulsar
    from metapulsar.selection_utils import create_staggered_selection
    from nltiming import TimingCoordinatePolicy, TimingInference, TimingSpec
    from nltiming.sampling.numpyro import ensure_x64

    ensure_x64()
    ds.config(kernels="metamath")
    fixture = FIXTURES / f"class_tracking_mpe_{pulsar_name}.json"
    if not fixture.is_file():
        pytest.skip(f"bake-point fixture missing: {fixture}")
    blob = json.loads(fixture.read_text(encoding="utf-8"))
    mpe, sigma = blob["mpe"], blob["sigma"]

    file_data = _file_data(pulsar_name)
    if len(file_data) < 2:
        pytest.skip(f"{pulsar_name}: need >= 2 PTAs under {DATA}")
    pulsar = create_metapulsar(
        file_data,
        combination_strategy="shared",
        combine_components=["astrometry", "spindown", "binary", "dispersion"],
        use_pulse_numbers="yes",
        clock_dir=_clock_dir(),
    )

    ecorr_sel = es.Selection(
        create_staggered_selection("ecorr", {("group", "g", "f"): None})
    )

    def selection(psr, _sel=ecorr_sel):
        masks = _sel(psr).masks
        flags = np.full(len(psr.toas), "", dtype="U64")
        for key, mask in masks.items():
            flags[np.asarray(mask, dtype=bool)] = (
                key.split("_", 1)[1] if "_" in key else key
            )
        return flags

    spec = TimingSpec(
        engines={"tempo2": "jug", "pint": "jug"},
        derivative_method="autodiff",
        tempo2_native="fixed_state_stripped",
        inference=TimingInference.sample_all(),
        whitening=None,
        binary_chart="auto",
        prior_policy="wide_default",
        coordinate_policy=TimingCoordinatePolicy(
            linear_scale=100.0, nonlinear_scale=100.0
        ),
        name="timing",
    )
    ctx = spec.for_pulsar(pulsar)
    use_ecorr = (
        ds.makegp_ecorr(
            pulsar, noisedict={}, enterprise=True, selection=selection
        ).F.shape[1]
        > 0
    )
    likelihood = ds.PulsarLikelihood(
        [
            pulsar.residuals,
            ds.makenoise_measurement(
                pulsar, {}, ecorr=use_ecorr, enterprise=True, selection=selection
            ),
            ds.makegp_fourier(pulsar, ds.powerlaw, components=15, name="red_noise"),
            ds.makegp_fourier(
                pulsar,
                ds.powerlaw,
                components=15,
                fourierbasis=ds.dmfourierbasis,
                name="dm_gp",
            ),
            *ctx.discovery_signals(joint=True),
        ]
    )
    return ctx, likelihood, mpe, sigma


def _white_names(params):
    return [n for n in params if n.endswith(tuple(_BOUNDS))]


def _points(mpe, sigma, seed=0):
    rng = np.random.default_rng(seed)
    wn = _white_names(mpe)
    pts = []
    for c in _BOX_SCALES:
        for _ in range(_DRAWS_PER_SCALE):
            th = dict(mpe)
            for n in wn:
                lo, hi = _bounds_for(n)
                th[n] = float(
                    np.clip(mpe[n] + c * sigma[n] * rng.standard_normal(), lo, hi)
                )
            pts.append(th)
    for suffixes, bump in (
        (("_efac",), None),
        (("_log10_t2equad",), 0.5),
        (("_log10_ecorr",), 0.5),
        (("_log10_t2equad", "_log10_ecorr"), 1.0),
    ):
        th = dict(mpe)
        for n in wn:
            if n.endswith(suffixes):
                lo, hi = _bounds_for(n)
                th[n] = float(
                    np.clip(mpe[n] * 1.25 if bump is None else mpe[n] + bump, lo, hi)
                )
        pts.append(th)
    return pts


def _geometry(transport, params, live_solve, W, r0):
    """cond of the whitened exact conditional precision, and the mode offset."""
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    d = transport.diagnostics(
        params, noise_solve=lambda X: live_solve(X, params=params)
    )
    cond = d["metric_eig_max"] / d["metric_eig_min"]
    cf, pinv, b = transport._factor(params)
    NmW, _ = live_solve(jnp.asarray(W), params=params)
    G_exact = np.asarray(W).T @ np.asarray(NmW)
    b_exact = np.asarray(NmW).T @ r0
    H = G_exact + np.diag(np.asarray(pinv))
    mu_chart = np.asarray(cho_solve(cf, b))
    mu_exact = np.linalg.solve(H, b_exact)
    L = np.tril(np.asarray(cf[0]))
    offset = float(np.abs(L.T @ (mu_exact - mu_chart)).max())
    return float(cond), offset


@pytest.mark.slow
@pytest.mark.real_data
@pytest.mark.requires_jug
@pytest.mark.requires_ipta_data
@pytest.mark.parametrize("pulsar_name", ["J1918-0642", "J1600-3053"])
def test_class_tracked_joint_transport_geometry(pulsar_name):
    if not DATA.is_dir():
        pytest.skip(f"IPTA DR2 data not present at {DATA}")
    pytest.importorskip("jax")
    from discovery import metamatrix
    from discovery import transport as dst
    from nltiming.sampling import numpyro as N

    ctx, likelihood, mpe, sigma = _build(pulsar_name)
    kernel = likelihood.white_noise_kernel
    wn0 = {n: mpe[n] for n in _white_names(mpe)}

    tracked = N.build_joint_transport(
        likelihood, ctx, reference_noise=N.class_tracking_reference(likelihood, wn0)
    )
    frozen = N.build_joint_transport(
        likelihood, ctx, reference_noise=dst.reference_noise_frozen(kernel, params0=wn0)
    )
    assert set(wn0) <= set(tracked.params)
    assert tracked.fingerprint() != frozen.fingerprint()

    live_solve = metamatrix.func(kernel.make_solve, working=dst.bake_dtype())
    W = np.asarray(tracked._W)
    r0 = np.asarray(
        ctx.linearization.transport_effective_residual(ctx.pulsar.residuals),
        dtype=float,
    )

    conds_t, conds_f, offsets = [], [], []
    for th in _points(mpe, sigma):
        c_t, off_t = _geometry(tracked, th, live_solve, W, r0)
        c_f, _ = _geometry(frozen, th, live_solve, W, r0)
        conds_t.append(c_t)
        conds_f.append(c_f)
        offsets.append(off_t)

    # exact at the bake point, up to roundoff amplified by the timing block's
    # ~1e16 dynamic range (J1600-3053 measures |cond - 1| = 2.5e-6)
    c0, off0 = _geometry(tracked, dict(mpe), live_solve, W, r0)
    assert abs(c0 - 1.0) < 1e-5 and off0 < 1e-4

    assert max(conds_t) <= 1.15, conds_t
    assert max(offsets) < 0.5, offsets
    assert sum(c > 10.0 for c in conds_f) >= 6, conds_f
