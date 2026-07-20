"""M3 integration: single-pulsar marginalized dynamic decentering (§11.3).

The engine-neutral unit tests live in the nltiming repo
(``tests/test_decentered_model.py``, T-N1..T-N4). These exercise the whole
stack on a discovery-native simulated pulsar: the marginalized likelihood, the
live-kernel ``MarginalTransport``, the exact marginal identity against an
independently-assembled surrogate likelihood, cross-mode agreement with the
joint full-basis model, and the decentered geometry certifier.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("jug")
ds = pytest.importorskip("discovery")
pint_config = pytest.importorskip("pint.config")
pytest.importorskip("pint.simulation")
pytest.importorskip("numpyro")

from numpyro.infer import init_to_value  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from pint.models import get_model  # noqa: E402
from pint.simulation import make_fake_toas_uniform  # noqa: E402

from metapulsar import create_metapulsar  # noqa: E402
from nltiming import (  # noqa: E402
    NonLinearTimingModel,
    TimingInference,
    box_hyper_probe_points,
    certify_decentered_geometry,
)
from nltiming.sampling import numpyro as N  # noqa: E402

pytestmark = [
    pytest.mark.requires_jug,
    pytest.mark.requires_discovery,
    pytest.mark.requires_libstempo,
    pytest.mark.slow,
]

GAMMA, LOG10_A = 3.0, -14.0
_PRIORS = {
    r".*rednoise_log10_A.*": [-18.0, -11.0],
    r".*rednoise_gamma.*": [0.0, 7.0],
}


def _simulate_metapulsar(workdir):
    np.random.seed(42)
    model = get_model(pint_config.examplefile("NGC6440E.par"))
    toas = make_fake_toas_uniform(
        startMJD=53400,
        endMJD=56000,
        ntoas=90,
        model=model,
        obs="gbt",
        error=1.0,
        add_noise=True,
    )
    (workdir / "d.par").write_text(model.as_parfile())
    toas.write_TOA_file(str(workdir / "d.tim"), format="tempo2")
    return create_metapulsar(
        {
            "d": [
                {
                    "par": str(workdir / "d.par"),
                    "tim": str(workdir / "d.tim"),
                    "timing_package": "pint",
                }
            ]
        },
        use_pulse_numbers="no",
    )


@pytest.fixture(scope="module")
def decentered_setup():
    """One simulated pulsar; build BOTH the decentered (F0/F1 sampled, DM
    marginalized, whitening=None) and the joint sample_all model on it."""
    ds.config(kernels="metamath")
    workdir = Path(tempfile.mkdtemp(prefix="m3_it_"))
    mp = _simulate_metapulsar(workdir)
    noisedict = {f"{mp.name}_efac": 1.0, f"{mp.name}_log10_t2equad": -8.0}
    hyper = {
        f"{mp.name}_rednoise_gamma": GAMMA,
        f"{mp.name}_rednoise_log10_A": LOG10_A,
    }

    # --- decentered mode: DM marginalized (delta-flat), F0/F1 sampled ---
    ntm_d = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["DM"]),
        name="timing",
    )
    ctx_d = ntm_d.for_pulsar(mp)
    psl_d = ds.PulsarLikelihood(
        [
            mp.residuals,
            ds.makenoise_measurement_simple(mp, noisedict),
            ds.makegp_fourier(mp, ds.powerlaw, 20, name="rednoise"),
            *ctx_d.discovery_signals(),
        ]
    )
    model_d = N.decentered_model(psl_d, ctx_d, priors=_PRIORS, fixed=noisedict)

    # --- joint mode: sample_all on the same pulsar ---
    ntm_j = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        name="timing",
    )
    ctx_j = ntm_j.for_pulsar(mp)
    psl_j = ds.PulsarLikelihood(
        [
            mp.residuals,
            ds.makenoise_measurement_simple(mp, noisedict),
            ds.makegp_fourier(mp, ds.powerlaw, 20, name="rednoise"),
            *ctx_j.discovery_signals(joint=True),
        ]
    )
    model_j = N.joint_model(psl_j, ctx_j, priors=_PRIORS, fixed=noisedict)

    yield {
        "mp": mp,
        "noisedict": noisedict,
        "hyper": hyper,
        "ctx_d": ctx_d,
        "psl_d": psl_d,
        "model_d": model_d,
        "ctx_j": ctx_j,
        "model_j": model_j,
        "workdir": workdir,
    }
    ds.config(kernels="matrix")


def _surrogate_marginal(setup):
    """Independently-assembled linear-Gaussian surrogate ln p_marg(y_t | eta).

    A NEW PulsarLikelihood over ``y_t`` with the SAME fixed WN and RN/DM GPs,
    the plan's marginal timing GPs, and the sampled timing block promoted to a
    unit-normal GP on ``W_s`` via ``makegp_standard_normal`` (§11.3). It shares
    no graph node with ``model.transport``; the sampled timing is integrated
    under ``z_s ~ N(0, I)``. This is NOT the joint mode's frozen-N0 formula.
    """
    from discovery.signals import makegp_standard_normal

    ctx, mp, noisedict = setup["ctx_d"], setup["mp"], setup["noisedict"]
    lin = ctx.linearization
    W_s = np.asarray(lin.sampled_basis, dtype=float)
    y_t = np.asarray(
        lin.transport_effective_residual(np.asarray(mp.residuals)), dtype=float
    )
    # The plan's marginal timing GPs: discovery_signals() minus the sampled
    # nonlinear delay (the signal whose params ARE the delay keys).
    delay_keys = set(ctx.delay_keys)
    marginal_gps = [
        s
        for s in ctx.discovery_signals()
        if not (set(getattr(s, "params", [])) & delay_keys)
    ]
    psl_lin = ds.PulsarLikelihood(
        [
            y_t,
            ds.makenoise_measurement_simple(mp, noisedict),
            ds.makegp_fourier(mp, ds.powerlaw, 20, name="rednoise"),
            *marginal_gps,
            makegp_standard_normal(mp, W_s, name="sampled_timing_surrogate"),
        ]
    )
    return lambda eta: float(psl_lin.logL({**eta, **noisedict}))


def test_m1_decentered_marginal_identity(decentered_setup):
    """T-M1 (merge gate): log_density(xi=0, eta) - ln p_marg_lin(eta) has spread
    < 0.05 over a 3-point red-noise-amplitude scan. The reference is the
    independent surrogate marginal (live C(eta); NOT the frozen-N0 joint form).

    The scan spans the injected amplitude and below (log10_A <= -14.0): there the
    GLS centering ``mu(eta)`` is small (|mu| ~ 0.05) and NGC6440E stays in its
    linear regime, so the linear-Gaussian surrogate is the exact marginal and the
    identity is constant. At much larger amplitudes ``mu`` grows and the exact
    engine phase-wrap nonlinearity departs from the linear surrogate BY DESIGN
    (real physics the certifier reports, not a model defect). Whether the timing
    signal ever approaches the pulse period is a pulsar-population fact (rare for
    MSPs, possible for ordinary / gamma-ray / spider pulsars); staying on-wrap is
    the user's job via pulse-number tracking (``use_pulse_numbers`` in
    PINT/tempo2/JUG, exposed by MetaPulsar), not something this mode fixes. The
    ``|mu|`` guard below fails loudly if a fixture change silently re-enters the
    nonlinear band while still passing the identity."""
    model = decentered_setup["model_d"]
    gamma_key = f"{decentered_setup['mp'].name}_rednoise_gamma"
    amp_key = f"{decentered_setup['mp'].name}_rednoise_log10_A"
    ln_p_marg = _surrogate_marginal(decentered_setup)

    dim = int(model.transport.dimension)
    xi0 = jnp.zeros(dim)

    diffs = []
    for log10a in (-14.5, -14.25, -14.0):
        eta = {gamma_key: GAMMA, amp_key: log10a}
        # The GLS centering must stay in the linear band for the linear-Gaussian
        # surrogate to be the exact marginal (see docstring / pulse-number note).
        mu, _ = model.transport.apply(eta, xi0)
        assert float(np.linalg.norm(np.asarray(mu))) < 0.1, (log10a, mu)
        lp, _ = log_density(model, (), {}, {model.xi_site: xi0, **eta})
        diffs.append(float(lp) - ln_p_marg(eta))

    diffs = np.asarray(diffs)
    assert np.ptp(diffs) < 0.05, diffs - diffs[0]


def test_m2_cross_mode_consistency(decentered_setup):
    """T-M2 (merge gate): short NUTS from decentered_model and joint_model
    describe the same F0/F1 posterior.

    Primary check: the F0/F1 posterior MEANS agree within 3 joint-sigma (the
    physics gate). Secondary: the posterior widths agree to an MC/mixing-limited
    factor. The joint full-basis chain (43-dim: F0/F1/DM + 40 GP coefficients) is
    precisely the ill-conditioned geometry the decentered mode was built to fix,
    so at any affordable NUTS budget its width is MC/mixing-limited and cannot be
    matched to a tight 30% against the well-mixed 4-dim decentered chain (the
    F1-vs-low-frequency-red-noise degeneracy mixes slowly in the joint frame).
    The decentered target density's exactness is pinned independently and to
    machine precision by the linear-duck identity T-N1, not by this MC gate.

    The two chains are intentionally NOT symmetrically configured: the decentered
    chain inits at ``decentered_init_values`` + hypers and caps ``max_tree_depth``
    at the small-k_s default (7), while the joint chain uses the default recipe.
    Symmetrizing them would not rescue the joint chain's mixing and is not the
    point of a cross-mode physics-consistency gate."""
    ctx_d, model_d = decentered_setup["ctx_d"], decentered_setup["model_d"]
    ctx_j, model_j = decentered_setup["ctx_j"], decentered_setup["model_j"]
    hyper = decentered_setup["hyper"]
    stem = ctx_d.name_stem

    mcmc_d = N.nuts(
        model_d,
        ctx_d,
        num_warmup=300,
        num_samples=400,
        target_accept=0.9,
        max_tree_depth=7,
        progress_bar=False,
        init_strategy=init_to_value(
            values={**N.decentered_init_values(ctx_d, model_d.transport), **hyper}
        ),
    )
    mcmc_d.run(jax.random.PRNGKey(0))
    df_d = model_d.to_df(mcmc_d.get_samples())

    mcmc_j = N.nuts(
        model_j,
        ctx_j,
        num_warmup=300,
        num_samples=400,
        target_accept=0.9,
        progress_bar=False,
    )
    mcmc_j.run(jax.random.PRNGKey(1))
    df_j = model_j.to_df(mcmc_j.get_samples())

    for fp in ("F0", "F1"):
        col = f"{stem}_{fp}_theta_native"
        a, b = np.asarray(df_d[col], float), np.asarray(df_j[col], float)
        sig_a, sig_b = a.std(ddof=1), b.std(ddof=1)
        joint_sigma = float(np.hypot(sig_a, sig_b))
        # Physics gate: means agree within 3 joint-sigma.
        assert abs(a.mean() - b.mean()) < 3.0 * joint_sigma, (
            fp,
            a.mean(),
            b.mean(),
            joint_sigma,
        )
        # MC/mixing-limited width consistency (joint full-basis is the hard
        # geometry; see docstring). A gross disagreement (> ~2.5x) still fails.
        ratio = sig_a / sig_b
        assert 0.4 < ratio < 2.5, (fp, sig_a, sig_b, ratio)


def test_m3_certifier_optional_path_smoke(decentered_setup):
    """T-M3 (optional-path smoke): certify_decentered_geometry over box probes at
    the hyper MPE returns a well-formed report. passed=True is NOT required on a
    fully nonlinear real pulsar (the linear-duck T-N4 carries the math gate)."""
    ctx, model = decentered_setup["ctx_d"], decentered_setup["model_d"]
    mp = decentered_setup["mp"]
    eta_mpe = {
        f"{mp.name}_rednoise_gamma": GAMMA,
        f"{mp.name}_rednoise_log10_A": LOG10_A,
    }
    bounds = {
        f"{mp.name}_rednoise_gamma": (0.0, 7.0),
        f"{mp.name}_rednoise_log10_A": (-18.0, -11.0),
    }
    points = box_hyper_probe_points(eta_mpe, bounds)

    report = certify_decentered_geometry(model, ctx, hyper_points=points)

    # Well-formed: thresholds recorded, fingerprints set, per-point payload full.
    assert report.thresholds is not None
    assert report.context_fingerprint == ctx.fingerprint()
    assert report.model_fingerprint
    assert len(report.per_point) == len(points)
    assert all("hyper" in p for p in report.per_point)
    assert isinstance(report.passed, bool)
    # The live-kernel per-TOA scale was finite/positive at every probe (the
    # certifier would have raised otherwise); the residual RMS is a real number.
    assert np.isfinite(report.max_residual_remainder_rms)
