"""MetaPulsar integration (PR-E3): Enterprise/PTMCMC parity for marginalized
dynamic decentering (feature_enterprise_dynamic_parity.md §10.2).

T-EM1 (cross-frontend posterior consistency): a short PTMCMC run over the
Enterprise decentered target vs a short Discovery ``decentered_model`` NUTS run
agree on the F0/F1 posterior (means within 3 joint-sigma; widths MC-limited).
T-EM2 (record integrity): the run manifest carries ``latent_decodable=false``,
the ``ptmcmc-decentered`` chain layout, and the E26 reconstruction recipe whose
digests match ``ctx.fingerprint()`` / the linearization; a Discovery-side
geometry report for the same context has a matching context fingerprint.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

pytest.importorskip("jug")
ds = pytest.importorskip("discovery")
pytest.importorskip("pint.config")
pytest.importorskip("numpyro")
pytest.importorskip("enterprise")
pytest.importorskip("PTMCMCSampler")

from enterprise.signals import (  # noqa: E402
    gp_signals,
    parameter,
    signal_base,
    white_signals,
)
from enterprise.signals import utils as ent_utils  # noqa: E402
from numpyro.infer import init_to_value  # noqa: E402
from pint.config import examplefile  # noqa: E402
from pint.models import get_model  # noqa: E402
from pint.simulation import make_fake_toas_uniform  # noqa: E402

from metapulsar import create_metapulsar  # noqa: E402

from nltiming import (  # noqa: E402
    NonLinearTimingModel,
    TimingInference,
    box_hyper_probe_points,
    certify_decentered_geometry,
    write_geometry_report,
)
from nltiming.decentering import NumpyMarginalTransport  # noqa: E402
from nltiming.likelihoods.enterprise import enterprise_marginal_products  # noqa: E402
from nltiming.metric import _column_digest, dynamic_transport_record  # noqa: E402
from nltiming.run_io import (  # noqa: E402
    DYNAMIC_FINAL_NAME,
    attach_decentered_reconstruction,
    build_run_manifest,
    decentered_reconstruction_recipe,
    load_run,
    save_ptmcmc_decentered_checkpoint,
)
from nltiming.sampling import numpyro as N  # noqa: E402
from nltiming.sampling.ptmcmc import (  # noqa: E402
    decentered_chain_layout,
    decentered_initial_point,
    decentered_sampler,
)

pytestmark = [
    pytest.mark.requires_jug,
    pytest.mark.requires_discovery,
    pytest.mark.requires_libstempo,
    pytest.mark.slow,
]

GAMMA, LOG10_A, COMPS = 3.0, -14.0, 10
_PRIORS = {"rednoise_log10_A": (-18.0, -11.0), "rednoise_gamma": (0.0, 7.0)}


def _simulate_metapulsar(workdir):
    np.random.seed(42)
    model = get_model(examplefile("NGC6440E.par"))
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
def enterprise_setup(tmp_path_factory):
    """One NGC6440E pulsar; the shared Discovery + Enterprise decentered stack."""
    ds.config(kernels="metamath")
    workdir = tmp_path_factory.mktemp("me3")
    mp = _simulate_metapulsar(workdir)
    tspan = float(np.asarray(mp.toas).max() - np.asarray(mp.toas).min())
    nd = {f"{mp.name}_efac": 1.0, f"{mp.name}_log10_t2equad": -8.0}
    priors = {f"{mp.name}_{k}": v for k, v in _PRIORS.items()}
    eta_mpe = {
        f"{mp.name}_rednoise_gamma": GAMMA,
        f"{mp.name}_rednoise_log10_A": LOG10_A,
    }

    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["DM"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(mp)

    # Discovery decentered_model.
    psl = ds.PulsarLikelihood(
        [
            mp.residuals,
            ds.makenoise_measurement_simple(mp, nd),
            ds.makegp_fourier(mp, ds.powerlaw, COMPS, T=tspan, name="rednoise"),
            *ctx.discovery_signals(),
        ]
    )
    model = N.decentered_model(psl, ctx, priors=priors, fixed=nd)

    # Enterprise decentered target on the SAME context / WN / RN.
    white = white_signals.MeasurementNoise(
        efac=parameter.Constant(1.0), log10_t2equad=parameter.Constant(-8.0)
    )
    pl = ent_utils.powerlaw(
        log10_A=parameter.Uniform(-18, -11), gamma=parameter.Uniform(0, 7)
    )
    rn = gp_signals.FourierBasisGP(
        spectrum=pl, components=COMPS, Tspan=tspan, name="rednoise"
    )
    pta = signal_base.PTA([(white + rn + ntm.enterprise_signal())(mp)])
    products = enterprise_marginal_products(pta, ctx, fixed_wn_params=nd)
    hyper_names = tuple(products.params)
    transport = NumpyMarginalTransport(
        products,
        dimension=len(ctx.plan.sampled),
        key=ctx.joint_site,
        params=hyper_names,
    )
    return {
        "mp": mp,
        "ctx": ctx,
        "nd": nd,
        "priors": priors,
        "eta_mpe": eta_mpe,
        "model": model,
        "pta": pta,
        "transport": transport,
        "hyper_names": hyper_names,
        "workdir": workdir,
    }


def test_em1_cross_frontend_posterior_consistency(enterprise_setup, tmp_path):
    """T-EM1 (merge gate): a short Enterprise PTMCMC run and a short Discovery
    decentered_model NUTS run agree on the F0/F1 posterior mean within 3
    joint-sigma; widths agree to an MC/mixing-limited factor (PTMCMC adaptive
    Metropolis vs NUTS; the density parity itself is pinned exactly by T-E1)."""
    s = enterprise_setup
    ctx, model = s["ctx"], s["model"]
    hyper_names, priors, nd, eta_mpe = (
        s["hyper_names"],
        s["priors"],
        s["nd"],
        s["eta_mpe"],
    )
    transport = s["transport"]
    stem = ctx.name_stem

    # --- Discovery NUTS ---
    mcmc = N.nuts(
        model,
        ctx,
        num_warmup=300,
        num_samples=400,
        target_accept=0.9,
        max_tree_depth=7,
        progress_bar=False,
        init_strategy=init_to_value(
            values={**N.decentered_init_values(ctx, model.transport), **eta_mpe}
        ),
    )
    mcmc.run(jax.random.PRNGKey(0))
    df = model.to_df(mcmc.get_samples())

    # --- Enterprise PTMCMC ---
    sampler = decentered_sampler(
        s["pta"],
        ctx,
        transport,
        tmp_path,
        hyper_names=hyper_names,
        hyper_bounds={n: priors[n] for n in hyper_names},
        fixed=nd,
        verbose=False,
    )
    p0 = decentered_initial_point(ctx, transport, hyper_names, eta_mpe)
    sampler.sample(
        p0,
        25000,
        burn=5000,
        thin=1,
        covUpdate=500,
        SCAMweight=30,
        AMweight=15,
        DEweight=50,
    )
    chain = np.loadtxt(tmp_path / "chain_1.txt")
    chain = chain[5000:]  # burn
    k = len(ctx.plan.sampled)
    from nltiming.decentering import decode_decentered_chain

    delta = decode_decentered_chain(
        chain[:, :k],
        chain[:, k : k + len(hyper_names)],
        hyper_names,
        transport,
        ctx.space,
    )
    theta = ctx.space.to_physical(delta, units="native", coord="delta")

    for fp in ("F0", "F1"):
        a = np.asarray(df[f"{stem}_{fp}_theta_native"], dtype=float)  # NUTS
        b = np.asarray(theta[fp], dtype=float)  # PTMCMC
        sig = float(np.hypot(a.std(ddof=1), b.std(ddof=1)))
        assert abs(a.mean() - b.mean()) < 3.0 * sig, (fp, a.mean(), b.mean(), sig)
        ratio = a.std(ddof=1) / b.std(ddof=1)
        assert 0.4 < ratio < 2.5, (fp, ratio)


def test_em2_record_integrity(enterprise_setup, tmp_path):
    """T-EM2: the manifest carries latent_decodable=false, the ptmcmc-decentered
    chain layout, and the E26 reconstruction recipe whose digests match
    ctx.fingerprint()/the linearization; a Discovery geometry report for the same
    context has a matching context fingerprint."""
    s = enterprise_setup
    ctx, transport, hyper_names, nd = (
        s["ctx"],
        s["transport"],
        s["hyper_names"],
        s["nd"],
    )
    lin = ctx.linearization

    # Synthetic chain (record integrity does not need a real sampler run).
    n, k = 12, len(ctx.plan.sampled)
    rng = np.random.default_rng(0)
    chain = np.column_stack(
        [
            rng.standard_normal((n, k)),
            np.column_stack([rng.uniform(1, 5, n), rng.uniform(-16, -12, n)]),
            rng.standard_normal((n, 4)),
        ]
    )

    manifest = build_run_manifest(
        ctx,
        likelihood="enterprise",
        sampler="ptmcmc-decentered",
        dynamic_transport=dynamic_transport_record(transport),
        chain_layout=decentered_chain_layout(ctx, hyper_names),
        checkpoint={"kind": "npz", "path": DYNAMIC_FINAL_NAME},
    )
    recipe = decentered_reconstruction_recipe(ctx, hyper_names, noisedict=nd)
    # Public path: attach + refresh the content digest (no private helper).
    attach_decentered_reconstruction(manifest, recipe)
    manifest.write(tmp_path)
    save_ptmcmc_decentered_checkpoint(
        tmp_path, chain, ctx, transport, manifest, hyper_names=hyper_names, final=True
    )

    # Assert against the LOADED run products, not the in-memory manifest.
    run = load_run(tmp_path)
    assert run.latent_decodable is False
    loaded_layout = run.run_meta["run_products"]["chain_layout"]
    assert loaded_layout["kind"] == "ptmcmc-decentered"
    assert loaded_layout["hyper_names"] == list(hyper_names)

    # E26 reconstruction recipe survives the round-trip and its digests bind to
    # the context / linearization.
    y_t = np.asarray(
        lin.transport_effective_residual(np.asarray(ctx.pulsar.residuals)), dtype=float
    )
    loaded_recipe = run.run_meta["sections"]["transport"]["reconstruction"]
    assert loaded_recipe["context_digest"] == ctx.fingerprint()
    assert loaded_recipe["linearization_fingerprint"] == lin.fingerprint()
    assert loaded_recipe["sampled_basis_digest"] == _column_digest(
        np.asarray(lin.sampled_basis, dtype=float)
    )
    assert loaded_recipe["effective_residual_digest"] == _column_digest(y_t)
    assert loaded_recipe["hyper_names"] == list(hyper_names)
    assert loaded_recipe["builder"].endswith("enterprise_marginal_products")

    # A Discovery-side geometry report for the same context matches the context
    # fingerprint (E17/§8): one context, one certification surface.
    report = certify_decentered_geometry(
        s["model"], ctx, hyper_points=box_hyper_probe_points(s["eta_mpe"], s["priors"])
    )
    assert report.context_fingerprint == ctx.fingerprint()
    write_geometry_report(report, tmp_path / "geom")
