"""J1 integration: single-pulsar joint full-basis timing (Track J, §6).

The engine-neutral unit tests live in the nltiming repo
(``tests/test_joint_model.py``). These exercise the whole stack on a
discovery-native simulated pulsar: three-way partition, the dynamic transport,
target exactness against a dense oracle, whitening geometry, decoding, the
soft-clamp, and the dynamic run manifest.
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

from pint.models import get_model  # noqa: E402
from pint.simulation import make_fake_toas_uniform  # noqa: E402

from metapulsar import create_metapulsar  # noqa: E402
from nltiming import NonLinearTimingModel  # noqa: E402
from nltiming.sampling import numpyro as N  # noqa: E402

pytestmark = [
    pytest.mark.requires_jug,
    pytest.mark.requires_discovery,
    pytest.mark.requires_libstempo,
    pytest.mark.slow,
]

GAMMA, LOG10_A = 3.0, -14.0


@pytest.fixture(scope="module")
def joint_setup():
    """Build a simulated pulsar, a joint (transform='none') context, a
    residual-form discovery likelihood, and the joint model once."""
    ds.config(kernels="metamath")
    workdir = Path(tempfile.mkdtemp(prefix="j1_it_"))
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
    mp = create_metapulsar(
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

    ntm = NonLinearTimingModel(
        engines="jug",
        sample=["F0", "F1"],
        sample_linear="remaining",
        transform="none",
        name="timing",
    )
    ctx = ntm.for_pulsar(mp)
    noisedict = {f"{mp.name}_efac": 1.0, f"{mp.name}_log10_t2equad": -8.0}
    psl = ds.PulsarLikelihood(
        [
            mp.residuals,
            ds.makenoise_measurement_simple(mp, noisedict),
            ds.makegp_fourier(mp, ds.powerlaw, 20, name="rednoise"),
            *ctx.discovery_signals(joint=True),
        ]
    )
    jm = N.joint_model(psl, ctx, fixed=noisedict)
    yield {
        "mp": mp,
        "ctx": ctx,
        "psl": psl,
        "jm": jm,
        "workdir": workdir,
        "noisedict": noisedict,
        "hyper": {
            f"{mp.name}_rednoise_gamma": GAMMA,
            f"{mp.name}_rednoise_log10_A": LOG10_A,
        },
    }
    ds.config(kernels="matrix")


def _build_pulsar(workdir, parname, tag, ntoas=70):
    model = get_model(pint_config.examplefile(parname))
    toas = make_fake_toas_uniform(
        startMJD=53400,
        endMJD=56000,
        ntoas=ntoas,
        model=model,
        obs="gbt",
        error=1.0,
        add_noise=True,
    )
    (workdir / f"{tag}.par").write_text(model.as_parfile())
    toas.write_TOA_file(str(workdir / f"{tag}.tim"), format="tempo2")
    return create_metapulsar(
        {
            tag: [
                {
                    "par": str(workdir / f"{tag}.par"),
                    "tim": str(workdir / f"{tag}.tim"),
                    "timing_package": "pint",
                }
            ]
        },
        use_pulse_numbers="no",
    )


@pytest.fixture(scope="module")
def joint_multi_setup():
    """Two genuinely different pulsars (ragged timing widths) for the
    multi-pulsar joint assembly (§6.4 / J2)."""
    ds.config(kernels="metamath")
    workdir = Path(tempfile.mkdtemp(prefix="j2_it_"))
    np.random.seed(7)
    mp_a = _build_pulsar(workdir, "NGC6440E.par", "a")
    mp_b = _build_pulsar(workdir, "J0613-sim.par", "b")

    likelihoods, ctxs = [], []
    for mp in (mp_a, mp_b):
        ntm = NonLinearTimingModel(
            engines="jug",
            sample=["F0", "F1"],
            sample_linear="remaining",
            transform="none",
            name="timing",
        )
        ctx = ntm.for_pulsar(mp)
        nd = {f"{mp.name}_efac": 1.0, f"{mp.name}_log10_t2equad": -8.0}
        psl = ds.PulsarLikelihood(
            [
                mp.residuals,
                ds.makenoise_measurement_simple(mp, nd),
                ds.makegp_fourier(mp, ds.powerlaw, 15, name="rednoise"),
                *ctx.discovery_signals(joint=True),
            ]
        )
        likelihoods.append(psl)
        ctxs.append(ctx)
    fixed = {}
    for mp in (mp_a, mp_b):
        fixed[f"{mp.name}_efac"] = 1.0
        fixed[f"{mp.name}_log10_t2equad"] = -8.0
    jm = N.joint_model_multi(likelihoods, ctxs, fixed=fixed)
    yield {
        "mps": (mp_a, mp_b),
        "ctxs": ctxs,
        "likelihoods": likelihoods,
        "jm": jm,
        "fixed": fixed,
    }
    ds.config(kernels="matrix")


def test_multi_pulsar_ragged_assembly(joint_multi_setup):
    """Two pulsars with different timing widths build independent per-pulsar
    transports and sample jointly; the joint path never routes through
    ArrayTransport (which requires equal widths)."""
    ctxs = joint_multi_setup["ctxs"]
    jm = joint_multi_setup["jm"]
    dims = [t.dimension for t in jm.transports]
    widths = [len(c.sampled) for c in ctxs]
    assert widths[0] != widths[1], "expected ragged timing widths"
    assert len(dims) == 2

    # Ragged per-pulsar transports cannot be stacked into an ArrayTransport.
    from discovery import transport as dst

    with pytest.raises(ValueError, match="equal per-pulsar dimension"):
        dst.ArrayTransport(list(jm.transports))


def test_multi_pulsar_joint_samples(joint_multi_setup):
    ctxs, jm = joint_multi_setup["ctxs"], joint_multi_setup["jm"]
    from numpyro.infer.util import log_density
    import jax.numpy as jnp

    seed = {}
    for c, t in zip(ctxs, jm.transports):
        seed[f"{c.name_stem}_joint_xi"] = jnp.zeros(t.dimension)
        seed[f"{c.pulsar.name}_rednoise_gamma"] = 3.0
        seed[f"{c.pulsar.name}_rednoise_log10_A"] = -14.0
    ld, _ = log_density(jm, (), {}, seed)
    assert np.isfinite(float(ld))

    # nuts() init uses ctxs[0]; the multi-model has one xi site per pulsar, so
    # init every timing site at zero (the local center).
    from numpyro.infer import init_to_value

    init = {
        f"{c.name_stem}_joint_xi": np.zeros(t.dimension)
        for c, t in zip(ctxs, jm.transports)
    }
    mcmc = N.nuts(
        jm,
        ctxs[0],
        num_warmup=60,
        num_samples=60,
        progress_bar=False,
        init_strategy=init_to_value(values=init),
    )
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()
    for c in ctxs:
        col = f"{c.name_stem}_F0_delta"
        assert col in samples
        assert np.all(np.isfinite(np.asarray(samples[col])))


@pytest.fixture(scope="module")
def joint_hd_setup():
    """Two pulsars sharing a correlated (HD) global GP, for the exact-HD joint
    assembly (§7). Each likelihood carries intrinsic RN + the GW residual delay."""
    ds.config(kernels="metamath")
    workdir = Path(tempfile.mkdtemp(prefix="hd_it_"))
    np.random.seed(11)
    mp_a = _build_pulsar(workdir, "NGC6440E.par", "a", ntoas=60)
    mp_b = _build_pulsar(workdir, "J0613-sim.par", "b", ntoas=60)
    mps = [mp_a, mp_b]

    gg = ds.makeglobalgp_fourier(
        mps, ds.powerlaw, ds.hd_orf, components=6, T=ds.getspan(mps), name="gw"
    )

    likelihoods, ctxs, fixed = [], [], {}
    for i, mp in enumerate(mps):
        ntm = NonLinearTimingModel(
            engines="jug",
            sample=["F0", "F1"],
            sample_linear="remaining",
            transform="none",
            name="timing",
        )
        ctx = ntm.for_pulsar(mp)
        nd = {f"{mp.name}_efac": 1.0, f"{mp.name}_log10_t2equad": -8.0}
        psl = ds.PulsarLikelihood(
            [
                mp.residuals,
                ds.makenoise_measurement_simple(mp, nd),
                ds.makegp_fourier(mp, ds.powerlaw, 8, name="rednoise"),
                N.gw_residual_delay(gg, i),
                *ctx.discovery_signals(joint=True),
            ]
        )
        likelihoods.append(psl)
        ctxs.append(ctx)
        fixed.update(nd)
    jm = N.joint_model_multi(likelihoods, ctxs, global_gp=gg, fixed=fixed)
    yield {
        "mps": mps,
        "ctxs": ctxs,
        "likelihoods": likelihoods,
        "gg": gg,
        "jm": jm,
        "fixed": fixed,
    }
    ds.config(kernels="matrix")


def test_hd_target_matches_dense_oracle(joint_hd_setup):
    """The correlated-GW joint target equals a dense NumPy oracle: per-pulsar
    (data + intrinsic-RN prior + timing prior) plus the exact dense cross-pulsar
    GW prior ``-½ cᵀ Φ_gw⁻¹ c - ½ log|Φ_gw|`` counted once (§7). Differencing two
    ξ draws cancels the fixed-hyperparameter constants."""
    from discovery import metamatrix

    ctxs, psls, gg = (
        joint_hd_setup["ctxs"],
        joint_hd_setup["likelihoods"],
        joint_hd_setup["gg"],
    )
    jm = joint_hd_setup["jm"]
    mps = joint_hd_setup["mps"]
    hyper = {"gw_gamma": 4.0, "gw_log10_A": -14.0}
    for mp in mps:
        hyper[f"{mp.name}_rednoise_gamma"] = 3.0
        hyper[f"{mp.name}_rednoise_log10_A"] = -14.5
    params_h = {**hyper, **joint_hd_setup["fixed"]}

    # Per-pulsar dense pieces.
    per = []
    for ctx, psl in zip(ctxs, psls):
        ntoa = len(ctx.pulsar.toas)
        noise = next(s for s in psl.signals if isinstance(s, ds.utils.Kernel))
        Ninv, _ = metamatrix.func(noise.make_solve)(np.eye(ntoa), params=params_h)
        rn = psl.sampled_gps[0]
        per.append(
            {
                "ctx": ctx,
                "psl": psl,
                "Ninv": np.asarray(Ninv),
                "phi_rn": np.asarray(rn.Phi.getN(params_h)),
                "rn_key": list(rn.index)[0],
                "F_rn": np.asarray(rn.F),
                "y": np.asarray(ctx.pulsar.residuals, float),
            }
        )

    Phi_gw = np.asarray(gg.Phi.getN(params_h))
    Phi_gw_inv = np.linalg.inv(Phi_gw)
    gw_logdet = np.linalg.slogdet(Phi_gw)[1]

    gw_keys = list(gg.index)

    def oracle_and_model(xis):
        oracle = 0.0
        model = 0.0
        c_gw_flat = []
        for i, (xi, tr, e) in enumerate(zip(xis, jm.transports, per)):
            ctx, psl = e["ctx"], psls[i]
            q, _ = tr.apply(params_h, jnp.asarray(xi))
            parts = tr.split(q)
            z = np.asarray(parts[ctx.joint_site])
            c_rn = np.asarray(parts[e["rn_key"]])
            c_gw = np.asarray(parts[gw_keys[i]])
            c_gw_flat.append(c_gw)
            delta = np.asarray(ctx.space.delta_from_z(jnp.asarray(z), jnp))
            idx = np.asarray(ctx.partition.idx_sampled)
            full = np.zeros(len(ctx.partition.fitpars))
            full[idx] = delta
            detres = (
                e["y"]
                + np.asarray(ctx.engine.residual_delta_jax(jnp.asarray(full)))
                - e["F_rn"] @ c_rn
                - np.asarray(gg.Fs[i]) @ c_gw
            )
            oracle += (
                -0.5 * detres @ e["Ninv"] @ detres
                - 0.5 * np.sum(c_rn * c_rn / e["phi_rn"])
                - 0.5 * np.sum(z * z)
            )
            # model physical target for this pulsar: clogL - 0.5||z||^2
            params = dict(params_h)
            for j, fp in enumerate(ctx.sampled_all):
                params[f"{ctx.name_stem}_{fp}"] = delta[j]
            params[e["rn_key"]] = c_rn
            params[gw_keys[i]] = c_gw
            model += float(psl.clogL(params)) - 0.5 * float(np.sum(z * z))
        c_flat = np.concatenate(c_gw_flat)
        oracle += -0.5 * c_flat @ Phi_gw_inv @ c_flat - 0.5 * gw_logdet
        model += float(N.global_gp_logprior(c_flat, gg, params_h, np))
        return oracle, model

    rng = np.random.default_rng(5)
    xis_a = [rng.standard_normal(t.dimension) for t in jm.transports]
    xis_b = [rng.standard_normal(t.dimension) for t in jm.transports]
    o_a, m_a = oracle_and_model(xis_a)
    o_b, m_b = oracle_and_model(xis_b)
    assert np.isclose((m_a - m_b), (o_a - o_b), rtol=1e-6, atol=1e-3)


def test_hd_joint_density_finite(joint_hd_setup):
    """The full HD model log-density evaluates finitely with the global-GP
    hyperparameters (gw_log10_A/gw_gamma) sampled as free parameters — they drive
    the transport and dense prior but appear in no per-pulsar clogL."""
    ctxs, jm = joint_hd_setup["ctxs"], joint_hd_setup["jm"]
    from numpyro.infer.util import log_density

    seed = {}
    for c, t in zip(ctxs, jm.transports):
        seed[f"{c.name_stem}_joint_xi"] = jnp.zeros(t.dimension)
        seed[f"{c.pulsar.name}_rednoise_gamma"] = 3.0
        seed[f"{c.pulsar.name}_rednoise_log10_A"] = -14.5
    seed["gw_gamma"] = 4.0
    seed["gw_log10_A"] = -14.0
    ld, _ = log_density(jm, (), {}, seed)
    assert np.isfinite(float(ld))


def _log_density(jm, ctx, xi, hyper):
    from numpyro.infer.util import log_density

    ld, _ = log_density(
        jm, (), {}, {f"{ctx.name_stem}_joint_xi": jnp.asarray(xi), **hyper}
    )
    return float(ld)


class _CWExtSignal:
    """A deterministic ExtSignal duck (CW-like): .Fs, .coeffs, .name (§4.4)."""

    def __init__(self, F, name="cw"):
        self.Fs = [np.asarray(F, dtype=float)]
        self.name = name

        def coeffs(params):
            return jnp.asarray([[params["cw_a"], params["cw_b"]]])

        coeffs.params = ["cw_a", "cw_b"]
        self.coeffs = coeffs


def _cw_delay(es, psr_slot=0):
    F = jnp.asarray(es.Fs[psr_slot])

    def delay(params):
        return F @ es.coeffs(params)[psr_slot]

    delay.params = list(es.coeffs.params)
    return delay


def test_cw_template_subtracted_centering(joint_setup):
    """CW ExtSignal template-subtracted centering (§4.4) wired through the joint
    model: it engages, is a pure translation (the conditioner A and the returned
    ``ldj`` are unchanged), and the joint model builds and evaluates finitely."""
    ctx, mp = joint_setup["ctx"], joint_setup["mp"]
    rng = np.random.default_rng(2)
    F_cw = rng.standard_normal((len(mp.toas), 2))
    cw = _CWExtSignal(F_cw)
    cw_params = {"cw_a": 0.5, "cw_b": -0.2}
    noisedict = joint_setup["noisedict"]

    psl = ds.PulsarLikelihood(
        [
            mp.residuals,
            ds.makenoise_measurement_simple(mp, noisedict),
            ds.makegp_fourier(mp, ds.powerlaw, 20, name="rednoise"),
            _cw_delay(cw),
            *ctx.discovery_signals(joint=True),
        ]
    )

    tr_with = N.build_joint_transport(psl, ctx, center_extsignals=[cw], psr_slot=0)
    tr_without = N.build_joint_transport(psl, ctx)
    assert tr_with.diagnostics()["center_extsignals"] == ["cw"]

    params = {**joint_setup["hyper"], **cw_params}
    # A (hence ldj) is independent of the ExtSignal translation.
    for xi in (np.zeros(tr_with.dimension), rng.standard_normal(tr_with.dimension)):
        q_w, ldj_w = tr_with.apply(params, jnp.asarray(xi))
        q_wo, ldj_wo = tr_without.apply(params, jnp.asarray(xi))
        assert np.isclose(float(ldj_w), float(ldj_wo), rtol=1e-10)

    # The shift is a pure (xi-independent), nonzero translation.
    s0 = np.asarray(
        tr_with.apply(params, jnp.zeros(tr_with.dimension))[0]
    ) - np.asarray(tr_without.apply(params, jnp.zeros(tr_with.dimension))[0])
    xr = rng.standard_normal(tr_with.dimension)
    s1 = np.asarray(tr_with.apply(params, jnp.asarray(xr))[0]) - np.asarray(
        tr_without.apply(params, jnp.asarray(xr))[0]
    )
    assert np.allclose(s0, s1, atol=1e-8)
    assert np.max(np.abs(s0)) > 0.0

    # The joint model builds and evaluates with CW centering.
    jm = N.joint_model(
        psl, ctx, center_extsignals=[cw], fixed={**noisedict, **cw_params}
    )
    ld = _log_density(jm, ctx, np.zeros(jm.transport.dimension), joint_setup["hyper"])
    assert np.isfinite(ld)


def test_partition_and_transport_structure(joint_setup):
    ctx, jm = joint_setup["ctx"], joint_setup["jm"]
    assert ctx.partition.nonlinear_sampled == ("F0", "F1")
    assert set(ctx.partition.linear_sampled) == set(ctx.sampled) - {"F0", "F1"}
    tr = jm.transport
    diag = tr.diagnostics()
    names = [b["name"] for b in diag["blocks"]]
    assert names == ["timing", "rednoise"]
    # timing block width == #sampled; GP block == 2*components.
    assert diag["blocks"][0]["k"] == len(ctx.sampled)
    assert diag["blocks"][1]["k"] == 40
    # Softclip is OFF by default: a clamp makes mu != q_hat and adds an
    # eta-dependent -1/2||L^T(mu-q_hat)||^2 to the xi=0 slice, destroying the
    # hyperparameter geometry (measured on IPTA J1640+2224).
    assert "softclip" not in diag
    tr_clip = N.build_joint_transport(joint_setup["psl"], ctx, softclip_zmax=4.0)
    assert tr_clip.diagnostics()["softclip"] == {"timing": 4.0}


def test_target_is_whitened_near_mode(joint_setup):
    """The dynamic transport whitens the true target: unit-step curvature of the
    actual log-density is ~1 in every sampled direction (§4.3). This is the
    geometry-improvement gate — without the fix it was O(1e10)."""
    ctx, jm = joint_setup["ctx"], joint_setup["jm"]
    hyper = joint_setup["hyper"]
    dim = jm.transport.dimension
    ld0 = _log_density(jm, ctx, np.zeros(dim), hyper)
    for k in (0, 1, len(ctx.sampled), dim - 1):  # timing + GP directions
        xi = np.zeros(dim)
        xi[k] = 1.0
        curv = -2.0 * (_log_density(jm, ctx, xi, hyper) - ld0)
        assert 0.5 < curv < 2.0, f"direction {k} curvature {curv} not ~1"


def test_target_density_is_exact(joint_setup):
    """The joint target equals a dense NumPy oracle of the exact Gaussian
    ``log p(y, c) - ½‖z‖²`` (data + GP prior + timing prior), pointwise up to a
    constant. Differencing two ``xi`` draws cancels the (fixed-hyperparameter)
    log-normalizers, so any mis-count of a prior or the residual would show."""
    ctx, psl, jm = joint_setup["ctx"], joint_setup["psl"], joint_setup["jm"]
    mp, hyper, noisedict = (
        joint_setup["mp"],
        joint_setup["hyper"],
        joint_setup["noisedict"],
    )
    tr = jm.transport
    params_h = {**hyper, **noisedict}

    # Dense N^-1 and Phi^-1 (fixed hyperparameters). make_solve returns
    # (N^-1 @ rhs, logdet); we only need the inverse action here.
    from discovery import metamatrix

    ntoa = len(mp.toas)
    noise_sig = next(s for s in psl.signals if isinstance(s, ds.utils.Kernel))
    Ninv, _ = metamatrix.func(noise_sig.make_solve)(np.eye(ntoa), params=params_h)
    Ninv = np.asarray(Ninv)
    gp = psl.sampled_gps[0]
    phi = np.asarray(gp.Phi.getN(params_h))  # (k,)
    y0 = np.asarray(mp.residuals, dtype=float)
    js, gp_key = ctx.joint_site, list(gp.index)[0]

    def oracle_and_model(xi):
        q, ldj = tr.apply(params_h, jnp.asarray(xi))
        parts = tr.split(q)
        z = np.asarray(parts[js])
        c = np.asarray(parts[gp_key])
        delta = np.asarray(ctx.space.delta_from_z(jnp.asarray(z), jnp))
        # exact nonlinear detres = y + residual_delta(full_delta)
        idx = np.asarray(ctx.partition.idx_sampled)
        full = np.zeros(len(ctx.partition.fitpars))
        full[idx] = delta
        detres = y0 + np.asarray(ctx.engine.residual_delta_jax(jnp.asarray(full)))
        detres = detres - np.asarray(gp.F) @ c
        oracle = (
            -0.5 * detres @ Ninv @ detres
            - 0.5 * np.sum(c * c / phi)
            - 0.5 * np.sum(z * z)
        )
        # model target (before ldj / base cancel) = clogL - 0.5||z||^2
        params = dict(params_h)
        for i, fp in enumerate(ctx.sampled_all):
            params[f"{ctx.name_stem}_{fp}"] = delta[i]
        params[gp_key] = c
        model_t = float(psl.clogL(params)) - 0.5 * float(np.sum(z * z))
        return oracle, model_t

    rng = np.random.default_rng(7)
    a = oracle_and_model(rng.standard_normal(tr.dimension))
    b = oracle_and_model(rng.standard_normal(tr.dimension))
    assert np.isclose((a[1] - b[1]), (a[0] - b[0]), rtol=1e-6, atol=1e-4)


def test_transport_jacobian_matches_ldj(joint_setup):
    """Autodiff log|∂q/∂ξ| equals the returned ``ldj`` (sign pinned)."""
    jm = joint_setup["jm"]
    params_h = joint_setup["hyper"]
    tr = jm.transport
    xi0 = jnp.asarray(np.random.default_rng(1).standard_normal(tr.dimension))
    _, ldj = tr.apply(params_h, xi0)
    Jac = jax.jacfwd(lambda xi: tr.apply(params_h, xi)[0])(xi0)
    sign, logabsdet = np.linalg.slogdet(np.asarray(Jac))
    assert sign > 0
    assert np.isclose(float(ldj), float(logabsdet), rtol=1e-8, atol=1e-8)


def test_exact_linear_param_is_linear(joint_setup):
    """A genuinely-linear sample_linear parameter's engine residual is exactly
    linear in its delta (``residual_delta(2δ) == 2·residual_delta(δ)``): the
    exact-linear design-column path (§6.2). (``sample_linear='remaining'`` also
    sweeps in nonlinear astrometry like RAJ/DECJ, which the engine evaluates
    natively — those are exact but not linear, so we test DM, a true linear
    dispersion term.)"""
    ctx = joint_setup["ctx"]
    lin = "DM"
    assert lin in ctx.partition.linear_sampled
    col = ctx.partition.idx_sampled[list(ctx.sampled).index(lin)]
    nfit = len(ctx.partition.fitpars)

    def resid(scale):
        full = np.zeros(nfit)
        full[col] = scale * 1e-6
        return np.asarray(ctx.engine.residual_delta_jax(jnp.asarray(full)))

    assert np.allclose(resid(2.0), 2.0 * resid(1.0), rtol=1e-9, atol=1e-12)


def test_nuts_recovers_reference_and_writes_dynamic_manifest(joint_setup):
    ctx, jm = joint_setup["ctx"], joint_setup["jm"]
    mcmc = N.nuts(jm, ctx, num_warmup=120, num_samples=120, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    df = jm.to_df(mcmc.get_samples())

    # F0 posterior brackets the reference; delta actually moves (not float-stuck).
    f0_ref = 61.485476  # NGC6440E F0
    f0 = np.asarray(df[f"{ctx.name_stem}_F0_theta_native"])
    assert abs(f0.mean() - f0_ref) < 1e-3
    assert np.asarray(df[f"{ctx.name_stem}_F0_delta"]).std() > 0.0
    num = df.select_dtypes("number").to_numpy()
    assert np.all(np.isfinite(num))

    # Dynamic run manifest: latent is NOT decodable (§6.6).
    manifest = N.joint_run_manifest(
        ctx,
        jm.transport,
        likelihood="discovery",
        sampler="numpyro-nuts",
    )
    assert manifest.latent_decodable is False
    assert manifest.transport["kind"] == "dynamic_transport"
    assert manifest.transport["structure"]["dimension"] == jm.transport.dimension
