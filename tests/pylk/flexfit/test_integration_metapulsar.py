"""End-to-end glue test on a real MetaPulsar pulsar + real timing backend.

Builds a real (tempo2-free) ``MetaPulsar`` from a mock libstempo pulsar and
drives the full flexfit path the paper notebook uses: the timing sign check
(real ``backend.residual_delta`` vs real ``pulsar.Mmat``), the marginalize-all
timing block, Discovery red/DM blocks from the pulsar's real TOAs/frequencies,
white noise, ``fastfit``, and the fixed-spectrum MLE reconstruction.

The nltiming ``TimingSpec.bind`` cheat-prior WLS is rank-deficient on
mock timing models (a mock-data property, not a flexfit one), so the ctx is
assembled directly from the pulsar's real backend — exactly the ``TimingSignal``
surface the adapter consumes. Skips unless MetaPulsar + nltiming + Discovery are
importable (devcontainer).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("discovery")
pytest.importorskip("nltiming")
pytest.importorskip("metapulsar")

from metapulsar.metapulsar import MetaPulsar  # noqa: E402
from metapulsar.mockpulsar import (  # noqa: E402
    create_mock_libstempo,
    write_mock_pta_files,
)
from nltiming.bijectors import PriorBijector  # noqa: E402
from nltiming.space import ParameterSpace  # noqa: E402

from pylk.flexfit import fastfit  # noqa: E402
from pylk.flexfit.adapters import discovery as fx_dx, nltiming as fx_nx  # noqa: E402

ENGINES = {"tempo2": "libstempo", "pint": "jug"}


@pytest.fixture(scope="module")
def pulsar(tmp_path_factory):
    psr = create_mock_libstempo(
        n_toas=200, name="J1857+0943", telescope="pta_a", seed=11
    )
    pulsars = {"pta_a": psr}
    return MetaPulsar(
        pulsars,
        combination_strategy="per_pta",
        pta_files=write_mock_pta_files(pulsars, tmp_path_factory.mktemp("pta_files")),
    )


def _plan(fitpars, sampled):
    """Duck-typed TimingParameterPlan for the flexfit nltiming adapter."""
    fitpars = tuple(fitpars)
    sampled = tuple(sampled)
    idx = {name: i for i, name in enumerate(fitpars)}
    sampled_idx = tuple(idx[name] for name in sampled)
    marg_idx = tuple(i for i, name in enumerate(fitpars) if name not in sampled)
    return SimpleNamespace(
        sampled=sampled,
        idx_sampled=sampled_idx,
        idx_analytically_marginalized=marg_idx,
        fitpars=fitpars,
    )


def _binding(pulsar, sample):
    """Assemble a TimingSignal-shaped object from the pulsar's real backend."""
    backend = pulsar.timing_engine(ENGINES)
    fitpars = tuple(pulsar.fitpars)
    sampled = tuple(sample) if sample else ()
    plan = _plan(fitpars, sampled)
    ref = backend.reference_theta_exact()
    if sampled:
        pb = PriorBijector.from_normal(
            sampled, means=np.zeros(len(sampled)), stds=np.ones(len(sampled))
        )
        space = ParameterSpace.build(
            {name: ref[name] for name in sampled},
            prior_bijector=pb,
            static_layer="identity",
        )
    else:
        space = None
    return SimpleNamespace(
        pulsar=pulsar,
        engine=backend,
        space=space,
        plan=plan,
        design_matrix=np.asarray(pulsar.Mmat, dtype=float),
    )


def _fitpars_by_base(pulsar, *bases: str) -> list[str]:
    """Resolve PTA-suffixed fitpars by base name (e.g. F0 -> F0_pta_a)."""
    out = []
    for base in bases:
        matches = [p for p in pulsar.fitpars if p == base or p.startswith(f"{base}_")]
        if not matches:
            raise KeyError(
                f"no fitpar matching base {base!r} in {list(pulsar.fitpars)}"
            )
        out.append(matches[0])
    return out


def test_sign_check_relates_backend_to_design(pulsar):
    sample = _fitpars_by_base(pulsar, "F0", "F1")
    ctx = _binding(pulsar, sample)
    errs = fx_nx.sign_check(ctx, parameters=tuple(ctx.plan.sampled[:2]))
    # Real backend.residual_delta finite differences must match pulsar.Mmat columns.
    assert max(errs.values()) < 1e-3


def test_discovery_blocks_from_real_host(pulsar):
    red = fx_dx.red_noise_block(pulsar, components=10, name="red")
    dm = fx_dx.dm_noise_block(pulsar, components=10, name="dm")
    n = len(pulsar.residuals)
    assert red.matrix.shape == (n, 20)
    assert dm.matrix.shape == (n, 20)
    # DM basis scales with observing frequency; red does not.
    assert not np.allclose(red.matrix, dm.matrix)


def test_full_notebook_path_reconstruction(pulsar):
    ctx = _binding(pulsar, None)  # marginalize all timing
    timing = fx_nx.timing_model(ctx, marginalize_all=True)
    assert timing.blocks()[0].matrix.shape == (
        len(pulsar.residuals),
        len(pulsar.fitpars),
    )

    white = fx_dx.white_noise_from_variance(
        np.asarray(pulsar.toaerrs, dtype=float) ** 2
    )
    free = [
        fx_dx.red_noise_block(pulsar, components=8, name="red"),
        fx_dx.dm_noise_block(pulsar, components=8, name="dm"),
    ]
    fit_fs = fastfit(
        noise=white,
        blocks=free,
        timing=fx_nx.timing_model(ctx, marginalize_all=True),
        n_sweeps=3,
    )
    assert set(fit_fs.block_names) == {"timing_marg", "red", "dm"}

    fixed = []
    for blk in free:
        proj = fx_dx.project_powerlaw(fit_fs, blk)
        assert proj.success
        builder = fx_dx.red_noise_block if blk.kind == "red" else fx_dx.dm_noise_block
        fixed.append(
            builder(
                pulsar,
                components=8,
                name=blk.name,
                log10_A=proj.values["log10_A"],
                gamma=proj.values["gamma"],
            )
        )
    fit = fastfit(
        noise=white,
        blocks=fixed,
        timing=fx_nx.timing_model(ctx, marginalize_all=True),
        n_sweeps=2,
    )
    residual = np.asarray(pulsar.residuals, dtype=float)
    subtracted = residual - fit.noise_waveform()
    assert subtracted.shape == residual.shape
    assert np.all(np.isfinite(subtracted))
    for name in ("red", "dm"):
        assert fit.waveform(name).shape == residual.shape


def test_sampled_timing_summary(pulsar):
    ctx = _binding(pulsar, _fitpars_by_base(pulsar, "F0"))
    white = fx_dx.white_noise_from_variance(
        np.asarray(pulsar.toaerrs, dtype=float) ** 2
    )
    timing = fx_nx.timing_model(ctx, sample_update_from_sweep=2)
    fit = fastfit(noise=white, blocks=[], timing=timing, n_sweeps=3)
    summary = fit.timing_summary
    assert "z" in summary and "physical" in summary
    assert len(summary["sampled_names"]) == len(ctx.plan.sampled)
