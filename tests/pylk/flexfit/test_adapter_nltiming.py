"""Tests for the nltiming timing-block adapter.

These use real ``nltiming`` coordinate machinery (``ParameterSpace``,
``PriorBijector``) with a trivial *linear* timing backend, so the adapter's
``J_z`` assembly, the mandated finite-difference sign check, and the
marginalize-all reconstruction are validated without a heavy pulsar pulsar. Skips
when nltiming is unavailable (pulsar runs); intended for the devcontainer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("nltiming")

from nltiming.bijectors import PriorBijector  # noqa: E402
from nltiming.space import ParameterSpace  # noqa: E402

from pylk.flexfit import BasisBlock, DiagonalNoise, fastfit  # noqa: E402
from pylk.flexfit.adapters import nltiming as nx  # noqa: E402


def _linear_binding(names, design, residuals, *, sampled, sign=1.0, prior="normal"):
    """Build a duck-typed nltiming ctx around a linear backend."""
    n_fit = len(names)
    idx = {name: i for i, name in enumerate(names)}
    # nltiming builds `space` over the *sampled* parameters only; mirror that.
    space_names = tuple(sampled) if sampled else tuple(names)
    n_space = len(space_names)
    if prior == "normal":
        pb = PriorBijector.from_normal(
            space_names, means=np.zeros(n_space), stds=np.ones(n_space)
        )
    else:
        pb = PriorBijector.from_uniform(
            space_names, lowers=-np.ones(n_space), uppers=np.ones(n_space)
        )
    space = ParameterSpace.build(
        {name: "0.0" for name in space_names},
        prior_bijector=pb,
        static_layer="identity",
    )
    backend = SimpleNamespace(
        residual_delta=lambda delta: sign * (np.asarray(design, dtype=float) @ delta)
    )
    sampled_idx = tuple(idx[name] for name in sampled)
    marg_idx = tuple(i for i in range(n_fit) if names[i] not in sampled)
    plan = SimpleNamespace(
        sampled=tuple(sampled),
        idx_sampled=sampled_idx,
        idx_analytically_marginalized=marg_idx,
        fitpars=tuple(names),
    )
    pulsar = SimpleNamespace(
        name="Jtest",
        residuals=np.asarray(residuals, dtype=float),
        fitpars=tuple(names),
    )
    return SimpleNamespace(
        pulsar=pulsar,
        engine=backend,
        space=space,
        plan=plan,
        design_matrix=np.asarray(design, dtype=float),
    )


def test_sign_check_passes_for_consistent_design():
    rng = np.random.default_rng(0)
    names = ["F0", "F1", "DM"]
    design = rng.standard_normal((50, 3))
    ctx = _linear_binding(names, design, np.zeros(50), sampled=("F0", "F1"))
    errors = nx.sign_check(ctx)
    assert max(errors.values()) < 1e-6


def test_sign_check_raises_for_flipped_convention():
    rng = np.random.default_rng(1)
    names = ["F0", "F1"]
    design = rng.standard_normal((40, 2))
    # backend residual_delta has the opposite sign of the stated design.
    ctx = _linear_binding(names, design, np.zeros(40), sampled=("F0", "F1"), sign=-1.0)
    with pytest.raises(ValueError, match="sign/scale check failed"):
        nx.sign_check(ctx)


def test_linear_jz_equals_design_for_normal_prior():
    rng = np.random.default_rng(2)
    names = ["F0", "F1", "DM"]
    design = rng.standard_normal((30, 3))
    ctx = _linear_binding(names, design, np.zeros(30), sampled=("F0", "DM"))
    model = nx.timing_model(ctx)
    sampled_block = next(b for b in model.blocks() if b.name == "timing")
    # normal(0,1) prior => d delta / d z = 1 => J_z == design columns.
    np.testing.assert_allclose(sampled_block.matrix, design[:, [0, 2]])


def test_marginalize_all_uses_full_normalized_design():
    from nltiming.whitening import normalized_basis

    rng = np.random.default_rng(3)
    names = ["A", "B", "C", "D"]
    design = rng.standard_normal((60, 4))
    ctx = _linear_binding(names, design, rng.standard_normal(60), sampled=())
    model = nx.timing_model(ctx, marginalize_all=True)
    blocks = model.blocks()
    assert len(blocks) == 1
    assert blocks[0].name == "timing_marg"
    np.testing.assert_allclose(blocks[0].matrix, normalized_basis(design))
    assert model.sampled_block is None


def test_fastfit_marginalizes_timing_and_recovers_red():
    rng = np.random.default_rng(4)
    n_obs = 300
    names = ["Offset", "F0", "F1"]
    t = np.linspace(-1, 1, n_obs)
    design = np.column_stack([np.ones(n_obs), t, t**2])
    # A red-like basis (independent of Discovery).
    n_freq = 8
    F = rng.standard_normal((n_obs, 2 * n_freq))
    red_wave = F @ (0.3 * rng.standard_normal(2 * n_freq))
    timing_signal = design @ np.array([2.0, -1.5, 0.7])
    white = 0.02 * rng.standard_normal(n_obs)
    residuals = timing_signal + red_wave + white

    ctx = _linear_binding(names, design, residuals, sampled=())
    timing = nx.timing_model(ctx, marginalize_all=True)

    from pylk.flexfit import fourier_pair_groups

    groups = fourier_pair_groups(
        F, prefix="red", n_freq=n_freq, sigma_min=1e-6, sigma_max=1e2
    )
    red_block = BasisBlock(
        "red", F, tuple(f"c{i}" for i in range(2 * n_freq)), groups, kind="red"
    )
    noise = DiagonalNoise(np.full(n_obs, 0.02**2))

    fit = fastfit(noise=noise, blocks=[red_block], timing=timing, n_sweeps=4)
    whitened = fit.whitened_residuals() - fit.waveform("timing_marg")
    # Timing + red waveforms together explain the residuals down to the white floor.
    assert np.sqrt(np.mean(whitened**2)) < 0.05
    assert np.corrcoef(fit.waveform("red"), red_wave)[0, 1] > 0.9


def test_fastfit_sampled_timing_learns_variance_from_sweep_two():
    rng = np.random.default_rng(5)
    n_obs = 250
    names = ["F0", "F1"]
    t = np.linspace(-1, 1, n_obs)
    design = np.column_stack([t, t**2])
    residuals = design @ np.array([1.0, -0.5]) + 0.01 * rng.standard_normal(n_obs)
    ctx = _linear_binding(names, design, residuals, sampled=("F0", "F1"))

    timing = nx.timing_model(ctx, sample_update_from_sweep=2, sample_sigma_max=1e2)
    noise = DiagonalNoise(np.full(n_obs, 0.01**2))
    # No GP blocks: pure timing fit. First-sweep phi is 1e40 for both.
    fit = fastfit(noise=noise, blocks=[], timing=timing, n_sweeps=3)
    # After sweep 2 the timing variances were updated away from 1e40.
    for name in ("timing_F0", "timing_F1"):
        assert fit.group_variances[name] < 1e40
    summ = fit.timing_summary
    assert summ["sampled_names"] == ("F0", "F1")
    assert summ["z"].shape == (2,)
