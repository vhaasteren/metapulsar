"""flexfit — fast, flexible-``Phi`` timing + Gaussian-process quick-look fits.

A deterministic empirical-Bayes fit over a joint timing-and-GP reduced-rank
basis, following the Enterprise ``T``/``Phi`` formulation. It reconstructs
quick-look waveforms (red noise, DM, chromatic, ...) and timing estimates for
diagnostics and initialization — labelled *quick-look empirical Bayes*, not a
substitute for production posterior inference.

Home: ``pylk.flexfit`` (incubated under MetaPulsar). The numerical core
depends only on NumPy/SciPy;
Discovery, Enterprise, and ``nltiming`` are reached through
``pylk.flexfit.adapters``.

Typical use::

    from pylk.flexfit import fastfit
    from pylk.flexfit.adapters import discovery as dx, nltiming as nx

    ctx = ntm.for_pulsar(pulsar)
    nx.sign_check(ctx)                       # verify the timing convention
    timing = nx.timing_model(ctx, marginalize_all=True)
    blocks = [dx.red_noise_block(pulsar, components=30),
              dx.dm_noise_block(pulsar, components=30)]
    noise = dx.white_noise(pulsar, noisedict)

    fit = fastfit(noise=noise, blocks=blocks, timing=timing, n_sweeps=3)
    whitened = fit.whitened_residuals()          # residuals minus red+DM waveforms
    red_wave = fit.waveform("red")
"""

from __future__ import annotations

from .basis import (
    INITIAL_TIMING_VARIANCE,
    AssembledModel,
    BasisBlock,
    VarianceGroup,
    assemble,
    column_rms_scale,
    fourier_pair_groups,
    per_column_groups,
    rho_bounds_from_rms,
)
from .fastfit import FastFitResult, fastfit
from .flexible_phi import (
    FlexiblePhiResult,
    bounded_variance_update,
    conditional_moments,
    solve_flexible_phi,
)
from .noise import DiagonalNoise, NoiseOperator, ShermanMorrisonNoise
from .projection import SpectrumProjection, project_spectrum, spectrum_objective
from .timing import TimingModel
from .whitenoise import WhiteNoiseResult, expected_squared_residuals, fit_white_noise

__all__ = [
    # orchestration
    "fastfit",
    "FastFitResult",
    # core solve
    "solve_flexible_phi",
    "FlexiblePhiResult",
    "conditional_moments",
    "bounded_variance_update",
    # containers
    "BasisBlock",
    "VarianceGroup",
    "AssembledModel",
    "assemble",
    "INITIAL_TIMING_VARIANCE",
    # group helpers
    "fourier_pair_groups",
    "per_column_groups",
    "rho_bounds_from_rms",
    "column_rms_scale",
    # noise
    "NoiseOperator",
    "DiagonalNoise",
    "ShermanMorrisonNoise",
    # white-noise estimation
    "fit_white_noise",
    "WhiteNoiseResult",
    "expected_squared_residuals",
    # projection
    "project_spectrum",
    "spectrum_objective",
    "SpectrumProjection",
    # timing interface
    "TimingModel",
]
