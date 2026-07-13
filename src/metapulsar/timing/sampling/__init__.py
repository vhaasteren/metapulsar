"""Sampler-facing glue over a ``TimingBinding``.

- ``numpyro``: NumPyro/Discovery model builder, NUTS setup, timing sites.
- ``ptmcmc``: PTMCMCSampler setup for Enterprise (or Discovery) likelihoods.

Modules import their sampler dependencies lazily; importing this package
requires neither numpyro nor PTMCMCSampler.
"""

from . import numpyro, ptmcmc

__all__ = ["numpyro", "ptmcmc"]
