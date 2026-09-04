"""Build flexfit GP blocks and white noise from Enterprise primitives.

**Status: planned / not implemented.**

This adapter is the Enterprise counterpart of
:mod:`pylk.flexfit.adapters.discovery`. The numerical core already speaks
Enterprise's ``T``/``Phi`` language; this module only needs to supply:

* Fourier design matrices — ``enterprise.signals.gp_bases.createfourierdesignmatrix_red``
  (and ``_dm`` / ``_chromatic``), interleaved sin/cos, ``f_k = k / Tspan``;
* power-law and free-spectrum diagonals — ``enterprise.signals.gp_priors.powerlaw``,
  ``free_spectrum``;
* white-noise diagonals (EFAC/EQUAD; t2equad vs tnequad conventions);
* optional basis-ECORR epoch-averaging blocks;
* projection of free-spectrum second moments onto a power law via
  :func:`pylk.flexfit.project_spectrum`.

Until this lands, build blocks by hand from Enterprise arrays and pass them to
:func:`pylk.flexfit.fastfit`, or use the Discovery adapter when Discovery is
available.
"""

from __future__ import annotations

__all__: list[str] = []


def red_noise_block(*args, **kwargs):  # noqa: ANN001, ANN003
    """Not implemented — see module docstring."""
    raise NotImplementedError(
        "pylk.flexfit.adapters.enterprise is planned but not yet implemented. "
        "Use pylk.flexfit.adapters.discovery, or assemble BasisBlocks from "
        "enterprise.signals.gp_bases arrays and call pylk.flexfit.fastfit."
    )


def dm_noise_block(*args, **kwargs):  # noqa: ANN001, ANN003
    """Not implemented — see module docstring."""
    raise NotImplementedError(
        "pylk.flexfit.adapters.enterprise is planned but not yet implemented."
    )


def white_noise(*args, **kwargs):  # noqa: ANN001, ANN003
    """Not implemented — see module docstring."""
    raise NotImplementedError(
        "pylk.flexfit.adapters.enterprise is planned but not yet implemented."
    )


def project_powerlaw(*args, **kwargs):  # noqa: ANN001, ANN003
    """Not implemented — see module docstring."""
    raise NotImplementedError(
        "pylk.flexfit.adapters.enterprise is planned but not yet implemented."
    )
