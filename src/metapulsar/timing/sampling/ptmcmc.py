"""PTMCMCSampler glue for nonlinear timing bindings.

Handles the sampler-vector ↔ likelihood-parameter mapping for both timing
coordinate layouts (whitening joint site vs standardized scalar columns) and
records the on-disk chain layout for the artifact sidecar.

Works with an Enterprise ``PTA`` (``get_lnlikelihood``/``get_lnprior``); the
same ``eval_params`` mapping applies to Discovery likelihoods driven by
PTMCMC with fixed hyperparameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CHAIN_FILENAME = "chain_1.txt"


def eval_params(
    binding,
    vec: np.ndarray,
    *,
    fixed: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Map a timing-block sampler vector to a likelihood parameter dict.

    With ``transform="standardized"`` the vector entries are scalar
    standardized coordinates keyed by delay-key names (Enterprise scalar
    ``UserParameter`` layout); otherwise the whole vector is the joint
    coordinate site.
    """
    params: dict[str, Any] = {
        key: float(value)
        for key, value in dict(fixed or {}).items()
        if isinstance(value, (int, float))
    }
    x = np.asarray(vec, dtype=float).reshape(-1)
    if x.size != len(binding.sampled):
        raise ValueError(
            f"expected vector of length {len(binding.sampled)}, got {x.size}"
        )
    if not binding.sampled:
        return params
    if binding.model.transform == "standardized":
        for i, key in enumerate(binding.delay_keys):
            params[key] = float(x[i])
    else:
        params[binding.coord_site_name()] = x
    return params


def initial_point(binding) -> np.ndarray:
    """Reference initial point: zero in sampling coordinates."""
    return np.zeros(len(binding.sampled), dtype=float)


def initial_cov(binding, *, nsamples: int = 2000, seed: int = 0) -> np.ndarray:
    """Proposal covariance from the WLS covariance, in sampling coordinates.

    The default coordinate transforms rescale axes but do not rotate away
    cross-parameter correlations, so the posterior in sampling coordinates can
    be a narrow correlated ridge. Seeding PTMCMC's jump proposals with the
    whitened-least-squares covariance (mapped through the coordinate stack by
    sampling) makes the chain mix immediately instead of relying on long
    adaptation.
    """
    from ..whitening import schur_delta_wls

    ndim = len(binding.sampled)
    if ndim == 0:
        raise ValueError("binding has no sampled timing parameters")
    wls = schur_delta_wls(
        pulsar=binding.pulsar,
        partition=binding.partition,
        variance=np.asarray(binding.pulsar.toaerrs, dtype=float) ** 2,
        design_matrix=binding.design_matrix,
    )
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        np.zeros(ndim), np.asarray(wls.covariance, dtype=float), size=nsamples
    )
    coords = np.stack(
        [
            np.asarray(
                binding.space.coord_from_delta(delta, np, coord=binding.coord),
                dtype=float,
            )
            for delta in draws
        ]
    )
    cov = np.atleast_2d(np.cov(coords.T))
    # Guard against numerically singular proposals.
    cov[np.diag_indices_from(cov)] += 1e-12
    return cov


def timing_param_names(binding) -> tuple[str, ...]:
    """Sampler-visible timing parameter names in vector order."""
    if not binding.sampled:
        return tuple()
    if binding.model.transform == "standardized":
        return tuple(binding.delay_keys)
    site = binding.coord_site_name()
    return tuple(f"{site}_{i}" for i in range(len(binding.sampled)))


def chain_layout(
    binding,
    param_names: Sequence[str],
    *,
    chain_file: str = CHAIN_FILENAME,
) -> dict[str, Any]:
    """Sidecar ``chain_layout`` spec locating timing columns in a PTMCMC chain.

    ``param_names`` is the sampler's parameter-name order (for a
    timing-only run, ``timing_param_names(binding)``; for an Enterprise PTA
    with free noise, ``pta.param_names``).
    """
    names = list(param_names)
    columns = []
    for key in timing_param_names(binding):
        try:
            columns.append(names.index(key))
        except ValueError as exc:
            raise ValueError(
                f"timing parameter {key!r} not found in sampler param names"
            ) from exc
    return {"kind": "ptmcmc", "file": chain_file, "columns": columns}


def sampler(
    pta,
    binding,
    outdir: str | Path,
    *,
    fixed: Mapping[str, float] | None = None,
    cov: np.ndarray | None = None,
    verbose: bool = True,
    **ptmcmc_kwargs: Any,
):
    """Configured ``PTSampler`` over the timing block with fixed non-timing params.

    Returns the sampler; run it with
    ``pts.sample(p0=initial_point(binding), Niter=..., burn=...)``. The chain
    lands in ``outdir/chain_1.txt`` (PTMCMC layout: ndim columns + lnpost,
    lnlik, accept, pt-accept). The default proposal covariance is
    :func:`initial_cov` — the WLS covariance mapped into sampling coordinates —
    which captures cross-parameter correlations the coordinate transform does
    not remove.
    """
    from PTMCMCSampler.PTMCMCSampler import PTSampler

    ndim = len(binding.sampled)
    if ndim == 0:
        raise ValueError("binding has no sampled timing parameters")
    if cov is None:
        cov = initial_cov(binding)

    def _params(vec: np.ndarray) -> dict[str, Any]:
        return eval_params(binding, vec, fixed=fixed)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return PTSampler(
        ndim,
        lambda vec: pta.get_lnlikelihood(_params(vec)),
        lambda vec: pta.get_lnprior(_params(vec)),
        cov,
        outDir=str(outdir),
        verbose=verbose,
        **ptmcmc_kwargs,
    )
