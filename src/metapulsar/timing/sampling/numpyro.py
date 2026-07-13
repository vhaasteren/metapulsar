"""NumPyro sampler glue for nonlinear timing bindings.

Owns everything NumPyro-specific: the joint timing sample site and its prior
factor (:func:`contribute_timing`), physical deterministic sites
(:func:`record_physical`), the standard Discovery model closure
(:func:`model`), and NUTS setup with init-at-reference (:func:`nuts`).

All functions take a :class:`~metapulsar.timing.nonlinear_timing_model.TimingBinding`
(from ``NonLinearTimingModel.bind``); none of this leaks into the model config.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


def ensure_x64() -> None:
    """Enable JAX float64 and fail loudly if it cannot take effect.

    Timing deltas need float64; silently sampling in float32 produces wrong
    posteriors. Call this before building likelihoods; it is invoked by
    :func:`nuts` as a safety net.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if jnp.zeros(1).dtype != jnp.float64:
        raise RuntimeError(
            "JAX x64 mode could not be enabled (arrays still default to "
            "float32). Set JAX_ENABLE_X64=1 or call ensure_x64() before any "
            "JAX arrays are created."
        )


def contribute_timing(
    binding,
    params: Mapping[str, Any],
    *,
    coord: str | None = None,
) -> Mapping[str, Any]:
    """Sample the joint timing site and inject per-parameter delta values.

    Adds one ``ImproperUniform`` sample site (``binding.coord_site_name()``)
    plus a ``numpyro.factor`` carrying the physical prior, and returns
    ``params`` extended with backend-facing delta keys
    (``{pulsar}_{model}_{fitpar}``).
    """
    sampled = binding.partition.sampled
    if not sampled:
        return params

    coord = binding.coord if coord is None else coord
    if coord not in {"delta", "z", "x"}:
        raise ValueError(f"Unsupported coord: {coord}")

    import jax.numpy as jnp
    import numpyro
    from numpyro import distributions as dist
    from numpyro.distributions import constraints

    space = binding.space
    site_name = binding.coord_site_name(coord)
    q = numpyro.sample(
        site_name,
        dist.ImproperUniform(constraints.real, (), (len(sampled),)),
    )
    numpyro.factor(f"{site_name}_logprior", space.logprior_coord(q, jnp, coord=coord))
    delta = space.delta_from_coord(q, jnp, coord=coord)

    out = dict(params)
    for i, name in enumerate(sampled):
        out[f"{binding.prefix}_{name}"] = delta[i]
    return out


def record_physical(
    binding,
    params: Mapping[str, Any],
    *,
    scope: str = "timing",
    coord: str | None = None,
) -> None:
    """Emit ``{prefix}_{fitpar}_theta_{native,display}`` deterministic sites.

    Operates on concrete (un-traced) parameter values — posterior draws or a
    reconstructed parameter dict — not inside a traced model function.
    """
    coord_was_explicit = coord is not None
    if coord is not None and coord not in {"delta", "z", "x"}:
        raise ValueError("coord must be one of {'delta', 'z', 'x'}")
    if coord is None:
        coord = binding.coord

    if scope == "all":
        raise NotImplementedError("scope='all' is deferred")
    if scope != "timing":
        raise ValueError("scope must be one of {'timing', 'all'}")

    sampled = binding.partition.sampled
    if not sampled:
        return

    import numpyro

    delta = binding.delta_from_params(
        params,
        coord=coord,
        coord_explicit=coord_was_explicit,
    )
    prefix = binding.prefix
    theta_native = binding.space.to_physical(
        delta[None, :], units="native", coord="delta"
    )
    theta_display = binding.space.to_physical(
        delta[None, :], units="display", coord="delta"
    )
    for name in sampled:
        numpyro.deterministic(f"{prefix}_{name}_theta_native", theta_native[name][0])
        numpyro.deterministic(f"{prefix}_{name}_theta_display", theta_display[name][0])


def model(
    likelihood,
    binding,
    *,
    priors: Mapping[str, Any] | None = None,
    fixed: Mapping[str, float] | None = None,
) -> Callable[[], None]:
    """Build the standard NumPyro model for a Discovery likelihood.

    Non-timing likelihood parameters are sampled from uniform priors resolved
    via ``discovery.prior.getprior_uniform(par, priors)``, unless pinned in
    ``fixed``. Timing parameters enter through :func:`contribute_timing`.

    Args:
        likelihood: Discovery ``PulsarLikelihood`` (or anything exposing
            ``logL`` with a ``params`` attribute).
        binding: ``TimingBinding`` for the pulsar in the likelihood.
        priors: prior overrides / noise dictionary for non-timing parameters.
        fixed: parameter values held constant (not sampled).
    """
    import numpyro
    from numpyro import distributions as dist

    fixed_params = {
        key: float(value)
        for key, value in dict(fixed or {}).items()
        if isinstance(value, (int, float))
    }
    priordict = dict(priors or {})
    free = [
        par
        for par in binding.non_timing_params(likelihood.logL.params)
        if par not in fixed_params
    ]
    if free:
        from discovery import prior as ds_prior

        bounds = {par: tuple(ds_prior.getprior_uniform(par, priordict)) for par in free}

    def nlt_model() -> None:
        params = dict(fixed_params)
        for par in free:
            params[par] = numpyro.sample(par, dist.Uniform(*bounds[par]))
        numpyro.factor("ll", likelihood.logL(contribute_timing(binding, params)))

    return nlt_model


def timing_init_values(binding) -> dict[str, Any]:
    """Init-at-reference values for the joint timing site (zero coordinates)."""
    ndim = len(binding.sampled)
    if not ndim:
        return {}
    import jax.numpy as jnp

    return {binding.coord_site_name(): jnp.zeros((ndim,), dtype=jnp.float64)}


def nuts(
    model_fn: Callable[[], None],
    binding,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    dense_mass: bool = True,
    target_accept: float = 0.65,
    **mcmc_kwargs: Any,
) -> Any:
    """Configured NUTS ``MCMC`` with x64 enforced and init-at-reference.

    Timing coordinates initialize at the backend reference (zero in sampling
    coordinates); other sites fall back to NumPyro's default init. Run with
    ``mcmc.run(jax.random.PRNGKey(seed))``.
    """
    ensure_x64()
    from numpyro.infer import MCMC, NUTS, init_to_value

    kernel = NUTS(
        model_fn,
        dense_mass=dense_mass,
        target_accept_prob=target_accept,
        init_strategy=init_to_value(values=timing_init_values(binding)),
    )
    return MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        **mcmc_kwargs,
    )


def timing_draws(samples: Mapping[str, Any], binding) -> np.ndarray:
    """Extract flattened joint-site draws ``(n_draws, ndim)`` from MCMC samples."""
    x = np.asarray(samples[binding.coord_site_name()], dtype=float)
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    return x
