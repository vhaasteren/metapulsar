"""Enterprise likelihood-frontend adapter for nonlinear timing.

This module wires ``NonLinearTimingModel`` into Enterprise's signal graph:
a deterministic nonlinear delay for numerically sampled fit parameters and an optional
``TimingModel`` GP basis for analytically marginalized linear nuisances.

Priors
------
Sampled-parameter priors come from the bound ``ParameterSpace`` and are
evaluated through Enterprise ``UserParameter`` hooks that call
``PriorBijector.logprior_physical`` / ``ParameterSpace.logprior_coord``,
including the PIT Jacobian for bounded families (``uniform``, truncated
normal, etc.).

With ``prior_policy="fallback"``, unresolved sampled priors use the reference-stack
*cheat* prior convention—not Gaussians at the WLS scale. Each axis is a flat
``uniform`` on ``[center ± cheat_prior_scale · σ]`` in delta space
(``center`` = par-file reference, ``σ`` = par-file uncertainty with WLS
fallback), clipped to ``native_physical_bounds`` (e.g. ``ECC ∈ [0, 1]``,
``M2 ≥ 0``). Over the typical posterior support these boxes are
effectively flat. The whitening/standardized coordinate map is for sampler
preconditioning only and does not alter the physical prior density.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from metapulsar.timing.bijectors import PriorBijector
from metapulsar.timing.partition import PartitionResult, resolve_partition
from metapulsar.timing.space import default_coord_for_transform


def _resolve_partition(host, partition_spec) -> PartitionResult:
    if isinstance(partition_spec, PartitionResult):
        return partition_spec
    if callable(partition_spec):
        resolved = partition_spec(host)
        if not isinstance(resolved, PartitionResult):
            raise TypeError("partition_spec(host) must return PartitionResult")
        return resolved
    return resolve_partition(host, analytically_marginalize=partition_spec)


def _coord_from_transform(transform: str) -> str:
    if transform == "standardized":
        return "standardized"
    return default_coord_for_transform(transform)


def _get_backend(host, backend_name: str, backend_kwargs: dict | None = None):
    kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
    return host.timing_backend(backend_name, **kwargs)


def _axis_bijector(*, space, idx: int) -> PriorBijector:
    return PriorBijector(
        names=(space.names[idx],),
        priors=(space.prior_bijector.priors[idx],),
    )


def _scalar_user_parameter(*, space, coord: str, idx: int):
    from enterprise.signals import parameter

    axis = _axis_bijector(space=space, idx=idx)

    def _logprior(value):
        q = np.asarray([value], dtype=float)
        if coord == "delta":
            return float(axis.logprior_physical(q, np))
        if coord == "z":
            delta = axis.delta_from_z(q, np)
            return float(
                axis.logprior_physical(delta, np) + axis.logabsdet_delta_from_z(q, np)
            )
        if coord == "standardized":
            z = np.asarray([space.linear.C[idx, idx] * value + space.linear.z0[idx]])
            delta = axis.delta_from_z(z, np)
            return float(
                axis.logprior_physical(delta, np)
                + axis.logabsdet_delta_from_z(z, np)
                + np.log(space.linear.C[idx, idx])
            )
        raise ValueError(f"Scalar Enterprise parameters do not support coord={coord!r}")

    def _ppf(u):
        cube = np.asarray([u], dtype=float)
        delta = axis.delta_from_u(cube, np)
        if coord == "delta":
            return float(delta[0])
        if coord == "z":
            return float(axis.z_from_delta(delta, np)[0])
        if coord == "standardized":
            z = float(axis.z_from_delta(delta, np)[0])
            return float((z - space.linear.z0[idx]) / space.linear.C[idx, idx])
        raise ValueError(f"Scalar Enterprise parameters do not support coord={coord!r}")

    return parameter.UserParameter(
        logprior=parameter.Function(_logprior),
        ppf=parameter.Function(_ppf),
    )


def _vector_user_parameter(*, space):
    from enterprise.signals import parameter

    def _logprior(value):
        q = np.asarray(value, dtype=float)
        return float(space.logprior_coord(q, np, coord="x"))

    def _ppf(u):
        cube = np.asarray(u, dtype=float)
        return np.asarray(space.coord_from_cube(cube, np, coord="x"), dtype=float)

    return parameter.UserParameter(
        logprior=parameter.Function(_logprior),
        ppf=parameter.Function(_ppf),
        size=space.ndim,
    )


def _validate_kwarg_name(name: str) -> None:
    if not name.isidentifier():
        raise ValueError(
            f"Fit parameter {name!r} cannot be used as an Enterprise waveform keyword"
        )


def _explicit_scalar_delay_function(sampled_names: tuple[str, ...], evaluator):
    for fitpar in sampled_names:
        _validate_kwarg_name(fitpar)
    signature = ", ".join(f"{fitpar}=None" for fitpar in sampled_names)
    call = ", ".join(f"{fitpar}={fitpar}" for fitpar in sampled_names)
    source = (
        f"def _delay_body(toas, psr=None, mask=None, {signature}):\n"
        f"    return _evaluator({call})\n"
    )
    namespace = {"_evaluator": evaluator}
    exec(source, namespace)
    return namespace["_delay_body"]


def _make_waveform(
    *,
    space_fn: Callable,
    backend_name: str,
    backend_kwargs: dict | None,
    partition_spec,
    coord: str,
):
    from enterprise.signals import parameter

    def waveform(signal_name, psr=None):
        if psr is None:
            raise ValueError("enterprise waveform requires psr binding")
        space = space_fn(psr)
        partition = _resolve_partition(psr, partition_spec)
        sampled_names = tuple(partition.sampled)
        sampled_indices = tuple(partition.idx_sampled)
        backend = _get_backend(psr, backend_name, backend_kwargs=backend_kwargs)
        ndim = len(partition.fitpars)

        if coord in {"delta", "z", "standardized"}:

            def _evaluate(**coord_values):
                q = np.asarray(
                    [coord_values[param] for param in sampled_names],
                    dtype=float,
                )
                space_coord = "x" if coord == "standardized" else coord
                delta_sampled = np.asarray(
                    space.delta_from_coord(q, np, coord=space_coord)
                )
                full_delta = np.zeros((ndim,), dtype=float)
                for i, col in enumerate(sampled_indices):
                    full_delta[col] = delta_sampled[i]
                return -backend.residual_delta(full_delta)

            delay_body = _explicit_scalar_delay_function(sampled_names, _evaluate)
            kwargs = {
                param: _scalar_user_parameter(
                    space=space,
                    coord=coord,
                    idx=i,
                )
                for i, param in enumerate(sampled_names)
            }
            return parameter.Function(delay_body, **kwargs)(signal_name, psr=psr)

        if coord != "x":
            raise ValueError(f"Unsupported enterprise timing coord: {coord}")

        def _delay_body(toas, psr=None, mask=None, x=None):
            q = np.asarray(x, dtype=float)
            delta_sampled = np.asarray(space.delta_from_coord(q, np, coord="x"))
            full_delta = np.zeros((ndim,), dtype=float)
            for i, col in enumerate(sampled_indices):
                full_delta[col] = delta_sampled[i]
            return -backend.residual_delta(full_delta)

        kwargs = {"x": _vector_user_parameter(space=space)}
        return parameter.Function(_delay_body, **kwargs)(signal_name, psr=psr)

    return waveform


def _make_marginalizing_signal(*, partition_spec, name: str):
    from enterprise.signals import gp_signals, signal_base

    class MarginalizingTimingModel(
        signal_base.Signal, metaclass=signal_base.MetaSignal
    ):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = f"{name}_timingmodel"

        def __init__(self, psr):
            super().__init__(psr)
            partition = _resolve_partition(psr, partition_spec)
            base = gp_signals.TimingModel(
                name=f"{name}_timingmodel",
                idx_exclude=partition.idx_sampled,
            )
            self._inner = base(psr)
            self.name = self._inner.name
            self._params = self._inner._params
            self.basis_params = list(self._inner.basis_params)
            self.prior_params = list(getattr(self._inner, "prior_params", []))
            self.delay_params = list(getattr(self._inner, "delay_params", []))
            self.basis_combine = getattr(self._inner, "basis_combine", False)

        def get_basis(self, params=None):
            return self._inner.get_basis(params=params)

        def get_phi(self, params):
            return self._inner.get_phi(params)

        def get_phiinv(self, params):
            return self._inner.get_phiinv(params)

        def get_delay(self, params):
            return self._inner.get_delay(params)

        def get_logsignalprior(self, params):
            return self._inner.get_logsignalprior(params)

        def set_default_params(self, params):
            self._inner.set_default_params(params)

    return MarginalizingTimingModel


def enterprise_signal(
    *,
    space_fn,
    backend_name: str,
    backend_kwargs: dict | None = None,
    partition_spec,
    name: str,
    transform: str,
):
    """Return a deferred Enterprise signal with deterministic delay + timing GP.

    Parameters
    ----------
    space_fn
        Callable ``host -> ParameterSpace`` (typically ``NonLinearTimingModel.space``).
        Supplies per-axis priors and the whitening/standardized linear map used
        by ``UserParameter`` log-prior and PPF hooks.
    backend_name
        Host timing backend identifier (``"jug"``, ``"pint"``, ``"tempo2"``).
    backend_kwargs
        Optional kwargs forwarded to ``host.timing_backend`` (e.g.
        ``jug_compatibility``).
    partition_spec
        ``PartitionResult``, ``analytically_marginalize`` spec, or callable
        ``host -> PartitionResult``.
    name
        Enterprise signal / component name prefix.
    transform
        ``NonLinearTimingModel`` transform mode: ``"none"``, ``"standardized"``,
        or ``"whitening"``. Selects the Enterprise sampling coordinate
        (``delta``, per-axis standardized, or joint ``x``).

    Returns
    -------
    type
        ``MetaSignal`` subclass that materializes on ``(psr)`` into either
        ``Deterministic(delay)``, ``TimingModel`` GP, or their sum.

    Notes
    -----
    Delay parameters are mapped from the sampling coordinate back to native
    ``delta_theta`` via ``space.delta_from_coord`` before calling
    ``backend.residual_delta``. Prior terms follow ``space`` exactly, so
    fallback cheat priors are the wide uniform boxes described in the module
    docstring—not informative Gaussians tied to the WLS covariance.
    """
    from enterprise.signals import deterministic_signals

    coord = _coord_from_transform(transform)
    waveform = _make_waveform(
        space_fn=space_fn,
        backend_name=backend_name,
        backend_kwargs=backend_kwargs,
        partition_spec=partition_spec,
        coord=coord,
    )
    delay_signal = deterministic_signals.Deterministic(waveform, name=name)
    timing_model = _make_marginalizing_signal(partition_spec=partition_spec, name=name)
    return delay_signal + timing_model
