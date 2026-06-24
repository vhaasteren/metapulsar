"""Enterprise frontend adapter for nonlinear timing."""

from __future__ import annotations

from typing import Callable

import numpy as np

from metapulsar.timing.bijectors import PriorBijector
from metapulsar.timing.partition import PartitionResult, resolve_partition


def _resolve_partition(host, partition_spec) -> PartitionResult:
    if isinstance(partition_spec, PartitionResult):
        return partition_spec
    if callable(partition_spec):
        resolved = partition_spec(host)
        if not isinstance(resolved, PartitionResult):
            raise TypeError("partition_spec(host) must return PartitionResult")
        return resolved
    return resolve_partition(host, marginalize=partition_spec)


def _coord_from_transform(transform: str) -> str:
    if transform == "none":
        return "delta"
    if transform == "standardized":
        return "z"
    if transform == "whitening":
        return "x"
    raise ValueError(f"Unsupported transform: {transform}")


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
        raise ValueError(f"Scalar Enterprise parameters do not support coord={coord!r}")

    def _ppf(u):
        cube = np.asarray([u], dtype=float)
        delta = axis.delta_from_u(cube, np)
        if coord == "delta":
            return float(delta[0])
        if coord == "z":
            return float(axis.z_from_delta(delta, np)[0])
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

        if coord in {"delta", "z"}:

            def _evaluate(**coord_values):
                q = np.asarray(
                    [coord_values[param] for param in sampled_names],
                    dtype=float,
                )
                delta_sampled = np.asarray(space.delta_from_coord(q, np, coord=coord))
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
    """Return deferred Enterprise signal with deterministic delay + timing GP."""
    from enterprise.signals import deterministic_signals, signal_base

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

    class NonlinearTimingEnterpriseSignal(metaclass=signal_base.MetaSignal):
        signal_id = name
        signal_name = name
        signal_type = "nonlinear timing"

        def __new__(cls, psr):
            partition = _resolve_partition(psr, partition_spec)
            if partition.sampled and partition.idx_marginalized:
                return (delay_signal + timing_model)(psr)
            if partition.sampled:
                return delay_signal(psr)
            if partition.idx_marginalized:
                return timing_model(psr)
            raise ValueError(
                "enterprise_signal requires sampled or marginalized fitpars"
            )

    return NonlinearTimingEnterpriseSignal
