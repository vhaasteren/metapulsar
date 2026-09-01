"""Per-PTA vela-jax timing engine.

vela-jax evaluates Vela.jl's component chain as a JAX function of the timing
parameters, over arrays frozen from either host: PINT reads a PINT leg, tempo2
reads a tempo2 leg (clocks, INCLUDE trees, ``TRACK -2`` pulse numbers), and the
delay physics is the same either way. That is why one impl token serves both
native packages -- the leg's own ``timing_package`` picks the reader, not the
physics.

Unlike :class:`~metapulsar.engines.vela.VelaEngine`, which calls ``SPNTA``
per delta, this leg is differentiable: it exposes ``residual_delta_jax``, so a
composite built from vela-jax legs is a ``PulsarJaxTimingEngine`` and NUTS can
sample the timing parameters directly rather than through a host callback.

Enterprise/Discovery still own EFAC/EQUAD/ECORR and the GPs; the par handed to
vela-jax is a delay-only ingest, stripped by the same classifier
:mod:`metapulsar.parfile_lines` uses.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from nltiming.engine_support import LinearModel, is_exact_linear_param
from nltiming.protocols import GaugeProvenance

from .hybrid import (
    is_hybrid_native_param,
    resolve_hybrid_partition,
    validate_nonlinear_params,
)


class VelaJaxEngine:
    """Native vela-jax residual-delta engine for one PTA leg.

    Nonlinear residual deltas come from the vela-jax ``Engine``; the design
    matrix and reference-theta metadata are served from the pulsar-derived
    ``LinearModel`` so the composite sees the same canonical columns as the
    pulsar design matrix. Fit parameters the engine cannot evaluate natively
    (another PTA's JUMPs, host-only gauge columns) go to the exact-linear path.

    ``nonlinear_params`` (``None`` | ``"binary"`` | ``"binary+"``) is the same
    hybrid residual mode every MetaPulsar engine family executes: under a
    hybrid mode only the binary axes (plus ``PX`` for ``"binary+"``) reach the
    engine and every other fitpar is evaluated through its design-matrix
    column (see :mod:`.hybrid`).
    """

    engine_name = "vela_jax"

    def __init__(
        self,
        *,
        engine: Any,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        native_fitpars: tuple[str, ...] | None = None,
        exact_linear_fitpars: frozenset[str] | set[str] | None = None,
        nonlinear_params: str | None = None,
    ):
        self._engine = engine
        self._model = linear_model
        self._param_mapping = dict(param_mapping or {})
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self.nonlinear_params = validate_nonlinear_params(nonlinear_params)

        native_fitpars, exact_linear_fitpars = resolve_hybrid_partition(
            fitpars=self.fitpars,
            param_mapping=self._param_mapping,
            mode=self.nonlinear_params,
            native_fitpars=native_fitpars,
            exact_linear_fitpars=exact_linear_fitpars,
        )
        self._native_fitpars = tuple(native_fitpars)
        self._exact_linear_fitpars = frozenset(exact_linear_fitpars)

        engine_index = {name: i for i, name in enumerate(engine.param_names)}
        self._host_indices = np.array(
            [self.fitpars.index(name) for name in self._native_fitpars], dtype=int
        )
        self._engine_indices = np.array(
            [
                engine_index[self._param_mapping.get(name, name)]
                for name in self._native_fitpars
            ],
            dtype=int,
        )
        self._exact_linear_indices = np.array(
            sorted(self.fitpars.index(name) for name in self._exact_linear_fitpars),
            dtype=int,
        )

    @classmethod
    def from_files(
        cls,
        par_file,
        tim_file,
        *,
        host: str = "pint",
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        nonlinear_params: str | None = None,
        binary_conventions: str = "pint",
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> "VelaJaxEngine":
        """Build a leg engine straight from the retained par/tim inputs.

        ``host`` is the leg's own timing package: ``"tempo2"`` legs are read by
        tempo2, ``"pint"`` legs by PINT. ``binary_conventions`` stays Vela's by
        default on both, so every leg of one pulsar uses one residual formula
        unless the caller deliberately asks otherwise.
        """
        from vela_jax import Engine

        kwargs = dict(engine_kwargs or {})
        kwargs["binary_conventions"] = binary_conventions
        if str(host).lower().startswith("tempo2"):
            engine = Engine.from_tempo2(str(par_file), str(tim_file), **kwargs)
        else:
            engine = Engine.from_files(str(par_file), str(tim_file), **kwargs)
        return cls.from_engine(
            engine,
            linear_model=linear_model,
            param_mapping=param_mapping,
            nonlinear_params=nonlinear_params,
        )

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        nonlinear_params: str | None = None,
    ) -> "VelaJaxEngine":
        """Partition the host fitpars against an already-built vela-jax engine."""
        mapping = dict(param_mapping or {})
        mode = validate_nonlinear_params(nonlinear_params)
        settable = set(engine.param_names)

        native_fitpars: list[str] = []
        exact_linear: list[str] = []
        for name in tuple(linear_model.fitpars):
            engine_param = mapping.get(name, name)
            if (
                is_exact_linear_param(engine_param)
                or engine_param not in settable
                or not is_hybrid_native_param(engine_param, mode)
            ):
                exact_linear.append(name)
                continue
            native_fitpars.append(name)

        # A hybrid mode on a pulsar without binary axes legitimately degenerates
        # to the pure ``-M delta`` residual, as it does for every other family.
        if not native_fitpars and mode is None:
            raise ValueError(
                "No vela-jax-evaluable fit parameters remain after filtering; "
                f"exact-linear candidates: {exact_linear}"
            )
        return cls(
            engine=engine,
            linear_model=linear_model,
            param_mapping=mapping,
            native_fitpars=tuple(native_fitpars),
            exact_linear_fitpars=frozenset(exact_linear),
            nonlinear_params=mode,
        )

    # --- residuals ---------------------------------------------------------

    def _check(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        return delta

    def residual_delta(self, delta_theta) -> np.ndarray:
        delta = self._check(delta_theta)
        step = np.zeros(len(self._engine.param_names))
        step[self._engine_indices] = delta[self._host_indices]
        return self._engine.residual_delta(step) + self._exact_linear_delta(delta)

    def residual_delta_jax(self, delta_theta):
        """The differentiable path: the leg's own JAX residual, plus -M delta."""
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        step = jnp.zeros((len(self._engine.param_names),), dtype=delta.dtype)
        step = step.at[self._engine_indices].set(delta[self._host_indices])
        native = self._engine.residual_delta_jax(step)
        if not len(self._exact_linear_indices):
            return native
        columns = jnp.asarray(
            self._model.design[:, self._exact_linear_indices], dtype=delta.dtype
        )
        return native - columns @ delta[self._exact_linear_indices]

    def _exact_linear_delta(self, delta: np.ndarray) -> np.ndarray:
        if not len(self._exact_linear_indices):
            return np.zeros(self.design_matrix().shape[0], dtype=float)
        columns = np.asarray(
            self._model.design[:, self._exact_linear_indices], dtype=float
        )
        return -(columns @ delta[self._exact_linear_indices])

    def residual_jacobian(self) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        zeros = jnp.zeros((len(self.fitpars),), dtype=jnp.float64)
        return np.asarray(jax.jacfwd(self.residual_delta_jax)(zeros), dtype=float)

    # --- metadata ----------------------------------------------------------

    def exact_linear_fitpars(self) -> frozenset[str]:
        """Pulsar fitpars evaluated exactly via the design matrix."""
        return self._exact_linear_fitpars

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Fitpars whose engine delay is affine in delta."""
        engine_linear = self._engine.identically_linear_params()
        native = {
            name
            for name in self._native_fitpars
            if self._param_mapping.get(name, name) in engine_linear
        }
        return self._exact_linear_fitpars | native

    def precision_critical_fitpars(self) -> frozenset[str]:
        critical = self._engine.precision_critical_params()
        return frozenset(
            name
            for name in self._native_fitpars
            if self._param_mapping.get(name, name) in critical
        )

    def binary_chart_capability(self, chart_family: str, suffix: str):
        from vela_jax.backend import TimingBackend

        return TimingBackend(self._engine).binary_chart_capability(chart_family, suffix)

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def gauge_provenance(self) -> GaugeProvenance:
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode="mean",
            reporting_weighted=True,
        )

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"

    def __repr__(self) -> str:
        return (
            f"<VelaJaxEngine host={self._engine.host} "
            f"native={list(self._native_fitpars)}>"
        )
