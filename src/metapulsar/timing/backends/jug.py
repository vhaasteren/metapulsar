"""Per-session JUG timing backend adapter."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .engines import infer_jug_param_mapping
from .base import LinearModel, LinearTimingBackend


class JugTimingBackend:
    """Native JUG adapter with NumPy and pure-JAX residual-delta paths.

    The nonlinear residuals are evaluated by a frozen JUG ``JaxTimingState``.
    ``design_matrix`` and reference theta metadata are intentionally served from
    the host-derived ``LinearModel`` so the composite backend uses the same
    canonical columns and analytically marginalized basis as ``MetaPulsar.Mmat``.
    """

    backend_name = "jug"

    def __init__(
        self,
        *,
        state: Any,
        linear_model: LinearModel,
        precision_critical: frozenset[str] | set[str] | None = None,
    ):
        self._state = state
        self._model = linear_model
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self._precision_critical = frozenset(precision_critical or frozenset())
        # Defaults for direct construction: every fitpar is JUG-evaluable and no
        # linear fallback. ``from_session`` overrides these when filtering params
        # JUG cannot handle natively (e.g. DMX/Offset).
        self._linear_fallback_fitpars: frozenset[str] = frozenset()
        self._linear_fallback_indices: tuple[int, ...] = tuple()
        self._jug_fitpars: tuple[str, ...] = self.fitpars
        self._jug_indices: tuple[int, ...] = tuple(range(len(self.fitpars)))

    @classmethod
    def from_session(
        cls,
        session: Any,
        *,
        linear_model: LinearModel,
        compatibility: str = "auto",
        param_mapping: Mapping[str, str] | None = None,
        subtract_tzr: bool = True,
        design_matrix_method: str = "analytic",
    ) -> "JugTimingBackend":
        """Build a native JUG backend from an already-created JUG session."""
        from .jug_jax_state import export_jax_timing_state

        fitpars = tuple(linear_model.fitpars)
        mapping = dict(param_mapping or {})
        if not mapping:
            mapping = infer_jug_param_mapping(
                fitpars, set(getattr(session, "params", {}).keys())
            )

        from jug.model.parameter_spec import validate_fit_param

        jug_fitpars: list[str] = []
        linear_fallback: list[str] = []
        for name in fitpars:
            backend_name = mapping.get(name, name)
            if _should_use_linear_fallback(backend_name):
                linear_fallback.append(name)
                continue
            try:
                validate_fit_param(backend_name)
            except ValueError:
                linear_fallback.append(name)
                continue
            jug_fitpars.append(name)

        if not jug_fitpars:
            raise ValueError(
                "No JUG-evaluable fit parameters remain after filtering; "
                f"linear fallback candidates: {linear_fallback}"
            )

        state = export_jax_timing_state(
            session,
            fit_params=tuple(jug_fitpars),
            subtract_tzr=subtract_tzr,
            compatibility=compatibility,
            param_mapping=mapping,
            isort=None,
            design_matrix_method=design_matrix_method,
        )
        backend = cls(
            state=state,
            linear_model=linear_model,
            precision_critical=_canonical_high_precision(tuple(jug_fitpars), mapping),
        )
        backend._linear_fallback_fitpars = frozenset(linear_fallback)
        backend._linear_fallback_indices = tuple(
            fitpars.index(name) for name in linear_fallback
        )
        backend._jug_fitpars = tuple(jug_fitpars)
        backend._jug_indices = tuple(fitpars.index(name) for name in jug_fitpars)
        return backend

    def linear_fallback_fitpars(self) -> frozenset[str]:
        """Host fitpars evaluated linearly via the composite design matrix."""
        return getattr(self, "_linear_fallback_fitpars", frozenset())

    def _jug_delta(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        return delta[np.asarray(self._jug_indices, dtype=int)]

    def _linear_fallback_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        indices = getattr(self, "_linear_fallback_indices", tuple())
        if not indices:
            return np.zeros(self.design_matrix().shape[0], dtype=float)
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        columns = np.asarray(self._model.design[:, list(indices)], dtype=float)
        return columns @ delta[np.asarray(indices, dtype=int)]

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        nonlinear = self._state.residual_delta_np(self._jug_delta(delta_theta))
        return nonlinear + self._linear_fallback_delta(delta_theta)

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def residual_delta_jax(self, delta_theta: Any) -> Any:
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        jug_delta = delta[jnp.asarray(self._jug_indices, dtype=int)]
        nonlinear = self._state.residual_delta_jax(jug_delta)

        indices = getattr(self, "_linear_fallback_indices", tuple())
        if not indices:
            return nonlinear
        design = jnp.asarray(self._model.design[:, list(indices)], dtype=delta.dtype)
        fallback_delta = delta[jnp.asarray(indices, dtype=int)]
        return nonlinear + design @ fallback_delta

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_critical


def _canonical_high_precision(
    fitpars: tuple[str, ...], mapping: Mapping[str, str]
) -> frozenset[str]:
    """Return canonical fitpars JUG marks as high-precision."""
    try:
        from jug.model.parameter_spec import get_high_precision_params
    except Exception:
        return frozenset()

    backend_high_precision = set(get_high_precision_params())
    return frozenset(
        name for name in fitpars if mapping.get(name, name) in backend_high_precision
    )


def _should_use_linear_fallback(backend_name: str) -> bool:
    """Return true for exact-linear timing columns JUG JAX should not own."""
    name = backend_name.upper()
    if name == "OFFSET":
        return True
    if name.startswith(("DMX", "JUMP", "FD")):
        return True
    return False


class LinearizedJugTimingBackend(LinearTimingBackend):
    """Explicit linearized JUG test double with JAX-capable surface."""

    backend_name = "jug"

    def __init__(
        self,
        model: LinearModel,
        *,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        super().__init__(model)
        self.compatibility = compatibility
        self._precision_critical = frozenset(precision_critical)

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ) -> "LinearizedJugTimingBackend":
        return cls(
            model,
            compatibility=compatibility,
            precision_critical=precision_critical,
        )

    def residual_delta_jax(self, delta_theta: Any) -> Any:
        import jax.numpy as jnp

        design = jnp.asarray(self.design_matrix(), dtype=jnp.asarray(delta_theta).dtype)
        delta = jnp.asarray(delta_theta)
        return design @ delta

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_critical
