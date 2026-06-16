"""JUG frozen-state JAX timing engine for Discovery/NUTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .engines import infer_jug_param_mapping
from .parameter_space import SampledTimingParameterSpace


@dataclass(frozen=True)
class JugLinearizedTimingContext:
    """Diagnostic linearized timing evaluator: ``delta_r ~= M @ delta_theta``.

    Convention (must match the nonlinear path so the linear curve is its exact
    tangent):

    * ``design_matrix`` is the timing design matrix ``d(residual)/d(theta)`` at
      theta=0 (the partial derivatives of the timing model wrt the timing
      parameters), built by ``export_jax_timing_state`` as adaptive finite
      differences of the host nonlinear residual -- no autodiff -- already in the
      engine's native-delta / output (isort) order exactly like
      ``reference_residuals_sec``. The residual delta is therefore a plain
      ``design_matrix @ delta_theta`` -- no sign flip and no further ``isort``.
    * ``isort`` is retained only so external consumers can align separately-built
      bases (e.g. the marginalized Woodbury basis) into the same output order.
    """

    param_names: tuple[str, ...]
    theta_ref: np.ndarray
    design_matrix: Any
    column_units: tuple[str, ...]
    reference_residuals_sec: Any
    isort: Any | None

    def residual_delta_jax(self, delta_theta):
        delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
        return jnp.asarray(self.design_matrix, dtype=jnp.float64) @ delta_theta

    def residual_delta_np(self, delta_theta: np.ndarray) -> np.ndarray:
        delta_theta = np.asarray(delta_theta, dtype=np.float64).reshape(-1)
        return np.asarray(self.design_matrix, dtype=np.float64) @ delta_theta


class JugJaxTimingEngine:
    """JAX-native JUG timing engine backed by exported frozen state."""

    def __init__(
        self,
        *,
        jax_state,
        parameter_space: SampledTimingParameterSpace,
        fitpars: Sequence[str],
        param_mapping: Mapping[str, str] | None = None,
        isort: np.ndarray | None = None,
        evaluation_mode: Literal["nonlinear", "linearized"] = "nonlinear",
        linearized_context: JugLinearizedTimingContext | None = None,
    ):
        from .jug_jax_state import JaxTimingState

        if not isinstance(jax_state, JaxTimingState):
            raise TypeError(
                "jax_state must be a metapulsar.nonlinear_timing_model.jug_jax_state.JaxTimingState"
            )

        self._state = jax_state
        self._parameter_space = parameter_space
        self._evaluation_mode = evaluation_mode
        self._linearized = linearized_context
        self._param_mapping = dict(param_mapping or {})
        self._isort = None if isort is None else np.asarray(isort, dtype=int)

        self.sampled_params = tuple(parameter_space.names)
        self.fitpars = list(fitpars)
        self.param_names = list(
            self._param_mapping.get(name, name) for name in self.sampled_params
        )
        self._reference_residuals = np.asarray(
            jax_state.reference_residuals_sec, dtype=np.float64
        )
        ref_residuals = getattr(jax_state, "reference_residuals_sec", None)
        self.output_shape = (int(np.asarray(ref_residuals).shape[0]),)
        self.output_dtype = jnp.float64

        if evaluation_mode == "linearized" and linearized_context is None:
            self._linearized = JugLinearizedTimingContext(
                param_names=tuple(jax_state.fit_params),
                theta_ref=np.asarray(jax_state.ref_theta, dtype=np.float64),
                design_matrix=np.asarray(jax_state.design_matrix, dtype=np.float64),
                column_units=tuple(jax_state.column_units),
                reference_residuals_sec=np.asarray(
                    jax_state.reference_residuals_sec, dtype=np.float64
                ),
                isort=self._isort,
            )

    @classmethod
    def from_session(
        cls,
        session,
        *,
        sampled_parameter_space: SampledTimingParameterSpace,
        fitpars: Sequence[str],
        param_mapping: Mapping[str, str] | None = None,
        reference_params: Mapping[str, float] | None = None,
        subtract_tzr: bool = True,
        isort: np.ndarray | None = None,
        evaluation_mode: Literal["nonlinear", "linearized"] = "nonlinear",
        compatibility: str | None = None,
    ) -> JugJaxTimingEngine:
        from .jug_jax_state import export_jax_timing_state

        del reference_params  # reference comes from the session params
        sampled = list(sampled_parameter_space.names)
        unknown = sorted(set(sampled) - set(fitpars))
        if unknown:
            raise ValueError(
                "Sampled parameters must be subset of fitpars for JUG JAX export: "
                f"{unknown}"
            )

        mapping = dict(param_mapping or {})
        if not mapping:
            backend = set(getattr(session, "params", {}).keys())
            mapping = infer_jug_param_mapping(fitpars, backend)

        state = export_jax_timing_state(
            session,
            fit_params=sampled,
            subtract_tzr=subtract_tzr,
            compatibility=compatibility or getattr(session, "compatibility", "pint"),
            param_mapping=mapping,
            isort=isort,
        )
        linearized = JugLinearizedTimingContext(
            param_names=tuple(state.fit_params),
            theta_ref=np.asarray(state.ref_theta, dtype=np.float64),
            design_matrix=np.asarray(state.design_matrix, dtype=np.float64),
            column_units=tuple(state.column_units),
            reference_residuals_sec=np.asarray(
                state.reference_residuals_sec, dtype=np.float64
            ),
            isort=isort,
        )
        return cls(
            jax_state=state,
            parameter_space=sampled_parameter_space,
            fitpars=fitpars,
            param_mapping=mapping,
            isort=isort,
            evaluation_mode=evaluation_mode,
            linearized_context=linearized,
        )

    @property
    def linearized_context(self) -> JugLinearizedTimingContext:
        if self._linearized is None:
            raise RuntimeError("Linearized timing context is unavailable.")
        return self._linearized

    def _delta_theta_from_z_np(self, z_flat: np.ndarray) -> np.ndarray:
        return self._parameter_space.delta_from_z_np(z_flat)

    def _delta_theta_from_z_jax(self, z_flat):
        return self._parameter_space.delta_from_z_jax(z_flat)

    def residual_delta_np(self, z_flat: np.ndarray) -> np.ndarray:
        delta_theta = self._delta_theta_from_z_np(z_flat)
        if self._evaluation_mode == "linearized":
            return self.linearized_context.residual_delta_np(delta_theta)
        return self._state.residual_delta_np(delta_theta)

    def residual_delta_jax(self, z_flat):
        delta_theta = self._delta_theta_from_z_jax(z_flat)
        if self._evaluation_mode == "linearized":
            return self.linearized_context.residual_delta_jax(delta_theta)
        return self._state.residual_delta_jax(delta_theta)

    def timing_delay_jax(self, z_flat):
        return -self.residual_delta_jax(z_flat)

    def timing_delay_np(self, z_flat: np.ndarray) -> np.ndarray:
        return -self.residual_delta_np(z_flat)
