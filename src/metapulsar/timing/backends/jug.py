"""Per-session JUG timing backend adapter."""

from __future__ import annotations

from typing import Any

from .base import LinearModel, LinearTimingBackend


class JugTimingBackend:
    """Native JUG adapter placeholder until Slice 3b wires live sessions."""

    backend_name = "jug"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Native JugTimingBackend construction requires Slice 3b host-session wiring"
        )


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
