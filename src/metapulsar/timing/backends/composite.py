"""Composite timing backend that assembles per-session adapters in host order."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any, Mapping

import numpy as np

from metapulsar.timing.protocols import JaxTimingBackend, TimingBackend


@dataclass(frozen=True)
class BackendSession:
    """One per-PTA timing session contribution to a composite backend."""

    name: str
    row_indices: np.ndarray
    backend: TimingBackend
    linear_fallback_fitpars: frozenset[str] = frozenset()
    fallback_reference_exact: Mapping[str, str] = field(default_factory=dict)


def _to_exact_str(value: str) -> str:
    with localcontext() as ctx:
        ctx.prec = 50
        return format(Decimal(value), "f")


class CompositeTimingBackend:
    """Canonical-row-order composite over per-session timing-backend adapters."""

    def __init__(
        self,
        *,
        fitpars: tuple[str, ...],
        nrows: int,
        sessions: list[BackendSession],
        missing_param_policy: str = "linear_fallback",
        host_design: np.ndarray | None = None,
    ):
        if missing_param_policy not in {"linear_fallback", "strict"}:
            raise ValueError(
                "missing_param_policy must be one of {'linear_fallback', 'strict'}"
            )
        self.fitpars = fitpars
        self.native_units = {name: "native" for name in fitpars}
        self._nrows = int(nrows)
        self._sessions = list(sessions)
        self._missing_param_policy = missing_param_policy
        self._global_index = {name: i for i, name in enumerate(self.fitpars)}
        for session in self._sessions:
            unknown_fallback = [
                name
                for name in session.linear_fallback_fitpars
                if name not in self._global_index
            ]
            if unknown_fallback:
                raise ValueError(
                    f"Session '{session.name}' declares fallback for unknown fitpars: "
                    f"{unknown_fallback}"
                )
        self._host_design = (
            None if host_design is None else np.asarray(host_design, dtype=float)
        )
        self._ref_exact = self._merge_reference_theta_exact()

    def _merge_reference_theta_exact(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for session in self._sessions:
            ref = session.backend.reference_theta_exact()
            ref = dict(ref) | dict(session.fallback_reference_exact)
            for name in self.fitpars:
                if name not in ref:
                    continue
                exact = _to_exact_str(str(ref[name]))
                if name in merged and merged[name] != exact:
                    raise ValueError(
                        f"Shared fitpar '{name}' disagrees across sessions: "
                        f"{merged[name]} != {exact} (session={session.name})"
                    )
                merged[name] = exact
        for name in self.fitpars:
            if name not in merged:
                raise ValueError(
                    f"No session provides reference_theta_exact for '{name}'"
                )
        return merged

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._ref_exact)

    def reference_theta(self) -> np.ndarray:
        return np.asarray(
            [float(self._ref_exact[name]) for name in self.fitpars], dtype=float
        )

    def _session_delta_and_fallback(
        self, session: BackendSession, delta_theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        local = np.zeros(len(session.backend.fitpars), dtype=float)
        missing_names: list[str] = []

        # Only explicitly declared mapped-but-not-nonlinear fitpars use the
        # missing-parameter policy. Canonical fitpars absent from this session
        # contribute zero on this row block by construction.
        for name in session.linear_fallback_fitpars:
            value = delta_theta[self._global_index[name]]
            if value == 0.0:
                continue
            missing_names.append(name)

        for i, name in enumerate(session.backend.fitpars):
            if name in self._global_index:
                local[i] = delta_theta[self._global_index[name]]
            else:
                missing_names.append(name)
        if missing_names and self._missing_param_policy == "strict":
            raise ValueError(
                f"Session '{session.name}' requires linear fallback for {missing_names}"
            )
        if missing_names and self._host_design is not None:
            rows = np.asarray(session.row_indices, dtype=int)
            fallback = np.zeros(len(rows), dtype=float)
            for name in missing_names:
                if name in self._global_index:
                    fallback += (
                        self._host_design[rows, self._global_index[name]]
                        * delta_theta[self._global_index[name]]
                    )
        elif missing_names:
            raise ValueError(
                f"Session '{session.name}' requires linear fallback for {missing_names}, "
                "but no host design matrix was provided"
            )
        else:
            fallback = np.zeros(len(session.row_indices), dtype=float)
        return local, fallback

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        out = np.zeros(self._nrows, dtype=float)
        for session in self._sessions:
            local_delta, fallback = self._session_delta_and_fallback(session, delta)
            block = (
                np.asarray(session.backend.residual_delta(local_delta), dtype=float)
                + fallback
            )
            out[np.asarray(session.row_indices, dtype=int)] = block
        return out

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        out = np.zeros((self._nrows, len(self.fitpars)), dtype=float)
        for session in self._sessions:
            block = np.asarray(
                session.backend.design_matrix(params=params), dtype=float
            )
            rows = np.asarray(session.row_indices, dtype=int)
            for local_j, name in enumerate(session.backend.fitpars):
                if name not in self._global_index:
                    if self._missing_param_policy == "strict":
                        raise ValueError(
                            f"Session '{session.name}' has unmapped fitpar '{name}'"
                        )
                    continue
                out[rows, self._global_index[name]] = block[:, local_j]
            if session.linear_fallback_fitpars:
                if self._missing_param_policy == "strict":
                    raise ValueError(
                        f"Session '{session.name}' requires linear fallback for "
                        f"{sorted(session.linear_fallback_fitpars)}"
                    )
                if self._host_design is None:
                    raise ValueError(
                        f"Session '{session.name}' requires linear fallback but no "
                        "host design matrix was provided"
                    )
                for name in session.linear_fallback_fitpars:
                    out[rows, self._global_index[name]] = self._host_design[
                        rows, self._global_index[name]
                    ]
        return out

    def linearized_design_matrix(self, params: Any | None = None) -> np.ndarray:
        """Assemble each session's selected linearized residual basis."""
        out = np.zeros((self._nrows, len(self.fitpars)), dtype=float)
        for session in self._sessions:
            matrix_fn = getattr(
                session.backend,
                "linearized_design_matrix",
                session.backend.design_matrix,
            )
            block = np.asarray(matrix_fn(params=params), dtype=float)
            rows = np.asarray(session.row_indices, dtype=int)
            for local_j, name in enumerate(session.backend.fitpars):
                if name not in self._global_index:
                    if self._missing_param_policy == "strict":
                        raise ValueError(
                            f"Session '{session.name}' has unmapped fitpar '{name}'"
                        )
                    continue
                out[rows, self._global_index[name]] = block[:, local_j]
            if session.linear_fallback_fitpars:
                if self._missing_param_policy == "strict":
                    raise ValueError(
                        f"Session '{session.name}' requires linear fallback for "
                        f"{sorted(session.linear_fallback_fitpars)}"
                    )
                if self._host_design is None:
                    raise ValueError(
                        f"Session '{session.name}' requires linear fallback but no "
                        "host design matrix was provided"
                    )
                for name in session.linear_fallback_fitpars:
                    out[rows, self._global_index[name]] = self._host_design[
                        rows, self._global_index[name]
                    ]
        return out


class CompositeJaxTimingBackend(CompositeTimingBackend):
    """Composite backend with JAX-capable path and precision-critical union."""

    def __init__(
        self,
        *,
        fitpars: tuple[str, ...],
        nrows: int,
        sessions: list[BackendSession],
        missing_param_policy: str = "linear_fallback",
        host_design: np.ndarray | None = None,
    ):
        super().__init__(
            fitpars=fitpars,
            nrows=nrows,
            sessions=sessions,
            missing_param_policy=missing_param_policy,
            host_design=host_design,
        )
        self._precision_union = frozenset().union(
            *[
                session.backend.precision_critical_fitpars()
                for session in sessions
                if isinstance(session.backend, JaxTimingBackend)
            ]
        )

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        out = jnp.zeros((self._nrows,), dtype=delta.dtype)
        for session in self._sessions:
            if not isinstance(session.backend, JaxTimingBackend):
                raise ValueError(
                    f"Session '{session.name}' does not provide a JAX backend path"
                )
            local = jnp.zeros((len(session.backend.fitpars),), dtype=delta.dtype)
            for i, name in enumerate(session.backend.fitpars):
                if name in self._global_index:
                    local = local.at[i].set(delta[self._global_index[name]])
                elif self._missing_param_policy == "strict":
                    raise ValueError(
                        f"Session '{session.name}' missing global mapping for fitpar '{name}'"
                    )
            block = jnp.asarray(
                session.backend.residual_delta_jax(local), dtype=delta.dtype
            )
            out = out.at[jnp.asarray(session.row_indices, dtype=int)].set(block)
        return out

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_union


def build_composite_backend(
    *,
    fitpars: tuple[str, ...],
    nrows: int,
    sessions: list[BackendSession],
    missing_param_policy: str = "linear_fallback",
    host_design: np.ndarray | None = None,
) -> TimingBackend:
    """Return JAX-capable composite only when all sessions are JAX-capable."""
    all_jax = all(isinstance(s.backend, JaxTimingBackend) for s in sessions)
    cls = CompositeJaxTimingBackend if all_jax else CompositeTimingBackend
    return cls(
        fitpars=fitpars,
        nrows=nrows,
        sessions=sessions,
        missing_param_policy=missing_param_policy,
        host_design=host_design,
    )
