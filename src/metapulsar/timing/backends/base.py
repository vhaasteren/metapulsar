"""Runtime validators and shared backend primitives."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Mapping

import numpy as np

from metapulsar.timing.protocols import EnterprisePulsarLike, TimingBackend

# JUG(tempo2) G1: reference vs _update_param longdouble promotion is ~1e-8 s on
# real IPTA data. See ref-packages/jug/PARITY_ROADMAP.md (G1).
_JUG_TEMPO2_ZERO_DELTA_TOL_SEC = 1e-7

# Lives here (not in jug_jax_state) so importing it never pulls in jax/jug.
_NUMPY_RESIDUAL_DEPRECATION = (
    "JUG NumPy residual path (residual_delta_np) is deprecated and will be "
    "removed once JAX residual_delta_jax reaches full parity. Use "
    "residual_delta_jax for new code."
)
_JUG_TEMPO2_COMPAT_MODES = frozenset(
    {"tempo2", "tempo2-compatible", "tempo2_compatible"}
)


def _as_1d_float(arr, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return out


def validate_enterprise_pulsar(pulsar: EnterprisePulsarLike) -> None:
    """Validate basic frozen pulsar-surface shape invariants."""
    toas = _as_1d_float(pulsar.toas, name="toas")
    residuals = _as_1d_float(pulsar.residuals, name="residuals")
    toaerrs = _as_1d_float(pulsar.toaerrs, name="toaerrs")
    freqs = _as_1d_float(pulsar.freqs, name="freqs")
    backend_flags = np.asarray(pulsar.backend_flags)
    mmat = np.asarray(pulsar.Mmat, dtype=float)

    nrows = len(toas)
    if len(residuals) != nrows or len(toaerrs) != nrows or len(freqs) != nrows:
        raise ValueError(
            "Pulsar arrays toas/residuals/toaerrs/freqs must be same length"
        )
    if len(backend_flags) != nrows:
        raise ValueError("backend_flags length mismatch with pulsar rows")
    if mmat.shape[0] != nrows:
        raise ValueError("Mmat row count must match pulsar arrays")
    if mmat.shape[1] != len(pulsar.fitpars):
        raise ValueError("Mmat column count must match fitpars length")


def _is_jug_tempo2_backend(backend: TimingBackend) -> bool:
    if getattr(backend, "backend_name", None) != "jug":
        return False
    mode = str(getattr(backend, "compatibility", "pint")).lower()
    return mode in _JUG_TEMPO2_COMPAT_MODES


def backend_uses_jug_tempo2_path(backend: TimingBackend) -> bool:
    """Return whether any nested session uses JUG with tempo2 compatibility."""
    if _is_jug_tempo2_backend(backend):
        return True
    sessions = getattr(backend, "_sessions", None)
    if sessions is None:
        return False
    return any(backend_uses_jug_tempo2_path(session.backend) for session in sessions)


def zero_delta_tolerance(backend: TimingBackend, requested: float) -> float:
    """Return the residual_delta(0) tolerance for ``backend``."""
    if backend_uses_jug_tempo2_path(backend):
        return max(float(requested), _JUG_TEMPO2_ZERO_DELTA_TOL_SEC)
    return float(requested)


def is_exact_linear_param(backend_name: str) -> bool:
    """Return true for fitpars that should use exact design-matrix columns."""
    name = backend_name.upper()
    if name == "OFFSET":
        return True
    if name.startswith(("DMX", "JUMP", "FD")):
        return True
    return False


def validate_backend_zero_delta(backend: TimingBackend, tol: float = 1e-12) -> None:
    """Check residual_delta(0) = 0 invariant."""
    effective_tol = zero_delta_tolerance(backend, tol)
    zero = np.zeros(len(backend.fitpars), dtype=float)
    residual = np.asarray(backend.residual_delta(zero), dtype=float)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    if max_abs <= effective_tol:
        if effective_tol > tol and max_abs > tol:
            warnings.warn(
                "JUG compatibility='tempo2' residual_delta(0) is within the relaxed "
                f"MetaPulsar tolerance ({effective_tol:.1e} s, max|delta|={max_abs:.1e} s) "
                "but not the strict check. Known reference-state gap (JUG "
                "PARITY_ROADMAP.md G1); nonlinear tempo2 parity is still experimental.",
                stacklevel=2,
            )
        return
    raise ValueError("residual_delta(0) must equal 0")


def validate_backend_shapes(backend: TimingBackend) -> None:
    """Check backend fitpar/dmatrix shape invariants."""
    design = np.asarray(backend.design_matrix(), dtype=float)
    if design.ndim != 2:
        raise ValueError("design_matrix must be 2D")
    if design.shape[1] != len(backend.fitpars):
        raise ValueError("design_matrix columns must match fitpars")
    ref_exact = backend.reference_theta_exact()
    missing = [name for name in backend.fitpars if name not in ref_exact]
    if missing:
        raise ValueError(f"reference_theta_exact missing fitpars: {missing}")


def validate_backend_against_pulsar(
    backend: TimingBackend, pulsar: EnterprisePulsarLike, tol: float = 1e-12
) -> None:
    """Validate backend outputs against pulsar canonical row and column ordering."""
    validate_enterprise_pulsar(pulsar)
    validate_backend_shapes(backend)
    validate_backend_zero_delta(backend, tol=tol)  # may relax tol for JUG(tempo2)
    design = np.asarray(backend.design_matrix(), dtype=float)
    host_design = np.asarray(pulsar.Mmat, dtype=float)
    nrows = len(pulsar.toas)
    if design.shape[0] != nrows:
        raise ValueError("Backend row count must match pulsar rows")
    if tuple(pulsar.fitpars) != tuple(backend.fitpars):
        raise ValueError("Backend fitpars must match pulsar fitpars in canonical order")
    if design.shape != host_design.shape:
        raise ValueError("Backend design_matrix shape must match pulsar.Mmat")
    if not np.allclose(design, host_design, atol=tol, rtol=0.0):
        raise ValueError(
            "Backend design_matrix must match pulsar.Mmat in canonical row order"
        )


@dataclass(frozen=True)
class LinearModel:
    """Simple linearized residual model around reference theta."""

    fitpars: tuple[str, ...]
    design: np.ndarray
    theta_exact: Mapping[str, str]
    native_units: Mapping[str, str]

    @classmethod
    def from_host(
        cls,
        *,
        fitpars: tuple[str, ...],
        design: np.ndarray,
        theta_exact: Mapping[str, str] | None = None,
        native_units: Mapping[str, str] | None = None,
    ) -> "LinearModel":
        if theta_exact is None:
            theta_exact = {name: "0.0" for name in fitpars}
        if native_units is None:
            native_units = {name: "native" for name in fitpars}
        return cls(
            fitpars=fitpars,
            design=np.asarray(design, dtype=float),
            theta_exact=dict(theta_exact),
            native_units=dict(native_units),
        )

    def reference_theta(self) -> np.ndarray:
        with localcontext() as ctx:
            ctx.prec = 50
            return np.asarray(
                [float(Decimal(self.theta_exact[name])) for name in self.fitpars],
                dtype=float,
            )

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        return self.design @ delta


class LinearTimingBackend:
    """Concrete TimingBackend wrapper around a LinearModel."""

    def __init__(self, model: LinearModel):
        self._model = model
        self.fitpars = model.fitpars
        self.native_units = dict(model.native_units)

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return self._model.residual_delta(delta_theta)

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)
