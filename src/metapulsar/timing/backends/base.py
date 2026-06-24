"""Runtime validators and shared backend primitives."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Mapping

import numpy as np

from metapulsar.timing.protocols import EnterprisePulsarLike, TimingBackend


def _as_1d_float(arr, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return out


def validate_enterprise_host(host: EnterprisePulsarLike) -> None:
    """Validate basic frozen host-surface shape invariants."""
    toas = _as_1d_float(host.toas, name="toas")
    residuals = _as_1d_float(host.residuals, name="residuals")
    toaerrs = _as_1d_float(host.toaerrs, name="toaerrs")
    freqs = _as_1d_float(host.freqs, name="freqs")
    backend_flags = np.asarray(host.backend_flags)
    mmat = np.asarray(host.Mmat, dtype=float)

    nrows = len(toas)
    if len(residuals) != nrows or len(toaerrs) != nrows or len(freqs) != nrows:
        raise ValueError("Host arrays toas/residuals/toaerrs/freqs must be same length")
    if len(backend_flags) != nrows:
        raise ValueError("backend_flags length mismatch with host rows")
    if mmat.shape[0] != nrows:
        raise ValueError("Mmat row count must match host arrays")
    if mmat.shape[1] != len(host.fitpars):
        raise ValueError("Mmat column count must match fitpars length")


def validate_backend_zero_delta(backend: TimingBackend, tol: float = 1e-12) -> None:
    """Check residual_delta(0) = 0 invariant."""
    zero = np.zeros(len(backend.fitpars), dtype=float)
    residual = np.asarray(backend.residual_delta(zero), dtype=float)
    if not np.allclose(residual, 0.0, atol=tol, rtol=0.0):
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


def validate_backend_against_host(
    backend: TimingBackend, host: EnterprisePulsarLike, tol: float = 1e-12
) -> None:
    """Validate backend outputs against host canonical row and column ordering."""
    validate_enterprise_host(host)
    validate_backend_shapes(backend)
    validate_backend_zero_delta(backend, tol=tol)
    design = np.asarray(backend.design_matrix(), dtype=float)
    host_design = np.asarray(host.Mmat, dtype=float)
    nrows = len(host.toas)
    if design.shape[0] != nrows:
        raise ValueError("Backend row count must match host rows")
    if tuple(host.fitpars) != tuple(backend.fitpars):
        raise ValueError("Backend fitpars must match host fitpars in canonical order")
    if design.shape != host_design.shape:
        raise ValueError("Backend design_matrix shape must match host.Mmat")
    if not np.allclose(design, host_design, atol=tol, rtol=0.0):
        raise ValueError(
            "Backend design_matrix must match host.Mmat in canonical row order"
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
