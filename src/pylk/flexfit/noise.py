"""White-noise operators for the flexible-Phi conditional solve.

The flexible-Phi fit treats the base (white) noise covariance ``N`` as fixed and
only requires the ability to apply ``N^-1`` to a vector or a basis matrix. This
keeps the numerical core agnostic to how ``N`` was built: a plain diagonal from
``efac``/``equad``, or a Sherman-Morrison update that folds ECORR into ``N``
rather than into the fitted basis.

Concrete operators here depend on NumPy/SciPy only. Frontend-specific builders
(e.g. reusing Discovery's ``makenoise_measurement``) live in ``adapters``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.linalg as sl


@runtime_checkable
class NoiseOperator(Protocol):
    """Apply ``N^-1`` to residual vectors and basis matrices.

    ``solve`` must accept a ``(n_obs,)`` vector or a ``(n_obs, k)`` matrix and
    return ``N^-1 @ v`` with the same shape.
    """

    @property
    def n_obs(self) -> int: ...

    def solve(self, v: np.ndarray) -> np.ndarray: ...

    def logdet(self) -> float: ...


@dataclass(frozen=True)
class DiagonalNoise:
    """Diagonal white noise ``N = diag(variance)`` in seconds squared."""

    variance: np.ndarray

    def __post_init__(self) -> None:
        var = np.asarray(self.variance, dtype=float)
        if var.ndim != 1 or var.size == 0:
            raise ValueError("variance must be a non-empty 1-D array")
        if np.any(~np.isfinite(var)) or np.any(var <= 0.0):
            raise ValueError("variance entries must be finite and positive")
        var.setflags(write=False)
        object.__setattr__(self, "variance", var)

    @classmethod
    def from_toaerrs(
        cls, toaerrs: np.ndarray, *, efac: float = 1.0, equad: float = 0.0
    ) -> "DiagonalNoise":
        """Build ``N = (efac * toaerr)^2 + equad^2`` (a single-selection model)."""
        errs = np.asarray(toaerrs, dtype=float)
        return cls((float(efac) * errs) ** 2 + float(equad) ** 2)

    @property
    def n_obs(self) -> int:
        return int(self.variance.shape[0])

    def solve(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.shape[0] != self.n_obs:
            raise ValueError(
                f"leading dimension {v.shape[0]} does not match n_obs {self.n_obs}"
            )
        if v.ndim == 1:
            return v / self.variance
        return v / self.variance[:, None]

    def logdet(self) -> float:
        return float(np.sum(np.log(self.variance)))


@dataclass(frozen=True)
class ShermanMorrisonNoise:
    """White noise with a low-rank update ``N = D + U diag(jitter) U^T``.

    Useful when ECORR (or any block-jitter term) is folded into the base
    covariance instead of being fitted as a basis block. ``D`` is the diagonal
    part (variance in s^2); ``U`` is the ``(n_obs, m)`` epoch-averaging basis and
    ``jitter`` its ``(m,)`` variances.
    """

    diagonal: np.ndarray
    u: np.ndarray
    jitter: np.ndarray

    def __post_init__(self) -> None:
        d = np.asarray(self.diagonal, dtype=float)
        u = np.asarray(self.u, dtype=float)
        j = np.asarray(self.jitter, dtype=float)
        if d.ndim != 1 or np.any(d <= 0.0):
            raise ValueError("diagonal must be a positive 1-D array")
        if u.ndim != 2 or u.shape[0] != d.shape[0]:
            raise ValueError("u must be (n_obs, m) aligned with diagonal")
        if j.shape != (u.shape[1],) or np.any(j <= 0.0):
            raise ValueError("jitter must be a positive (m,) array aligned with u")
        for arr in (d, u, j):
            arr.setflags(write=False)
        object.__setattr__(self, "diagonal", d)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "jitter", j)
        # Capacitance C = jitter^-1 + U^T D^-1 U (small, m x m).
        dinv_u = u / d[:, None]
        capacitance = np.diag(1.0 / j) + u.T @ dinv_u
        object.__setattr__(self, "_dinv_u", dinv_u)
        object.__setattr__(self, "_cap_cf", sl.cho_factor(capacitance))

    @property
    def n_obs(self) -> int:
        return int(self.diagonal.shape[0])

    def solve(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.shape[0] != self.n_obs:
            raise ValueError(
                f"leading dimension {v.shape[0]} does not match n_obs {self.n_obs}"
            )
        squeeze = v.ndim == 1
        mat = v[:, None] if squeeze else v
        dinv_v = mat / self.diagonal[:, None]
        correction = self._dinv_u @ sl.cho_solve(self._cap_cf, self.u.T @ dinv_v)
        out = dinv_v - correction
        return out[:, 0] if squeeze else out

    def logdet(self) -> float:
        sign, logabsdet = np.linalg.slogdet(
            np.diag(1.0 / self.jitter) + self.u.T @ self._dinv_u
        )
        if sign <= 0:
            raise ValueError("capacitance matrix is not positive definite")
        return (
            float(np.sum(np.log(self.diagonal)))
            + float(np.sum(np.log(self.jitter)))
            + float(logabsdet)
        )
