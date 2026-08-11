"""White-noise operators for the flexible-Phi conditional solve.

The flexible-Phi fit treats the base (white) noise covariance ``N`` as fixed and
only requires the ability to apply ``N^-1`` to a vector or a basis matrix. This
keeps the numerical core agnostic to how ``N`` was built: a plain diagonal from
``efac``/``equad``, a Sherman-Morrison update that folds ECORR into ``N``, or
the memory-light :class:`EpochKernelNoise` for column-disjoint epoch jitter.

Concrete operators here depend on NumPy/SciPy only. Frontend-specific builders
live in ``adapters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp


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


@dataclass(frozen=True)
class EpochKernelNoise:
    """``N = D + E diag(jitter) E^T`` for a column-disjoint 0/1 indicator ``E``.

    ``E`` is stored as one epoch index per TOA (``-1`` = TOA in no ECORR epoch),
    so memory is ``O(n)`` and the capacitance
    ``C = Lambda^-1 + E^T D^-1 E`` is diagonal. This is the structure the
    fast-TNT downdate already assumes.

    ``epoch_backends`` (length ``n_ep``) names the backend that owns each epoch
    column; it is set by :meth:`from_backends` and is required for the §3.7
    per-backend jitter M-step / :func:`~pylk.flexfit.adapters.discovery.ecorr_from_kernel`.
    """

    diagonal: np.ndarray  # (n,) D, s^2, > 0
    epoch: np.ndarray  # (n,) int in [-1, n_ep)
    jitter: np.ndarray  # (n_ep,) s^2, > 0
    epoch_backends: tuple[str, ...] | None = None
    _dinv: np.ndarray = field(init=False, repr=False, compare=False)
    _valid: np.ndarray = field(init=False, repr=False, compare=False)
    _s: np.ndarray = field(init=False, repr=False, compare=False)
    _indicator: sp.csr_matrix | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        d = np.asarray(self.diagonal, dtype=float)
        ep = np.asarray(self.epoch, dtype=np.int64)
        j = np.asarray(self.jitter, dtype=float)
        if d.ndim != 1 or np.any(d <= 0.0):
            raise ValueError("diagonal must be a positive 1-D array")
        if ep.shape != d.shape:
            raise ValueError("epoch must be aligned with diagonal")
        if j.ndim != 1 or np.any(j <= 0.0) or int(ep.max(initial=-1)) >= j.size:
            raise ValueError("jitter must be a positive (n_ep,) array covering epoch")
        if self.epoch_backends is not None:
            backends = tuple(str(b) for b in self.epoch_backends)
            if len(backends) != j.size:
                raise ValueError(
                    f"epoch_backends length {len(backends)} != n_ep {j.size}"
                )
            object.__setattr__(self, "epoch_backends", backends)
        dinv = 1.0 / d
        valid = ep >= 0
        t = np.bincount(ep[valid], weights=dinv[valid], minlength=j.size)
        for name, arr in (
            ("diagonal", d),
            ("epoch", ep),
            ("jitter", j),
            ("_dinv", dinv),
            ("_valid", valid),
            ("_s", 1.0 / (1.0 / j + t)),
        ):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @property
    def n_obs(self) -> int:
        return int(self.diagonal.shape[0])

    def solve(self, v: np.ndarray) -> np.ndarray:
        """``N^-1 v = D^-1 v - D^-1 E s (E^T D^-1 v)``, for ``(n,)`` or ``(n, m)``."""
        v = np.asarray(v, dtype=float)
        if v.shape[0] != self.n_obs:
            raise ValueError(
                f"leading dimension {v.shape[0]} does not match n_obs {self.n_obs}"
            )
        squeeze = v.ndim == 1
        mat = v[:, None] if squeeze else v
        dv = self._dinv[:, None] * mat
        # E^T D^-1 v and D^-1 E s (...) through the CSR indicator: rows with
        # epoch < 0 are empty, so they pass through untouched. (An equivalent
        # np.add.at scatter is ~10x slower at (n, k) = (7e4, 2e2).)
        e = self.indicator
        c = np.asarray(e.T @ dv)
        out = dv - self._dinv[:, None] * np.asarray(e @ (self._s[:, None] * c))
        return out[:, 0] if squeeze else out

    def logdet(self) -> float:
        """``log|N| = sum log D_i + sum log(1 + jitter_e t_e)``."""
        return float(
            np.sum(np.log(self.diagonal)) - np.sum(np.log(self._s / self.jitter))
        )

    def diagonal_variance(self) -> np.ndarray:
        """``diag(N) = D_i + jitter_{e(i)}`` (``D_i`` alone where ``epoch < 0``).

        Marginal per-TOA variance for callers that do not subtract the epoch
        waveform. Not the whitening denominator (see waveform analysis).
        """
        out = np.array(self.diagonal, dtype=float)
        out[self._valid] += self.jitter[self.epoch[self._valid]]
        return out

    @property
    def capacitance_scale(self) -> np.ndarray:
        """``s_e = 1 / (1/jitter_e + sum_{i in e} 1/D_i)``, read-only ``(n_ep,)``."""
        return self._s

    @property
    def indicator(self) -> sp.csr_matrix:
        """``E`` as CSR with ``nnz = (epoch >= 0).sum()``; cached on first use."""
        cached = self._indicator
        if cached is not None:
            return cached
        n = self.n_obs
        valid = self._valid
        rows = np.flatnonzero(valid)
        cols = self.epoch[valid]
        built = sp.csr_matrix(
            (np.ones(rows.size, dtype=float), (rows, cols)),
            shape=(n, self.jitter.size),
        )
        object.__setattr__(self, "_indicator", built)
        return built

    @classmethod
    def from_backends(
        cls,
        diagonal: np.ndarray,
        toas: np.ndarray,
        backend_flags: np.ndarray,
        ecorr: Mapping[str, float],
        *,
        dt: float = 1.0,
        ecorr_min: float = 1.0e-9,
    ) -> "EpochKernelNoise":
        """Per-backend ``dt``-quantized epochs, ECORR amplitudes (seconds) by label.

        Matches ``adapters.discovery.ecorr_blocks``: quantize each backend's
        TOAs separately, keep only epochs with more than one TOA, skip backends
        with no such epoch. ``jitter_e = max(ecorr_b, ecorr_min)^2``.

        Every *key* of ``ecorr`` must occur in ``backend_flags`` (else
        ``ValueError``); a label absent from ``ecorr`` gets no epochs.
        """
        from .fasttnt import quantize

        d = np.asarray(diagonal, dtype=float)
        toas_arr = np.asarray(toas, dtype=float)
        labels = np.asarray(backend_flags)
        if d.shape != toas_arr.shape or labels.shape != d.shape:
            raise ValueError("diagonal, toas, and backend_flags must be aligned")

        label_set = {str(x) for x in labels.tolist() if str(x)}
        ecorr_keys = {str(k) for k in ecorr}
        missing = sorted(ecorr_keys - label_set)
        if missing:
            if not (ecorr_keys & label_set):
                raise ValueError(
                    "no ecorr keys occur in backend_flags — selection mismatch "
                    f"(ecorr keys {sorted(ecorr_keys)!r} vs labels "
                    f"{sorted(label_set)!r})"
                )
            raise ValueError(f"ecorr keys absent from backend_flags: {missing!r}")

        epoch = np.full(d.shape, -1, dtype=np.int64)
        jitter_list: list[float] = []
        backend_list: list[str] = []
        next_ep = 0
        for backend in sorted(label_set):
            if backend not in ecorr:
                continue
            mask = labels == backend
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            bins = quantize(toas_arr[idx], dt=dt)
            # Map local bins -> global; keep only multi-TOA epochs.
            uniques, counts = np.unique(bins, return_counts=True)
            amp = max(float(ecorr[backend]), float(ecorr_min))
            jitter_e = amp * amp
            local_to_global: dict[int, int] = {}
            for local_id, cnt in zip(uniques.tolist(), counts.tolist()):
                if cnt <= 1:
                    continue
                local_to_global[int(local_id)] = next_ep
                jitter_list.append(jitter_e)
                backend_list.append(backend)
                next_ep += 1
            if not local_to_global:
                continue
            for j, local_bin in zip(idx, bins):
                g = local_to_global.get(int(local_bin))
                if g is not None:
                    epoch[j] = g

        if not jitter_list:
            # Degenerate: no multi-TOA epochs — keep a dummy epoch unused by epoch=-1.
            return cls(diagonal=d, epoch=epoch, jitter=np.array([1.0], dtype=float))
        return cls(
            diagonal=d,
            epoch=epoch,
            jitter=np.asarray(jitter_list, dtype=float),
            epoch_backends=tuple(backend_list),
        )


def ecorr_from_kernel(noise: EpochKernelNoise) -> dict[str, float]:
    """Per-backend ECORR amplitudes (seconds) from an :class:`EpochKernelNoise`.

    Requires ``noise.epoch_backends`` (set by :meth:`EpochKernelNoise.from_backends`).
    All epochs of a backend share one jitter after construction / the §3.7 M-step;
    the reported amplitude is ``sqrt`` of that common value.
    """
    if noise.epoch_backends is None:
        raise ValueError(
            "ecorr_from_kernel requires EpochKernelNoise.epoch_backends "
            "(build the operator with EpochKernelNoise.from_backends)"
        )
    if not noise.epoch_backends:
        return {}
    out: dict[str, float] = {}
    for backend in sorted(set(noise.epoch_backends)):
        idxs = [i for i, b in enumerate(noise.epoch_backends) if b == backend]
        out[backend] = float(np.sqrt(max(float(noise.jitter[idxs[0]]), 0.0)))
    return out
