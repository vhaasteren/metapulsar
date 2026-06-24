"""Linear transform builders for timing-space whitening/standardization."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bijectors import WhiteningLinear


@dataclass(frozen=True)
class LinearTransform:
    """Serializable linear transform representation."""

    C: np.ndarray
    z0: np.ndarray
    name: str = "custom"

    def to_whitening_linear(self) -> WhiteningLinear:
        return WhiteningLinear(C=self.C, z0=self.z0)


def _as_columns(matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    return np.asarray(matrix[:, indices], dtype=float)


def _weighted_cross(
    left: np.ndarray, weights: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return left.T @ (weights[:, None] * right)


def _schur_fisher_and_mean(
    *,
    host,
    partition,
    variance: np.ndarray,
    prior_bijector=None,
    jitter: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    mmat = np.asarray(host.Mmat, dtype=float)
    residuals = np.asarray(host.residuals, dtype=float)
    weights = 1.0 / np.asarray(variance, dtype=float)

    sampled = _as_columns(mmat, tuple(partition.idx_sampled))
    marginalized = _as_columns(mmat, tuple(partition.idx_marginalized))
    ndim = sampled.shape[1]
    if ndim == 0:
        return np.eye(0, dtype=float), np.zeros(0, dtype=float)

    fisher_ss = _weighted_cross(sampled, weights, sampled)
    rhs_s = sampled.T @ (weights * residuals)

    if marginalized.shape[1]:
        fisher_sm = _weighted_cross(sampled, weights, marginalized)
        fisher_mm = _weighted_cross(marginalized, weights, marginalized)
        rhs_m = marginalized.T @ (weights * residuals)
        fisher_mm_inv_fms = np.linalg.solve(fisher_mm, fisher_sm.T)
        fisher_ss = fisher_ss - fisher_sm @ fisher_mm_inv_fms
        rhs_s = rhs_s - fisher_sm @ np.linalg.solve(fisher_mm, rhs_m)

    fisher_ss = 0.5 * (fisher_ss + fisher_ss.T)
    fisher_ss = fisher_ss + jitter * np.eye(ndim, dtype=float)
    mean_delta = np.linalg.solve(fisher_ss, rhs_s)
    if prior_bijector is None:
        z0 = mean_delta
    else:
        z0 = np.asarray(prior_bijector.z_from_delta(mean_delta, np), dtype=float)
    return fisher_ss, z0


def diagonal_white(
    ndim: int | None = None,
    *,
    host=None,
    partition=None,
    prior_bijector=None,
    jitter: float = 1e-12,
) -> LinearTransform:
    """Default diagonal-white Fisher/WLS preconditioner."""
    if host is None or partition is None:
        if ndim is None:
            raise ValueError("ndim is required when host/partition are not provided")
        return LinearTransform(
            C=np.eye(ndim, dtype=float),
            z0=np.zeros(ndim, dtype=float),
            name="diagonal_white",
        )

    variance = np.asarray(host.toaerrs, dtype=float) ** 2
    fisher, z0 = _schur_fisher_and_mean(
        host=host,
        partition=partition,
        variance=variance,
        prior_bijector=prior_bijector,
        jitter=jitter,
    )
    return LinearTransform(
        C=np.linalg.cholesky(fisher),
        z0=z0,
        name="diagonal_white",
    )


def _resolve_noise_value(value, labels: np.ndarray, default: float) -> np.ndarray:
    if value is None:
        return np.full(len(labels), default, dtype=float)
    if isinstance(value, dict):
        return np.asarray(
            [float(value.get(label, default)) for label in labels], dtype=float
        )
    return np.full(len(labels), float(value), dtype=float)


def fixed_hyperparameters(
    ndim: int | None = None,
    hyperparameters: dict | None = None,
    *,
    host=None,
    partition=None,
    prior_bijector=None,
    jitter: float = 1e-12,
) -> LinearTransform:
    """Deterministic linear transform from fixed hyperparameter snapshot.

    This mirrors diagonal-white construction with serialized EFAC/EQUAD values.
    Red-noise covariance support is intentionally rejected until a named pure
    NumPy builder is added.
    """
    hyperparameters = hyperparameters or {}
    if host is None or partition is None:
        if ndim is None:
            raise ValueError("ndim is required when host/partition are not provided")
        center = hyperparameters.get("center", None)
        z0 = (
            np.zeros(ndim, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        if z0.shape != (ndim,):
            raise ValueError("fixed_hyperparameters center must match ndim")
        return LinearTransform(
            C=np.eye(ndim, dtype=float), z0=z0, name="fixed_hyperparameters"
        )

    red_noise = hyperparameters.get("red_noise", None)
    if red_noise is not None:
        raise NotImplementedError(
            "fixed_hyperparameters red_noise requires a named pure NumPy builder"
        )

    labels = np.asarray(host.backend_flags)
    efac = _resolve_noise_value(hyperparameters.get("efac", 1.0), labels, 1.0)
    equad = _resolve_noise_value(hyperparameters.get("equad", 0.0), labels, 0.0)
    toaerrs = np.asarray(host.toaerrs, dtype=float)
    variance = (efac * toaerrs) ** 2 + equad**2
    fisher, z0 = _schur_fisher_and_mean(
        host=host,
        partition=partition,
        variance=variance,
        prior_bijector=prior_bijector,
        jitter=jitter,
    )
    return LinearTransform(
        C=np.linalg.cholesky(fisher),
        z0=z0,
        name="fixed_hyperparameters",
    )
