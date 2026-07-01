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


@dataclass(frozen=True)
class DeltaWLS:
    """Schur-complement WLS approximation in sampled delta coordinates."""

    fisher: np.ndarray
    covariance: np.ndarray
    mean: np.ndarray


def _as_columns(matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    return np.asarray(matrix[:, indices], dtype=float)


def _weighted_cross(
    left: np.ndarray, weights: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return left.T @ (weights[:, None] * right)


def schur_delta_wls(
    *,
    host,
    partition,
    variance: np.ndarray,
    jitter: float = 1e-12,
) -> DeltaWLS:
    """Return sampled-block Fisher, covariance, and WLS mean in delta units."""
    mmat = np.asarray(host.Mmat, dtype=float)
    residuals = np.asarray(host.residuals, dtype=float)
    weights = 1.0 / np.asarray(variance, dtype=float)

    sampled = _as_columns(mmat, tuple(partition.idx_sampled))
    analytically_marginalized_cols = _as_columns(
        mmat, tuple(partition.idx_analytically_marginalized)
    )
    ndim = sampled.shape[1]
    if ndim == 0:
        empty = np.eye(0, dtype=float)
        return DeltaWLS(
            fisher=empty,
            covariance=empty,
            mean=np.zeros(0, dtype=float),
        )

    fisher_ss = _weighted_cross(sampled, weights, sampled)
    rhs_s = sampled.T @ (weights * residuals)

    if analytically_marginalized_cols.shape[1]:
        fisher_sm = _weighted_cross(sampled, weights, analytically_marginalized_cols)
        fisher_mm = _weighted_cross(
            analytically_marginalized_cols, weights, analytically_marginalized_cols
        )
        rhs_m = analytically_marginalized_cols.T @ (weights * residuals)
        fisher_mm_inv_fms = np.linalg.solve(fisher_mm, fisher_sm.T)
        fisher_ss = fisher_ss - fisher_sm @ fisher_mm_inv_fms
        rhs_s = rhs_s - fisher_sm @ np.linalg.solve(fisher_mm, rhs_m)

    fisher_ss = 0.5 * (fisher_ss + fisher_ss.T)
    fisher_ss = fisher_ss + jitter * np.eye(ndim, dtype=float)
    covariance = np.linalg.inv(fisher_ss)
    covariance = 0.5 * (covariance + covariance.T)
    mean_delta = np.linalg.solve(fisher_ss, rhs_s)
    return DeltaWLS(fisher=fisher_ss, covariance=covariance, mean=mean_delta)


def _z_space_wls(wls: DeltaWLS, prior_bijector) -> tuple[np.ndarray, np.ndarray]:
    """Map delta-space WLS mean/covariance into local PIT z coordinates."""
    if prior_bijector is None:
        return wls.mean, wls.covariance
    mean_z = np.asarray(prior_bijector.z_from_delta(wls.mean, np), dtype=float)
    jac = np.asarray(prior_bijector.jacobian_diag_delta_from_z(mean_z, np), dtype=float)
    if np.any(~np.isfinite(jac)) or np.any(jac <= 0.0):
        raise ValueError(
            "Invalid prior bijector Jacobian while building timing transform"
        )
    inv_jac = np.diag(1.0 / jac)
    covariance_z = inv_jac @ wls.covariance @ inv_jac
    covariance_z = 0.5 * (covariance_z + covariance_z.T)
    return mean_z, covariance_z


def _linear_from_z_covariance(covariance_z: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "standardized":
        return np.diag(np.sqrt(np.diag(covariance_z)))
    if mode == "whitening":
        return np.linalg.cholesky(covariance_z)
    raise ValueError(f"Unsupported transform mode for WLS linear layer: {mode}")


def _linear_transform_from_wls(
    wls: DeltaWLS,
    *,
    prior_bijector,
    mode: str,
    name: str,
) -> LinearTransform:
    z0, covariance_z = _z_space_wls(wls, prior_bijector)
    return LinearTransform(
        C=_linear_from_z_covariance(covariance_z, mode=mode),
        z0=z0,
        name=name,
    )


def diagonal_white(
    ndim: int | None = None,
    *,
    host=None,
    partition=None,
    prior_bijector=None,
    mode: str = "whitening",
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
    wls = schur_delta_wls(
        host=host,
        partition=partition,
        variance=variance,
        jitter=jitter,
    )
    return _linear_transform_from_wls(
        wls,
        prior_bijector=prior_bijector,
        mode=mode,
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
    mode: str = "whitening",
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
    wls = schur_delta_wls(
        host=host,
        partition=partition,
        variance=variance,
        jitter=jitter,
    )
    return _linear_transform_from_wls(
        wls,
        prior_bijector=prior_bijector,
        mode=mode,
        name="fixed_hyperparameters",
    )
