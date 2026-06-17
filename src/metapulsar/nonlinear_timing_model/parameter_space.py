"""Canonical sampled-timing transform and prior module for Discovery/Enterprise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .priors import SampledTimingParameterRegistry


@dataclass(frozen=True)
class SampledTimingParameterSpace:
    """Single source of truth for z <-> theta and induced priors."""

    names: tuple[str, ...]
    theta_ref: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    z_lower: np.ndarray
    z_upper: np.ndarray
    prior_kind: tuple[str, ...]
    theta_lower: np.ndarray
    theta_upper: np.ndarray

    @classmethod
    def from_registry(
        cls, registry: SampledTimingParameterRegistry
    ) -> SampledTimingParameterSpace:
        names = tuple(registry.sampled_params)
        theta_ref = np.array(
            [registry._params[name].theta_ref for name in names],
            dtype=np.float64,
        )
        center = np.array(
            [registry._params[name].transform.center for name in names],
            dtype=np.float64,
        )
        scale = np.array(
            [registry._params[name].transform.scale for name in names],
            dtype=np.float64,
        )
        z_bounds = [registry._params[name].z_bounds() for name in names]
        z_lower = np.array([lo for lo, _hi in z_bounds], dtype=np.float64)
        z_upper = np.array([hi for _lo, hi in z_bounds], dtype=np.float64)
        prior_kind = tuple(
            (
                "log_uniform"
                if registry._params[name].prior.is_log_uniform
                else "uniform" if registry._params[name].prior.is_uniform else "other"
            )
            for name in names
        )
        theta_bounds = [registry._params[name].prior.support() for name in names]
        theta_lower = np.array([lo for lo, _hi in theta_bounds], dtype=np.float64)
        theta_upper = np.array([hi for _lo, hi in theta_bounds], dtype=np.float64)
        return cls(
            names=names,
            theta_ref=theta_ref,
            center=center,
            scale=scale,
            z_lower=z_lower,
            z_upper=z_upper,
            prior_kind=prior_kind,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
        )

    @classmethod
    def from_transform_registry(
        cls,
        *,
        names: Sequence[str],
        theta_ref: Mapping[str, float] | None = None,
        center: Mapping[str, float] | None = None,
        scale: Mapping[str, float] | None = None,
        z_lower: Mapping[str, float] | None = None,
        z_upper: Mapping[str, float] | None = None,
    ) -> SampledTimingParameterSpace:
        """Build a parameter space from transform metadata without PINT priors."""
        theta_ref = theta_ref or {}
        center = center or {}
        scale = scale or {}
        z_lower = z_lower or {}
        z_upper = z_upper or {}
        return cls(
            names=tuple(names),
            theta_ref=np.array(
                [float(theta_ref.get(name, 0.0)) for name in names],
                dtype=np.float64,
            ),
            center=np.array(
                [float(center.get(name, 0.0)) for name in names],
                dtype=np.float64,
            ),
            scale=np.array(
                [float(scale.get(name, 1.0)) for name in names],
                dtype=np.float64,
            ),
            z_lower=np.array(
                [float(z_lower.get(name, -np.inf)) for name in names],
                dtype=np.float64,
            ),
            z_upper=np.array(
                [float(z_upper.get(name, np.inf)) for name in names],
                dtype=np.float64,
            ),
            prior_kind=tuple("uniform" for _ in names),
            theta_lower=np.full(len(names), -np.inf, dtype=np.float64),
            theta_upper=np.full(len(names), np.inf, dtype=np.float64),
        )

    def delta_from_z_np(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape[0] != len(self.names):
            raise ValueError(
                f"z length {z.shape[0]} != {len(self.names)} sampled parameters."
            )
        return self.center + self.scale * z

    def theta_from_z_np(self, z: np.ndarray) -> np.ndarray:
        return self.theta_ref + self.delta_from_z_np(z)

    def z_from_theta_np(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        return (theta - self.theta_ref - self.center) / self.scale

    def _logprior_z_contrib(self, z: np.ndarray, idx: int) -> float:
        kind = self.prior_kind[idx]
        z_lo = float(self.z_lower[idx])
        z_hi = float(self.z_upper[idx])
        a = float(self.theta_lower[idx])
        b = float(self.theta_upper[idx])
        scale = float(self.scale[idx])
        theta = float(self.theta_ref[idx] + self.center[idx] + scale * float(z[idx]))
        if kind == "uniform":
            if not np.isfinite(z_lo) or not np.isfinite(z_hi):
                return 0.0
            if float(z[idx]) < z_lo or float(z[idx]) > z_hi:
                return -np.inf
            return -np.log(z_hi - z_lo) + np.log(abs(scale))
        if kind == "log_uniform":
            if theta < a or theta > b:
                return -np.inf
            log_span = np.log(b / a)
            return -np.log(theta) - np.log(log_span) + np.log(abs(scale))
        raise NotImplementedError(
            f"NumPy prior for {self.names[idx]!r} requires uniform or log_uniform support."
        )

    def logprior_z_np(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        total = 0.0
        for idx in range(len(self.names)):
            lp = self._logprior_z_contrib(z, idx)
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return float(total)

    def delta_from_z_jax(self, z):
        import jax.numpy as jnp

        z = jnp.asarray(z, dtype=jnp.float64).reshape(-1)
        return self.center + self.scale * z

    def theta_from_z_jax(self, z):
        return self.theta_ref + self.delta_from_z_jax(z)

    def prior_transform_z_np(self, q: np.ndarray, idx: int) -> np.ndarray:
        kind = self.prior_kind[idx]
        a = float(self.theta_lower[idx])
        b = float(self.theta_upper[idx])
        q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
        if kind == "uniform":
            theta = a + q * (b - a)
        elif kind == "log_uniform":
            theta = a * (b / a) ** q
        else:
            raise NotImplementedError(
                f"NumPy prior transform for {self.names[idx]!r} requires "
                "uniform or log_uniform support."
            )
        return (theta - self.theta_ref[idx] - self.center[idx]) / self.scale[idx]

    def prior_transform_z_jax(self, q, idx: int):
        import jax.numpy as jnp

        kind = self.prior_kind[idx]
        a = float(self.theta_lower[idx])
        b = float(self.theta_upper[idx])
        q = jnp.clip(jnp.asarray(q, dtype=jnp.float64), 0.0, 1.0)
        if kind == "uniform":
            theta = a + q * (b - a)
        elif kind == "log_uniform":
            theta = a * (b / a) ** q
        else:
            raise NotImplementedError(
                f"JAX prior transform for {self.names[idx]!r} requires "
                "uniform or log_uniform support."
            )
        return (theta - self.theta_ref[idx] - self.center[idx]) / self.scale[idx]

    def logprior_z_jax(self, z):
        import jax.numpy as jnp

        z = jnp.asarray(z, dtype=jnp.float64).reshape(-1)
        lp = jnp.array(0.0, dtype=jnp.float64)
        for idx, kind in enumerate(self.prior_kind):
            z_lo = float(self.z_lower[idx])
            z_hi = float(self.z_upper[idx])
            a = float(self.theta_lower[idx])
            b = float(self.theta_upper[idx])
            scale = self.scale[idx]
            theta = self.theta_ref[idx] + self.center[idx] + scale * z[idx]
            if kind == "uniform":
                if np.isfinite(z_lo) and np.isfinite(z_hi):
                    inside = (z[idx] >= z_lo) & (z[idx] <= z_hi)
                    lp = lp + jnp.where(
                        inside,
                        -jnp.log(z_hi - z_lo) + jnp.log(jnp.abs(scale)),
                        -jnp.inf,
                    )
            elif kind == "log_uniform":
                inside = (theta >= a) & (theta <= b)
                log_span = jnp.log(b / a)
                lp = lp + jnp.where(
                    inside,
                    -jnp.log(theta) - jnp.log(log_span) + jnp.log(jnp.abs(scale)),
                    -jnp.inf,
                )
            else:
                raise NotImplementedError(
                    f"JAX prior for {self.names[idx]!r} requires "
                    "uniform or log_uniform support."
                )
        return lp

    def log_abs_dtheta_dz_jax(self):
        import jax.numpy as jnp

        return jnp.sum(jnp.log(jnp.abs(self.scale)))

    def delta_dict_from_z_np(self, z: np.ndarray) -> dict[str, float]:
        delta = self.delta_from_z_np(z)
        return {name: float(delta[idx]) for idx, name in enumerate(self.names)}

    def metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, name in enumerate(self.names):
            rows.append(
                {
                    "name": name,
                    "theta_ref": float(self.theta_ref[idx]),
                    "center": float(self.center[idx]),
                    "scale": float(self.scale[idx]),
                    "z_lower": float(self.z_lower[idx]),
                    "z_upper": float(self.z_upper[idx]),
                    "prior_kind": self.prior_kind[idx],
                    "theta_lower": float(self.theta_lower[idx]),
                    "theta_upper": float(self.theta_upper[idx]),
                }
            )
        return rows
