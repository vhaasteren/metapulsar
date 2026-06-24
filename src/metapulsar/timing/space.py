"""ParameterSpace single-source-of-truth for coordinate transforms."""

from __future__ import annotations

from collections import namedtuple
from decimal import Decimal, localcontext
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .bijectors import AxisPrior, PriorBijector, WhiteningLinear
from .precision import ExactNativeRef
from .units import to_display


DensityParts = namedtuple(
    "DensityParts",
    ["logprior_physical", "logjacobian", "logprior_coord", "loglike", "logpost"],
)


class ParameterSpace:
    """Own x <-> z <-> delta_theta transforms and related densities."""

    def __init__(
        self,
        names: tuple[str, ...],
        theta_ref: ExactNativeRef,
        prior_bijector: PriorBijector,
        linear: WhiteningLinear,
        *,
        transform: str = "standardized",
    ):
        if transform not in {"none", "standardized", "whitening"}:
            raise ValueError(f"Unknown transform mode: {transform}")
        if names != theta_ref.names:
            raise ValueError("names must match theta_ref names")
        self.names = names
        self.theta_ref = theta_ref
        self.prior_bijector = prior_bijector
        self.linear = linear
        self.transform = transform
        self.ndim = len(names)
        if self.linear.C.shape != (self.ndim, self.ndim):
            raise ValueError("linear transform dimension mismatch")

    @classmethod
    def build(
        cls,
        theta_ref_mapping: dict[str, str | float | int],
        prior_bijector: PriorBijector | None = None,
        *,
        transform: str = "standardized",
        linear_transform: WhiteningLinear | None = None,
    ) -> "ParameterSpace":
        if all(isinstance(v, str) for v in theta_ref_mapping.values()):
            exact = ExactNativeRef.from_mapping(theta_ref_mapping)  # type: ignore[arg-type]
        else:
            exact = ExactNativeRef.from_float_mapping(theta_ref_mapping)  # type: ignore[arg-type]
        names = exact.names
        if prior_bijector is None:
            prior_bijector = PriorBijector.from_normal(
                names=names,
                means=np.zeros(len(names), dtype=float),
                stds=np.ones(len(names), dtype=float),
            )
        if linear_transform is None:
            linear_transform = WhiteningLinear.identity(len(names))
        return cls(
            names=names,
            theta_ref=exact,
            prior_bijector=prior_bijector,
            linear=linear_transform,
            transform=transform,
        )

    def z_from_x(self, x, xp):
        return self.linear.z_from_x(x, xp)

    def x_from_z(self, z, xp):
        return self.linear.x_from_z(z, xp)

    def delta_from_z(self, z, xp):
        return self.prior_bijector.delta_from_z(z, xp)

    def z_from_delta(self, delta, xp):
        return self.prior_bijector.z_from_delta(delta, xp)

    def delta_from_coord(self, q, xp, coord: str = "x"):
        if coord == "delta":
            return xp.asarray(q)
        if coord == "z":
            return self.delta_from_z(q, xp)
        if coord == "x":
            z = self.z_from_x(q, xp)
            return self.delta_from_z(z, xp)
        raise ValueError(f"Unsupported coord: {coord}")

    def coord_from_delta(self, delta, xp, coord: str = "x"):
        if coord == "delta":
            return xp.asarray(delta)
        if coord == "z":
            return self.z_from_delta(delta, xp)
        if coord == "x":
            z = self.z_from_delta(delta, xp)
            return self.x_from_z(z, xp)
        raise ValueError(f"Unsupported coord: {coord}")

    def delta_from_cube(self, u, xp):
        return self.prior_bijector.delta_from_u(u, xp)

    def coord_from_cube(self, u, xp, coord: str = "x"):
        delta = self.delta_from_cube(u, xp)
        return self.coord_from_delta(delta, xp, coord=coord)

    def logprior_physical(self, delta, xp):
        return self.prior_bijector.logprior_physical(delta, xp)

    def logjacobian(self, q, xp, coord: str = "x"):
        if coord == "delta":
            return xp.asarray(0.0)
        if coord == "z":
            return self.prior_bijector.logabsdet_delta_from_z(q, xp)
        if coord == "x":
            z = self.z_from_x(q, xp)
            return self.prior_bijector.logabsdet_delta_from_z(z, xp) + xp.asarray(
                self.linear.logabsdet
            )
        raise ValueError(f"Unsupported coord: {coord}")

    def logprior_coord(self, q, xp, coord: str = "x"):
        delta = self.delta_from_coord(q, xp, coord=coord)
        return self.logprior_physical(delta, xp) + self.logjacobian(q, xp, coord=coord)

    def density_parts(
        self,
        q,
        loglike_fn: Callable[[np.ndarray], float],
        xp,
        coord: str = "x",
    ) -> DensityParts:
        delta = self.delta_from_coord(q, xp, coord=coord)
        logprior_physical = self.logprior_physical(delta, xp)
        logjacobian = self.logjacobian(q, xp, coord=coord)
        logprior_coord = logprior_physical + logjacobian
        loglike = xp.asarray(loglike_fn(delta))
        logpost = loglike + logprior_coord
        return DensityParts(
            logprior_physical=logprior_physical,
            logjacobian=logjacobian,
            logprior_coord=logprior_coord,
            loglike=loglike,
            logpost=logpost,
        )

    def theta_from_delta(self, delta) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (self.ndim,):
            raise ValueError("delta has wrong shape")
        with localcontext() as ctx:
            ctx.prec = 50
            out = []
            for ref_str, d in zip(self.theta_ref.values, delta, strict=True):
                out.append(float(Decimal(ref_str) + Decimal(str(float(d)))))
        return np.asarray(out, dtype=float)

    def delta_from_theta(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError("theta has wrong shape")
        with localcontext() as ctx:
            ctx.prec = 50
            out = []
            for ref_str, th in zip(self.theta_ref.values, theta, strict=True):
                out.append(float(Decimal(str(float(th))) - Decimal(ref_str)))
        return np.asarray(out, dtype=float)

    def to_physical(self, samples, units: str = "display", coord: str | None = None):
        if coord is None:
            coord = (
                "delta"
                if self.transform == "none"
                else ("z" if self.transform == "standardized" else "x")
            )
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[-1] != self.ndim:
            raise ValueError("samples final dimension mismatch")
        delta = np.stack(
            [self.delta_from_coord(row, np, coord=coord) for row in arr], axis=0
        )
        theta = np.stack([self.theta_from_delta(row) for row in delta], axis=0)
        if units == "native":
            return {name: theta[:, i] for i, name in enumerate(self.names)}
        if units == "display":
            return {
                name: np.asarray(to_display(name, theta[:, i]), dtype=float)
                for i, name in enumerate(self.names)
            }
        raise ValueError(f"Unknown units mode: {units}")

    def metadata(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "theta_ref": self.theta_ref.as_mapping(),
            "transform": self.transform,
        }

    def save(self, path: str | Path) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(base) + ".npz",
            C=self.linear.C,
            z0=self.linear.z0,
        )
        prior_payload = []
        for p in self.prior_bijector.priors:
            prior_payload.append(
                {
                    "family": p.family,
                    "lower": p.lower,
                    "upper": p.upper,
                    "mean": p.mean,
                    "std": p.std,
                }
            )
        payload = self.metadata() | {"priors": prior_payload}
        (Path(str(base) + ".json")).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ParameterSpace":
        base = Path(path)
        arrays = np.load(str(base) + ".npz")
        payload = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
        names = tuple(payload["names"])
        theta_ref = ExactNativeRef.from_mapping(payload["theta_ref"])
        priors = tuple(AxisPrior(**entry) for entry in payload["priors"])
        prior_bijector = PriorBijector(names=names, priors=priors)
        linear = WhiteningLinear(C=arrays["C"], z0=arrays["z0"])
        return cls(
            names=names,
            theta_ref=theta_ref,
            prior_bijector=prior_bijector,
            linear=linear,
            transform=payload["transform"],
        )
