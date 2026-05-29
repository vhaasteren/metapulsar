"""Standardization transforms for nonlinear timing parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AffineTransform:
    """Affine standardized transform: ``delta = center + scale * z``."""

    center: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if float(self.scale) == 0.0:
            raise ValueError("Transform scale must be non-zero.")

    def to_physical(self, z_value: float) -> float:
        return float(self.center) + float(self.scale) * float(z_value)

    def to_standardized(self, delta_value: float) -> float:
        return (float(delta_value) - float(self.center)) / float(self.scale)


def _coerce_transform(param: str, spec) -> AffineTransform:
    if spec is None:
        return AffineTransform()
    if isinstance(spec, AffineTransform):
        return spec
    if isinstance(spec, Mapping):
        return AffineTransform(
            center=float(spec.get("center", 0.0)),
            scale=float(spec.get("scale", 1.0)),
        )
    raise TypeError(f"Unsupported transform spec for '{param}': {type(spec)!r}")


class TransformRegistry:
    """Per-parameter reversible transform registry for sampled timing params."""

    def __init__(
        self,
        sampled_params: list[str],
        standardization: Mapping[str, object] | None = None,
    ):
        self.sampled_params = list(sampled_params)
        self._transforms: dict[str, AffineTransform] = {}
        standardization = standardization or {}
        for param in self.sampled_params:
            self._transforms[param] = _coerce_transform(
                param, standardization.get(param)
            )

    @property
    def transforms(self) -> dict[str, AffineTransform]:
        return dict(self._transforms)

    def to_physical(self, z_params: Mapping[str, float]) -> dict[str, float]:
        self._ensure_known_params(z_params, field_name="standardized")
        return {
            param: self._transforms[param].to_physical(float(z_params[param]))
            for param in self.sampled_params
            if param in z_params
        }

    def to_standardized(self, delta_params: Mapping[str, float]) -> dict[str, float]:
        self._ensure_known_params(delta_params, field_name="physical")
        return {
            param: self._transforms[param].to_standardized(float(delta_params[param]))
            for param in self.sampled_params
            if param in delta_params
        }

    def metadata(self) -> dict[str, dict[str, float]]:
        return {
            param: {"center": tfm.center, "scale": tfm.scale}
            for param, tfm in self._transforms.items()
        }

    def validate_roundtrip(self, atol: float = 1e-12) -> None:
        for param, tfm in self._transforms.items():
            probe_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
            recovered = np.array(
                [tfm.to_standardized(tfm.to_physical(v)) for v in probe_values],
                dtype=float,
            )
            if not np.allclose(probe_values, recovered, atol=atol, rtol=0.0):
                raise ValueError(f"Transform roundtrip failed for '{param}'.")

    def _ensure_known_params(
        self, params: Mapping[str, float], field_name: str
    ) -> None:
        unknown = sorted(set(params.keys()) - set(self.sampled_params))
        if unknown:
            unknown_csv = ", ".join(unknown)
            raise KeyError(f"Unknown {field_name} parameter(s): {unknown_csv}")
