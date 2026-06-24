"""High-precision reference parameter utilities for timing transforms."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np


@dataclass(frozen=True)
class ExactNativeRef:
    """Exact decimal-string representation of native reference parameters."""

    names: tuple[str, ...]
    values: tuple[str, ...]

    @classmethod
    def from_mapping(cls, mapping: dict[str, str]) -> "ExactNativeRef":
        """Construct from exact decimal strings only."""
        names = tuple(mapping.keys())
        values = []
        for name in names:
            value = mapping[name]
            if not isinstance(value, str):
                raise TypeError(
                    "from_mapping expects exact decimal strings; "
                    f"use from_float_mapping for numeric values (key={name!r})"
                )
            values.append(value)
        return cls(names=names, values=tuple(values))

    @classmethod
    def from_float_mapping(cls, mapping: dict[str, float | int]) -> "ExactNativeRef":
        """Construct from numeric values with explicit float-string conversion."""
        names = tuple(mapping.keys())
        values = tuple(str(float(mapping[name])) for name in names)
        return cls(names=names, values=values)

    def as_mapping(self) -> dict[str, str]:
        return dict(zip(self.names, self.values, strict=True))

    def as_decimal_array(self) -> tuple[Decimal, ...]:
        with localcontext() as ctx:
            ctx.prec = 50
            return tuple(Decimal(v) for v in self.values)

    def as_float_array(self) -> np.ndarray:
        return np.asarray([float(v) for v in self.values], dtype=float)
