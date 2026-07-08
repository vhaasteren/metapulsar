"""Nonlinear timing policy objects for advanced MetaPulsar configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .backends import normalize_engines

if TYPE_CHECKING:
    from jug.timing import Tempo2NativeConfig


@dataclass(frozen=True)
class NLTTimingPolicy:
    """Advanced timing-engine policy for ``NonLinearTimingModel``."""

    engines: str | Mapping[str, str] = "jug"
    design_matrix_method: str = "analytic"
    tempo2_native: "str | Tempo2NativeConfig | None" = None
    prime_sessions: bool = True
    verify_wiring: bool = False
    subtract_tzr: bool = False

    def normalized_engines(self) -> dict[str, str]:
        return normalize_engines(self.engines)

    def normalized_design_matrix_method(self) -> str:
        method = str(self.design_matrix_method or "analytic").lower()
        if method not in ("analytic", "autodiff"):
            raise ValueError(
                "design_matrix_method must be 'analytic' or 'autodiff'; "
                f"got {self.design_matrix_method!r}"
            )
        return method


__all__ = ["NLTTimingPolicy"]
