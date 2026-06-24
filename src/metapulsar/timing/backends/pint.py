"""Per-session PINT timing backend adapter."""

from __future__ import annotations

from .base import LinearModel, LinearTimingBackend


class PintTimingBackend:
    """Native PINT adapter placeholder until Slice 3b wires live host sessions."""

    backend_name = "pint"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Native PintTimingBackend construction requires Slice 3b host-session wiring"
        )


class LinearizedPintTimingBackend(LinearTimingBackend):
    """Explicit linearized PINT test double using a frozen design matrix."""

    backend_name = "pint"

    @classmethod
    def from_linear_model(cls, model: LinearModel) -> "LinearizedPintTimingBackend":
        return cls(model)
