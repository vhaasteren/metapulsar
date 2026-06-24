"""Per-session Tempo2/libstempo timing backend adapter."""

from __future__ import annotations

from .base import LinearModel, LinearTimingBackend


class Tempo2TimingBackend:
    """Native Tempo2/libstempo adapter placeholder until Slice 3b wiring."""

    backend_name = "tempo2"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Native Tempo2TimingBackend construction requires Slice 3b host-session wiring"
        )


class LinearizedTempo2TimingBackend(LinearTimingBackend):
    """Explicit linearized Tempo2 test double using a frozen design matrix."""

    backend_name = "tempo2"

    @classmethod
    def from_linear_model(cls, model: LinearModel) -> "LinearizedTempo2TimingBackend":
        return cls(model)
